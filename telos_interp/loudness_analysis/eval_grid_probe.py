"""Score a multiclass grid-cell probe on a prepared grid_tile split, one row per (token, cell).

The grid twin of `rollouts/eval_local_belief.py`, and built to sit in the same cross-selection
matrix: one probe, one split dir, a block of text per cell. For probes written by
`train_cognitive_map_probe` on a token-major manifest (`input_dim = D + 2`: the token's
activation plus the cell's (row, col)).

  * BALANCED acc, the trainer's definition (mean recall over classes with support), so the
    diagonal of a cross-eval reproduces what the trainer reported as its FINAL number
  * balanced acc WITHOUT padding: every grid is padded to 15, so `+` is an easy class that is
    most of a small grid's cells, and the first number counts it
  * per-class precision / recall
  * the same, split by whether the TOKEN ITSELF is a signal word -- the verbalization
    confound: a token that already spells out a grid word may be read off trivially

Cells whose symbol the probe never saw in training are excluded, as the trainer's eval
loader excludes them. `--signal-json` takes any `{class: [tokens]}` file, as in
eval_local_belief.py.
"""

import argparse
import json
from pathlib import Path

import torch

from telos_interp.commands.prepare_activations_for_probing.manifest_loader import load_v3_manifest
from telos_interp.commands.train_cognitive_map_probe.train_cognitive_map_probe_fn import (
    _load_grid_tile_compact_cached,
)
from telos_interp.grid_utils import CELL_ID_TO_SYMBOL, CELL_SYMBOL_TO_ID
from telos_interp.probe_models import create_classification_model

PADDING_ID = CELL_SYMBOL_TO_ID["+"]


def bal_acc(pred: torch.Tensor, y: torch.Tensor, exclude: tuple[int, ...] = ()) -> float:
    """Mean per-class recall over the classes present in `y`, minus `exclude`.

    The trainer's statistic (train_cognitive_map_probe::_evaluate): tp / gt_support per class,
    averaged unweighted over classes with support.

    Example:
        >>> bal_acc(torch.tensor([0, 1, 1, 1]), torch.tensor([0, 0, 1, 1]))
        0.75
        >>> bal_acc(torch.tensor([0, 1, 1, 1]), torch.tensor([0, 0, 1, 1]), exclude=(1,))
        0.5
    """
    recalls = [(pred[y == c] == c).float().mean().item() for c in torch.unique(y).tolist() if c not in exclude]
    return sum(recalls) / len(recalls) if recalls else float("nan")


def load_probe(path: Path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    if "positive_class" in d or "label_to_idx" not in d:
        raise SystemExit(f"{path} is not a multiclass probe from train_cognitive_map_probe")
    m = create_classification_model(
        d["model_type"], d["input_dim"], d["num_classes"], d["hidden_dims"] or [], d["dropout"] or 0.0
    )
    m.load_state_dict(d["model_state_dict"])
    m.eval()
    return m, d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("probe", type=Path)
    ap.add_argument("eval_dir", type=Path, help="a prepared grid_tile split dir")
    ap.add_argument("--signal-json", required=True, help="Signal vocabulary as {class: [tokens]}.")
    ap.add_argument("--signal-name", default="signal", help="What to call its words in the output, e.g. 'grid'.")
    ap.add_argument("--cache-activations", action="store_true", help="Use the trainer's packed cache.")
    ap.add_argument("--batch-size", type=int, default=65536, help="Rows per forward pass.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Also write the global block (per-class support / correct / predicted, keyed by cell id) "
        "as JSON: what a per-token decile notebook's check cell compares against.",
    )
    args = ap.parse_args()

    manifest_path = args.eval_dir / "manifest.json"
    manifest = load_v3_manifest(manifest_path)
    if manifest.get("probe_type") != "grid_tile":
        raise SystemExit(f"{args.eval_dir} is not a grid_tile manifest")
    entries = manifest["trajectories"]

    model, d = load_probe(args.probe)
    if manifest["activation_dim"] + 2 != d["input_dim"]:
        raise SystemExit(
            f"probe input_dim {d['input_dim']} != activation_dim {manifest['activation_dim']} + 2: "
            "different layer or token category"
        )
    model = model.to(args.device)
    mean, std = d.get("scaler_mean"), d.get("scaler_std")
    if mean is not None:
        mean, std = mean.to(args.device), std.to(args.device)

    compact = _load_grid_tile_compact_cached(manifest, manifest_path, args.cache_activations, False)
    X, pos, raw = compact["base_act"], compact["positions"], compact["labels"]
    if len(entries) != X.shape[0]:
        raise SystemExit(f"{len(entries)} manifest entries but {X.shape[0]} loaded rows")

    # The trainer's NaN filter, applied to the entry list too so the token split stays aligned.
    keep = ~torch.isnan(X).any(dim=1)
    X, pos, raw = X[keep], pos[keep], raw[keep]
    entries = [e for e, k in zip(entries, keep.tolist(), strict=True) if k]
    T, C = raw.shape

    label_to_idx = d["label_to_idx"]
    idx_to_label = d["idx_to_label"]
    y = torch.full_like(raw, -1)
    for original, idx in label_to_idx.items():
        y[raw == original] = idx
    valid = y >= 0

    pred = torch.empty((T, C), dtype=torch.long)
    per_chunk = max(1, args.batch_size // C)
    with torch.no_grad():
        for s in range(0, T, per_chunk):
            e = min(s + per_chunk, T)
            a = X[s:e].float().to(args.device)
            p = pos[s:e].float().to(args.device)
            rows = torch.cat([a.unsqueeze(1).expand(e - s, C, a.shape[1]), p], dim=2).reshape(-1, a.shape[1] + 2)
            if mean is not None:
                rows = (rows - mean) / std
            pred[s:e] = model(rows).argmax(-1).reshape(e - s, C).cpu()

    signal_vocab = set()
    for v in json.loads(Path(args.signal_json).read_text()).values():
        signal_vocab |= set(v)

    def norm(t):
        # manifest tokens are raw byte-level BPE ("Ġup", "Ċ"); a signal vocab is decoded text.
        if t is None:
            return None
        return t.replace("Ġ", " ").replace("Ċ", "\n")

    pad_idx = tuple(i for i, lab in idx_to_label.items() if lab == PADDING_ID)
    classes = [(i, CELL_ID_TO_SYMBOL[idx_to_label[i]]) for i in sorted(idx_to_label)]

    def block(mask_tc: torch.Tensor) -> tuple[float, float, float, torch.Tensor, torch.Tensor]:
        m = mask_tc & valid
        yy, pp = y[m], pred[m]
        return bal_acc(pp, yy), bal_acc(pp, yy, pad_idx), (pp == yy).float().mean().item(), yy, pp

    everything = torch.ones((T, C), dtype=torch.bool)
    b, b_np, acc, yy, pp = block(everything)
    print(
        f"{args.probe.name}  entries={T}  rows={int(valid.sum())}  "
        f"signal={Path(args.signal_json).name} ({len(signal_vocab)} words)"
    )
    print(f"  bal acc (trainer, incl. padding) : {b:.4f}")
    print(f"  bal acc (no padding)             : {b_np:.4f}")
    print(f"  acc                              : {acc:.4f}")
    print(f"  {'class':<6}{'precision':>10}{'recall':>9}{'support':>10}{'predicted':>11}")
    per_class = {}
    for i, sym in classes:
        tp = int(((pp == i) & (yy == i)).sum())
        sup, prd = int((yy == i).sum()), int((pp == i).sum())
        print(f"  {sym:<6}{tp / prd if prd else 0:>10.4f}{tp / sup if sup else 0:>9.4f}{sup:>10}{prd:>11}")
        # Keyed by the ORIGINAL cell id, which is what a per-token table's n_true_{c} uses.
        per_class[str(idx_to_label[i])] = {"symbol": sym, "support": sup, "correct": tp, "predicted": prd}

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(
                {
                    "probe": str(args.probe),
                    "data": str(args.eval_dir),
                    "global": {
                        "n_entries": T,
                        "n_rows": int(valid.sum()),
                        "accuracy": acc,
                        "balanced_accuracy": b,
                        "balanced_accuracy_no_padding": b_np,
                        "per_class": per_class,
                    },
                },
                indent=2,
            )
        )
        print(f"  -> {args.out_json}")

    # As in eval_local_belief.py: a manifest whose entries carry no `token` would put every row in
    # the NOT bucket and restate the top line as if it were a finding. Say so instead.
    if not any(e.get("token") is not None for e in entries):
        print(f"  [{args.signal_name}-word split unavailable: this manifest's entries carry no 'token' field]")
        return 0

    is_signal = torch.tensor([norm(e.get("token")) in signal_vocab for e in entries])
    for name, tok_mask in (
        (f"token IS a {args.signal_name} word", is_signal),
        (f"token is NOT a {args.signal_name} word", ~is_signal),
    ):
        if tok_mask.sum() == 0:
            continue
        b, b_np, acc, _, _ = block(tok_mask.unsqueeze(1).expand(T, C))
        print(f"  [{name}] N={int(tok_mask.sum())} tokens  bal={b:.4f}  bal_no_pad={b_np:.4f}  acc={acc:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
