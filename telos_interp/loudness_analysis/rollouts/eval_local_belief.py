"""Compare a local-belief probe's predictions to both the local belief and the final action.

For each eval split (its `label` is the local belief, `final_label` the trajectory's
agent_action):

  * acc and BALANCED acc vs local belief   (balanced reproduces the training report)
  * acc and BALANCED acc vs final action   (where the trajectory ended)
  * on the rows where local != final: does the probe follow local or final?
  * the same three, split by whether the CUT TOKEN ITSELF is a direction word --
    the verbalization confound of ICLR log 42(e)/43: if the model already typed
    " up", both the probe activation and the rollout answer read that, trivially.
"""

import argparse
import json
from pathlib import Path

import torch

from telos_interp.commands.prepare_activations_for_probing.manifest_loader import load_next_action_compact
from telos_interp.probe_models import create_classification_model

DIR_JSON = "/workspace/jlens/direction_tokens_full.json"
ID2A = {0: "LEFT", 1: "UP", 2: "RIGHT", 3: "DOWN"}


def bal_acc(pred: torch.Tensor, y: torch.Tensor) -> float:
    """Mean per-class recall over the classes actually present in `y`.

    This is the statistic the probe checkpoints report as `best_balanced_accuracy`, computed the
    same way as train_next_action_probe::_evaluate: per-class tp/gt_support, averaged over classes
    with support, unweighted. Plain accuracy is NOT a substitute here -- the local-belief label is
    imbalanced (UP runs ~.35 of rows against DOWN's ~.17 on these splits), so a probe that leans
    on the majority class is flattered by accuracy and corrected by this.
    """
    if y.numel() == 0:
        return float("nan")
    recalls = []
    for c in torch.unique(y):
        m = y == c
        recalls.append((pred[m] == c).float().mean().item())
    return sum(recalls) / len(recalls)


def load_probe(path: Path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    m = create_classification_model(
        d["model_type"], d["input_dim"], d["num_classes"], d["hidden_dims"] or [], d["dropout"] or 0.0
    )
    m.load_state_dict(d["model_state_dict"])
    m.eval()
    return m, d.get("scaler_mean"), d.get("scaler_std")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("probe", type=Path)
    ap.add_argument("eval_dir", type=Path, help="the *_eval split dir")
    ap.add_argument("--direction-json", default=DIR_JSON)
    args = ap.parse_args()

    manifest = json.loads((args.eval_dir / "manifest.json").read_text())
    samples = manifest["samples"]
    data = load_next_action_compact(manifest, args.eval_dir / "manifest.json")
    X = data["base_act"].float()
    y_local = data["labels"].long()
    y_final = torch.tensor([s.get("final_label", s["label"]) for s in samples], dtype=torch.long)

    dirvocab = set()
    raw = json.loads(Path(args.direction_json).read_text())
    for v in raw.values():
        dirvocab |= set(v)

    def norm(t):
        # manifest / selection tokens are raw byte-level BPE ("Ġup", "Ċ"); the direction
        # vocab is decoded text (" up", "\n"). Normalise before matching.
        if t is None:
            return None
        return t.replace("Ġ", " ").replace("Ċ", "\n")

    is_dir = torch.tensor([norm(s.get("token")) in dirvocab for s in samples])

    m, mean, std = load_probe(args.probe)
    if mean is not None:
        X = (X - mean) / std
    with torch.no_grad():
        pred = m(X).argmax(-1)

    n = len(pred)
    acc_local = (pred == y_local).float().mean().item()
    acc_final = (pred == y_final).float().mean().item()
    print(f"{args.probe.name}  N={n}")
    print(f"  bal acc vs LOCAL belief : {bal_acc(pred, y_local):.4f}")
    print(f"  bal acc vs FINAL action : {bal_acc(pred, y_final):.4f}")
    print(f"  acc vs LOCAL belief : {acc_local:.4f}")
    print(f"  acc vs FINAL action : {acc_final:.4f}")
    print(f"  local == final      : {(y_local == y_final).float().mean().item():.4f} of rows")

    diff = y_local != y_final
    if diff.any():
        pd = pred[diff]
        print(f"  --- on the {diff.sum().item()} rows where local != final ---")
        print(f"    pred == local  : {(pd == y_local[diff]).float().mean().item():.4f}")
        print(f"    pred == final  : {(pd == y_final[diff]).float().mean().item():.4f}")
        print(f"    pred == neither: {((pd != y_local[diff]) & (pd != y_final[diff])).float().mean().item():.4f}")

    # The split below keys on each sample's `token` field. Several prepared splits do not carry
    # one -- their entries were written without it -- and then `s.get("token")` is None for every
    # row, every row lands in the "NOT a direction word" bucket, and the split restates the
    # top-line numbers under a heading that reads like a finding ("no cut token is a direction
    # word"). Say so instead of printing that.
    if not any(s.get("token") is not None for s in samples):
        print("  [direction-word split unavailable: this manifest's samples carry no 'token' field]")
        return 0

    for name, mask in (("cut token IS a direction word", is_dir), ("cut token is NOT a direction word", ~is_dir)):
        if mask.sum() == 0:
            continue
        mm = mask
        al = (pred[mm] == y_local[mm]).float().mean().item()
        af = (pred[mm] == y_final[mm]).float().mean().item()
        eq = (y_local[mm] == y_final[mm]).float().mean().item()
        d2 = mm & diff
        line = (
            f"  [{name}] N={mm.sum().item()}  bal_local={bal_acc(pred[mm], y_local[mm]):.4f}  "
            f"acc_local={al:.4f}  acc_final={af:.4f}  local==final={eq:.4f}"
        )
        if d2.any():
            line += (
                f"  | on local!=final (N={d2.sum().item()}): "
                f"pred=local {(pred[d2] == y_local[d2]).float().mean().item():.3f}, "
                f"pred=final {(pred[d2] == y_final[d2]).float().mean().item():.3f}"
            )
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
