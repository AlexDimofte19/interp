#!/usr/bin/env python3
"""A signal-mass table for a NEW vocabulary, from activations already on disk -- no model.

A mass table is baked at gather time: `build_loudness_tables.py` unembeds each layer's
residual stream while it is on the device and keeps only `log P(any signal word)` for the
one vocabulary it was given. Asking a second vocabulary's question of the same tokens looked
like it needed a second forward pass. It does not, wherever the tree kept the `.pt`: the
gather saves `blocks[layer]` to disk and lenses the very same tensor, so

    .pt -> float -> apply_lens_transport -> lens_predictions

is the gather's own arithmetic on the gather's own input. What it costs is the unembed
(`gpt-oss-20b_unembed.pt`, lm_head + final norm) and, for the jlens, the fitted `J` -- the
two files `--jlens_dir` already holds. No model weights, no MXFP4 kernels, fits any GPU.

WHICH TOKENS. The source tree's own mass table (any vocabulary, `--source-lens`) lists every
reasoning token with its (step, reasoning_pos, abs_pos, token); a row is written for each one
whose `.pt` exists at `--layer`. A PRUNED tree (jlens_mass_l15) keeps only its selected
tokens, so the output covers exactly those -- every token a probe on that tree can be scored
on, and nothing a join could misread: a token without a `.pt` has no row, so the join leaves
its cells empty (absent, not quiet).

THE OUTPUT IS A LENS TREE. Per trajectory, `{stem}_{lens}_direction_mass.csv` plus its
`.meta.json` (signal name, fingerprint, `source: "activations"`), and a symlink to the source
analysis CSV, which is what `lens_io.trajectory_dirs` finds a trajectory folder by. So
`join_signal_loudness.py --lens-root OUT --signal-name NAME` reads it unchanged.

`--self-check` recomputes the SOURCE tree's own vocabulary on a few trajectories and compares
it with the source table: agreement is the evidence that this path is the gather's.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from telos_interp.jlens_utils import load_direction_tokens, mass_header, read_mass_meta, write_mass_meta
from telos_interp.jlens_utils.jlens_csv import output_start, step_folder_index
from telos_interp.jlens_utils.methods import direction_mass_path
from telos_interp.loudness_analysis import signals
from telos_interp.loudness_analysis.lens_io import trajectory_dirs

csv.field_size_limit(10**9)
PREFIX = ["size", "complexity", "run", "step", "reasoning_pos", "abs_pos", "token", "agent_action"]


def source_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def lens_trajectory(rows, act_folder: Path, traj: dict, layer: int, lenses, assets, mass_ids, torch):
    """{lens: [(prefix cells, logmass)]} for every row whose .pt exists at `layer`."""
    from telos_interp.loudness_analysis.build_loudness_tables import apply_lens_transport

    kept, acts = [], []
    starts: dict[int, tuple[int, int]] = {}
    for r in rows:
        step = int(r["step"])
        if step not in starts:
            starts[step] = (output_start(traj, step), step_folder_index(traj, step))
        start, folder = starts[step]
        pt = act_folder / f"layer_{layer}" / f"step_{folder}" / "output" / f"{int(r['abs_pos']) - start}.pt"
        if pt.exists():
            kept.append([r[c] for c in PREFIX])
            acts.append(torch.load(pt, map_location="cpu", weights_only=True))
    if not kept:
        return {lens: [] for lens in lenses}
    dev = assets["dev"]
    x = torch.stack(acts).to(dev)
    out = {}
    for lens in lenses:
        J_stack, J_rows = assets["transport"][lens]
        masses = []
        for i in range(0, len(x), assets["batch"]):
            with torch.no_grad():
                h = apply_lens_transport(x[i : i + assets["batch"]].float().unsqueeze(0), J_stack, J_rows,
                                         assets["norm_w"], assets["eps"])[0]
                logits = (h.to(assets["lm_head"].dtype) @ assets["lm_head"].T).float()
                norm = logits.logsumexp(-1)
                masses += (logits[:, mass_ids].logsumexp(-1) - norm).cpu().tolist()
        out[lens] = list(zip(kept, masses, strict=True))
    return out


def build_assets(args, lenses, torch):
    """The gather's own unembed, norm and transports, built through its own helpers."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.jlens_action_ranks import action_token_ids, ensure_unembed_assets
    from telos_interp.jlens_utils import get_model
    from telos_interp.loudness_analysis.build_loudness_tables import build_lens_transports

    spec = get_model(args.model_id)
    dev = torch.device(args.device)
    raw = ensure_unembed_assets(args.jlens_dir, spec)
    _, tok = action_token_ids(spec)
    layers_by_lens, transport = build_lens_transports(lenses, [args.layer], args.jlens_dir, dev, spec)
    for lens in lenses:
        if args.layer not in layers_by_lens[lens]:
            raise SystemExit(f"{lens} has no lens at layer {args.layer}")
    return spec, tok, {
        "dev": dev,
        "lm_head": raw["lm_head"].to(dev),
        "norm_w": spec.lens_norm_weight(raw["norm_weight"].float().to(dev)),
        "eps": raw["rms_eps"],
        "transport": transport,
        "batch": args.batch_size,
    }


def vocab_ids(signal_json: Path, tok, torch, dev) -> tuple:
    from telos_interp.loudness_analysis.build_loudness_tables import resolve_direction_ids

    vocab = load_direction_tokens(signal_json, "all")
    ids, dropped = resolve_direction_ids(vocab, lambda t: tok.encode(t, add_special_tokens=False),
                                         lambda i: tok.decode([i]))
    if not ids:
        raise SystemExit(f"{signal_json} resolved to no single-token ids")
    print(f"{signal_json.name}: {len(ids)} token ids, {len(dropped)} dropped {dropped[:8]}", flush=True)
    return torch.tensor(ids, device=dev), dropped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-root", type=Path, required=True, help="Lens tree holding the .pt and a mass table.")
    ap.add_argument("--trajectories-dir", type=Path, required=True, help="Trajectory JSONs (for output_start).")
    ap.add_argument("--signal-json", type=Path, required=True, help="The NEW vocabulary, {class: [tokens]}.")
    ap.add_argument("--signal-name", required=True, help="Names the vocabulary in every sidecar and column.")
    ap.add_argument("--out", type=Path, required=True, help="New lens tree to write.")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--lenses", default="jlens,logitlens")
    ap.add_argument("--jlens_dir", type=Path, default=Path("/workspace/jlens/gridenv"),
                    help="Lens + unembed. gridenv/ is the J every direction tree was gathered with (--self-check).")
    ap.add_argument("--model-id", default="openai/gpt-oss-20b")
    ap.add_argument("--source-lens", default="jlens", help="Whose mass table lists the tokens (default jlens).")
    ap.add_argument("--names-file", type=Path, default=None, help="Only these trajectory stems.")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--self-check", type=int, default=0, metavar="N",
                    help="Recompute the SOURCE vocabulary on N trajectories, compare, and exit.")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch

    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    lenses = [s for s in args.lenses.split(",") if s]
    spec, tok, assets = build_assets(args, lenses, torch)
    traj_paths = {p.stem: p for p in args.trajectories_dir.rglob("*.json")}
    folders = [f for f in trajectory_dirs(args.source_root) if f.name in traj_paths]
    if args.names_file:
        wanted = set(args.names_file.read_text().split())
        folders = [f for f in folders if f.name in wanted]
    if args.limit or args.self_check:
        folders = folders[: args.self_check or args.limit]
    print(f"{len(folders)} trajectories under {args.source_root}", flush=True)

    if args.self_check:
        return self_check(args, folders, traj_paths, lenses, assets, tok, torch)

    mass_ids, dropped = vocab_ids(args.signal_json, tok, torch, assets["dev"])
    signal = signals.resolve(args.signal_name, args.signal_json)
    meta = {
        "signal_json": str(args.signal_json), "signal_name": signal.name,
        "signal_fingerprint": signal.fingerprint(args.signal_json), "direction_classes": "all",
        "num_direction_tokens": int(mass_ids.numel()), "num_dropped": len(dropped), "dropped": dropped,
        "model": spec.model_id, "final_norm_offset": spec.norm_offset,
        "source": "activations", "source_root": str(args.source_root),
    }
    n_rows = 0
    for k, folder in enumerate(folders):
        stem = folder.name
        dest = args.out / folder.parent.name / stem
        paths = {lens: direction_mass_path(dest, lens) for lens in lenses}
        if all(p.exists() for p in paths.values()):
            continue
        rows = source_rows(direction_mass_path(folder, args.source_lens))
        act = next((c for c in folder.iterdir() if c.is_dir()), None)
        if act is None:
            print(f"{stem}: no activation folder, skipped", flush=True)
            continue
        traj = json.loads(traj_paths[stem].read_text())
        result = lens_trajectory(rows, act, traj, args.layer, lenses, assets, mass_ids, torch)
        dest.mkdir(parents=True, exist_ok=True)
        for src in folder.glob(f"{stem}_*_analysis.csv"):
            link = dest / src.name
            if not link.exists():
                link.symlink_to(src)
        for lens in lenses:
            tmp = paths[lens].with_suffix(".csv.tmp")
            with open(tmp, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(mass_header([args.layer]))
                w.writerows([*pre, round(m, 6)] for pre, m in result[lens])
            tmp.replace(paths[lens])
            write_mass_meta(paths[lens], {**meta, "lens": lens, "layers": [args.layer]})
        n_rows += len(result[lenses[0]])
        if k % 50 == 0:
            print(f"  [{k + 1}/{len(folders)}] {stem}: {len(result[lenses[0]])} tokens", flush=True)
    print(f"done: {n_rows} token rows -> {args.out}", flush=True)
    return 0


def self_check(args, folders, traj_paths, lenses, assets, tok, torch) -> int:
    """Recompute each source table's OWN vocabulary from the .pt and compare at --layer."""
    worst = 0.0
    for folder in folders:
        stem = folder.name
        traj = json.loads(traj_paths[stem].read_text())
        act = next(c for c in folder.iterdir() if c.is_dir())
        for lens in lenses:
            src = direction_mass_path(folder, lens)
            if not src.exists():
                continue
            vocab = Path(read_mass_meta(src)["signal_json"])
            if not vocab.exists():
                vocab = Path(__file__).resolve().parents[2] / "data/jlens" / vocab.name
            ids, _ = vocab_ids(vocab, tok, torch, assets["dev"])
            rows = source_rows(src)
            got = lens_trajectory(rows, act, traj, args.layer, [lens], assets, ids, torch)[lens]
            ref = {(r["step"], r["abs_pos"]): float(r[f"L{args.layer}"]) for r in rows if r[f"L{args.layer}"] != ""}
            diffs = [abs(m - ref[(pre[3], pre[5])]) for pre, m in got]
            worst = max(worst, max(diffs, default=0.0))
            print(f"{stem} {lens}: {len(diffs)} tokens, max |diff| {max(diffs, default=0):.2e}", flush=True)
    # A wrong lens misses by nats (the wikitext J against a gridenv tree: 2-4). The .pt of a
    # SELECTING gather come from a second forward pass with a different batch shape, and bf16
    # moves those by up to ~0.3 (the batching noise floor), so exact agreement is only seen on
    # unselected trees. The threshold separates the two, not bf16 from bf16.
    print(f"self-check: worst |diff| {worst:.2e} ({'OK' if worst < 1.0 else 'WRONG LENS?'})")
    return 0 if worst < 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
