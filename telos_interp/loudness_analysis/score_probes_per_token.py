#!/usr/bin/env python3
"""Score trained probes on EVERY reasoning token of a trajectory set, under any lens.

The arm scripts answer "how well does a probe do on the tokens the lens SELECTED". This
answers the prior question: how does the probe do on each token, and does the lens score
predict where it does well? One row per (trajectory, step, token), carrying both the probe's
verdict and every lens score available for that token, so the two can be crossed afterwards
without re-running anything.

THIS IS ALREADY A LOUDNESS TABLE. Beside every probe's prediction it writes, per lens, the
top-k count and the full-vocabulary mass at the requested layer and at the token's own best
layer. That is why there is no separate "join loudness to probe predictions" step -- the
loudness is here, and `join_rollouts.py` only has to add what the ROLLOUT knows.

It needs two trees, because the two artifacts have different layer coverage by design:

  --activations-dir   a tree gathered with NO --signal-json, so every token has a .pt at
                      --layer (the probe's layer).
  --lens-dir          a CSV-only tree (--no-save-activations) over the SAME trajectories,
                      covering every layer, which is where the scores come from. Defaults to
                      --activations-dir when one tree holds both.

`--probe-type` replaces what used to be a second copy of this file. `next_action` labels a
token with its step's `agent_action`; `grid_tile` labels every CELL of the step's grid and
puts per-class counts on the row instead of a single verdict. Everything else -- the tree
walk, the lens tables, the .pt lookup -- is shared, which is the point: a difference between
an action row and a grid row is the label and nothing else.

The label is a property of the STEP, shared by every token of it, which is exactly why the
eval trajectories must be disjoint from the probe's training set -- scoring a probe on a
different token of a trajectory it trained on leaks that trajectory's single label.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

from telos_interp.jlens_utils.jlens_csv import output_start, step_folder_index
from telos_interp.jlens_utils.methods import scored_methods
from telos_interp.loudness_analysis import columns as cols
from telos_interp.loudness_analysis import provenance, signals
from telos_interp.loudness_analysis.lens_io import (
    find_act_folder,
    load_trajectory,
    read_lens_tables,
    trajectory_dirs,
)
from telos_interp.loudness_analysis.probes import DEFAULT_PROBE_TYPE, get_probe_type, probe_type_names


def lens_cells(counts: dict, masses: dict, lenses: list[str], key: tuple[int, int], layer: int):
    """(decoded token, the lens score columns) for one (step, abs_pos).

    Order matches `columns.score_columns`: count, mass at `layer`, best layer, best mass.
    """
    token = ""
    cells: list = []
    for lens in lenses:
        sc = counts.get(lens, {}).get(key)
        mm = masses.get(lens, {}).get(key, {})
        token = token or (sc.token if sc else "")
        best_layer, best = ("", "")
        if mm:
            best_layer = max(mm, key=lambda ly: mm[ly])
            best = mm[best_layer]
        cells += [sc.total() if sc else "", mm.get(layer, ""), best_layer, best]
    return token, cells


def build_parser(probe_type_name: str | None = None) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--probe-type",
        default=DEFAULT_PROBE_TYPE,
        choices=probe_type_names(),
        help="What the probes decode. Decides the label, the features and the row shape "
        "(default: %(default)s).",
    )
    ap.add_argument("--probe", type=Path, action="append", required=True, help="Trained probe .pt (repeatable).")
    ap.add_argument("--activations-dir", type=Path, required=True, help="Tree holding the per-token .pt files.")
    ap.add_argument("--lens-dir", type=Path, default=None, help="Tree holding the lens CSVs (default: --activations-dir).")
    ap.add_argument("--trajectories-dir", type=Path, required=True, help="Trajectory JSONs.")
    ap.add_argument("--signal-json", type=Path, required=True, help="Signal vocabulary JSON.")
    ap.add_argument(
        "--signal-name",
        default=None,
        help="What that vocabulary measures; names the loudness columns. Inferred from the "
        "filename when omitted.",
    )
    ap.add_argument("--layer", type=int, default=15, help="Layer the probe reads (default 15).")
    ap.add_argument("--out", type=Path, required=True, help="Per-token CSV to write.")
    ap.add_argument("--signal-classes", "--direction-classes", dest="direction_classes", default="all")
    ap.add_argument("--full-probs", action="store_true", help="Also write each class's probability.")
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=None, help="Process at most N trajectories.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Probe-type flags are added after the type is known, so `--help` shows only the relevant
    # ones and an unrelated flag is an error rather than a silent no-op.
    if probe_type_name:
        get_probe_type(probe_type_name).add_arguments(ap)
    return ap


def main() -> int:
    # Two passes: the first only to learn --probe-type, the second with its flags registered.
    pre, _ = build_parser().parse_known_args()
    args = build_parser(pre.probe_type).parse_args()

    ptype = get_probe_type(args.probe_type)
    signal = signals.resolve(args.signal_name, args.signal_json)
    lens_dir = args.lens_dir or args.activations_dir

    probes = {}
    for path in args.probe:
        probes[ptype.probe_key(path)] = ptype.load_probe(path)
    print(f"{len(probes)} {args.probe_type} probe(s): {', '.join(probes)}", flush=True)

    signal_tokens = signal.load(args.signal_json, args.direction_classes)
    folders = trajectory_dirs(lens_dir)
    if args.limit:
        folders = folders[: args.limit]
    print(f"{len(folders)} trajectory folder(s) under {lens_dir}", flush=True)

    lenses = list(scored_methods())
    score_cols: list[str] = []
    for lens in lenses:
        score_cols += cols.score_columns(lens, signal.name, args.layer)
    header = ptype.prefix_columns() + score_cols + ptype.result_columns(list(probes), args.full_probs)

    cfg = provenance.RunConfig("loudness_analysis/score_probes_per_token.py")
    cfg.measurement(lens=lenses[0], signal=signal.name, layer=args.layer, signal_json=args.signal_json)
    cfg.input("activations", args.activations_dir)
    cfg.input("lens_tables", lens_dir)
    cfg.input("trajectories", args.trajectories_dir)
    cfg.params.update(
        {
            "probe_type": args.probe_type,
            "probes": sorted(probes),
            "layer": args.layer,
            "lenses": lenses,
            "signal_classes": args.direction_classes,
            "full_probs": args.full_probs,
        }
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = skipped_missing_pt = 0
    with open(args.out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)

        for fi, folder in enumerate(folders, 1):
            stem = folder.name
            traj = load_trajectory(args.trajectories_dir, stem)
            if traj is None:
                print(f"  {stem}: no trajectory JSON, skipped", flush=True)
                continue

            counts, masses = read_lens_tables(folder, stem, lenses, signal_tokens)
            universe = sorted({k for d in list(counts.values()) + list(masses.values()) for k in d})
            if not universe:
                continue

            act_folder = find_act_folder(args.activations_dir, folder, stem)
            if act_folder is None:
                continue

            rows, acts, states = [], [], []
            starts: dict[int, tuple[int, int]] = {}
            step_states: dict[int, object] = {}
            for step, abs_pos in universe:
                if step not in starts:
                    try:
                        starts[step] = (step_folder_index(traj, step), output_start(traj, step))
                    except (IndexError, KeyError):
                        starts[step] = (-1, -1)
                folder_idx, start = starts[step]
                if folder_idx < 0:
                    continue

                # Cached: the label (or the cell draw) is a property of the step, and a
                # trajectory's ~240 tokens must not redo it 240 times.
                if step not in step_states:
                    step_states[step] = ptype.step_state(traj, step, stem, args)
                state = step_states[step]
                if state is None:
                    continue

                token_idx = abs_pos - start
                pt = act_folder / f"layer_{args.layer}" / f"step_{folder_idx}" / "output" / f"{token_idx}.pt"
                if not pt.exists():
                    skipped_missing_pt += 1
                    continue

                token, cells = lens_cells(counts, masses, lenses, (step, abs_pos), args.layer)
                rows.append(ptype.row_prefix(stem, traj, step, abs_pos, token_idx, token, state) + cells)
                acts.append(torch.load(pt, map_location="cpu", weights_only=True).float())
                states.append(state)

            if not rows:
                continue
            results = ptype.score(probes, acts, states, args.batch_size, args.device, args.full_probs)
            for row, result in zip(rows, results, strict=True):
                writer.writerow(row + result)
            written += len(rows)
            if fi % 25 == 0 or fi == len(folders):
                print(f"  [{fi}/{len(folders)}] {written} token rows", flush=True)

    print(f"\nwrote {written} token row(s) -> {args.out}")
    if skipped_missing_pt:
        print(f"  {skipped_missing_pt} token(s) had no .pt at layer {args.layer} and were skipped")

    cfg.rows("token_rows", written)
    cfg.rows("skipped_missing_pt", skipped_missing_pt)
    cfg.outputs.append(args.out)
    cfg.write(args.out.parent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
