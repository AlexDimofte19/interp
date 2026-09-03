#!/usr/bin/env python3
"""Score a trained next_action probe on EVERY reasoning token of a trajectory set.

The arm scripts answer "how well does a probe do on the tokens the lens SELECTED". This
answers the prior question: how does the probe do on each token, and does the lens score
predict where it does well? One row per (trajectory, step, token), carrying both the probe's
verdict and every lens score available for that token, so the two can be crossed afterwards
without re-running anything.

It needs two trees, because the two artifacts have different layer coverage by design:

  --activations-dir   a tree gathered with NO --signal-json, so every token has a .pt at
                      --layer (the probe's layer).
  --lens-dir          a CSV-only tree (--no-save-activations) over the SAME trajectories,
                      covering every layer, which is where the scores come from. Defaults to
                      --activations-dir when one tree holds both.

The label is the step's `agent_action`: every token of a step shares it, which is exactly why
the eval trajectories must be disjoint from the probe's training set -- scoring a probe on a
different token of a trajectory it trained on leaks that trajectory's single label.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from telos_interp.commands.prepare_activations_for_probing.prepare_activations_for_probing_fn import (  # noqa: E402
    NEXT_ACTION_TO_ID,
)
from telos_interp.commands.train_next_action_probe.train_next_action_probe_fn import (  # noqa: E402
    ACTION_ID_TO_NAME,
    NextActionProbe,
)
from telos_interp.jlens_utils.jlens_csv import (  # noqa: E402
    load_direction_tokens,
    output_start,
    read_direction_scores,
    step_folder_index,
)
from telos_interp.jlens_utils.methods import scored_methods  # noqa: E402

# Column order for --full-probs. Fixed and probe-independent: NEXT_ACTION_TO_ID's order, so
# {probe}_p_LEFT means the same column for every arm regardless of its own label_to_idx.
ACTION_COLS = ("LEFT", "UP", "RIGHT", "DOWN")

MASS_PREFIX = ("size", "complexity", "run", "step", "reasoning_pos", "abs_pos", "token", "agent_action")


def trajectory_dirs(root: Path) -> list[Path]:
    """Every folder under `root` holding at least one lens artifact, sizeN nesting or flat."""
    seen = {p.parent for p in root.glob("size*/*/*_analysis.csv")}
    seen |= {p.parent for p in root.glob("*/*_analysis.csv")}
    return sorted(seen)


def read_mass_columns(path: Path) -> dict[tuple[int, int], dict[int, float]]:
    """{(step, abs_pos): {layer: log P(direction)}} from a direction-mass table."""
    out: dict[tuple[int, int], dict[int, float]] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            layers = {}
            for key, val in row.items():
                if key and key.startswith("L") and key[1:].isdigit() and val not in (None, ""):
                    layers[int(key[1:])] = float(val)
            out[(int(row["step"]), int(row["abs_pos"]))] = layers
    return out


def load_trajectory(trajectories_dir: Path, stem: str) -> dict | None:
    hits = list(trajectories_dir.glob(f"size*/{stem}.json")) + list(trajectories_dir.glob(f"{stem}.json"))
    return json.loads(hits[0].read_text()) if hits else None


def read_lens_tables(
    folder: Path, stem: str, lenses: list[str], direction_tokens: dict
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Per-lens (count scores, direction-mass columns) for one trajectory, keyed by (step, abs_pos)."""
    counts: dict[str, dict] = {}
    masses: dict[str, dict] = {}
    for lens in lenses:
        acsv = folder / f"{stem}_{lens}_analysis.csv"
        if acsv.exists():
            counts[lens] = read_direction_scores(acsv, direction_tokens, score_mode="count")
        mcsv = folder / f"{stem}_{lens}_direction_mass.csv"
        if mcsv.exists():
            masses[lens] = read_mass_columns(mcsv)
    return counts, masses


def find_act_folder(activations_dir: Path, folder: Path, stem: str) -> Path | None:
    """The `{...}/{model}` folder holding this trajectory's .pt tree, sizeN-nested or flat.

    Falls back to a model folder sitting inside the lens folder itself, for the case where one
    tree holds both artifacts.
    """
    for d in (activations_dir / folder.parent.name / stem, activations_dir / stem):
        if d.exists():
            return next((c for c in d.iterdir() if c.is_dir()), None)
    return next((d for d in folder.iterdir() if d.is_dir()), None)


def score_batch(
    probes: dict, batch: "torch.Tensor", labels: "torch.Tensor", batch_size: int, full_probs: bool
) -> tuple[dict, dict, dict, dict]:
    """Run every probe over one trajectory's activations -> (preds, corrects, p_true, p_full).

    `p_full` is empty unless `full_probs`; when present its columns are ACTION_COLS order, NOT
    the probe's own, so the same column name means the same action across arms.
    """
    preds, corrects, ptrue, pfull = {}, {}, {}, {}
    for name, probe in probes.items():
        out_pred, out_p, out_full = [], [], []
        order = [probe.label_to_idx[NEXT_ACTION_TO_ID[a]] for a in ACTION_COLS]
        for i in range(0, batch.shape[0], batch_size):
            chunk = batch[i : i + batch_size]
            probs = probe.predict_proba(chunk).cpu()
            idx = probs.argmax(dim=-1)
            out_pred.append(torch.tensor([probe.idx_to_label[j.item()] for j in idx]))
            cols = torch.tensor([probe.label_to_idx.get(int(v), 0) for v in labels[i : i + chunk.shape[0]]])
            out_p.append(probs.gather(1, cols[:, None]).squeeze(1))
            if full_probs:
                out_full.append(probs[:, order])
        preds[name] = torch.cat(out_pred)
        ptrue[name] = torch.cat(out_p)
        corrects[name] = (preds[name] == labels).int()
        if full_probs:
            pfull[name] = torch.cat(out_full)
    return preds, corrects, ptrue, pfull


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--probe",
        type=Path,
        required=True,
        action="append",
        help="A saved next_action probe .pt. Repeat to score several on one pass "
        "over the activations (they are loaded once and reused).",
    )
    ap.add_argument(
        "--activations-dir",
        type=Path,
        required=True,
        help="Tree holding every token's .pt at --layer (gathered with no --signal-json).",
    )
    ap.add_argument(
        "--lens-dir",
        type=Path,
        default=None,
        help="Tree holding the analysis CSVs and direction-mass tables. Defaults to --activations-dir.",
    )
    ap.add_argument(
        "--trajectories-dir",
        type=Path,
        required=True,
        help="Trajectory JSONs, for agent_action and the abs_pos -> filename offset.",
    )
    ap.add_argument(
        "--signal-json", type=Path, required=True, help="Direction vocabulary, for the top-k count/logprob scores."
    )
    ap.add_argument("--layer", type=int, default=15, help="Layer the probe reads (default 15).")
    ap.add_argument("--out", type=Path, required=True, help="Per-token CSV to write.")
    ap.add_argument("--direction-classes", default="all")
    ap.add_argument(
        "--full-probs",
        action="store_true",
        help="Also write the FULL 4-way softmax per token as {probe}_p_{ACTION}. "
        "{probe}_p_true is only the probability of the step's own action, which "
        "cannot say what else the probe believed; the commitment-boundary work "
        "needs p(the belief the model would report here) too.",
    )
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=None, help="Process at most N trajectories.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    lens_dir = args.lens_dir or args.activations_dir

    # Key by <parent dir>/<stem>: the same probe filename legitimately exists in several
    # probe directories (one per experiment), and a bare stem would silently overwrite.
    def probe_key(path: Path) -> str:
        return f"{path.parent.name}.{path.stem}".replace("next_action_probe_", "")

    probes = {probe_key(p): NextActionProbe.load(p, device=args.device) for p in args.probe}
    if len(probes) != len(args.probe):
        raise SystemExit("duplicate --probe paths: probe keys must be unique")
    print(f"loaded {len(probes)} probe(s): {', '.join(probes)}", flush=True)

    direction_tokens = load_direction_tokens(args.signal_json, args.direction_classes)
    folders = trajectory_dirs(lens_dir)
    if args.limit:
        folders = folders[: args.limit]
    print(f"{len(folders)} trajectory folder(s) under {lens_dir}", flush=True)

    lenses = list(scored_methods())
    score_cols: list[str] = []
    for lens in lenses:
        score_cols += [f"{lens}_count", f"{lens}_mass_L{args.layer}", f"{lens}_mass_best_layer", f"{lens}_mass_best"]
    header = (
        ["name", "size", "complexity", "step", "abs_pos", "token_idx", "token", "label", "label_name"]
        + score_cols
        + [f"{n}_pred" for n in probes]
        + [f"{n}_correct" for n in probes]
        + [f"{n}_p_true" for n in probes]
        + ([f"{n}_p_{a}" for n in probes for a in ACTION_COLS] if args.full_probs else [])
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

            counts, masses = read_lens_tables(folder, stem, lenses, direction_tokens)
            universe = sorted({k for d in list(counts.values()) + list(masses.values()) for k in d})
            if not universe:
                continue

            act_folder = find_act_folder(args.activations_dir, folder, stem)
            if act_folder is None:
                continue

            rows, acts = [], []
            starts: dict[int, tuple[int, int]] = {}
            for step, abs_pos in universe:
                if step not in starts:
                    try:
                        starts[step] = (step_folder_index(traj, step), output_start(traj, step))
                    except (IndexError, KeyError):
                        starts[step] = (-1, -1)
                folder_idx, start = starts[step]
                if folder_idx < 0:
                    continue
                token_idx = abs_pos - start
                pt = act_folder / f"layer_{args.layer}" / f"step_{folder_idx}" / "output" / f"{token_idx}.pt"
                if not pt.exists():
                    skipped_missing_pt += 1
                    continue
                action = traj["steps"][step].get("agent_action", "")
                label = NEXT_ACTION_TO_ID.get(action.upper())
                if label is None:
                    continue

                cells: list = []
                token = ""
                for lens in lenses:
                    sc = counts.get(lens, {}).get((step, abs_pos))
                    mm = masses.get(lens, {}).get((step, abs_pos), {})
                    token = token or (sc.token if sc else "")
                    best_layer, best = ("", "")
                    if mm:
                        best_layer = max(mm, key=lambda ly: mm[ly])
                        best = mm[best_layer]
                    cells += [sc.total() if sc else "", mm.get(args.layer, ""), best_layer, best]

                rows.append(
                    [
                        stem,
                        traj["grid_params"].get("grid_width", ""),
                        traj["grid_params"].get("grid_complexity", ""),
                        step,
                        abs_pos,
                        token_idx,
                        token,
                        label,
                        ACTION_ID_TO_NAME.get(label, ""),
                    ]
                    + cells
                )
                acts.append(torch.load(pt, map_location="cpu", weights_only=True).float())

            if not rows:
                continue
            batch = torch.stack(acts)
            labels = torch.tensor([r[7] for r in rows])
            preds, corrects, ptrue, pfull = score_batch(probes, batch, labels, args.batch_size, args.full_probs)

            for i, row in enumerate(rows):
                writer.writerow(
                    row
                    + [int(preds[n][i]) for n in probes]
                    + [int(corrects[n][i]) for n in probes]
                    + [f"{float(ptrue[n][i]):.6f}" for n in probes]
                    + ([f"{float(v):.6f}" for n in probes for v in pfull[n][i]] if args.full_probs else [])
                )
            written += len(rows)
            if fi % 25 == 0 or fi == len(folders):
                print(f"  [{fi}/{len(folders)}] {written} token rows", flush=True)

    print(f"\nwrote {written} token row(s) -> {args.out}")
    if skipped_missing_pt:
        print(f"  {skipped_missing_pt} token(s) had no .pt at layer {args.layer} and were skipped")


if __name__ == "__main__":
    main()
