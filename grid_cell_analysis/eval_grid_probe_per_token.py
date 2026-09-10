#!/usr/bin/env python3
"""Score a trained grid_tile ("cognitive map") probe on EVERY reasoning token of a set.

The grid twin of `scripts/eval_probe_per_token.py`, and it exists to answer a SPECIFICITY
question rather than a selection one. Log entries 37/38 established that the jlens
**direction** mass predicts where the *next action* is decodable: balanced accuracy climbs
monotonically across deciles of that score. The obvious worry is that the score is not
direction-specific at all -- that a "loud" token is merely an informative token, and any
probe would look better there. Reading the *grid cell identity* off the same tokens, ranked
by the same direction mass, is the control: a flat curve says the score is specific to the
action, a rising one says it is a generic saliency measure.

So the direction vocabulary is deliberately the right one to rank by here even though the
label is the grid. The grid vocabulary (`data/jlens/grid_tokens_full.json`) is a different
experiment and is NOT what this script is for.

One row per (trajectory, step, token) -- NOT per cell. A token owns C cells and its
per-class hit counts are what balanced accuracy is built from, so each row carries
`n_true_{c}` / `n_correct_{c}` for every cell class. Balanced accuracy for any bucket of
tokens is then a group-by: per class, sum the correct and sum the true, divide, and average
the classes. Emitting one row per cell instead would multiply the file by ~100 and buy
nothing, since no analysis needs an individual cell's verdict.

Feature layout matches the trainer exactly: `[activation(D), row, col] -> cell_id`, i.e. one
activation, D+2 wide. See `manifest_loader.build_flat_grid_tile`.

Cells are read at the grid's NATIVE size by default. `parse_grid_state` pads to the right
and bottom only, so a real cell's (row, col) is identical padded or not, and dropping the
padding class keeps balanced accuracy over cells the model could actually see.
`--pad-to-size` restores the padded view if the padding class is wanted.
"""

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from telos_interp.commands.train_cognitive_map_probe.train_cognitive_map_probe_fn import (  # noqa: E402
    CognitiveMapProbe,
)
from telos_interp.grid_utils import CELL_ID_TO_SYMBOL, parse_grid_state  # noqa: E402
from telos_interp.jlens_utils.jlens_csv import (  # noqa: E402
    load_direction_tokens,
    output_start,
    read_direction_scores,
    step_folder_index,
)
from telos_interp.jlens_utils.methods import scored_methods  # noqa: E402

# Fixed and probe-independent, so `n_true_3` means "empty cell" in every CSV this writes
# regardless of which classes a particular probe happened to be trained on.
CELL_CLASSES = tuple(sorted(CELL_ID_TO_SYMBOL))


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
    """The `{...}/{model}` folder holding this trajectory's .pt tree, sizeN-nested or flat."""
    for d in (activations_dir / folder.parent.name / stem, activations_dir / stem):
        if d.exists():
            return next((c for c in d.iterdir() if c.is_dir()), None)
    return next((d for d in folder.iterdir() if d.is_dir()), None)


def step_cells(
    traj: dict, step: int, pad_to_size: int | None, max_cells: int | None, seed: int, stem: str
) -> list[list[int]]:
    """The [row, col, cell_id] triples this step is scored on.

    The draw is seeded per (trajectory, step) rather than from a global stream, for the same
    reason `prepare_activations_for_probing` does it: two runs that consume different numbers
    of draws would otherwise be scored on different cells.
    """
    grid_state = traj["steps"][step].get("grid_state")
    if not grid_state:
        return []
    triples = parse_grid_state(grid_state, pad_to_size=pad_to_size)
    if max_cells is not None and len(triples) > max_cells:
        triples = random.Random(f"{stem}|{step}|{seed}").sample(triples, max_cells)
    return triples


def lens_cells(counts: dict, masses: dict, lenses: list[str], key: tuple[int, int], layer: int) -> tuple[str, list]:
    """(decoded token, the lens score columns) for one (step, abs_pos)."""
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


def collect_trajectory(
    traj: dict,
    stem: str,
    universe: list[tuple[int, int]],
    act_folder: Path,
    counts: dict,
    masses: dict,
    lenses: list[str],
    args: argparse.Namespace,
) -> tuple[list, list, list, dict, int]:
    """One trajectory's CSV rows, activations, per-token cell refs, cell cache, n missing .pt.

    Split out of `main` so the per-token filtering stays readable; this is the only place
    that decides which (step, abs_pos) survives into the output.

    Cells are a property of the STEP, so each step is parsed once and shared across that
    step's tokens -- a trajectory's ~240 tokens would otherwise re-parse and re-draw the
    same grid 240 times.
    """
    cells_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    starts: dict[int, tuple[int, int]] = {}
    rows: list = []
    acts: list = []
    cellrefs: list = []
    missing = 0

    for step, abs_pos in universe:
        if step not in starts:
            try:
                starts[step] = (step_folder_index(traj, step), output_start(traj, step))
            except (IndexError, KeyError):
                starts[step] = (-1, -1)
        folder_idx, start = starts[step]
        if folder_idx < 0:
            continue

        if step not in cells_cache:
            triples = step_cells(traj, step, args.pad_to_size, args.max_cells, args.seed, stem)
            if triples:
                pos = torch.tensor([[t[0], t[1]] for t in triples], dtype=torch.float32)
                lab = torch.tensor([t[2] for t in triples], dtype=torch.long)
            else:
                pos, lab = torch.empty(0, 2), torch.empty(0, dtype=torch.long)
            cells_cache[step] = (pos, lab)
        lab = cells_cache[step][1]
        if lab.numel() == 0:
            continue

        token_idx = abs_pos - start
        pt = act_folder / f"layer_{args.layer}" / f"step_{folder_idx}" / "output" / f"{token_idx}.pt"
        if not pt.exists():
            missing += 1
            continue

        token, score_cells = lens_cells(counts, masses, lenses, (step, abs_pos), args.layer)
        hist: dict[int, int] = defaultdict(int)
        for v in lab.tolist():
            hist[v] += 1

        rows.append(
            [
                stem,
                traj["grid_params"].get("grid_width", ""),
                traj["grid_params"].get("grid_complexity", ""),
                step,
                abs_pos,
                token_idx,
                token,
                int(lab.numel()),
            ]
            + [hist.get(c, 0) for c in CELL_CLASSES]
            + score_cells
        )
        acts.append(torch.load(pt, map_location="cpu", weights_only=True).float())
        cellrefs.append(step)

    return rows, acts, cellrefs, cells_cache, missing


def score_trajectory(
    probes: dict,
    acts: list[torch.Tensor],
    cellrefs: list[int],
    cells_cache: dict[int, tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
    device: str,
) -> dict[str, list[tuple[int, float, float, dict[int, int]]]]:
    """Per (token, probe): (n_correct, accuracy, mean p_true, {cell_class: n_correct}).

    Tokens are batched together rather than scored one at a time: a token contributes C rows
    of width D+2, and one 8k-row forward over several tokens is far cheaper than ~240
    forwards of ~100 rows. The activation is broadcast across its own cells, so the
    (row, col) columns are the only part that varies within a token.
    """
    out: dict[str, list] = {n: [] for n in probes}
    n_tokens = len(acts)
    i = 0
    while i < n_tokens:
        # Grow the batch until one more token would exceed batch_size rows.
        j, rows_in_batch = i, 0
        while j < n_tokens:
            c = int(cells_cache[cellrefs[j]][1].numel())
            if rows_in_batch and rows_in_batch + c > batch_size:
                break
            rows_in_batch += c
            j += 1

        chunk_x, chunk_y, spans = [], [], []
        for k in range(i, j):
            pos, lab = cells_cache[cellrefs[k]]
            a = acts[k].unsqueeze(0).expand(lab.numel(), -1)
            chunk_x.append(torch.cat([a, pos], dim=1))
            chunk_y.append(lab)
            spans.append(lab.numel())
        x = torch.cat(chunk_x).to(device)
        y = torch.cat(chunk_y)

        for name, probe in probes.items():
            probs = probe.predict_proba(x).cpu()
            pred = torch.tensor([probe.idx_to_label[t.item()] for t in probs.argmax(dim=-1)])
            # A label the probe never saw has no column; p_true is 0 there, which is the
            # honest reading -- the probe assigns it no mass at all.
            cols = torch.tensor([probe.label_to_idx.get(int(v), -1) for v in y])
            p_true = probs.gather(1, cols.clamp(min=0)[:, None]).squeeze(1)
            p_true[cols < 0] = 0.0
            correct = (pred == y).int()

            off = 0
            for span in spans:
                sl = slice(off, off + span)
                c_sl, y_sl = correct[sl], y[sl]
                per_class: dict[int, int] = defaultdict(int)
                for v, ok in zip(y_sl.tolist(), c_sl.tolist(), strict=True):
                    per_class[v] += ok
                out[name].append(
                    (int(c_sl.sum()), float(c_sl.float().mean()), float(p_true[sl].mean()), dict(per_class))
                )
                off += span
        i = j
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--probe",
        type=Path,
        required=True,
        action="append",
        help="A saved grid_tile probe .pt. Repeat to score several on one pass over the activations.",
    )
    ap.add_argument(
        "--activations-dir",
        type=Path,
        required=True,
        help="Tree holding every token's .pt at --layer (e.g. heldout360_l15).",
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
        help="Trajectory JSONs, for grid_state and the abs_pos -> filename offset.",
    )
    ap.add_argument(
        "--signal-json",
        type=Path,
        required=True,
        help="DIRECTION vocabulary. The ranking under test is the direction one even though "
        "the label is the grid -- that contrast is the whole point.",
    )
    ap.add_argument("--layer", type=int, default=15, help="Layer the probe reads (default 15).")
    ap.add_argument("--out", type=Path, required=True, help="Per-token CSV to write.")
    ap.add_argument("--direction-classes", default="all")
    ap.add_argument(
        "--pad-to-size",
        type=int,
        default=None,
        help="Pad every grid to this size, adding the padding class. Default: native size, real cells only.",
    )
    ap.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="Cap cells scored per (trajectory, step); seeded per step. Default: every cell.",
    )
    ap.add_argument(
        "--exclude-names",
        type=Path,
        default=None,
        help="File of trajectory names (one per line) to SKIP -- the probe's own training "
        "set. Scoring a probe on a trajectory it trained on leaks that trajectory's grid, "
        "which every token of it shares. Not needed when the eval tree is disjoint by "
        "construction, but the trees here are only PARTLY disjoint: heldout360 shares 0 "
        "trajectories with jlens_mass_l15 but 33 with jlens_reasoning_tokens.",
    )
    ap.add_argument("--seed", type=int, default=42, help="Seeds the per-step cell draw.")
    ap.add_argument("--batch-size", type=int, default=8192, help="Rows (token x cell) per forward.")
    ap.add_argument("--limit", type=int, default=None, help="Process at most N trajectories.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    lens_dir = args.lens_dir or args.activations_dir

    def probe_key(path: Path) -> str:
        return f"{path.parent.name}.{path.stem}".replace("grid_probe_", "").replace("cognitive_map_probe_", "")

    probes = {probe_key(p): CognitiveMapProbe.load(p, device=args.device) for p in args.probe}
    if len(probes) != len(args.probe):
        raise SystemExit("duplicate --probe paths: probe keys must be unique")
    for name, pr in probes.items():
        print(f"loaded {name}: input_dim={pr.input_dim} classes={sorted(pr.label_to_idx)}", flush=True)

    direction_tokens = load_direction_tokens(args.signal_json, args.direction_classes)
    folders = trajectory_dirs(lens_dir)
    if args.exclude_names:
        excluded = {line.strip() for line in args.exclude_names.read_text().splitlines() if line.strip()}
        before = len(folders)
        folders = [f for f in folders if f.name not in excluded]
        print(f"excluded {before - len(folders)} trajectory folder(s) named in {args.exclude_names}", flush=True)
    if args.limit:
        folders = folders[: args.limit]
    print(f"{len(folders)} trajectory folder(s) under {lens_dir}", flush=True)

    lenses = list(scored_methods())
    score_cols: list[str] = []
    for lens in lenses:
        score_cols += [f"{lens}_count", f"{lens}_mass_L{args.layer}", f"{lens}_mass_best_layer", f"{lens}_mass_best"]
    per_probe_cols: list[str] = []
    for n in probes:
        per_probe_cols += [f"{n}_n_correct", f"{n}_acc", f"{n}_mean_p_true"]
        per_probe_cols += [f"{n}_correct_{c}" for c in CELL_CLASSES]
    header = (
        ["name", "size", "complexity", "step", "abs_pos", "token_idx", "token", "n_cells"]
        + [f"n_true_{c}" for c in CELL_CLASSES]
        + score_cols
        + per_probe_cols
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = skipped_missing_pt = total_cells = 0
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
            act_folder = find_act_folder(args.activations_dir, folder, stem) if universe else None
            if act_folder is None:
                continue

            rows, acts, cellrefs, cells_cache, missing = collect_trajectory(
                traj, stem, universe, act_folder, counts, masses, lenses, args
            )
            skipped_missing_pt += missing
            if not rows:
                continue

            results = score_trajectory(probes, acts, cellrefs, cells_cache, args.batch_size, args.device)
            for i, row in enumerate(rows):
                extra: list = []
                for n in probes:
                    ncorr, acc, meanp, per_class = results[n][i]
                    extra += [ncorr, f"{acc:.6f}", f"{meanp:.6f}"]
                    extra += [per_class.get(c, 0) for c in CELL_CLASSES]
                writer.writerow(row + extra)
            written += len(rows)
            total_cells += sum(r[7] for r in rows)
            if fi % 25 == 0 or fi == len(folders):
                print(f"  [{fi}/{len(folders)}] {written} token rows, {total_cells} cell evals", flush=True)

    print(f"\nwrote {written} token row(s) ({total_cells} cell evaluations) -> {args.out}")
    if skipped_missing_pt:
        print(f"  {skipped_missing_pt} token(s) had no .pt at layer {args.layer} and were skipped")


if __name__ == "__main__":
    main()
