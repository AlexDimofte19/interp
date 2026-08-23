"""Prune an already-gathered activation tree down to the jlens selection.

Damage control for trajectories gathered before `jlens_reasoning_tokens.py` learned to
filter as it writes. Those folders hold a `.pt` for *every* (reasoning token, layer) — a
700-token chain over 17 layers is ~12k files and ~68 MB per step — of which a probe reads
around 40 tokens' worth.

This applies the same `jlens_top_filter` negatively: keep what the filter selects, delete
everything else. Because both scripts go through that one function, pruning a full tree
lands on exactly the files a filtered gather would have written — which is what
`tests/test_delete_non_jlens_selected.py` asserts byte-for-byte.

Two things it will not do:

  * touch the `{stem}_jlens_analysis.csv`. It is both the analysis artefact and the marker
    that says a trajectory has been processed; the whole point of pruning is that the CSV
    survives while the activations it describes mostly do not.
  * delete anything without `--apply`. The default is a dry run that reports what it would
    remove.

It also writes the same `{stem}_jlens_selection.json` a filtered gather writes, which is
what makes a pruned trajectory usable by
`prepare_activations_for_probing --token-selection recorded_jlens|recorded_random`. That
record is the *only* surviving trace of the random control arm: after pruning, a uniform
draw over the reasoning chain can no longer be made, so it has to be read back rather than
recomputed.

No torch, so this runs anywhere the tree is mounted.

Usage:
  python scripts/delete_non_jlens_selected.py \
    --activations-dir /workspace/activations/jlens_reasoning_tokens \
    --trajectories-dir /workspace/trajectories/reveng/trajectories_train_single_step \
    --signal-json data/jlens/direction_tokens_full.json \
    --apply
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.jlens_reasoning_tokens import parse_name  # noqa: E402
from telos_interp.jlens_utils import (  # noqa: E402
    DEFAULT_ALWAYS_LAYERS,
    DEFAULT_JLENS_CSV_SUFFIX,
    build_record,
    jlens_top_filter,
    output_start,
    record_path,
    step_folder_index,
    to_disk_coords,
    write_selection_record,
)


@dataclass
class Outcome:
    """What happened to one trajectory."""

    stem: str
    status: str  # "pruned" | "skipped"
    reason: str = ""
    kept: int = 0
    deleted: int = 0
    freed: int = 0


def model_folder(trajectory_folder: Path) -> Path | None:
    """The `<model>` folder inside a trajectory folder.

    Reimplemented rather than imported from `telos_interp.activation_loading`, which pulls
    in torch — this script has no other reason to load it.
    """
    subdirs = sorted(p for p in trajectory_folder.iterdir() if p.is_dir())
    for item in subdirs:
        if "__" in item.name:
            return item
    return subdirs[0] if subdirs else None


def find_trajectory_json(trajectories_dir: Path, stem: str) -> Path | None:
    """`{dir}/size{S}/{stem}.json`, falling back to a flat `{dir}/{stem}.json`."""
    size = parse_name(stem)["size"]
    candidates = [trajectories_dir / f"size{size}" / f"{stem}.json"] if size else []
    candidates.append(trajectories_dir / f"{stem}.json")
    return next((p for p in candidates if p.exists()), None)


def matches_filters(stem: str, sizes: set[str] | None, complexities: set[str] | None) -> bool:
    name = parse_name(stem)
    return (sizes is None or name["size"] in sizes) and (complexities is None or name["comp"] in complexities)


def find_jlens_csvs(activations_dir: Path) -> list[Path]:
    """Locate every trajectory's jlens CSV without walking the activation tree.

    The CSV sits at `{root}/size{S}/{stem}/{stem}_jlens_analysis.csv`, or `{root}/{stem}/...`
    for a trajectory whose filename carries no size. `rglob` finds those too -- but only
    after descending into every `{model}/layer_N/step_M/output/` folder underneath, which is
    the tens of millions of `.pt` files this script exists to delete. Globbing at the two
    known depths keeps the search proportional to the number of trajectories instead.
    """
    found = set(activations_dir.glob(f"*/*/*{DEFAULT_JLENS_CSV_SUFFIX}"))
    found.update(activations_dir.glob(f"*/*{DEFAULT_JLENS_CSV_SUFFIX}"))
    return sorted(found)


def prune_empty_dirs(root: Path) -> None:
    """Remove directories left empty by the deletion, deepest first."""
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def prune_trajectory(csv_path: Path, args) -> Outcome:
    """Filter one trajectory and delete the activations it does not select."""
    trajectory_folder = csv_path.parent
    stem = trajectory_folder.name

    model = model_folder(trajectory_folder)
    if model is None:
        return Outcome(stem, "skipped", "no model folder")

    trajectory_json = find_trajectory_json(args.trajectories_dir, stem)
    if trajectory_json is None:
        # The JSON is the authoritative source of output_start, which is what maps a CSV row
        # onto a filename. Guessing it on a destructive path is not worth the convenience.
        return Outcome(stem, "skipped", "no trajectory JSON")
    trajectory_data = json.loads(trajectory_json.read_text())

    kept = jlens_top_filter(
        args.signal_json,
        csv_path,
        num_tokens=args.select_num_tokens,
        num_layers=args.select_num_layers,
        always_layers=args.always_layers,
        random_tokens=args.select_random_tokens,
        random_layers=args.select_random_layers,
        seed=args.select_seed,
        seed_key=stem,
        candidate_layers=args.candidate_layers,
        direction_classes=args.direction_classes,
        top_k=args.jlens_top_k,
    )
    kept = to_disk_coords(kept, trajectory_data)
    keep_paths = kept.activation_paths(model)

    if not keep_paths:
        return Outcome(stem, "skipped", "filter selected nothing")

    absent = [p for p in keep_paths if not p.exists()]
    if absent and not args.tolerate_missing:
        # Either the coordinate mapping is wrong or this trajectory was gathered with a
        # different layer set. Deleting on that basis would destroy the wrong files.
        return Outcome(stem, "skipped", f"{len(absent)}/{len(keep_paths)} selected files missing")

    on_disk = list(model.rglob("*.pt"))
    doomed = [p for p in on_disk if p not in keep_paths]
    freed = sum(p.stat().st_size for p in doomed)

    if args.apply:
        for path in doomed:
            path.unlink()
        prune_empty_dirs(model)
        write_selection_record(
            record_path(trajectory_folder),
            build_record(
                kept,
                stem=stem,
                model=model.name,
                config={
                    "signal_json": str(args.signal_json),
                    "direction_classes": args.direction_classes,
                    "num_tokens": args.select_num_tokens,
                    "num_layers": args.select_num_layers,
                    "always_layers": list(args.always_layers),
                    "random_tokens": args.select_random_tokens,
                    "random_layers": args.select_random_layers,
                    "seed": args.select_seed,
                    "top_k": args.jlens_top_k,
                    "candidate_layers": args.candidate_layers,
                    "pruned_from": len(on_disk),
                },
                output_starts=_output_starts(trajectory_data),
            ),
        )

    return Outcome(stem, "pruned", "", kept=len(on_disk) - len(doomed), deleted=len(doomed), freed=freed)


def _output_starts(trajectory_data: dict) -> dict[int, int]:
    """{step folder index: that step's output_start}, for the record's abs_pos column."""
    starts: dict[int, int] = {}
    for index in range(len(trajectory_data.get("steps", []))):
        try:
            starts[step_folder_index(trajectory_data, index)] = output_start(trajectory_data, index)
        except (IndexError, KeyError):
            continue
    return starts


def human(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}TB"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--activations-dir", type=Path, required=True,
                    help="Root of the gather_activations-style tree to prune.")
    ap.add_argument("--trajectories-dir", type=Path, required=True,
                    help="Trajectory JSONs, used to map a CSV row's abs_pos onto a .pt filename. "
                         "A trajectory whose JSON is missing is left completely untouched.")
    ap.add_argument("--signal-json", type=Path, required=True,
                    help="JSON mapping UP/DOWN/LEFT/RIGHT to token strings, e.g. "
                         "data/jlens/direction_tokens_full.json.")
    ap.add_argument("--apply", action="store_true",
                    help="Actually delete. Without it this is a dry run that only reports.")
    ap.add_argument("--tolerate-missing", action="store_true",
                    help="Prune even when some selected files are absent. Off by default: a "
                         "selection pointing at files that do not exist usually means the "
                         "coordinate mapping is wrong, and deleting on that basis is unrecoverable.")
    ap.add_argument("--sizes", default=None, help="Comma-separated grid sizes to prune, e.g. '11,15'.")
    ap.add_argument("--complexities", default=None, help="Comma-separated complexities, e.g. '0.0,0.2'.")
    ap.add_argument("--select-num-tokens", type=int, default=20)
    ap.add_argument("--select-num-layers", type=int, default=3)
    ap.add_argument("--select-always-layers", default=",".join(str(i) for i in DEFAULT_ALWAYS_LAYERS),
                    help="Layers kept for every selected token regardless of score (default 15). "
                         "Added on top of --select-num-layers. Empty string to disable.")
    ap.add_argument("--select-random-tokens", type=int, default=20,
                    help="Size of the matched control arm to preserve (default 20). Set 0 only if "
                         "you accept that no uniform-draw control can ever be recovered from these "
                         "trajectories again.")
    ap.add_argument("--select-random-layers", type=int, default=None)
    ap.add_argument("--select-seed", type=int, default=42)
    ap.add_argument("--direction-classes", default="all")
    ap.add_argument("--jlens-top-k", type=int, default=20)
    ap.add_argument("--candidate-layers", default=None,
                    help="Comma-separated layer pool to choose from; default is every layer the "
                         "CSV has rows for.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N trajectories. Worth using for a first --apply run: "
                         "check the result on a handful before committing to the whole tree.")
    ap.add_argument("--verbose", action="store_true", help="One line per trajectory.")
    args = ap.parse_args()

    args.always_layers = tuple(int(p) for p in args.select_always_layers.split(",") if p.strip())
    args.candidate_layers = (
        [int(p) for p in args.candidate_layers.split(",") if p.strip()] if args.candidate_layers else None
    )
    sizes = {s.strip() for s in args.sizes.split(",")} if args.sizes else None
    complexities = {c.strip() for c in args.complexities.split(",")} if args.complexities else None

    print(f"scanning {args.activations_dir} for trajectories...", flush=True)
    csvs = find_jlens_csvs(args.activations_dir)
    csvs = [p for p in csvs if matches_filters(p.parent.name, sizes, complexities)]
    if not csvs:
        raise SystemExit(f"no {DEFAULT_JLENS_CSV_SUFFIX} files under {args.activations_dir}")
    if args.limit is not None:
        csvs = csvs[: args.limit]

    mode = "APPLYING" if args.apply else "DRY RUN (pass --apply to delete)"
    print(f"{mode}: {len(csvs)} trajectory folder(s) under {args.activations_dir}", flush=True)
    if args.select_random_tokens == 0:
        print("WARNING: --select-random-tokens 0 discards the matched control arm permanently",
              flush=True)

    # Reported as we go rather than collected first: each trajectory means listing and
    # unlinking ~12k files, so a batch of thousands is a long time to sit silent.
    outcomes: list[Outcome] = []
    for index, csv_path in enumerate(csvs, start=1):
        outcome = prune_trajectory(csv_path, args)
        outcomes.append(outcome)
        if args.verbose or outcome.status == "skipped":
            detail = outcome.reason if outcome.status == "skipped" else (
                f"kept {outcome.kept}, deleted {outcome.deleted} ({human(outcome.freed)})"
            )
            print(f"  [{index}/{len(csvs)}] {outcome.status:<8} {outcome.stem}: {detail}", flush=True)
        elif index % 25 == 0 or index == len(csvs):
            freed = human(sum(o.freed for o in outcomes))
            print(f"  [{index}/{len(csvs)}] {freed} so far", flush=True)

    pruned = [o for o in outcomes if o.status == "pruned"]
    skipped = [o for o in outcomes if o.status == "skipped"]
    print("")
    print(f"{'would prune' if not args.apply else 'pruned'}: {len(pruned)} trajectory folder(s)")
    print(f"  kept    {sum(o.kept for o in pruned)} activation file(s)")
    print(f"  deleted {sum(o.deleted for o in pruned)} activation file(s)")
    print(f"  freed   {human(sum(o.freed for o in pruned))}")
    if skipped:
        print(f"skipped: {len(skipped)} (untouched) -- rerun with --verbose for the reasons")
    if not args.apply:
        print("\nNothing was deleted. Re-run with --apply.")


if __name__ == "__main__":
    main()
