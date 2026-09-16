#!/usr/bin/env python3
"""Which single layer is most direction-loaded, averaged over a whole activation tree.

This is the **join**: `build_loudness_tables.py` leaves one direction-mass table (or one
analysis CSV) per trajectory, and nothing yet relates them. This walks the tree, folds
every table into one per-layer mean, and names the argmax.

Per-token layer selection gives each token its own best layer, and a probe trained on the
result pools rows from layers that are not in a shared basis — one weight vector cannot
read layer 7 and layer 22 the same way. Fixing **one** layer for the dataset removes that,
and this is how the layer is chosen: the highest mean direction score over every reasoning
token of every trajectory.

Why not read it off a prepared manifest
---------------------------------------
Because a manifest only holds the layers that were *selected*. A layer appears there when
it won some token's top-`M`, so its mean is conditioned on having won — while layer 15,
force-kept for every token (`top_filter.DEFAULT_ALWAYS_LAYERS`), carries an unconditional
mean. Comparing the two systematically favours the rarely-selected layers.

The analysis CSVs have no such hole: every reasoning token is scored at every layer the
lens covers, selected or not. They also survive pruning — `delete_non_jlens_selected.py`
removes `.pt` files, never CSVs — so this runs on the pruned tree as it stands.

    python scripts/jlens_layer_profile.py /workspace/activations \\
        --signal-json /workspace/jlens/direction_tokens_full.json \\
        --direction-score logprob_mass --lens jlens --out layer_profile.json

prints the per-layer table, names the argmax layer, says whether that argmax is
distinguishable from the layer below it, and writes the numbers as JSON (and, with
`--out-csv`, as one tidy row per layer for plotting). Feed the layer to
`split_next_action_manifest.py --single-layer L` (or to a
`prepare_activations_for_probing --layers L`).

Read the `z` before pinning the layer
-------------------------------------
The mean says which layer won; `z` says whether winning meant anything. It is the gap to
the runner-up over the standard error of that gap, **paired within each trajectory and
pooled over trajectories** — not over tokens, because the tokens of one chain share a
prompt, a grid and a train of thought and are nowhere near independent. Under about 2, the
argmax is a coin flip between two adjacent layers and the choice should be made on some
other ground. That matters most on a sampled gather (`--data_sample_p`), whose means are
unbiased but noisier; this script prints the sample fraction it finds in the sidecars so
the number is never mistaken for a full-dataset one.

A logprob score needs CSVs carrying the `top_i_logprob` columns; `--direction-score count`
works on every CSV ever written. `--signal-json` is required for those, and optional for
`logprob_mass_full`, whose vocabulary was fixed at gather time and is named in each table's
`.meta.json` sidecar — passing one there cannot change the numbers, so this refuses to let
it look as though it could.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from telos_interp.jlens_utils import (  # noqa: E402
    DEFAULT_SCORE,
    LayerProfile,
    artifact_layers,
    format_profile,
    format_separation,
    get_score,
    load_direction_tokens,
    read_direction_scores,
    read_mass_meta,
    score_artifact_path,
    score_names,
    scored_methods,
)

# Sidecar fields that must agree across the tree for the join to mean anything. The
# fingerprint is a hash of the vocabulary's *contents*, so it catches a file edited in
# place between two halves of a gather, which the path never would.
PINNED_META = ("signal_name", "signal_fingerprint", "direction_classes", "data_sample_p")


def trajectory_folders(activations_dir: Path) -> list[Path]:
    """Every folder that could hold an analysis CSV, size-sharded tree or flat.

    `build_loudness_tables.py` writes `size{S}/{stem}/{stem}_{lens}_analysis.csv`; a
    single-size run writes `{stem}/...`. Both are just "directories containing a CSV", so
    glob for the CSVs and take their parents rather than encoding the layout twice.
    """
    folders = {path.parent for path in activations_dir.rglob("*_analysis.csv")}
    folders |= {path.parent for path in activations_dir.rglob("*_direction_mass.csv")}
    return sorted(folders)


def check_meta_agrees(metas: list[dict]) -> dict:
    """One sidecar's worth of facts, or raise naming the field the tree disagrees on.

    Averaging two vocabularies' mass tables together produces a number with no referent,
    and this repo deliberately points two vocabularies at the same trees. Same for a
    partially re-gathered tree where half the trajectories were sampled and half were not:
    the mean is still unbiased, but the layer profile's error bars are not, and nothing
    downstream would ever notice.
    """
    if not metas:
        return {}
    first = metas[0]
    for field in PINNED_META:
        values = {json.dumps(meta.get(field), sort_keys=True) for meta in metas}
        if len(values) > 1:
            raise SystemExit(
                f"the direction-mass tables under this tree disagree on {field!r}: "
                f"{', '.join(sorted(values))}. They cannot be averaged together — "
                "re-gather the odd ones out, or profile the two sets separately."
            )
    return first


def profile_tree(
    activations_dir: Path,
    signal_json: Path | None,
    *,
    lens: str = "jlens",
    direction_score: str = DEFAULT_SCORE,
    direction_classes: str = "all",
    top_k: int = 20,
    max_trajectories: int | None = None,
    layers: list[int] | None = None,
    verbose: bool = False,
) -> tuple[LayerProfile, list[dict]]:
    """Accumulate the per-layer mean direction score over every trajectory in the tree.

    `layers` pins the layer set so a trajectory whose CSV covers fewer layers still counts
    at all of them (missing ones at the score's `empty`). Left out, it is taken from the
    first CSV read — the lens covers the same layers for every trajectory of a run, and
    letting each CSV define its own would make the denominators disagree.

    Returns the profile and every sidecar found, so the caller can check they agree. One
    trajectory is one `add()`, which is what makes the profile's error bars clustered by
    trajectory rather than by token.
    """
    source = get_score(direction_score).source
    direction_tokens = load_direction_tokens(signal_json, direction_classes) if signal_json else set()
    profile = LayerProfile(score_mode=direction_score)
    metas: list[dict] = []
    folders = trajectory_folders(activations_dir)
    if max_trajectories is not None:
        folders = folders[:max_trajectories]

    for folder in folders:
        path = score_artifact_path(folder, lens, direction_score)
        if path is None or not path.exists():
            continue
        if layers is None:
            layers = artifact_layers(path, direction_score)
        scores = read_direction_scores(path, direction_tokens, top_k=top_k, score_mode=direction_score)
        profile.add(scores, layers)
        if source == "mass":
            metas.append(read_mass_meta(path))
        if verbose:
            print(f"  {folder.name}: {len(scores)} tokens", flush=True)
    return profile, metas


def describe_provenance(meta: dict) -> list[str]:
    """What the sidecars say about numbers the table itself cannot show."""
    if not meta:
        return []
    lines = [
        f"  mass vocabulary: {meta.get('signal_json')} "
        f"({meta.get('num_direction_tokens')} tokens, classes={meta.get('direction_classes')}, "
        f"fingerprint={meta.get('signal_fingerprint')})"
    ]
    if meta.get("weights_model_id"):
        lines.append(f"  weights loaded from: {meta['weights_model_id']}")
    if meta.get("data_sample_p") is not None:
        lines.append(
            f"  SAMPLED GATHER: p={meta['data_sample_p']} (seed {meta.get('data_sample_seed')}) — "
            "the means are unbiased, the error bars are wider than a full pass would give"
        )
    return lines


def write_csv(path: Path, profile: LayerProfile) -> None:
    """One tidy row per layer: what a plot or a spreadsheet wants out of the join."""
    summary = profile.to_dict()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "mean", "se", "rows", "score_mode", "is_best"])
        for row in summary["layers"]:
            writer.writerow(
                [
                    row["layer"],
                    f"{row['mean']:.6f}",
                    "" if row["se"] is None else f"{row['se']:.6f}",
                    row["rows"],
                    summary["score_mode"],
                    int(row["layer"] == summary["best_layer"]),
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("activations_dir", type=Path, help="tree holding the per-trajectory analysis CSVs")
    parser.add_argument(
        "--signal-json",
        type=Path,
        default=None,
        help="JSON mapping UP/DOWN/LEFT/RIGHT to token strings "
        "(/workspace/jlens/direction_tokens_full.json on the GPU host). Required for the "
        "top-k scores; refused for logprob_mass_full, whose vocabulary is the one in the "
        "table's sidecar and cannot be changed after the gather",
    )
    parser.add_argument(
        "--lens", choices=scored_methods(), default="jlens", help="which lens' CSV to profile (default jlens)"
    )
    parser.add_argument(
        "--direction-score",
        choices=score_names(),
        default=DEFAULT_SCORE,
        help="see jlens_utils/scoring.py. 'logprob_mass_full' profiles the "
        "direction-mass table instead of the analysis CSV, which is the "
        "unbiased-over-the-vocabulary number as well as the "
        "unbiased-over-layers one",
    )
    parser.add_argument("--direction-classes", default="all", help="'all' or e.g. 'UP,DOWN'")
    parser.add_argument("--top-k", type=int, default=20, help="how many top_i columns to scan")
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="profile only the first N trajectories (the mean converges fast; "
        "use this to sanity-check before a full pass)",
    )
    parser.add_argument(
        "--layers", default=None, help="comma-separated layer pool (default: the layers the first CSV covers)"
    )
    parser.add_argument("--out", type=Path, default=None, help="write the numbers here as JSON")
    parser.add_argument("--out-csv", type=Path, default=None, help="write one row per layer here, for plotting")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    source = get_score(args.direction_score).source
    if source == "mass" and args.signal_json is not None:
        raise SystemExit(
            f"--direction-score {args.direction_score} reads the direction-mass table, whose vocabulary was "
            "fixed at gather time and is recorded in each table's .meta.json. Passing --signal-json here "
            "would have no effect on the numbers; drop it (this run prints the vocabulary it finds)."
        )
    if source != "mass" and args.signal_json is None:
        raise SystemExit(f"--direction-score {args.direction_score} scores the analysis CSV and needs --signal-json")

    layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    profile, metas = profile_tree(
        args.activations_dir,
        args.signal_json,
        lens=args.lens,
        direction_score=args.direction_score,
        direction_classes=args.direction_classes,
        top_k=args.top_k,
        max_trajectories=args.max_trajectories,
        layers=layers,
        verbose=args.verbose,
    )
    if not profile.tokens:
        raise SystemExit(f"no {args.lens} analysis CSVs found under {args.activations_dir}")

    print(
        f"{args.lens} / {args.direction_score} over {profile.tokens} (token, trajectory) rows "
        f"from {profile.clusters} trajectories"
    )
    # A mass table's numbers depend on the vocabulary it was gathered against, and this repo
    # points two different ones at the same trees. Say which, rather than let the reader
    # assume it was --signal-json -- and check every table agrees before averaging them.
    for line in describe_provenance(check_meta_agrees(metas)):
        print(line)
    print(format_profile(profile))
    print(f"\nbest layer: {profile.best_layer()}")
    print(f"  {format_separation(profile)}")
    sep = profile.separation()
    if sep is not None and sep["z"] < 2.0:
        print("  WARNING: under z=2 the argmax is not distinguishable from the layer below it")
    print(f"  split_next_action_manifest.py ... --single-layer {profile.best_layer()}")

    if args.out:
        summary = profile.to_dict()
        summary.update(
            lens=args.lens,
            activations_dir=str(args.activations_dir),
            direction_classes=args.direction_classes,
            top_k=args.top_k,
            mass_meta=check_meta_agrees(metas) or None,
        )
        args.out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(f"wrote {args.out}")
    if args.out_csv:
        write_csv(args.out_csv, profile)
        print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
