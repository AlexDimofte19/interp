#!/usr/bin/env python3
"""Which single layer is most direction-loaded, averaged over a whole activation tree.

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

    python scripts/jlens_layer_profile.py /workspace/activations \
        --signal-json /workspace/jlens/direction_tokens_full.json \
        --direction-score logprob_mass --lens jlens --out layer_profile.json

prints the per-layer table, names the argmax layer, and writes the numbers as JSON. Feed
the layer to `split_next_action_manifest.py --single-layer L` (or to a
`prepare_activations_for_probing --layers L`).

A logprob score needs CSVs carrying the `top_i_logprob` columns; `--direction-score count`
works on every CSV ever written.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from telos_interp.jlens_utils import (  # noqa: E402
    DEFAULT_SCORE,
    LayerProfile,
    artifact_layers,
    format_profile,
    load_direction_tokens,
    read_direction_scores,
    read_mass_meta,
    score_artifact_path,
    score_names,
    scored_methods,
)


def trajectory_folders(activations_dir: Path) -> list[Path]:
    """Every folder that could hold an analysis CSV, size-sharded tree or flat.

    `jlens_reasoning_tokens.py` writes `size{S}/{stem}/{stem}_{lens}_analysis.csv`; a
    single-size run writes `{stem}/...`. Both are just "directories containing a CSV", so
    glob for the CSVs and take their parents rather than encoding the layout twice.
    """
    folders = {path.parent for path in activations_dir.rglob("*_analysis.csv")}
    folders |= {path.parent for path in activations_dir.rglob("*_direction_mass.csv")}
    return sorted(folders)


def profile_tree(
    activations_dir: Path,
    signal_json: Path,
    *,
    lens: str = "jlens",
    direction_score: str = DEFAULT_SCORE,
    direction_classes: str = "all",
    top_k: int = 20,
    max_trajectories: int | None = None,
    layers: list[int] | None = None,
    verbose: bool = False,
) -> LayerProfile:
    """Accumulate the per-layer mean direction score over every trajectory in the tree.

    `layers` pins the layer set so a trajectory whose CSV covers fewer layers still counts
    at all of them (missing ones at the score's `empty`). Left out, it is taken from the
    first CSV read — the lens covers the same layers for every trajectory of a run, and
    letting each CSV define its own would make the denominators disagree.
    """
    direction_tokens = load_direction_tokens(signal_json, direction_classes)
    profile = LayerProfile(score_mode=direction_score)
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
        if verbose:
            print(f"  {folder.name}: {len(scores)} tokens", flush=True)
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("activations_dir", type=Path, help="tree holding the per-trajectory analysis CSVs")
    parser.add_argument(
        "--signal-json",
        type=Path,
        required=True,
        help="JSON mapping UP/DOWN/LEFT/RIGHT to token strings "
        "(/workspace/jlens/direction_tokens_full.json on the GPU host)",
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
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    profile = profile_tree(
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

    print(f"{args.lens} / {args.direction_score} over {profile.tokens} (token, trajectory) rows")
    # A mass table's numbers depend on the vocabulary it was gathered against, and this repo
    # points two different ones at the same trees. Say which, rather than let the reader
    # assume it was --signal-json.
    meta = (
        read_mass_meta(score_artifact_path(folders[0], args.lens, args.direction_score))
        if (folders := trajectory_folders(args.activations_dir))
        else {}
    )
    if meta:
        print(
            f"  mass vocabulary: {meta.get('signal_json')} "
            f"({meta.get('num_direction_tokens')} tokens, classes={meta.get('direction_classes')})"
        )
    print(format_profile(profile))
    print(f"\nbest layer: {profile.best_layer()}")
    print(f"  split_next_action_manifest.py ... --single-layer {profile.best_layer()}")

    if args.out:
        summary = profile.to_dict()
        summary.update(
            lens=args.lens,
            activations_dir=str(args.activations_dir),
            direction_classes=args.direction_classes,
            top_k=args.top_k,
        )
        args.out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
