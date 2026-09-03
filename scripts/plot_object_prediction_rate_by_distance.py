#!/usr/bin/env python3
"""
Plot the object (goal / agent) prediction rate as a function of distance, one bar
panel per grid complexity level.

Consumes the JSON produced by the ``eval_cognitive_map_probe_per_distance``
command (see ``telos_interp/commands/eval_cognitive_map_probe_per_distance``).
For the object class (goal ``G`` or agent ``A``), at each Chebyshev distance ``d``
from that object we compute a spatial *localization / smearing* profile:

    object_prediction_rate(d) = (# cells at distance d predicted as the object)
                               / (# cells at distance d)

At ``d = 0`` this is the true-positive rate at the object's own location; at
``d >= 1`` it is the false-positive rate — how often the probe wrongly "sees" the
object that far away. Both quantities come straight from the eval JSON (no re-eval
needed):

    numerator   = by_complexity_distance[comp][<axis>][<dist>]["per_class"][<obj>]["predicted"]
    denominator = by_complexity_distance[comp][<axis>][<dist>]["total_samples"]

where ``<axis>`` is ``by_goal_distance`` / ``by_agent_distance`` and ``<obj>`` is
the class index whose ``class_labels`` symbol is ``G`` / ``A``.

For each requested object this produces one figure: a 2x3 grid of complexity
panels, each a bar plot with X = distance, Y = object-prediction-rate. The
``d = 0`` bar (true-positive location) is highlighted in a distinct colour from
the ``d >= 1`` false-positive bars.

Usage
-----
    python scripts/plot_object_prediction_rate_by_distance.py results.json
    python scripts/plot_object_prediction_rate_by_distance.py results.json --object goal
    python scripts/plot_object_prediction_rate_by_distance.py results.json --out-dir figures/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; remove if you want interactive windows
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Chart chrome (light surface) — shared with plot_per_class_accuracy_by_distance.py
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

# Bar colours: distance 0 (true positive) vs distance >=1 (false positive).
COLOR_TP = "#2a78d6"  # blue  — object's own location
COLOR_FP = "#e34948"  # red   — false positives away from the object

# Object -> (distance-axis key in each complexity block, grid symbol).
OBJECT_TO_AXIS = {"goal": "by_goal_distance", "agent": "by_agent_distance"}
OBJECT_TO_SYMBOL = {"goal": "G", "agent": "A"}


def _complexity_levels(by_complexity_distance: dict) -> list[str]:
    """Return complexity keys sorted numerically (e.g. '0.0', '0.2', ..., '1.0')."""
    return sorted(by_complexity_distance.keys(), key=float)


def _object_index(class_labels: dict, symbol: str) -> str:
    """Return the class-index string whose label is ``symbol`` (e.g. 'G' -> '2')."""
    for idx, sym in class_labels.items():
        if sym == symbol:
            return idx
    raise ValueError(f"Object symbol '{symbol}' not found in class_labels {class_labels}")


def make_figure(data: dict, obj: str, out_path: Path) -> None:
    """Render one figure (all complexity panels) for a single object."""
    axis_key = OBJECT_TO_AXIS[obj]
    symbol = OBJECT_TO_SYMBOL[obj]
    by_cd = data["by_complexity_distance"]
    obj_idx = _object_index(data["class_labels"], symbol)
    complexities = _complexity_levels(by_cd)

    # Per-complexity {distance: rate}, plus global max distance / rate for shared axes.
    rates_by_comp: dict[str, dict[int, float]] = {}
    max_dist = 0
    max_rate = 0.0
    for comp in complexities:
        dist_block = by_cd[comp][axis_key]
        rates: dict[int, float] = {}
        for dk, m in dist_block.items():
            total = m["total_samples"]
            if total <= 0:
                continue
            rate = m["per_class"][obj_idx]["predicted"] / total
            rates[int(dk)] = rate
            max_dist = max(max_dist, int(dk))
            max_rate = max(max_rate, rate)
        rates_by_comp[comp] = rates

    # Panel grid sized to the number of complexity levels (defaults to 2x3 for 6).
    ncols = 3 if len(complexities) > 4 else len(complexities)
    nrows = -(-len(complexities) // ncols)  # ceil division
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.25 * nrows), sharex=True, sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    flat_axes = axes.flat

    for ax, comp in zip(flat_axes, complexities, strict=False):
        ax.set_facecolor(SURFACE)
        rates = rates_by_comp[comp]
        dists = sorted(rates.keys())
        heights = [rates[d] for d in dists]
        colors = [COLOR_TP if d == 0 else COLOR_FP for d in dists]
        ax.bar(dists, heights, color=colors, width=0.85, edgecolor=SURFACE, linewidth=0.6)

        ax.set_title(f"Complexity {comp}", color=INK, fontsize=12)
        ax.set_xlim(-0.6, max_dist + 0.6)
        ax.set_ylim(0, max_rate * 1.08 if max_rate > 0 else 1.0)
        ax.grid(True, axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED)
        for spine in ax.spines.values():
            spine.set_color(GRID)

    # Hide any unused panels.
    for ax in list(flat_axes)[len(complexities) :]:
        ax.set_visible(False)

    for ax in axes[-1, :]:
        ax.set_xlabel(f"Distance to {obj} (Chebyshev)", color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"Fraction of cells predicted as {symbol}", color=INK)

    legend_handles = [
        Patch(facecolor=COLOR_TP, label="distance 0 (true positive)"),
        Patch(facecolor=COLOR_FP, label="distance ≥1 (false positive)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        f"{obj.capitalize()} ({symbol}) prediction rate vs distance to {obj}, by grid complexity",
        color=INK,
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}  (object='{symbol}' idx={obj_idx}, max_dist={max_dist}, max_rate={max_rate:.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot object (goal/agent) prediction rate vs distance, one bar panel per complexity."
    )
    parser.add_argument(
        "results_json",
        type=Path,
        help="Path to the eval_cognitive_map_probe_per_distance JSON results file",
    )
    parser.add_argument(
        "--object",
        choices=["goal", "agent", "both"],
        default="both",
        help="Which object to plot (default: both, producing two figures)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for the PNG figures (default: alongside the input JSON)",
    )
    args = parser.parse_args()

    if not args.results_json.is_file():
        raise FileNotFoundError(f"Results file not found: {args.results_json}")

    data = json.loads(args.results_json.read_text(encoding="utf-8"))
    if "by_complexity_distance" not in data or "class_labels" not in data:
        raise ValueError(
            "JSON does not look like eval_cognitive_map_probe_per_distance output "
            "(missing 'by_complexity_distance' or 'class_labels')"
        )

    out_dir = args.out_dir if args.out_dir is not None else args.results_json.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.results_json.stem

    objects = ["goal", "agent"] if args.object == "both" else [args.object]
    for obj in objects:
        out_path = out_dir / f"{stem}__{obj}_prediction_rate_by_distance.png"
        make_figure(data, obj, out_path)


if __name__ == "__main__":
    main()
