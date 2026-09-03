#!/usr/bin/env python3
"""
Plot per-class probe accuracy as a function of distance, one panel per grid
complexity level.

Consumes the JSON produced by the ``eval_cognitive_map_probe_per_distance``
command (see ``telos_interp/commands/eval_cognitive_map_probe_per_distance``).
That file stores, for each complexity level, a per-class breakdown of metrics at
every (Chebyshev) distance bucket to the goal and to the agent:

    by_complexity_distance
      └─ "<complexity>"
           └─ by_goal_distance_per_class  /  by_agent_distance_per_class
                └─ "<class_index>"
                     └─ "<distance>" -> {accuracy, precision, recall, f1,
                                         gt_support, predicted}

Class indices are decoded to grid symbols (A, #, G, _, ...) via the top-level
``class_labels`` mapping, which the eval command writes into the JSON.

For each requested distance axis (goal and/or agent) this produces one figure: a
2x3 grid of complexity panels, with X = distance, Y = per-class accuracy
(class-conditional recall), and one line per class. Points are only drawn where a
class actually has ground-truth support at that distance, and classes with no
support anywhere (e.g. padding ``+``) are dropped.

Usage
-----
    python scripts/plot_per_class_accuracy_by_distance.py results.json
    python scripts/plot_per_class_accuracy_by_distance.py results.json --axis goal
    python scripts/plot_per_class_accuracy_by_distance.py results.json --out-dir figures/
    python scripts/plot_per_class_accuracy_by_distance.py results.json --metric f1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; remove if you want interactive windows
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Categorical palette (dataviz reference, light mode), assigned in fixed slot
# order so a class keeps the same colour across every panel and figure.
PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

# Chart chrome (light surface)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

# Axis name -> key inside each complexity block of ``by_complexity_distance``.
AXIS_TO_KEY = {
    "goal": "by_goal_distance_per_class",
    "agent": "by_agent_distance_per_class",
}


def _complexity_levels(by_complexity_distance: dict) -> list[str]:
    """Return complexity keys sorted numerically (e.g. '0.0', '0.2', ..., '1.0')."""
    return sorted(by_complexity_distance.keys(), key=float)


def _collect_classes(by_complexity_distance: dict, per_class_key: str) -> list[str]:
    """Class ids (as strings) with >0 ground-truth support at any complexity/distance.

    Ordered by integer class id for a stable colour/marker assignment.
    """
    present: set[str] = set()
    for block in by_complexity_distance.values():
        for cls_id, dist_map in block[per_class_key].items():
            if any(cm["gt_support"] > 0 for cm in dist_map.values()):
                present.add(cls_id)
    return sorted(present, key=int)


def make_figure(
    data: dict,
    axis_name: str,
    out_path: Path,
    metric: str = "accuracy",
) -> None:
    """Render one figure (all complexity panels) for a single distance axis."""
    per_class_key = AXIS_TO_KEY[axis_name]
    by_cd = data["by_complexity_distance"]
    class_labels = data["class_labels"]
    complexities = _complexity_levels(by_cd)

    classes = _collect_classes(by_cd, per_class_key)
    if not classes:
        raise ValueError(f"No classes with ground-truth support for axis '{axis_name}'")

    color_of = {cls: PALETTE[i % len(PALETTE)] for i, cls in enumerate(classes)}
    marker_of = {cls: MARKERS[i % len(MARKERS)] for i, cls in enumerate(classes)}

    # Shared x-range across panels: max distance where any class has support.
    max_dist = 0
    for block in by_cd.values():
        for cls in classes:
            for dk, cm in block[per_class_key].get(cls, {}).items():
                if cm["gt_support"] > 0:
                    max_dist = max(max_dist, int(dk))

    # Panel grid sized to the number of complexity levels (defaults to 2x3 for 6).
    ncols = 3 if len(complexities) > 4 else len(complexities)
    nrows = -(-len(complexities) // ncols)  # ceil division
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.25 * nrows), sharex=True, sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    flat_axes = axes.flat

    for ax, comp in zip(flat_axes, complexities, strict=False):
        ax.set_facecolor(SURFACE)
        pivot = by_cd[comp][per_class_key]
        for cls in classes:
            # Only plot distances where this class has ground-truth support.
            pts = sorted((int(dk), cm[metric]) for dk, cm in pivot.get(cls, {}).items() if cm["gt_support"] > 0)
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(
                xs,
                ys,
                color=color_of[cls],
                marker=marker_of[cls],
                markersize=6,
                linewidth=2,
                markeredgecolor=SURFACE,
                markeredgewidth=0.6,
                label=class_labels[cls],
            )

        ax.set_title(f"Complexity {comp}", color=INK, fontsize=12)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlim(-0.5, max_dist + 0.5)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.tick_params(colors=MUTED)
        for spine in ax.spines.values():
            spine.set_color(GRID)

    # Hide any unused panels.
    for ax in list(flat_axes)[len(complexities) :]:
        ax.set_visible(False)

    for ax in axes[-1, :]:
        ax.set_xlabel(f"Distance to {axis_name} (Chebyshev)", color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel(
            f"{metric.capitalize()} (per-class recall)" if metric == "accuracy" else metric.capitalize(), color=INK
        )

    # One shared legend; identity is colour + marker, never colour alone.
    handles = [
        Line2D(
            [0],
            [0],
            color=color_of[cls],
            marker=marker_of[cls],
            markersize=7,
            linewidth=2,
            markeredgecolor=SURFACE,
            label=class_labels[cls],
        )
        for cls in classes
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(classes),
        frameon=False,
        title="Class",
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        f"Per-class probe {metric} vs distance to {axis_name}, by grid complexity",
        color=INK,
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}  (classes: {[class_labels[c] for c in classes]}, max_dist={max_dist})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-class probe accuracy vs distance, one panel per complexity."
    )
    parser.add_argument(
        "results_json",
        type=Path,
        help="Path to the eval_cognitive_map_probe_per_distance JSON results file",
    )
    parser.add_argument(
        "--axis",
        choices=["goal", "agent", "both"],
        default="both",
        help="Which distance axis to plot (default: both, producing two figures)",
    )
    parser.add_argument(
        "--metric",
        choices=["accuracy", "precision", "recall", "f1"],
        default="accuracy",
        help="Per-class metric to plot on the Y axis (default: accuracy)",
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

    axes = ["goal", "agent"] if args.axis == "both" else [args.axis]
    for axis_name in axes:
        out_path = out_dir / f"{stem}__per_class_{args.metric}_by_{axis_name}_distance.png"
        make_figure(data, axis_name, out_path, metric=args.metric)


if __name__ == "__main__":
    main()
