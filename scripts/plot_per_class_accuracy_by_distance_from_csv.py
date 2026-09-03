#!/usr/bin/env python3
"""
Plot per-class probe accuracy as a function of distance, faceted by grid size, by
complexity, or by both — computed directly from the per-prediction CSV emitted by
``eval_cognitive_map_probe_per_distance``.

This is the CSV-based sibling of ``plot_per_class_accuracy_by_distance.py`` (which
reads the aggregated JSON and only facets by complexity). Working from the raw
per-cell rows lets us slice along ``grid_size`` and combine dimensions.

For each class and each Chebyshev distance ``d`` (to the goal or agent), metrics
are computed with the same definitions as the eval accumulators:

    gt_support = # cells with gt == class at distance d
    tp         = # of those predicted correctly
    predicted  = # cells predicted as class at distance d
    recall = accuracy = tp / gt_support      precision = tp / predicted
    f1 = harmonic mean(precision, recall)

Points are only drawn where a class actually has ground-truth support at that
distance; classes with no support anywhere (e.g. padding ``+``, which carries a
NaN distance and is dropped) never appear.

The ``--by`` flag selects the panel breakdown:
  * ``size``       — one panel per grid size (default).
  * ``complexity`` — one panel per complexity (reproduces the original figure).
  * ``both``       — a 2-D grid, rows = grid size, cols = complexity.

Usage
-----
    python scripts/plot_per_class_accuracy_by_distance_from_csv.py preds.csv
    python scripts/plot_per_class_accuracy_by_distance_from_csv.py preds.csv --by both
    python scripts/plot_per_class_accuracy_by_distance_from_csv.py preds.csv --axis goal
    python scripts/plot_per_class_accuracy_by_distance_from_csv.py preds.csv --metric f1
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; remove if you want interactive windows
import matplotlib.pyplot as plt
import pandas as pd
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

# Axis name -> distance column in the CSV.
AXIS_TO_COL = {"goal": "dist_to_goal", "agent": "dist_to_agent"}

# ``--by`` -> (filename suffix, title phrase).
BY_TO_SUFFIX = {"size": "size", "complexity": "complexity", "both": "size_x_complexity"}
BY_TO_TITLE = {"size": "grid size", "complexity": "grid complexity", "both": "grid size and complexity"}


def _per_class_metrics(sub: pd.DataFrame, axis_col: str) -> dict[int, dict[int, dict]]:
    """Return ``{class_idx: {distance: {accuracy, precision, recall, f1, gt_support}}}``.

    Padding rows (NaN distance) are dropped by ``dropna``, mirroring the eval
    accumulators which never see padding cells in the distance buckets.
    """
    s = sub.dropna(subset=[axis_col])
    if s.empty:
        return {}
    correct = s["gt_idx"] == s["pred_idx"]
    gt_grp = s.groupby([axis_col, "gt_idx"])
    gt_support = gt_grp.size()
    tp = correct.groupby([s[axis_col], s["gt_idx"]]).sum()
    predicted = s.groupby([axis_col, "pred_idx"]).size()

    out: dict[int, dict[int, dict]] = defaultdict(dict)
    for (dist, cls), supp in gt_support.items():
        t = int(tp.loc[(dist, cls)])
        pc = int(predicted.get((dist, cls), 0))
        recall = t / supp if supp > 0 else 0.0
        precision = t / pc if pc > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        out[int(cls)][int(dist)] = {
            "accuracy": recall,
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "gt_support": int(supp),
        }
    return out


def _facets(df: pd.DataFrame, by: str):
    """Return (nrows, ncols, panels, sizes, comps).

    ``panels`` is a list of (ax_pos, title, subframe); ``ax_pos`` is a flat index
    for 1-D breakdowns or an (i, j) tuple for the 2-D ``both`` grid.
    """
    if by == "both":
        sizes = sorted(df["grid_size"].unique())
        comps = sorted(df["complexity"].unique())
        panels = []
        for i, s in enumerate(sizes):
            for j, c in enumerate(comps):
                sub = df[(df["grid_size"] == s) & (df["complexity"] == c)]
                panels.append(((i, j), f"Size {int(s)} · Comp {c}", sub))
        return len(sizes), len(comps), panels, sizes, comps

    if by == "size":
        vals = sorted(df["grid_size"].unique())
        col, label = "grid_size", lambda v: f"Size {int(v)}"
    else:  # complexity
        vals = sorted(df["complexity"].unique())
        col, label = "complexity", lambda v: f"Complexity {v}"
    n = len(vals)
    ncols = 3 if n > 4 else n
    nrows = -(-n // ncols)  # ceil division
    panels = [(k, label(v), df[df[col] == v]) for k, v in enumerate(vals)]
    return nrows, ncols, panels, None, None


def make_figure(df: pd.DataFrame, axis_name: str, by: str, out_path: Path, metric: str = "accuracy") -> None:
    """Render one figure (all panels) for a single distance axis and breakdown."""
    axis_col = AXIS_TO_COL[axis_name]

    # Classes with ground-truth support anywhere (padding has NaN distance -> absent).
    valid = df.dropna(subset=[axis_col])
    classes = sorted(int(c) for c in valid["gt_idx"].unique())
    if not classes:
        raise ValueError(f"No classes with ground-truth support for axis '{axis_name}'")
    # Class index -> symbol, from the (gt_idx, gt_symbol) pairs in the CSV.
    idx_to_symbol = dict(zip(valid["gt_idx"].astype(int), valid["gt_symbol"]))

    color_of = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(classes)}
    marker_of = {c: MARKERS[i % len(MARKERS)] for i, c in enumerate(classes)}

    nrows, ncols, panels, sizes, comps = _facets(df, by)

    # Precompute per-panel metrics and the shared x-range.
    panel_metrics = []
    max_dist = 0
    for pos, title, sub in panels:
        m = _per_class_metrics(sub, axis_col)
        for dist_map in m.values():
            for dk, cm in dist_map.items():
                if cm["gt_support"] > 0:
                    max_dist = max(max_dist, dk)
        panel_metrics.append((pos, title, m))

    figsize = (3.6 * ncols, 3.0 * nrows) if by == "both" else (5 * ncols, 4.25 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)

    for pos, title, m in panel_metrics:
        ax = axes[pos[0]][pos[1]] if by == "both" else axes.flat[pos]
        ax.set_facecolor(SURFACE)
        for cls in classes:
            dist_map = m.get(cls, {})
            pts = sorted((dk, cm[metric]) for dk, cm in dist_map.items() if cm["gt_support"] > 0)
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
                label=idx_to_symbol.get(cls, str(cls)),
            )
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlim(-0.5, max_dist + 0.5)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.tick_params(colors=MUTED)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        if by != "both":
            ax.set_title(title, color=INK, fontsize=12)

    ylabel = f"{metric.capitalize()} (per-class recall)" if metric == "accuracy" else metric.capitalize()
    if by == "both":
        # Matrix-style labelling: complexity across the top, size down the left.
        for j, c in enumerate(comps):
            axes[0][j].set_title(f"Comp {c}", color=INK, fontsize=12)
        for i, s in enumerate(sizes):
            axes[i][0].set_ylabel(f"Size {int(s)}", color=INK, fontsize=12)
        for ax in axes[-1, :]:
            ax.set_xlabel(f"Distance to {axis_name} (Chebyshev)", color=INK)
        fig.supylabel(ylabel, color=INK, x=0.005)
    else:
        for ax in list(axes.flat)[len(panels) :]:
            ax.set_visible(False)
        for ax in axes[-1, :]:
            ax.set_xlabel(f"Distance to {axis_name} (Chebyshev)", color=INK)
        for ax in axes[:, 0]:
            ax.set_ylabel(ylabel, color=INK)

    # One shared legend; identity is colour + marker, never colour alone.
    handles = [
        Line2D(
            [0],
            [0],
            color=color_of[c],
            marker=marker_of[c],
            markersize=7,
            linewidth=2,
            markeredgecolor=SURFACE,
            label=idx_to_symbol.get(c, str(c)),
        )
        for c in classes
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
        f"Per-class probe {metric} vs distance to {axis_name}, by {BY_TO_TITLE[by]}",
        color=INK,
        fontsize=15,
    )
    fig.tight_layout(rect=(0.04 if by == "both" else 0, 0.05, 1, 0.96))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}  (by={by}, classes={[idx_to_symbol.get(c, c) for c in classes]}, max_dist={max_dist})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-class probe accuracy vs distance from the per-prediction CSV."
    )
    parser.add_argument(
        "predictions_csv",
        type=Path,
        help="Path to the eval_cognitive_map_probe_per_distance per-prediction CSV",
    )
    parser.add_argument(
        "--by",
        choices=["size", "complexity", "both"],
        default="size",
        help="Panel breakdown: grid size, complexity, or both (2-D grid). Default: size",
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
        help="Directory for the PNG figures (default: alongside the input CSV)",
    )
    args = parser.parse_args()

    if not args.predictions_csv.is_file():
        raise FileNotFoundError(f"CSV file not found: {args.predictions_csv}")

    df = pd.read_csv(
        args.predictions_csv,
        usecols=["grid_size", "complexity", "gt_idx", "gt_symbol", "pred_idx", "dist_to_goal", "dist_to_agent"],
    )

    out_dir = args.out_dir if args.out_dir is not None else args.predictions_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.predictions_csv.stem

    axes = ["goal", "agent"] if args.axis == "both" else [args.axis]
    for axis_name in axes:
        out_path = out_dir / f"{stem}__per_class_{args.metric}_by_{axis_name}_distance__by_{BY_TO_SUFFIX[args.by]}.png"
        make_figure(df, axis_name, args.by, out_path, metric=args.metric)


if __name__ == "__main__":
    main()
