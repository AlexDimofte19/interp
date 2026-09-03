#!/usr/bin/env python3
"""
Plot the object (goal / agent) prediction rate as a function of distance, faceted
by grid size, by complexity, or by both — computed directly from the
per-prediction CSV emitted by ``eval_cognitive_map_probe_per_distance``.

This is the CSV-based sibling of ``plot_object_prediction_rate_by_distance.py``
(which reads the aggregated JSON and only facets by complexity). Working from the
raw per-cell rows lets us slice along ``grid_size`` and combine dimensions.

For the object class (goal ``G`` or agent ``A``), at each Chebyshev distance ``d``
from that object we compute a spatial *localization / smearing* profile:

    object_prediction_rate(d) = (# cells at distance d predicted as the object)
                               / (# cells at distance d)

At ``d = 0`` this is the true-positive rate at the object's own location; at
``d >= 1`` it is the false-positive rate — how often the probe wrongly "sees" the
object that far away. Padding cells carry no distance (NaN) and are dropped, so
the denominator matches the JSON's per-distance ``total_samples``.

The ``--by`` flag selects the panel breakdown:
  * ``size``       — one panel per grid size (default).
  * ``complexity`` — one panel per complexity (reproduces the original figure).
  * ``both``       — a 2-D grid, rows = grid size, cols = complexity.

Usage
-----
    python scripts/plot_object_prediction_rate_by_distance_from_csv.py preds.csv
    python scripts/plot_object_prediction_rate_by_distance_from_csv.py preds.csv --by both
    python scripts/plot_object_prediction_rate_by_distance_from_csv.py preds.csv --object goal
    python scripts/plot_object_prediction_rate_by_distance_from_csv.py preds.csv --out-dir figures/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; remove if you want interactive windows
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

# Chart chrome (light surface) — shared with plot_object_prediction_rate_by_distance.py
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

# Bar colours: distance 0 (true positive) vs distance >=1 (false positive).
COLOR_TP = "#2a78d6"  # blue  — object's own location
COLOR_FP = "#e34948"  # red   — false positives away from the object

# Object -> (distance column in the CSV, grid symbol).
OBJECT_TO_AXIS = {"goal": "dist_to_goal", "agent": "dist_to_agent"}
OBJECT_TO_SYMBOL = {"goal": "G", "agent": "A"}

# ``--by`` -> (title prefix, filename suffix).
BY_TO_SUFFIX = {"size": "size", "complexity": "complexity", "both": "size_x_complexity"}
BY_TO_TITLE = {"size": "grid size", "complexity": "grid complexity", "both": "grid size and complexity"}


def _object_rates(sub: pd.DataFrame, axis_col: str, symbol: str) -> dict[int, float]:
    """{distance: fraction of cells predicted as ``symbol``} for one panel's rows.

    Padding rows (NaN distance) are dropped by ``groupby``, so the denominator is
    the count of real cells at each distance — matching the JSON ``total_samples``.
    """
    s = sub.dropna(subset=[axis_col])
    if s.empty:
        return {}
    total = s.groupby(axis_col).size()
    predicted = s[s["pred_symbol"] == symbol].groupby(axis_col).size()
    predicted = predicted.reindex(total.index, fill_value=0)
    return {int(d): predicted[d] / total[d] for d in total.index}


def _facets(df: pd.DataFrame, by: str):
    """Return (nrows, ncols, panels) where panels is a list of (ax_pos, title, subframe).

    ``ax_pos`` is a flat panel index for 1-D breakdowns, or an (i, j) tuple for the
    2-D ``both`` grid.
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


def make_figure(df: pd.DataFrame, obj: str, by: str, out_path: Path) -> None:
    """Render one figure (all panels) for a single object and breakdown."""
    axis_col = OBJECT_TO_AXIS[obj]
    symbol = OBJECT_TO_SYMBOL[obj]

    nrows, ncols, panels, sizes, comps = _facets(df, by)

    # Precompute per-panel {distance: rate} and shared axis extents.
    panel_rates = []
    max_dist = 0
    max_rate = 0.0
    for pos, title, sub in panels:
        rates = _object_rates(sub, axis_col, symbol)
        if rates:
            max_dist = max(max_dist, max(rates))
            max_rate = max(max_rate, max(rates.values()))
        panel_rates.append((pos, title, rates))

    figsize = (3.6 * ncols, 3.0 * nrows) if by == "both" else (5 * ncols, 4.25 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)

    for pos, title, rates in panel_rates:
        ax = axes[pos[0]][pos[1]] if by == "both" else axes.flat[pos]
        ax.set_facecolor(SURFACE)
        dists = sorted(rates.keys())
        heights = [rates[d] for d in dists]
        colors = [COLOR_TP if d == 0 else COLOR_FP for d in dists]
        ax.bar(dists, heights, color=colors, width=0.85, edgecolor=SURFACE, linewidth=0.6)
        ax.set_xlim(-0.6, max_dist + 0.6)
        ax.set_ylim(0, max_rate * 1.08 if max_rate > 0 else 1.0)
        ax.grid(True, axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        if by != "both":
            ax.set_title(title, color=INK, fontsize=12)

    if by == "both":
        # Matrix-style labelling: complexity across the top, size down the left.
        for j, c in enumerate(comps):
            axes[0][j].set_title(f"Comp {c}", color=INK, fontsize=12)
        for i, s in enumerate(sizes):
            axes[i][0].set_ylabel(f"Size {int(s)}", color=INK, fontsize=12)
        for ax in axes[-1, :]:
            ax.set_xlabel(f"Distance to {obj} (Chebyshev)", color=INK)
        fig.supylabel(f"Fraction of cells predicted as {symbol}", color=INK, x=0.005)
    else:
        # Hide unused panels in the 1-D grid.
        for ax in list(axes.flat)[len(panels) :]:
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
        f"{obj.capitalize()} ({symbol}) prediction rate vs distance to {obj}, by {BY_TO_TITLE[by]}",
        color=INK,
        fontsize=15,
    )
    fig.tight_layout(rect=(0.04 if by == "both" else 0, 0.05, 1, 0.96))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}  (object='{symbol}', by={by}, max_dist={max_dist}, max_rate={max_rate:.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot object (goal/agent) prediction rate vs distance from the per-prediction CSV."
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
        "--object",
        choices=["goal", "agent", "both"],
        default="both",
        help="Which object to plot (default: both, producing two figures)",
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
        usecols=["grid_size", "complexity", "pred_symbol", "dist_to_goal", "dist_to_agent"],
    )

    out_dir = args.out_dir if args.out_dir is not None else args.predictions_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.predictions_csv.stem

    objects = ["goal", "agent"] if args.object == "both" else [args.object]
    for obj in objects:
        out_path = out_dir / f"{stem}__{obj}_prediction_rate_by_distance__by_{BY_TO_SUFFIX[args.by]}.png"
        make_figure(df, obj, args.by, out_path)


if __name__ == "__main__":
    main()
