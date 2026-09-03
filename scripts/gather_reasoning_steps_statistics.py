#!/usr/bin/env python3
"""
Count .pt activation files per (maze size, complexity) category and visualise how
many reasoning steps (sentences) are used as a function of those variables.

Model of the data
------------------
Each .pt file == one sentence in a trajectory's reasoning chain. So for a single
trajectory directory (e.g. ``..._size9_comp1.0_989``) the number of .pt files is
the length of that trajectory's reasoning chain.

    n_steps(trajectory) = number of .pt files for that trajectory, AT A SINGLE LAYER.

We restrict to one layer because activations are duplicated across layers (the
same sentences are stored once per layer); counting across all layers would
multiply every count by the number of layers. Restricting to one layer is also
robust to *how* sentences are enumerated on disk:

    structure A:  .../layer_15/step_0/output/{0,1,2,...}.pt   (many pt, one step dir)
    structure B:  .../layer_15/step_{0,1,2}/output/0.pt        (one pt, many step dirs)

In both cases the count of .pt files under one layer == number of sentences.

Usage
-----
    python count_reasoning_steps.py /path/to/main_repo
    python count_reasoning_steps.py /path/to/root --layer 15 --out figures/
    python count_reasoning_steps.py /path/to/root --all-layers   # disable dedup
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless-safe; remove if you want interactive windows
import matplotlib.pyplot as plt

# --- parsing -----------------------------------------------------------------
SIZE_RE = re.compile(r"size(\d+)")
COMP_RE = re.compile(r"comp(\d+(?:\.\d+)?)")
LAYER_RE = re.compile(r"layer_(\d+)")
STEP_RE = re.compile(r"step_(\d+)")


def parse_pt_path(p: Path):
    """Return (trajectory, size, comp, layer, step) parsed from a .pt file path.

    The trajectory directory is identified as the single path component that
    contains BOTH a ``sizeN`` and a ``compN.M`` token, so model names like
    ``gpt-oss-20b`` are never mistaken for a size.
    """
    traj = size = comp = layer = step = None
    for part in p.parts:
        s, c = SIZE_RE.search(part), COMP_RE.search(part)
        if s and c:
            traj, size, comp = part, int(s.group(1)), float(c.group(1))
        if m := LAYER_RE.search(part):
            layer = int(m.group(1))
        if m := STEP_RE.search(part):
            step = int(m.group(1))
    return traj, size, comp, layer, step


def build_dataframe(root: Path) -> pd.DataFrame:
    pt_files = list(root.rglob("*.pt"))
    if not pt_files:
        sys.exit(f"No .pt files found under {root}")
    rows = []
    for p in pt_files:
        traj, size, comp, layer, step = parse_pt_path(p)
        if traj is None or size is None or comp is None:
            continue  # .pt not under a size/comp trajectory dir -> ignore
        rows.append(dict(path=str(p), traj=traj, size=size, comp=comp, layer=layer, step=step))
    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("Found .pt files, but none under a 'size..comp..' trajectory dir.")
    return df


# --- counting ----------------------------------------------------------------
def choose_layer(df: pd.DataFrame, requested):
    layers = sorted(df["layer"].dropna().unique().tolist())
    if not layers:
        return None
    if requested is not None:
        if requested not in layers:
            sys.exit(f"Requested layer {requested} not found. Available: {layers}")
        return requested
    if len(layers) == 1:
        return layers[0]
    pref = 15 if 15 in layers else layers[0]
    print(
        f"[warn] Multiple layers found {layers}. Using layer {pref} to avoid "
        f"double-counting sentences. Override with --layer or use --all-layers.",
        file=sys.stderr,
    )
    return pref


def per_trajectory_counts(df: pd.DataFrame, layer, use_all: bool) -> pd.DataFrame:
    sub = df if (use_all or layer is None) else df[df["layer"] == layer]
    return sub.groupby(["traj", "size", "comp"]).size().rename("n_steps").reset_index()


def print_diagnostics(df: pd.DataFrame) -> None:
    print("=" * 64)
    print("STRUCTURE DIAGNOSTICS")
    print("=" * 64)
    print(f"total .pt files (all layers/steps): {len(df):,}")
    print(f"trajectories                       : {df['traj'].nunique():,}")
    print(f"layers found                       : {sorted(df['layer'].dropna().unique())}")
    print(f"steps  found                       : {sorted(df['step'].dropna().unique())}")
    print(f"sizes  found                       : {sorted(df['size'].unique())}")
    print(f"comps  found                       : {sorted(df['comp'].unique())}")
    # how many pt files sit in a single output dir, and how many step dirs / traj
    by_out = df.groupby(["traj", "layer", "step"]).size()
    print(
        f"pt files per (traj,layer,step)     : min={by_out.min()}, median={int(by_out.median())}, max={by_out.max()}"
    )
    print()


# --- grids -------------------------------------------------------------------
def make_grids(t: pd.DataFrame):
    comps = sorted(t["comp"].unique())
    sizes = sorted(t["size"].unique())

    def piv(agg):
        return t.pivot_table(index="size", columns="comp", values="n_steps", aggfunc=agg).reindex(
            index=sizes, columns=comps
        )

    return comps, sizes, piv("sum"), piv("mean"), piv("std"), piv("count")


# --- plots -------------------------------------------------------------------
def plot_data_amount(t, out, layer_label):
    """Two-panel bar chart of data volume (# .pt files) per grid size and per
    complexity, with the grand total of counted .pt files in the title."""
    total_pt = int(t["n_steps"].sum())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    by_size = t.groupby("size")["n_steps"].sum()
    axes[0].bar([str(s) for s in by_size.index], by_size.values, color="indianred")
    axes[0].set(xlabel="maze size", ylabel="amount of data (# .pt files)", title="Data per maze size")
    for x, v in enumerate(by_size.values):
        axes[0].text(x, v, f"{int(v):,}", ha="center", va="bottom", fontsize=8)

    by_comp = t.groupby("comp")["n_steps"].sum()
    axes[1].bar([f"{c:g}" for c in by_comp.index], by_comp.values, color="steelblue")
    axes[1].set(xlabel="complexity", ylabel="amount of data (# .pt files)", title="Data per complexity")
    for x, v in enumerate(by_comp.values):
        axes[1].text(x, v, f"{int(v):,}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(f"Amount of data \u2014 total {total_pt:,} .pt files ({layer_label})", fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(out / "data_amount.png", dpi=150)
    plt.close(fig)


def _imshow_grid(ax, grid, comps, sizes, cmap):
    im = ax.imshow(grid.values, origin="lower", aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(comps)))
    ax.set_xticklabels([f"{c:g}" for c in comps])
    ax.set_yticks(range(len(sizes)))
    ax.set_yticklabels(sizes)
    return im


def plot2_heatmap(comps, sizes, mean, std, ntraj, out):
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    im = _imshow_grid(ax, mean, comps, sizes, "magma")
    vmax = np.nanmax(mean.values)
    for i in range(len(sizes)):
        for j in range(len(comps)):
            m = mean.values[i, j]
            if np.isnan(m):
                continue
            s = std.values[i, j]
            n = ntraj.values[i, j]
            s_txt = f"±{s:.1f}" if not np.isnan(s) else "±–"
            ax.text(
                j,
                i,
                f"{m:.1f}{s_txt}\n(n={int(n)})",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if m < vmax * 0.6 else "black",
            )
    ax.set(xlabel="complexity", ylabel="maze size", title="Mean ± std reasoning steps per category")
    fig.colorbar(im, ax=ax, label="mean reasoning steps")
    fig.tight_layout()
    fig.savefig(out / "heatmap_mean_std.png", dpi=150)
    plt.close(fig)


# --- main --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="repository root to scan recursively")
    ap.add_argument("--layer", type=int, default=None, help="restrict counting to this layer (default: auto)")
    ap.add_argument(
        "--all-layers", action="store_true", help="count across ALL layers (disables dedup; usually wrong)"
    )
    ap.add_argument("--out", type=Path, default=Path("reasoning_step_figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = build_dataframe(args.root)
    print_diagnostics(df)

    layer = None if args.all_layers else choose_layer(df, args.layer)
    use_all = args.all_layers or layer is None
    if not use_all:
        print(f"Counting .pt files at layer {layer}.\n")

    t = per_trajectory_counts(df, layer, use_all)
    comps, sizes, total, mean, std, ntraj = make_grids(t)

    # save tables
    t.to_csv(args.out / "per_trajectory_counts.csv", index=False)
    mean.to_csv(args.out / "table_mean_steps.csv")
    std.to_csv(args.out / "table_std_steps.csv")
    ntraj.to_csv(args.out / "table_n_trajectories.csv")

    # plots
    layer_label = "all layers" if use_all else f"layer {layer}"
    plot_data_amount(t, args.out, layer_label)
    plot2_heatmap(comps, sizes, mean, std, ntraj, args.out)

    # console summary
    print("MEAN reasoning steps per (size x complexity):")
    print(mean.round(2).to_string())
    print(f"\nWrote 2 figures + 4 CSVs to {args.out.resolve()}")


if __name__ == "__main__":
    main()
