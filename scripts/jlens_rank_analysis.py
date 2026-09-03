"""Compact supervisor-facing figures from a jlens action-rank CSV.

Usage:
  python scripts/jlens_rank_analysis.py \
    --csv C:/Uni/Thesis/data/jlens_exploration/training_all_jlens_action_ranks_sample1000.csv \
    --out_dir C:/Uni/Thesis/data/jlens_exploration/figs
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ACTIONS = ["RIGHT", "LEFT", "UP", "DOWN"]

# dataviz reference palette (light mode)
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"
SERIES = ["#2a78d6", "#008300", "#eb6834", "#4a3aa7"]  # blue, green, orange, violet


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, default=Path("figs"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    # drop layers with only stray rows (partial remote runs)
    counts = df.groupby("layer").size()
    layers = sorted(counts[counts >= 1000].index)
    df = df[df["layer"].isin(layers)].copy()
    df = df[df["agent_action"].isin(ACTIONS)]
    print(f"{len(df)} rows, layers {layers}")

    ranks = df[[f"{a}_position" for a in ACTIONS]].to_numpy()
    lps = df[[f"{a}_logprob" for a in ACTIONS]].to_numpy()
    taken = df["agent_action"].map({a: i for i, a in enumerate(ACTIONS)}).to_numpy()
    rows = np.arange(len(df))
    df["taken_rank"] = ranks[rows, taken]
    df["taken_lp"] = lps[rows, taken]
    others_lp = lps.copy()
    others_lp[rows, taken] = -np.inf
    df["best_other_lp"] = others_lp.max(1)
    df["correct"] = df["taken_rank"] == ranks.min(1)

    # ---- Fig 1: accuracy (taken action ranked best of 4) vs layer, per action
    fig, ax = plt.subplots(figsize=(6, 3.6), facecolor=SURFACE)
    for a, c in zip(ACTIONS, SERIES):
        acc = df[df["agent_action"] == a].groupby("layer")["correct"].mean()
        ax.plot(acc.index, acc.values, color=c, linewidth=2, marker="o", markersize=5, label=a)
    ax.axhline(0.25, color=MUTED, linewidth=1, linestyle="--")
    ax.annotate("chance (25%)", (layers[0], 0.25), textcoords="offset points", xytext=(0, 5), color=MUTED, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_xlabel("layer")
    ax.set_ylabel("P(taken action ranked best of 4)")
    style(ax, "Decoding accuracy by layer, per taken action")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(args.out_dir / "fig1_accuracy_by_layer.png", dpi=150, facecolor=SURFACE)

    # ---- Fig 2: median lens logprob of each action token by layer
    fig, ax = plt.subplots(figsize=(6, 3.6), facecolor=SURFACE)
    g = df.groupby("layer")
    for a, c in zip(ACTIONS, SERIES):
        med = g[f"{a}_logprob"].median()
        lo, hi = g[f"{a}_logprob"].quantile(0.25), g[f"{a}_logprob"].quantile(0.75)
        ax.plot(med.index, med.values, color=c, linewidth=2, marker="o", markersize=5, label=a)
        ax.fill_between(med.index, lo, hi, color=c, alpha=0.12, linewidth=0)
    ax.set_xlabel("layer")
    ax.set_ylabel("lens logprob (median, IQR)")
    style(ax, "Lens logprob of each action token by layer")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(args.out_dir / "fig2_logprob_by_layer.png", dpi=150, facecolor=SURFACE)

    # ---- Fig 3: median vocab rank of each action token by layer (log scale)
    fig, ax = plt.subplots(figsize=(6, 3.6), facecolor=SURFACE)
    for a, c in zip(ACTIONS, SERIES):
        med = g[f"{a}_position"].median()
        ax.plot(med.index, med.values + 1, color=c, linewidth=2, marker="o", markersize=5, label=a)
    ax.set_yscale("log")
    ax.set_xlabel("layer")
    ax.set_ylabel("median vocab rank (log)")
    style(ax, "Each action token's rank in the full vocabulary (201k tokens)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(args.out_dir / "fig3_vocab_rank_by_layer.png", dpi=150, facecolor=SURFACE)

    # ---- Fig 4: accuracy heatmap size x complexity at deepest layer
    last = layers[-1]
    piv = df[df["layer"] == last].pivot_table(index="size", columns="complexity", values="correct")
    fig, ax = plt.subplots(figsize=(5.6, 3.6), facecolor=SURFACE)
    im = ax.imshow(
        piv.values,
        cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#0d366b"]),
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    ax.set_xticks(range(len(piv.columns)), [f"{c:g}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="#ffffff" if v > 0.55 else INK)
    ax.set_xlabel("complexity")
    ax.set_ylabel("maze size")
    style(ax, f"Decoding accuracy at layer {last}, by maze difficulty")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.85).ax.tick_params(colors=MUTED, labelsize=8)
    fig.tight_layout()
    fig.savefig(args.out_dir / "fig4_accuracy_heatmap.png", dpi=150, facecolor=SURFACE)

    print(f"wrote 4 figures to {args.out_dir}")


if __name__ == "__main__":
    main()
