#!/usr/bin/env python3
"""Figures for the probe-loudness analysis: what the local-belief probes of ICLR log entry
45 do as a function of how loud the token they read is.

Point estimates only -- ``analyze_probe_loudness.py`` carries the bootstrap CIs and writes
the same cuts as CSV. Balanced accuracy over the four actions everywhere, for the reason
that script gives: the loud bins are direction-word-heavy and a plain accuracy would move
with the class mix.

Seven figures per run:

  loudness_distribution        where the selected tokens sit against every reasoning token
  {rs}_by_mass_decile          Q1, entry 37's monotonicity with the local-belief label
  {rs}_by_sentence_frac        Q2a, the same accuracy against position in the sentence
  {rs}_mass_x_position         Q2b, the 3x3 that separates the two
  {rs}_by_rel_sentence         Q3, against the commitment boundary
  {rs}_follows_by_mass         Q5, local vs final on the disagreement rows, by loudness
  p2_label_comparison          the headline: ONE token set, three probes
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ACTIONS = ["LEFT", "UP", "RIGHT", "DOWN"]
PROBES = {
    "p1_full": ["p1_lr", "p1_mlp"],
    "p1_top20": ["p1t20_lr", "p1t20_mlp"],
    "p2": ["p2_lr", "p2_mlp", "base_lr", "base_mlp", "rand_lr", "rand_mlp"],
}
LABEL = {
    "p1_lr": "P1 all, lr",
    "p1_mlp": "P1 all, mlp",
    "p1t20_lr": "P1 top-20, lr",
    "p1t20_mlp": "P1 top-20, mlp",
    "p2_lr": "P2 global top-20, lr",
    "p2_mlp": "P2 global top-20, mlp",
    "base_lr": "baseline (final label), lr",
    "base_mlp": "baseline (final label), mlp",
    "rand_lr": "random control, lr",
    "rand_mlp": "random control, mlp",
}
TITLE = {
    "p1_full": "P1 - every sentence's loudest token",
    "p1_top20": "P1 - top 20 loudest per trajectory",
    "p2": "P2 - the 20 globally loudest tokens",
}
COLOR = {
    "p1_lr": "#5a7fb5",
    "p1_mlp": "#1f3f7a",
    "p1t20_lr": "#5a7fb5",
    "p1t20_mlp": "#1f3f7a",
    "p2_lr": "#5a7fb5",
    "p2_mlp": "#1f3f7a",
    "base_lr": "#d08a4a",
    "base_mlp": "#a8541c",
    "rand_lr": "#9a9a9a",
    "rand_mlp": "#5a5a5a",
}


def bal_acc(truth: np.ndarray, pred: np.ndarray) -> float:
    r = [float((pred[truth == a] == a).mean()) for a in ACTIONS if (truth == a).sum()]
    return float(np.mean(r)) if r else float("nan")


def acc_by(g: pd.DataFrame, probe: str, truth: str) -> float:
    return bal_acc(g[truth].to_numpy(), g[f"{probe}_pred"].to_numpy())


def qbin(s: pd.Series, q: int) -> pd.Series:
    try:
        return pd.qcut(s, q, labels=False, duplicates="drop")
    except ValueError:
        return pd.cut(s, q, labels=False)


def curve(d: pd.DataFrame, bincol: str, probes: list[str], truth: str) -> pd.DataFrame:
    rows = []
    for b, g in d.groupby(bincol, observed=True):
        r = {"bin": b, "n": len(g), "x": float(g["dir_logmass"].mean())}
        for p in probes:
            r[p] = acc_by(g, p, truth)
        rows.append(r)
    return pd.DataFrame(rows).sort_values("bin")


def fig_distribution(df: pd.DataFrame, allref: pd.DataFrame | None, out: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    bins = np.linspace(-12, 0.5, 80)
    if allref is not None:
        ax[0].hist(
            allref["dir_logmass_L15"],
            bins=bins,
            density=True,
            color="#cccccc",
            label=f"every reasoning token (n={len(allref):,})",
        )
    for rs in PROBES:
        d = df[df["rowset"] == rs]
        if len(d):
            ax[0].hist(
                d["dir_logmass"], bins=bins, density=True, histtype="step", lw=1.8, label=f"{rs} (n={len(d):,})"
            )
    ax[0].set_xlabel("layer-15 direction log-mass  (loudness)")
    ax[0].set_ylabel("density")
    ax[0].set_title("What the probes are fed, against the whole chain")
    ax[0].legend(fontsize=7)

    for rs in PROBES:
        d = df[df["rowset"] == rs]
        if len(d):
            ax[1].hist(
                d["mass_pct_in_traj"], bins=np.linspace(0, 1, 51), density=True, histtype="step", lw=1.8, label=rs
            )
    ax[1].set_xlabel("loudness rank within its own chain  (0 = loudest token)")
    ax[1].set_ylabel("density")
    ax[1].set_title("Rank, not level")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "loudness_distribution.png", dpi=150)
    plt.close(fig)


def fig_by_bin(d: pd.DataFrame, probes: list[str], rs: str, bincol: str, xlabel: str, fname: str, out: Path) -> None:
    loc = curve(d, bincol, probes, "label_local")
    fin = curve(d, bincol, probes, "label_final")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for i, (tab, name) in enumerate([(loc, "vs LOCAL belief"), (fin, "vs FINAL action")]):
        for p in probes:
            ax[i].plot(tab["bin"], tab[p], marker="o", ms=4, color=COLOR[p], label=LABEL[p])
        ax[i].axhline(0.25, ls=":", c="k", lw=1)
        ax[i].set_xlabel(xlabel)
        ax[i].set_title(name)
        ax[i].grid(alpha=0.25)
    ax[0].set_ylabel("balanced accuracy")
    ax[1].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"{TITLE[rs]}   (n={len(d):,} held-out tokens)")
    fig.tight_layout()
    fig.savefig(out / f"{rs}_{fname}.png", dpi=150)
    plt.close(fig)


def fig_grid(d: pd.DataFrame, probe: str, rs: str, out: Path) -> None:
    d = d.copy()
    d["mass_t"] = qbin(d["dir_logmass"], 3)
    d["pos_t"] = qbin(d["sentence_frac"], 3)
    M = np.full((3, 3), np.nan)
    N = np.zeros((3, 3), dtype=int)
    for (mt, pt), g in d.groupby(["mass_t", "pos_t"], observed=True):
        M[int(mt), int(pt)] = acc_by(g, probe, "label_local")
        N[int(mt), int(pt)] = len(g)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(M, origin="lower", cmap="viridis")
    for i in range(3):
        for j in range(3):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.3f}\nn={N[i, j]:,}", ha="center", va="center", color="w", fontsize=8)
    ax.set_xticks(range(3), ["start", "middle", "end"])
    ax.set_yticks(range(3), ["quiet", "mid", "loud"])
    ax.set_xlabel("position in its sentence")
    ax.set_ylabel("loudness")
    ax.set_title(f"{TITLE[rs]}\n{LABEL[probe]}, balanced acc vs local belief")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out / f"{rs}_mass_x_position.png", dpi=150)
    plt.close(fig)


def fig_follows(d: pd.DataFrame, probes: list[str], rs: str, out: Path) -> None:
    d = d.copy()
    d["mass_decile"] = qbin(d["dir_logmass"], 10)
    dd = d[d["label_local"] != d["label_final"]]
    if not len(dd):
        return
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for p in probes:
        loc, fin, xs = [], [], []
        for b, g in dd.groupby("mass_decile", observed=True):
            xs.append(b)
            loc.append(float((g[f"{p}_pred"] == g["label_local"]).mean()))
            fin.append(float((g[f"{p}_pred"] == g["label_final"]).mean()))
        ax.plot(xs, loc, marker="o", ms=4, color=COLOR[p], label=f"{LABEL[p]} -> local")
        ax.plot(xs, fin, marker="s", ms=4, ls="--", color=COLOR[p], alpha=0.6, label=f"{LABEL[p]} -> final")
    ax.axhline(0.25, ls=":", c="k", lw=1)
    ax.set_xlabel("loudness decile  (9 = loudest)")
    ax.set_ylabel("share of predictions")
    ax.set_title(
        f"Where the probe lands when belief != ending\n{TITLE[rs]}  -  n={len(dd):,} disagreement rows", fontsize=11
    )
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / f"{rs}_follows_by_mass.png", dpi=150)
    plt.close(fig)


def fig_chain_length(d: pd.DataFrame, probes: list[str], rs: str, out: Path) -> None:
    """The loudness gradient with chain length held fixed.

    A fixed top-K per trajectory takes the K loudest of a short chain and the K loudest of a
    long one, so inside such an arm loudness is correlated with chain LENGTH -- and long
    chains are harder. The two run opposite ways and flatten the raw decile curve. Re-cut
    the mass terciles inside chain-length quartiles and the loudness gradient stands alone.
    """
    d = d.copy()
    d["len_q"] = qbin(d["n_reasoning_tokens"], 4)
    d["mass_t"] = d.groupby("len_q", observed=True)["dir_logmass"].transform(lambda s: qbin(s, 3))
    qs = sorted(d["len_q"].dropna().unique())
    fig, ax = plt.subplots(1, len(qs), figsize=(3.1 * len(qs), 4.0), sharey=True)
    for i, q in enumerate(qs):
        g = d[d["len_q"] == q]
        for p in probes:
            xs, ys = [], []
            for t, gg in g.groupby("mass_t", observed=True):
                xs.append(t)
                ys.append(acc_by(gg, p, "label_local"))
            ax[i].plot(xs, ys, marker="o", ms=5, color=COLOR[p], label=LABEL[p])
        ax[i].set_xticks([0, 1, 2], ["quiet", "mid", "loud"])
        ax[i].set_title(f"chain length q{int(q)}\nmean {g['n_reasoning_tokens'].mean():.0f} tokens", fontsize=9)
        ax[i].grid(alpha=0.25)
        ax[i].set_xlabel("loudness tercile")
    ax[0].set_ylabel("balanced accuracy vs local belief")
    ax[-1].legend(fontsize=7)
    fig.suptitle(f"{TITLE[rs]}: the loudness gradient with chain length held fixed")
    fig.tight_layout()
    fig.savefig(out / f"{rs}_chain_length_control.png", dpi=150)
    plt.close(fig)


def fig_label_comparison(d: pd.DataFrame, out: Path) -> None:
    """ONE token set, three training regimes. Everything that differs here is the label or
    the selection the probe was trained under, never the tokens."""
    d = d.copy()
    d["mass_decile"] = qbin(d["dir_logmass"], 10)
    probes = [p for p in ["p2_mlp", "base_mlp", "rand_mlp", "p2_lr", "base_lr", "rand_lr"] if f"{p}_pred" in d.columns]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for i, truth in enumerate(["label_local", "label_final"]):
        tab = curve(d, "mass_decile", probes, truth)
        for p in probes:
            ax[i].plot(
                tab["bin"],
                tab[p],
                marker="o",
                ms=4,
                color=COLOR[p],
                ls="-" if p.endswith("mlp") else "--",
                label=LABEL[p],
            )
        ax[i].axhline(0.25, ls=":", c="k", lw=1)
        ax[i].set_xlabel("loudness decile  (9 = loudest)")
        ax[i].set_title("vs LOCAL belief" if truth == "label_local" else "vs FINAL action")
        ax[i].grid(alpha=0.25)
    ax[0].set_ylabel("balanced accuracy")
    ax[1].legend(fontsize=7, loc="lower right")
    fig.suptitle("Same 14,391 held-out tokens, three probes: local-belief, final-action baseline, random control")
    fig.tight_layout()
    fig.savefig(out / "p2_label_comparison.png", dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--per-token", type=Path, default=Path("/workspace/reasoning_theatre/probe_loudness/per_token.csv")
    )
    ap.add_argument(
        "--all-token-loudness", type=Path, default=Path("/workspace/reasoning_theatre/loudness/per_token.csv")
    )
    ap.add_argument("--out", type=Path, default=Path("/workspace/reasoning_theatre/probe_loudness/plots"))
    args = ap.parse_args()

    df = pd.read_csv(args.per_token, keep_default_na=False, na_values=[""], low_memory=False)
    args.out.mkdir(parents=True, exist_ok=True)
    allref = None
    if args.all_token_loudness.exists():
        allref = pd.read_csv(args.all_token_loudness, usecols=["dir_logmass_L15"])

    fig_distribution(df, allref, args.out)
    print("loudness_distribution.png", flush=True)

    for rs, all_probes in PROBES.items():
        d = df[df["rowset"] == rs].copy()
        if not len(d):
            continue
        probes = [p for p in all_probes if f"{p}_pred" in d.columns]
        d["mass_decile"] = qbin(d["dir_logmass"], 10)
        d["sentfrac_decile"] = qbin(d["sentence_frac"], 10)
        fig_by_bin(d, probes, rs, "mass_decile", "loudness decile  (9 = loudest)", "by_mass_decile", args.out)
        fig_by_bin(
            d,
            probes,
            rs,
            "sentfrac_decile",
            "position in its sentence  (0 = first token)",
            "by_sentence_frac",
            args.out,
        )
        rel = d[d["rel_sentence"].notna()].copy()
        if len(rel):
            rel["rel_clipped"] = rel["rel_sentence"].clip(-6, 6).astype(int)
            fig_by_bin(rel, probes, rs, "rel_clipped", "sentence_idx - convinced_idx", "by_rel_sentence", args.out)
        mlps = [p for p in probes if p.endswith("mlp")]
        fig_grid(d, mlps[0], rs, args.out)
        fig_follows(d, mlps, rs, args.out)
        fig_chain_length(d, mlps, rs, args.out)
        print(f"{rs}: 6 figures", flush=True)

    d2 = df[df["rowset"] == "p2"]
    if len(d2):
        fig_label_comparison(d2, args.out)
        print("p2_label_comparison.png", flush=True)
    print(f"-> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
