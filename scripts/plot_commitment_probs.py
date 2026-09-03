#!/usr/bin/env python3
"""The commitment boundary in the probe's PROBABILITIES, not its argmax.

Every other figure in this line plots an agreement RATE: how often the probe's argmax equals
some comparator. That throws away everything the probe says about the other three actions, and
an argmax can only move in whole steps. This plots, at every reasoning token, the probability
mass the probe puts on each comparator:

  p_this   -- the action the model answers if reasoning is cut at the end of THIS sentence
  p_prev   -- the action it answered at the end of the PREVIOUS sentence
  p_final  -- the action it actually takes after all reasoning (the probe's training label)

The three are three columns of one 4-way softmax, so they sum to <= 1 and a rise in one is
mass taken from another. Chance is 0.25. Confidence (max probability) is plotted alongside,
because a probability curve that rises only because the probe grows sharper everywhere is a
different claim from one that reallocates mass toward the answer.

Needs `eval_probe_per_token.py --full-probs`; the x coordinate is the one
`plot_commitment_all_tokens.py` documents.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

HEADLINE = "next_action_l15.jlens_topall_mlp"
ACTIONS = ("LEFT", "UP", "RIGHT", "DOWN")
C = {"this": "#2b6cb0", "prev": "#c05621", "final": "#2f855a", "conf": "#6b46c1", "grey": "#718096"}
SERIES = (
    ("p_this", "this", "p(belief at the end of THIS sentence)"),
    ("p_prev", "prev", "p(belief at the end of the PREVIOUS sentence)"),
    ("p_final", "final", "p(final action)"),
)
N_BOOT = 400


def load(probs_csv: Path, meta_csv: Path, arm: str) -> pd.DataFrame:
    meta_cols = [
        "name",
        "step",
        "token_idx",
        "label_name",
        "sentence_idx",
        "pos_in_sentence",
        "sentence_len",
        "is_sentence_end",
        "sent_model_action",
        "prev_model_action",
        "convinced_sentence_idx",
    ]
    # keep_default_na=False: decoded tokens include the literal string "NA".
    meta = pd.read_csv(meta_csv, keep_default_na=False, na_values=[""], usecols=meta_cols)
    pcols = ["name", "step", "token_idx"] + [f"{arm}_p_{a}" for a in ACTIONS]
    probs = pd.read_csv(probs_csv, keep_default_na=False, na_values=[""], usecols=pcols)
    df = meta.merge(probs, on=["name", "step", "token_idx"], how="inner", validate="one_to_one")

    p = df[[f"{arm}_p_{a}" for a in ACTIONS]].to_numpy(float)
    col = {a: i for i, a in enumerate(ACTIONS)}
    rows = np.arange(len(df))
    df["p_this"] = p[rows, df["sent_model_action"].map(col).to_numpy()]
    df["p_prev"] = p[rows, df["prev_model_action"].map(col).to_numpy()]
    df["p_final"] = p[rows, df["label_name"].map(col).to_numpy()]
    df["p_max"] = p.max(axis=1)
    df["entropy"] = -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=1)
    df["rel_sentence"] = df["sentence_idx"] - df["convinced_sentence_idx"]
    df["x"] = df["rel_sentence"] - 1 + (df["pos_in_sentence"] + 1) / df["sentence_len"]
    df["within"] = (df["pos_in_sentence"] + 1) / df["sentence_len"]
    return df[df["convinced_sentence_idx"] >= 1].reset_index(drop=True)


def clustered_band(df: pd.DataFrame, col: str, bin_id: np.ndarray, n_bins: int, seed: int = 0):
    """Mean per bin + 95% band from resampling TRAJECTORIES (rows within one are dependent)."""
    vals = df[col].to_numpy(float)
    codes, _ = pd.factorize(df["name"])
    order = np.argsort(codes, kind="stable")
    starts = np.searchsorted(codes[order], np.arange(codes.max() + 1))
    ends = np.append(starts[1:], len(order))

    def means(idx: np.ndarray) -> np.ndarray:
        b = bin_id[idx]
        keep = b >= 0
        b, v = b[keep], vals[idx][keep]
        cnt = np.bincount(b, minlength=n_bins)
        tot = np.bincount(b, weights=v, minlength=n_bins)
        return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)

    point = means(np.arange(len(vals)))
    rng = np.random.default_rng(seed)
    boots = np.empty((N_BOOT, n_bins))
    for b in range(N_BOOT):
        pick = rng.integers(0, len(starts), len(starts))
        boots[b] = means(np.concatenate([order[starts[i] : ends[i]] for i in pick]))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    return point, lo, hi


def binner(vals: np.ndarray, edges: np.ndarray) -> np.ndarray:
    bid = np.clip(np.digitize(vals, edges) - 1, -1, len(edges) - 2)
    bid[(vals <= edges[0]) | (vals > edges[-1])] = -1
    return bid


def fig_boundary(df: pd.DataFrame, arm: str, out: Path, window: int, per_sentence: int) -> pd.DataFrame:
    lo_x, hi_x = -window - 1, window
    d = df[(df["x"] > lo_x) & (df["x"] <= hi_x)].reset_index(drop=True)
    edges = np.arange(lo_x, hi_x + 1e-9, 1 / per_sentence)
    centers = (edges[:-1] + edges[1:]) / 2
    bid = binner(d["x"].to_numpy(), edges)

    fig, (ax, axc) = plt.subplots(
        2, 1, figsize=(12, 7.5), sharex=True, gridspec_kw={"height_ratios": [2.6, 1.2], "hspace": 0.09}
    )
    table = {"x": centers}
    for a in (ax, axc):
        for s in range(lo_x, hi_x):
            if s % 2 == 0:
                a.axvspan(s, s + 1, color=C["grey"], alpha=0.05, lw=0)
        for s in range(lo_x, hi_x + 1):
            a.axvline(s, color=C["grey"], lw=0.6, alpha=0.35)
        a.axvline(0, color="black", lw=1.8)

    ax.axhline(0.25, color=C["grey"], ls=":", lw=1.2)
    for col, key, label in SERIES:
        pt, lo, hi = clustered_band(d, col, bid, len(centers))
        ax.fill_between(centers, lo, hi, color=C[key], alpha=0.15, lw=0)
        ax.plot(centers, pt, color=C[key], lw=2, label=label)
        table[col] = pt
        ends = d[d["is_sentence_end"] == 1].groupby("rel_sentence")[col].mean()
        ends = ends[(ends.index >= lo_x + 1) & (ends.index <= hi_x)]
        ax.plot(ends.index, ends.values, "o", color=C[key], ms=6, mec="white", mew=1.2, zorder=5)
    ax.text(hi_x - 0.05, 0.25, "chance", ha="right", va="bottom", color=C["grey"], fontsize=9)
    ax.text(0.06, ax.get_ylim()[1], "convinced\nboundary", ha="left", va="top", fontsize=9)
    ax.set_ylabel("mean probe probability")
    ax.set_title(
        f"What the probe's probabilities do around the commitment boundary\n{arm}  ·  "
        f"{d['name'].nunique()} trajectories, {len(d):,} tokens  ·  markers = sentence ends",
        fontsize=12,
    )
    ax.legend(loc="lower left", frameon=True, fontsize=9, framealpha=0.92)

    for col, color, label in (
        ("p_max", C["conf"], "confidence  max p"),
        ("entropy", C["grey"], "entropy  (right axis)"),
    ):
        pt, lo, hi = clustered_band(d, col, bid, len(centers))
        table[col] = pt
        target = axc if col == "p_max" else axc.twinx()
        if col != "p_max":
            target.grid(False)
            target.set_ylabel("entropy (nats)", color=C["grey"], fontsize=9)
            target.tick_params(axis="y", colors=C["grey"], labelsize=8)
        target.fill_between(centers, lo, hi, color=color, alpha=0.15, lw=0)
        target.plot(centers, pt, color=color, lw=1.8, ls="-" if col == "p_max" else "--", label=label)
    axc.set_ylabel("max p", color=C["conf"], fontsize=9)
    axc.tick_params(axis="y", colors=C["conf"], labelsize=8)
    axc.set_xlabel("sentence index relative to the convinced sentence (0 = its last token; each band is one sentence)")
    axc.set_xticks(range(lo_x, hi_x + 1))
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return pd.DataFrame(table)


def fig_within(df: pd.DataFrame, arm: str, out: Path, n_bins: int) -> pd.DataFrame:
    # `this` and `final` coincide from rel 0 on: the convinced sentence is by definition the
    # first whose truncation already answers the final action, so the green line hides the blue.
    panels = [
        ("rel \u2264 \u22122", df["rel_sentence"] <= -2, "earlier reasoning"),
        ("rel = \u22121", df["rel_sentence"] == -1, "one before the convinced"),
        ("rel = 0", df["rel_sentence"] == 0, "THE CONVINCED SENTENCE\nthis \u2261 final here"),
        ("rel \u2265 +1", df["rel_sentence"] >= 1, "after commitment\nthis \u2261 prev \u2261 final"),
    ]
    edges = np.linspace(0, 1, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4), sharey=True)
    out_rows = []
    for axp, (tag, mask, sub) in zip(axes, panels, strict=True):
        d = df[mask].reset_index(drop=True)
        bid = binner(d["within"].to_numpy(), edges)
        axp.axhline(0.25, color=C["grey"], ls=":", lw=1.2)
        for col, key, _ in SERIES:
            pt, lo, hi = clustered_band(d, col, bid, len(centers))
            axp.fill_between(centers, lo, hi, color=C[key], alpha=0.15, lw=0)
            axp.plot(centers, pt, color=C[key], lw=2)
            out_rows += [{"panel": tag, "x": c, "series": col, "mean": v} for c, v in zip(centers, pt, strict=True)]
        axp.set_title(f"{tag}   ·   {len(d):,} tokens\n{sub}", fontsize=9.5)
        axp.set_xlabel("position through the sentence")
    axes[0].set_ylabel("mean probe probability")
    handles = [plt.Line2D([], [], color=C[k], lw=2, label=lbl) for _, k, lbl in SERIES]
    axes[-1].legend(handles=handles, loc="lower right", fontsize=8, frameon=True, framealpha=0.92)
    fig.suptitle(
        f"Probe probabilities across a reasoning sentence, by where the sentence sits "
        f"relative to commitment  ·  {arm}",
        fontsize=12,
        y=1.10,
    )
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return pd.DataFrame(out_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path("/workspace/reasoning_theatre/probe_vs_rollout")
    ap.add_argument("--probs-csv", type=Path, default=root / "per_token_probs.csv")
    ap.add_argument("--per-token", type=Path, default=root / "per_token.csv")
    ap.add_argument("--out-dir", type=Path, default=root / "plots")
    ap.add_argument("--table-dir", type=Path, default=root / "tables")
    ap.add_argument("--arm", default=HEADLINE)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--bins-per-sentence", type=int, default=5)
    ap.add_argument("--within-bins", type=int, default=10)
    args = ap.parse_args()

    sns.set_theme(style="whitegrid", context="notebook")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    df = load(args.probs_csv, args.per_token, args.arm)
    print(f"{len(df):,} tokens from {df['name'].nunique()} trajectories (convinced sentence >= 1)")
    # The headline arm owns the unsuffixed filenames; any other arm writes beside it.
    sfx = "" if args.arm == HEADLINE else "_" + args.arm.replace(".", "_")
    t1 = fig_boundary(
        df, args.arm, args.out_dir / f"commit_probs_by_sentence{sfx}.png", args.window, args.bins_per_sentence
    )
    t1.to_csv(args.table_dir / f"commit_probs_by_sentence{sfx}.csv", index=False)
    t2 = fig_within(df, args.arm, args.out_dir / f"commit_probs_within_sentence{sfx}.png", args.within_bins)
    t2.to_csv(args.table_dir / f"commit_probs_within_sentence{sfx}.csv", index=False)


if __name__ == "__main__":
    main()
