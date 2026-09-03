#!/usr/bin/env python3
"""Where the Jacobian lens is LOUD -- inside a sentence, and along the whole chain.

Reads `build_sentence_loudness.py`'s per-token CSV (full-vocabulary direction probability
mass at layer 15, placed inside its reasoning sentence) and answers two questions:

  Q1  how does loudness evolve THROUGH a sentence?
      q1_within_sentence      -- mean +- sd against percent of sentence completion.
      q1_around_commitment    -- the same, split by where the sentence sits relative to
                                 the convinced one. Sentences are indexed by their RIGHT
                                 endpoint, so the convinced sentence spans x in [-1, 0]:
                                   [-1, +1]  the convinced sentence and the one after it,
                                             on one continuous 2-sentence axis;
                                   >= +1     already convinced;
                                   -1        the sentence just before commitment;
                                   <= -2     everything earlier.
  Q2  where in the WHOLE chain is it loudest?
      q2_along_chain          -- mean +- sd against percent of reasoning completion, with
                                 the convinced eos token's position marked.

Every figure is produced over all the data and then again split by complexity.

The cohort is the trajectories whose belief CHANGES during reasoning (convinced sentence
index >= 1): a trajectory convinced at index 0 was committed before writing anything and
has no pre-commitment sentences at all, so keeping it would put a different set of
trajectories on each side of the boundary.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COMPLEXITIES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
CMAP = plt.get_cmap("viridis")
CCOLOR = {c: CMAP(i / (len(COMPLEXITIES) - 1) * 0.9) for i, c in enumerate(COMPLEXITIES)}
MAIN = "#2b6cb0"
BAND = "#2b6cb0"
REF = "#999999"
VALUES = {
    "dir_prob_L15": "direction probability mass (layer 15)",
    "dir_logmass_L15": "log direction mass (layer 15)",
}


def binned(df: pd.DataFrame, xcol: str, value: str, edges: np.ndarray) -> pd.DataFrame:
    """mean / sd / n / trajectory-clustered sem of `value` in each bin of `xcol`."""
    idx = np.clip(np.digitize(df[xcol].to_numpy(), edges) - 1, 0, len(edges) - 2)
    out = []
    centers = (edges[:-1] + edges[1:]) / 2
    for b in range(len(edges) - 1):
        g = df[idx == b]
        if len(g) == 0:
            out.append(
                {"x": centers[b], "mean": np.nan, "median": np.nan, "sd": np.nan, "n": 0, "n_traj": 0, "sem": np.nan}
            )
            continue
        per_traj = g.groupby("name")[value].mean()
        sem = per_traj.std(ddof=1) / np.sqrt(len(per_traj)) if len(per_traj) > 1 else np.nan
        out.append(
            {
                "x": centers[b],
                "mean": g[value].mean(),
                "median": g[value].median(),
                "sd": g[value].std(ddof=1),
                "n": len(g),
                "n_traj": g["name"].nunique(),
                "sem": sem,
            }
        )
    return pd.DataFrame(out)


def draw(ax, b: pd.DataFrame, color=MAIN, label=None, band=True, sd=True, median=True):
    """mean +- sd (faint) and +- 1.96 sem over trajectories (solid); the median is the line
    to read when a bin is small, since the mass is heavy-tailed and a few near-1 tokens move
    the mean."""
    ok = b["n"] > 0
    if band and sd:
        ax.fill_between(
            b["x"][ok], (b["mean"] - b["sd"])[ok], (b["mean"] + b["sd"])[ok], color=color, alpha=0.13, lw=0
        )
    if band:
        ax.fill_between(
            b["x"][ok],
            (b["mean"] - 1.96 * b["sem"])[ok],
            (b["mean"] + 1.96 * b["sem"])[ok],
            color=color,
            alpha=0.35,
            lw=0,
        )
    ax.plot(b["x"][ok], b["mean"][ok], "-o", color=color, ms=3.5, lw=1.6, label=label)
    if median:
        ax.plot(
            b["x"][ok], b["median"][ok], ":", color=color, lw=1.2, alpha=0.8, label=None if label is None else "median"
        )
    ax.grid(alpha=0.25)


def save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}", flush=True)


# --------------------------------------------------------------------------- groups


def group_frames(df: pd.DataFrame, min_len: int) -> dict[str, tuple[pd.DataFrame, str, np.ndarray, str]]:
    """The four commitment-relative panels: (rows, x column, bin edges, title)."""
    long = df[df["sentence_len"] >= min_len]
    # The two-sentence window is restricted to trajectories that HAVE both sentences, so the
    # composition does not change at x = 0 (the boundary itself).
    have0 = set(long[long["rel_sentence"] == 0]["name"])
    have1 = set(long[long["rel_sentence"] == 1]["name"])
    both = have0 & have1
    win = long[(long["rel_sentence"].isin([0, 1])) & (long["name"].isin(both))]
    return {
        "window": (
            win,
            "x_sentence",
            np.linspace(-1, 1, 21),
            "the convinced sentence and the next\n(x = 0 is the convinced eos token)",
        ),
        "after": (
            long[long["rel_sentence"] >= 1],
            "sentence_frac",
            np.linspace(0, 1, 11),
            "already convinced (sentences >= +1)",
        ),
        "before1": (
            long[long["rel_sentence"] == -1],
            "sentence_frac",
            np.linspace(0, 1, 11),
            "not yet convinced (sentence -1)",
        ),
        "before2": (
            long[long["rel_sentence"] <= -2],
            "sentence_frac",
            np.linspace(0, 1, 11),
            "not yet convinced (sentences <= -2)",
        ),
    }


# --------------------------------------------------------------------------- figures


def q1_within(df: pd.DataFrame, ref: pd.DataFrame, value: str, min_len: int, out: Path, tab: Path, suffix: str):
    edges = np.linspace(0, 1, 11)
    sub = df[df["sentence_len"] >= min_len]
    b = binned(sub, "sentence_frac", value, edges)
    rb = binned(ref[ref["sentence_len"] >= min_len], "sentence_frac", value, edges)
    b.to_csv(tab / f"q1_within_sentence{suffix}.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    draw(ax, b, label=f"belief-change cohort ({sub['name'].nunique()} trajectories)")
    ax.plot(rb["x"], rb["mean"], "--", color=REF, lw=1.4, label=f"all training trajectories ({ref['name'].nunique()})")
    ax.set_xlabel("position within the sentence  (0 = first token, 1 = final token)")
    ax.set_ylabel(VALUES[value])
    ax.set_title("loudness through a reasoning sentence", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.bar(b["x"], b["n"], width=0.085, color=REF)
    ax.set_xlabel("position within the sentence")
    ax.set_ylabel("tokens")
    ax.set_title("tokens per bin", fontsize=10)
    ax.grid(alpha=0.25)
    fig.suptitle(
        f"Q1 -- how loud is the lens through a sentence?  (sentences of >= {min_len} tokens; "
        "band = 95% CI over trajectories, faint band = +-1 sd over tokens)",
        fontsize=10,
    )
    save(fig, out / f"q1_within_sentence{suffix}.png")


def q1_within_by_complexity(df: pd.DataFrame, value: str, min_len: int, out: Path, tab: Path, suffix: str):
    edges = np.linspace(0, 1, 11)
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True)
    for ax, c in zip(axes.ravel(), COMPLEXITIES, strict=True):
        sub = df[(df["complexity"] == c) & (df["sentence_len"] >= min_len)]
        b = binned(sub, "sentence_frac", value, edges)
        b["complexity"] = c
        rows.append(b)
        draw(ax, b, color=CCOLOR[c])
        ax.set_title(f"complexity {c}  ({sub['name'].nunique()} traj, {len(sub)} tokens)", fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("position within the sentence")
    for ax in axes[:, 0]:
        ax.set_ylabel(VALUES[value])
    fig.suptitle("Q1 -- loudness through a sentence, split by complexity", fontsize=11)
    save(fig, out / f"q1_within_sentence_by_complexity{suffix}.png")

    tabdf = pd.concat(rows, ignore_index=True)
    tabdf.to_csv(tab / f"q1_within_sentence_by_complexity{suffix}.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for c in COMPLEXITIES:
        b = tabdf[tabdf["complexity"] == c]
        ax.plot(b["x"], b["mean"], "-o", color=CCOLOR[c], ms=3.5, lw=1.6, label=f"comp {c}")
    ax.set_xlabel("position within the sentence  (0 = first token, 1 = final token)")
    ax.set_ylabel(VALUES[value])
    ax.set_title("Q1 -- loudness through a sentence, all complexities overlaid", fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    save(fig, out / f"q1_within_sentence_overlay{suffix}.png")


def q1_commitment(df: pd.DataFrame, value: str, min_len: int, out: Path, tab: Path, suffix: str):
    groups = group_frames(df, min_len)
    rows = []
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.3))
    for ax, (key, (sub, xcol, edges, title)) in zip(axes, groups.items(), strict=True):
        b = binned(sub, xcol, value, edges)
        b["group"] = key
        rows.append(b)
        draw(ax, b)
        if key == "window":
            ax.axvline(0, color="red", ls="--", lw=1.2)
            ax.set_xlabel("sentences relative to the convinced eos token")
        else:
            ax.set_xlabel("position within the sentence")
        ax.set_title(f"{title}\n{sub['name'].nunique()} traj, {len(sub)} tokens", fontsize=9)
    axes[0].set_ylabel(VALUES[value])
    fig.suptitle(
        f"Q1 -- loudness through a sentence, split by the commitment boundary (sentences of >= {min_len} tokens)",
        fontsize=11,
    )
    save(fig, out / f"q1_around_commitment{suffix}.png")
    pd.concat(rows, ignore_index=True).to_csv(tab / f"q1_around_commitment{suffix}.csv", index=False)


def q1_commitment_by_complexity(df: pd.DataFrame, value: str, min_len: int, out: Path, tab: Path, suffix: str):
    rows = []
    fig, axes = plt.subplots(6, 4, figsize=(18, 21), squeeze=False)
    overlay = {}
    for r, c in enumerate(COMPLEXITIES):
        groups = group_frames(df[df["complexity"] == c], min_len)
        for k, (key, (sub, xcol, edges, title)) in enumerate(groups.items()):
            b = binned(sub, xcol, value, edges)
            b["group"], b["complexity"] = key, c
            rows.append(b)
            overlay.setdefault(key, {})[c] = (b, title, xcol)
            ax = axes[r][k]
            draw(ax, b, color=CCOLOR[c])
            if key == "window":
                ax.axvline(0, color="red", ls="--", lw=1.2)
            if r == 0:
                ax.set_title(title, fontsize=9)
            if k == 0:
                ax.set_ylabel(f"complexity {c}\n{VALUES[value]}", fontsize=9)
            ax.tick_params(labelsize=8)
    for ax, key in zip(axes[-1], overlay, strict=True):
        ax.set_xlabel("sentences rel. to convinced eos" if key == "window" else "position within the sentence")
    fig.suptitle("Q1 -- loudness around the commitment boundary, one row per complexity", fontsize=12)
    save(fig, out / f"q1_around_commitment_by_complexity{suffix}.png")
    pd.concat(rows, ignore_index=True).to_csv(tab / f"q1_around_commitment_by_complexity{suffix}.csv", index=False)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.3))
    for ax, key in zip(axes, overlay, strict=True):
        for c in COMPLEXITIES:
            b, title, xcol = overlay[key][c]
            ax.plot(b["x"], b["mean"], "-o", color=CCOLOR[c], ms=3, lw=1.5, label=f"comp {c}")
        if key == "window":
            ax.axvline(0, color="red", ls="--", lw=1.2)
            ax.set_xlabel("sentences relative to the convinced eos token")
        else:
            ax.set_xlabel("position within the sentence")
        ax.set_title(overlay[key][COMPLEXITIES[0]][1], fontsize=9)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(VALUES[value])
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle("Q1 -- loudness around the commitment boundary, complexities overlaid", fontsize=11)
    save(fig, out / f"q1_around_commitment_overlay{suffix}.png")


def q1_by_rel_sentence(df: pd.DataFrame, value: str, min_len: int, out: Path, tab: Path, suffix: str, span: int = 6):
    """Sentence-level loudness against distance from the convinced sentence.

    Three readouts because the raw one has two confounds. `mean` is the plain sentence mean.
    `centered` subtracts each trajectory's own mean loudness, so a trajectory that is loud
    throughout cannot tilt the profile. `mid-chain` keeps only tokens in the middle 60% of the
    reasoning chain, where the Q2 profile is flat -- the convinced sentence sits late in the
    chain on average, and the chain has an end-of-reasoning spike.
    """
    long = df[df["sentence_len"] >= min_len].copy()
    long["centered"] = long[value] - long.groupby("name")[value].transform("mean")
    rel = long["rel_sentence"].clip(-span, span)
    long = long.assign(rel_c=rel)
    mid = long[(long["reasoning_frac"] > 0.2) & (long["reasoning_frac"] < 0.8)]

    rows = []
    readouts = (
        ("mean", long, value),
        ("centered", long, "centered"),
        ("mid-chain", mid, value),
        # The standing confound: after commitment the model writes the answer down, so more
        # tokens ARE direction words. Dropping them asks whether the bump survives without them.
        ("non-direction tokens", long[long["is_direction_token"] == 0], value),
        (
            "direction-token share",
            long.assign(**{"is_direction_token": long["is_direction_token"].astype(float)}),
            "is_direction_token",
        ),
    )
    for label, g, col in readouts:
        for comp in ["all", *COMPLEXITIES]:
            h = g if comp == "all" else g[g["complexity"] == comp]
            for r, k in h.groupby("rel_c"):
                per = k.groupby("name")[col].mean()
                rows.append(
                    {
                        "readout": label,
                        "complexity": comp,
                        "rel_sentence": r,
                        "mean": k[col].mean(),
                        "sd": k[col].std(ddof=1),
                        "n": len(k),
                        "n_traj": len(per),
                        "sem": per.std(ddof=1) / np.sqrt(len(per)) if len(per) > 1 else np.nan,
                    }
                )
    t = pd.DataFrame(rows)
    t.to_csv(tab / f"q1_by_rel_sentence{suffix}.csv", index=False)

    def panel(ax, h, ylabel, title, color=MAIN):
        ax.errorbar(h["rel_sentence"], h["mean"], yerr=1.96 * h["sem"], fmt="-o", color=color, ms=4, lw=1.6, capsize=2)
        ax.axvline(0, color="red", ls="--", lw=1.2)
        ax.set_xlabel("sentence index − convinced index  (0 = the convinced sentence)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.25)

    fig, axes = plt.subplots(1, 5, figsize=(24, 4.3))
    a = t[(t["complexity"] == "all")]
    panel(axes[0], a[a["readout"] == "mean"], VALUES[value], "raw sentence mean")
    panel(
        axes[1], a[a["readout"] == "centered"], "loudness − trajectory mean", "centered on each trajectory's own mean"
    )
    panel(
        axes[2],
        a[a["readout"] == "mid-chain"],
        VALUES[value],
        "middle 60% of the chain only\n(controls for where in the chain commitment lands)",
    )
    panel(
        axes[3],
        a[a["readout"] == "non-direction tokens"],
        VALUES[value],
        "tokens that are NOT themselves direction words",
    )
    panel(
        axes[4],
        a[a["readout"] == "direction-token share"],
        "P(token is a direction word)",
        "the confound itself: how often the token\nIS a direction word",
    )
    fig.suptitle(
        f"Q1 -- loudness against distance from the convinced sentence "
        f"(|rel| >= {span} pooled into the end points; bars = 95% CI over trajectories)",
        fontsize=10,
    )
    save(fig, out / f"q1_by_rel_sentence{suffix}.png")

    fig, axes = plt.subplots(1, 5, figsize=(24, 4.3))
    for ax, readout, ylabel in zip(
        axes,
        ("mean", "centered", "mid-chain", "non-direction tokens", "direction-token share"),
        (VALUES[value], "loudness − trajectory mean", VALUES[value], VALUES[value], "P(token is a direction word)"),
        strict=True,
    ):
        for c in COMPLEXITIES:
            h = t[(t["readout"] == readout) & (t["complexity"] == c)]
            ax.plot(h["rel_sentence"], h["mean"], "-o", color=CCOLOR[c], ms=3.5, lw=1.5, label=f"comp {c}")
        ax.axvline(0, color="red", ls="--", lw=1.2)
        ax.set_xlabel("sentence index − convinced index")
        ax.set_ylabel(ylabel)
        ax.set_title(readout, fontsize=9)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle("Q1 -- loudness against distance from the convinced sentence, split by complexity", fontsize=11)
    save(fig, out / f"q1_by_rel_sentence_by_complexity{suffix}.png")


def q2_chain(df: pd.DataFrame, value: str, out: Path, tab: Path, suffix: str, min_sentences: int):
    edges = np.linspace(0, 1, 21)
    sub = df[df["n_sentences"] >= min_sentences]
    b = binned(sub, "reasoning_frac", value, edges)
    b.to_csv(tab / f"q2_along_chain{suffix}.csv", index=False)
    conv = sub.groupby("name")["convinced_reasoning_frac"].first().dropna()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    draw(ax, b)
    ax.axvline(conv.mean(), color="red", ls="--", lw=1.4, label=f"mean convinced eos ({conv.mean():.2f})")
    ax.axvline(conv.median(), color="darkorange", ls=":", lw=1.4, label=f"median convinced eos ({conv.median():.2f})")
    ax.set_xlabel("position in the reasoning chain  (0 = first token, 1 = last)")
    ax.set_ylabel(VALUES[value])
    ax.set_title(f"loudness along the whole chain ({sub['name'].nunique()} traj, {len(sub)} tokens)", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.hist(conv, bins=20, range=(0, 1), color=REF)
    ax.axvline(conv.mean(), color="red", ls="--", lw=1.4)
    ax.set_xlabel("position of the convinced eos token")
    ax.set_ylabel("trajectories")
    ax.set_title("where commitment lands", fontsize=10)
    ax.grid(alpha=0.25)
    fig.suptitle(
        f"Q2 -- where in the reasoning chain is the lens loudest? "
        f"(trajectories with >= {min_sentences} reasoning sentences)",
        fontsize=10,
    )
    save(fig, out / f"q2_along_chain{suffix}.png")


def q2_chain_by_complexity(df: pd.DataFrame, value: str, out: Path, tab: Path, suffix: str, min_sentences: int):
    edges = np.linspace(0, 1, 21)
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True)
    for ax, c in zip(axes.ravel(), COMPLEXITIES, strict=True):
        sub = df[(df["complexity"] == c) & (df["n_sentences"] >= min_sentences)]
        b = binned(sub, "reasoning_frac", value, edges)
        b["complexity"] = c
        rows.append(b)
        draw(ax, b, color=CCOLOR[c])
        conv = sub.groupby("name")["convinced_reasoning_frac"].first().dropna()
        if len(conv):
            ax.axvline(conv.mean(), color="red", ls="--", lw=1.3)
            ax.axvline(conv.median(), color="darkorange", ls=":", lw=1.3)
        ax.set_title(
            f"complexity {c}  ({sub['name'].nunique()} traj; convinced eos mean "
            f"{conv.mean():.2f}, median {conv.median():.2f})",
            fontsize=9,
        )
    for ax in axes[-1]:
        ax.set_xlabel("position in the reasoning chain")
    for ax in axes[:, 0]:
        ax.set_ylabel(VALUES[value])
    fig.suptitle(
        "Q2 -- loudness along the chain, split by complexity "
        "(red dashed = mean convinced eos, orange dotted = median)",
        fontsize=11,
    )
    save(fig, out / f"q2_along_chain_by_complexity{suffix}.png")

    tabdf = pd.concat(rows, ignore_index=True)
    tabdf.to_csv(tab / f"q2_along_chain_by_complexity{suffix}.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for c in COMPLEXITIES:
        b = tabdf[tabdf["complexity"] == c]
        ax.plot(b["x"], b["mean"], "-o", color=CCOLOR[c], ms=3.5, lw=1.6, label=f"comp {c}")
        sub = df[(df["complexity"] == c) & (df["n_sentences"] >= min_sentences)]
        conv = sub.groupby("name")["convinced_reasoning_frac"].first().dropna()
        if len(conv):
            ax.axvline(conv.mean(), color=CCOLOR[c], ls="--", lw=1.0, alpha=0.7)
    ax.set_xlabel("position in the reasoning chain  (0 = first token, 1 = last)")
    ax.set_ylabel(VALUES[value])
    ax.set_title(
        "Q2 -- loudness along the chain, complexities overlaid\n"
        "(dashed verticals = that complexity's mean convinced eos)",
        fontsize=10,
    )
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    save(fig, out / f"q2_along_chain_overlay{suffix}.png")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-token", type=Path, default=Path("/workspace/reasoning_theatre/loudness/per_token.csv"))
    ap.add_argument("--out", type=Path, default=Path("/workspace/reasoning_theatre/loudness"))
    ap.add_argument("--min-sentence-len", type=int, default=5)
    ap.add_argument("--min-sentences", type=int, default=5)
    ap.add_argument("--values", nargs="+", default=list(VALUES), choices=list(VALUES))
    args = ap.parse_args()

    plots, tables = args.out / "plots", args.out / "tables"
    plots.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    cols = [
        "name",
        "complexity",
        "sentence_len",
        "sentence_frac",
        "rel_sentence",
        "x_sentence",
        "reasoning_frac",
        "n_sentences",
        "convinced_idx",
        "convinced_reasoning_frac",
        "is_direction_token",
        *VALUES,
    ]
    # keep_default_na off everywhere a token could appear; here only numeric columns are read.
    ref = pd.read_csv(args.per_token, usecols=cols, keep_default_na=False, na_values=[""])
    df = ref[ref["convinced_idx"] >= 1]
    print(
        f"{len(ref)} rows / {ref['name'].nunique()} trajectories; "
        f"belief-change cohort {len(df)} rows / {df['name'].nunique()} trajectories",
        flush=True,
    )

    for value in args.values:
        suffix = "" if value == "dir_prob_L15" else "_logmass"
        print(f"[{value}]", flush=True)
        q1_within(df, ref, value, args.min_sentence_len, plots, tables, suffix)
        q1_within_by_complexity(df, value, args.min_sentence_len, plots, tables, suffix)
        q1_commitment(df, value, args.min_sentence_len, plots, tables, suffix)
        q1_commitment_by_complexity(df, value, args.min_sentence_len, plots, tables, suffix)
        q1_by_rel_sentence(df, value, args.min_sentence_len, plots, tables, suffix)
        q2_chain(df, value, plots, tables, suffix, args.min_sentences)
        q2_chain_by_complexity(df, value, plots, tables, suffix, args.min_sentences)
    return 0


if __name__ == "__main__":
    sys.exit(main())
