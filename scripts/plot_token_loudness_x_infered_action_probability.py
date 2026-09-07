#!/usr/bin/env python3
"""Figures for ``build_token_loudness_x_infered_action_probability.py``'s table.

The join asks one question: **does a token being LOUD under a lens predict what the model
answers if you cut its reasoning there?** Three figures answer it, in the order the claim
has to be defended.

``01_loudness_vs_answer.png``   the result. Per-lens decile of layer-15 direction mass
                                against three things the truncated model does: how much
                                probability it puts on its answer, whether that answer is
                                right, and whether it CHANGES the answer the previous cutoff
                                gave. Deciles are per lens because a lens is used as a
                                RANKER -- "cut at the loudest token" -- so its own ordering
                                is the axis that matters.

                                NOT plotted, because it is not a second measurement here:
                                "answers what it will answer after the whole chain". All 360
                                held-out chains end on the ground-truth action, so that
                                series is identically the correctness one. Worth knowing
                                before reading correctness as anything other than agreement
                                with the model's own settled answer.
``02_rulers.png``               the two lenses as selectors (per-trajectory rank buckets,
                                the shape entry 37(c) reports) and against each other, so
                                "the jlens ranks better" is visible rather than asserted.
``03_position_control.png``     the confound, and it is not a small one. Loudness rises
                                through a reasoning chain and so does commitment, so a
                                loudness effect could be a position effect wearing a hat.
                                The third panel re-runs the decile curve WITHIN terciles of
                                chain position, where that explanation is unavailable.

Bands are 95% bootstrap intervals resampling TRAJECTORIES, never rows: 87k rows come from
360 chains, tokens inside a chain share a label and a commitment boundary, and a row-level
bootstrap would call every band several times too narrow. ``clustered_band`` is imported
from ``plot_commitment_all_tokens.py`` rather than re-derived, so every figure in this line
gets its uncertainty the same way.

Palette: the two-hue categorical pair already used by ``plot_loud_vs_sentence_end.py``,
checked with the data-viz validator on the ``#fcfcfb`` light surface these figures ship on
-- OKLCH L 0.575/0.671, chroma 0.163/0.175, WCAG 4.30/3.12, and OKLab dE 33.6 unsimulated,
24.7 protan / 31.7 deutan / 32.7 tritan. Colour never carries identity alone: every series
is also direct-labelled or legended.
"""

import argparse
import importlib
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The trajectory-clustered bootstrap, imported so this line has exactly one of them.
clustered_band = importlib.import_module("scripts.plot_commitment_all_tokens").clustered_band  # noqa: E402

DEFAULT_CSV = Path("/workspace/reasoning_theatre/loudness_vs_answer_prob/heldout360_per_token.csv")
DEFAULT_OUT = Path("/workspace/reasoning_theatre/loudness_vs_answer_prob/plots")

# Fixed hue order, never cycled: slot 0 is always the jlens, slot 1 always the logit lens.
LENS_COLOR = {"jlens": "#2a78d6", "logitlens": "#eb6834"}
LENS_LABEL = {"jlens": "Jacobian lens", "logitlens": "logit lens"}
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#d9d8d4", "#fcfcfb"
CHANCE = 0.25  # four actions

# The three things a cut at this token produces. (column, axis label, reference line)
METRICS = (
    ("answer_prob", "P(the answer it gave)", None),
    ("correct", "answers the ground-truth action", CHANCE),
    ("switch", "changes the answer the\nPREVIOUS token's cutoff gave", None),
)

# Per-trajectory loudness rank -> bucket, mirroring the shape entry 37(c) reports.
RANK_EDGES = [(1, 1, "top-1"), (2, 5, "2-5"), (6, 20, "6-20"), (21, 99, "21-99"), (100, 10**9, "100+")]


def load(path: Path, lenses: list[str], layer: int) -> pd.DataFrame:
    """The join, plus the two per-chain quantities the figures need but it does not store."""
    cols = ["name", "step", "token_idx", "reasoning_pos", "model_action", "answer_prob", "correct"]
    cols += [f"{lens}_logmass_L{layer}" for lens in lenses]
    # keep_default_na=False: decoded tokens include the literal string "NA".
    df = pd.read_csv(path, keep_default_na=False, na_values=[""], usecols=cols)

    # Sorted once, index reset: every per-chain quantity below is positional ("the LAST
    # cutoff of this chain"), so the frame has to be in chain order to begin with.
    df = df.sort_values(["name", "step", "token_idx"]).reset_index(drop=True)
    chain = df.groupby(["name", "step"], sort=False)
    # What the model answers with the whole chain in front of it: the last cutoff's action.
    # Whether cutting one token later changes the answer. The chain's first cutoff has no
    # predecessor and is left out of the mean rather than counted as "no change".
    previous = chain["model_action"].shift(1)
    df["switch"] = np.where(previous.isna(), np.nan, df["model_action"] != previous).astype(float)
    # Position in the chain, 0 at the first reasoning token and 1 at the last.
    n_tokens = chain["reasoning_pos"].transform("max")
    df["chain_frac"] = np.where(n_tokens > 0, df["reasoning_pos"] / n_tokens.clip(lower=1), 1.0)
    df["correct"] = df["correct"].astype(float)
    return df


def decile(series: pd.Series) -> np.ndarray:
    """0-based decile index of a value within its own lens's distribution."""
    return pd.qcut(series, 10, labels=False, duplicates="drop").to_numpy()


def rank_bucket(df: pd.DataFrame, col: str) -> pd.Series:
    """Each token's 1-based loudness rank inside its own chain, bucketed."""
    rank = df.groupby(["name", "step"], sort=False)[col].rank(ascending=False, method="first")
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    for lo, hi, label in RANK_EDGES:
        out[(rank >= lo) & (rank <= hi)] = label
    return out


def style_axis(ax, *, reference: float | None = None, label: str = "chance", label_x: float = 0.99) -> None:
    """Recessive grid and spines, plus an optional reference line.

    ``label_x`` moves the reference's tag off whichever corner the legend took.
    """
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    if reference is not None:
        ax.axhline(reference, color=INK_MUTED, ls=":", lw=1.2, zorder=1)
        ax.text(
            label_x,
            reference,
            label,
            transform=ax.get_yaxis_transform(),
            ha="right" if label_x > 0.5 else "left",
            va="bottom",
            color=INK_MUTED,
            fontsize=8.5,
        )


def finish(fig, out: Path) -> None:
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def fig_loudness_vs_answer(df: pd.DataFrame, lenses: list[str], layer: int, out: Path) -> pd.DataFrame:
    """The headline: what the model does at a token, against how loud that token is."""
    fig, axes = plt.subplots(1, len(METRICS), figsize=(15, 4.6))
    x = np.arange(1, 11)
    records = []
    for ax, (col, ylabel, reference) in zip(axes, METRICS, strict=True):
        # A metric may be undefined for some tokens -- "switch" needs a predecessor, so each
        # chain's first token has none. Drop those rows rather than counting them as zero,
        # which would report a change rate 360 tokens too low.
        sub = df[df[col].notna()].reset_index(drop=True) if df[col].isna().any() else df
        for lens in lenses:
            bins = decile(sub[f"{lens}_logmass_L{layer}"])
            point, lo, hi = clustered_band(sub, col, bins, 10)
            ax.fill_between(x, lo, hi, color=LENS_COLOR[lens], alpha=0.15, lw=0)
            ax.plot(
                x,
                point,
                color=LENS_COLOR[lens],
                lw=2,
                marker="o",
                ms=5,
                mec=SURFACE,
                mew=1.2,
                label=LENS_LABEL[lens],
                zorder=3,
            )
            for d in range(10):
                records.append(
                    {
                        "lens": lens,
                        "metric": col,
                        "decile": d + 1,
                        "mean": point[d],
                        "lo": lo[d],
                        "hi": hi[d],
                        "mean_logmass": sub.loc[bins == d, f"{lens}_logmass_L{layer}"].mean(),
                    }
                )
        style_axis(ax, reference=reference)
        ax.set_xticks(x)
        ax.set_xlabel(
            f"decile of layer-{layer} direction mass\n(1 = quietest, 10 = loudest)", fontsize=9.5, color=INK_MUTED
        )
        ax.set_ylabel(ylabel, fontsize=10, color=INK)
        ax.set_xlim(0.5, 10.5)
    # One legend per panel: with the direct labels gone, this is the only thing naming a
    # series, and the reader should not have to look left to decode the panel in front of them.
    # A panel carrying a reference line puts its legend on the left, where that line's label
    # is not; the others keep the empty lower-right corner.
    for ax, (_, _, reference) in zip(axes, METRICS, strict=True):
        ax.legend(
            loc="lower left" if reference is not None else "lower right",
            frameon=True,
            framealpha=0.92,
            edgecolor=GRID,
            fontsize=9,
        )
    fig.subplots_adjust(wspace=0.34)
    fig.suptitle(
        f"Cut the chain at one token: how the model answers, against how loud that token is\n"
        f"held-out 360  ·  {len(df):,} reasoning tokens, {df['name'].nunique()} trajectories  ·  "
        f"bands are 95% CIs bootstrapped over trajectories",
        fontsize=12.5,
        color=INK,
        y=1.06,
    )
    finish(fig, out)
    return pd.DataFrame(records)


def fig_rulers(df: pd.DataFrame, lenses: list[str], layer: int, out: Path) -> pd.DataFrame:
    """The two lenses as selectors, and against each other."""
    fig, (ax_rank, ax_scatter) = plt.subplots(1, 2, figsize=(13.5, 4.8), width_ratios=[1.25, 1])
    labels = [label for _, _, label in RANK_EDGES]
    width = 0.38
    records = []
    for i, lens in enumerate(lenses):
        buckets = rank_bucket(df, f"{lens}_logmass_L{layer}")
        means = [df.loc[buckets == label, "correct"].mean() for label in labels]
        counts = [int((buckets == label).sum()) for label in labels]
        # 2px surface gap between adjacent bars, and the value stated rather than guessed.
        pos = np.arange(len(labels)) + (i - 0.5) * width
        ax_rank.bar(
            pos,
            means,
            width=width * 0.94,
            color=LENS_COLOR[lens],
            label=LENS_LABEL[lens],
            edgecolor=SURFACE,
            lw=1.0,
            zorder=3,
        )
        for p, m in zip(pos, means, strict=True):
            ax_rank.text(p, m + 0.012, f"{m:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK)
        records += [
            {"lens": lens, "bucket": label, "n": n, "correct": m}
            for label, n, m in zip(labels, counts, means, strict=True)
        ]
    style_axis(ax_rank, reference=CHANCE)
    ax_rank.set_xticks(range(len(labels)), labels)
    ax_rank.set_xlabel("loudness rank of the token within its own chain", fontsize=9.5, color=INK_MUTED)
    ax_rank.set_ylabel("answers the ground-truth action", fontsize=10, color=INK)
    ax_rank.legend(loc="upper right", frameon=False, fontsize=9)
    ax_rank.set_title("As a selector: cut at the chain's loudest tokens", fontsize=10.5, color=INK, loc="left")

    jl, ll = (df[f"{lens}_logmass_L{layer}"] for lens in ("jlens", "logitlens"))
    hb = ax_scatter.hexbin(jl, ll, gridsize=55, bins="log", cmap="Blues", mincnt=1, linewidths=0)
    lims = [min(jl.min(), ll.min()), max(jl.max(), ll.max())]
    ax_scatter.plot(lims, lims, color=INK_MUTED, ls="--", lw=1.1, zorder=3)
    ax_scatter.annotate(
        "equal",
        (lims[1], lims[1]),
        xytext=(-10, -14),
        textcoords="offset points",
        color=INK_MUTED,
        fontsize=8.5,
        ha="right",
    )
    style_axis(ax_scatter)
    ax_scatter.set_xlabel(f"Jacobian lens, L{layer} log direction mass", fontsize=9.5, color=LENS_COLOR["jlens"])
    ax_scatter.set_ylabel(f"logit lens, L{layer} log direction mass", fontsize=9.5, color=LENS_COLOR["logitlens"])
    pearson, spearman = jl.corr(ll), jl.corr(ll, method="spearman")
    ax_scatter.set_title(
        f"The same quantity, two rulers:  r = {pearson:.2f},  rho = {spearman:.2f}",
        fontsize=10.5,
        color=INK,
        loc="left",
    )
    cb = fig.colorbar(hb, ax=ax_scatter, pad=0.02)
    cb.set_label("tokens per cell (log)", fontsize=8.5, color=INK_MUTED)
    cb.ax.tick_params(labelsize=8, colors=INK_MUTED)
    finish(fig, out)
    return pd.DataFrame(records)


def fig_position_control(df: pd.DataFrame, lenses: list[str], layer: int, out: Path) -> None:
    """The confound: loudness and commitment both rise through a chain."""
    fig, (ax_loud, ax_commit, ax_within) = plt.subplots(1, 3, figsize=(15.5, 4.6))
    edges = np.linspace(0, 1, 11)
    centers = (edges[:-1] + edges[1:]) / 2
    bins = np.clip(np.digitize(df["chain_frac"].to_numpy(), edges) - 1, 0, 9)

    for lens in lenses:
        point, lo, hi = clustered_band(df, f"{lens}_logmass_L{layer}", bins, 10)
        ax_loud.fill_between(centers, lo, hi, color=LENS_COLOR[lens], alpha=0.15, lw=0)
        ax_loud.plot(centers, point, color=LENS_COLOR[lens], lw=2, label=LENS_LABEL[lens])
    style_axis(ax_loud)
    ax_loud.set_ylabel(f"mean L{layer} log direction mass", fontsize=10, color=INK)
    ax_loud.set_xlabel("position in the reasoning chain", fontsize=9.5, color=INK_MUTED)
    ax_loud.legend(loc="lower right", frameon=False, fontsize=9)
    ax_loud.set_title("Loudness rises through a chain...", fontsize=10.5, color=INK, loc="left")

    point, lo, hi = clustered_band(df, "correct", bins, 10)
    ax_commit.fill_between(centers, lo, hi, color=INK_MUTED, alpha=0.15, lw=0)
    ax_commit.plot(centers, point, color=INK, lw=2)
    style_axis(ax_commit, reference=CHANCE)
    ax_commit.set_ylabel("answers the ground-truth action", fontsize=10, color=INK)
    ax_commit.set_xlabel("position in the reasoning chain", fontsize=9.5, color=INK_MUTED)
    ax_commit.set_title("...and so does commitment. Hence panel 3.", fontsize=10.5, color=INK, loc="left")

    # Within a tercile of chain position, "later in the chain" cannot explain the slope.
    tercile = pd.qcut(df["chain_frac"], 3, labels=False, duplicates="drop")
    shades = {0: "#86b6ef", 1: "#2a78d6", 2: "#104281"}
    names = {0: "first third", 1: "middle third", 2: "last third"}
    x = np.arange(1, 11)
    for t in sorted(shades):
        sub = df[tercile == t].reset_index(drop=True)
        point, _, _ = clustered_band(sub, "correct", decile(sub[f"jlens_logmass_L{layer}"]), 10)
        ax_within.plot(
            x,
            point,
            color=shades[t],
            lw=2,
            marker="o",
            ms=4,
            mec=SURFACE,
            mew=1.0,
            label=f"{names[t]} of the chain",
        )
    style_axis(ax_within, reference=CHANCE, label_x=0.01)
    ax_within.set_xticks(x)
    ax_within.set_ylabel("answers the ground-truth action", fontsize=10, color=INK)
    ax_within.set_xlabel(
        f"decile of Jacobian-lens L{layer} mass,\nWITHIN the position tercile", fontsize=9.5, color=INK_MUTED
    )
    ax_within.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor=GRID, fontsize=9)
    ax_within.set_title("The loudness slope, position held", fontsize=10.5, color=INK, loc="left")

    fig.suptitle(
        "Is it loudness, or is it just being late in the chain?\n"
        f"held-out 360  ·  {len(df):,} reasoning tokens  ·  bands bootstrapped over trajectories",
        fontsize=12.5,
        color=INK,
        y=1.06,
    )
    finish(fig, out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-token", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--lenses", default="jlens,logitlens")
    ap.add_argument("--layer", type=int, default=15)
    args = ap.parse_args()

    lenses = [lens.strip() for lens in args.lenses.split(",") if lens.strip()]
    sns.set_theme(style="white", context="notebook")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load(args.per_token, lenses, args.layer)
    print(f"{len(df):,} tokens over {df['name'].nunique()} trajectories from {args.per_token}", flush=True)

    deciles = fig_loudness_vs_answer(df, lenses, args.layer, args.out_dir / "01_loudness_vs_answer.png")
    buckets = fig_rulers(df, lenses, args.layer, args.out_dir / "02_rulers.png")
    fig_position_control(df, lenses, args.layer, args.out_dir / "03_position_control.png")

    print("\ndecile means (the numbers behind figure 1):", flush=True)
    print(deciles.pivot_table(index="decile", columns=["metric", "lens"], values="mean").round(3), flush=True)
    print("\nrank buckets (figure 2), fraction correct:", flush=True)
    order = [label for _, _, label in RANK_EDGES]
    table = buckets.pivot_table(index="bucket", columns="lens", values="correct").reindex(order)
    print(table.round(3), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
