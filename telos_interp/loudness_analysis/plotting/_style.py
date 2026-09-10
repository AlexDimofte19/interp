"""Shared drawing scaffolding for every loudness figure.

Three plotters each carried their own copy of this: two ways to save a figure, three quantile
binners, two balanced accuracies and three sets of colour constants. A figure that styles its
own axes slightly differently from the one beside it is not comparable with it, which is the
whole reason this is one module.

`axis_label` comes from `columns`, so a figure cannot name its ruler differently from the
table it was drawn from.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from telos_interp.loudness_analysis import columns as _cols  # noqa: E402
from telos_interp.loudness_analysis import stats as _stats  # noqa: E402
from telos_interp.loudness_analysis.analysis.probe_accuracy_by_loudness import (  # noqa: E402
    REL_TOKEN_BINS,
    REL_TOKEN_LABELS,
)

axis_label = _cols.axis_label

clustered_band = _stats.clustered_band

# NOTE: `qbin` and `bal_acc` below are the figures' OWN versions, kept verbatim rather than
# aliased to stats.qbin / stats.bal_acc. They are not interchangeable: this `qbin` hardcodes
# labels=False and returns integer bin codes, which every figure indexes as an integer, while
# stats.qbin defaults to labelled bins. The analysis layer wants labels; the drawing layer
# wants codes. Aliasing one to the other changes how each figure bins, and nothing would look
# wrong until the plot came out.


# ---- constants -----------------------------------------------------------------------
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



# ---- helpers -------------------------------------------------------------------------
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


