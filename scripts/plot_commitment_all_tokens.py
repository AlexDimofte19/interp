#!/usr/bin/env python3
"""The commitment boundary at TOKEN resolution, not sentence resolution.

`plot_probe_rollout.py`'s commitment figure has one point per sentence END -- the only place
the rollout actually measures the model's belief. This plots the same comparison over EVERY
reasoning token, so the within-sentence shape is visible and the sentence ends are just the
points where the curve crosses an integer.

The x axis is a continuous sentence coordinate relative to the convinced sentence: a token at
0-indexed position p of a sentence of length L, whose sentence is `s`, sits at

    x = (s - convinced) - 1 + (p + 1) / L

so a sentence occupies one unit, its last token lands exactly on an integer, and x = 0 IS the
convinced boundary (the eos of the convinced sentence -- the first cutoff from which every
later truncation already answers the final action). Integer gridlines are therefore sentence
boundaries and the alternating bands are individual sentences.

Three comparators for the probe's argmax at each token:

  this  -- what the model answers if reasoning is cut at the END of this token's own sentence
           (a measurement in the token's future, for every token but the last of the sentence)
  prev  -- what it answers at the end of the PREVIOUS sentence: the belief already on record
           when this token was emitted
  final -- the action it actually takes after all reasoning (the probe's training label)

Trajectories whose convinced sentence is 0 (the model was already committed before it wrote
anything) are excluded: they have no tokens before the boundary, so including them would put a
different set of trajectories on each side of x = 0.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

# Must match plot_probe_rollout.py's HEADLINE -- the two drifted apart once and the write-up
# ended up mixing two arms.
HEADLINE = "next_action_l15.jlens_topall_mlp"
C = {"this": "#2b6cb0", "prev": "#c05621", "final": "#2f855a", "grey": "#718096"}
N_BOOT = 500


def load(path: Path, arm: str) -> pd.DataFrame:
    cols = [
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
        "sent_answer_prob",
        "convinced_sentence_idx",
        f"{arm}_pred",
    ]
    # keep_default_na=False: decoded tokens include the literal string "NA".
    df = pd.read_csv(path, keep_default_na=False, na_values=[""], usecols=cols)
    actions = ("LEFT", "UP", "RIGHT", "DOWN")
    pred = df[f"{arm}_pred"].map(lambda i: actions[int(i)])
    df["m_this"] = (pred == df["sent_model_action"]).astype(float)
    df["m_prev"] = (pred == df["prev_model_action"]).astype(float)
    df["m_final"] = (pred == df["label_name"]).astype(float)
    df["rel_sentence"] = df["sentence_idx"] - df["convinced_sentence_idx"]
    df["x"] = df["rel_sentence"] - 1 + (df["pos_in_sentence"] + 1) / df["sentence_len"]
    return df[df["convinced_sentence_idx"] >= 1].reset_index(drop=True)


def clustered_band(df: pd.DataFrame, col: str, bin_id: np.ndarray, n_bins: int, seed: int = 0):
    """Mean of `col` per bin, plus a 95% band from resampling TRAJECTORIES, not rows.

    Rows inside a trajectory share a label and a boundary, so a row-level bootstrap would call
    every band several times too narrow.
    """
    vals = df[col].to_numpy(float)
    codes, _ = pd.factorize(df["name"])
    order = np.argsort(codes, kind="stable")
    starts = np.searchsorted(codes[order], np.arange(codes.max() + 1))
    ends = np.append(starts[1:], len(order))
    n_traj = len(starts)

    def means(idx: np.ndarray) -> np.ndarray:
        b = bin_id[idx]
        keep = b >= 0
        b, v = b[keep], vals[idx][keep]
        cnt = np.bincount(b, minlength=n_bins)
        tot = np.bincount(b, weights=v, minlength=n_bins)
        with np.errstate(invalid="ignore"):
            return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)

    point = means(np.arange(len(vals)))
    rng = np.random.default_rng(seed)
    boots = np.empty((N_BOOT, n_bins))
    for b in range(N_BOOT):
        pick = rng.integers(0, n_traj, n_traj)
        boots[b] = means(np.concatenate([order[starts[i] : ends[i]] for i in pick]))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    return point, lo, hi


def fig_by_sentence(df: pd.DataFrame, arm: str, out: Path, window: int, per_sentence: int) -> None:
    lo_x, hi_x = -window - 1, window
    d = df[(df["x"] > lo_x) & (df["x"] <= hi_x)].reset_index(drop=True)
    edges = np.arange(lo_x, hi_x + 1e-9, 1 / per_sentence)
    centers = (edges[:-1] + edges[1:]) / 2
    bin_id = np.clip(np.digitize(d["x"].to_numpy(), edges) - 1, -1, len(centers) - 1)
    bin_id[(d["x"].to_numpy() <= edges[0]) | (d["x"].to_numpy() > edges[-1])] = -1

    fig, (ax, axn) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0.08}
    )
    for s in range(lo_x, hi_x):  # alternating bands = one sentence each
        if s % 2 == 0:
            ax.axvspan(s, s + 1, color=C["grey"], alpha=0.05, lw=0)
            axn.axvspan(s, s + 1, color=C["grey"], alpha=0.05, lw=0)
    for s in range(lo_x, hi_x + 1):
        ax.axvline(s, color=C["grey"], lw=0.6, alpha=0.35)
    ax.axvline(0, color="black", lw=1.8)
    ax.axhline(0.25, color=C["grey"], ls=":", lw=1.2)

    for col, key, label in (
        ("m_this", "this", "belief at the end of THIS sentence"),
        ("m_prev", "prev", "belief at the end of the PREVIOUS sentence"),
        ("m_final", "final", "final action (the probe's label)"),
    ):
        pt, lo, hi = clustered_band(d, col, bin_id, len(centers))
        ax.fill_between(centers, lo, hi, color=C[key], alpha=0.15, lw=0)
        ax.plot(centers, pt, color=C[key], lw=2, label=label)
        ends = d[d["is_sentence_end"] == 1].groupby("rel_sentence")[col].mean()
        ends = ends[(ends.index >= lo_x + 1) & (ends.index <= hi_x)]
        ax.plot(ends.index, ends.values, "o", color=C[key], ms=6, mec="white", mew=1.2, zorder=5)

    lo_y = min(0.15, ax.get_ylim()[0])
    ax.set_ylim(lo_y, ax.get_ylim()[1] + 0.04)
    ax.text(hi_x - 0.05, 0.25, "chance", ha="right", va="bottom", color=C["grey"], fontsize=9)
    ax.text(0.06, ax.get_ylim()[1] - 0.01, "convinced\nboundary", ha="left", va="top", fontsize=9)
    # Past x = 0 every later truncation already answers the final action -- that IS the
    # definition of the convinced sentence -- so all three comparators are one series there.
    ax.text(
        (hi_x + 0.1) / 2,
        lo_y + 0.02,
        "past the boundary the three comparators are\nthe same series by definition of \u201cconvinced\u201d",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=C["grey"],
        style="italic",
    )
    ax.set_ylabel("probe argmax agreement")
    ax.set_title(
        f"Probe agreement at every reasoning token, around the commitment boundary\n"
        f"{arm}  ·  {d['name'].nunique()} trajectories, {len(d):,} tokens  ·  "
        f"markers = sentence ends",
        fontsize=12,
    )
    ax.legend(loc="upper left", frameon=True, fontsize=9, framealpha=0.92, bbox_to_anchor=(0.0, 0.90))

    cnt = pd.Series(bin_id[bin_id >= 0]).value_counts().reindex(range(len(centers)), fill_value=0)
    axn.bar(centers, cnt.values, width=1 / per_sentence * 0.9, color=C["grey"], alpha=0.5)
    for s in range(lo_x, hi_x + 1):
        axn.axvline(s, color=C["grey"], lw=0.6, alpha=0.35)
    axn.axvline(0, color="black", lw=1.8)
    axn.set_ylabel("tokens")
    axn.set_xlabel("sentence index relative to the convinced sentence (0 = its last token; each band is one sentence)")
    axn.set_xticks(range(lo_x, hi_x + 1))
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_by_token(df: pd.DataFrame, arm: str, out: Path, span: int, step_tok: int) -> None:
    conv = df[(df["sentence_idx"] == df["convinced_sentence_idx"]) & (df["is_sentence_end"] == 1)]
    conv = conv[["name", "step", "token_idx"]].rename(columns={"token_idx": "conv_tok"})
    d = df.merge(conv, on=["name", "step"], how="inner")
    d["dt"] = d["token_idx"] - d["conv_tok"]
    d = d[(d["dt"] >= -span) & (d["dt"] <= span)].reset_index(drop=True)

    edges = np.arange(-span, span + 1e-9, step_tok)
    centers = (edges[:-1] + edges[1:]) / 2
    bin_id = np.clip(np.digitize(d["dt"].to_numpy(), edges) - 1, -1, len(centers) - 1)
    bin_id[(d["dt"].to_numpy() < edges[0]) | (d["dt"].to_numpy() > edges[-1])] = -1

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axvline(0, color="black", lw=1.8)
    ax.axhline(0.25, color=C["grey"], ls=":", lw=1.2)
    for col, key, label in (
        ("m_this", "this", "belief at the end of THIS sentence"),
        ("m_prev", "prev", "belief at the end of the PREVIOUS sentence"),
        ("m_final", "final", "final action (the probe's label)"),
    ):
        pt, lo, hi = clustered_band(d, col, bin_id, len(centers))
        ax.fill_between(centers, lo, hi, color=C[key], alpha=0.15, lw=0)
        ax.plot(centers, pt, color=C[key], lw=2, label=label)
    ax.set_ylabel("probe argmax agreement")
    ax.set_xlabel("reasoning tokens from the convinced boundary")
    ax.legend(loc="lower left", frameon=True, fontsize=9, framealpha=0.92)
    # The bin straddling 0 holds the boundary token itself (whose previous sentence can still
    # be wrong); every bin beyond it has the three comparators identical, as above.

    # The sentence index on the same x, so the sentence structure is legible in raw-token space.
    ax2 = ax.twinx()
    ax2.grid(False)
    sent = d.groupby(pd.cut(d["dt"], edges), observed=True)["rel_sentence"].mean()
    ax2.step(centers, sent.values, where="mid", color=C["grey"], lw=1.4, alpha=0.8)
    ax2.axhline(0, color=C["grey"], lw=0.6, alpha=0.5)
    ax2.set_ylabel("mean sentence index relative to convinced", color=C["grey"])
    ax2.tick_params(axis="y", colors=C["grey"])
    ax.set_title(
        f"The same comparison in raw token distance\n{arm}  ·  "
        f"{d['name'].nunique()} trajectories, {len(d):,} tokens  ·  "
        f"grey step = relative sentence index",
        fontsize=12,
    )
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path("/workspace/reasoning_theatre/probe_vs_rollout")
    ap.add_argument("--per-token", type=Path, default=root / "per_token.csv")
    ap.add_argument("--out-dir", type=Path, default=root / "plots")
    ap.add_argument("--arm", default=HEADLINE)
    ap.add_argument("--window", type=int, default=4, help="Sentences either side of the boundary.")
    ap.add_argument("--bins-per-sentence", type=int, default=5)
    ap.add_argument("--token-span", type=int, default=120)
    ap.add_argument("--token-bin", type=int, default=10)
    args = ap.parse_args()

    sns.set_theme(style="whitegrid", context="notebook")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load(args.per_token, args.arm)
    print(f"{len(df):,} tokens from {df['name'].nunique()} trajectories with a convinced sentence >= 1")
    fig_by_sentence(
        df, args.arm, args.out_dir / "commit_all_tokens_by_sentence.png", args.window, args.bins_per_sentence
    )
    fig_by_token(df, args.arm, args.out_dir / "commit_all_tokens_by_token.png", args.token_span, args.token_bin)


if __name__ == "__main__":
    main()
