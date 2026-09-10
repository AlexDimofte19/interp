"""Every loudness figure, behind one registry.

Three scripts drew these: plot_probe_loudness.py (6 figures), plot_sentence_loudness.py (7)
and plot_token_loudness_x_infered_action_probability.py (3). The bodies are unchanged -- they
draw published figures and this is a relocation, not a redesign -- but the scaffolding they
each carried now lives in `_style`, and one CLI reaches all of them.

A figure declares WHICH TABLE it needs, because the three groups read different per-token
tables and asking for a figure without its input should be an error rather than a traceback
from inside matplotlib:

  probe         a joined probe table (join_rollouts.py output)
  distribution  a per-token loudness table with sentence coordinates
  rollout       a per-token table carrying the rollout's answer and its probability
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from telos_interp.loudness_analysis import provenance as _prov
from telos_interp.loudness_analysis.plotting._style import (  # noqa: F401
    CCOLOR,
    CHANCE,
    COLOR,
    COMPLEXITIES,
    GRID,
    INK,
    INK_MUTED,
    LABEL,
    LENS_COLOR,
    LENS_LABEL,
    MAIN,
    METRICS,
    PROBES,
    RANK_EDGES,
    REF,
    REL_TOKEN_BINS,
    REL_TOKEN_LABELS,
    SURFACE,
    TITLE,
    VALUES,
    acc_by,
    axis_label,
    bal_acc,
    binned,
    clustered_band,
    curve,
    decile,
    draw,
    finish,
    group_frames,
    load,
    np,
    pd,
    plt,
    qbin,
    rank_bucket,
    save,
    sns,
    style_axis,
)


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


# ---------------------------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Figure:
    """One figure: what it draws, and which table it needs to draw it."""

    name: str
    table: str
    fn: Callable
    description: str = ""


FIGURES: dict[str, Figure] = {
    f.name: f
    for f in [
        Figure(
            "loudness_distribution",
            "probe",
            fig_distribution,
            "Where the selected tokens sit in the whole-chain loudness distribution.",
        ),
        Figure(
            "accuracy_by_bin",
            "probe",
            fig_by_bin,
            "Balanced accuracy per bin of a chosen axis, against both label definitions.",
        ),
        Figure("loudness_x_position", "probe", fig_grid, "The 3x3 loudness x within-sentence-position grid."),
        Figure(
            "which_label_followed",
            "probe",
            fig_follows,
            "On disagreement rows, which label the probe follows, by loudness decile.",
        ),
        Figure(
            "chain_length_control",
            "probe",
            fig_chain_length,
            "Loudness terciles re-cut inside chain-length quartiles.",
        ),
        Figure(
            "label_comparison",
            "probe",
            fig_label_comparison,
            "Belief label vs final-action baseline vs random control on one token set.",
        ),
        Figure("within_sentence", "distribution", q1_within, "Loudness against position within a sentence."),
        Figure(
            "within_sentence_by_complexity",
            "distribution",
            q1_within_by_complexity,
            "The same, split by grid complexity.",
        ),
        Figure(
            "around_commitment",
            "distribution",
            q1_commitment,
            "Loudness in a +-1 sentence window around the commitment boundary.",
        ),
        Figure(
            "around_commitment_by_complexity",
            "distribution",
            q1_commitment_by_complexity,
            "The same, split by grid complexity.",
        ),
        Figure(
            "by_rel_sentence",
            "distribution",
            q1_by_rel_sentence,
            "Five readouts against sentence distance from the boundary.",
        ),
        Figure("along_chain", "distribution", q2_chain, "Loudness along the whole reasoning chain, 20 bins."),
        Figure(
            "along_chain_by_complexity", "distribution", q2_chain_by_complexity, "The same, split by grid complexity."
        ),
        Figure(
            "loudness_vs_answer",
            "rollout",
            fig_loudness_vs_answer,
            "Loudness decile against the answer the model gives if cut there.",
        ),
        Figure("rulers", "rollout", fig_rulers, "The two lenses as selectors, plus their agreement."),
        Figure(
            "position_control",
            "rollout",
            fig_position_control,
            "The loudness curve re-run within terciles of chain position.",
        ),
    ]
}


def figure_names() -> list[str]:
    """Registered figures, in registry order.

    >>> "loudness_distribution" in figure_names()
    True
    >>> sorted({f.table for f in FIGURES.values()})
    ['distribution', 'probe', 'rollout']
    """
    return list(FIGURES)


def get_figure(name: str) -> Figure:
    """Look up one figure.

    >>> get_figure("rulers").table
    'rollout'
    >>> get_figure("nope")
    Traceback (most recent call last):
        ...
    ValueError: Unknown figure 'nope'; available: ...
    """
    try:
        return FIGURES[name]
    except KeyError:
        raise ValueError(f"Unknown figure {name!r}; available: {sorted(FIGURES)}") from None


# ---------------------------------------------------------------------------------------------
# Per-table drawers, ported from the three scripts' main() bodies.
#
# The registry above is a CATALOGUE, not a call interface: the sixteen figure functions have
# seven different signatures, and the orchestration -- which rowsets, which bin columns, which
# value columns -- lived in each script's main(). These keep that orchestration verbatim.
# ---------------------------------------------------------------------------------------------


def draw_probe(df: pd.DataFrame, args) -> None:
    """Every probe-table figure, exactly as plot_probe_loudness.py drew them."""
    # A spare palette for arms the registry never named: distinct from the blue (belief),
    # orange (final-label baseline) and grey (control) families already in COLOR.
    spare = ["#7a1f5a", "#b5568f", "#1f7a5a", "#4aa88a", "#7a5a1f", "#b58f56"]
    sep = ";" if ";" in args.extra_probes else ","
    for n, entry in enumerate(t.strip() for t in args.extra_probes.split(sep) if t.strip()):
        name, eq, rest = entry.partition("=")
        if not eq:
            raise SystemExit(f"--extra-probes entry {entry!r} is not name=Label[=#rrggbb]")
        label, _, colour = rest.partition("=")
        if args.extra_probes_rowset not in PROBES:
            raise SystemExit(f"unknown --extra-probes-rowset {args.extra_probes_rowset!r}")
        PROBES[args.extra_probes_rowset].append(name)
        LABEL[name] = label
        COLOR[name] = colour or spare[n % len(spare)]

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
        n_fig = 4
        if len(rel):
            rel["rel_clipped"] = rel["rel_sentence"].clip(-6, 6).astype(int)
            fig_by_bin(rel, probes, rs, "rel_clipped", "sentence_idx - convinced_idx", "by_rel_sentence", args.out)
            n_fig += 1
        # The same boundary, located per token instead of at a sentence end. Drop the
        # steps already committed at the no-reasoning cutoff: they have no pre-boundary
        # side, so they would load the positive bins only.
        live = d[(d["rel_token"].notna()) & (d["convinced_before_reasoning"] == 0)].copy()
        if len(live):
            live["rel_sent_tok_clipped"] = live["rel_sentence_token"].clip(-6, 6).astype(int)
            fig_by_bin(
                live,
                probes,
                rs,
                "rel_sent_tok_clipped",
                "sentence_idx - sentence of the PER-TOKEN boundary",
                "by_rel_sentence_token",
                args.out,
            )
            live["rel_token_bin"] = pd.cut(
                live["rel_token"], bins=REL_TOKEN_BINS, labels=REL_TOKEN_LABELS, include_lowest=True
            )
            fig_by_bin(
                live,
                probes,
                rs,
                "rel_token_bin",
                "tokens from the per-token commitment boundary",
                "by_rel_token",
                args.out,
            )
            n_fig += 2
        mlps = [p for p in probes if p.endswith("mlp")]
        fig_grid(d, mlps[0], rs, args.out)
        fig_follows(d, mlps, rs, args.out)
        fig_chain_length(d, mlps, rs, args.out)
        print(f"{rs}: {n_fig} figures", flush=True)

    d2 = df[df["rowset"] == "p2"]
    if len(d2):
        fig_label_comparison(d2, args.out)
        print("p2_label_comparison.png", flush=True)
    print(f"-> {args.out}", flush=True)
    return 0


def draw_distribution(df: pd.DataFrame, args) -> None:
    """Every distribution figure, exactly as plot_sentence_loudness.py drew them."""

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


def draw_rollout(df: pd.DataFrame, args) -> None:
    """Every rollout figure, exactly as plot_token_loudness_x_infered_action_probability.py drew them."""

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


DRAWERS = {
    "probe": draw_probe,
    "distribution": draw_distribution,
    "rollout": draw_rollout,
}


TABLE_FLAGS = {
    "probe": "--probe-table",
    "distribution": "--distribution-table",
    "rollout": "--rollout-table",
}


def _read(path: Path) -> pd.DataFrame:
    """Loudness tables quote a token column that contains the literal string "NA", embedded
    commas and newlines -- so NA handling is off, as everywhere else in this line."""
    return pd.read_csv(path, keep_default_na=False, na_values=[""], low_memory=False)


def main(argv: list[str] | None = None) -> int:
    """`argv` is taken rather than read from sys.argv so the CLI is testable."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--figure",
        default="all",
        help="Figure to draw, or 'all' (default) for every one whose table was supplied. "
        f"Available: {', '.join(figure_names())}",
    )
    ap.add_argument("--probe-table", type=Path, default=None, help="join_rollouts.py output.")
    ap.add_argument(
        "--distribution-table", type=Path, default=None, help="Per-token loudness with sentence coordinates."
    )
    ap.add_argument("--rollout-table", type=Path, default=None, help="Per-token table with the rollout answer.")
    ap.add_argument("--out", type=Path, required=True, help="Directory to write the PNGs into.")
    ap.add_argument("--lens", default="jlens", help="Ruler the axis labels name (default: %(default)s).")
    ap.add_argument("--signal-name", default="direction", help="Vocabulary the axis labels name.")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--list", action="store_true", help="List the figures and their tables, then exit.")
    # --- names the ported drawers read -------------------------------------------------------
    # Kept verbatim from the three scripts, so the CLI supplies the names their bodies use
    # rather than the bodies being rewritten around new ones.
    ap.add_argument(
        "--all-token-loudness",
        type=Path,
        default=None,
        help="Whole-chain loudness table, for the distribution figure's reference histogram.",
    )
    ap.add_argument(
        "--extra-probes",
        default="",
        help="name=Label[=#rrggbb] entries added to --extra-probes-rowset. Use ';' between "
        "entries if a label contains a comma. With no flag every figure is unchanged.",
    )
    ap.add_argument("--extra-probes-rowset", default="p2", help="Rowset the --extra-probes join.")
    ap.add_argument("--min-sentence-len", type=int, default=5)
    ap.add_argument("--min-sentences", type=int, default=5)
    ap.add_argument(
        "--values",
        default=",".join(VALUES),
        help="Which value column(s) the distribution figures draw (default: both scales).",
    )
    ap.add_argument("--lenses", default="jlens,logitlens", help="Lenses the rollout figures compare.")
    args = ap.parse_args(argv)

    # Two of the three scripts named the output directory differently; the drawers still use
    # their own name, so both point at the same place.
    args.out_dir = args.out
    args.per_token = args.probe_table or args.distribution_table or args.rollout_table
    # draw_rollout splits --lenses itself, so it stays a string here; draw_distribution
    # ITERATES --values, so that one is split. The two drawers genuinely differ.
    if isinstance(args.values, str):
        args.values = [x.strip() for x in args.values.split(",") if x.strip()]

    if args.list:
        width = max(len(n) for n in figure_names())
        for name, fig in FIGURES.items():
            print(f"{name.ljust(width)}  {fig.table:13s}  {fig.description}")
        return 0

    tables = {
        "probe": args.probe_table,
        "distribution": args.distribution_table,
        "rollout": args.rollout_table,
    }

    if args.figure == "all":
        wanted = [f for f in FIGURES.values() if tables.get(f.table)]
        if not wanted:
            raise SystemExit("no input tables given; pass at least one of " + ", ".join(sorted(TABLE_FLAGS.values())))
    else:
        fig = get_figure(args.figure)
        if not tables.get(fig.table):
            raise SystemExit(f"{args.figure!r} needs a {fig.table} table: pass {TABLE_FLAGS[fig.table]}")
        # A drawer emits its table's whole set; naming one figure selects its table.
        wanted = [f for f in FIGURES.values() if f.table == fig.table]

    args.out.mkdir(parents=True, exist_ok=True)
    drawn = []
    for table in dict.fromkeys(f.table for f in wanted):
        df = _read(tables[table])
        print(f"read {len(df):,} rows for the {table} figures", flush=True)
        DRAWERS[table](df, args)
        drawn += [f.name for f in wanted if f.table == table]

    cfg = _prov.RunConfig("loudness_analysis/plotting/figures.py")
    cfg.measurement(lens=args.lens, signal=args.signal_name, layer=args.layer)
    for kind, path in tables.items():
        if path:
            cfg.input(kind, path)
    cfg.params.update({"figures": drawn})
    cfg.guard(args.out, "lens", "signal", "layer")
    cfg.write(args.out)
    print(f"\nwrote {len(drawn)} figure(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
