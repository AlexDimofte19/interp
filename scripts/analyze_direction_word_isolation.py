#!/usr/bin/env python3
"""Isolate the direction words: is loudness reading a word the model already typed?

THE CONFOUND. The tokens a lens calls loud are disproportionately the direction words the
model has *already verbalized* -- CLAUDE.md states it plainly. If a probe scores well only on
those, "the residual carries the action" collapses into "the token says ``up``".

It has been tested twice before and survived both times, but never like this:

  * the held-out per-token scoring split the SELECTED top-20 bucket and found identical
    accuracy (.6878 vs .6884) -- but only inside the loud regime, and against the
    FINAL-action label (ICLR log entry 37).
  * probe accuracy read by loudness dropped direction words entirely and the jlens-loudness
    gradient held -- but on the eval-720 split, with the selection still truncating the
    loudness axis (ICLR log entry 46).

Here the selection is gone (all 87,221 held-out tokens), the label is the at-token local
belief, sixteen probes are in play, and -- the part no previous pass did -- **every question
is asked separately of jlens loudness and of logitlens loudness**, which disagree about which
tokens are loud (~50% top-20 overlap at layer 15).

FOUR QUESTIONS, in the order they have to be answered:

  1. CONCENTRATION. Does loudness select verbalized tokens, and do the two lenses do it to
     the same degree? Direction words are 6.2% of all tokens; entry 37 measured 34.6% inside
     the jlens-selected top-20 and 24.5% inside the logitlens one, so the answer is already
     "yes, and differently" -- this quantifies it per decile.
  2. SURVIVAL. Within NON-direction words alone, does accuracy still rise with loudness?
     That is the confound-free version of the headline claim.
  3. THE GAP. How much better is a probe on a direction word than on a non-direction word at
     the SAME loudness? A gap that vanishes once loudness is matched means loudness, not the
     word, was doing the work.
  4. THE LABEL EFFECT, CLEAN. Does relabelling to local belief still pay on non-direction
     words? Entry 49's headline result should not depend on verbalized tokens.

Balanced accuracy throughout (mean per-class recall over the classes present; chance .25).
Cells with fewer than MIN_N rows are reported but not plotted -- direction words are 6.2% of
the data and thin out fast in the quiet deciles, which is itself finding 1.

Usage:
    python scripts/analyze_direction_word_isolation.py
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ACTIONS = ["LEFT", "UP", "RIGHT", "DOWN"]
MIN_N = 120  # below this a balanced-accuracy cell is too noisy to plot
DECILES = 10

# The probes worth carrying through every panel: one mlp per selection x label cell.
FOCUS = ["p2_mlp", "base_mlp", "rand_mlp", "randb_mlp", "ll1_mlp", "ll2_mlp"]
NICE = {
    "p2_mlp": "jlens top-20, belief",
    "base_mlp": "jlens top-20, final",
    "rand_mlp": "random, final",
    "randb_mlp": "random, belief",
    "ll1_mlp": "logitlens P1, belief",
    "ll2_mlp": "logitlens P2, belief",
}
COLOR = {
    "p2_mlp": "#1f3f7a",
    "base_mlp": "#a8541c",
    "rand_mlp": "#6a6a76",
    "randb_mlp": "#8a1f5e",
    "ll1_mlp": "#1f7a5a",
    "ll2_mlp": "#4aa88a",
}
LENSES = {"jlens": "per_token_jlens_loudness.csv", "logitlens": "per_token_logitlens_loudness.csv"}


def bal_acc(truth: np.ndarray, pred: np.ndarray) -> float:
    r = [float((pred[truth == a] == a).mean()) for a in ACTIONS if (truth == a).sum()]
    return float(np.mean(r)) if r else float("nan")


def load(path: Path) -> pd.DataFrame:
    # csv.DictReader semantics via pandas: NA handling off, because decoded tokens include
    # the literal string "NA" and pandas would turn it into a missing value.
    df = pd.read_csv(path, keep_default_na=False, na_values=[""], low_memory=False)
    d = df[df["rowset"] == "p2"].copy()
    d["is_dir"] = d["is_direction_token"].astype(str).str.lower().isin(["1", "true"])
    d["decile"] = pd.qcut(d["dir_logmass"].rank(method="first"), DECILES, labels=False)
    return d


def q1_concentration(cuts: dict[str, pd.DataFrame], out: Path) -> pd.DataFrame:
    rows = []
    for lens, d in cuts.items():
        for dec, g in d.groupby("decile"):
            rows.append(
                {
                    "lens": lens,
                    "decile": int(dec),
                    "n": len(g),
                    "n_direction": int(g["is_dir"].sum()),
                    "share_direction": float(g["is_dir"].mean()),
                    "mean_logmass": float(g["dir_logmass"].mean()),
                }
            )
    t = pd.DataFrame(rows)
    t.to_csv(out / "tables" / "q1_direction_share_by_decile.csv", index=False)
    return t


def q2q3_split(cuts: dict[str, pd.DataFrame], out: Path) -> pd.DataFrame:
    rows = []
    for lens, d in cuts.items():
        for probe in FOCUS:
            col = f"{probe}_pred"
            if col not in d.columns:
                continue
            for is_dir, sub in d.groupby("is_dir"):
                for dec, raw in sub.groupby("decile"):
                    cell = raw[raw[col].notna() & raw["label_local"].notna()]
                    if not len(cell):
                        continue
                    rows.append(
                        {
                            "lens": lens,
                            "probe": probe,
                            "subset": "direction" if is_dir else "non-direction",
                            "decile": int(dec),
                            "n": len(cell),
                            "bal_acc_local": bal_acc(cell["label_local"].to_numpy(), cell[col].to_numpy()),
                            "bal_acc_final": bal_acc(cell["label_final"].to_numpy(), cell[col].to_numpy()),
                        }
                    )
    t = pd.DataFrame(rows)
    t.to_csv(out / "tables" / "q2q3_accuracy_by_decile_split.csv", index=False)
    return t


def q4_overall(cuts: dict[str, pd.DataFrame], out: Path) -> pd.DataFrame:
    """Per-probe accuracy on each subset, and the loudness-matched gap in the loudest decile."""
    rows = []
    for lens, d in cuts.items():
        top = d[d["decile"] == DECILES - 1]
        for probe in [c[:-5] for c in d.columns if c.endswith("_pred")]:
            col = f"{probe}_pred"
            rec = {"lens": lens, "probe": probe}
            for tag, frame in (("all", d), ("top_decile", top)):
                for sub_tag, raw in (("dir", frame[frame["is_dir"]]), ("nodir", frame[~frame["is_dir"]])):
                    sub = raw[raw[col].notna() & raw["label_local"].notna()]
                    rec[f"{tag}_{sub_tag}_n"] = len(sub)
                    rec[f"{tag}_{sub_tag}_local"] = (
                        bal_acc(sub["label_local"].to_numpy(), sub[col].to_numpy()) if len(sub) else float("nan")
                    )
                    rec[f"{tag}_{sub_tag}_final"] = (
                        bal_acc(sub["label_final"].to_numpy(), sub[col].to_numpy()) if len(sub) else float("nan")
                    )
            rec["gap_all"] = rec["all_dir_local"] - rec["all_nodir_local"]
            rec["gap_top_decile"] = rec["top_decile_dir_local"] - rec["top_decile_nodir_local"]
            rows.append(rec)
    t = pd.DataFrame(rows)
    t.to_csv(out / "tables" / "q4_probe_summary_by_subset.csv", index=False)
    return t


# --------------------------------------------------------------------------------------
# Figures. Every one is two panels: jlens loudness left, logitlens loudness right.
# --------------------------------------------------------------------------------------
def _lens_axes(title: str, ylabel: str, xlabel: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, lens in zip(axes, LENSES, strict=True):
        ax.set_title(f"{lens} loudness", fontsize=11, color="#8a1f5e")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.grid(alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel(ylabel, fontsize=9)
    fig.suptitle(title, fontsize=13)
    return fig, axes


def fig_concentration(t: pd.DataFrame, out: Path) -> None:
    fig, axes = _lens_axes(
        "Direction words concentrate in the loud tokens — and the two lenses differ",
        "share of tokens that ARE direction words",
        "loudness decile (9 = loudest)",
    )
    for ax, lens in zip(axes, LENSES, strict=True):
        s = t[t["lens"] == lens].sort_values("decile")
        ax.bar(s["decile"], s["share_direction"], color="#8a1f5e", alpha=0.85)
        ax.axhline(0.062, ls="--", lw=1, color="#444", label="all tokens: 6.2%")
        for _, r in s.iterrows():
            ax.text(
                r["decile"], r["share_direction"] + 0.006, f"{r['share_direction']:.0%}", ha="center", fontsize=7.5
            )
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "plots" / "q1_direction_share_by_loudness.png", dpi=150)
    plt.close(fig)


def fig_split(t: pd.DataFrame, out: Path) -> None:
    for subset, fname in (
        ("non-direction", "q2_gradient_nondirection.png"),
        ("direction", "q3_gradient_direction.png"),
    ):
        fig, axes = _lens_axes(
            f"Balanced accuracy vs local belief — {subset} tokens only",
            "balanced accuracy (local belief)",
            "loudness decile (9 = loudest)",
        )
        for ax, lens in zip(axes, LENSES, strict=True):
            for probe in FOCUS:
                s = t[(t["lens"] == lens) & (t["probe"] == probe) & (t["subset"] == subset)].sort_values("decile")
                s = s[s["n"] >= MIN_N]
                if not len(s):
                    continue
                ax.plot(
                    s["decile"],
                    s["bal_acc_local"],
                    "o-",
                    ms=3.5,
                    lw=1.6,
                    color=COLOR[probe],
                    label=NICE.get(probe, probe),
                )
            ax.axhline(0.25, ls=":", lw=1, color="#888")
        axes[1].legend(fontsize=8, loc="upper left")
        fig.tight_layout()
        fig.savefig(out / "plots" / fname, dpi=150)
        plt.close(fig)


def fig_gap(t4: pd.DataFrame, out: Path) -> None:
    """Direction-word advantage overall vs inside the loudest decile (loudness matched)."""
    fig, axes = _lens_axes(
        "Is the direction-word advantage just loudness? (bar = accuracy on direction words − on non-direction words)",
        "balanced-accuracy gap",
        "",
    )
    for ax, lens in zip(axes, LENSES, strict=True):
        s = t4[(t4["lens"] == lens) & (t4["probe"].isin(FOCUS))].set_index("probe").loc[FOCUS]
        x = np.arange(len(FOCUS))
        ax.bar(x - 0.2, s["gap_all"], 0.4, label="all tokens", color="#8a1f5e", alpha=0.85)
        ax.bar(x + 0.2, s["gap_top_decile"], 0.4, label="loudest decile only", color="#4aa88a", alpha=0.9)
        ax.axhline(0, color="#444", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([NICE[p] for p in FOCUS], rotation=30, ha="right", fontsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "plots" / "q3_direction_word_gap.png", dpi=150)
    plt.close(fig)


def fig_label_effect(t4: pd.DataFrame, out: Path) -> None:
    """The entry-49 label effect, recomputed on NON-direction words only."""
    pairs = [("rand_mlp", "randb_mlp", "random selection"), ("base_mlp", "p2_mlp", "jlens top-20")]
    fig, axes = _lens_axes(
        "The label effect on NON-direction words only (final-label probe → belief-label probe)",
        "balanced accuracy vs local belief",
        "",
    )
    for ax, lens in zip(axes, LENSES, strict=True):
        s = t4[t4["lens"] == lens].set_index("probe")
        x = np.arange(len(pairs))
        fin = [s.loc[a, "all_nodir_local"] for a, _, _ in pairs]
        bel = [s.loc[b, "all_nodir_local"] for _, b, _ in pairs]
        ax.bar(x - 0.2, fin, 0.4, label="final-action label", color="#a8541c", alpha=0.85)
        ax.bar(x + 0.2, bel, 0.4, label="local-belief label", color="#1f3f7a", alpha=0.9)
        for i, (f, b) in enumerate(zip(fin, bel, strict=True)):
            ax.text(i + 0.2, b + 0.006, f"+{(b - f) * 100:.1f} pp", ha="center", fontsize=8.5, color="#1f3f7a")
        ax.set_xticks(x)
        ax.set_xticklabels([c for _, _, c in pairs], fontsize=9)
        ax.axhline(0.25, ls=":", lw=1, color="#888")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "plots" / "q4_label_effect_nondirection.png", dpi=150)
    plt.close(fig)


def q5_lens_agreement(cuts: dict[str, pd.DataFrame], out: Path) -> dict:
    """jlens loudness against logitlens loudness, with and without the direction words.

    The two rulers are used interchangeably nowhere in this project precisely because they
    disagree -- but "disagree" has been a top-20 overlap number until now. This is the whole
    joint distribution, and it separates the part of the agreement that is carried by the
    verbalized tokens (which BOTH lenses light up on, so they agree there almost by
    construction) from the agreement on everything else.

    Pearson r on the log-masses AND Spearman rho on the ranks: rho is the one that matters,
    since every use of loudness in this project is a RANKING, not a regression.
    """
    j = cuts["jlens"][["name", "step", "token_id", "dir_logmass", "is_dir"]].rename(
        columns={"dir_logmass": "jlens_logmass"}
    )
    l = cuts["logitlens"][["name", "step", "token_id", "dir_logmass"]].rename(
        columns={"dir_logmass": "logitlens_logmass"}
    )
    # merged on the token key, never on row order -- the two cuts are written by separate runs
    m = j.merge(l, on=["name", "step", "token_id"], how="inner", validate="one_to_one")
    if len(m) != len(j):
        raise SystemExit(f"lens merge lost rows: {len(j)} -> {len(m)}")

    stats = {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharex=True, sharey=True)
    for ax, (tag, sub, title) in zip(
        axes,
        [
            ("all", m, f"all {len(m):,} reasoning tokens"),
            ("nodir", m[~m["is_dir"]], f"direction words removed ({int((~m['is_dir']).sum()):,} tokens)"),
        ],
        strict=True,
    ):
        x = sub["jlens_logmass"].to_numpy()
        y = sub["logitlens_logmass"].to_numpy()
        r = float(np.corrcoef(x, y)[0, 1])
        rho = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
        stats[tag] = {"n": int(len(sub)), "pearson_r": r, "spearman_rho": rho}
        hb = ax.hexbin(x, y, gridsize=70, bins="log", cmap="magma_r", mincnt=1, linewidths=0)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("jlens loudness  (layer-15 log direction mass)", fontsize=9)
        ax.grid(alpha=0.2, linewidth=0.5)
        ax.text(
            0.03,
            0.97,
            f"Pearson r = {r:.3f}\nSpearman ρ = {rho:.3f}\nn = {len(sub):,}",
            transform=ax.transAxes,
            va="top",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.45", "fc": "white", "ec": "#8a1f5e", "alpha": 0.92},
        )
        fig.colorbar(hb, ax=ax, label="tokens per bin (log)")
    axes[0].set_ylabel("logitlens loudness  (layer-15 log direction mass)", fontsize=9)
    fig.suptitle("How much do the two loudness rulers agree, and is it the direction words?", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "plots" / "q5_lens_agreement.png", dpi=150)
    plt.close(fig)

    pd.DataFrame([{"subset": k, **v} for k, v in stats.items()]).to_csv(
        out / "tables" / "q5_lens_agreement.csv", index=False
    )
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--src", type=Path, default=Path("/workspace/reasoning_theatre/probe_loudness_heldout360_16probes")
    )
    ap.add_argument("--out", type=Path, default=None, help="default: <src>/direction_words")
    args = ap.parse_args()
    out = args.out or args.src / "direction_words"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)

    cuts = {}
    for lens, fname in LENSES.items():
        print(f"loading {lens} loudness cut: {fname}", flush=True)
        cuts[lens] = load(args.src / fname)
    n = len(next(iter(cuts.values())))
    share = float(next(iter(cuts.values()))["is_dir"].mean())
    print(f"{n} evaluated tokens, {share:.1%} are direction words", flush=True)

    t1 = q1_concentration(cuts, out)
    t23 = q2q3_split(cuts, out)
    t4 = q4_overall(cuts, out)
    fig_concentration(t1, out)
    fig_split(t23, out)
    fig_gap(t4, out)
    fig_label_effect(t4, out)
    agree = q5_lens_agreement(cuts, out)

    summary = {
        "n_tokens": n,
        "share_direction_all": share,
        "min_n_plotted": MIN_N,
        "concentration": {
            lens: {
                "quietest_decile_share": float(t1[(t1.lens == lens) & (t1.decile == 0)]["share_direction"].iloc[0]),
                "loudest_decile_share": float(
                    t1[(t1.lens == lens) & (t1.decile == DECILES - 1)]["share_direction"].iloc[0]
                ),
            }
            for lens in LENSES
        },
        "gaps": t4[t4["probe"].isin(FOCUS)][["lens", "probe", "gap_all", "gap_top_decile"]].to_dict("records"),
        "lens_agreement": agree,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {out} — 4 tables, 6 figures, summary.json", flush=True)
    for tag, v in agree.items():
        print(
            f"  jlens vs logitlens loudness [{tag:5s}] n={v['n']:,}  "
            f"Pearson r={v['pearson_r']:.3f}  Spearman rho={v['spearman_rho']:.3f}",
            flush=True,
        )
    for lens in LENSES:
        c = summary["concentration"][lens]
        print(
            f"  {lens:9s} direction-word share: quietest {c['quietest_decile_share']:.1%} "
            f"-> loudest {c['loudest_decile_share']:.1%}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
