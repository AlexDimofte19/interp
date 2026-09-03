#!/usr/bin/env python3
"""Figures for the probe-vs-rollout comparison. Reads only `tables/` written by
`analyze_probe_rollout.py`, so re-styling never re-runs the group-bys."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Must match analyze_probe_rollout.py's HEADLINE[0] — see the note there.
HEADLINE = "next_action_l15.jlens_topall_mlp"
C = {"this": "#2b6cb0", "prev": "#c05621", "final": "#2f855a", "lens": "#805ad5", "grey": "#718096"}


def fig(path: Path, f):
    f.tight_layout()
    f.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(f)
    print(f"  {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("/workspace/reasoning_theatre/probe_vs_rollout"))
    args = ap.parse_args()
    t = args.dir / "tables"
    p = args.dir / "plots"
    p.mkdir(parents=True, exist_ok=True)

    # 1. Q2 -- rollout vs final, per arm, on the disagreement rows.
    q2 = pd.read_csv(t / "q2_sentence_end_agreement.csv")
    d = q2[q2["subset"] == "rollout_disagrees_with_final"].sort_values("match_final")
    f, ax = plt.subplots(figsize=(9, 0.28 * len(d) + 2))
    y = range(len(d))
    ax.barh(
        [i - 0.2 for i in y],
        d["match_rollout"],
        height=0.4,
        color=C["this"],
        label="matches the rollout (what the model would say here)",
    )
    ax.barh([i + 0.2 for i in y], d["match_final"], height=0.4, color=C["final"], label="matches the final action")
    ax.set_yticks(list(y))
    ax.set_yticklabels(d["arm"], fontsize=7)
    ax.axvline(0.25, color=C["grey"], ls=":", lw=1, label="chance (4 actions)")
    ax.set_xlabel("agreement")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(
        f"Sentence ends where the truncated rollout ≠ the final action (n={int(d['n'].iloc[0])})", fontsize=10
    )
    fig(p / "q2_disagreement_by_arm.png", f)

    # 2. Q2 -- all sentence ends, per arm.
    a = q2[q2["subset"] == "all_sentence_ends"].sort_values("match_rollout")
    f, ax = plt.subplots(figsize=(9, 0.28 * len(a) + 2))
    y = range(len(a))
    ax.barh([i - 0.2 for i in y], a["match_rollout"], height=0.4, color=C["this"], label="matches the rollout")
    ax.barh([i + 0.2 for i in y], a["match_final"], height=0.4, color=C["final"], label="matches the final action")
    ax.set_yticks(list(y))
    ax.set_yticklabels(a["arm"], fontsize=7)
    ax.axvline(0.25, color=C["grey"], ls=":", lw=1)
    ax.set_xlabel("agreement")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(f"All sentence ends (n={int(a['n'].iloc[0])})", fontsize=10)
    fig(p / "q2_all_ends_by_arm.png", f)

    # 3. Q3 -- within-sentence trajectory of the headline probe.
    q3 = pd.read_csv(t / "q3_within_sentence.csv")
    f, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    panels = (("all", "all sentences"), ("switch_sentences", "sentences where the model's answer CHANGES"))
    for ax, (sub, title) in zip(axes, panels, strict=True):
        g = q3[(q3["arm"] == HEADLINE) & (q3["subset"] == sub)]
        ax.plot(
            g["frac_mid"],
            g["match_this_sentence_end"],
            "-o",
            color=C["this"],
            ms=4,
            label="= this sentence's end-state",
        )
        ax.plot(
            g["frac_mid"],
            g["match_prev_sentence_end"],
            "-o",
            color=C["prev"],
            ms=4,
            label="= previous sentence's end-state",
        )
        ax.plot(g["frac_mid"], g["match_final"], "-o", color=C["final"], ms=4, label="= final action")
        ax.axhline(0.25, color=C["grey"], ls=":", lw=1)
        ax.set_xlabel("position within the sentence")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("probe agreement")
    axes[0].legend(fontsize=8)
    f.suptitle(f"How {HEADLINE} moves through a sentence", fontsize=11)
    fig(p / "q3_within_sentence.png", f)

    # 4. Q3 -- token offset from the sentence end.
    q3o = pd.read_csv(t / "q3_offset_from_end.csv")
    f, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, sub, title in zip(axes, ("all", "switch_sentences"), ("all sentences", "switch sentences"), strict=False):
        g = q3o[(q3o["arm"] == HEADLINE) & (q3o["subset"] == sub)].sort_values("offset_from_end")
        ax.plot(
            g["offset_from_end"],
            g["match_this_sentence_end"],
            "-o",
            color=C["this"],
            ms=4,
            label="= this sentence's end-state",
        )
        ax.plot(
            g["offset_from_end"],
            g["match_prev_sentence_end"],
            "-o",
            color=C["prev"],
            ms=4,
            label="= previous end-state",
        )
        ax.plot(g["offset_from_end"], g["match_final"], "-o", color=C["final"], ms=4, label="= final action")
        ax.axhline(0.25, color=C["grey"], ls=":", lw=1)
        ax.set_xlabel("tokens before the sentence end (0 = the end token)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("probe agreement")
    axes[0].legend(fontsize=8)
    fig(p / "q3_offset_from_end.png", f)

    # 5. Q4 -- lens direction through a sentence.
    q4 = pd.read_csv(t / "q4_within_sentence.csv")
    f, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, sub, title in zip(
        axes[:2], ("all", "switch_sentences"), ("all sentences", "switch sentences"), strict=False
    ):
        g = q4[q4["subset"] == sub]
        ax.plot(g["frac_mid"], g["act_match_this"], "-o", color=C["this"], ms=4, label="lens argmax = this end-state")
        ax.plot(g["frac_mid"], g["act_match_prev"], "-o", color=C["prev"], ms=4, label="= previous end-state")
        ax.plot(g["frac_mid"], g["act_match_final"], "-o", color=C["final"], ms=4, label="= final action")
        ax.plot(
            g["frac_mid"], g["top20_match_final"], "--s", color=C["lens"], ms=4, label="top-20 class = final action"
        )
        ax.axhline(0.25, color=C["grey"], ls=":", lw=1)
        ax.set_xlabel("position within the sentence")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("lens agreement")
    axes[0].legend(fontsize=7)
    g = q4[q4["subset"] == "all"]
    ax = axes[2]
    ax.plot(g["frac_mid"], g["jlens_mass_L15"], "-o", color=C["lens"], ms=4)
    ax.set_xlabel("position within the sentence")
    ax.set_ylabel("mean jlens direction mass @L15")
    ax.set_title("how direction-loaded the token is", fontsize=10)
    ax.grid(alpha=0.25)
    f.suptitle("Layer-15 Jacobian lens through a reasoning sentence", fontsize=11)
    fig(p / "q4_lens_within_sentence.png", f)

    # 6. Q4 -- around the commitment boundary.
    q4c = pd.read_csv(t / "q4_around_commitment.csv").sort_values("rel_sentence")
    f, axes = plt.subplots(1, 3, figsize=(14, 4))
    ax = axes[0]
    ax.plot(q4c["rel_sentence"], q4c["rollout_correct"], "-o", color="black", ms=4, label="rollout is correct")
    ax.plot(q4c["rel_sentence"], q4c["answer_prob"], "-o", color=C["grey"], ms=4, label="rollout answer prob")
    ax.axvline(0, color="red", ls="--", lw=1)
    ax.set_title("the commitment boundary itself", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax = axes[1]
    ax.plot(
        q4c["rel_sentence"], q4c[f"{HEADLINE}|match_final"], "-o", color=C["final"], ms=4, label="probe = final action"
    )
    ax.plot(
        q4c["rel_sentence"],
        q4c[f"{HEADLINE}|match_rollout"],
        "-o",
        color=C["this"],
        ms=4,
        label="probe = rollout here",
    )
    ax.axvline(0, color="red", ls="--", lw=1)
    ax.axhline(0.25, color=C["grey"], ls=":", lw=1)
    ax.set_title(f"probe ({HEADLINE.split('.')[-1]})", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax = axes[2]
    ax.plot(
        q4c["rel_sentence"], q4c["act_match_final"], "-o", color=C["lens"], ms=4, label="lens argmax = final action"
    )
    ax.plot(
        q4c["rel_sentence"],
        q4c["act_logprob_true"] / 10,
        "-s",
        color=C["grey"],
        ms=4,
        label="lens logprob(true action) / 10",
    )
    ax.axvline(0, color="red", ls="--", lw=1)
    ax.set_title("jacobian lens", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    for ax in axes:
        ax.set_xlabel("sentence index − convinced index")
    fig(p / "q4_around_commitment.png", f)

    # 7. Q4 -- the centered lens, which is the only fair version of "which way does it lean".
    sh = pd.read_csv(t / "q4_lens_shape.csv")
    f, axes = plt.subplots(1, 3, figsize=(14, 4))
    panels = (("all", "all sentences"), ("switch_sentences", "sentences where the answer CHANGES"))
    for ax, (sub, title) in zip(axes[:2], panels, strict=True):
        g = sh[sh["subset"] == sub]
        ax.plot(
            g["bin"] / 10 + 0.05,
            g["z_act_roll_rel"],
            "-o",
            color=C["this"],
            ms=4,
            label="lean toward THIS sentence's answer",
        )
        ax.plot(
            g["bin"] / 10 + 0.05,
            g["z_act_true_rel"],
            "-o",
            color=C["final"],
            ms=4,
            label="lean toward the FINAL action",
        )
        ax.axhline(0, color=C["grey"], lw=1)
        ax.set_xlabel("position within the sentence")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("centered action-word score, own − others (sd)")
    axes[0].legend(fontsize=8)
    g = sh[sh["subset"] == "all"]
    ax = axes[2]
    ax.plot(g["bin"] / 10 + 0.05, g["lens_top20_n_direction"], "-o", color=C["lens"], ms=4)
    ax.set_xlabel("position within the sentence")
    ax.set_ylabel("direction words in the lens top-20")
    ax.set_title("how loud the lens is", fontsize=10)
    ax.grid(alpha=0.25)
    f.suptitle("The layer-15 lens with the word-frequency prior removed", fontsize=11)
    fig(p / "q4_lens_centered.png", f)

    # 8. Q4 -- lens loudness around the commitment boundary.
    lc = pd.read_csv(t / "q4_lens_around_commitment.csv").sort_values("rel_sentence")
    f, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.plot(lc["rel_sentence"], lc["lens_top20_n_direction"], "-o", color=C["lens"], ms=5)
    ax.axvline(0, color="red", ls="--", lw=1)
    ax.set_ylabel("direction words in the lens top-20")
    ax.set_title("lens loudness", fontsize=10)
    ax = axes[1]
    ax.plot(lc["rel_sentence"], lc["z_act_roll_rel"], "-o", color=C["this"], ms=5, label="lean toward the answer here")
    ax.plot(
        lc["rel_sentence"], lc["z_act_true_rel"], "-o", color=C["final"], ms=5, label="lean toward the final action"
    )
    ax.axvline(0, color="red", ls="--", lw=1)
    ax.axhline(0, color=C["grey"], lw=1)
    ax.set_title("lens direction lean", fontsize=10)
    ax.legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("sentence index − convinced index")
        ax.grid(alpha=0.25)
    f.suptitle("Does the Jacobian lens see the commitment boundary?", fontsize=11)
    fig(p / "q4_lens_around_commitment.png", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
