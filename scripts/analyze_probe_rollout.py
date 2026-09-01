#!/usr/bin/env python3
"""Answer the four reasoning-theatre questions from `per_token.csv`.

  Q2  At each sentence end, does the probe's prediction match what the model WOULD infer if
      reasoning stopped there (the rollout), or only the final action it eventually takes?
      The two hypotheses are separable exactly on the rows where they disagree.
  Q3  How does the probe's prediction move THROUGH a sentence — does it already hold the
      sentence's end-state at the first token, or does it flip partway?
  Q4  Does the layer-15 Jacobian lens see the commitment boundary at all: does its own
      directional preference (the literal action words, and the top-20 direction vocabulary)
      track the rollout the way the probe does?

Every table is a group-by over the join; nothing here re-reads an activation.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ACTIONS = ["LEFT", "UP", "RIGHT", "DOWN"]
CLASSES = ["UP", "DOWN", "LEFT", "RIGHT"]
# Entry 38(b): the strongest arms on this held-out set, plus a control and a pooled-layer arm.
# The single arm every figure and every quoted number in the write-up uses. Kept identical to
# plot_probe_rollout.py's HEADLINE: the two drifted apart once and the report mixed two arms.
HEADLINE = [
    "next_action_l15.jlens_topall_mlp",
    "next_action_l15.jlens_topall_lr",
    "next_action_mass_l15.jlens_topall_mlp",
    "next_action_seeds.logitlens_l15_seed42_mlp",
    "next_action_mass_l15.random_topall_mlp",
    "next_action.jlens_topall_mlp",
]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — sane at the small n the per-bin tables reach."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def zscore(a: np.ndarray) -> np.ndarray:
    """Column-wise standardisation over the corpus, so the four columns become comparable."""
    return (a - a.mean(0)) / a.std(0)


def cluster_bootstrap(df: pd.DataFrame, arm: str, n_boot: int = 2000, seed: int = 0) -> dict:
    """Trajectory-clustered bootstrap of (match_rollout - match_final) for one arm.

    Every row of a trajectory carries that trajectory's single label, so an i.i.d. row
    bootstrap would understate the spread by a lot. Resample trajectory names instead.
    """
    pred = df[f"{arm}_pred"].map(lambda i: ACTIONS[int(i)])
    hit_r = (pred == df["sent_model_action"]).to_numpy()
    hit_f = (pred == df["label_name"]).to_numpy()
    names = df["name"].to_numpy()
    uniq, inv = np.unique(names, return_inverse=True)
    by = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        pick = np.concatenate([by[i] for i in rng.integers(0, len(uniq), len(uniq))])
        diffs[b] = hit_r[pick].mean() - hit_f[pick].mean()
    return {
        "n": int(len(df)),
        "n_trajectories": int(len(uniq)),
        "match_rollout": float(hit_r.mean()),
        "match_final": float(hit_f.mean()),
        "diff": float(hit_r.mean() - hit_f.mean()),
        "diff_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
    }


def agreement_table(df: pd.DataFrame, arms: list[str]) -> pd.DataFrame:
    """Per-arm agreement with the rollout vs. with the final action, on `df`."""
    rows = []
    for arm in arms:
        pred = df[f"{arm}_pred"].map(lambda i: ACTIONS[int(i)])
        n = len(df)
        k_roll = int((pred == df["sent_model_action"]).sum())
        k_lab = int((pred == df["label_name"]).sum())
        lo, hi = wilson(k_roll, n)
        rows.append(
            {
                "arm": arm,
                "n": n,
                "match_rollout": k_roll / n if n else np.nan,
                "match_rollout_lo": lo,
                "match_rollout_hi": hi,
                "match_final": k_lab / n if n else np.nan,
                "mean_p_true": float(df[f"{arm}_p_true"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--per-token", type=Path, default=Path("/workspace/reasoning_theatre/probe_vs_rollout/per_token.csv")
    )
    ap.add_argument("--out-dir", type=Path, default=Path("/workspace/reasoning_theatre/probe_vs_rollout"))
    ap.add_argument(
        "--headline-extra",
        default="",
        help="Comma-separated arms to APPEND to HEADLINE for the per-arm tables (q3, q4). "
        "Empty (the default) leaves every output byte-identical to the entry-39/41 run -- "
        "HEADLINE itself must not be edited, see the note on it. Use this to add an arm to "
        "a CLONE of the report rather than to change the original.",
    )
    args = ap.parse_args()
    out = args.out_dir
    (out / "tables").mkdir(parents=True, exist_ok=True)

    # keep_default_na off: decoded tokens include the literal string "NA".
    df = pd.read_csv(args.per_token, keep_default_na=False, na_values=[""])
    arms = sorted(c[:-5] for c in df.columns if c.endswith("_pred"))
    extra = [a for a in args.headline_extra.split(",") if a]
    missing = [a for a in extra if f"{a}_pred" not in df.columns]
    if missing:
        raise SystemExit(f"--headline-extra arms not in {args.per_token}: {missing}")
    headline = HEADLINE + [a for a in extra if a not in HEADLINE]
    print(f"{len(df)} rows, {df['name'].nunique()} trajectories, {len(arms)} arms")
    if extra:
        print(f"  headline extended by {extra}")

    summary: dict = {"n_rows": int(len(df)), "n_trajectories": int(df["name"].nunique()), "arms": arms}

    ends = df[df["is_sentence_end"] == 1].copy()
    # sentence 0 owns no reasoning tokens, so every row here is a real sentence end.
    summary["n_sentence_end_rows"] = int(len(ends))
    summary["rollout_matches_final_at_end"] = float((ends["sent_model_action"] == ends["label_name"]).mean())

    # ---------------- Q2: probe vs rollout at sentence ends ----------------
    t_all = agreement_table(ends, arms).assign(subset="all_sentence_ends")
    disagree = ends[ends["sent_model_action"] != ends["label_name"]]
    t_dis = agreement_table(disagree, arms).assign(subset="rollout_disagrees_with_final")
    agree = ends[ends["sent_model_action"] == ends["label_name"]]
    t_agr = agreement_table(agree, arms).assign(subset="rollout_agrees_with_final")
    q2 = pd.concat([t_all, t_dis, t_agr], ignore_index=True)
    q2.to_csv(out / "tables" / "q2_sentence_end_agreement.csv", index=False)

    # Before vs after the model has committed (convinced_sentence_idx).
    rows = []
    for arm in arms:
        for label, sub in (
            ("before_convinced", ends[ends["is_after_convinced"] == 0]),
            ("after_convinced", ends[ends["is_after_convinced"] == 1]),
        ):
            if not len(sub):
                continue
            pred = sub[f"{arm}_pred"].map(lambda i: ACTIONS[int(i)])
            rows.append(
                {
                    "arm": arm,
                    "regime": label,
                    "n": len(sub),
                    "match_rollout": float((pred == sub["sent_model_action"]).mean()),
                    "match_final": float((pred == sub["label_name"]).mean()),
                    "rollout_correct": float(sub["sent_correct"].mean()),
                    "mean_p_true": float(sub[f"{arm}_p_true"].mean()),
                }
            )
    pd.DataFrame(rows).to_csv(out / "tables" / "q2_by_commitment_regime.csv", index=False)

    # ---------------- Q3: evolution through a sentence ----------------
    # Only sentences long enough to have an inside; `switch` sentences are the ones where the
    # model's answer actually changes, so "does the probe flip early or late" is answerable.
    long = df[df["sentence_len"] >= 5].copy()
    long["bin"] = pd.cut(long["frac_in_sentence"], bins=np.linspace(0, 1, 11), include_lowest=True, labels=range(10))
    switch = long[long["prev_model_action"] != long["sent_model_action"]]
    hold = long[long["prev_model_action"] == long["sent_model_action"]]
    rows = []
    for arm in headline + [a for a in arms if a not in headline]:
        for label, sub in (("all", long), ("switch_sentences", switch), ("hold_sentences", hold)):
            pred_all = sub[f"{arm}_pred"].map(lambda i: ACTIONS[int(i)])
            g = sub.assign(_p=pred_all).groupby("bin", observed=True)
            for b, gg in g:
                rows.append(
                    {
                        "arm": arm,
                        "subset": label,
                        "bin": int(b),
                        "frac_mid": (int(b) + 0.5) / 10,
                        "n": len(gg),
                        "match_this_sentence_end": float((gg["_p"] == gg["sent_model_action"]).mean()),
                        "match_prev_sentence_end": float((gg["_p"] == gg["prev_model_action"]).mean()),
                        "match_final": float((gg["_p"] == gg["label_name"]).mean()),
                        "mean_p_true": float(gg[f"{arm}_p_true"].mean()),
                    }
                )
    pd.DataFrame(rows).to_csv(out / "tables" / "q3_within_sentence.csv", index=False)

    # Token offset from the sentence end, -12..0, which does not squash short sentences.
    df["offset_from_end"] = df["pos_in_sentence"] - (df["sentence_len"] - 1)
    off = df[df["offset_from_end"] >= -12]
    rows = []
    for arm in headline:
        for label, sub in (
            ("all", off),
            ("switch_sentences", off[off["prev_model_action"] != off["sent_model_action"]]),
        ):
            pred_all = sub[f"{arm}_pred"].map(lambda i: ACTIONS[int(i)])
            for o, gg in sub.assign(_p=pred_all).groupby("offset_from_end"):
                rows.append(
                    {
                        "arm": arm,
                        "subset": label,
                        "offset_from_end": int(o),
                        "n": len(gg),
                        "match_this_sentence_end": float((gg["_p"] == gg["sent_model_action"]).mean()),
                        "match_prev_sentence_end": float((gg["_p"] == gg["prev_model_action"]).mean()),
                        "match_final": float((gg["_p"] == gg["label_name"]).mean()),
                        "mean_p_true": float(gg[f"{arm}_p_true"].mean()),
                    }
                )
    pd.DataFrame(rows).to_csv(out / "tables" / "q3_offset_from_end.csv", index=False)

    # ---------------- Q4: what the lens itself sees ----------------
    lens = df[df["lens_act_argmax"].notna()].copy()
    summary["n_lens_rows"] = int(len(lens))
    lens_ends = lens[lens["is_sentence_end"] == 1]
    q4_rows = [
        {
            "readout": "lens_act_argmax",
            "subset": s,
            "n": len(sub),
            "match_rollout": float((sub["lens_act_argmax"] == sub["sent_model_action"]).mean()),
            "match_final": float((sub["lens_act_argmax"] == sub["label_name"]).mean()),
        }
        for s, sub in (
            ("sentence_ends", lens_ends),
            ("sentence_ends_rollout_disagrees", lens_ends[lens_ends["sent_model_action"] != lens_ends["label_name"]]),
            ("all_tokens", lens),
        )
    ]
    top20 = lens[lens["lens_top20_argmax"].notna()]
    top20_ends = top20[top20["is_sentence_end"] == 1]
    q4_rows += [
        {
            "readout": "lens_top20_argmax",
            "subset": s,
            "n": len(sub),
            "match_rollout": float((sub["lens_top20_argmax"] == sub["sent_model_action"]).mean()),
            "match_final": float((sub["lens_top20_argmax"] == sub["label_name"]).mean()),
        }
        for s, sub in (
            ("sentence_ends", top20_ends),
            (
                "sentence_ends_rollout_disagrees",
                top20_ends[top20_ends["sent_model_action"] != top20_ends["label_name"]],
            ),
            ("all_tokens", top20),
        )
    ]
    pd.DataFrame(q4_rows).to_csv(out / "tables" / "q4_lens_agreement.csv", index=False)

    # Lens direction through a sentence, same bins as Q3.
    lens_long = lens[lens["sentence_len"] >= 5].copy()
    lens_long["bin"] = pd.cut(
        lens_long["frac_in_sentence"], bins=np.linspace(0, 1, 11), include_lowest=True, labels=range(10)
    )
    rows = []
    for label, sub in (
        ("all", lens_long),
        ("switch_sentences", lens_long[lens_long["prev_model_action"] != lens_long["sent_model_action"]]),
    ):
        for b, gg in sub.groupby("bin", observed=True):
            has20 = gg[gg["lens_top20_argmax"].notna()]
            rows.append(
                {
                    "subset": label,
                    "bin": int(b),
                    "frac_mid": (int(b) + 0.5) / 10,
                    "n": len(gg),
                    "act_match_this": float((gg["lens_act_argmax"] == gg["sent_model_action"]).mean()),
                    "act_match_prev": float((gg["lens_act_argmax"] == gg["prev_model_action"]).mean()),
                    "act_match_final": float((gg["lens_act_argmax"] == gg["label_name"]).mean()),
                    "act_margin": float(gg["lens_act_margin"].mean()),
                    "act_logprob_true": float(
                        gg.apply(lambda r: r[f"lens_act_logprob_{r['label_name']}"], axis=1).mean()
                    ),
                    "n_top20": len(has20),
                    "top20_match_this": float((has20["lens_top20_argmax"] == has20["sent_model_action"]).mean())
                    if len(has20)
                    else np.nan,
                    "top20_match_final": float((has20["lens_top20_argmax"] == has20["label_name"]).mean())
                    if len(has20)
                    else np.nan,
                    "top20_frac_any_direction": float((gg["lens_top20_n_direction"] > 0).mean()),
                    "mean_n_direction": float(gg["lens_top20_n_direction"].mean()),
                    "jlens_mass_L15": float(gg["jlens_mass_L15"].mean()),
                }
            )
    pd.DataFrame(rows).to_csv(out / "tables" / "q4_within_sentence.csv", index=False)

    # Everything as a function of sentence index relative to the commitment boundary.
    rel = df[df["convinced_sentence_idx"].notna()].copy()
    rel["rel_sentence"] = rel["sentence_idx"] - rel["convinced_sentence_idx"]
    rel = rel[rel["rel_sentence"].between(-4, 4)]
    rows = []
    for r, gg in rel.groupby("rel_sentence"):
        rec = {
            "rel_sentence": int(r),
            "n": len(gg),
            "rollout_correct": float(gg["sent_correct"].mean()),
            "answer_prob": float(gg["sent_answer_prob"].mean()),
            "jlens_mass_L15": float(gg["jlens_mass_L15"].mean()),
        }
        gl = gg[gg["lens_act_argmax"].notna()]
        rec["act_match_final"] = float((gl["lens_act_argmax"] == gl["label_name"]).mean()) if len(gl) else np.nan
        rec["act_match_rollout"] = (
            float((gl["lens_act_argmax"] == gl["sent_model_action"]).mean()) if len(gl) else np.nan
        )
        rec["act_logprob_true"] = (
            float(gl.apply(lambda x: x[f"lens_act_logprob_{x['label_name']}"], axis=1).mean()) if len(gl) else np.nan
        )
        for arm in headline:
            p = gg[f"{arm}_pred"].map(lambda i: ACTIONS[int(i)])
            rec[f"{arm}|match_final"] = float((p == gg["label_name"]).mean())
            rec[f"{arm}|match_rollout"] = float((p == gg["sent_model_action"]).mean())
            rec[f"{arm}|p_true"] = float(gg[f"{arm}_p_true"].mean())
        rows.append(rec)
    pd.DataFrame(rows).to_csv(out / "tables" / "q4_around_commitment.csv", index=False)

    write_q4_centered(df, ends, disagree, out, summary)

    # ---------------- clustered bootstrap on the headline claim ----------------
    # Rows inside one trajectory share a label and a rollout, so resample trajectories.
    summary["bootstrap"] = {}
    for arm in headline[:3] + list(headline[len(HEADLINE) :]):
        summary["bootstrap"][arm] = cluster_bootstrap(disagree, arm)

    # Confusion of the headline probe against the rollout, at sentence ends.
    arm = headline[0]
    p = ends[f"{arm}_pred"].map(lambda i: ACTIONS[int(i)])
    pd.crosstab(ends["sent_model_action"], p, rownames=["rollout"], colnames=["probe"]).to_csv(
        out / "tables" / "q2_confusion_headline.csv"
    )

    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2)[:800])
    print(f"tables -> {out / 'tables'}")
    return 0


def write_q4_centered(df: pd.DataFrame, ends: pd.DataFrame, disagree: pd.DataFrame, out: Path, summary: dict) -> None:
    """Q4 with the lens's word-frequency prior removed, and the shape tables that follow."""
    # ---------------- Q4b: the lens with its word-frequency prior removed ----------------
    # Raw argmax over the four action-word logprobs is degenerate -- "DOWN" and "UP" are simply
    # commoner strings -- so it measures the vocabulary, not the token. Centering each column on
    # its corpus mean/sd asks the only fair question: relative to how much this lens usually
    # likes "UP", does it like "UP" MORE here?
    za = zscore(df[[f"lens_act_logprob_{a}" for a in ACTIONS]].to_numpy(float))
    zm = zscore(df[[f"lens_top20_mass_{c}" for c in CLASSES]].to_numpy(float))
    df["lens_act_z_argmax"] = [ACTIONS[i] for i in za.argmax(1)]
    df["lens_top20_z_argmax"] = [CLASSES[i] for i in zm.argmax(1)]
    idx = np.arange(len(df))
    for tag, arr, cols in (("act", za, ACTIONS), ("mass", zm, CLASSES)):
        for who, col in (("true", "label_name"), ("roll", "sent_model_action")):
            own = arr[idx, df[col].map({c: i for i, c in enumerate(cols)}).to_numpy()]
            # own minus the mean of the other three: a within-token contrast, prior-free.
            df[f"z_{tag}_{who}_rel"] = own - (arr.sum(1) - own) / 3

    rows = []
    for name, part in (("all_tokens", df), ("sentence_ends", ends), ("ends_rollout_disagrees", disagree)):
        # ends/disagree were sliced before the z-columns existed; re-select to pick them up.
        sub = df.loc[part.index]
        for col in ("lens_act_argmax", "lens_act_z_argmax", "lens_top20_argmax", "lens_top20_z_argmax"):
            s2 = sub[sub[col].notna()]
            rows.append(
                {
                    "subset": name,
                    "readout": col,
                    "n": len(s2),
                    "match_rollout": float((s2[col] == s2["sent_model_action"]).mean()),
                    "match_final": float((s2[col] == s2["label_name"]).mean()),
                }
            )
    pd.DataFrame(rows).to_csv(out / "tables" / "q4_centered_agreement.csv", index=False)

    # Prior-free continuous version: AUC over all (token, action) pairs.
    rows = []
    for tag, arr, cols in (("action_word", za, ACTIONS), ("top20_class_mass", zm, CLASSES)):
        for tgt in ("sent_model_action", "label_name"):
            tv = df[tgt].to_numpy()
            pos = np.concatenate([arr[tv == a, i] for i, a in enumerate(cols)])
            neg = np.concatenate([arr[tv != a, i] for i, a in enumerate(cols)])
            r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
            auc = (r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
            rows.append({"readout": tag, "target": tgt, "n_pos": len(pos), "auc": float(auc)})
    pd.DataFrame(rows).to_csv(out / "tables" / "q4_lens_auc.csv", index=False)

    # The shape question the user asked: does the direction MIX move through a sentence?
    rel_cols = [
        "z_act_true_rel",
        "z_act_roll_rel",
        "z_mass_true_rel",
        "z_mass_roll_rel",
        "lens_top20_n_direction",
        "jlens_mass_L15",
    ]
    long2 = df[df["sentence_len"] >= 5].copy()
    long2["bin"] = pd.cut(long2["frac_in_sentence"], np.linspace(0, 1, 11), include_lowest=True, labels=range(10))
    parts = []
    for label, sub in (
        ("all", long2),
        ("switch_sentences", long2[long2["prev_model_action"] != long2["sent_model_action"]]),
    ):
        g = sub.groupby("bin", observed=True)
        tb = g[rel_cols].mean()
        tb["n"] = g.size()
        for c in CLASSES:
            tb[f"share_{c}"] = g.apply(lambda d, c=c: float((d[f"lens_top20_mass_{c}"] > -40).mean()))
        parts.append(tb.reset_index().assign(subset=label))
    pd.concat(parts, ignore_index=True).to_csv(out / "tables" / "q4_lens_shape.csv", index=False)

    rel2 = df[df["convinced_sentence_idx"].notna()].copy()
    rel2["rel_sentence"] = rel2["sentence_idx"] - rel2["convinced_sentence_idx"]
    rel2 = rel2[rel2["rel_sentence"].between(-4, 4)]
    g = rel2.groupby("rel_sentence")
    tb = g[rel_cols + ["sent_answer_prob", "sent_correct"]].mean()
    tb["n"] = g.size()
    tb.reset_index().to_csv(out / "tables" / "q4_lens_around_commitment.csv", index=False)

    for c in ("z_act_true_rel", "z_mass_true_rel", "z_act_roll_rel", "z_mass_roll_rel"):
        x = df[c].to_numpy()
        summary[f"{c}_mean"] = float(x.mean())
        summary[f"{c}_t"] = float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


if __name__ == "__main__":
    sys.exit(main())
