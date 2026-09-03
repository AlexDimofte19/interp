#!/usr/bin/env python3
"""The Jacobian lens read through the FULL direction vocabulary, split by class.

The earlier Q4 pass scored the lens by the four literal uppercase action words
(`LEFT`/`UP`/`RIGHT`/`DOWN`), which is one token per class and mostly measures how common
that string is. This asks the same question of `direction_tokens_full.json` — 446 tokens
across four disjoint classes, casing/punctuation variants and eight languages — so a token
whose lens top-20 says " gauche", " izquierda" or "_left" counts as LEFT.

Three readouts of the same top-20 block, because they fail differently:

  mass_c   logsumexp of the matched top-20 logprobs — a hit weighted by belief
  count_c  how many of the top 20 are in class c — the original score, belief-free
  rank_c   the best rank class c reaches inside the top 20 (NO_RANK when absent)

and three prior corrections, because the classes are NOT the same size (UP 149, RIGHT 126,
DOWN 119, LEFT 52) and a raw argmax over them mostly recovers that:

  raw        argmax as-is
  bysize     mass − log|class|, count / |class| — the analytic correction
  centered   each column standardised over the corpus — the empirical one, which absorbs
             vocabulary size and anything else that biases a class

The headline evaluation is prior-free: AUC over all (token, class) pairs, which cannot be
gamed by a constant per class. Everything is layer 15, the only layer heldout360_l15 holds.

NOTE on what this CANNOT do. The direction-MASS table beside each analysis CSV is computed
over the whole vocabulary on-device, but its `direction_classes` is "all" — one union column
per layer — so it cannot be split by class. A full-vocabulary per-class readout would need
four mass tables (one `--direction-mass-json` per class) and a fresh gather. Everything here
therefore sees only direction words that reached rank 20, which is 77.7% of tokens.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["UP", "DOWN", "LEFT", "RIGHT"]
NO_MATCH = -40.0
NO_RANK = 21


def zscore(a: np.ndarray) -> np.ndarray:
    """Column-wise standardisation over the corpus."""
    return (a - a.mean(0)) / a.std(0)


def uniformize(a: np.ndarray) -> np.ndarray:
    """Column-wise percentile rank over the corpus, ties averaged.

    Standardising does NOT equalise these columns. A class present in only 26% of top-20 blocks
    sits at the floor for the other 74%, so its z distribution is a spike plus a long tail and its
    argmax rate stays far from a quarter. Mapping each column onto its own uniform [0,1] makes
    the four exactly exchangeable, which is what an argmax across them assumes.
    """
    out = np.empty_like(a, dtype=float)
    for j in range(a.shape[1]):
        out[:, j] = pd.Series(a[:, j]).rank(method="average").to_numpy() / len(a)
    return out


def pooled_auc(score: np.ndarray, truth: np.ndarray, classes: list[str]) -> float:
    """AUC over all (token, class) pairs: does class c score higher where c IS the answer?

    Prior-free by construction — a constant added to one column moves its positives and its
    negatives together, so a class's own base rate cannot buy AUC.
    """
    pos = np.concatenate([score[truth == c, i] for i, c in enumerate(classes)])
    neg = np.concatenate([score[truth != c, i] for i, c in enumerate(classes)])
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def one_vs_rest_auc(score: np.ndarray, truth: np.ndarray, i: int, c: str) -> float:
    """AUC of column `i` for 'the answer is c', on that column alone."""
    pos, neg = score[truth == c, i], score[truth != c, i]
    if not len(pos) or not len(neg):
        return float("nan")
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def argmax_names(score: np.ndarray, classes: list[str]) -> np.ndarray:
    return np.array(classes, dtype=object)[score.argmax(1)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--per-token", type=Path, default=Path("/workspace/reasoning_theatre/probe_vs_rollout/per_token.csv")
    )
    ap.add_argument("--direction-json", type=Path, default=Path("/workspace/jlens/direction_tokens_full.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("/workspace/reasoning_theatre/probe_vs_rollout"))
    args = ap.parse_args()
    out = args.out_dir
    (out / "tables").mkdir(parents=True, exist_ok=True)

    with open(args.direction_json, encoding="utf-8") as f:
        vocab = json.load(f)
    size = np.array([len(vocab[c]) for c in CLASSES], dtype=float)

    df = pd.read_csv(args.per_token, keep_default_na=False, na_values=[""])
    df = df[df["lens_top20_n_direction"].notna()].copy()
    truth_final = df["label_name"].to_numpy()
    truth_roll = df["sent_model_action"].to_numpy()

    mass = df[[f"lens_top20_mass_{c}" for c in CLASSES]].to_numpy(float)
    count = df[[f"lens_top20_count_{c}" for c in CLASSES]].to_numpy(float)
    rank = df[[f"lens_top20_bestrank_{c}" for c in CLASSES]].to_numpy(float)

    # Every readout is oriented "higher is better" so one comparator serves all of them.
    readouts = {
        "mass_raw": mass,
        "mass_bysize": mass - np.log(size),
        "mass_centered": zscore(mass),
        "count_raw": count,
        "count_bysize": count / size,
        "count_centered": zscore(count),
        "rank_raw": -rank,
        "rank_centered": -zscore(rank),
        "mass_uniform": uniformize(mass),
        "count_uniform": uniformize(count),
        "rank_uniform": uniformize(-rank),
    }
    # The four literal uppercase action words, carried through the identical comparator so the
    # vocabulary version can be read against the thing it replaces rather than against prose.
    literal = df[[f"lens_act_logprob_{c}" for c in CLASSES]].to_numpy(float)
    readouts["literal4_raw"] = literal
    readouts["literal4_centered"] = zscore(literal)
    readouts["literal4_uniform"] = uniformize(literal)
    summary: dict = {
        "n_rows": int(len(df)),
        "class_sizes": {c: int(s) for c, s in zip(CLASSES, size, strict=True)},
        "frac_any_direction_in_top20": float((df["lens_top20_n_direction"] > 0).mean()),
        # NaN != "" is True, so the top-1 class must be tested with notna(), not against "".
        "frac_top1_is_direction": float(df["lens_top20_top1_class"].notna().mean()),
        "mean_direction_words_in_top20": float(df["lens_top20_n_direction"].mean()),
        "class_hit_rate": {c: float((df[f"lens_top20_count_{c}"] > 0).mean()) for c in CLASSES},
        "label_marginal": {k: float(v) for k, v in df["label_name"].value_counts(normalize=True).items()},
    }

    # ---- 1. agreement, over the subsets that matter -------------------------------------
    ends = df["is_sentence_end"] == 1
    loud = df["lens_top20_n_direction"] > 0
    top1 = df["lens_top20_top1_class"].notna()
    subsets = {
        "all_tokens": np.ones(len(df), bool),
        "all_tokens_loud": loud.to_numpy(),
        "all_tokens_top1_direction": top1.to_numpy(),
        "sentence_ends": ends.to_numpy(),
        "sentence_ends_loud": (ends & loud).to_numpy(),
        "ends_rollout_disagrees": (ends & (df["sent_model_action"] != df["label_name"])).to_numpy(),
    }
    rows = []
    for sname, m in subsets.items():
        for rname, sc in readouts.items():
            pred = argmax_names(sc[m], CLASSES)
            marg = pd.Series(pred).value_counts(normalize=True)
            rows.append(
                {
                    "subset": sname,
                    "readout": rname,
                    "n": int(m.sum()),
                    # No per-column correction fixes the argmax when the classes fire at very
                    # different rates: with 74% of RIGHT at the floor, both standardising and
                    # uniformizing leave its ties above another class's. Quote the AUC, which
                    # a per-class constant cannot move, and read this column as the caveat.
                    "argmax_max_share": float(marg.max()),
                    "argmax_mode": str(marg.index[0]),
                    "match_rollout": float((pred == truth_roll[m]).mean()),
                    "match_final": float((pred == truth_final[m]).mean()),
                    "auc_rollout": pooled_auc(sc[m], truth_roll[m], CLASSES),
                    "auc_final": pooled_auc(sc[m], truth_final[m], CLASSES),
                }
            )
    agree = pd.DataFrame(rows)
    agree.to_csv(out / "tables" / "q4v_class_agreement.csv", index=False)

    # The lens's own top-1 word, when it is a direction word at all: the most literal readout
    # there is, and the one a reader will assume was tested.
    t1 = df[top1]
    summary["top1_class_readout"] = {
        "n": int(len(t1)),
        "match_rollout": float((t1["lens_top20_top1_class"] == t1["sent_model_action"]).mean()),
        "match_final": float((t1["lens_top20_top1_class"] == t1["label_name"]).mean()),
        "marginal": {k: float(v) for k, v in t1["lens_top20_top1_class"].value_counts(normalize=True).items()},
    }

    # ---- 2. per class, one-vs-rest ------------------------------------------------------
    rows = []
    for rname in ("mass_centered", "count_centered", "rank_centered", "count_uniform", "literal4_uniform"):
        sc = readouts[rname]
        for i, c in enumerate(CLASSES):
            rows.append(
                {
                    "readout": rname,
                    "class": c,
                    "vocab_size": int(size[i]),
                    "n_true": int((truth_final == c).sum()),
                    "hit_rate": float((df[f"lens_top20_count_{c}"] > 0).mean()),
                    "auc_final": one_vs_rest_auc(sc, truth_final, i, c),
                    "auc_rollout": one_vs_rest_auc(sc, truth_roll, i, c),
                }
            )
    pd.DataFrame(rows).to_csv(out / "tables" / "q4v_per_class_auc.csv", index=False)

    best = "count_uniform"
    pred_best = argmax_names(readouts[best], CLASSES)
    pd.crosstab(pd.Series(truth_final, name="true action"), pd.Series(pred_best, name="lens class argmax")).to_csv(
        out / "tables" / "q4v_confusion.csv"
    )

    # ---- 3. does the class mix move through a sentence? ---------------------------------
    # The "lift" is the honest per-class effect: how much higher class c scores at tokens whose
    # answer IS c than at the rest, in sd. A class that is merely common has a lift of zero.
    # Lift and lean are read on the uniform scale for the same reason the argmax is: on the raw
    # logprob scale a class that is usually floored produces lifts that are mostly floor arithmetic.
    z = readouts["count_uniform"]
    idx = np.arange(len(df))
    ci = {c: i for i, c in enumerate(CLASSES)}
    df["z_class_final"] = z[idx, df["label_name"].map(ci).to_numpy()]
    df["z_class_roll"] = z[idx, df["sent_model_action"].map(ci).to_numpy()]
    df["z_class_final_rel"] = df["z_class_final"] - (z.sum(1) - df["z_class_final"]) / 3
    df["z_class_roll_rel"] = df["z_class_roll"] - (z.sum(1) - df["z_class_roll"]) / 3

    for i, c in enumerate(CLASSES):
        df[f"u_{c}"] = z[:, i]

    long = df[df["sentence_len"] >= 5].copy()
    long["bin"] = pd.cut(long["frac_in_sentence"], np.linspace(0, 1, 11), include_lowest=True, labels=range(10))
    parts = []
    for label, sub in (
        ("all", long),
        ("switch_sentences", long[long["prev_model_action"] != long["sent_model_action"]]),
        ("loud", long[long["lens_top20_n_direction"] > 0]),
    ):
        g = sub.groupby("bin", observed=True)
        tb = g[["z_class_roll_rel", "z_class_final_rel", "lens_top20_n_direction"]].mean()
        tb["n"] = g.size()
        for c in CLASSES:
            tb[f"hit_{c}"] = g.apply(
                lambda d, c=c: float((d[f"lens_top20_count_{c}"] > 0).mean()), include_groups=False
            )
            tb[f"lift_{c}"] = g.apply(
                lambda d, c=c: float(
                    d.loc[d["label_name"] == c, f"u_{c}"].mean() - d.loc[d["label_name"] != c, f"u_{c}"].mean()
                ),
                include_groups=False,
            )
        parts.append(tb.reset_index().assign(subset=label))
    pd.concat(parts, ignore_index=True).to_csv(out / "tables" / "q4v_within_sentence.csv", index=False)

    rel = df[df["convinced_sentence_idx"].notna()].copy()
    rel["rel_sentence"] = rel["sentence_idx"] - rel["convinced_sentence_idx"]
    rel = rel[rel["rel_sentence"].between(-4, 4)]
    g = rel.groupby("rel_sentence")
    tb = g[["z_class_roll_rel", "z_class_final_rel", "lens_top20_n_direction", "sent_answer_prob"]].mean()
    tb["n"] = g.size()
    for c in CLASSES:
        tb[f"hit_{c}"] = g.apply(lambda d, c=c: float((d[f"lens_top20_count_{c}"] > 0).mean()), include_groups=False)
    tb.reset_index().to_csv(out / "tables" / "q4v_around_commitment.csv", index=False)

    for c in ("z_class_final_rel", "z_class_roll_rel"):
        x = df[c].to_numpy()
        summary[f"{c}_mean"] = float(x.mean())
        summary[f"{c}_t"] = float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))

    with open(out / "summary_direction_classes.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\ntables -> {out / 'tables'}/q4v_*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
