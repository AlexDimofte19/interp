#!/usr/bin/env python3
"""Numbers behind the loudness figures: within-sentence decay, and whether the commitment
boundary moves the level at all.

Bootstraps resample TRAJECTORY NAMES, not rows -- tokens inside a trajectory share a
sentence structure and a boundary.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from telos_interp.loudness_analysis import columns as _cols
from telos_interp.loudness_analysis import provenance as _prov

COMPLEXITIES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
VAL = "loudness_prob"


def boot_mean(g: pd.DataFrame, value: str, n: int, rng) -> tuple[float, float, float]:
    per = g.groupby("name")[value].agg(["sum", "count"])
    s, c = per["sum"].to_numpy(), per["count"].to_numpy()
    if len(s) < 2:
        return float(s.sum() / c.sum()) if c.sum() else np.nan, np.nan, np.nan
    idx = rng.integers(0, len(s), size=(n, len(s)))
    draws = s[idx].sum(1) / c[idx].sum(1)
    return float(s.sum() / c.sum()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def paired_open(df: pd.DataFrame, a: int, b: int, value: str, n: int, rng) -> dict:
    """Within-trajectory difference of the OPENING decile between two rel-sentence classes."""
    first = df[df["sentence_frac"] < 0.1]
    pa = first[first["rel_sentence"] == a].groupby("name")[value].mean()
    pb = first[first["rel_sentence"] == b].groupby("name")[value].mean()
    both = pa.index.intersection(pb.index)
    d = (pb.loc[both] - pa.loc[both]).to_numpy()
    if len(d) < 2:
        return {}
    idx = rng.integers(0, len(d), size=(n, len(d)))
    draws = d[idx].mean(1)
    return {
        "n_traj": int(len(d)),
        "mean_diff": float(d.mean()),
        "lo": float(np.percentile(draws, 2.5)),
        "hi": float(np.percentile(draws, 97.5)),
    }


def lens_agreement(path, args) -> dict:
    """Correlation between two lenses' loudness on the same rows, or {} if only one is present.

    At layer 15 the two lenses' top-20 sets overlap only about half, so this is what says
    whether a result binned by one ruler would survive the other.
    """
    header = pd.read_csv(path, nrows=0, keep_default_na=False).columns.tolist()
    try:
        a = _cols.resolve(header, args.lens, args.signal_name, args.layer)
        b = _cols.resolve(header, args.compare_lens, args.signal_name, args.layer)
    except KeyError:
        return {}
    if a == b:
        return {}

    member = _cols.membership_column(args.signal_name)
    want = [c for c in (a, b, member, "is_direction_token") if c in header]
    d = pd.read_csv(path, usecols=want, keep_default_na=False, na_values=[""])
    flag = member if member in d.columns else ("is_direction_token" if "is_direction_token" in d.columns else None)

    def pair(sub: pd.DataFrame) -> dict:
        return {
            "n": int(len(sub)),
            "pearson": float(sub[a].corr(sub[b])),
            "spearman": float(sub[a].corr(sub[b], method="spearman")),
        }

    out = {
        "lens_a": args.lens,
        "lens_b": args.compare_lens,
        "column_a": a,
        "column_b": b,
        "all_tokens": pair(d),
    }
    if flag:
        # Agreement carried entirely by the words the model already typed is not agreement
        # about the residual stream.
        out["excluding_signal_words"] = pair(d[d[flag].astype(int) == 0])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-token", type=Path, default=Path("/workspace/reasoning_theatre/loudness/per_token.csv"))
    ap.add_argument("--out", type=Path, default=Path("/workspace/reasoning_theatre/loudness"))
    ap.add_argument("--min-sentence-len", type=int, default=5)
    ap.add_argument("--min-sentences", type=int, default=5)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument(
        "--lens",
        default="jlens",
        help="Which lens's loudness is being profiled (default: %(default)s). Named in every "
        "table and caption -- an unqualified 'loudness' is not a quantity.",
    )
    ap.add_argument("--signal-name", default="direction", help="Which vocabulary (default: %(default)s).")
    ap.add_argument(
        "--compare-lens",
        default="logitlens",
        help="Second lens to correlate the first against, when the table carries both "
        "(default: %(default)s). Skipped silently when only one ruler is present.",
    )
    ap.add_argument("--layer", type=int, default=15, help="Layer the loudness is read at.")
    ap.add_argument(
        "--loudness-column",
        default=None,
        help="Read this column as the value instead of the one --lens/--signal-name/--layer "
        "imply. Legacy spellings are resolved automatically.",
    )
    ap.add_argument(
        "--exclude-signal-words",
        action="store_true",
        help="Drop rows whose token is a signal word before anything else, and produce the "
        "SAME tables from what is left. The per-regime breakdown below always reports the "
        "filtered variants too; this makes the filter the whole analysis rather than one row "
        "of it.",
    )
    ap.add_argument(
        "--exclude-radius",
        type=int,
        default=0,
        help="With --exclude-signal-words, also drop rows within N tokens of a signal word "
        "(0 = only the word itself).",
    )
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    cols = [
        "name",
        "reasoning_pos",
        "complexity",
        "sentence_len",
        "sentence_frac",
        "rel_sentence",
        "reasoning_frac",
        "n_sentences",
        "convinced_idx",
        "convinced_reasoning_frac",
        "is_signal_word",
        VAL,
    ]
    # Resolve what this table actually calls the two signal columns before asking for them by
    # name: a table written before the naming convention spells them dir_prob_L15 and
    # is_direction_token, and usecols on a missing name is a hard failure.
    header = pd.read_csv(args.per_token, nrows=0, keep_default_na=False).columns.tolist()
    member_col = _cols.membership_column(args.signal_name)
    if member_col not in header:
        member_col = "is_direction_token" if "is_direction_token" in header else "is_signal_word"
    if args.loudness_column:
        value_col = args.loudness_column
    elif VAL in header:
        value_col = VAL
    else:
        prob = _cols.prob_column(args.lens, args.signal_name, args.layer)
        value_col = prob if prob in header else "dir_prob_L15"
    if value_col not in header:
        raise SystemExit(
            f"{args.per_token} has no loudness value column; looked for {VAL!r}, "
            f"{_cols.prob_column(args.lens, args.signal_name, args.layer)!r} and 'dir_prob_L15'. "
            "Pass --loudness-column."
        )

    on_disk = [value_col if c == VAL else member_col if c == "is_signal_word" else c for c in cols]
    ref = pd.read_csv(args.per_token, usecols=on_disk, keep_default_na=False, na_values=[""])
    ref = ref.rename(columns={value_col: VAL, member_col: "is_signal_word"})
    ref = ref.sort_values(["name", "reasoning_pos"])

    n_before = len(ref)
    if args.exclude_signal_words:
        drop = ref["is_signal_word"].astype(int).astype(bool)
        if args.exclude_radius > 0:
            grouped = ref["is_signal_word"].astype(int).groupby(ref["name"])
            for shift in range(1, args.exclude_radius + 1):
                drop |= grouped.shift(shift).fillna(0).astype(bool)
                drop |= grouped.shift(-shift).fillna(0).astype(bool)
        ref = ref[~drop].copy()
        print(
            f"--exclude-signal-words (radius {args.exclude_radius}): "
            f"{n_before} -> {len(ref)} rows ({n_before - len(ref)} dropped)",
            flush=True,
        )
    # `is_signal_word` flags the token the MODEL EMITTED, so dropping those rows asks whether the
    # boundary bump survives where the model is not writing a signal word. It does not control for
    # PROXIMITY: the lens predicts the next tokens, so the token just before " up" is loud without
    # being a signal word itself. near{k} widens the exclusion to a +-k window, which is the
    # control that actually separates "the residual stream is direction-loaded here" from "a
    # signal word is about to be written".
    flag = ref.groupby("name")["is_signal_word"]
    for k in (1, 2, 3):
        near = np.zeros(len(ref), dtype=bool)
        for shift in range(-k, k + 1):
            near |= flag.shift(shift).fillna(0).to_numpy().astype(bool)
        ref[f"near{k}"] = near
    df = ref[ref["convinced_idx"] >= 1]
    long = df[df["sentence_len"] >= args.min_sentence_len]

    out: dict = {
        "rows": int(len(ref)),
        "trajectories": int(ref["name"].nunique()),
        "cohort_rows": int(len(df)),
        "cohort_trajectories": int(df["name"].nunique()),
        "min_sentence_len": args.min_sentence_len,
        "min_sentences": args.min_sentences,
    }

    rows = []
    regimes = {
        "all sentences": long,
        "convinced sentence (rel 0)": long[long["rel_sentence"] == 0],
        "already convinced (rel >= +1)": long[long["rel_sentence"] >= 1],
        "not convinced (rel = -1)": long[long["rel_sentence"] == -1],
        "not convinced (rel <= -2)": long[long["rel_sentence"] <= -2],
    }
    for comp in ["all", *COMPLEXITIES]:
        for label, regime in regimes.items():
            g = regime if comp == "all" else regime[regime["complexity"] == comp]
            if g.empty:
                continue
            op = g[g["sentence_frac"] < 0.1]
            cl = g[g["sentence_frac"] > 0.9]
            m, lo, hi = boot_mean(g, VAL, args.boot, rng)
            mo, olo, ohi = boot_mean(op, VAL, args.boot, rng)
            mc, clo, chi = boot_mean(cl, VAL, args.boot, rng)
            rows.append(
                {
                    "complexity": comp,
                    "regime": label,
                    "n_tokens": len(g),
                    "n_traj": g["name"].nunique(),
                    "mean": m,
                    "mean_lo": lo,
                    "mean_hi": hi,
                    "open_mean": mo,
                    "open_lo": olo,
                    "open_hi": ohi,
                    "close_mean": mc,
                    "close_lo": clo,
                    "close_hi": chi,
                    "open_over_close": mo / mc if mc else np.nan,
                }
            )
    tab = pd.DataFrame(rows)
    (args.out / "tables").mkdir(parents=True, exist_ok=True)
    tab.to_csv(args.out / "tables" / "loudness_summary.csv", index=False)

    out["paired_opening"] = {
        "convinced sentence -> next sentence (rel 0 -> +1)": paired_open(long, 0, 1, VAL, args.boot, rng),
        "pre-boundary -> convinced sentence (rel -1 -> 0)": paired_open(long, -1, 0, VAL, args.boot, rng),
        "rel -2 -> rel -1": paired_open(long, -2, -1, VAL, args.boot, rng),
    }

    # The boundary step, under every control: is the convinced sentence louder than the one
    # before it? Paired within trajectory, so a trajectory that is loud throughout cannot
    # produce the difference.
    def paired_sentence(sub: pd.DataFrame, a: int, b: int, col: str = VAL) -> dict:
        pa = sub[sub["rel_sentence"] == a].groupby("name")[col].mean()
        pb = sub[sub["rel_sentence"] == b].groupby("name")[col].mean()
        both = pa.index.intersection(pb.index)
        d = (pb.loc[both] - pa.loc[both]).to_numpy()
        if len(d) < 2:
            return {}
        idx = rng.integers(0, len(d), size=(args.boot, len(d)))
        draws = d[idx].mean(1)
        return {
            "n_traj": int(len(d)),
            "mean_diff": float(d.mean()),
            "lo": float(np.percentile(draws, 2.5)),
            "hi": float(np.percentile(draws, 97.5)),
            "frac_positive": float((d > 0).mean()),
        }

    mid = long[(long["reasoning_frac"] > 0.2) & (long["reasoning_frac"] < 0.8)]
    subsets = {
        "all tokens": long,
        "token is not a signal word": long[long["is_signal_word"] == 0],
        "no signal word within +-1": long[~long["near1"]],
        "no signal word within +-2": long[~long["near2"]],
        "no signal word within +-3": long[~long["near3"]],
    }
    out["boundary_step"] = {
        "whole sentence, rel -1 -> 0": paired_sentence(long, -1, 0),
        "whole sentence, rel -1 -> +1": paired_sentence(long, -1, 1),
        "mid-chain only, rel -1 -> 0": paired_sentence(mid, -1, 0),
        "non-direction tokens, rel -1 -> 0": paired_sentence(subsets["token is not a signal word"], -1, 0),
        "signal-word share, rel -1 -> 0": paired_sentence(long, -1, 0, "is_signal_word"),
        "placebo: rel -2 -> -1": paired_sentence(long, -2, -1),
        "placebo: rel -3 -> -2": paired_sentence(long, -3, -2),
    }
    # Each exclusion re-run with its own placebos: a step that shrinks toward its placebo is
    # a step that was mostly proximity to a verbalized signal word.
    out["verbalization_proximity"] = {
        label: {
            "tokens_kept": float(len(g) / len(long)),
            "profile_rel_-6_to_+6": [
                float(v) for v in g.assign(r=g["rel_sentence"].clip(-6, 6)).groupby("r")[VAL].mean()
            ],
            "boundary rel -1 -> 0": paired_sentence(g, -1, 0),
            "boundary rel -1 -> +1": paired_sentence(g, -1, 1),
            "placebo rel -2 -> -1": paired_sentence(g, -2, -1),
            "placebo rel -3 -> -2": paired_sentence(g, -3, -2),
        }
        for label, g in subsets.items()
    }

    chain = df[df["n_sentences"] >= args.min_sentences]
    conv = chain.groupby("name")["convinced_reasoning_frac"].first().dropna()
    out["convinced_eos_fraction"] = {
        "n": int(len(conv)),
        "mean": float(conv.mean()),
        "median": float(conv.median()),
        "q25": float(conv.quantile(0.25)),
        "q75": float(conv.quantile(0.75)),
    }
    prof = []
    edges = np.linspace(0, 1, 21)
    idx = np.clip(np.digitize(chain["reasoning_frac"].to_numpy(), edges) - 1, 0, 19)
    for b in range(20):
        g = chain[idx == b]
        m, lo, hi = boot_mean(g, VAL, args.boot, rng)
        prof.append({"x": float((edges[b] + edges[b + 1]) / 2), "n": int(len(g)), "mean": m, "lo": lo, "hi": hi})
    out["chain_profile"] = prof
    agreement = lens_agreement(args.per_token, args)
    if agreement:
        out["lens_agreement"] = agreement
        print(
            f"lens agreement {agreement['lens_a']} vs {agreement['lens_b']}: "
            f"pearson {agreement['all_tokens']['pearson']:.3f}",
            flush=True,
        )
    (args.out / "summary.json").write_text(json.dumps(out, indent=2))

    cfg = _prov.RunConfig("loudness_analysis/analysis/loudness_distribution.py")
    cfg.measurement(lens=args.lens, signal=args.signal_name, layer=args.layer)
    cfg.input("per_token", args.per_token)
    cfg.params.update(
        {
            "min_sentence_len": args.min_sentence_len,
            "min_sentences": args.min_sentences,
            "exclude_signal_words": args.exclude_signal_words,
            "exclude_radius": args.exclude_radius,
        }
    )
    cfg.aggregation(
        statistic="mean loudness on the probability scale",
        resolved_value_column=value_col,
        resolved_membership_column=member_col,
        bootstrap="trajectory-clustered",
        n_boot=args.boot,
        seed=0,
    )
    cfg.rows("input", n_before)
    cfg.rows("after_signal_word_filter", len(ref))
    cfg.guard(args.out, "lens", "signal", "layer")
    cfg.write(args.out)

    pd.set_option("display.width", 250)
    print(tab[tab["complexity"] == "all"].to_string(index=False))
    print()
    print(json.dumps({k: out[k] for k in ("boundary_step", "paired_opening", "convinced_eos_fraction")}, indent=2))
    print(f"\nwrote {args.out / 'tables' / 'loudness_summary.csv'} and {args.out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
