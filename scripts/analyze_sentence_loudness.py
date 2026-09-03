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

COMPLEXITIES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
VAL = "dir_prob_L15"


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-token", type=Path, default=Path("/workspace/reasoning_theatre/loudness/per_token.csv"))
    ap.add_argument("--out", type=Path, default=Path("/workspace/reasoning_theatre/loudness"))
    ap.add_argument("--min-sentence-len", type=int, default=5)
    ap.add_argument("--min-sentences", type=int, default=5)
    ap.add_argument("--boot", type=int, default=500)
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
        "is_direction_token",
        VAL,
    ]
    ref = pd.read_csv(args.per_token, usecols=cols, keep_default_na=False, na_values=[""])
    ref = ref.sort_values(["name", "reasoning_pos"])
    # `is_direction_token` flags the token the MODEL EMITTED, so dropping those rows asks whether the
    # boundary bump survives where the model is not writing a direction word. It does not control for
    # PROXIMITY: the lens predicts the next tokens, so the token just before " up" is loud without
    # being a direction word itself. near{k} widens the exclusion to a +-k window, which is the
    # control that actually separates "the residual stream is direction-loaded here" from "a
    # direction word is about to be written".
    flag = ref.groupby("name")["is_direction_token"]
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
        "token is not a direction word": long[long["is_direction_token"] == 0],
        "no direction word within +-1": long[~long["near1"]],
        "no direction word within +-2": long[~long["near2"]],
        "no direction word within +-3": long[~long["near3"]],
    }
    out["boundary_step"] = {
        "whole sentence, rel -1 -> 0": paired_sentence(long, -1, 0),
        "whole sentence, rel -1 -> +1": paired_sentence(long, -1, 1),
        "mid-chain only, rel -1 -> 0": paired_sentence(mid, -1, 0),
        "non-direction tokens, rel -1 -> 0": paired_sentence(subsets["token is not a direction word"], -1, 0),
        "direction-token share, rel -1 -> 0": paired_sentence(long, -1, 0, "is_direction_token"),
        "placebo: rel -2 -> -1": paired_sentence(long, -2, -1),
        "placebo: rel -3 -> -2": paired_sentence(long, -3, -2),
    }
    # Each exclusion re-run with its own placebos: a step that shrinks toward its placebo is
    # a step that was mostly proximity to a verbalized direction word.
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
    (args.out / "summary.json").write_text(json.dumps(out, indent=2))

    pd.set_option("display.width", 250)
    print(tab[tab["complexity"] == "all"].to_string(index=False))
    print()
    print(json.dumps({k: out[k] for k in ("boundary_step", "paired_opening", "convinced_eos_fraction")}, indent=2))
    print(f"\nwrote {args.out / 'tables' / 'loudness_summary.csv'} and {args.out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
