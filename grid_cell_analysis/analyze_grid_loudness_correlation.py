#!/usr/bin/env python3
"""Does DIRECTION loudness predict where the GRID is decodable? (It should not.)

Consumes the per-token CSV from `score_probes_per_token.py --probe-type grid_tile` and produces the
table that answers it: balanced accuracy of the grid_tile probe binned by the token's jlens
layer-15 direction mass.

The comparison this is built for is log entry 37(c), which ran the identical binning with
the `next_action` label and found balanced accuracy climbing monotonically across all ten
deciles (.291 -> .536 lr, .313 -> .573 mlp) with no reversal. That is the ALTERNATIVE
hypothesis here. If the direction score were a generic "this token is informative" measure,
the grid curve would climb too; if it is specific to the action, the grid curve is flat and
the decile-10 minus decile-1 gap straddles zero.

Balanced accuracy is rebuilt from per-class counts rather than averaged over rows: a token
owns C cells, and per-class recall over a whole bucket is
`sum(n_correct_c) / sum(n_true_c)` averaged over the classes with support. Averaging each
token's own balanced accuracy would instead weight a token with one goal cell like a token
with forty wall cells.

Bootstraps resample TRAJECTORY NAMES, not rows -- a trajectory's ~240 tokens share one grid
and one set of cells, so rows within it are anything but independent.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Cells the model can actually see. 7 is padding, which only appears with --pad-to-size and
# is trivially predictable, so it is excluded from balanced accuracy by default.
REAL_CLASSES = (0, 1, 2, 3, 4, 5, 6)
CLASS_SYMBOL = {0: "A", 1: "#", 2: "G", 3: "_", 4: "D", 5: "K", 6: "?", 7: "+"}


def balanced_accuracy(df: pd.DataFrame, probe: str, classes=REAL_CLASSES) -> tuple[float, dict]:
    """Balanced accuracy over a bucket of token rows, plus the per-class recalls.

    A class with no support in the bucket is dropped rather than counted as 0 -- it was
    never asked about.
    """
    recalls: dict[int, float] = {}
    for c in classes:
        tcol, ccol = f"n_true_{c}", f"{probe}_correct_{c}"
        if tcol not in df.columns or ccol not in df.columns:
            continue
        n_true = float(df[tcol].sum())
        if n_true > 0:
            recalls[c] = float(df[ccol].sum()) / n_true
    ba = float(np.mean(list(recalls.values()))) if recalls else float("nan")
    return ba, recalls


def plain_accuracy(df: pd.DataFrame, probe: str) -> float:
    n = float(df["n_cells"].sum())
    return float(df[f"{probe}_n_correct"].sum()) / n if n else float("nan")


def decile_table(df: pd.DataFrame, probe: str, score: str, n_bins: int) -> pd.DataFrame:
    """Balanced accuracy per bin of `score`, lowest bin first."""
    # qcut on ranks, so ties (a floor of NO_MATCH_LOGPROB is common) cannot collapse bins.
    df = df.copy()
    df["_bin"] = pd.qcut(df[score].rank(method="first"), n_bins, labels=False)
    rows = []
    for b in range(n_bins):
        g = df[df["_bin"] == b]
        if g.empty:
            continue
        ba, recalls = balanced_accuracy(g, probe)
        rows.append(
            {
                "bin": b + 1,
                "n_tokens": len(g),
                "n_cells": int(g["n_cells"].sum()),
                "mean_score": float(g[score].mean()),
                "balanced_acc": ba,
                "accuracy": plain_accuracy(g, probe),
                **{f"recall_{CLASS_SYMBOL[c]}": v for c, v in recalls.items()},
            }
        )
    return pd.DataFrame(rows)


def gap_bootstrap(df: pd.DataFrame, probe: str, score: str, n_bins: int, n_boot: int, seed: int) -> dict:
    """Trajectory-clustered bootstrap of (top bin BA - bottom bin BA).

    The bins are recomputed inside each resample: the decile edges are themselves a function
    of the sample, and holding them fixed would understate the spread.
    """
    names = df["name"].unique()
    rng = np.random.default_rng(seed)
    by_name = dict(df.groupby("name").__iter__())
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(len(names), size=len(names), replace=True)
        rs = pd.concat([by_name[names[i]] for i in pick], ignore_index=True)
        t = decile_table(rs, probe, score, n_bins)
        if len(t) < 2:
            continue
        draws.append(t["balanced_acc"].iloc[-1] - t["balanced_acc"].iloc[0])
    if not draws:
        return {}
    base = decile_table(df, probe, score, n_bins)
    return {
        "gap": float(base["balanced_acc"].iloc[-1] - base["balanced_acc"].iloc[0]),
        "lo": float(np.percentile(draws, 2.5)),
        "hi": float(np.percentile(draws, 97.5)),
        "n_boot": len(draws),
    }


def spearman(df: pd.DataFrame, probe: str, score: str) -> float:
    """Rank correlation between a token's loudness and its own per-token accuracy."""
    return float(df[score].corr(df[f"{probe}_acc"], method="spearman"))


def verdict(gap: float, lo: float, hi: float, reference: float) -> str:
    """Read the gap as an EFFECT SIZE against a reference, not as a significance test.

    At 87k tokens a 2pp gap is comfortably resolvable, so "the CI excludes zero" is nearly
    guaranteed and says almost nothing on its own -- reporting it alone would call a -0.02
    drift "correlated" and invite exactly the wrong conclusion. What the specificity question
    turns on is whether the gap approaches the same-token, same-layer ACTION gap (entry 37's
    +0.155 for the matched uniform-trained probe), so that is the yardstick.
    """
    if lo <= 0 <= hi:
        return f"straddles zero: no detectable trend (reference action gap {reference:+.3f})"
    direction = "RISES with loudness" if gap > 0 else "FALLS slightly as loudness rises"
    frac = abs(gap) / abs(reference) if reference else float("inf")
    if gap < 0:
        return (
            f"{direction}; |gap| is {frac:.0%} of the action reference {reference:+.3f} "
            "and OPPOSITE in sign -> loudness does not predict this label"
        )
    if frac < 0.25:
        return (
            f"{direction} but |gap| is only {frac:.0%} of the action reference "
            f"{reference:+.3f} -> far weaker than the action effect"
        )
    return f"{direction}, {frac:.0%} of the action reference {reference:+.3f} -> comparable to the action effect"


def monotone_runs(t: pd.DataFrame) -> int:
    """How many times the binned curve reverses direction. 0 = perfectly monotone."""
    d = np.diff(t["balanced_acc"].to_numpy())
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return 0
    return int((np.sign(d[1:]) != np.sign(d[:-1])).sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("per_token", type=Path, help="CSV from score_probes_per_token.py --probe-type grid_tile")
    ap.add_argument("--probe", action="append", default=None, help="Probe key(s); default every one found.")
    ap.add_argument("--score", default="jlens_mass_L15", help="Loudness column to bin by.")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--reference",
        type=float,
        default=0.1559,
        help="The ACTION decile gap to read this one against: entry 37/38's matched "
        "uniform-trained next_action probe on these same 87,221 tokens "
        "(next_action_mass_l15.random_topall_mlp, .3924 -> .5483). A significance test "
        "alone is uninformative at this n; the effect SIZE against this bar is the result.",
    )
    ap.add_argument("--out", type=Path, default=None, help="Write the tables here as JSON.")
    args = ap.parse_args()

    # DictReader semantics matter for lens CSVs; this one is written by us and holds no raw
    # decoded tokens in numeric columns, but keep_default_na=False still protects the token
    # column, which legitimately contains "NA".
    df = pd.read_csv(args.per_token, keep_default_na=False, na_values=[""])
    if args.score not in df.columns:
        raise SystemExit(f"no column {args.score!r}; have {[c for c in df.columns if 'mass' in c]}")
    df = df[df[args.score].notna()].copy()
    df[args.score] = df[args.score].astype(float)

    probes = args.probe or sorted({c[: -len("_n_correct")] for c in df.columns if c.endswith("_n_correct")})
    print(f"{len(df)} token rows over {df['name'].nunique()} trajectories, {int(df['n_cells'].sum())} cell evals")
    print(f"binning by {args.score} into {args.bins} bins\n")

    out: dict = {"per_token": str(args.per_token), "score": args.score, "bins": args.bins, "probes": {}}
    for probe in probes:
        overall_ba, overall_recalls = balanced_accuracy(df, probe)
        t = decile_table(df, probe, args.score, args.bins)
        g = gap_bootstrap(df, probe, args.score, args.bins, args.n_boot, args.seed)
        rho = spearman(df, probe, args.score)

        print("=" * 78)
        print(f"{probe}   overall balanced acc {overall_ba:.4f}   accuracy {plain_accuracy(df, probe):.4f}")
        print("  per-class recall: " + "  ".join(f"{CLASS_SYMBOL[c]}={v:.3f}" for c, v in overall_recalls.items()))
        print("=" * 78)
        cols = ["bin", "n_tokens", "n_cells", "mean_score", "balanced_acc", "accuracy"]
        print(t[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(f"\n  top-bin minus bottom-bin balanced acc: {g.get('gap', float('nan')):+.4f}")
        if g:
            print(f"    95% CI (trajectory-clustered, {g['n_boot']} draws): [{g['lo']:+.4f}, {g['hi']:+.4f}]")
            print(f"    {verdict(g['gap'], g['lo'], g['hi'], args.reference)}")
        print(f"  Spearman(loudness, per-token accuracy) = {rho:+.4f}")
        print(f"  direction reversals in the binned curve: {monotone_runs(t)} (0 = monotone)\n")

        out["probes"][probe] = {
            "overall_balanced_acc": overall_ba,
            "overall_accuracy": plain_accuracy(df, probe),
            "per_class_recall": {CLASS_SYMBOL[c]: v for c, v in overall_recalls.items()},
            "table": t.to_dict(orient="records"),
            "gap": g,
            "spearman": rho,
            "reversals": monotone_runs(t),
        }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
