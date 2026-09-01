#!/usr/bin/env python3
"""Clone ``probe_vs_rollout/per_token.csv`` with extra probe arms merged in.

``build_probe_rollout_join.py`` joins one ``eval_probe_per_token.py`` run to the rollouts and
the sentence structure. Scoring a NEW probe on the same trajectory set does not need any of
that redone -- the sentence columns are a property of the trajectories, not of the probe --
so this merges the new run's ``{arm}_pred`` / ``_correct`` / ``_p_true`` / ``_p_{ACTION}``
columns onto the existing table on ``(name, step, abs_pos)``.

It writes a CLONE and never touches the input, because the entry-39/41 report is published
off the original and its numbers must not move under it.

The merge is validated, not assumed: both sides must be unique on the key and the extra run
must cover every row of the base. A partial overlap means the two runs saw different
trajectory sets, which would silently produce an arm scored on a subset -- refused rather
than filled with NaN.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

KEY = ["name", "step", "abs_pos"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--base",
        type=Path,
        default=Path("/workspace/reasoning_theatre/probe_vs_rollout/per_token.csv"),
        help="the joined table to clone (read-only).",
    )
    ap.add_argument(
        "--extra",
        type=Path,
        action="append",
        required=True,
        help="an eval_probe_per_token.py CSV holding the new arm(s). Repeatable.",
    )
    ap.add_argument("--out", type=Path, required=True, help="the clone to write.")
    args = ap.parse_args()

    # keep_default_na off: decoded tokens include the literal string "NA".
    base = pd.read_csv(args.base, keep_default_na=False, na_values=[""], low_memory=False)
    print(f"base {len(base)} rows, {sum(c.endswith('_pred') for c in base.columns)} arms", flush=True)
    if base.duplicated(KEY).any():
        raise SystemExit(f"base is not unique on {KEY}")

    merged = base
    for path in args.extra:
        ex = pd.read_csv(path, keep_default_na=False, na_values=[""], low_memory=False)
        arms = sorted(c[: -len("_pred")] for c in ex.columns if c.endswith("_pred"))
        if not arms:
            raise SystemExit(f"{path}: no *_pred columns")
        if ex.duplicated(KEY).any():
            raise SystemExit(f"{path} is not unique on {KEY}")
        # every column the new run carries FOR ITS OWN ARMS, and nothing else: the shared
        # columns (token, label, lens scores) already exist in the base and must not be
        # overwritten by a second run's copy of them.
        cols = [c for c in ex.columns if any(c.startswith(f"{a}_") for a in arms)]
        clash = [c for c in cols if c in merged.columns]
        if clash:
            raise SystemExit(f"{path}: arms already present in the base: {clash[:4]}")

        before = len(merged)
        merged = merged.merge(ex[KEY + cols], on=KEY, how="left", validate="one_to_one")
        if len(merged) != before:
            raise SystemExit(f"{path}: merge changed the row count {before} -> {len(merged)}")
        missing = int(merged[cols[0]].isna().sum())
        if missing:
            raise SystemExit(
                f"{path}: {missing} of {before} base rows have no row in the extra run "
                "-- the two runs saw different trajectories, refusing to write a partial arm"
            )
        print(f"  + {path.name}: {arms} ({len(cols)} columns), full coverage", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    n_arms = sum(c.endswith("_pred") for c in merged.columns)
    print(f"wrote {len(merged)} rows x {len(merged.columns)} cols, {n_arms} arms -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
