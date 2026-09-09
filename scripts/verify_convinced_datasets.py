#!/usr/bin/env python3
"""Check the four convinced-classifier datasets against artifacts built by other code.

The first two checks are cheap; the last two are the ones worth having. Both re-derive the same
numbers through a *different* script written months earlier, so agreement is evidence the join is
right rather than evidence it is self-consistent:

  * ``loudness/per_token.csv`` (entry 42, ``build_sentence_loudness.py``) covers every reasoning
    token of the same 2880 training trajectories at L15. Every training row must appear in it
    with the same loudness, sentence placement and convinced index.
  * ``probe_vs_rollout/per_token.csv`` (entry 39, ``build_probe_rollout_join.py``) covers all
    87,221 held-out tokens and carries its own ``is_after_convinced`` and ``jlens_mass_L15``,
    derived independently -- including via the hardcoded ``reasoning_pos = token_idx - 3`` this
    project distrusts. Agreement there also retires that worry for these trajectories.
"""

import argparse
import csv
import sys
from pathlib import Path

csv.field_size_limit(10**9)


def read_rows(path: Path, keep: set[str] | None = None) -> list[dict]:
    """Rows of a dataset CSV, optionally narrowed to the columns a check needs."""
    with open(path, encoding="utf-8", newline="") as f:
        return [{k: r[k] for k in keep} if keep else r for r in csv.DictReader(f)]


def names_of(path: Path) -> set[str]:
    with open(path, encoding="utf-8", newline="") as f:
        return {r["name"] for r in csv.DictReader(f)}


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}", flush=True)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("/workspace/reasoning_theatre/convinced_classifier"))
    ap.add_argument("--loudness", type=Path, default=Path("/workspace/reasoning_theatre/loudness/per_token.csv"))
    ap.add_argument(
        "--probe-rollout", type=Path, default=Path("/workspace/reasoning_theatre/probe_vs_rollout/per_token.csv")
    )
    args = ap.parse_args()
    passed = True

    train = args.root / "train_random20.csv"
    val1 = args.root / "val1_random20.csv"
    val2 = args.root / "val2_jlens20.csv"
    ev = args.root / "eval_heldout360_all.csv"

    print("SPLIT DISJOINTNESS", flush=True)
    n_train, n_val1, n_val2, n_ev = (names_of(p) for p in (train, val1, val2, ev))
    passed &= check("train n=2880", len(n_train) == 2880, f"{len(n_train)}")
    passed &= check("val1 n=720", len(n_val1) == 720, f"{len(n_val1)}")
    passed &= check("train n val1 empty", not (n_train & n_val1), f"{len(n_train & n_val1)} shared")
    passed &= check("val1 == val2 names", n_val1 == n_val2, f"{len(n_val1 ^ n_val2)} differ")
    passed &= check("eval n=360", len(n_ev) == 360, f"{len(n_ev)}")
    passed &= check(
        "eval disjoint from train+val", not (n_ev & (n_train | n_val1)), f"{len(n_ev & (n_train | n_val1))} shared"
    )

    print("\nVAL1 / VAL2 COVER THE SAME TRAJECTORIES, DIFFERENTLY SELECTED", flush=True)
    r1, r2 = read_rows(val1, {"name", "abs_pos"}), read_rows(val2, {"name", "abs_pos"})
    passed &= check("same row count", len(r1) == len(r2), f"{len(r1)} vs {len(r2)}")
    k1 = {(r["name"], r["abs_pos"]) for r in r1}
    k2 = {(r["name"], r["abs_pos"]) for r in r2}
    print(f"         overlap {len(k1 & k2)} of {len(k1)} tokens -- the selections really differ", flush=True)

    print("\nTRAINING ROWS vs loudness/per_token.csv (entry 42, different script)", flush=True)
    cols = {"name", "step", "reasoning_pos", "dir_logmass_L15", "sentence_idx", "convinced_idx", "is_direction_token"}
    want = {(r["name"], r["step"], r["reasoning_pos"]): r for r in read_rows(train, cols | {"is_convinced"})}
    found, mismatch = 0, []
    with open(args.loudness, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            key = (r["name"], r["step"], r["reasoning_pos"])
            mine = want.get(key)
            if mine is None:
                continue
            found += 1
            for col in ("dir_logmass_L15", "sentence_idx", "convinced_idx", "is_direction_token"):
                if r[col] != mine[col]:
                    mismatch.append((key, col, r[col], mine[col]))
            rel = r["rel_sentence"]
            expect = 0 if rel == "" else int(int(rel) >= 0)
            if expect != int(mine["is_convinced"]):
                mismatch.append((key, "is_convinced", rel, mine["is_convinced"]))
    passed &= check("every training row present", found == len(want), f"{found}/{len(want)}")
    passed &= check("all shared columns agree", not mismatch, f"{len(mismatch)} mismatches {mismatch[:2]}")

    print("\nHELD-OUT ROWS vs probe_vs_rollout/per_token.csv (entry 39, different script)", flush=True)
    ev_cols = {"name", "step", "abs_pos", "is_convinced", "dir_logmass_L15"}
    mine_ev = {(r["name"], r["step"], r["abs_pos"]): r for r in read_rows(ev, ev_cols)}
    found, label_bad, mass_bad = 0, 0, 0
    with open(args.probe_rollout, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            mine = mine_ev.get((r["name"], r["step"], r["abs_pos"]))
            if mine is None:
                continue
            found += 1
            theirs = 0 if r["is_after_convinced"] == "" else int(r["is_after_convinced"])
            if theirs != int(mine["is_convinced"]):
                label_bad += 1
            if abs(float(r["jlens_mass_L15"]) - float(mine["dir_logmass_L15"])) > 1e-6:
                mass_bad += 1
    passed &= check("every held-out row present", found == len(mine_ev), f"{found}/{len(mine_ev)}")
    passed &= check("labels agree", label_bad == 0, f"{label_bad} disagree")
    passed &= check("L15 loudness agrees to 1e-6", mass_bad == 0, f"{mass_bad} differ")

    print(f"\n{'ALL CHECKS PASSED' if passed else 'SOME CHECKS FAILED'}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
