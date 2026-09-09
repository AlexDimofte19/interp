#!/usr/bin/env python3
"""The equal-N correction, as a table: entry 45/49/51 arms against their entry-52 rebuilds.

Reads `results.best_balanced_accuracy` out of each checkpoint -- the number
`train_next_action_probe::_evaluate` wrote on the arm's OWN eval-720 selection -- and pairs
each superseded probe with the one that replaces it.

**The correction is not spread evenly across the arms, and that is the whole point.** The
final sentence is ~4.8% of every chain, but the loud and random arms already held nearly all
of theirs -- they lost only the 30 / 67 / 776 cases where their pick happened to land on the
chain's last token. `eos` lost ALL 3,600, because its pick IS each sentence's last token. So
restoring them moves eos by ~4.8% of its rows and the others by 0.04-1.0%, and those rows are
the near-deterministic ones (99.8% agree with the final action). eos was being graded on a
distribution with the easy rows stripped out while its comparators kept theirs.

Expect, therefore: eos rises materially, the other three barely move, and the eos-to-jlens gap
-- entry 51's "+2.8 pp for loudness" -- shrinks. The decomposition below is computed within
each round rather than by differencing the two columns, since only same-round gaps are
comparable.

    python scripts/compare_equal_n_arms.py [--heldout CSV] [--json-out FILE]
"""

import argparse
import json
from pathlib import Path

import torch

PROBES = Path("/workspace/probes")
LB = Path("/workspace/reasoning_theatre/local_belief_probes/probes")
BASE = PROBES / "local_belief_baselines"
NEW = PROBES / "local_belief_equalN"

# arm -> (label, old checkpoint stem builder, new checkpoint stem builder)
ARMS = [
    ("p1", "random", "random in span", lambda m: BASE / f"next_action_probe_random_sentence_belief_{m}.pt"),
    ("p1", "eos", "last token of sentence", lambda m: BASE / f"next_action_probe_eos_belief_{m}.pt"),
    ("p1", "logitlens", "logitlens loudest", lambda m: BASE / f"next_action_probe_logitlens_p1_{m}.pt"),
    ("p1", "jlens", "jlens loudest", lambda m: LB / f"local_belief_p1_{m}.pt"),
    (
        "p1-top20",
        "random_top20",
        "random in span, to 20",
        lambda m: BASE / f"next_action_probe_random_sentence_belief_top20_{m}.pt",
    ),
    (
        "p1-top20",
        "logitlens_top20",
        "logitlens loudest, to 20",
        lambda m: BASE / f"next_action_probe_logitlens_p1_top20_{m}.pt",
    ),
    ("p1-top20", "jlens_top20", "jlens loudest, to 20", lambda m: LB / f"local_belief_p1_top20_{m}.pt"),
]


def balanced(path: Path) -> tuple[float | None, int | None]:
    """(best balanced accuracy, train n) from a checkpoint, or (None, None) if absent."""
    if not path.exists():
        return None, None
    ck = torch.load(path, map_location="cpu", weights_only=False)
    res = ck.get("results", {})
    return res.get("best_balanced_accuracy"), res.get("train_samples") or res.get("n_train")


def fmt(x: float | None) -> str:
    return f"{x:.4f}" if isinstance(x, float) else "  --  "


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    rows = []
    print("EVAL-720, each probe on its OWN selection (balanced accuracy)")
    print("levels are NOT comparable across rounds -- see the module docstring; gaps are")
    print()
    print(
        f"{'cadence':<9} {'arm':<16} {'what it cuts at':<24} {'mt':<4} {'entry 45/49/51':>14} {'entry 52':>10} {'delta':>8}"
    )
    for cadence, arm, label, old_path in ARMS:
        for mt in ("lr", "mlp"):
            old, _ = balanced(old_path(mt))
            sub = "p1" if cadence == "p1" else "p1-top20"
            new, _ = balanced(NEW / sub / f"next_action_probe_{arm}_{mt}.pt")
            delta = f"{new - old:+.4f}" if isinstance(old, float) and isinstance(new, float) else "   --   "
            print(f"{cadence:<9} {arm:<16} {label:<24} {mt:<4} {fmt(old):>14} {fmt(new):>10} {delta:>8}")
            rows.append({"cadence": cadence, "arm": arm, "model_type": mt, "old": old, "new": new})

    # The decomposition, computed WITHIN each round so the near-deterministic rows that both
    # arms of a pair now carry cancel out of the difference.
    print()
    print("POSITION vs LOUDNESS, within each round (mlp, uncapped p1)")
    by = {(r["arm"], r["model_type"]): r for r in rows}
    for era in ("old", "new"):
        r_, e_, j_ = (by[(a, "mlp")][era] for a in ("random", "eos", "jlens"))
        if not all(isinstance(v, float) for v in (r_, e_, j_)):
            print(f"  {era:>3}: incomplete")
            continue
        pos, loud = e_ - r_, j_ - e_
        total = j_ - r_
        share = f"{pos / total:.0%} / {loud / total:.0%}" if total else "n/a"
        print(
            f"  {era:>3}: random {r_:.4f} -> eos {e_:.4f} = {pos:+.4f} (cut at a sentence boundary); "
            f"eos -> jlens {j_:.4f} = {loud:+.4f} (cut at the loudest point);  position/loudness = {share}"
        )

    if args.json_out:
        args.json_out.write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
