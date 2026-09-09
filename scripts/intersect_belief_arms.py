#!/usr/bin/env python3
"""Cut a set of per-sentence belief arms down to the sentences they all hold.

The arms exist to vary one thing -- which token inside a sentence the reasoning is cut at --
so any difference in *which sentences* they cover is a confound rather than a result. Linking
the final-sentence rows back in (`link_end_of_reasoning_activations.py`) removes the large,
systematic differences; this removes the small residual ones, which come from cutoffs whose
`model_action` was null or unparseable in one arm's rollout but not another's (2-4 rows per
arm) and from tensors genuinely absent from a tree (3).

The result is stronger than equal row counts: the arms are paired row for row on
`(name, step, cut_sentence_idx)`, so an eval-720 half holds the same sentences in every arm and
arm-to-arm differences can be read as paired.

    python scripts/intersect_belief_arms.py OUT_SUFFIX ARM_DIR [ARM_DIR ...]

writes `{ARM_DIR}{OUT_SUFFIX}/manifest.json` for each input.
"""

import argparse
import json
from pathlib import Path

ENTRY_KEYS = ("samples", "trajectories")


def entries_key(manifest: dict) -> str:
    """Which manifest key holds the per-sample entries."""
    for key in ENTRY_KEYS:
        if key in manifest:
            return key
    raise ValueError(f"manifest has none of {ENTRY_KEYS}")


def sentence_keys(samples: list[dict]) -> set[tuple]:
    """The `(name, step, cut_sentence_idx)` triples an arm covers.

    **`cut_sentence_idx`, not `sentence_idx`.** The latter is the cutoff's ordinal in its
    rollout's eval list, so it counts cutoffs rather than sentences and differs between arms
    for the same sentence -- pairing on it would intersect two arms to almost nothing and look
    like a data problem rather than a field mix-up.

    Rows without one cannot be paired across arms and are excluded rather than silently kept:
    a row no other arm can match is exactly the asymmetry this removes.

    >>> sorted(sentence_keys([{"name": "a", "step": 0, "cut_sentence_idx": 1},
    ...                       {"name": "a", "step": 0, "cut_sentence_idx": 2},
    ...                       {"name": "a", "step": 0}]))
    [('a', 0, 1), ('a', 0, 2)]
    """
    return {(s["name"], s["step"], s["cut_sentence_idx"]) for s in samples if s.get("cut_sentence_idx") is not None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_suffix", help="appended to each input dir name, e.g. '_eq'")
    ap.add_argument("arm_dirs", type=Path, nargs="+", help="prepared dirs holding the relabelled arms")
    args = ap.parse_args()

    if len(args.arm_dirs) < 2:
        raise SystemExit("intersecting needs at least two arms")

    loaded: list[tuple[Path, dict, str, list[dict]]] = []
    for d in args.arm_dirs:
        m = json.loads((d / "manifest.json").read_text())
        k = entries_key(m)
        loaded.append((d, m, k, m[k]))

    per_arm = [sentence_keys(rows) for _, _, _, rows in loaded]
    common = set.intersection(*per_arm)
    print(f"common (trajectory, step, sentence) triples: {len(common)}")
    print(f"common trajectories: {len({k[0] for k in common})}")

    for (d, m, k, rows), keys in zip(loaded, per_arm, strict=True):
        kept = [s for s in rows if (s["name"], s["step"], s.get("cut_sentence_idx")) in common]
        out_dir = d.with_name(d.name + args.out_suffix)
        out = dict(m)
        out[k] = kept
        out["intersected_with"] = {
            "arms": [str(p.resolve()) for p in args.arm_dirs],
            "n_in": len(rows),
            "n_out": len(kept),
            "n_common_sentences": len(common),
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "manifest.json").write_text(json.dumps(out))
        print(f"  {d.name:48s} {len(rows):6d} -> {len(kept):6d} (dropped {len(keys - common):4d}) -> {out_dir.name}")

    sizes = {
        len([s for s in rows if (s["name"], s["step"], s.get("cut_sentence_idx")) in common])
        for _, _, _, rows in loaded
    }
    if len(sizes) != 1:
        raise SystemExit(f"arms still differ in size after intersecting: {sorted(sizes)}")
    print(f"all arms now hold {sizes.pop()} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
