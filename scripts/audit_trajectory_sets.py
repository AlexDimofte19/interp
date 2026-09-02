#!/usr/bin/env python3
"""Which trajectory set produced each artifact on disk?

There are three canonical sets and they are mutually disjoint, so any analysis is scored on
exactly one of them -- and which one decides what its numbers may be compared against:

  train 2,880    the mass tree minus the pinned eval names. DISTRIBUTION REFERENCES ONLY;
                 no accuracy number may come from here.
  eval 720       `next_action_mass_l15_eval_names.txt`. Every probe arm shares this exact
                 partition, which is what makes same-token comparisons legitimate.
  heldout 360    a disjoint TREE -- no shared draw with the 3,600 at all. The stronger
                 claim, for generalisation rather than a matched comparison.

This reads the `name` column of each artifact and reports what it actually is, rather than
trusting the write-up. An artifact that lands on "all 3600" is train and eval mixed: fine
for a rollout-vs-rollout analysis that fits no model, never comparable to a probe number.
"""

import argparse
import json
import pathlib
import sys

import pandas as pd

PREP = pathlib.Path("/workspace/prepared")
RT = pathlib.Path("/workspace/reasoning_theatre")
PROBES = pathlib.Path("/workspace/probes")


def load_reference(prep: pathlib.Path, heldout_lens: pathlib.Path) -> list[tuple[str, set]]:
    ev = set((prep / "next_action_mass_l15_eval_names.txt").read_text().split())
    tr = {s["name"] for s in json.loads((prep / "local_belief_p2_split_train/manifest.json").read_text())["samples"]}
    hd = {p.name for p in heldout_lens.glob("size*/*") if p.is_dir()}
    return [("train 2880", tr), ("eval 720", ev), ("heldout 360", hd), ("all 3600", tr | ev)]


def classify(names: set, ref: list[tuple[str, set]]) -> str:
    for label, s in ref:
        if s and names == s:
            return f"= {label}"
    parts = [f"{len(names & s)}/{len(names)} in {label}" for label, s in ref if s and names & s]
    return "; ".join(parts) if parts else "no overlap with any known set"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepared", type=pathlib.Path, default=PREP)
    ap.add_argument(
        "--heldout-lens", type=pathlib.Path, default=pathlib.Path("/workspace/activations/heldout360_lens")
    )
    args = ap.parse_args()

    ref = load_reference(args.prepared, args.heldout_lens)
    sizes = ", ".join(f"{lab}={len(s)}" for lab, s in ref)
    print(f"reference sets: {sizes}")
    tr, ev, hd = ref[0][1], ref[1][1], ref[2][1]
    print(f"train n eval = {len(tr & ev)},  3600 n heldout360 = {len(hd & (tr | ev))}  (both must be 0)")
    print("=" * 112)

    def from_csv(tag: str, path: pathlib.Path) -> None:
        if not path.exists():
            print(f"{tag:<58} MISSING {path}")
            return
        try:
            s = pd.read_csv(path, usecols=["name"], keep_default_na=False, na_values=[""], low_memory=False)
        except ValueError:
            print(f"{tag:<58} no 'name' column")
            return
        names = set(s["name"].dropna().unique())
        print(f"{tag:<58} n={len(names):<5} {classify(names, ref)}")

    def from_files(tag: str, folder: pathlib.Path, pattern: str = "*.json") -> None:
        """Trajectory names taken from filenames, for artifacts that are one file per trajectory."""
        if not folder.exists():
            print(f"{tag:<58} MISSING {folder}")
            return
        names = {f.stem for f in folder.glob(pattern)}
        print(f"{tag:<58} n={len(names):<5} {classify(names, ref)}")

    from_csv("loudness/  (entry 42, 22 figures)", RT / "loudness/per_token.csv")
    from_csv("probe_vs_rollout/  (entries 39/40/41, 15 figures)", RT / "probe_vs_rollout/per_token.csv")
    from_csv("probe_vs_rollout/per_token_probs.csv  (entry 41)", RT / "probe_vs_rollout/per_token_probs.csv")
    from_csv("probe_loudness/  (entry 46, 20 figures)", RT / "probe_loudness/per_token.csv")
    from_csv("probe_loudness_heldout360/  (entry 48, 20 figures)", RT / "probe_loudness_heldout360/per_token.csv")
    from_files(
        "rollout_strategies_heldout360/every_token  (entry 48)",
        RT / "rollout_strategies_heldout360/every_token",
    )
    from_csv("probes/heldout360_all_probes.csv  (entry 38)", PROBES / "heldout360_all_probes.csv")
    from_csv(
        "probes/.../heldout360_per_token.csv  (entry 37)", PROBES / "next_action_mass_l15/heldout360_per_token.csv"
    )
    for arm in ("jlens_argmax_per_sentence", "jlens_top_k_global"):
        from_csv(
            f"rollout_strategies/comparison {arm}  (entry 44d)",
            RT / f"rollout_strategies/comparison/loud_vs_sentence_end_{arm}.csv",
        )

    for d in sorted(args.prepared.glob("*_split_eval")) + sorted(args.prepared.glob("next_action_mass_l15_*_eval")):
        names = {s["name"] for s in json.loads((d / "manifest.json").read_text())["samples"]}
        print(f"{'prepared/' + d.name:<58} n={len(names):<5} {classify(names, ref)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
