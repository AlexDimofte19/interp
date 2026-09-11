#!/usr/bin/env python3
"""Check that the mass-era train/eval split is two separate, complete, disjoint datasets.

`build_mass_era_split.sh` writes two name lists and two symlink views. This is the contract
they have to satisfy, and it is worth a script rather than a comment because three of the
checks are the ones that would silently produce a wrong number rather than an error:

  * **the eval half is the SAME 720.** Every probe on disk was scored with `--eval-names`
    pinned to `next_action_mass_l15_eval_names.txt`. If the materialised list drifts from it
    by even one name, every cross-arm comparison quietly stops being like-for-like.
  * **the train half is the SAME 2,880.** Before this split existed, "train 2880" was
    recovered by reading the names out of one arm's prepared manifest. That definition is
    checked against the complement here, so materialising the set cannot redefine it.
  * **the views resolve.** A view is symlinks; a dangling one is an empty directory to
    `prepare_activations_for_probing`, which drops the trajectory and reports nothing.

It also checks what the whole point of the split is: that neither view can reach a trajectory
belonging to the other half, and that a view carries the gather's per-trajectory artifacts
(analysis CSV, direction-mass table + `.meta.json`, selection record) alongside the tensors --
a view that lost them would score against a different vocabulary without saying so.

Run it directly, or through `build_mass_era_split.sh`. Paths come from the environment so the
two agree by construction; exits non-zero on the first failed check.
"""

import json
import os
import pathlib
import sys

WS = pathlib.Path(os.environ.get("WS", "/workspace"))
PREPARED = pathlib.Path(os.environ.get("PREPARED", WS / "prepared"))
TRAIN_NAMES = pathlib.Path(os.environ.get("TRAIN_NAMES", WS / "splits/mass_train_2880.txt"))
EVAL_NAMES = pathlib.Path(os.environ.get("EVAL_NAMES", WS / "splits/mass_eval_720.txt"))
MASS_NAMES = pathlib.Path(os.environ.get("MASS_NAMES", WS / "reasoning_theatre/rollout_strategies/mass_l15_names.txt"))
TRAIN_VIEW = pathlib.Path(os.environ.get("TRAIN_VIEW", WS / "activations/mass_train2880_view"))
EVAL_VIEW = pathlib.Path(os.environ.get("EVAL_VIEW", WS / "activations/mass_eval720_view"))
SRC_TREE = pathlib.Path(os.environ.get("SRC_TREE", WS / "activations/jlens_mass_l15"))
HELDOUT_NAMES = pathlib.Path(os.environ.get("HELDOUT_NAMES", WS / "trajectories/heldout360_names.txt"))

# The pinned eval list every probe on disk was scored against, and the one arm's manifest the
# 2,880 used to be recovered from. Both are references, not outputs: this script never writes.
PINNED_EVAL = PREPARED / "next_action_mass_l15_eval_names.txt"
HISTORICAL_TRAIN_MANIFEST = PREPARED / "local_belief_p2_split_train/manifest.json"

# One gather writes all four beside the tensors; a view that lost them is not a usable dataset.
LENS_ARTIFACTS = ("_jlens_analysis.csv", "_jlens_direction_mass.csv", "_jlens_direction_mass.csv.meta.json")

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"      {'ok  ' if ok else 'FAIL'}  {label}{('  -- ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def names(path: pathlib.Path) -> set[str]:
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def view_names(view: pathlib.Path, kind: str) -> set[str]:
    root = view / kind
    if not root.is_dir():
        return set()
    suffix = ".json" if kind == "trajectories" else ""
    return {p.name[: -len(suffix)] if suffix else p.name for p in root.glob("size*/*")}


def main() -> int:
    train, ev, mass = names(TRAIN_NAMES), names(EVAL_NAMES), names(MASS_NAMES)

    print("    name lists")
    # The counts are DERIVED, not asserted: the contract is "train is the complement of eval",
    # and the canonical 2880/720 follows from it whenever the source really is the 3,600. Only
    # then is it worth stating, because only then does it mean the published partition.
    check("train and eval are disjoint", not (train & ev), f"{len(train & ev)} shared")
    check(
        f"train ({len(train)}) + eval ({len(ev)}) == the source set ({len(mass)})",
        train | ev == mass,
        f"symmetric diff {len((train | ev) ^ mass)}",
    )
    if len(mass) == 3600:
        check(
            "the canonical mass-era split is 2880 / 720",
            (len(train), len(ev)) == (2880, 720),
            f"got {len(train)} / {len(ev)}",
        )
    else:
        print(f"      --    source set is {len(mass)}, not the canonical 3600; skipping the 2880/720 check")

    if PINNED_EVAL.exists():
        pinned = names(PINNED_EVAL)
        check(
            "eval half is the SAME 720 every probe was scored on",
            ev == pinned,
            f"{len(ev ^ pinned)} names differ from {PINNED_EVAL.name}",
        )
    if HISTORICAL_TRAIN_MANIFEST.exists():
        hist = {s["name"] for s in json.loads(HISTORICAL_TRAIN_MANIFEST.read_text())["samples"]}
        check(
            "train half is the SAME 2880 the project already called train",
            train == hist,
            f"{len(train ^ hist)} names differ from {HISTORICAL_TRAIN_MANIFEST.parent.name}",
        )
    if HELDOUT_NAMES.exists():
        held = names(HELDOUT_NAMES)
        check("neither half touches heldout360", not ((train | ev) & held), f"{len((train | ev) & held)} shared")

    print("    views")
    for label, view, want in (("train", TRAIN_VIEW, train), ("eval", EVAL_VIEW, ev)):
        acts, trajs = view_names(view, "activations"), view_names(view, "trajectories")
        check(f"{label} view holds exactly its {len(want)} activation dirs", acts == want, f"got {len(acts)}")
        check(f"{label} view holds exactly its {len(want)} trajectory JSONs", trajs == want, f"got {len(trajs)}")

        dangling = [p for p in (view / "activations").glob("size*/*") if not p.resolve().exists()]
        dangling += [p for p in (view / "trajectories").glob("size*/*") if not p.resolve().exists()]
        check(f"{label} view has no dangling symlinks", not dangling, f"{len(dangling)} dangling")

        # Every link must land inside the source tree, and nowhere else -- that is what makes
        # the view a closed dataset rather than a filter someone can forget to apply.
        outside = [p for p in (view / "activations").glob("size*/*") if SRC_TREE.resolve() not in p.resolve().parents]
        check(f"{label} view points only into {SRC_TREE.name}", not outside, f"{len(outside)} outside")

    print("    per-trajectory lens artifacts reachable through the views")
    for label, view in (("train", TRAIN_VIEW), ("eval", EVAL_VIEW)):
        missing: list[str] = []
        for p in sorted((view / "activations").glob("size*/*"))[:200]:
            for suffix in LENS_ARTIFACTS:
                if not (p / f"{p.name}{suffix}").exists():
                    missing.append(f"{p.name}{suffix}")
        check(
            f"{label} view carries the lens CSV + mass table + meta (first 200)",
            not missing,
            f"{len(missing)} missing",
        )

    print()
    if failures:
        print(f"    {len(failures)} CHECK(S) FAILED: " + "; ".join(failures))
        return 1
    print("    all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
