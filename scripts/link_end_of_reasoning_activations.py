#!/usr/bin/env python3
"""Restore the final-sentence row that `_dedupe` removed from a per-sentence arm.

Every per-sentence truncation strategy emits one cutoff per sentence *and* an
`end_of_reasoning` bookend, then merges cutoffs that share a position. The bookend sits at
`eos[-1]`, the last token of the last span, so a pick landing there merges with it and the
survivor is tagged `end_of_reasoning` -- a kind the p1 datasets drop. The sentence therefore
leaves the dataset entirely, and it does so at a rate that is a property of the rule rather
than of the data: `jlens_argmax_per_sentence` 30 times in 3600, `logitlens` 67, a uniform
draw 776, and `eos` all 3600 (its pick *is* the last token, so this is a definition, not a
coincidence). Arms meant to differ only in which token inside a span they cut ended up
differing in how many spans they hold, which is the confound this repairs.

Only the LAST sentence can be affected. `reasoning_eos_positions` puts the analysis header
at `eos[0]` and `sentence_spans` starts sentence 1 at `eos[0] + 1`, so the `no_reasoning`
bookend lies outside every span and can never collide; a pick landing on some other
sentence's final token stays an interior cutoff. This therefore adds at most one row per
(trajectory, step) and cannot double-count.

The tensor needed is the step's last reasoning token, which is the same position for every
arm -- so it does not have to be gathered again. `--source-tree` (the eos tree) already holds
it, and this symlinks it into the arm's own tree, where a re-`prepare` picks it up as an
ordinary entry. The alternative is ~873 forward passes for tensors that already exist.

    python scripts/link_end_of_reasoning_activations.py ROLLOUT_DIR ARM_TREE --source-tree EOS_TREE

Dry-run by default; pass --apply to create the links.
"""

import argparse
import json
import os
from pathlib import Path

LAYER = 15
CATEGORY = "output"
KIND_END_OF_REASONING = "end_of_reasoning"


def steps_missing_final_sentence(rollout_doc: dict) -> dict[int, int]:
    """``{step_id: end_of_reasoning position}`` for steps whose final sentence has no interior cutoff.

    A step qualifies when its highest sentence is carried *only* by the `end_of_reasoning`
    cutoff -- i.e. the arm's pick for that sentence merged into the bookend. When the arm kept
    a distinct interior pick for the last sentence, the sentence is already represented.

    **Read `cut_sentence_idx`, never `sentence_idx`.** In the rollout schema `sentence_idx` is
    the cutoff's ORDINAL in the eval list (`no_reasoning` is 0, then 1, 2, ...), so the bookend
    is always the unique maximum and every step looks like it lost its final sentence.
    `cut_sentence_idx` is the sentence the cut actually lands in -- the bookend shares it with
    that sentence's own pick, which is exactly the distinction this function needs. The two
    coincide only for `eos`, which emits one cutoff per sentence and no extra bookend, so a
    field mix-up survives an eos spot-check and fails everywhere else.

    >>> doc = {"steps": [
    ...     {"step_id": 0, "sentence_evals": [                       # pick merged into the bookend
    ...         {"cut_sentence_idx": 1, "cutoff_kind": "loudest_in_sentence", "eos_token_pos": 4},
    ...         {"cut_sentence_idx": 2, "cutoff_kind": "end_of_reasoning", "eos_token_pos": 9}]},
    ...     {"step_id": 1, "sentence_evals": [                       # sentence 2 kept its own pick
    ...         {"cut_sentence_idx": 1, "cutoff_kind": "loudest_in_sentence", "eos_token_pos": 2},
    ...         {"cut_sentence_idx": 2, "cutoff_kind": "loudest_in_sentence", "eos_token_pos": 7},
    ...         {"cut_sentence_idx": 2, "cutoff_kind": "end_of_reasoning", "eos_token_pos": 9}]}]}
    >>> steps_missing_final_sentence(doc)
    {0: 9}
    """
    out: dict[int, int] = {}
    for step in rollout_doc["steps"]:
        evals = [ev for ev in step["sentence_evals"] if ev.get("cut_sentence_idx") is not None]
        if not evals:
            continue
        last = max(ev["cut_sentence_idx"] for ev in evals)
        in_last = [ev for ev in evals if ev["cut_sentence_idx"] == last]
        if any(ev.get("cutoff_kind") != KIND_END_OF_REASONING for ev in in_last):
            continue  # the arm kept an interior pick for the final sentence
        for ev in in_last:
            if ev.get("eos_token_pos") is not None:
                out[step["step_id"]] = ev["eos_token_pos"]
    return out


def step_dir(tree: Path, size: str, name: str, model: str, step_id: int) -> Path:
    """`{tree}/size{N}/{name}/{model}/layer_15/step_{M}/output` -- the standard tree layout.

    Note the `size{N}` level sits ABOVE the trajectory name; a path built without it silently
    matches nothing rather than raising.
    """
    return tree / f"size{size}" / name / model / f"layer_{LAYER}" / f"step_{step_id}" / CATEGORY


def model_dir_name(tree: Path, size: str, name: str) -> str | None:
    """The single model directory under a trajectory, or None when the trajectory is absent."""
    base = tree / f"size{size}" / name
    if not base.is_dir():
        return None
    subs = [p.name for p in base.iterdir() if p.is_dir()]
    return subs[0] if len(subs) == 1 else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rollout_dir", type=Path, help="the arm's run_inference.py output directory")
    ap.add_argument("arm_tree", type=Path, help="the arm's activation tree, linked into in place")
    ap.add_argument(
        "--source-tree",
        type=Path,
        required=True,
        help="tree holding the end_of_reasoning tensors (the eos tree). The position is the "
        "step's last reasoning token and is strategy-independent, so any tree that gathered "
        "it holds the same tensor.",
    )
    ap.add_argument("--names-file", type=Path, default=None, help="restrict to these trajectory names")
    ap.add_argument("--apply", action="store_true", help="create the links (default: report only)")
    args = ap.parse_args()

    keep = None
    if args.names_file is not None:
        keep = {ln.strip() for ln in args.names_file.read_text().split() if ln.strip()}

    files = sorted(p for p in args.rollout_dir.rglob("*.json") if p.parent.name != "logs")
    if keep is not None:
        files = [p for p in files if p.stem in keep]
    print(f"{len(files)} rollout file(s) under {args.rollout_dir}")

    n_steps = n_linked = n_present = n_no_source = n_no_model = 0
    missing_examples: list[str] = []
    for rf in files:
        name = rf.stem
        size = name.split("_size")[1].split("_")[0]
        wanted = steps_missing_final_sentence(json.loads(rf.read_text()))
        if not wanted:
            continue
        model = model_dir_name(args.source_tree, size, name)
        if model is None:
            n_no_model += len(wanted)
            continue
        for step_id, pos in wanted.items():
            n_steps += 1
            src = step_dir(args.source_tree, size, name, model, step_id) / f"{pos}.pt"
            dst = step_dir(args.arm_tree, size, name, model, step_id) / f"{pos}.pt"
            if dst.exists():
                n_present += 1
                continue
            if not src.exists():
                n_no_source += 1
                if len(missing_examples) < 5:
                    missing_examples.append(str(src))
                continue
            if args.apply:
                dst.parent.mkdir(parents=True, exist_ok=True)
                # Relative link so the tree stays movable, matching the manifest contract that
                # a prepared dataset's activations_root can be relocated.
                dst.symlink_to(os.path.relpath(src, dst.parent))
            n_linked += 1

    verb = "linked" if args.apply else "would link"
    print(f"final sentences with no interior cutoff: {n_steps}")
    print(f"  {verb}: {n_linked}")
    print(f"  already present: {n_present}")
    if n_no_source:
        print(f"  MISSING from --source-tree: {n_no_source}")
        for example in missing_examples:
            print(f"    {example}")
    if n_no_model:
        print(f"  trajectory absent from --source-tree: {n_no_model}")
    if not args.apply and n_linked:
        print("\nDry run. Re-run with --apply to create the links.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
