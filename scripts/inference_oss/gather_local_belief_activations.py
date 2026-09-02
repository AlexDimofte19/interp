"""Gather layer-15 residual streams for the per-sentence-loudest reasoning tokens.

Probe-1 half of the "local belief" experiment. The `jlens_argmax_per_sentence`
rollout (`run_inference.py --strategy jlens_argmax_per_sentence`) cut each
reasoning sentence at its LOUDEST token and recorded what the model then answers.
Those cut positions are one per sentence (`cutoff_kind == "loudest_in_sentence"`),
~22 per trajectory, and the pruned `/workspace/activations/jlens_mass_l15` tree
only has `.pt` files for ~36% of them (its intersection with the global top-20).

This walks the rollout output, and for every `loudest_in_sentence` cutoff does one
full forward pass per trajectory (prefix + grid + suffix + the WHOLE reasoning
chain, exactly as `gather_activations` builds it) and saves the layer-15
residual stream at those token positions, in the standard activation-tree layout:

    {out}/{stem}/{model}/layer_15/step_{step_id}/output/{eos_token_pos}.pt

`eos_token_pos` indexes `step["output_tokens"]` (== `token_id` in a prepared
manifest, == `token_idx` in a `_jlens_selection.json` pick), so the tree drops
straight into `prepare_activations_for_probing --token-selection all`.

The extraction internals (`build_truncated_input`, `extract_activations_single_pass`,
`save_activations_to_files`) are imported from the `gather_activations` command, so
the tensors are byte-identical to what that command would write for the same
positions -- an unbatched, single-sequence forward pass, no bf16 packing.

Run from the repo so telos_interp imports:
    uv run --project . --extra gpu python scripts/inference_oss/gather_local_belief_activations.py

GPU host only. One forward pass per trajectory; ~3600 passes.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from telos_interp.commands.gather_activations.gather_activations_fn import _resolve_torch_dtype
from telos_interp.commands.gather_activations.gather_activations_utils import (
    build_truncated_input,
    extract_activations_single_pass,
    sanitize_model_id,
    save_activations_to_files,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

LAYER = 15
CATEGORY = "output"
INTERIOR_KIND = "loudest_in_sentence"


def positions_by_step(rollout_doc: dict, kinds: set[str]) -> dict[int, list[int]]:
    """{step_id: sorted unique eos_token_pos} for cutoffs whose kind is in `kinds`."""
    out: dict[int, list[int]] = {}
    for step in rollout_doc["steps"]:
        pos = {
            ev["eos_token_pos"]
            for ev in step["sentence_evals"]
            if ev.get("cutoff_kind") in kinds and ev.get("eos_token_pos") is not None
        }
        if pos:
            out[step["step_id"]] = sorted(pos)
    return out


def output_start(trajectory: dict, step: dict) -> int:
    """Absolute index of output token 0 = len(prefix) + len(grid) + len(suffix)."""
    return (
        len(trajectory["prompt"]["prompt_prefix_tokens"])
        + len(step["grid_state_tokens"])
        + len(trajectory["prompt"]["prompt_suffix_tokens"])
    )


def already_done(output_base: Path, step_id: int, positions: list[int]) -> bool:
    d = output_base / f"layer_{LAYER}" / f"step_{step_id}" / CATEGORY
    return d.is_dir() and all((d / f"{p}.pt").exists() for p in positions)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--rollout-dir",
        type=Path,
        default=Path("/workspace/reasoning_theatre/rollout_strategies/jlens_argmax_per_sentence"),
        help="Directory of per-trajectory {stem}.json from run_inference.py --strategy jlens_argmax_per_sentence.",
    )
    ap.add_argument(
        "--trajectory-paths",
        type=Path,
        default=Path("/workspace/trajectories/reveng/trajectories_train_single_step"),
        help="Directory holding size{N}/{stem}.json trajectory files.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/workspace/activations/argmax_per_sentence_l15"),
    )
    ap.add_argument(
        "--names-file",
        type=Path,
        default=Path("/workspace/reasoning_theatre/rollout_strategies/mass_l15_names.txt"),
        help="Whitespace-separated trajectory stems to process (defaults to the rollout's own names file).",
    )
    ap.add_argument(
        "--include-endpoints",
        action="store_true",
        help="Also save the no_reasoning / end_of_reasoning cutoff positions (default: interior only).",
    )
    ap.add_argument("--torch-dtype", default="auto")
    ap.add_argument("--device-map", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="Process at most this many trajectories (0 = all).")
    ap.add_argument("--dry-run", action="store_true", help="Report coverage and the work plan, load no model.")
    args = ap.parse_args()

    kinds = {INTERIOR_KIND}
    if args.include_endpoints:
        kinds |= {"no_reasoning", "end_of_reasoning"}

    keep = set(args.names_file.read_text().split()) if args.names_file and args.names_file.exists() else None

    rollout_files = sorted(args.rollout_dir.glob("*.json"))
    if keep is not None:
        rollout_files = [p for p in rollout_files if p.stem in keep]
    if args.limit:
        rollout_files = rollout_files[: args.limit]
    print(f"{len(rollout_files)} rollout file(s) from {args.rollout_dir}")

    # Plan: (traj_path, output_base, {step_id: [positions]}), skipping finished work.
    plan: list[tuple[Path, Path, dict[int, list[int]]]] = []
    n_pos_total = 0
    n_pos_todo = 0
    n_missing_traj = 0
    model_id = None
    for rf in rollout_files:
        stem = rf.stem
        size = stem.split("_size")[1].split("_")[0]
        traj_path = args.trajectory_paths / f"size{size}" / f"{stem}.json"
        if not traj_path.exists():
            n_missing_traj += 1
            continue
        by_step = positions_by_step(json.loads(rf.read_text()), kinds)
        n_pos_total += sum(len(v) for v in by_step.values())
        with open(traj_path) as f:
            trajectory = json.load(f)
        if model_id is None:
            model_id = trajectory["model_params"]["model_id"]
        # size{N}/{stem}/{model}/... -- the multi-size layout prepare_activations_for_probing
        # and the jlens_mass_l15 tree use.
        output_base = args.out / f"size{size}" / stem / sanitize_model_id(model_id)
        todo = {sid: pos for sid, pos in by_step.items() if not already_done(output_base, sid, pos)}
        if todo:
            plan.append((traj_path, output_base, todo))
            n_pos_todo += sum(len(v) for v in todo.values())

    print(
        f"positions: {n_pos_total} total across {len(rollout_files)} trajectories; "
        f"{n_pos_todo} to gather in {len(plan)} unfinished trajectories "
        f"({n_missing_traj} trajectory JSONs missing)"
    )
    if args.dry_run or not plan:
        return 0

    print(f"Loading {model_id}")
    resolved = _resolve_torch_dtype(args.torch_dtype, model_id)
    _ = AutoTokenizer.from_pretrained(model_id)  # parity with gather_activations; not otherwise needed
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map=args.device_map, dtype=resolved if resolved is not None else "auto"
    )
    model.eval()

    t0 = time.time()
    done = 0
    nan_total = 0
    for traj_path, output_base, todo in plan:
        with open(traj_path) as f:
            trajectory = json.load(f)
        steps_by_id = {s["step_id"]: s for s in trajectory["steps"]}
        for step_id, positions in todo.items():
            step = steps_by_id.get(step_id)
            if step is None:
                print(f"  WARNING: {traj_path.stem} has no step_id {step_id}; skipping")
                continue
            base = output_start(trajectory, step)
            n_out = len(step["output_tokens"])
            abs_indices = [base + p for p in positions if 0 <= p < n_out]
            if not abs_indices:
                continue
            input_ids = build_truncated_input(trajectory, step, max(abs_indices))
            acts = extract_activations_single_pass(model, input_ids, abs_indices, [LAYER])
            remapped: dict[int, dict[int, torch.Tensor]] = {LAYER: {}}
            for p in positions:
                a = base + p
                if a in acts[LAYER]:
                    remapped[LAYER][p] = acts[LAYER][a]
            nan_total += save_activations_to_files(remapped, output_base, step_idx=step_id, category=CATEGORY)
        done += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if done % 50 == 0 or done == len(plan):
            el = time.time() - t0
            rate = done / el
            eta = (len(plan) - done) / rate if rate else 0
            print(
                f"  {done}/{len(plan)} trajectories  {el / 60:.1f} min  "
                f"{rate * 60:.1f}/min  ETA {eta / 60:.0f} min  nan={nan_total}",
                flush=True,
            )

    print(f"done: {done} trajectories, nan activations: {nan_total}, {(time.time() - t0) / 60:.1f} min")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
