"""Re-run gpt-oss next-action inference from pre-computed reasoning chains.

For each step in a trajectory JSON file, this rebuilds the harmony-format prompt
token-faithfully (system+user prompt, grid state, and the model's *own* previously
computed analysis/reasoning) up to the point just before the final action value, then
asks the model to emit only the action (LEFT/RIGHT/UP/DOWN) and compares it against the
ground-truth ``agent_action``.

This mirrors ``telos_interp.commands.gather_activations``: same transformers model
loading and the same prefix+grid+suffix+output token concatenation, so no extra
dependencies (no ``openai_harmony``) are required.

Run on the GPU host, e.g.:
    python scripts/inference_oss/run_inference.py \
        scripts/inference_oss/together_ai_openai_gpt-oss-20b_size11_comp1.0_987.json
"""

import argparse
import json
import re
from glob import glob
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from telos_interp.commands.gather_activations.gather_activations_fn import _resolve_torch_dtype

DEFAULT_TRAJECTORY = str(Path(__file__).with_name("together_ai_openai_gpt-oss-20b_size11_comp1.0_987.json"))
DEFAULT_OUTPUT = str(Path(__file__).with_name("inference_results.json"))

# Harmony stop tokens (present in the gpt-oss tokenizer vocabulary).
HARMONY_STOP_TOKENS = ("<|return|>", "<|end|>", "<|call|>")

ACTION_RE = re.compile(r"\b(LEFT|RIGHT|UP|DOWN)\b")


def expand_paths(patterns: list[str]) -> list[str]:
    """Expand patterns into a sorted list of trajectory JSON files.

    Each pattern may be a single ``.json`` file, a glob (``**`` supported), or a
    directory. Directories (and directories matched by a glob) are searched
    recursively for ``*.json``, so the ``<root>/size*/*.json`` trajectory layout
    works whether you pass the root, a ``size*`` folder, or an explicit glob.
    """
    all_paths: list[str] = []
    for pattern in patterns:
        p = Path(pattern)
        if p.is_dir():
            all_paths.extend(str(f) for f in p.rglob("*.json"))
            continue
        matched = glob(pattern, recursive=True)
        if matched:
            for m in matched:
                mp = Path(m)
                if mp.is_dir():
                    all_paths.extend(str(f) for f in mp.rglob("*.json"))
                else:
                    all_paths.append(m)
        elif p.exists():
            all_paths.append(pattern)
    if not all_paths:
        raise ValueError(f"No files found matching patterns: {patterns}")
    return sorted(set(all_paths))


def is_trajectory(data: object) -> bool:
    """Whether a parsed JSON object looks like a trajectory file we can process."""
    return isinstance(data, dict) and {"prompt", "steps", "model_params"} <= data.keys()


def find_action_cut(output_tokens: list[dict]) -> int | None:
    """Return the index of the first final-channel action token in ``output_tokens``.

    Everything before this index is the reasoning plus the final-channel header
    (``...<|channel|>final<|message|>{\\n  "action": "``), which is exactly the prompt
    we want to feed the model. Returns ``None`` if no such token exists.
    """
    for i, token in enumerate(output_tokens):
        groups = set(token.get("token_groups", []))
        if {"final", "action"} <= groups:
            return i
    return None


def build_prompt_ids(trajectory: dict, step: dict) -> list[int] | None:
    """Reconstruct the prompt token ids up to (but excluding) the action value."""
    cut = find_action_cut(step["output_tokens"])
    if cut is None:
        return None

    prefix = [t["token_id"] for t in trajectory["prompt"]["prompt_prefix_tokens"]]
    grid = [t["token_id"] for t in step["grid_state_tokens"]]
    suffix = [t["token_id"] for t in trajectory["prompt"]["prompt_suffix_tokens"]]
    reasoning = [t["token_id"] for t in step["output_tokens"][:cut]]
    return prefix + grid + suffix + reasoning


def resolve_stop_ids(tokenizer) -> list[int]:
    """Resolve harmony stop tokens to ids, dropping any that map to the unk id."""
    unk_id = tokenizer.unk_token_id
    stop_ids = []
    for tok in HARMONY_STOP_TOKENS:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != unk_id:
            stop_ids.append(tid)
    return stop_ids


def parse_action(text: str) -> str | None:
    """Extract the first LEFT/RIGHT/UP/DOWN action from generated text."""
    match = ACTION_RE.search(text.upper())
    return match.group(1) if match else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trajectory_paths", nargs="*", default=[DEFAULT_TRAJECTORY], help="Trajectory JSON file(s); glob patterns supported.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to write the results JSON.")
    parser.add_argument("--max-new-tokens", type=int, default=8, help="Max tokens to generate for the action value.")
    parser.add_argument("--device-map", default="auto", help="device_map for model loading.")
    parser.add_argument("--torch-dtype", default="auto", help="Torch dtype: auto, bfloat16, or float16.")
    parser.add_argument("--dry-run", action="store_true", help="Decode and print the first step's prompt, then exit (no model load).")
    args = parser.parse_args()

    candidate_paths = expand_paths(args.trajectory_paths)

    # Keep only files that actually look like trajectories (recursive globbing of a
    # directory may turn up results files or other JSON).
    paths: list[str] = []
    first_traj: dict | None = None
    for path in candidate_paths:
        with open(path) as f:
            data = json.load(f)
        if is_trajectory(data):
            paths.append(path)
            if first_traj is None:
                first_traj = data
        else:
            print(f"  Skipping non-trajectory JSON: {path}")

    if not paths:
        raise ValueError(f"No valid trajectory files found in: {args.trajectory_paths}")

    print(f"Found {len(paths)} trajectory file(s) to process")
    model_id = first_traj["model_params"]["model_id"]

    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if args.dry_run:
        step = first_traj["steps"][0]
        prompt_ids = build_prompt_ids(first_traj, step)
        decoded = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        print(f"\n[DRY RUN] Reconstructed prompt for step {step['step_id']} ({len(prompt_ids)} tokens):\n{decoded}")
        print(f"\n[DRY RUN] Ground truth action: {step['agent_action']}")
        return

    print(f"Loading model: {model_id}")
    resolved_dtype = _resolve_torch_dtype(args.torch_dtype, model_id)
    dtype = resolved_dtype if resolved_dtype is not None else "auto"
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=args.device_map, dtype=dtype)
    model.eval()
    model_device = next(model.parameters()).device

    stop_ids = resolve_stop_ids(tokenizer)
    print(f"Harmony stop token ids: {stop_ids}")

    results: list[dict] = []
    overall_correct = 0
    overall_total = 0

    for traj_path in paths:
        with open(traj_path) as f:
            trajectory = json.load(f)

        model_params = trajectory["model_params"]
        seed = model_params.get("seed", 0)
        temperature = model_params.get("temperature", 1.0)
        top_p = model_params.get("top_p", 1.0)

        file_stem = Path(traj_path).stem
        file_correct = 0
        file_total = 0

        for step in trajectory["steps"]:
            prompt_ids = build_prompt_ids(trajectory, step)
            if prompt_ids is None:
                print(f"  WARNING: no final-channel action token in {file_stem} step {step['step_id']}; skipping")
                continue

            input_ids = torch.tensor([prompt_ids], device=model_device)
            torch.manual_seed(seed)
            with torch.no_grad():
                generated = model.generate(
                    input_ids,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_id=stop_ids or None,
                    pad_token_id=tokenizer.eos_token_id,
                )

            new_tokens = generated[0, input_ids.shape[1]:]
            raw_output = tokenizer.decode(new_tokens, skip_special_tokens=False)
            model_action = parse_action(raw_output)
            ground_truth = step["agent_action"]
            correct = model_action == ground_truth

            file_total += 1
            file_correct += int(correct)
            results.append({
                "file": file_stem,
                "step_id": step["step_id"],
                "model_action": model_action,
                "ground_truth": ground_truth,
                "correct": correct,
                "raw_output": raw_output,
            })

        overall_correct += file_correct
        overall_total += file_total
        acc = file_correct / file_total if file_total else 0.0
        print(f"  {file_stem}: {file_correct}/{file_total} correct ({acc:.1%})")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    overall_acc = overall_correct / overall_total if overall_total else 0.0
    print(f"\nOverall: {overall_correct}/{overall_total} correct ({overall_acc:.1%})")
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
