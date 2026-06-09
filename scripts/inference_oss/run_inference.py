"""Reasoning-theatre inference: gpt-oss next-action at each reasoning sentence boundary.

For each step in a trajectory JSON file, this truncates the model's *own* previously
computed analysis/reasoning at every sentence boundary, appends the fixed final-channel
answer prefix (``<|end|>...{\\n  "action": "``), and asks the model to emit the action
(LEFT/RIGHT/UP/DOWN). Comparing the action across sentence cutoffs against the
ground-truth ``agent_action`` shows *when* during reasoning the model commits to the
correct move. The last cutoff uses the full reasoning, reproducing the plain
full-reasoning inference.

This mirrors ``telos_interp.commands.gather_activations``: same transformers model
loading, the same prefix+grid+suffix+output token concatenation, and the same
sentence-end detector (``get_indices_for_eos_tokens``), so no extra dependencies (no
``openai_harmony``) are required.

Run on the GPU host, e.g.:
    python scripts/inference_oss/run_inference.py \
        --trajectory-paths scripts/inference_oss/together_ai_openai_gpt-oss-20b_size11_comp1.0_987.json
"""

import argparse
import json
import re
from glob import glob
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from telos_interp.commands.gather_activations.gather_activations_fn import _resolve_torch_dtype
from telos_interp.commands.gather_activations.gather_activations_utils import get_indices_for_eos_tokens

DEFAULT_TRAJECTORY = str(Path(__file__).with_name("together_ai_openai_gpt-oss-20b_size11_comp1.0_987.json"))
DEFAULT_OUTPUT_DIR = str(Path(__file__).with_name("inference_results"))

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
    (``...<|channel|>final<|message|>{\\n  "action": "``). Returns ``None`` if no such
    token exists.
    """
    for i, token in enumerate(output_tokens):
        if {"final", "action"} <= set(token.get("token_groups", [])):
            return i
    return None


def analysis_positions(output_tokens: list[dict]) -> list[int]:
    """Indices of the analysis/reasoning tokens in ``output_tokens``."""
    return [i for i, t in enumerate(output_tokens) if "analysis" in t.get("token_groups", [])]


def get_final_prefix_ids(output_tokens: list[dict]) -> list[int] | None:
    """Token ids of the fixed ``<|end|>...{\\n  "action": "`` slice.

    This is the verbatim sequence between the last reasoning token and the action value
    (``output_tokens[last_analysis + 1 : action_cut]``), so it is lifted from the data
    without re-tokenizing. Returns ``None`` if reasoning or the action value is missing.
    """
    cut = find_action_cut(output_tokens)
    ana = analysis_positions(output_tokens)
    if cut is None or not ana:
        return None
    return [t["token_id"] for t in output_tokens[max(ana) + 1 : cut]]


def reasoning_eos_positions(output_tokens: list[dict]) -> list[int]:
    """Positions of reasoning sentence-ends, via gather_activations' EOS detector.

    Restricted to the analysis region. Always includes a no-reasoning cutoff (the
    analysis-header ``<|message|>``, just before the first reasoning token) as the first
    position, so sentence_idx 0 is inference with zero reasoning. The end-of-reasoning
    position is always included as the final cutoff even if it does not end in sentence
    punctuation.
    """
    ana = analysis_positions(output_tokens)
    if not ana:
        return []
    last_analysis = max(ana)
    ana_set = set(ana)
    positions = [p for p in get_indices_for_eos_tokens(output_tokens) if p in ana_set]
    if last_analysis not in positions:
        positions.append(last_analysis)
    no_reasoning = min(ana) - 1  # analysis header <|message|>; keeps empty analysis channel
    if no_reasoning >= 0:
        positions.append(no_reasoning)
    return sorted(set(positions))


def build_prompt_ids_at(trajectory: dict, step: dict, eos_pos: int, final_prefix: list[int]) -> list[int]:
    """Prompt ids = prefix + grid + suffix + reasoning[:eos_pos+1] + final-channel prefix.

    Keeping ``output_tokens[:eos_pos + 1]`` retains the analysis header
    (``<|channel|>analysis<|message|>``) plus reasoning through the sentence end;
    appending ``final_prefix`` closes the analysis message and primes the final answer.
    """
    prefix = [t["token_id"] for t in trajectory["prompt"]["prompt_prefix_tokens"]]
    grid = [t["token_id"] for t in step["grid_state_tokens"]]
    suffix = [t["token_id"] for t in trajectory["prompt"]["prompt_suffix_tokens"]]
    reasoning = [t["token_id"] for t in step["output_tokens"][: eos_pos + 1]]
    return prefix + grid + suffix + reasoning + final_prefix


def commitment_metrics(corrects: list[bool]) -> tuple[int | None, int | None]:
    """Return (first_correct_idx, convinced_idx) from ordered per-sentence correctness.

    - first_correct: smallest index that is correct (None if never correct).
    - convinced: smallest index from which *all* later evals (incl. the last) are correct
      (None if the final eval is wrong).
    """
    first_correct = next((i for i, c in enumerate(corrects) if c), None)
    convinced = None
    if corrects and corrects[-1]:
        convinced = 0
        for i in range(len(corrects) - 1, -1, -1):
            if not corrects[i]:
                convinced = i + 1
                break
    return first_correct, convinced


def _fraction(idx: int | None, n: int) -> float | None:
    """Normalize a sentence index to [0, 1] over n sentences (None passes through)."""
    if idx is None:
        return None
    return idx / (n - 1) if n > 1 else 0.0


def _mean_or_none(values: list[float | int | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


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


def evaluate_step(
    trajectory: dict,
    step: dict,
    *,
    model,
    tokenizer,
    model_device,
    stop_ids: list[int],
    seed: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> dict | None:
    """Re-ask for the action at every reasoning sentence boundary of a single step.

    Returns a step record (with per-sentence evals and commitment metrics), or ``None``
    if the step has no reconstructable reasoning/action.
    """
    output_tokens = step["output_tokens"]
    final_prefix = get_final_prefix_ids(output_tokens)
    eos_positions = reasoning_eos_positions(output_tokens)
    if final_prefix is None or not eos_positions:
        return None

    ground_truth = step["agent_action"]
    n_sentences = len(eos_positions)
    sentence_evals: list[dict] = []
    corrects: list[bool] = []

    for sentence_idx, eos_pos in enumerate(eos_positions):
        prompt_ids = build_prompt_ids_at(trajectory, step, eos_pos, final_prefix)
        input_ids = torch.tensor([prompt_ids], device=model_device)
        torch.manual_seed(seed)
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                eos_token_id=stop_ids or None,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = generated[0, input_ids.shape[1]:]
        raw_output = tokenizer.decode(new_tokens, skip_special_tokens=False)
        model_action = parse_action(raw_output)
        correct = model_action == ground_truth
        corrects.append(correct)
        sentence_evals.append({
            "sentence_idx": sentence_idx,
            "eos_token_pos": eos_pos,
            "n_prompt_tokens": len(prompt_ids),
            "model_action": model_action,
            "correct": correct,
            "raw_output": raw_output,
        })

    first_correct, convinced = commitment_metrics(corrects)
    return {
        "step_id": step["step_id"],
        "ground_truth": ground_truth,
        "n_reasoning_sentences": n_sentences,
        "first_correct_sentence_idx": first_correct,
        "convinced_sentence_idx": convinced,
        "first_correct_fraction": _fraction(first_correct, n_sentences),
        "convinced_fraction": _fraction(convinced, n_sentences),
        "sentence_evals": sentence_evals,
    }


def summarize_file(file_stem: str, step_records: list[dict]) -> dict:
    """Aggregate per-step records into a file-level summary."""
    n_evals = sum(len(s["sentence_evals"]) for s in step_records)
    n_correct = sum(e["correct"] for s in step_records for e in s["sentence_evals"])
    n_final_correct = sum(1 for s in step_records if s["sentence_evals"][-1]["correct"])

    if len(step_records) == 1:
        # Single-step files: report that step's values verbatim (no averaging).
        s = step_records[0]
        first_correct = s["first_correct_sentence_idx"]
        convinced = s["convinced_sentence_idx"]
        first_correct_frac = s["first_correct_fraction"]
        convinced_frac = s["convinced_fraction"]
    else:
        first_correct = _mean_or_none([s["first_correct_sentence_idx"] for s in step_records])
        convinced = _mean_or_none([s["convinced_sentence_idx"] for s in step_records])
        first_correct_frac = _mean_or_none([s["first_correct_fraction"] for s in step_records])
        convinced_frac = _mean_or_none([s["convinced_fraction"] for s in step_records])

    return {
        "file": file_stem,
        "n_steps": len(step_records),
        "n_evals": n_evals,
        "sentence_accuracy": n_correct / n_evals if n_evals else 0.0,
        "final_sentence_accuracy": n_final_correct / len(step_records) if step_records else 0.0,
        "first_correct_sentence_idx": first_correct,
        "convinced_sentence_idx": convinced,
        "first_correct_fraction": first_correct_frac,
        "convinced_fraction": convinced_frac,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trajectory-paths", nargs="+", default=[DEFAULT_TRAJECTORY], help="Trajectory JSON file(s), directory, or glob pattern(s).")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory to write per-file results JSON into.")
    parser.add_argument("--max-new-tokens", type=int, default=8, help="Max tokens to generate for the action value.")
    parser.add_argument("--device-map", default="auto", help="device_map for model loading.")
    parser.add_argument("--torch-dtype", default="auto", help="Torch dtype: auto, bfloat16, or float16.")
    parser.add_argument("--dry-run", action="store_true", help="Print the first step's sentence cutoffs and prompt tails, then exit (no generation).")
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
        output_tokens = step["output_tokens"]
        final_prefix = get_final_prefix_ids(output_tokens)
        eos_positions = reasoning_eos_positions(output_tokens)
        print(f"\n[DRY RUN] Step {step['step_id']}: {len(eos_positions)} reasoning sentence cutoffs; "
              f"ground truth = {step['agent_action']}")
        print(f"[DRY RUN] final-channel prefix: {tokenizer.decode(final_prefix or [], skip_special_tokens=False)!r}")
        for sentence_idx, eos_pos in enumerate(eos_positions):
            prompt_ids = build_prompt_ids_at(first_traj, step, eos_pos, final_prefix)
            tail = tokenizer.decode(prompt_ids[-24:], skip_special_tokens=False)
            print(f"\n[DRY RUN] sentence {sentence_idx} (eos_pos={eos_pos}, {len(prompt_ids)} tokens) tail:\n...{tail}")
        return

    print(f"Loading model: {model_id}")
    resolved_dtype = _resolve_torch_dtype(args.torch_dtype, model_id)
    dtype = resolved_dtype if resolved_dtype is not None else "auto"
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=args.device_map, dtype=dtype)
    model.eval()
    model_device = next(model.parameters()).device

    stop_ids = resolve_stop_ids(tokenizer)
    print(f"Harmony stop token ids: {stop_ids}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_correct = 0
    overall_evals = 0
    overall_final_correct = 0
    overall_steps = 0
    overall_first_correct_fracs: list[float | None] = []
    overall_convinced_fracs: list[float | None] = []

    for traj_path in paths:
        with open(traj_path) as f:
            trajectory = json.load(f)

        model_params = trajectory["model_params"]
        seed = model_params.get("seed", 0)
        temperature = model_params.get("temperature", 1.0)
        top_p = model_params.get("top_p", 1.0)

        file_stem = Path(traj_path).stem
        step_records: list[dict] = []

        for step in trajectory["steps"]:
            record = evaluate_step(
                trajectory,
                step,
                model=model,
                tokenizer=tokenizer,
                model_device=model_device,
                stop_ids=stop_ids,
                seed=seed,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=args.max_new_tokens,
            )
            if record is None:
                print(f"  WARNING: no reconstructable reasoning/action in {file_stem} step {step['step_id']}; skipping")
                continue
            step_records.append(record)

        if not step_records:
            print(f"  WARNING: no usable steps in {file_stem}; no output written")
            continue

        summary = summarize_file(file_stem, step_records)
        out_path = output_dir / f"{file_stem}.json"
        with open(out_path, "w") as f:
            json.dump({"summary": summary, "steps": step_records}, f, indent=2)

        overall_correct += sum(e["correct"] for s in step_records for e in s["sentence_evals"])
        overall_evals += summary["n_evals"]
        overall_final_correct += sum(1 for s in step_records if s["sentence_evals"][-1]["correct"])
        overall_steps += len(step_records)
        overall_first_correct_fracs.extend(s["first_correct_fraction"] for s in step_records)
        overall_convinced_fracs.extend(s["convinced_fraction"] for s in step_records)

        print(
            f"  {file_stem}: sentence acc {summary['sentence_accuracy']:.1%}, "
            f"final acc {summary['final_sentence_accuracy']:.1%}, "
            f"first_correct={summary['first_correct_sentence_idx']}, "
            f"convinced={summary['convinced_sentence_idx']} -> {out_path}"
        )

    sentence_acc = overall_correct / overall_evals if overall_evals else 0.0
    final_acc = overall_final_correct / overall_steps if overall_steps else 0.0
    mean_first = _mean_or_none(overall_first_correct_fracs)
    mean_convinced = _mean_or_none(overall_convinced_fracs)
    print(
        f"\nOverall ({overall_steps} steps, {overall_evals} sentence-evals): "
        f"sentence acc {sentence_acc:.1%}, final acc {final_acc:.1%}, "
        f"mean first-correct fraction {mean_first}, mean convinced fraction {mean_convinced}"
    )
    print(f"Results written to {output_dir}/")


if __name__ == "__main__":
    main()
