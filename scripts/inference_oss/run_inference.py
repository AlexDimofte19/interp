"""Reasoning-theatre inference: gpt-oss next-action at cutoffs inside its own reasoning.

For each step in a trajectory JSON file, this truncates the model's *own* previously
computed analysis/reasoning at a series of cutoffs, appends the fixed final-channel
answer prefix (``<|end|>...{\\n  "action": "``), and asks the model to emit the action
(LEFT/RIGHT/UP/DOWN). Comparing the action across cutoffs against the ground-truth
``agent_action`` shows *when* during reasoning the model commits to the correct move.

WHERE the cutoffs go is ``--strategy``, and it is the only thing that differs between
runs (see ``truncation_strategies.py``):

  ``eos``                        every reasoning sentence end -- the original grid, and
                                 what every rollout currently on disk was measured on.
  ``jlens_argmax_per_sentence``  one cutoff per sentence, at its LOUDEST token.
  ``jlens_top_k_global``         the K loudest tokens of the whole reasoning chain.

Loudness is the layer-15 full-vocabulary direction mass of ICLR log entry 42, read from
the direction-mass tables beside the analysis CSVs; the two jlens strategies therefore
only cover trajectories that have one (``--lens-root``, ``--names-file``).

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
from telos_interp.commands.gather_activations.gather_activations_fn import _resolve_torch_dtype
from transformers import AutoModelForCausalLM, AutoTokenizer

try:  # run as a script: its own directory is on sys.path
    from truncation_strategies import (
        DEFAULT_LAYER,
        DEFAULT_LENS_ROOT,
        DEFAULT_TOP_K,
        STRATEGIES,
        Cutoff,
        LoudnessUnavailable,
        TruncationStrategy,
        analysis_positions,
        build_strategy,
        find_action_cut,
        get_final_prefix_ids,
        reasoning_eos_positions,
    )
except ImportError:  # imported as scripts.inference_oss.run_inference
    from scripts.inference_oss.truncation_strategies import (  # noqa: F401
        DEFAULT_LAYER,
        DEFAULT_LENS_ROOT,
        DEFAULT_TOP_K,
        STRATEGIES,
        Cutoff,
        LoudnessUnavailable,
        TruncationStrategy,
        analysis_positions,
        build_strategy,
        find_action_cut,
        get_final_prefix_ids,
        reasoning_eos_positions,
    )

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


def build_prompt_ids_at(trajectory: dict, step: dict, cut_pos: int, final_prefix: list[int]) -> list[int]:
    """Prompt ids = prefix + grid + suffix + reasoning[:cut_pos+1] + final-channel prefix.

    Keeping ``output_tokens[:cut_pos + 1]`` retains the analysis header
    (``<|channel|>analysis<|message|>``) plus reasoning through the cutoff; appending
    ``final_prefix`` closes the analysis message and primes the final answer. The cutoff
    need not be a sentence end -- the jlens strategies cut mid-sentence, which is the
    point of them -- and the truncation is by token, so nothing rewrites the text.
    """
    prefix = [t["token_id"] for t in trajectory["prompt"]["prompt_prefix_tokens"]]
    grid = [t["token_id"] for t in step["grid_state_tokens"]]
    suffix = [t["token_id"] for t in trajectory["prompt"]["prompt_suffix_tokens"]]
    reasoning = [t["token_id"] for t in step["output_tokens"][: cut_pos + 1]]
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


def _build_padded_batch(prompts: list[list[int]], pad_id: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pad a list of token-id lists into (input_ids, attention_mask) on ``device``.

    Left padding keeps each row's real last token at position -1, so the next-token
    logits line up across the batch regardless of differing prompt lengths.
    """
    max_len = max(len(p) for p in prompts)
    input_ids = torch.full((len(prompts), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(prompts), max_len), dtype=torch.long)
    for i, p in enumerate(prompts):
        input_ids[i, max_len - len(p) :] = torch.tensor(p, dtype=torch.long)
        attention_mask[i, max_len - len(p) :] = 1
    return input_ids.to(device), attention_mask.to(device)


def build_step_prompts(
    trajectory: dict, step: dict, strategy: TruncationStrategy, traj_name: str
) -> tuple[dict, list[list[int]]] | None:
    """Build one prompt per ``strategy`` cutoff for a single step.

    Returns ``(meta, prompts)`` where ``prompts`` is aligned with ``meta["cutoffs"]``, or
    ``None`` if the step has no reconstructable reasoning/action. ``meta`` carries the
    step identity needed to assemble the record later, so generation can be decoupled from
    step boundaries (see ``generate_actions``). ``meta["eos_positions"]`` is kept as the
    plain list of cut positions under its historical name, since downstream joins
    (``scripts/build_sentence_loudness.py``, ``scripts/build_probe_rollout_join.py``) key
    on ``eos_token_pos``.

    Propagates ``LoudnessUnavailable`` so the caller can skip the whole trajectory.
    """
    output_tokens = step["output_tokens"]
    final_prefix = get_final_prefix_ids(output_tokens)
    cutoffs = strategy.cutoffs(trajectory, step, traj_name)
    if final_prefix is None or not cutoffs:
        return None

    prompts = [build_prompt_ids_at(trajectory, step, c.pos, final_prefix) for c in cutoffs]
    meta = {
        "step_id": step["step_id"],
        "ground_truth": step["agent_action"],
        "cutoffs": cutoffs,
        "eos_positions": [c.pos for c in cutoffs],
    }
    return meta, prompts


def _length_sorted_batches(
    order: list[int],
    prompts: list[list[int]],
    batch_size: int,
    max_batch_tokens: int,
    max_attn_elems: int = 0,
) -> list[list[int]]:
    """Group the length-sorted ``order`` into batches bounded by rows, area AND attention memory.

    A batch is flushed when adding the next prompt would exceed any of three caps. Since
    ``order`` is ascending by length, each newly added prompt is the batch's longest, so its
    length sets the padded width for every row — adding it makes the batch ``(len(batch) + 1)``
    rows of width ``L``:

    * ``batch_size`` — the row cap;
    * ``max_batch_tokens`` — the padded-token AREA, ``rows * L``, which bounds everything whose
      cost is linear in tokens (the MoE forward, the KV cache);
    * ``max_attn_elems`` — ``rows * L**2``, which is what actually bounds EAGER ATTENTION.
      gpt-oss has no SDPA kernel and flash-attn is not installed here, so
      ``eager_attention_forward`` materializes a ``[rows, heads, L, L]`` score tensor:
      ``rows * 64 * L**2 * 2`` bytes at bf16, with two or three of them live at once. The area
      cap is LINEAR in ``L`` and does not bound that — 16 rows x 1587 tokens is an area of
      25,392, well inside a 49,152 budget, and a 4.80 GiB attention tensor that OOMs a 32 GB
      card. That is the failure this cap exists to prevent; ``0`` disables it.

    A single prompt larger than the whole budget still forms its own one-row batch (work is
    never dropped) — see ``generate_actions``, which splits a batch and retries if one OOMs
    anyway.
    """
    batches: list[list[int]] = []
    batch: list[int] = []
    for i in order:
        length = len(prompts[i])
        rows = len(batch) + 1
        too_big = (
            len(batch) >= batch_size
            or rows * length > max_batch_tokens
            or (max_attn_elems > 0 and rows * length * length > max_attn_elems)
        )
        if batch and too_big:
            batches.append(batch)
            batch = []
        batch.append(i)
    if batch:
        batches.append(batch)
    return batches


def generate_actions(
    prompts: list[list[int]],
    *,
    model,
    tokenizer,
    model_device,
    stop_ids: list[int],
    batch_size: int,
    max_batch_tokens: int,
    max_new_tokens: int,
    max_attn_elems: int = 0,
) -> list[dict]:
    """Greedily decode the action for a flat list of prompts; results align with ``prompts``.

    Prompts are left-padded into batches and decoded greedily, so each ``model.generate`` handles
    many cutoffs at once regardless of which step they came from. Prompts are processed in
    length-sorted order and grouped by ``_length_sorted_batches`` so each batch is length-uniform
    (minimizing padding) and bounded by a row cap (``batch_size``), a padded-token-area budget
    (``max_batch_tokens``) and an eager-attention budget (``max_attn_elems``, ``rows * L**2``) —
    the last two shrink the row count on long-sequence batches so memory stays bounded. Results
    are then scattered back to the original positions.

    Should a batch OOM anyway, it is halved and retried rather than allowed to kill the run: a
    single bad batch part-way through an arm would otherwise discard hours of finished windows.
    """
    results: list[dict | None] = [None] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    batches = _length_sorted_batches(order, prompts, batch_size, max_batch_tokens, max_attn_elems)

    print(
        f"    Generating actions for {len(prompts)} prompt(s) in {len(batches)} batch(es) "
        f"(<= {batch_size} rows, <= {max_batch_tokens} padded tokens, <= {max_attn_elems} attn elems each)"
    )

    for batch_no, idxs in enumerate(batches, start=1):
        seq_len = max(len(prompts[i]) for i in idxs)
        print(
            f"      batch {batch_no}/{len(batches)}: {len(idxs)} prompt(s), padded to {seq_len} tokens "
            f"({len(idxs) * seq_len} area, {len(idxs) * seq_len * seq_len} attn elems)"
        )
        _generate_batch_into(
            idxs,
            prompts,
            results,
            model=model,
            tokenizer=tokenizer,
            model_device=model_device,
            stop_ids=stop_ids,
            max_new_tokens=max_new_tokens,
        )

    return results  # type: ignore[return-value]  # every slot is filled above


def _generate_batch_into(
    idxs: list[int],
    prompts: list[list[int]],
    results: list[dict | None],
    *,
    model,
    tokenizer,
    model_device,
    stop_ids: list[int],
    max_new_tokens: int,
) -> None:
    """Decode one batch and write its results into ``results``; on OOM, halve and recurse.

    The caps in ``_length_sorted_batches`` are sized from a memory model, and a model is not a
    guarantee — fragmentation and a long tail of prompt lengths can still put a batch over.
    Halving is tried down to a single row, at which point the OOM is real and is raised.
    """
    chunk = [prompts[i] for i in idxs]
    # Bound to names up front: they are frame locals, so on an OOM they would keep this
    # batch's GPU blocks alive across the retry that happens inside this same frame.
    input_ids = attention_mask = generated = new_tokens = first_probs = None
    try:
        input_ids, attention_mask = _build_padded_batch(chunk, tokenizer.eos_token_id, model_device)
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                eos_token_id=stop_ids or None,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_logits=True,
            )
        new_tokens = generated.sequences[:, input_ids.shape[1] :]
        # The final-channel prefix primes `... "action": "`, so the first generated token
        # *is* the action value. Record the model's raw probability for that token (its
        # confidence in the answer it gave), matching the trajectory's `probabilities`.
        first_probs = torch.softmax(generated.logits[0].float(), dim=-1)
        for j, orig_i in enumerate(idxs):
            raw_output = tokenizer.decode(new_tokens[j], skip_special_tokens=False)
            first_token_id = int(new_tokens[j, 0])
            results[orig_i] = {
                "model_action": parse_action(raw_output),
                "answer_token": tokenizer.decode(new_tokens[j, :1], skip_special_tokens=False),
                "answer_prob": float(first_probs[j, first_token_id]),
                "raw_output": raw_output,
            }

        # Drop this batch's GPU tensors before the next (larger, length-sorted) batch so the
        # caching allocator can reuse the blocks instead of growing its reserved pool.
        del generated, first_probs, new_tokens, input_ids, attention_mask
    except torch.OutOfMemoryError:
        if len(idxs) == 1:
            raise
        # Drop the failed batch's references before empty_cache(), or the allocator cannot
        # reclaim the blocks and the halved retry OOMs too. Assigned-never-read on purpose.
        input_ids = attention_mask = generated = new_tokens = first_probs = None  # noqa: F841
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        half = len(idxs) // 2
        seq_len = max(len(prompts[i]) for i in idxs)
        print(
            f"      WARNING: OOM on {len(idxs)} row(s) x {seq_len} tokens; retrying as {half} + {len(idxs) - half}",
            flush=True,
        )
        for part in (idxs[:half], idxs[half:]):
            _generate_batch_into(
                part,
                prompts,
                results,
                model=model,
                tokenizer=tokenizer,
                model_device=model_device,
                stop_ids=stop_ids,
                max_new_tokens=max_new_tokens,
            )


def assemble_step_record(meta: dict, prompts: list[list[int]], results: list[dict]) -> dict:
    """Build a step record (per-cutoff evals + commitment metrics) from this step's slice.

    ``prompts`` and ``results`` are this step's cutoffs in ``meta["cutoffs"]`` order.

    The schema is the ``eos`` one for every strategy, so ``analysis.py`` and the join
    scripts read all three unchanged: ``sentence_evals`` is the ordered list of cutoffs,
    ``sentence_idx`` its running index, ``eos_token_pos`` the cut position in
    ``step["output_tokens"]``, and ``n_reasoning_sentences`` the number of cutoffs (a
    misnomer under ``jlens_top_k_global``; ``n_cutoffs`` is the same number under an
    honest name). What the strategy chose is added per eval: ``cutoff_kind``, the
    sentence the cut LANDS IN on the eos grid (``cut_sentence_idx``,
    ``pos_in_sentence``, ``sentence_len``) and its loudness (``dir_logmass``,
    ``dir_prob``), so a cutoff can be placed in sentence coordinates without redoing the
    join.
    """
    ground_truth = meta["ground_truth"]
    cutoffs: list[Cutoff] = meta["cutoffs"]
    n_cutoffs = len(cutoffs)

    sentence_evals: list[dict] = []
    corrects: list[bool] = []
    for sentence_idx, (cut, prompt, res) in enumerate(zip(cutoffs, prompts, results, strict=True)):
        correct = res["model_action"] == ground_truth
        corrects.append(correct)
        sentence_evals.append(
            {
                "sentence_idx": sentence_idx,
                "eos_token_pos": cut.pos,
                "cutoff_kind": cut.kind,
                "cut_sentence_idx": cut.sentence_idx,
                "pos_in_sentence": cut.pos_in_sentence,
                "sentence_len": cut.sentence_len,
                "dir_logmass": cut.logmass,
                "dir_prob": cut.prob_mass,
                "n_prompt_tokens": len(prompt),
                "model_action": res["model_action"],
                "correct": correct,
                "answer_token": res["answer_token"],
                "answer_prob": res["answer_prob"],
                "raw_output": res["raw_output"],
            }
        )

    first_correct, convinced = commitment_metrics(corrects)
    return {
        "step_id": meta["step_id"],
        "ground_truth": ground_truth,
        "n_reasoning_sentences": n_cutoffs,
        "n_cutoffs": n_cutoffs,
        "first_correct_sentence_idx": first_correct,
        "convinced_sentence_idx": convinced,
        "first_correct_fraction": _fraction(first_correct, n_cutoffs),
        "convinced_fraction": _fraction(convinced, n_cutoffs),
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
        "n_cutoffs": n_evals,
        "n_evals": n_evals,
        "sentence_accuracy": n_correct / n_evals if n_evals else 0.0,
        "final_sentence_accuracy": n_final_correct / len(step_records) if step_records else 0.0,
        "first_correct_sentence_idx": first_correct,
        "convinced_sentence_idx": convinced,
        "first_correct_fraction": first_correct_frac,
        "convinced_fraction": convinced_frac,
    }


def _flush_window(
    window_prompts: list[list[int]],
    window_files: list[dict],
    *,
    model,
    tokenizer,
    model_device,
    stop_ids: list[int],
    batch_size: int,
    max_batch_tokens: int,
    max_new_tokens: int,
    max_attn_elems: int,
    strategy: TruncationStrategy,
    overall: dict,
) -> None:
    """Generate over a whole window of prompts and write each file's results.

    ``window_prompts`` is the flat prompt list shared by every file in ``window_files``;
    each file's ``spans`` index into it. Batches inside ``generate_actions`` cross file and
    step boundaries freely, so a window only needs to hold whole files (never split one).
    Folds per-file metrics into the mutable ``overall`` accumulator.
    """
    if not window_files:
        return

    stems = ", ".join(fw["file_stem"] for fw in window_files)
    print(f"  Flushing window: {len(window_files)} file(s), {len(window_prompts)} prompt(s) [{stems}]")
    results = generate_actions(
        window_prompts,
        model=model,
        tokenizer=tokenizer,
        model_device=model_device,
        stop_ids=stop_ids,
        batch_size=batch_size,
        max_batch_tokens=max_batch_tokens,
        max_new_tokens=max_new_tokens,
        max_attn_elems=max_attn_elems,
    )

    # Return this window's reserved (but now unused) GPU blocks to the driver so the pool
    # doesn't accumulate across windows (mirrors gather_activations' per-trajectory cleanup).
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for fw in window_files:
        step_records = [assemble_step_record(meta, window_prompts[s:e], results[s:e]) for meta, s, e in fw["spans"]]
        summary = summarize_file(fw["file_stem"], step_records)
        # strategy.config() is read here, not at startup: for the jlens strategies it
        # only knows the mass table's vocabulary sidecar once a table has been loaded.
        with open(fw["out_path"], "w") as f:
            json.dump({"strategy": strategy.config(), "summary": summary, "steps": step_records}, f, indent=2)

        overall["correct"] += sum(e["correct"] for s in step_records for e in s["sentence_evals"])
        overall["evals"] += summary["n_evals"]
        overall["final_correct"] += sum(1 for s in step_records if s["sentence_evals"][-1]["correct"])
        overall["steps"] += len(step_records)
        overall["first_correct_fracs"].extend(s["first_correct_fraction"] for s in step_records)
        overall["convinced_fracs"].extend(s["convinced_fraction"] for s in step_records)

        print(
            f"  {fw['file_stem']}: sentence acc {summary['sentence_accuracy']:.1%}, "
            f"final acc {summary['final_sentence_accuracy']:.1%}, "
            f"first_correct={summary['first_correct_sentence_idx']}, "
            f"convinced={summary['convinced_sentence_idx']} -> {fw['out_path']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--trajectory-paths",
        nargs="+",
        default=[DEFAULT_TRAJECTORY],
        help="Trajectory JSON file(s), directory, or glob pattern(s).",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory to write per-file results JSON into."
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1,
        help="Max tokens to generate for the action value (the action is a single token, so 1 suffices).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Max rows per batched generate call (upper bound; long-sequence batches may use fewer rows due to --max-batch-tokens).",
    )
    parser.add_argument(
        "--max-batch-tokens",
        type=int,
        default=49152,
        help="Max padded token area (rows * padded_len) per batch. Caps memory on long-sequence batches by shrinking their row count. For eager attention memory scales ~rows*seq_len^2, so lower this if you still OOM.",
    )
    parser.add_argument(
        "--max-attn-elems",
        type=int,
        default=16_000_000,
        help=(
            "Max eager-attention elements (rows * padded_len^2) per batch. This is the cap that "
            "actually bounds memory here: gpt-oss falls back to eager attention, which "
            "materializes a [rows, heads, L, L] tensor (rows*64*L^2*2 bytes at bf16, 2-3 live at "
            "once), so --max-batch-tokens (linear in L) does not bound it -- 16 rows x 1587 "
            "tokens is a 25k area and a 4.80 GiB tensor. 0 disables. The default keeps a single "
            "score tensor near 1.9 GiB, which leaves 16 rows untouched below ~1000 tokens."
        ),
    )
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="Attention backend passed to from_pretrained (e.g. flex_attention, flash_attention_2, eager). Default: let transformers choose. gpt-oss has no SDPA, so without flash-attn it falls back to slow eager; flex_attention avoids the O(seq_len^2) memory.",
    )
    parser.add_argument(
        "--max-window-prompts",
        type=int,
        default=None,
        help="Accumulate prompts across trajectory files until this many, then generate+write as one window (batches cross file boundaries). Larger = better length-grouping but more RAM. Defaults to batch_size * 32.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip trajectory files whose output JSON already exists in --output-dir (resume a previous run).",
    )
    parser.add_argument(
        "--strategy",
        default="eos",
        choices=sorted(STRATEGIES),
        help="Where to place the truncation cutoffs (see truncation_strategies.py). "
        "eos: every reasoning sentence end. jlens_argmax_per_sentence: the loudest token of "
        "each sentence. jlens_top_k_global: the --top-k loudest tokens of the whole chain. "
        "every_token: no selection at all -- every reasoning token, or every --stride-th. "
        "recorded_selection: replay the --selection-arm picks of an existing gather's record.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="every_token only: keep every STRIDE-th reasoning token, counting from the first. "
        "1 is the dense grid; 2 halves the cost at the price of a label that is one token stale "
        "for the tokens it skips. Endpoints are kept whatever the stride.",
    )
    parser.add_argument(
        "--selection-arm",
        default="random",
        help="recorded_selection only: which arm of {stem}_jlens_selection.json to replay. "
        "'random' is the control whose seeded draw cannot be re-made once a tree is pruned.",
    )
    parser.add_argument(
        "--selection-root",
        default=None,
        help="recorded_selection only: tree holding the selection records (default: --lens-root, "
        "since one gather writes the record beside the mass table it ranked with).",
    )
    parser.add_argument(
        "--names-file",
        default=None,
        help="File of trajectory stems (whitespace-separated) to KEEP. The jlens strategies only "
        "cover trajectories with a direction-mass table, so this is how all three strategies are "
        "run over the same trajectory set.",
    )
    parser.add_argument(
        "--lens-root",
        default=str(DEFAULT_LENS_ROOT),
        help="Root of the gather tree holding the direction-mass tables (jlens strategies only).",
    )
    parser.add_argument("--lens", default="jlens", help="Which lens' mass table to read: jlens or logitlens.")
    parser.add_argument(
        "--loudness-layer",
        type=int,
        default=DEFAULT_LAYER,
        help="Mass-table layer whose direction mass defines loudness. Everything since log entry 36 is layer 15.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="jlens_top_k_global only: how many of the loudest reasoning tokens to cut at, per step.",
    )
    parser.add_argument(
        "--no-endpoint-cutoffs",
        action="store_true",
        help="jlens strategies only: drop the no-reasoning and full-reasoning cutoffs that are added "
        "so every strategy's first and last eval is the same prompt. Doing so makes final accuracy "
        "and the commitment indices incomparable with the eos arm.",
    )
    parser.add_argument("--device-map", default="auto", help="device_map for model loading.")
    parser.add_argument("--torch-dtype", default="auto", help="Torch dtype: auto, bfloat16, or float16.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the first step's sentence cutoffs and prompt tails, then exit (no generation).",
    )
    args = parser.parse_args()

    strategy = build_strategy(
        args.strategy,
        lens_root=Path(args.lens_root),
        lens=args.lens,
        layer=args.loudness_layer,
        top_k=args.top_k,
        include_endpoints=not args.no_endpoint_cutoffs,
        stride=args.stride,
        selection_arm=args.selection_arm,
        selection_root=Path(args.selection_root) if args.selection_root else None,
    )

    paths = expand_paths(args.trajectory_paths)
    if args.names_file:
        keep = set(Path(args.names_file).read_text().split())
        before = len(paths)
        paths = [p for p in paths if Path(p).stem in keep]
        print(f"--names-file {args.names_file}: {len(keep)} name(s) -> kept {len(paths)} of {before} file(s)")
    if not paths:
        raise ValueError(f"No valid trajectory files found in: {args.trajectory_paths}")
    with open(paths[0]) as f:
        first_traj = json.load(f)

    print(f"Found {len(paths)} trajectory file(s) to process")
    print(f"Truncation strategy: {args.strategy} {strategy.config()}")
    model_id = first_traj["model_params"]["model_id"]

    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if args.dry_run:
        step = first_traj["steps"][0]
        output_tokens = step["output_tokens"]
        final_prefix = get_final_prefix_ids(output_tokens)
        cutoffs = strategy.cutoffs(first_traj, step, Path(paths[0]).stem)
        eos_positions = reasoning_eos_positions(output_tokens)
        print(
            f"\n[DRY RUN] {Path(paths[0]).stem} step {step['step_id']}: {len(cutoffs)} cutoff(s) from "
            f"{args.strategy} over {max(len(eos_positions) - 1, 0)} reasoning sentence(s); "
            f"ground truth = {step['agent_action']}"
        )
        print(f"[DRY RUN] sentence ends at {eos_positions}")
        print(f"[DRY RUN] final-channel prefix: {tokenizer.decode(final_prefix or [], skip_special_tokens=False)!r}")
        for cut_idx, cut in enumerate(cutoffs):
            prompt_ids = build_prompt_ids_at(first_traj, step, cut.pos, final_prefix)
            tail = tokenizer.decode(prompt_ids[-24:], skip_special_tokens=False)
            mass = "-" if cut.prob_mass is None else f"{cut.prob_mass:.4f}"
            print(
                f"\n[DRY RUN] cutoff {cut_idx} pos={cut.pos} kind={cut.kind} "
                f"sentence={cut.sentence_idx} pos_in_sentence={cut.pos_in_sentence}/{cut.sentence_len} "
                f"dir_mass={mass} ({len(prompt_ids)} tokens) tail:\n...{tail}"
            )
        return

    print(f"Loading model: {model_id}")
    resolved_dtype = _resolve_torch_dtype(args.torch_dtype, model_id)
    dtype = resolved_dtype if resolved_dtype is not None else "auto"
    load_kwargs = {"device_map": args.device_map, "dtype": dtype}
    if args.attn_implementation:
        load_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()
    model_device = next(model.parameters()).device
    print(f"Attention implementation: {getattr(model.config, '_attn_implementation', 'unknown')}")

    stop_ids = resolve_stop_ids(tokenizer)
    print(f"Harmony stop token ids: {stop_ids}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overall = {
        "correct": 0,
        "evals": 0,
        "final_correct": 0,
        "steps": 0,
        "first_correct_fracs": [],
        "convinced_fracs": [],
    }
    max_window_prompts = args.max_window_prompts or args.batch_size * 32

    flush_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "model_device": model_device,
        "stop_ids": stop_ids,
        "batch_size": args.batch_size,
        "max_batch_tokens": args.max_batch_tokens,
        "max_attn_elems": args.max_attn_elems,
        "max_new_tokens": args.max_new_tokens,
        "strategy": strategy,
        "overall": overall,
    }

    # Accumulate whole files' cutoffs into a shared window, then flush (generate + write) once
    # it reaches max_window_prompts. Windows close only at file boundaries, so each file's
    # cutoffs stay contiguous; batches inside the flush still cross file/step boundaries.
    window_prompts: list[list[int]] = []
    window_files: list[dict] = []
    n_skipped_loudness = 0

    for traj_path in paths:
        file_stem = Path(traj_path).stem
        out_path = output_dir / f"{file_stem}.json"
        if args.skip_existing and out_path.exists():
            print(f"  Skipping {file_stem}: output already exists at {out_path}")
            continue

        with open(traj_path) as f:
            trajectory = json.load(f)

        print(f"  Loaded {file_stem}: {len(trajectory['steps'])} step(s)")

        # ``spans`` records each step's slice [start, end) into the shared window prompt list.
        window_start = len(window_prompts)
        spans: list[tuple[dict, int, int]] = []
        for step in trajectory["steps"]:
            try:
                built = build_step_prompts(trajectory, step, strategy, file_stem)
            except LoudnessUnavailable as exc:
                # No mass table for this trajectory: the jlens strategies cannot place a
                # cutoff, so drop the whole file rather than mixing in an eos-shaped one.
                print(f"  WARNING: {exc}; skipping {file_stem}")
                n_skipped_loudness += 1
                del window_prompts[window_start:]
                spans = []
                break
            if built is None:
                print(
                    f"  WARNING: no reconstructable reasoning/action in {file_stem} step {step['step_id']}; skipping"
                )
                continue
            meta, prompts = built
            spans.append((meta, len(window_prompts), len(window_prompts) + len(prompts)))
            window_prompts.extend(prompts)

        if not spans:
            print(f"  WARNING: no usable steps in {file_stem}; no output written")
            continue

        n_cutoffs = len(window_prompts) - window_start
        print(
            f"    {file_stem}: {len(spans)} usable step(s), {n_cutoffs} cutoff(s) added (window now {len(window_prompts)}/{max_window_prompts})"
        )
        window_files.append({"file_stem": file_stem, "out_path": out_path, "spans": spans})

        if len(window_prompts) >= max_window_prompts:
            _flush_window(window_prompts, window_files, **flush_kwargs)
            window_prompts = []
            window_files = []

    # Drain the final partial window.
    _flush_window(window_prompts, window_files, **flush_kwargs)

    sentence_acc = overall["correct"] / overall["evals"] if overall["evals"] else 0.0
    final_acc = overall["final_correct"] / overall["steps"] if overall["steps"] else 0.0
    mean_first = _mean_or_none(overall["first_correct_fracs"])
    mean_convinced = _mean_or_none(overall["convinced_fracs"])
    print(
        f"\nOverall [{args.strategy}] ({overall['steps']} steps, {overall['evals']} cutoff-evals): "
        f"cutoff acc {sentence_acc:.1%}, final acc {final_acc:.1%}, "
        f"mean first-correct fraction {mean_first}, mean convinced fraction {mean_convinced}"
    )
    if n_skipped_loudness:
        print(f"Skipped {n_skipped_loudness} trajectory file(s) with no usable direction-mass table")
    print(f"Results written to {output_dir}/")


if __name__ == "__main__":
    main()
