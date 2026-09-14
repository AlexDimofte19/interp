#!/usr/bin/env python3
"""Fit a Jacobian lens on Qwen3.6-35B-A3B using its own replayed grid trajectories.

The sibling of ``jlens_fit_gpt_oss.py``, which is left untouched -- it is stage 2 of
``scripts/reproduce_all.sh`` and its defaults are the record of the gpt-oss fit. Everything about
the corpus is shared with it by import, so any difference between the two lenses is the model and
the fit window, and nothing else.

Two things differ, and both are forced by the model.

Model
-----
``Qwen/Qwen3.6-35B-A3B`` is ``Qwen3_5MoeForConditionalGeneration`` (``model_type: qwen3_5_moe``):
a vision-language wrapper around a 40-layer hybrid text stack -- 30 gated-DeltaNet
``linear_attention`` layers and 10 ``full_attention`` layers at indices 3, 7, ... 39 -- with
d_model 2048, 256 experts and top-8 routing. The architecture lives in ``transformers`` itself
(there is no ``trust_remote_code`` fallback), and it first appears in **transformers 5.17**;
4.57.x fails at ``from_pretrained`` with an unrecognised model type.

We fit the nine full-attention layers below the top, with layer 39 -- itself a full-attention
layer, and the last block -- as the target. That is the same arrangement gpt-oss has at its layer
23: the lens is the identity at the target by construction.

Fit window
----------
Qwen's chains are ~30x longer than gpt-oss's (median 18,094 output tokens against a few hundred),
so "the first 1024 tokens" is no longer most of a trajectory -- it is the instruction prefix plus
the opening of the reasoning. We want the Jacobian averaged over a **mid-chain** window while the
forward pass still sees **everything before it**, so those activations carry their real context.

Done naively that is a retained graph over ~9,500 positions. It does not have to be: by causality,
every path from ``h_l[p]`` to ``h_target[p']`` (p' >= p, both in the window) stays inside the
window, so prefix activations are constants with respect to every position we differentiate. So
the prefix is run under ``no_grad`` into a cache and only the window is run with a graph --
numerically identical to a full-graph fit at ``max_seq_len=end, skip_first=end-W``, at ``W/end``
of the backward cost and graph memory. ``tests/test_jlens_fit_qwen.py`` asserts that equality
against a full-graph reference; it is the load-bearing test here.

Usage
-----
    # corpus, window statistics and the round-trip check; no weights needed
    python jlens/jlens_fit_qwen.py --dry-run --check-roundtrip

    # the real fit (GPU host, >= 80GB)
    python jlens/jlens_fit_qwen.py --n-prompts 144 --dim-batch 1

or through ``wrappers/jlens_fit_qwen.sh``, which is the recorded invocation.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import statistics
import sys
import time
import types
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# One retained forward against ~2000 backward passes fragments the caching allocator badly, so
# expandable segments matter here even more than they do for the gpt-oss fit. That fit sets
# PYTORCH_CUDA_ALLOC_CONF, which torch >= 2.9 accepts but warns is deprecated -- the spelling it
# reads now is PYTORCH_ALLOC_CONF, and a setting that is only warned about today is one that
# stops applying tomorrow. Both are set, both with setdefault so an explicit env var still wins,
# and both before anything imports torch.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# The corpus, convergence and provenance helpers are shared with the gpt-oss fit verbatim.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from jlens_fit_gpt_oss import (  # noqa: E402
    FILENAME_RE,
    _git_sha,
    _jacobian_stats,
    _release_gpu_memory,
    check_roundtrip,
    reconstruct_prompt,
    sample_files,
    write_config,
)

logger = logging.getLogger("jlens_fit_qwen")

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
TRAJECTORIES_DIR = Path("/workspace/trajectories/qwen3.6-35b/replayed_single_step/mass_train_576")

# The full-attention layers below the target. config.text_config.layer_types makes every fourth
# layer from index 3 a full_attention layer; 39 is the tenth of them and is the target.
FULL_ATTENTION_SOURCE_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35]

# The gpt-oss script's CONVERGENCE_COLUMNS with `max_seq_len` split in two, because the window
# moves per prompt and one number can no longer describe it.
CONVERGENCE_COLUMNS = [
    "n_done",
    "prompt_idx",
    "window_end",
    "window_size",
    "elapsed_s",
    "identity_distance",
    "mean_rel_change",
]


class WindowTooShort(ValueError):
    """A chain that cannot hold a full window ending at its target fraction."""


# --------------------------------------------------------------------------
# Phase A: corpus, with a window per trajectory
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowedPrompt:
    """One trajectory, the text the model saw, and where its Jacobian window sits.

    ``end`` and ``start`` are indices into the tokenization of ``text``: the forward pass runs on
    ``ids[:end]`` and the Jacobian is averaged over ``[start, end - 1)`` -- the last position is
    dropped for the same reason jlens drops it, it has no next-token target.
    """

    path: Path
    grid_size: int
    complexity: float
    text: str
    reasoning_start: int
    total_len: int
    end: int
    window_size: int

    @property
    def start(self) -> int:
        return self.end - self.window_size

    @property
    def frac_achieved(self) -> float:
        """Where in the chain the window ends, as a fraction of the reasoning."""
        return (self.end - self.reasoning_start) / (self.total_len - self.reasoning_start)


def resolve_window(
    trajectory: dict, step: dict, *, frac: float, window_size: int, max_window_end: int | None
) -> tuple[int, int, int]:
    """``(reasoning_start, total_len, end)`` for one trajectory, from its stored token counts.

    No tokenizer is involved: the counts are recorded per trajectory and ``--check-roundtrip``
    proves they are the tokenizer's. ``frac=0`` anchors the window at the first reasoning token,
    which is the gpt-oss behaviour and is kept as an ablation.

    Raises:
        WindowTooShort: If the chain cannot hold ``window_size`` tokens before ``end``.
    """
    prompt = trajectory["prompt"]
    reasoning_start = prompt["prompt_prefix_n_tokens"] + step["grid_state_n_tokens"] + prompt["prompt_suffix_n_tokens"]
    output_n = step["output_n_tokens"]
    total_len = reasoning_start + output_n

    offset = window_size if frac == 0 else round(frac * output_n)
    if max_window_end is not None:
        offset = min(offset, max_window_end)
    offset = min(offset, output_n)
    if offset < window_size:
        raise WindowTooShort(
            f"chain of {output_n} reasoning tokens gives a window end {offset} tokens in, "
            f"short of the {window_size}-token window"
        )
    return reasoning_start, total_len, reasoning_start + offset


def build_windowed_corpus(
    root: Path,
    n_prompts: int,
    seed: int,
    sizes: list[int] | None,
    *,
    frac: float,
    window_size: int,
    max_window_end: int | None,
) -> tuple[list[WindowedPrompt], Counter, Counter]:
    """Draw ``n_prompts`` trajectories balanced over size x complexity, windowed.

    ``sample_files`` is asked for every file so its round-robin gives a balanced *order*, and we
    then take the first ``n_prompts`` that can hold a window. Sampling exactly ``n_prompts`` and
    dropping the short ones afterwards would quietly unbalance the strata.
    """
    available = sum(1 for _ in root.glob("size*/*.json"))
    ordered = sample_files(root, available, seed, sizes)

    prompts: list[WindowedPrompt] = []
    histogram: Counter = Counter()
    skipped: Counter = Counter()
    for path in ordered:
        if len(prompts) == n_prompts:
            break
        trajectory = json.loads(path.read_text(encoding="utf-8"))
        step = trajectory["steps"][0]
        match = FILENAME_RE.search(path.name)
        key = (int(match["size"]), float(match["comp"]))
        try:
            reasoning_start, total_len, end = resolve_window(
                trajectory, step, frac=frac, window_size=window_size, max_window_end=max_window_end
            )
        except WindowTooShort:
            skipped[key] += 1
            continue
        prompts.append(
            WindowedPrompt(
                path=path,
                grid_size=key[0],
                complexity=key[1],
                text=reconstruct_prompt(trajectory, step),
                reasoning_start=reasoning_start,
                total_len=total_len,
                end=end,
                window_size=window_size,
            )
        )
        histogram[key] += 1
    return prompts, histogram, skipped


def log_window_statistics(prompts: list[WindowedPrompt], skipped: Counter) -> None:
    """Print the forward length the fit will actually pay for, per grid size."""
    by_size: dict[int, list[WindowedPrompt]] = {}
    for prompt in prompts:
        by_size.setdefault(prompt.grid_size, []).append(prompt)

    logger.info("window statistics (forward runs on [0, end), Jacobian over [end-W, end-1))")
    logger.info("  %4s %5s %9s %9s %9s %9s", "size", "n", "median_end", "min_end", "max_end", "med_chain")
    for size in sorted(by_size):
        ends = sorted(p.end for p in by_size[size])
        chains = sorted(p.total_len - p.reasoning_start for p in by_size[size])
        logger.info(
            "  %4d %5d %9d %9d %9d %9d",
            size,
            len(ends),
            statistics.median(ends),
            ends[0],
            ends[-1],
            statistics.median(chains),
        )
    if prompts:
        ends = sorted(p.end for p in prompts)
        logger.info("  all %d prompts: median end %d, max end %d", len(prompts), statistics.median(ends), ends[-1])
    if skipped:
        logger.info(
            "  skipped %d trajectories whose chain is shorter than the window: %s",
            sum(skipped.values()),
            ", ".join(f"size{s}_comp{c}={n}" for (s, c), n in sorted(skipped.items())),
        )


# --------------------------------------------------------------------------
# Phase B: the model
# --------------------------------------------------------------------------


def load_lens_model(args, tokenizer):
    """Load the HF model and wrap it as a jlens ``LensModel``.

    ``Qwen3_5MoeForConditionalGeneration`` is a VLM wrapper, so ``AutoModelForCausalLM`` is the
    wrong auto class for it. jlens's own layout detection needs no help: ``Layout("model")`` fails
    (``Qwen3_5MoeModel`` holds only ``.visual`` and ``.language_model``) and
    ``Layout("model.language_model")`` matches, with ``lm_head`` on the wrapper.
    """
    import jlens
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(args.model_id)
    model_class = args.model_class
    if model_class == "auto":
        architectures = getattr(config, "architectures", None) or []
        is_wrapped = hasattr(config, "vision_config") or any(
            a.endswith("ForConditionalGeneration") for a in architectures
        )
        model_class = "image-text" if is_wrapped else "causal-lm"
        logger.info("model_class=auto resolved to %r from %s", model_class, architectures or "config")

    if model_class == "image-text":
        from transformers import AutoModelForImageTextToText

        loader = AutoModelForImageTextToText
    else:
        loader = AutoModelForCausalLM

    hf_model = loader.from_pretrained(args.model_id, dtype=getattr(torch, args.dtype), device_map=args.device_map)

    # The text path never calls the vision tower -- jlens forwards through .language_model
    # directly -- so its weights are pure overhead on a card that has little to spare.
    if args.drop_vision_tower:
        inner = getattr(hf_model, "model", None)
        if inner is not None and getattr(inner, "visual", None) is not None:
            inner.visual = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("dropped the vision tower")

    model = jlens.from_hf(hf_model, tokenizer, compile=args.compile)
    logger.info("model: %r", model)

    text_config = hf_model.config.get_text_config()
    layer_types = getattr(text_config, "layer_types", None)
    if layer_types is not None:
        wrong = [layer for layer in args.source_layers if layer_types[layer] != "full_attention"]
        if wrong:
            logger.warning(
                "source layers %s are not full_attention layers (they are %s); fitting them anyway",
                wrong,
                sorted({layer_types[layer] for layer in wrong}),
            )
    return model


# --------------------------------------------------------------------------
# Phase C: the estimator, with the prefix out of the graph
# --------------------------------------------------------------------------


def _freeze_cache(cache) -> None:
    """Stop the windowed forward from writing its own state back into the prefilled cache.

    ``LinearAttentionLayer.update_recurrent_state`` does ``self.recurrent_states[i].copy_(new)``
    -- in place, to keep a static address for cudagraphs. The gated delta rule took that same
    tensor as ``initial_state`` and saved it for backward (it is a matmul operand; being
    ``requires_grad=False`` does not exempt it from the version check), so the write bumps its
    version and the *second* backward pass dies with "one of the variables needed for gradient
    computation has been modified by an inplace operation". Nothing reads the cache after the
    window, so the write is pure loss, and the DeltaNet discards the return value.

    The conv-state write is left alone on purpose: ``update_conv_state`` returns a fresh ``cat``
    and only then copies into ``conv_states``, a different tensor from the one in the graph, so it
    is not the same hazard -- and reimplementing its prefill/padding branches here would duplicate
    internals that are free to change.
    """
    try:
        cache.update_recurrent_state = lambda recurrent_states, *args, **kwargs: recurrent_states
    except AttributeError:  # a cache that forbids attribute assignment; only the KV path then
        logger.debug("could not freeze %s; if the backward trips on an in-place write this is why", type(cache))


def jacobian_for_windowed_prompt(
    model,
    input_ids,
    *,
    window_start: int,
    source_layers: list[int],
    target_layer: int,
    dim_batch: int = 1,
):
    """``jlens.fitting.jacobian_for_prompt`` with the prefix prefilled instead of retained.

    ``input_ids`` is ``[end]``; the forward runs on all of it, but only ``[window_start, end)``
    carries an autograd graph. The estimator, the cotangent construction, the ``retain_graph``
    schedule and the reduction are identical to jlens's, and the valid positions are the same set
    ``valid_position_mask(end, skip_first=window_start)`` would give -- the whole window except its
    last position. So this returns exactly what a full-graph fit at ``max_seq_len=end,
    skip_first=window_start`` returns, and ``tests/test_jlens_fit_qwen.py`` asserts it.

    Returns:
        ``(jacobians, seq_len, n_valid_positions)``, ``jacobians`` mapping each source layer to a
        ``[d_model, d_model]`` fp32 CPU tensor.
    """
    import torch
    from jlens.hooks import ActivationRecorder

    d_model = model.d_model
    device = model.input_device
    ids = input_ids.to(device).unsqueeze(0).expand(dim_batch, -1)
    prefix, window = ids[:, :window_start], ids[:, window_start:]
    window_len = window.shape[1]
    if window_len < 2:
        raise ValueError(f"window of {window_len} tokens leaves no valid positions")

    text_module = model._text_module  # noqa: SLF001 -- jlens's own forward() pins use_cache=False

    # The prefix is replicated to dim_batch rather than prefilled once and expanded: the cache
    # classes differ per architecture and expanding one is not part of any stable API. At the
    # dim_batch of 1-2 this fit runs at, the duplicated prefill is cheap and it is no-grad.
    cache = None
    if window_start > 0:
        with torch.no_grad():
            prefilled = text_module(input_ids=prefix, use_cache=True)
        cache = prefilled.past_key_values
        del prefilled
        _freeze_cache(cache)

    jacobians = {layer: torch.zeros(d_model, d_model, dtype=torch.float32) for layer in source_layers}
    n_passes = math.ceil(d_model / dim_batch)

    with (
        ActivationRecorder(
            model.layers, at=[*source_layers, target_layer], start_graph_at=min(source_layers)
        ) as recorder,
        torch.enable_grad(),
    ):
        # cache_position is passed rather than inferred so the window's positions are the ones it
        # actually holds in the sequence -- a rotary offset of `window_start` is the difference
        # between reading mid-chain activations and reading the chain's opening.
        text_module(
            input_ids=window,
            past_key_values=cache,
            use_cache=cache is not None,
            cache_position=torch.arange(window_start, window_start + window_len, device=device),
        )
        target_activation = recorder.activations[target_layer]
        source_activations = [recorder.activations[layer] for layer in source_layers]

        # Window-relative positions. The final position of the truncated sequence is dropped, as
        # valid_position_mask does: it has no next-token target.
        valid_positions = torch.arange(window_len - 1, device=target_activation.device)
        batch_indices = torch.arange(dim_batch, device=target_activation.device)
        cotangent = torch.zeros_like(target_activation)

        for pass_idx, dim_start in enumerate(range(0, d_model, dim_batch)):
            n_dims_this_pass = min(dim_batch, d_model - dim_start)
            cotangent.zero_()
            cotangent[
                batch_indices[:n_dims_this_pass, None],
                valid_positions[None, :],
                dim_start + batch_indices[:n_dims_this_pass, None],
            ] = 1.0
            grads = torch.autograd.grad(
                outputs=target_activation,
                inputs=source_activations,
                grad_outputs=cotangent,
                retain_graph=(pass_idx < n_passes - 1),
            )
            for layer, grad in zip(source_layers, grads, strict=True):
                positions_on_device = valid_positions.to(grad.device, non_blocking=True)
                rows = grad[:n_dims_this_pass, positions_on_device, :].float().mean(dim=1)
                jacobians[layer][dim_start : dim_start + n_dims_this_pass, :] = rows.cpu()
            del grads

    del cache
    return jacobians, window_start + window_len, int(valid_positions.numel())


# --------------------------------------------------------------------------
# Phase D: accumulation, checkpointing and convergence
# --------------------------------------------------------------------------


def _atomic_save(obj, path: Path) -> None:
    """``torch.save`` to a temp file then ``os.replace``; jlens's own, since we do not call fit."""
    import torch

    tmp_path = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def fit_windowed(
    model,
    prompts: list[WindowedPrompt],
    *,
    out_dir: Path,
    source_layers: list[int],
    target_layer: int,
    dim_batch: int,
    eval_every: int,
    stop_at_delta: float,
    min_prompts: int,
    window_frac: float,
    window_size: int,
    resume: bool = True,
):
    """Accumulate per-prompt Jacobians into a running mean, with convergence records.

    ``jlens.fit`` cannot be used here: it takes one ``skip_first``/``max_seq_len`` for the whole
    call, and our window moves per prompt. The checkpoint keeps fit's schema so the resume
    semantics are the same, plus the window parameters -- resuming a checkpoint under a different
    window would silently average two different quantities.
    """
    import torch
    from jlens.fitting import _check_layer_indices
    from jlens.lens import JacobianLens

    source_layers, target_layer = _check_layer_indices(source_layers, target_layer, model.n_layers)
    d_model = model.d_model
    checkpoint_path = out_dir / "ckpt.pt"
    convergence_path = out_dir / "convergence.csv"

    jacobian_sum = {layer: torch.zeros(d_model, d_model, dtype=torch.float32) for layer in source_layers}
    n_done = 0
    next_idx = 0
    rows = [",".join(CONVERGENCE_COLUMNS)]

    if resume and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        for key, expected in (
            ("source_layers", source_layers),
            ("target_layer", target_layer),
            ("window_frac", window_frac),
            ("window_size", window_size),
        ):
            if key in state and state[key] != expected:
                raise ValueError(
                    f"checkpoint at {checkpoint_path} was fitted with {key}={state[key]!r}, not "
                    f"{expected!r}; delete it or pass the original value"
                )
        jacobian_sum, n_done, next_idx = state["jacobian_sum"], state["n_done"], state["next_idx"]
        logger.info("resuming from checkpoint: %d/%d prompts processed", next_idx, len(prompts))
        if convergence_path.exists():
            rows = convergence_path.read_text(encoding="utf-8").rstrip("\n").split("\n")

    def write_checkpoint() -> None:
        _atomic_save(
            {
                "jacobian_sum": jacobian_sum,
                "n_done": n_done,
                "next_idx": next_idx,
                "source_layers": source_layers,
                "target_layer": target_layer,
                "window_frac": window_frac,
                "window_size": window_size,
            },
            checkpoint_path,
        )

    previous: dict | None = None
    started = time.time()
    sqrt_d = math.sqrt(d_model)

    for prompt_idx, prompt in enumerate(prompts):
        if prompt_idx < next_idx:
            continue
        prompt_started = time.perf_counter()
        input_ids = model.encode(prompt.text, max_length=prompt.end)[0]
        if input_ids.shape[0] != prompt.end:
            # The stored token counts and the tokenizer disagree, so the window is not where the
            # counts say. Never guess -- --check-roundtrip is the place that proves they agree.
            logger.warning(
                "skipping %s: tokenized to %d tokens, expected the window to end at %d",
                prompt.path.name,
                input_ids.shape[0],
                prompt.end,
            )
            next_idx = prompt_idx + 1
            continue

        per_prompt_J, seq_len, n_valid = jacobian_for_windowed_prompt(
            model,
            input_ids,
            window_start=prompt.start,
            source_layers=source_layers,
            target_layer=target_layer,
            dim_batch=dim_batch,
        )

        if n_done == 0:
            # CLAUDE.md records this MoE family producing NaNs under a multi-GPU device_map. A
            # NaN here poisons the running mean irrecoverably, so it is caught before the sum.
            nonfinite = [layer for layer, J in per_prompt_J.items() if not torch.isfinite(J).all()]
            if nonfinite:
                raise RuntimeError(
                    f"non-finite Jacobian at layers {nonfinite} on the first prompt "
                    f"({prompt.path.name}); check --device-map and --dtype before going further"
                )

        prompt_norm = max(per_prompt_J[layer].norm().item() for layer in source_layers) / sqrt_d
        for layer in source_layers:
            jacobian_sum[layer] += per_prompt_J[layer]
        n_done += 1
        next_idx = prompt_idx + 1
        peak_gb = _release_gpu_memory()
        logger.info(
            "  prompt %d/%d  %s  end=%d window=[%d,%d) n_valid=%d  %.0fs  max||J||/sqrt(d)=%.3f  peak_gpu=%.1fGB",
            prompt_idx + 1,
            len(prompts),
            prompt.path.name,
            seq_len,
            prompt.start,
            prompt.end,
            n_valid,
            time.perf_counter() - prompt_started,
            prompt_norm,
            peak_gb,
        )

        if n_done % eval_every == 0 or prompt_idx == len(prompts) - 1:
            mean = {layer: jacobian_sum[layer] / n_done for layer in source_layers}
            identity_distance, mean_rel_change = _jacobian_stats(types.SimpleNamespace(jacobians=mean), previous)
            previous = {layer: J.clone() for layer, J in mean.items()}
            rows.append(
                ",".join(
                    str(v)
                    for v in (
                        n_done,
                        prompt_idx,
                        prompt.end,
                        window_size,
                        f"{time.time() - started:.3f}",
                        f"{identity_distance:.6f}",
                        f"{mean_rel_change:.8f}",
                    )
                )
            )
            convergence_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            write_checkpoint()
            logger.info(
                "  %d prompts | identity_distance=%.6f mean_rel_change=%.8f",
                n_done,
                identity_distance,
                mean_rel_change,
            )
            if n_done >= min_prompts and mean_rel_change < stop_at_delta:
                logger.info("converged at %d prompts (< %g), stopping early", n_done, stop_at_delta)
                break

    write_checkpoint()
    if n_done == 0:
        raise ValueError("no prompts were fitted")
    mean = {layer: jacobian_sum[layer] / n_done for layer in source_layers}
    return JacobianLens(jacobians=mean, n_prompts=n_done, d_model=d_model)


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trajectories-dir", type=Path, default=TRAJECTORIES_DIR, help="Root containing size*/")
    parser.add_argument("--out-dir", type=Path, default=Path("/workspace/jlens/qwen3_6_35b"))
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument(
        "--model-class",
        choices=["auto", "image-text", "causal-lm"],
        default="auto",
        help="Auto class to load with; qwen3_5_moe is a ForConditionalGeneration, so image-text",
    )
    parser.add_argument("--n-prompts", type=int, default=144)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sizes", type=int, nargs="*", default=None, help="Restrict to these grid sizes")

    parser.add_argument(
        "--window-frac",
        type=float,
        default=0.5,
        help="Where in the reasoning the window ENDS, as a fraction of the chain. 0 anchors it at "
        "the first reasoning token (the gpt-oss behaviour), as an ablation",
    )
    parser.add_argument("--window-size", type=int, default=1024, help="Positions the Jacobian is averaged over")
    parser.add_argument(
        "--max-window-end",
        type=int,
        default=None,
        help="Cap the window end this many tokens into the reasoning, bounding the prefill on long chains",
    )

    parser.add_argument(
        "--dim-batch",
        type=int,
        default=1,
        help="Output dims per backward pass. Memory scales with dim_batch * window_size; 71.9GB of "
        "weights on an 80GB card leaves little, so this starts at 1",
    )
    parser.add_argument(
        "--source-layers",
        type=int,
        nargs="*",
        default=FULL_ATTENTION_SOURCE_LAYERS,
        help="Layers to fit. Defaults to the full-attention layers below the target",
    )
    parser.add_argument("--target-layer", type=int, default=None, help="Defaults to the last block, 39")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--keep-vision-tower",
        dest="drop_vision_tower",
        action="store_false",
        help="Keep the vision tower on the device; it is never called by the text path",
    )

    parser.add_argument("--eval-every", type=int, default=8, help="Prompts between convergence records")
    parser.add_argument("--stop-at-delta", type=float, default=0.002)
    parser.add_argument("--min-prompts", type=int, default=48)

    parser.add_argument("--dump-prompts", type=Path, default=None, help="Write the corpus to this .jsonl")
    parser.add_argument("--dry-run", action="store_true", help="Build the corpus and the windows, then stop")
    parser.add_argument("--check-roundtrip", action="store_true", help="Verify the text re-tokenizes to stored ids")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    prompts, histogram, skipped = build_windowed_corpus(
        args.trajectories_dir,
        args.n_prompts,
        args.seed,
        args.sizes,
        frac=args.window_frac,
        window_size=args.window_size,
        max_window_end=args.max_window_end,
    )
    logger.info("built %d prompts from %s", len(prompts), args.trajectories_dir)
    for (size, comp), count in sorted(histogram.items()):
        logger.info("  size %2d comp %.1f: %d", size, comp, count)
    log_window_statistics(prompts, skipped)
    if len(prompts) < args.n_prompts:
        logger.warning("only %d prompts available, wanted %d", len(prompts), args.n_prompts)

    if args.dump_prompts is not None:
        args.dump_prompts.parent.mkdir(parents=True, exist_ok=True)
        with args.dump_prompts.open("w", encoding="utf-8") as fh:
            for prompt in prompts:
                fh.write(
                    json.dumps(
                        {
                            "file": str(prompt.path),
                            "window": [prompt.start, prompt.end],
                            "reasoning_start": prompt.reasoning_start,
                            "total_len": prompt.total_len,
                            "text": prompt.text,
                        }
                    )
                    + "\n"
                )
        logger.info("dumped corpus -> %s", args.dump_prompts)

    tokenizer = None
    if args.check_roundtrip or not args.dry_run:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        if args.check_roundtrip:
            check_roundtrip(tokenizer, [p.path for p in prompts])

    if args.dry_run:
        logger.info("--dry-run: stopping before the fit")
        return

    import transformers

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = load_lens_model(args, tokenizer)
    target = args.target_layer if args.target_layer is not None else model.n_layers - 1
    logger.info(
        "retained graph spans layers %d..%d (%d blocks) x dim_batch=%d x window=%d; the prefix is prefilled",
        min(args.source_layers),
        target,
        target - min(args.source_layers),
        args.dim_batch,
        args.window_size,
    )

    started = time.time()
    lens = fit_windowed(
        model,
        prompts,
        out_dir=args.out_dir,
        source_layers=args.source_layers,
        target_layer=args.target_layer,
        dim_batch=args.dim_batch,
        eval_every=args.eval_every,
        stop_at_delta=args.stop_at_delta,
        min_prompts=args.min_prompts,
        window_frac=args.window_frac,
        window_size=args.window_size,
    )
    elapsed = time.time() - started

    stem = Path(args.model_id).name
    lens_path = args.out_dir / f"{stem}_gridenv_jacobian_lens.pt"
    lens.save(str(lens_path))
    logger.info("saved %s (%s)", lens_path, lens)

    ends = sorted(p.end for p in prompts)
    write_config(
        args.out_dir,
        {
            "hf_model_name": args.model_id,
            "git_sha": _git_sha(),
            "transformers_version": transformers.__version__,
            "dataset": {
                "root": str(args.trajectories_dir),
                "n_files_sampled": len(prompts),
                "seed": args.seed,
                "sizes": args.sizes or "all",
                "histogram": {f"size{s}_comp{c}": n for (s, c), n in sorted(histogram.items())},
                "skipped_too_short": sum(skipped.values()),
            },
            "fit": {
                "n_prompts_requested": args.n_prompts,
                "n_prompts_fitted": lens.n_prompts,
                "window_frac": args.window_frac,
                "window_size": args.window_size,
                "max_window_end": args.max_window_end,
                "window_end_median": statistics.median(ends) if ends else None,
                "window_end_max": ends[-1] if ends else None,
                "prefix_prefilled": True,
                "dim_batch": args.dim_batch,
                "target_layer": args.target_layer,
                "source_layers": args.source_layers,
                "dtype": args.dtype,
                "device_map": args.device_map,
                "compile": args.compile,
                "drop_vision_tower": args.drop_vision_tower,
                "stop_at_delta": args.stop_at_delta,
                "min_prompts": args.min_prompts,
                "eval_every": args.eval_every,
            },
            "results": {
                "d_model": lens.d_model,
                "source_layers_fitted": len(lens.source_layers),
                "elapsed_s": round(elapsed, 1),
            },
        },
    )
    logger.info("wrote config.yaml and convergence.csv -> %s", args.out_dir)


if __name__ == "__main__":
    main()
