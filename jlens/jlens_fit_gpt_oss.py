#!/usr/bin/env python3
"""Fit a Jacobian lens on gpt-oss-20b using our own grid-environment prompts.

The lens shipped in ``data/jlens/`` was fit by Neuronpedia on WikiText-103
(1000 prompts x 128 tokens). Applying it to grid-navigation activations means
transporting with a matrix calibrated on encyclopedia prose. This script fits
one on the distribution we actually study: Harmony-formatted grid prompts plus
the model's own chain-of-thought.

Corpus
------
One prompt per trajectory file from ``trajectories_train_single_step`` (36k
files, one step each), sampled evenly across grid sizes and complexities. The
test set is deliberately untouched. Each prompt is the exact string the model
saw: the rendered Harmony template with the grid substituted, followed by the
raw completion (analysis channel included).

Fit window
----------
Every trajectory shares a byte-identical 414-token instruction prefix. jlens
averages the Jacobian over positions ``[skip_first, seq_len-1)`` and truncates
from the left only, so ``skip_first`` is the sole knob for excluding that
boilerplate -- otherwise most averaged positions would be the same instruction
block repeated across all 1000 prompts. We resolve it by tokenizing the prefix
rather than hardcoding 414, since ``from_hf(force_bos=True)`` can shift every
position by one.

Usage
-----
    # inspect the corpus, no model weights needed
    python jlens/jlens_fit_gpt_oss.py --n-prompts 20 --dump-prompts corpus.jsonl --dry-run

    # the real fit (server)
    python jlens/jlens_fit_gpt_oss.py \
        --trajectories-dir /workspace/trajectories/reveng/trajectories_train_single_step \
        --out-dir /workspace/jlens/gridenv --n-prompts 1000 --dim-batch 16
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import re
import subprocess
import time
from collections import Counter
from pathlib import Path

# The fit alternates one big retained-graph forward with ~180 backward passes,
# which fragments the caching allocator badly. Must be set before torch loads
# CUDA, hence module scope; setdefault so an explicit env var still wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

logger = logging.getLogger("jlens_fit")

MODEL_ID = "openai/gpt-oss-20b"
GRID_PLACEHOLDER = "{{grid_state}}"

# size7_comp0.4_123.json -> ("7", "0.4")
FILENAME_RE = re.compile(r"size(?P<size>\d+)_comp(?P<comp>[\d.]+)_\d+\.json$")

CONVERGENCE_COLUMNS = [
    "n_done",
    "prompt_idx",
    "max_seq_len",
    "elapsed_s",
    "identity_distance",
    "mean_rel_change",
]


# --------------------------------------------------------------------------
# Phase A: corpus
# --------------------------------------------------------------------------


def reconstruct_prompt(trajectory: dict, step: dict) -> str:
    """The exact text the model saw for one step: rendered prompt + completion.

    Mirrors ``build_truncated_input`` in gather_activations_utils.py, which does
    the same prefix+grid+suffix+output concatenation in token space. The
    ``token`` fields there are byte-level (``Ġ``/``Ċ``), so we rebuild from the
    text fields and let the tokenizer re-derive the ids; ``--check-roundtrip``
    proves the two agree.
    """
    template = trajectory["prompt"]["prompt_template"]
    if GRID_PLACEHOLDER not in template:
        raise ValueError(f"prompt_template is missing {GRID_PLACEHOLDER!r}")
    grid = "\n".join(step["grid_state"])
    return template.replace(GRID_PLACEHOLDER, grid) + step["output_text"]


def sample_files(root: Path, n_prompts: int, seed: int, sizes: list[int] | None) -> list[Path]:
    """Sample ``n_prompts`` trajectory files, balanced over size x complexity.

    Round-robins across the (size, complexity) strata so no single grid size
    dominates the average; remaining strata absorb the shortfall if one runs dry.
    """
    strata: dict[tuple[str, str], list[Path]] = {}
    for path in root.glob("size*/*.json"):
        m = FILENAME_RE.search(path.name)
        if m is None:
            continue
        if sizes is not None and int(m["size"]) not in sizes:
            continue
        strata.setdefault((m["size"], m["comp"]), []).append(path)

    if not strata:
        raise FileNotFoundError(f"no trajectory files matching size*/*.json under {root}")

    rng = random.Random(seed)
    pools = {key: rng.sample(files, len(files)) for key, files in strata.items()}
    order = sorted(pools, key=lambda k: (int(k[0]), float(k[1])))

    chosen: list[Path] = []
    while len(chosen) < n_prompts:
        drained = True
        for key in order:
            if not pools[key]:
                continue
            drained = False
            chosen.append(pools[key].pop())
            if len(chosen) == n_prompts:
                break
        if drained:  # every stratum exhausted
            logger.warning("only %d files available, wanted %d", len(chosen), n_prompts)
            break
    return chosen


def build_prompts(
    root: Path, n_prompts: int, seed: int, sizes: list[int] | None
) -> tuple[list[str], list[Path], Counter]:
    """Return (prompt strings, source files, size x complexity histogram)."""
    files = sample_files(root, n_prompts, seed, sizes)
    prompts, kept, histogram = [], [], Counter()
    for path in files:
        trajectory = json.loads(path.read_text(encoding="utf-8"))
        step = trajectory["steps"][0]
        prompts.append(reconstruct_prompt(trajectory, step))
        kept.append(path)
        m = FILENAME_RE.search(path.name)
        histogram[(int(m["size"]), float(m["comp"]))] += 1
    return prompts, kept, histogram


# --------------------------------------------------------------------------
# Phase B: where the boilerplate ends
# --------------------------------------------------------------------------


def resolve_skip_first(tokenizer, files: list[Path], n_check: int = 25) -> int:
    """Token index where the grid state begins, measured not assumed.

    Tokenizes the instruction prefix of several trajectories and requires them
    all to agree -- a mismatch would mean ``skip_first`` lands mid-grid for some
    prompts, silently averaging over the wrong positions.
    """
    lengths: Counter = Counter()
    stored: set[int] = set()
    for path in files[:n_check]:
        trajectory = json.loads(path.read_text(encoding="utf-8"))
        prompt = trajectory["prompt"]
        prefix_text = prompt["prompt_template"].split(GRID_PLACEHOLDER)[0]
        lengths[len(tokenizer(prefix_text).input_ids)] += 1
        stored.add(prompt["prompt_prefix_n_tokens"])

    if len(lengths) != 1:
        raise ValueError(
            f"instruction prefix tokenizes to inconsistent lengths {dict(lengths)}; "
            "skip_first would land in different places per prompt -- pass --skip-first "
            "explicitly if this is expected"
        )
    resolved = next(iter(lengths))
    logger.info(
        "resolved skip_first=%d from %d files (trajectories record prompt_prefix_n_tokens=%s)",
        resolved,
        min(n_check, len(files)),
        sorted(stored),
    )
    return resolved


# --------------------------------------------------------------------------
# Phase C/D: fit, with convergence tracking
# --------------------------------------------------------------------------


def _jacobian_stats(lens, previous: dict | None) -> tuple[float, float]:
    """(identity_distance, mean_rel_change), both averaged over source layers.

    ``identity_distance`` = ||J - I||_F / ||I||_F, how far the transport is from
    a plain logit lens. ``mean_rel_change`` = ||J_new - J_old||_F / ||J_old||_F,
    the early-stop signal: once new prompts stop moving J, the corpus has
    saturated.
    """
    import torch

    id_dists, rel_changes = [], []
    for layer, J in lens.jacobians.items():
        identity = torch.eye(J.shape[0], dtype=J.dtype)
        id_dists.append((J - identity).norm().item() / identity.norm().item())
        if previous is not None:
            denominator = previous[layer].norm().item()
            if denominator > 0:
                rel_changes.append((J - previous[layer]).norm().item() / denominator)
    mean_rel = sum(rel_changes) / len(rel_changes) if rel_changes else float("nan")
    return sum(id_dists) / len(id_dists), mean_rel


def _release_gpu_memory() -> float:
    """Drop cached blocks between chunks; return peak allocated GB, then reset it.

    Note this does *not* address the OOM most people hit here: the peak happens
    *inside* one prompt, where jacobian_for_prompt retains the forward graph
    across all ~d_model/dim_batch backward passes. Nothing is leaked between
    prompts (ActivationRecorder drops its hooks and tensors on exit). This only
    returns fragmented cache to the driver and reports the high-water mark, so
    --dim-batch / --source-layers / --max-seq-len can be tuned from real numbers.
    """
    import torch

    if not torch.cuda.is_available():
        return float("nan")
    peak = torch.cuda.max_memory_allocated() / 1024**3
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return peak


def fit_with_convergence(
    model,
    prompts: list[str],
    *,
    out_dir: Path,
    chunk: int,
    stop_at_delta: float,
    min_prompts: int,
    **fit_kwargs,
):
    """``jlens.fit`` in prefix-slices, logging convergence between them.

    jlens.fit has no callback hook, so we call it on ``prompts[:k]`` for growing
    k. Because each slice is a *prefix* of the same list, fit's own
    checkpoint/resume bookkeeping (``next_idx``) stays consistent across calls --
    no reimplementation of the running mean. ``checkpoint_every=None`` means one
    checkpoint write per chunk rather than per prompt; the file is
    ``n_layers * d_model**2 * 4`` bytes (~760 MB here), so per-prompt writes
    would dominate runtime.
    """
    import jlens

    convergence_path = out_dir / "convergence.csv"
    rows = [",".join(CONVERGENCE_COLUMNS)]
    previous: dict | None = None
    lens = None
    last_n = 0
    started = time.time()

    for k in range(min(chunk, len(prompts)), len(prompts) + 1, chunk):
        lens = jlens.fit(model, prompts[:k], checkpoint_every=None, **fit_kwargs)
        peak_gb = _release_gpu_memory()
        if lens.n_prompts == last_n:
            # Resumed run: this chunk was already in the checkpoint, so J did not
            # move. Recording it would write a row of spurious zero change.
            continue
        last_n = lens.n_prompts
        identity_distance, mean_rel_change = _jacobian_stats(lens, previous)
        previous = {layer: J.clone() for layer, J in lens.jacobians.items()}

        rows.append(
            ",".join(
                str(v)
                for v in (
                    lens.n_prompts,
                    k - 1,
                    fit_kwargs.get("max_seq_len", ""),
                    f"{time.time() - started:.3f}",
                    f"{identity_distance:.6f}",
                    f"{mean_rel_change:.8f}",
                )
            )
        )
        convergence_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        logger.info(
            "  %d/%d prompts | identity_distance=%.6f mean_rel_change=%.8f peak_gpu=%.1fGB",
            k,
            len(prompts),
            identity_distance,
            mean_rel_change,
            peak_gb,
        )

        # NaN on the first chunk (no previous J) correctly fails this comparison.
        if k >= min_prompts and mean_rel_change < stop_at_delta:
            logger.info("converged at %d prompts (< %g), stopping early", k, stop_at_delta)
            break

    return lens


def write_config(out_dir: Path, payload: dict) -> None:
    """Mirror the metadata sidecar the Neuronpedia fit produced."""
    import yaml

    (out_dir / "config.yaml").write_text(
        "# Jacobian lens fit on grid-environment trajectories\n"
        "# jlens by Anthropic PBC (https://github.com/anthropics/jlens), Apache-2.0\n"
        + yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent, text=True
        ).strip()
    except Exception:
        return "unknown"


def check_roundtrip(tokenizer, files: list[Path], n_check: int = 5) -> None:
    """Assert the reconstructed text re-tokenizes to the stored token ids.

    The load-bearing check: if this fails, the strings we fit on are not the
    strings the model saw, and skip_first does not mean what we think.
    """
    for path in files[:n_check]:
        trajectory = json.loads(path.read_text(encoding="utf-8"))
        step = trajectory["steps"][0]
        expected = (
            [t["token_id"] for t in trajectory["prompt"]["prompt_prefix_tokens"]]
            + [t["token_id"] for t in step["grid_state_tokens"]]
            + [t["token_id"] for t in trajectory["prompt"]["prompt_suffix_tokens"]]
            + [t["token_id"] for t in step["output_tokens"]]
        )
        actual = tokenizer(reconstruct_prompt(trajectory, step)).input_ids
        if actual != expected:
            first = next(
                (i for i, (a, b) in enumerate(zip(actual, expected)) if a != b),
                min(len(actual), len(expected)),
            )
            raise AssertionError(
                f"{path.name}: re-tokenization differs at index {first} "
                f"(len {len(actual)} vs stored {len(expected)}); "
                f"got {actual[first : first + 8]}, stored {expected[first : first + 8]}"
            )
    logger.info("roundtrip OK: %d prompts re-tokenize to the stored ids", min(n_check, len(files)))


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--trajectories-dir",
        type=Path,
        default=Path(r"C:\Uni\Thesis\data\reveng\trajectories_train_single_step"),
        help="Root containing size*/ trajectory dirs",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("out/jlens_gridenv"))
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--n-prompts", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sizes", type=int, nargs="*", default=None, help="Restrict to these grid sizes (default: all)"
    )

    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=1024,
        help="Left-truncation length. Must clear the 414-token prefix plus "
        "the grid (517 tokens at size 15) to reach any reasoning",
    )
    parser.add_argument(
        "--dim-batch",
        type=int,
        default=16,
        help="Output dims per backward pass. Memory scales with "
        "dim_batch * max_seq_len -- tune this first on the server",
    )
    parser.add_argument(
        "--skip-first",
        type=int,
        default=None,
        help="Override the measured instruction-prefix length (e.g. 16 for the jlens paper default, as an ablation)",
    )
    parser.add_argument("--target-layer", type=int, default=None)
    parser.add_argument("--source-layers", type=int, nargs="*", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--compile", action="store_true")

    parser.add_argument(
        "--eval-every", type=int, default=10, help="Prompts between convergence records / checkpoint writes"
    )
    parser.add_argument(
        "--stop-at-delta", type=float, default=0.002, help="Stop once mean_rel_change falls below this"
    )
    parser.add_argument("--min-prompts", type=int, default=100, help="Never stop early before this many prompts")

    parser.add_argument(
        "--dump-prompts", type=Path, default=None, help="Write the corpus to this .jsonl for inspection"
    )
    parser.add_argument("--dry-run", action="store_true", help="Build (and optionally dump) the corpus, then stop")
    parser.add_argument(
        "--check-roundtrip", action="store_true", help="Verify reconstructed text re-tokenizes to the stored ids"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    prompts, files, histogram = build_prompts(args.trajectories_dir, args.n_prompts, args.seed, args.sizes)
    logger.info("built %d prompts from %s", len(prompts), args.trajectories_dir)
    for (size, comp), count in sorted(histogram.items()):
        logger.info("  size %2d comp %.1f: %d", size, comp, count)

    if args.dump_prompts is not None:
        args.dump_prompts.parent.mkdir(parents=True, exist_ok=True)
        with args.dump_prompts.open("w", encoding="utf-8") as fh:
            for path, text in zip(files, prompts):
                fh.write(json.dumps({"file": str(path), "text": text}) + "\n")
        logger.info("dumped corpus -> %s", args.dump_prompts)

    tokenizer = None
    skip_first = args.skip_first
    if args.check_roundtrip or not args.dry_run:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        if args.check_roundtrip:
            check_roundtrip(tokenizer, files)
        if skip_first is None:
            skip_first = resolve_skip_first(tokenizer, files)

    if args.dry_run:
        logger.info("--dry-run: stopping before the fit")
        return

    if args.max_seq_len <= skip_first + 1:
        raise ValueError(
            f"max_seq_len={args.max_seq_len} <= skip_first={skip_first}+1: every prompt would be "
            "truncated to nothing but instruction boilerplate"
        )

    import jlens
    import torch
    from transformers import AutoModelForCausalLM

    args.out_dir.mkdir(parents=True, exist_ok=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=getattr(torch, args.dtype), device_map=args.device_map
    )
    model = jlens.from_hf(hf_model, tokenizer, compile=args.compile)
    logger.info("model: n_layers=%d d_model=%d", model.n_layers, model.d_model)

    # The retained graph spans min(source_layers) -> target_layer, and that span
    # is what dominates peak memory. Defaulting source_layers to "everything"
    # retains the full stack; restricting it is the cheapest OOM fix available.
    graph_start = min(args.source_layers) if args.source_layers else 0
    target = args.target_layer if args.target_layer is not None else model.n_layers - 1
    logger.info(
        "retained graph spans layers %d..%d (%d blocks) x dim_batch=%d x max_seq_len=%d%s",
        graph_start,
        target,
        target - graph_start,
        args.dim_batch,
        args.max_seq_len,
        "" if args.source_layers else "  -- pass --source-layers to shrink this",
    )

    started = time.time()
    lens = fit_with_convergence(
        model,
        prompts,
        out_dir=args.out_dir,
        chunk=args.eval_every,
        stop_at_delta=args.stop_at_delta,
        min_prompts=args.min_prompts,
        source_layers=args.source_layers,
        target_layer=args.target_layer,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        skip_first=skip_first,
        checkpoint_path=str(args.out_dir / "ckpt.pt"),
    )
    elapsed = time.time() - started

    stem = Path(args.model_id).name
    lens_path = args.out_dir / f"{stem}_gridenv_jacobian_lens.pt"
    lens.save(str(lens_path))
    logger.info("saved %s (%s)", lens_path, lens)

    write_config(
        args.out_dir,
        {
            "hf_model_name": args.model_id,
            "git_sha": _git_sha(),
            "dataset": {
                "root": str(args.trajectories_dir),
                "n_files_sampled": len(files),
                "seed": args.seed,
                "sizes": args.sizes or "all",
                "histogram": {f"size{s}_comp{c}": n for (s, c), n in sorted(histogram.items())},
            },
            "fit": {
                "n_prompts_requested": args.n_prompts,
                "n_prompts_fitted": lens.n_prompts,
                "skip_first": skip_first,
                "skip_first_source": "explicit" if args.skip_first is not None else "measured",
                "max_seq_len": args.max_seq_len,
                "dim_batch": args.dim_batch,
                "target_layer": args.target_layer,
                "source_layers": args.source_layers or "all below target",
                "dtype": args.dtype,
                "device_map": args.device_map,
                "compile": args.compile,
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
