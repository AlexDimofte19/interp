"""Full j-space sweep over reasoning tokens: one row per (reasoning token, layer).

Combines gather_activations' single-pass activation extraction with
jlens_action_ranks' lens application: for every reasoning ("analysis") token of
every step, capture the residual stream at every requested layer in one forward
pass, apply the Jacobian lens, and record the top-20 predicted tokens plus each
token's position in the reasoning chain.

Each forward pass now produces two artefacts, so downstream analyses never have
to re-run gather_activations over the same trajectories:

  {activations_dir}/size{S}/{stem}/{stem}_jlens_analysis.csv
  {activations_dir}/size{S}/{stem}/{model}/layer_{N}/step_{M}/output/{idx}.pt

The .pt layout is exactly gather_activations' (reasoning tokens are output
tokens, hence the `output` category, keyed by output-relative index), so every
existing activation loader works unchanged.

Both the activation hooks and the jlens fit capture decoder-block *outputs*, so
layer indices line up directly; layer 23 is the jlens target (lens is identity
there: just norm + unembed).

Needs a GPU that fits gpt-oss-20b — run on the server, not the laptop.
First run downloads one 4.2 GB shard to cache lm_head + final norm (see
ensure_unembed_assets), reused from jlens_action_ranks_sampled.py.

Usage:
  python scripts/jlens_reasoning_tokens.py \
    --trajectory-paths /workspace/trajectories/trajectories_test_full \
    --jlens_dir /workspace/jlens \
    --layers 7:23 \
    --per-combo 200 \
    --activations-dir /workspace/activations/jlens_reasoning_tokens

--per-combo caps the run count per (size, complexity) cell rather than globally,
so the sweep stays evenly spread over the grid: 200 across a 6x6 grid = 7200
trajectories. It is a target for what ends up on disk, not a per-run batch size:
trajectories that already have a CSV count towards the 200, so re-running only
fills the gaps and cells that are already full (or fuller) are left untouched.

Throughput
----------
Everything here scales with the length of the reasoning chain, which is why the
high-complexity cells (comp 0.6-1.0) dominate the runtime: a step emits one CSV
row and one .pt file per (reasoning token, layer), so a 700-token chain over 17
layers is ~12k files and ~12k rows *per step*. The GPU is not the bottleneck -
the file writes and the per-row Python are. Hence:

  --io-workers        writes the .pt tree from a thread pool, overlapping it with
                      the next forward pass (the single biggest win)
  --forward-batch-*   packs several steps into one padded forward pass
  --batch-size        bounds the [B, vocab] lens logits
  --profile           per-phase wall time, so the split is measured not guessed
  --benchmark-pt-write / --no-save-activations
                      isolate what the activation tree costs on this host

Run `--profile` on one comp-0.2 and one comp-0.8 trajectory before tuning; the
balance between the buckets depends on whether --activations-dir is local NVMe
or a network volume.
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

# Running this file directly puts scripts/ on sys.path[0], not the repo root, so the
# sibling `scripts.*` imports below would miss. Put the repo root first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Heavy deps (torch/transformers) and sibling-script imports are done lazily inside
# main() so --self-test runs on any machine with only the stdlib.

TARGET_LAYER = 23  # jlens target: final decoder block, lens is identity here
ACTIONS = ["RIGHT", "LEFT", "UP", "DOWN"]
TOP_K = 20
# together_ai_openai_gpt-oss-20b_size11_comp1.0_987.json -> size / comp / run
NAME_RE = re.compile(r"size(?P<size>\d+)_comp(?P<comp>[\d.]+)_(?P<run>\d+)")


def reasoning_token_positions(trajectory: dict, step: dict) -> list[tuple[int, int, str]]:
    """(reasoning_pos, abs_pos, token_text) for each analysis token of a step.

    reasoning_pos is the 0-based index within the reasoning chain; abs_pos is the
    token's absolute position in prefix+grid+suffix+output (what the forward pass
    and the lens see). Falls back to all output tokens if none are tagged analysis.
    """
    n_prefix = len(trajectory["prompt"]["prompt_prefix_tokens"])
    n_grid = len(step["grid_state_tokens"])
    n_suffix = len(trajectory["prompt"]["prompt_suffix_tokens"])
    output_start = n_prefix + n_grid + n_suffix

    out = step["output_tokens"]
    idxs = [i for i, t in enumerate(out) if "analysis" in t.get("token_groups", [])] or list(range(len(out)))
    return [(rp, output_start + oi, out[oi].get("token", "")) for rp, oi in enumerate(idxs)]


def parse_name(stem: str) -> dict:
    m = NAME_RE.search(stem)
    return m.groupdict() if m else {"size": "", "comp": "", "run": ""}


def combo_sort_key(combo: tuple[str, str]) -> tuple[float, float]:
    """Numeric ordering for (size, complexity) pairs; unparseable fields sort first."""
    size, comp = combo
    return (float(size) if size else -1.0, float(comp) if comp else -1.0)


def select_balanced(
    paths: list[str], per_combo: int, seed: int | None = None,
    is_done: Callable[[str], bool] | None = None,
) -> tuple[list[str], dict[tuple[str, str], tuple[int, int]]]:
    """Top each (size, complexity) cell up to `per_combo` trajectories on disk.

    Gives every cell of the size x complexity grid the same target, so the hard
    limit is per_combo * (number of cells) rather than a global count that would
    over-sample whichever cells happen to have the most runs.

    `is_done` reports whether a trajectory has already been processed. Those
    count towards the cell's target and are never re-selected, so a cell already
    at (or past) `per_combo` is left alone rather than redone or trimmed, and a
    partially filled cell only gets the difference. Without it, every trajectory
    counts as pending and each cell selects a full `per_combo`.

    Without a seed, takes the lowest run indices in each cell (deterministic and
    stable as new runs land); with a seed, samples inside each cell instead.
    Cells with too few trajectories contribute all they have.

    Returns the selected paths (sorted) and, per cell, (already done, selected).
    """
    by_combo: dict[tuple[str, str], list[str]] = {}
    for p in paths:
        name = parse_name(Path(p).stem)
        by_combo.setdefault((name["size"], name["comp"]), []).append(p)

    rng = random.Random(seed) if seed is not None else None
    kept: list[str] = []
    counts: dict[tuple[str, str], tuple[int, int]] = {}
    for combo in sorted(by_combo, key=combo_sort_key):
        group = by_combo[combo]
        done = [p for p in group if is_done(p)] if is_done is not None else []
        pending = [p for p in group if p not in set(done)]
        quota = max(0, per_combo - len(done))
        if len(pending) <= quota:
            chosen = pending
        elif rng is not None:
            chosen = rng.sample(pending, quota)
        else:
            # sort by run index numerically: lexicographic order would put run 100 before 99
            chosen = sorted(pending, key=lambda p: int(parse_name(Path(p).stem)["run"] or 0))[:quota]
        kept.extend(chosen)
        counts[combo] = (len(done), len(chosen))
    return sorted(kept), counts


def trajectory_activation_dir(activations_dir: Path, stem: str) -> Path:
    """<activations_dir>/size{S}/<stem>, mirroring the gather_activations layout.

    Falls back to <activations_dir>/<stem> for trajectories whose filename carries
    no size{N} (rather than creating a bare `size/` folder).
    """
    size = parse_name(stem)["size"]
    return activations_dir / f"size{size}" / stem if size else activations_dir / stem


def jlens_csv_path(activations_dir: Path, stem: str) -> Path:
    """The trajectory's jlens CSV; its existence is what marks the run as done."""
    return trajectory_activation_dir(activations_dir, stem) / f"{stem}_jlens_analysis.csv"


def group_consecutive(sizes: list[int], max_items: int, max_tokens: int) -> list[list[int]]:
    """Split 0..len(sizes)-1 into consecutive groups honouring both budgets.

    Used to pack several steps into one padded forward pass. Sequences are padded to the
    longest member, so a group of k sequences costs `k * max(sizes in group)` tokens.
    Groups are always consecutive and never reordered, which keeps every downstream
    artefact (CSV rows, .pt folders) in step order.

    A single sequence larger than `max_tokens` still gets its own group rather than being
    dropped.

    >>> group_consecutive([10, 10, 10, 10], max_items=2, max_tokens=10_000)
    [[0, 1], [2, 3]]
    >>> group_consecutive([10, 10, 90], max_items=8, max_tokens=100)
    [[0, 1], [2]]
    >>> group_consecutive([500], max_items=8, max_tokens=100)
    [[0]]
    """
    groups: list[list[int]] = []
    current: list[int] = []
    current_max = 0
    for i, n in enumerate(sizes):
        candidate_max = max(current_max, n)
        if current and (len(current) + 1 > max_items or (len(current) + 1) * candidate_max > max_tokens):
            groups.append(current)
            current, candidate_max = [], n
        current.append(i)
        current_max = candidate_max
    if current:
        groups.append(current)
    return groups


def resolve_trajectory_paths(args, expand_paths: Callable[[list[str]], list[str]]) -> list[str]:
    """Which trajectories this invocation will process, and why.

    Expands --trajectory-paths, applies the size/complexity filters, then whichever cap
    was asked for. `expand_paths` is passed in because it lives in a module that imports
    transformers at import time, which main() defers.
    """
    paths = expand_paths(args.trajectory_paths)
    if args.sizes or args.complexities:
        sizes = {s.strip() for s in args.sizes.split(",")} if args.sizes else None
        comps = {c.strip() for c in args.complexities.split(",")} if args.complexities else None
        paths = [
            p for p in paths
            if (sizes is None or parse_name(Path(p).stem)["size"] in sizes)
            and (comps is None or parse_name(Path(p).stem)["comp"] in comps)
        ]
        print(f"{len(paths)} after size/complexity filter", flush=True)
    if args.per_combo is not None:
        # --overwrite means "rebuild", so nothing counts as done and each cell
        # re-selects a full per_combo instead of topping up what is on disk.
        is_done = None if args.overwrite else (
            lambda p: jlens_csv_path(args.activations_dir, Path(p).stem).exists()
        )
        paths, counts = select_balanced(paths, args.per_combo, args.seed, is_done)
        print(f"{len(counts)} size x complexity cell(s), target {args.per_combo} each:", flush=True)
        for combo in sorted(counts, key=combo_sort_key):
            size, comp = combo
            done, added = counts[combo]
            short = "  <-- SHORT (no more trajectories)" if done + added < args.per_combo else ""
            print(f"  size{size or '?'} comp{comp or '?'}: {done} done + {added} new "
                  f"= {done + added}{short}", flush=True)
    elif args.max_trajectories is not None and args.max_trajectories < len(paths):
        if args.seed is not None:
            paths = sorted(random.Random(args.seed).sample(paths, args.max_trajectories))
        else:
            paths = paths[: args.max_trajectories]
    print(f"{len(paths)} trajectory file(s)", flush=True)
    return paths


def apply_lens_transport(h, J_stack, J_rows, norm_w, eps):
    """Transport a [L, b, d] stack of residual streams into the lens target space.

    One `bmm` covers every layer that has a jlens matrix, replacing one `h @ J.T` per
    layer; rows not listed in `J_rows` pass straight through, which is the TARGET_LAYER
    case where the lens is the identity. Then the model's final RMS norm, so the caller
    is left with just the unembed.

    Mutates `h` in place — callers build it fresh from the residual streams each chunk.

    Args:
        h: [L, b, d] fp32 stack, layer-major, in the same order as `lens_layers`
        J_stack: [T, d, d] jlens matrices, in `J_rows` order
        J_rows: rows of `h` that `J_stack` applies to
        norm_w: the model's final RMS norm weight, [d]
        eps: the model's rms_norm_eps

    Returns:
        [L, b, d] normed activations ready for the unembed
    """
    import torch

    if J_rows.numel():
        h[J_rows] = torch.bmm(h[J_rows], J_stack.transpose(1, 2))
    return h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * norm_w


class Profiler:
    """Wall-clock accounting per phase, off by default.

    Timing GPU work needs a device sync to mean anything, and a sync on the hot path is
    exactly what we are trying to avoid, so the whole thing is inert unless --profile.
    """

    BUCKETS = ("build", "forward", "save_pt", "drain_pt", "lens", "rows", "csv")

    def __init__(self, enabled: bool, sync: bool) -> None:
        self.enabled = enabled
        self.sync = sync
        self.totals: dict[str, float] = defaultdict(float)
        self.run_totals: dict[str, float] = defaultdict(float)
        self.steps = 0
        self.tokens = 0
        self.run_steps = 0
        self.run_tokens = 0

    @contextmanager
    def __call__(self, bucket: str):
        if not self.enabled:
            yield
            return
        import torch

        if self.sync:
            torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            if self.sync:
                torch.cuda.synchronize()
            self.totals[bucket] += time.perf_counter() - start

    def report(self, label: str, reset: bool = True) -> None:
        """Print the accumulated split; by default fold it into the run total and reset."""
        if not self.enabled:
            return
        total = sum(self.totals.values())
        if total:
            per_step = f", {total / self.steps:.3f}s/step" if self.steps else ""
            print(f"  [profile] {label}: {total:.2f}s over {self.steps} step(s), "
                  f"{self.tokens} reasoning token(s){per_step}", flush=True)
            for bucket in self.BUCKETS:
                seconds = self.totals.get(bucket, 0.0)
                if seconds:
                    print(f"    {bucket:<9} {seconds:7.2f}s  {100 * seconds / total:5.1f}%", flush=True)
        if reset:
            for bucket, seconds in self.totals.items():
                self.run_totals[bucket] += seconds
            self.run_steps += self.steps
            self.run_tokens += self.tokens
            self.totals.clear()
            self.steps = self.tokens = 0

    def report_run(self) -> None:
        """Print the whole run, folding in whatever has not been reported yet."""
        if not self.enabled:
            return
        self.report("", reset=True)
        self.totals.update(self.run_totals)
        self.steps, self.tokens = self.run_steps, self.run_tokens
        self.report("run total", reset=False)


def benchmark_pt_write(out_dir: Path, count: int, dim: int = 2880) -> None:
    """Time both torch.save container formats on this host's filesystem, then clean up.

    The per-token .pt tree is the one part of this script whose cost is set by the
    machine rather than by the model, so it is worth measuring before tuning --io-workers
    or switching --pt-format.
    """
    import shutil

    import torch

    bench_dir = out_dir / "_pt_write_benchmark"
    shutil.rmtree(bench_dir, ignore_errors=True)
    try:
        for use_zip in (True, False):
            target = bench_dir / ("zip" if use_zip else "legacy")
            target.mkdir(parents=True, exist_ok=True)
            tensors = [torch.randn(dim, dtype=torch.bfloat16) for _ in range(count)]
            start = time.perf_counter()
            for i, tensor in enumerate(tensors):
                torch.save(tensor, target / f"{i}.pt", _use_new_zipfile_serialization=use_zip)
            elapsed = time.perf_counter() - start
            size = sum(p.stat().st_size for p in target.iterdir())
            print(f"  {'zip' if use_zip else 'legacy':<7} {count} files in {elapsed:.2f}s "
                  f"({count / elapsed:.0f} files/s, {size / count:.0f} B/file)", flush=True)
    finally:
        shutil.rmtree(bench_dir, ignore_errors=True)


def main() -> None:
    import torch
    from scripts.inference_oss.run_inference import expand_paths
    from scripts.jlens_action_ranks_sampled import action_token_ids, ensure_unembed_assets
    from telos_interp.commands.gather_activations.gather_activations_fn import _resolve_torch_dtype
    from telos_interp.commands.gather_activations.gather_activations_utils import (
        ActivationWriter,
        extract_activations_batched,
        parse_index_specification,
        sanitize_model_id,
    )
    from transformers import AutoModelForCausalLM

    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory-paths", nargs="+", required=True,
                    help="Trajectory JSON file(s), directory, or glob(s).")
    ap.add_argument("--jlens_dir", type=Path, required=True)
    ap.add_argument("--activations-dir", type=Path, required=True,
                    help="Base dir for the gather_activations-style tree: per-trajectory "
                         "activations under size{S}/{stem}/{model}/layer_N/step_M/output/ and the "
                         "trajectory's jlens CSV at size{S}/{stem}/{stem}_jlens_analysis.csv.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Reprocess trajectories whose jlens CSV already exists (default: skip).")
    ap.add_argument("--sizes", default=None,
                    help="Comma-separated grid sizes to keep (matched against size{S} in the "
                         "filename), e.g. '11,15'. Default: all sizes.")
    ap.add_argument("--complexities", default=None,
                    help="Comma-separated complexities to keep (matched against comp{C}), "
                         "e.g. '0.0,1.0'. Default: all complexities.")
    ap.add_argument("--batch-size", type=int, default=256,
                    help="Reasoning tokens per lens matmul (caps the [B, vocab] logits). "
                         "Default 256; 0 means a whole step's reasoning chain at once, which "
                         "on a long chain allocates gigabytes of logits.")
    ap.add_argument("--max-trajectories", type=int, default=None,
                    help="Process at most N trajectory files (default: all). Global cap: use "
                         "--per-combo instead to spread the budget over the size x complexity grid.")
    ap.add_argument("--per-combo", type=int, default=None,
                    help="Top each (size, complexity) cell up to N processed trajectories, i.e. a "
                         "hard limit of N * (number of cells) evenly spread across the grid "
                         "(e.g. 200 over a 6x6 grid = 7200). Trajectories that already have a CSV "
                         "count towards N and are never redone, so a cell at or past N is skipped "
                         "and a partial cell only gets the difference. Mutually exclusive with "
                         "--max-trajectories.")
    ap.add_argument("--seed", type=int, default=None,
                    help="With --max-trajectories/--per-combo, randomly sample using this seed "
                         "(default: take the lowest run indices).")
    ap.add_argument("--layers", default="all",
                    help="Comma/range spec of layer indices, or 'all' (default).")
    ap.add_argument("--steps", default="all",
                    help="Comma/range spec of step indices, or 'all' (default).")
    ap.add_argument("--forward-batch-size", type=int, default=4,
                    help="Steps packed into one padded forward pass (default 4; 1 disables "
                         "packing). Sequences are right-padded, which under causal attention "
                         "cannot change a real token's activations, but batched GEMMs do "
                         "reassociate bf16 reductions - use 1 for bit-exact agreement with an "
                         "unbatched run.")
    ap.add_argument("--forward-batch-tokens", type=int, default=16384,
                    help="Padded-token budget per forward batch (default 16384). A group costs "
                         "len(group) * longest(group) tokens; whichever of this and "
                         "--forward-batch-size binds first closes the group.")
    ap.add_argument("--io-workers", type=int, default=16,
                    help="Threads writing the per-token .pt files (default 16; 0 writes inline "
                         "on the main thread). These writes are the dominant cost on long "
                         "reasoning chains - one file per (token, layer) - and overlap with the "
                         "next forward pass.")
    ap.add_argument("--pt-format", choices=["zip", "legacy"], default="zip",
                    help="torch.save container for the per-token files. 'legacy' skips the zip "
                         "directory; both load under weights_only=True. Measure with "
                         "--benchmark-pt-write before switching.")
    ap.add_argument("--no-save-activations", dest="save_activations", action="store_false",
                    help="Write only the jlens CSV, no .pt files. Diagnostic: the difference "
                         "against a normal run is exactly what the activation tree costs.")
    ap.add_argument("--profile", action="store_true",
                    help="Report per-phase wall time. Adds CUDA syncs, so timings are honest "
                         "but the run is slightly slower.")
    ap.add_argument("--benchmark-pt-write", type=int, default=None,
                    help="Time N per-token .pt writes in both container formats under "
                         "--activations-dir, print the rates, and exit.")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--torch-dtype", default="auto")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device for the lens matmuls (defaults to cuda if available).")
    args = ap.parse_args()
    if args.benchmark_pt_write is not None:
        args.activations_dir.mkdir(parents=True, exist_ok=True)
        print(f"benchmarking {args.benchmark_pt_write} .pt writes under {args.activations_dir}", flush=True)
        benchmark_pt_write(args.activations_dir, args.benchmark_pt_write)
        return
    if args.per_combo is not None and args.max_trajectories is not None:
        ap.error("--per-combo and --max-trajectories are mutually exclusive")
    paths = resolve_trajectory_paths(args, expand_paths)

    with open(paths[0]) as f:
        model_id = json.load(f)["model_params"]["model_id"]

    print(f"loading model: {model_id}", flush=True)
    resolved = _resolve_torch_dtype(args.torch_dtype, model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map=args.device_map, dtype=resolved if resolved is not None else "auto"
    )
    model.eval()
    sanitized_model = sanitize_model_id(model_id)
    num_layers = model.config.num_hidden_layers
    layer_indices = parse_index_specification(args.layers, num_layers)
    print(f"{num_layers} layers; extracting {layer_indices}", flush=True)

    print("loading jlens + unembed assets...", flush=True)
    lens = torch.load(args.jlens_dir / "gpt-oss-20b_jacobian_lens.pt", map_location="cpu")
    assets = ensure_unembed_assets(args.jlens_dir)
    ids, tok = action_token_ids()
    id_cols = [ids[a] for a in ACTIONS]

    dev = torch.device(args.device)
    lm_head = assets["lm_head"].to(dev)
    norm_w = assets["norm_weight"].float().to(dev)
    eps = assets["rms_eps"]

    # The lens matrices never change, so they move to the device once for the whole run
    # rather than once per step (17 layers x 2880^2 fp32 is ~566 MB of H2D per step).
    # A layer with no J is dropped from the CSV exactly as before, but still gets its .pt;
    # TARGET_LAYER is the identity case and is checked first even if lens["J"] has an
    # entry for it.
    lens_layers = [i for i in layer_indices if i == TARGET_LAYER or i in lens["J"]]
    if not lens_layers:
        raise SystemExit(f"none of the requested layers {layer_indices} have a jlens matrix")
    transported = [i for i in lens_layers if i != TARGET_LAYER]
    J_stack = (
        torch.stack([lens["J"][i].float() for i in transported]).to(dev)
        if transported else torch.empty(0, device=dev)
    )
    # rows of the per-chunk [len(lens_layers), b, d] stack that J applies to
    J_rows = torch.tensor([lens_layers.index(i) for i in transported], dtype=torch.long, device=dev)
    skipped = sorted(set(layer_indices) - set(lens_layers))
    if skipped:
        print(f"no jlens matrix for layers {skipped}: activations saved, no CSV rows", flush=True)

    # tok.decode() is a round-trip into the Rust tokenizer, and the row loop asks for TOP_K
    # of them per row - millions per trajectory on a long chain. The top-k sets repeat
    # heavily across rows and layers, so a memo turns nearly all of that into a dict hit.
    decoded: dict[int, str] = {}

    def decode_id(token_id: int) -> str:
        text = decoded.get(token_id)
        if text is None:
            text = decoded[token_id] = tok.decode([token_id])
        return text

    prof = Profiler(args.profile, args.profile and dev.type == "cuda")
    writer_pool = ActivationWriter(
        max_workers=args.io_workers if args.save_activations else 0,
        use_zipfile=args.pt_format == "zip",
    )

    header = (
        ["size", "complexity", "run", "step", "reasoning_pos", "abs_pos", "token", "layer",
         "agent_action"]
        + [f"{a}_rank" for a in ACTIONS]
        + [f"{a}_logprob" for a in ACTIONS]
        + [f"top_{i}" for i in range(1, TOP_K + 1)]
    )
    written = 0

    for traj_path in paths:
        stem = Path(traj_path).stem
        traj_dir = trajectory_activation_dir(args.activations_dir, stem)
        csv_path = jlens_csv_path(args.activations_dir, stem)
        if csv_path.exists() and not args.overwrite:
            print(f"{stem}: CSV exists, skipping (use --overwrite to redo)", flush=True)
            continue

        with open(traj_path) as f:
            trajectory = json.load(f)
        name = parse_name(stem)
        n_steps = len(trajectory["steps"])
        step_idxs = parse_index_specification(args.steps, n_steps, clamp=True)
        print(f"{stem}: {len(step_idxs)}/{n_steps} steps", flush=True)

        n_prefix = len(trajectory["prompt"]["prompt_prefix_tokens"])
        n_suffix = len(trajectory["prompt"]["prompt_suffix_tokens"])
        output_base = traj_dir / sanitized_model
        traj_dir.mkdir(parents=True, exist_ok=True)
        # write to a temp file and rename at the end, so a crashed run leaves no CSV
        # and is redone on the next invocation instead of being skipped as done
        tmp_path = csv_path.with_suffix(".csv.tmp")
        nan_count = 0
        total_count = 0

        with open(tmp_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)

            # Everything a step needs, resolved up front so steps can be grouped by the
            # length of the sequence they will need before any of them runs.
            with prof("build"):
                prefix_ids = [t["token_id"] for t in trajectory["prompt"]["prompt_prefix_tokens"]]
                suffix_ids = [t["token_id"] for t in trajectory["prompt"]["prompt_suffix_tokens"]]
                pending = []
                for si in step_idxs:
                    step = trajectory["steps"][si]
                    positions = reasoning_token_positions(trajectory, step)
                    if not positions:
                        continue
                    abs_positions = [p[1] for p in positions]
                    # full prompt = prefix + grid + suffix + output; truncate at the last needed pos
                    all_ids = (
                        prefix_ids
                        + [t["token_id"] for t in step["grid_state_tokens"]]
                        + suffix_ids
                        + [t["token_id"] for t in step["output_tokens"]]
                    )
                    pending.append({
                        "si": si,
                        "step_id": step["step_id"],
                        "agent_action": step.get("agent_action", ""),
                        "positions": positions,
                        "abs_positions": abs_positions,
                        "output_start": n_prefix + len(step["grid_state_tokens"]) + n_suffix,
                        "ids": all_ids[: max(abs_positions) + 1],
                    })

            groups = group_consecutive(
                [len(p["ids"]) for p in pending],
                max_items=max(1, args.forward_batch_size),
                max_tokens=max(1, args.forward_batch_tokens),
            )

            for group in groups:
                batch = [pending[i] for i in group]
                with prof("build"):
                    width = max(len(item["ids"]) for item in batch)
                    # right padding: under causal attention a real token at position i only
                    # attends to positions <= i, all of which are real, so the pad id is
                    # never read and the activations match an unpadded pass
                    input_ids = torch.zeros((len(batch), width), dtype=torch.long)
                    attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
                    padded = False
                    for row, item in enumerate(batch):
                        n = len(item["ids"])
                        input_ids[row, :n] = torch.tensor(item["ids"], dtype=torch.long)
                        attention_mask[row, :n] = 1
                        padded |= n < width
                    # nothing padded -> no mask, which keeps the model on the same code path
                    # (and so bit-identical) as an unbatched --forward-batch-size 1 run
                    if not padded:
                        attention_mask = None

                with prof("forward"):
                    # one (N, d) tensor per layer, rows in batch order; left on the device
                    # because the lens is about to read them
                    blocks = extract_activations_batched(
                        model, input_ids, attention_mask,
                        [item["abs_positions"] for item in batch],
                        layer_indices, keep_on_device=True,
                    )

                offset = 0
                for item in batch:
                    positions = item["positions"]
                    n_pos = len(positions)
                    rows = slice(offset, offset + n_pos)
                    offset += n_pos
                    prof.steps += 1
                    prof.tokens += n_pos

                    # Save the raw residual streams in gather_activations' layout before the
                    # lens math, so layers without a jlens matrix still get written.
                    # Reasoning tokens are output tokens, keyed by output-relative index.
                    if args.save_activations:
                        with prof("save_pt"):
                            output_start = item["output_start"]
                            for layer in layer_indices:
                                block = blocks[layer][rows].to("cpu")
                                writer_pool.submit(
                                    # clone: torch.save serialises a view's whole storage
                                    {layer: {ap - output_start: block[j].clone()
                                             for j, ap in enumerate(item["abs_positions"])}},
                                    output_base, step_idx=item["step_id"], category="output",
                                )
                            total_count += n_pos * len(layer_indices)

                    # chunk reasoning tokens so [B, vocab] logits stay bounded (bs=0 -> whole step)
                    bs = args.batch_size or n_pos
                    # buffered per layer so the CSV keeps its layer-major row order while the
                    # lens transport runs as one bmm across layers
                    rows_by_layer: dict[int, list] = {layer: [] for layer in lens_layers}
                    row_prefix = [name["size"], name["comp"], name["run"], item["si"]]
                    for i in range(0, n_pos, bs):
                        chunk = positions[i : i + bs]
                        with torch.no_grad():
                            with prof("lens"):
                                # [L, b, d] straight off the device, no host round-trip
                                h = torch.stack([blocks[layer][rows][i : i + bs] for layer in lens_layers]).float()
                                h = apply_lens_transport(h, J_stack, J_rows, norm_w, eps)
                            for li, layer in enumerate(lens_layers):
                                with prof("lens"):
                                    logits = (h[li].to(lm_head.dtype) @ lm_head.T).float()  # [b, vocab]
                                    ranks = torch.stack(
                                        [(logits > logits[:, t : t + 1]).sum(1) for t in id_cols], dim=1
                                    )  # [b, 4], rank 0 = argmax
                                    own = torch.stack([logits[:, t] for t in id_cols], dim=1)
                                    logprobs = own - logits.logsumexp(-1, keepdim=True)
                                    topk = logits.topk(TOP_K, dim=1).indices  # [b, TOP_K]
                                    ranks = ranks.cpu().tolist()
                                    logprobs = logprobs.cpu().tolist()
                                    topk = topk.cpu().tolist()
                                    del logits, own
                                with prof("rows"):
                                    rows_by_layer[layer].extend(
                                        row_prefix + [rp, ap, token, layer, item["agent_action"]]
                                        + r
                                        + [round(x, 4) for x in lp]
                                        + [decode_id(t) for t in tk]
                                        for (rp, ap, token), r, lp, tk in zip(chunk, ranks, logprobs, topk, strict=True)
                                    )
                    with prof("csv"):
                        for layer in lens_layers:
                            writer.writerows(rows_by_layer[layer])
                del blocks

        with prof("drain_pt"):
            nan_count += writer_pool.drain()
        os.replace(tmp_path, csv_path)
        written += 1
        print(f"  wrote {csv_path} ({total_count} activations)", flush=True)
        if nan_count > 0:
            print(f"  WARNING: {nan_count}/{total_count} activations contain NaN values!")
            print("  This is often caused by multi-GPU setups. Try using CUDA_VISIBLE_DEVICES=0")
        prof.report(stem)

    writer_pool.close()
    prof.report_run()
    print(f"done: {written} trajectory folder(s) under {args.activations_dir}", flush=True)


def _self_test() -> None:
    """Position math only — no model, no torch load. Run: python ... --self-test."""
    traj = {"prompt": {"prompt_prefix_tokens": [{}] * 3, "prompt_suffix_tokens": [{}] * 2}}
    step = {
        "grid_state_tokens": [{}] * 4,  # output_start = 3 + 4 + 2 = 9
        "output_tokens": [
            {"token": "a", "token_groups": ["analysis"]},
            {"token": "b", "token_groups": ["final", "action"]},
            {"token": "c", "token_groups": ["analysis"]},
        ],
    }
    pos = reasoning_token_positions(traj, step)
    assert pos == [(0, 9, "a"), (1, 11, "c")], pos
    # no analysis tags -> all output tokens
    step2 = {"grid_state_tokens": [], "output_tokens": [{"token": "x", "token_groups": []}]}
    traj2 = {"prompt": {"prompt_prefix_tokens": [], "prompt_suffix_tokens": []}}
    assert reasoning_token_positions(traj2, step2) == [(0, 0, "x")]
    assert parse_name("foo_size11_comp1.0_987") == {"size": "11", "comp": "1.0", "run": "987"}
    # output layout mirrors gather_activations: <root>/size{S}/<stem>/...
    stem = "together_ai_openai_gpt-oss-20b_size11_comp1.0_987"
    assert trajectory_activation_dir(Path("/acts"), stem) == Path("/acts/size11") / stem
    assert trajectory_activation_dir(Path("/acts"), "nosize") == Path("/acts/nosize")

    # balanced selection: 2 per (size, complexity) cell, lowest run indices first
    def p(size, comp, run):
        return f"/t/size{size}/m_size{size}_comp{comp}_{run}.json"

    pool = (
        [p(11, "1.0", r) for r in (5, 99, 100, 1)]  # 4 runs -> 2 kept
        + [p(5, "0.0", r) for r in (7, 3)]  # exactly 2 -> both kept
        + [p(5, "0.5", 9)]  # short cell -> the one it has
    )
    kept, counts = select_balanced(pool, 2)
    assert counts == {("5", "0.0"): (0, 2), ("5", "0.5"): (0, 1), ("11", "1.0"): (0, 2)}, counts
    assert kept == sorted([p(11, "1.0", 1), p(11, "1.0", 5), p(5, "0.0", 3), p(5, "0.0", 7), p(5, "0.5", 9)]), kept
    seeded, seeded_counts = select_balanced(pool, 2, seed=0)
    assert seeded_counts == counts and len(seeded) == 5, (seeded, seeded_counts)
    assert seeded == select_balanced(pool, 2, seed=0)[0]  # same seed -> same picks
    assert select_balanced(pool, 100)[0] == sorted(pool)  # target above every cell keeps everything
    assert combo_sort_key(("11", "1.0")) > combo_sort_key(("5", "1.0"))  # numeric, not lexicographic

    # already-processed trajectories count towards the target instead of being redone
    done = {p(11, "1.0", 5), p(5, "0.0", 7), p(5, "0.5", 9)}
    kept, counts = select_balanced(pool, 2, is_done=done.__contains__)
    assert counts == {("5", "0.0"): (1, 1), ("5", "0.5"): (1, 0), ("11", "1.0"): (1, 1)}, counts
    assert not done & set(kept), kept  # never re-selects what is done
    assert kept == sorted([p(5, "0.0", 3), p(11, "1.0", 1)]), kept
    # a cell already at (or past) the target is left alone, extras and all
    full = {p(11, "1.0", r) for r in (1, 5, 99, 100)}
    kept, counts = select_balanced([p(11, "1.0", r) for r in (1, 5, 99, 100)], 2, is_done=full.__contains__)
    assert kept == [] and counts == {("11", "1.0"): (4, 0)}, (kept, counts)
    assert select_balanced(pool, 2, is_done=lambda _: False) == select_balanced(pool, 2)

    # forward-pass grouping: consecutive only, and both budgets bind
    assert group_consecutive([10, 10, 10, 10], 2, 10_000) == [[0, 1], [2, 3]]
    assert group_consecutive([10, 10, 90], 8, 100) == [[0, 1], [2]]
    assert group_consecutive([500], 8, 100) == [[0]]  # oversized sequence still runs
    assert group_consecutive([], 4, 100) == []
    assert group_consecutive([7, 3, 9], 1, 10_000) == [[0], [1], [2]]  # batch size 1 = off
    # every step lands in exactly one group, in order
    sizes = [11, 40, 5, 900, 7, 7]
    flat = [i for g in group_consecutive(sizes, 3, 200) for i in g]
    assert flat == list(range(len(sizes))), flat

    _lens_transport_self_test()
    print("self-test ok")


def _lens_transport_self_test() -> None:
    """Batched transport == the per-layer loop it replaced. Skipped if torch is absent."""
    try:
        import torch
    except ImportError:
        print("torch not installed, skipping lens transport check")
        return

    torch.manual_seed(0)
    d, b, eps = 16, 5, 1e-5
    # three layers, the last one standing in for TARGET_LAYER (identity, no J)
    Js = [torch.randn(d, d), torch.randn(d, d)]
    h_rows = [torch.randn(b, d) for _ in range(3)]
    norm_w = torch.randn(d)

    expected = []
    for i, h in enumerate(h_rows):
        x = h @ Js[i].T if i < len(Js) else h
        expected.append(x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * norm_w)

    got = apply_lens_transport(
        torch.stack(h_rows),
        torch.stack(Js),
        torch.tensor([0, 1], dtype=torch.long),
        norm_w,
        eps,
    )
    assert torch.allclose(got, torch.stack(expected), atol=1e-6), (got - torch.stack(expected)).abs().max()
    # the untransported row must pass through untouched, not merely close
    assert torch.equal(got[2], expected[2])


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        main()
