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

Selective gathering
-------------------
Saving every (token, layer) is what fills the disk: a 700-token chain over 17
layers is ~12k files and ~68 MB *per step*, of which a probe reads maybe 40
tokens' worth. Pass --signal-json to keep only what a probe will actually use:

  1. the full CSV is written as before (nothing is lost analytically),
  2. `jlens_top_filter` picks the top --select-num-tokens tokens and their top
     --select-num-layers layers, plus --select-always-layers (15) and a matched
     --select-random-tokens control arm,
  3. a second, much cheaper forward pass saves just those,
  4. `{stem}_jlens_selection.json` records the choice, which is what
     `prepare_activations_for_probing --token-selection recorded_*` reads.

The control arm is not optional bookkeeping: once the tree holds only the
top-scoring tokens, a uniform draw over the reasoning chain can never be made
again, so it has to be reserved before the rest is dropped.

`scripts/delete_non_jlens_selected.py` applies the same filter to trajectories
that were already gathered in full, and lands on the same files.

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
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

# telos_interp is an installed package; jlens_utils is stdlib-only, so importing it here
# costs nothing and keeps --self-test runnable without torch.
from telos_interp.jlens_utils import (
    DEFAULT_ALWAYS_LAYERS,
    DEFAULT_METHODS,
    METHODS,
    build_record,
    get_method,
    merge_records,
    parse_methods,
    read_raw_record,
    read_selection_record,
    record_path,
    to_disk_coords,
    top_filter,
    write_selection_record,
)

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


def analysis_csv(activations_dir: Path, stem: str, lens: str) -> Path:
    """The trajectory's CSV for one lens; its existence is what marks that lens as done.

    Named by `METHODS[lens].csv_suffix`, so the gather script, the pruner and
    `prepare_activations_for_probing` all agree on the filename without repeating it.
    """
    return trajectory_activation_dir(activations_dir, stem) / f"{stem}{METHODS[lens].csv_suffix}"


def jlens_csv_path(activations_dir: Path, stem: str) -> Path:
    """Back-compat alias for the Jacobian lens CSV."""
    return analysis_csv(activations_dir, stem, "jlens")


def parse_lenses(spec: str) -> list[str]:
    """--lens value to the list of lenses to compute.

    >>> parse_lenses("both")
    ['jlens', 'logitlens']
    >>> parse_lenses("logitlens")
    ['logitlens']
    """
    return ["jlens", "logitlens"] if spec == "both" else [spec]


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
        # Done means *every* requested lens has a CSV: a trajectory analysed with the jlens
        # alone is unfinished under --lens both, and must not count towards the cell.
        is_done = None if args.overwrite else (
            lambda p: all(
                analysis_csv(args.activations_dir, Path(p).stem, lens).exists()
                for lens in parse_lenses(args.lens)
            )
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

    **This is also the logit lens.** Pass an empty `J_rows` and nothing is transported: each
    layer's residual stream is normed and unembedded exactly where it sits, which is what
    the logit lens is. The two lenses are therefore the same function called two ways, not
    two implementations — the only other difference is which layers are eligible (the jlens
    needs a fitted `J`, the logit lens does not).

    Mutates `h` in place — callers build it fresh from the residual streams each chunk.

    Args:
        h: [L, b, d] fp32 stack, layer-major, in the same order as the caller's layer list
        J_stack: [T, d, d] jlens matrices, in `J_rows` order
        J_rows: rows of `h` that `J_stack` applies to; empty for the logit lens
        norm_w: the model's final RMS norm weight, [d]
        eps: the model's rms_norm_eps

    Returns:
        [L, b, d] normed activations ready for the unembed
    """
    import torch

    if J_rows.numel():
        h[J_rows] = torch.bmm(h[J_rows], J_stack.transpose(1, 2))
    return h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * norm_w


def lens_predictions(h_layer, lm_head, id_cols):
    """Unembed one layer's normed activations into (action ranks, action logprobs, top-k ids).

    Split out of the row loop because it is the only part that differs per layer rather than
    per lens, and because the `[b, vocab]` logits it allocates are the reason `--batch-size`
    exists.

    Returns plain Python lists — everything downstream is CSV text.
    """
    import torch

    logits = (h_layer.to(lm_head.dtype) @ lm_head.T).float()  # [b, vocab]
    ranks = torch.stack([(logits > logits[:, t : t + 1]).sum(1) for t in id_cols], dim=1)  # rank 0 = argmax
    own = torch.stack([logits[:, t] for t in id_cols], dim=1)
    logprobs = own - logits.logsumexp(-1, keepdim=True)
    topk = logits.topk(TOP_K, dim=1).indices  # [b, TOP_K]
    out = (ranks.cpu().tolist(), logprobs.cpu().tolist(), topk.cpu().tolist())
    del logits, own, ranks, logprobs, topk
    return out


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


def parse_always_layers(spec: str) -> tuple[int, ...]:
    """Comma-separated layer list for --select-always-layers; empty string means none.

    >>> parse_always_layers("15")
    (15,)
    >>> parse_always_layers(" 7, 15 ")
    (7, 15)
    >>> parse_always_layers("")
    ()
    """
    return tuple(int(part) for part in spec.split(",") if part.strip())


def build_lens_transports(lenses: list[str], layer_indices: list[int], jlens_dir: Path, dev):
    """Per lens: the layers it can score, and the (J_stack, J_rows) transport it applies.

    The two lenses differ in exactly two ways, both captured here:

      * **Eligible layers.** The Jacobian lens needs a fitted `J` for a layer (TARGET_LAYER
        excepted, where it is the identity), so a layer without one gets no jlens CSV rows —
        it still gets its `.pt`, it just cannot be scored, and so must never be selectable.
        The logit lens needs nothing fitted and therefore covers every requested layer.
      * **Transport.** The jlens moves each layer into the layer-23 space first; the logit
        lens does not. Empty J tensors make `apply_lens_transport` the identity, which *is*
        the logit lens — the two share one code path rather than having two implementations.

    Returns:
        ({lens: layers}, {lens: (J_stack, J_rows)}), ready for the per-chunk loop. The J
        matrices move to the device once for the whole run rather than once per step (17
        layers x 2880^2 fp32 is ~566 MB of H2D per step).
    """
    import torch

    empty_J = torch.empty(0, device=dev)
    empty_rows = torch.tensor([], dtype=torch.long, device=dev)
    layers_by_lens: dict[str, list[int]] = {}
    transport_by_lens: dict[str, tuple] = {}

    if "logitlens" in lenses:
        layers_by_lens["logitlens"] = list(layer_indices)
        transport_by_lens["logitlens"] = (empty_J, empty_rows)

    if "jlens" in lenses:
        lens = torch.load(jlens_dir / "gpt-oss-20b_jacobian_lens.pt", map_location="cpu")
        jlens_layers = [i for i in layer_indices if i == TARGET_LAYER or i in lens["J"]]
        if not jlens_layers:
            raise SystemExit(f"none of the requested layers {layer_indices} have a jlens matrix")
        transported = [i for i in jlens_layers if i != TARGET_LAYER]
        J_stack = (
            torch.stack([lens["J"][i].float() for i in transported]).to(dev)
            if transported else empty_J
        )
        # rows of the per-chunk [len(jlens_layers), b, d] stack that J applies to
        J_rows = torch.tensor([jlens_layers.index(i) for i in transported], dtype=torch.long, device=dev)
        layers_by_lens["jlens"] = jlens_layers
        transport_by_lens["jlens"] = (J_stack, J_rows)
        skipped = sorted(set(layer_indices) - set(jlens_layers))
        if skipped:
            print(f"no jlens matrix for layers {skipped}: activations saved, no jlens CSV rows", flush=True)

    return layers_by_lens, transport_by_lens


def validate_selection_args(args, selective: bool, select_methods: list[str], lenses: list[str],
                            always_layers: tuple[int, ...]) -> None:
    """Reject impossible selection flag combinations, and say what this run will select.

    The one that matters is a scored arm whose lens is not being computed: without its CSV
    that arm would come back silently empty after a full forward pass.
    """
    if args.extend and not selective:
        raise SystemExit("--extend needs --signal-json: there is no selection to extend without one")
    if args.dry_run and not args.extend:
        raise SystemExit("--dry-run currently only applies to --extend")
    if not selective:
        return

    scored = [m for m in select_methods if get_method(m).scored]
    unscored = [m for m in select_methods if not get_method(m).scored]
    missing_csv = [m for m in scored if m not in lenses]
    if missing_csv:
        raise SystemExit(
            f"--select-methods asks for {missing_csv} but --lens {args.lens} writes no CSV for "
            f"{'it' if len(missing_csv) == 1 else 'them'}; use --lens both"
        )
    print(f"selective gathering [{', '.join(select_methods)}]: top {args.select_num_tokens} "
          f"token(s) x {args.select_num_layers} layer(s)"
          + (f" + always {list(always_layers)}" if always_layers else "")
          + (f", plus {args.select_random_tokens} control token(s)"
             if unscored and args.select_random_tokens else ""), flush=True)
    if unscored and args.select_random_tokens:
        return
    # Not fatal: --extend inherits the control the record already holds, which is the normal
    # way to add a lens arm to a tree that has already been pruned.
    print("  NOTE: no control arm is being drawn. "
          + ("--extend will inherit the one in the record."
             if args.extend
             else "WARNING: on a fresh trajectory this is unrecoverable -- once the tree holds "
                  "only the lens arm, a uniform draw over the chain cannot be made again. "
                  "Add 'random' to --select-methods."), flush=True)


@dataclass
class TrajectoryPlan:
    """What one trajectory needs from this invocation, or why it needs nothing."""

    new_methods: list[str]  # arms to compute (all of them, unless --extend)
    active: list[str]  # lenses whose CSV this run writes
    existing_record: dict | None  # the record to merge into, under --extend
    skip: str = ""  # non-empty means: report this and move on


def plan_trajectory(traj_dir: Path, csv_paths: dict[str, Path], select_methods: list[str],
                    lenses: list[str], extend: bool, overwrite: bool) -> TrajectoryPlan:
    """Decide what still has to be done for one trajectory.

    Under `--extend` this is what makes an already-pruned tree safe to add an arm to: the
    arms the record already holds are neither recomputed nor redrawn, and only the lenses
    whose arm is genuinely missing get a CSV pass.
    """
    if not extend:
        active = list(lenses)
        have = [lens for lens in active if csv_paths[lens].exists()]
        if have and not overwrite and len(have) == len(active):
            return TrajectoryPlan([], [], None, skip=f"CSV(s) for {have} exist (use --overwrite to redo)")
        return TrajectoryPlan(list(select_methods), active, None)

    path = record_path(traj_dir)
    if not path.exists():
        # Refusing here rather than falling back to a full selection: on a pruned tree a
        # full gather would be a very different, much larger operation than asked for.
        return TrajectoryPlan([], [], None, skip=f"--extend but no selection record at {path}")

    existing = read_raw_record(path)
    recorded, _ = read_selection_record(path)
    new_methods = [m for m in select_methods if m not in recorded.arms]
    if not new_methods:
        return TrajectoryPlan([], [], None, skip=f"record already holds {recorded.names}, nothing to add")
    active = [lens for lens in lenses if lens in new_methods or not csv_paths[lens].exists()]
    return TrajectoryPlan(new_methods, active, existing)


def build_pending(trajectory: dict, step_idxs: list[int], n_prefix: int, n_suffix: int) -> list[dict]:
    """Everything each step needs, resolved before any of them runs.

    Resolved up front so steps can be grouped by the length of the sequence they will need,
    and so pass 2 can re-forward from the same records without re-reading the trajectory.
    Steps with no reasoning tokens are dropped rather than carried as empty work.
    """
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
    return pending


def finalize_selection(*, args, stem, traj_dir, trajectory, pending, new_methods, tmp_paths,
                       always_layers, existing_record, model, output_base, sanitized_model,
                       writer_pool, prof) -> int | None:
    """Filter the freshly-written CSVs, gather what they select, and record the choice.

    Runs while the CSVs are still under their `.tmp` names: the selection is computed from
    them and the activations gathered *before* they are renamed, so an interrupted run
    leaves no CSV and is redone rather than skipped with a half-filled activation tree.

    Returns:
        The number of activation files written, or **None** when nothing was committed —
        a `--dry-run`, or a record that was refused because writing it would have destroyed
        an arm. The caller discards the `.tmp` CSVs in that case, leaving the trajectory
        exactly as it was.
    """
    kept = top_filter(
        args.signal_json,
        traj_dir,
        methods=new_methods,
        num_tokens=args.select_num_tokens,
        num_layers=args.select_num_layers,
        always_layers=always_layers,
        random_tokens=args.select_random_tokens,
        random_layers=args.select_random_layers,
        seed=args.select_seed,
        seed_key=stem,
        direction_classes=args.direction_classes,
        top_k=TOP_K,
        # The CSVs are still under their .tmp names, which is not where analysis_csv_path
        # looks. Layer pools are left to default: each arm takes the layers its own CSV has
        # rows for, which is exactly what delete_non_jlens_selected.py derives from the CSV
        # alone -- and what keeps a pruned tree and a filtered gather on the same files.
        csv_paths=tmp_paths,
    )
    kept = to_disk_coords(kept, trajectory)
    summary = ", ".join(f"{len(kept[m])} {m}" for m in kept.names)

    if args.dry_run:
        would, present = _extension_cost(kept, output_base)
        print(f"  DRY RUN: {summary} token(s); {present} file(s) already present, "
              f"{would} to write", flush=True)
        return None

    record = build_record(
        kept,
        stem=stem,
        model=sanitized_model,
        config={
            "signal_json": str(args.signal_json),
            "direction_classes": args.direction_classes,
            "num_tokens": args.select_num_tokens,
            "num_layers": args.select_num_layers,
            "always_layers": list(always_layers),
            "random_tokens": args.select_random_tokens,
            "random_layers": args.select_random_layers,
            "seed": args.select_seed,
            "top_k": TOP_K,
            "lens": args.lens,
        },
        output_starts={item["step_id"]: item["output_start"] for item in pending},
    )
    if existing_record is not None:
        # Arms already recorded keep their picks and their config verbatim. Redrawing a
        # control against a pruned chain would sample the survivors, not the chain.
        record = merge_records(existing_record, record)
    elif not args.overwrite_record:
        dropped = _arms_that_would_be_lost(record_path(traj_dir), record)
        if dropped:
            print(f"  REFUSED: writing this record would drop arm(s) {dropped} that cannot be "
                  f"recomputed; re-run with --extend (to merge) or --overwrite-record", flush=True)
            return None

    kept_by_step: dict[int, dict[int, tuple[int, ...]]] = defaultdict(dict)
    for (step_folder, token_idx), layers in kept.merged().items():
        kept_by_step[step_folder][token_idx] = layers
    total_count = save_selected_activations(
        model, pending, kept_by_step, output_base, writer_pool, prof,
        args.forward_batch_size, args.forward_batch_tokens,
        skip_existing=args.extend,
    )
    write_selection_record(record_path(traj_dir), record)
    print(f"  selected {summary} token(s) -> {total_count} activations", flush=True)
    return total_count


def _extension_cost(kept, output_base: Path) -> tuple[int, int]:
    """(files this selection would write, files it already has) under `output_base`.

    What --dry-run reports. The arms overlap heavily on genuinely direction-loaded tokens,
    so the second number is usually large and the first is what actually costs disk.
    """
    paths = kept.activation_paths(output_base)
    present = sum(1 for p in paths if p.exists())
    return len(paths) - present, present


def _arms_that_would_be_lost(path: Path, record: dict) -> list[str]:
    """Arms an existing record holds that `record` does not — i.e. what writing it destroys.

    An unscored control arm is a uniform draw over the *full* reasoning chain. Once the tree
    is pruned, that draw exists nowhere else, so silently replacing the record that holds it
    is unrecoverable data loss rather than an overwrite.
    """
    if not path.exists():
        return []
    try:
        existing = read_raw_record(path)
    except (ValueError, json.JSONDecodeError):
        return []
    return sorted(set(existing.get("arms", {})) - set(record.get("arms", {})))


def save_selected_activations(
    model,
    pending: list[dict],
    kept_by_step: dict[int, dict[int, tuple[int, ...]]],
    output_base: Path,
    writer_pool,
    prof: "Profiler",
    forward_batch_size: int,
    forward_batch_tokens: int,
    skip_existing: bool = False,
) -> int:
    """Second pass: re-forward only what the selection needs, write only what it keeps.

    The selection cannot be known until the whole trajectory's CSV exists, and buffering a
    long multi-step chain's residual streams to wait for it would cost hundreds of MB. A
    second forward is the cheaper trade: the GPU was never the bottleneck here (the
    per-token .pt writes were, per this module's docstring), and this pass drops ~99% of
    them — as well as truncating each sequence to its last *kept* token and hooking only
    the layers something actually wants.

    Args:
        pending: the per-step records built for pass 1 (`ids`, `abs_positions`, `step_id`,
            `output_start`).
        kept_by_step: {step folder: {output-relative token index: layers to keep}}.
        skip_existing: drop (token, layer) pairs whose `.pt` is already on disk. Used when
            extending a tree that already holds another arm — the arms overlap heavily on
            genuinely direction-loaded tokens, and rewriting an identical tensor is pure IO.

    Returns:
        Number of activation files queued for writing.
    """
    import torch
    from telos_interp.commands.gather_activations.gather_activations_utils import extract_activations_batched

    jobs = []
    for item in pending:
        wanted = kept_by_step.get(item["step_id"])
        if not wanted:
            continue
        start = item["output_start"]
        if skip_existing:
            wanted = {
                token_idx: tuple(
                    layer
                    for layer in layers
                    if not (output_base / f"layer_{layer}" / f"step_{item['step_id']}"
                            / "output" / f"{token_idx}.pt").exists()
                )
                for token_idx, layers in wanted.items()
            }
            wanted = {token_idx: layers for token_idx, layers in wanted.items() if layers}
            if not wanted:
                continue
        # back to absolute positions, which is what the forward pass indexes by
        by_abs = {token_idx + start: layers for token_idx, layers in wanted.items()}
        abs_positions = sorted(p for p in by_abs if p in set(item["abs_positions"]))
        if not abs_positions:
            continue
        jobs.append({
            "step_id": item["step_id"],
            "output_start": start,
            "abs_positions": abs_positions,
            "layers_by_abs": by_abs,
            "ids": item["ids"][: abs_positions[-1] + 1],
        })

    if not jobs:
        return 0

    saved = 0
    groups = group_consecutive(
        [len(job["ids"]) for job in jobs],
        max_items=max(1, forward_batch_size),
        max_tokens=max(1, forward_batch_tokens),
    )
    for group in groups:
        batch = [jobs[i] for i in group]
        # hook the union the batch needs: index_select keeps only the requested rows, so a
        # layer nothing wants is the only real waste, and dropping it is free
        layers = sorted({layer for job in batch for layers in job["layers_by_abs"].values() for layer in layers})

        with prof("build"):
            width = max(len(job["ids"]) for job in batch)
            input_ids = torch.zeros((len(batch), width), dtype=torch.long)
            attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
            padded = False
            for row, job in enumerate(batch):
                n = len(job["ids"])
                input_ids[row, :n] = torch.tensor(job["ids"], dtype=torch.long)
                attention_mask[row, :n] = 1
                padded |= n < width
            if not padded:
                attention_mask = None

        with prof("forward"):
            blocks = extract_activations_batched(
                model, input_ids, attention_mask,
                [job["abs_positions"] for job in batch], layers, keep_on_device=True,
            )

        offset = 0
        for job in batch:
            rows = slice(offset, offset + len(job["abs_positions"]))
            offset += len(job["abs_positions"])
            with prof("save_pt"):
                for layer in layers:
                    block = blocks[layer][rows].to("cpu")
                    tokens = {
                        abs_pos - job["output_start"]: block[j].clone()
                        for j, abs_pos in enumerate(job["abs_positions"])
                        if layer in job["layers_by_abs"][abs_pos]
                    }
                    if not tokens:
                        continue
                    writer_pool.submit(
                        {layer: tokens}, output_base, step_idx=job["step_id"], category="output"
                    )
                    saved += len(tokens)
        del blocks
    return saved


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
                    help="Write only the lens CSV(s), no .pt files. Diagnostic: the difference "
                         "against a normal run is exactly what the activation tree costs.")
    ap.add_argument("--lens", choices=["jlens", "logitlens", "both"], default="jlens",
                    help="Which lens(es) to apply. 'jlens' (default) transports each layer into "
                         "the layer-23 space through its fitted Jacobian before the unembed; "
                         "'logitlens' unembeds each layer where it sits, needs no fitted matrix, "
                         "and therefore covers every requested layer rather than only the lensed "
                         "ones. 'both' emits both CSVs from ONE forward pass -- the extra cost is "
                         "a second unembed per chunk, not a second pass. Only 'jlens'/'both' load "
                         "gpt-oss-20b_jacobian_lens.pt.")
    ap.add_argument("--signal-json", type=Path, default=None,
                    help="Enable selective gathering: JSON mapping UP/DOWN/LEFT/RIGHT to token "
                         "strings (e.g. data/jlens/direction_tokens_full.json). The full CSV(s) "
                         "are still written, but only the tokens/layers selected get a .pt, cutting "
                         "the activation tree ~75x. Without this the script saves everything, as "
                         "before.")
    ap.add_argument("--select-methods", default=",".join(DEFAULT_METHODS),
                    help="Comma-separated arms to select and record (default 'jlens,random'). "
                         f"Available: {','.join(METHODS)}. An unscored arm needs a lens alongside "
                         "it to enumerate the reasoning chain. Every arm's files are kept -- the "
                         "union is what survives on disk.")
    ap.add_argument("--extend", action="store_true",
                    help="Add arms to a trajectory that already has a selection record, instead of "
                         "replacing it. Arms already recorded are preserved VERBATIM (picks and "
                         "config both) and never redrawn; only arms the record lacks are computed, "
                         "and only their not-yet-present .pt files are written. This is the only "
                         "safe way to add a lens to a tree that has already been pruned: redrawing "
                         "a control arm against a pruned chain would sample the survivors, not the "
                         "chain.")
    ap.add_argument("--dry-run", action="store_true",
                    help="With --extend: report per trajectory what would be added (new tokens per "
                         "arm, files already present, files to write, estimated bytes) and write "
                         "nothing. Run this before committing disk to an extension.")
    ap.add_argument("--overwrite-record", action="store_true",
                    help="Permit replacing a selection record that holds arms the new one does not. "
                         "Refused by default: an unscored control arm cannot be redrawn once the "
                         "tree is pruned, so dropping it from the record destroys it permanently.")
    ap.add_argument("--select-num-tokens", type=int, default=20,
                    help="Tokens per trajectory kept by each scored (lens) arm (default 20).")
    ap.add_argument("--select-num-layers", type=int, default=3,
                    help="Top layers kept per selected token (default 3), before --select-always-layers.")
    ap.add_argument("--select-always-layers", default=",".join(str(i) for i in DEFAULT_ALWAYS_LAYERS),
                    help="Layers kept for every selected token of every arm regardless of score "
                         "(default 15, the project's standing comparison layer). Added on top of "
                         "--select-num-layers, not counted against it. Empty string to disable.")
    ap.add_argument("--select-random-tokens", type=int, default=20,
                    help="Size of an unscored (control) arm (default 20; 0 disables it). A lens "
                         "result means nothing without a uniform-draw control, and once the tree "
                         "is pruned that draw can no longer be made - so it is reserved now. Only "
                         "leave the control out when the record already holds one to inherit "
                         "(see --extend), or when you will never compare the two.")
    ap.add_argument("--select-random-layers", type=int, default=None,
                    help="Layers per control token (default: --select-num-layers).")
    ap.add_argument("--select-seed", type=int, default=42,
                    help="Seed for the control draw; combined with the trajectory stem so each "
                         "trajectory's sample is stable regardless of processing order.")
    ap.add_argument("--direction-classes", default="all",
                    help="Which lists in --signal-json to count: 'all' or e.g. 'UP,DOWN'.")
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

    lenses = parse_lenses(args.lens)
    print(f"loading unembed assets ({', '.join(lenses)})...", flush=True)
    assets = ensure_unembed_assets(args.jlens_dir)
    ids, tok = action_token_ids()
    id_cols = [ids[a] for a in ACTIONS]

    dev = torch.device(args.device)
    lm_head = assets["lm_head"].to(dev)
    norm_w = assets["norm_weight"].float().to(dev)
    eps = assets["rms_eps"]

    layers_by_lens, transport_by_lens = build_lens_transports(lenses, layer_indices, args.jlens_dir, dev)

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
    # Selective mode defers every write to a second pass, so pass 1 runs as if
    # --no-save-activations had been given.
    selective = args.signal_json is not None
    always_layers = parse_always_layers(args.select_always_layers)
    select_methods = parse_methods(args.select_methods)
    save_in_pass_one = args.save_activations and not selective
    validate_selection_args(args, selective, select_methods, lenses, always_layers)
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
        csv_paths = {lens: analysis_csv(args.activations_dir, stem, lens) for lens in lenses}

        plan = plan_trajectory(traj_dir, csv_paths, select_methods, lenses, args.extend, args.overwrite)
        if plan.skip:
            print(f"{stem}: {plan.skip}, skipping", flush=True)
            continue
        new_methods, active, existing_record = plan.new_methods, plan.active, plan.existing_record

        with open(traj_path) as f:
            trajectory = json.load(f)
        name = parse_name(stem)
        n_steps = len(trajectory["steps"])
        step_idxs = parse_index_specification(args.steps, n_steps, clamp=True)
        print(f"{stem}: {len(step_idxs)}/{n_steps} steps"
              + (f" (extending with {new_methods})" if args.extend else ""), flush=True)

        n_prefix = len(trajectory["prompt"]["prompt_prefix_tokens"])
        n_suffix = len(trajectory["prompt"]["prompt_suffix_tokens"])
        output_base = traj_dir / sanitized_model
        traj_dir.mkdir(parents=True, exist_ok=True)
        # write to temp files and rename at the end, so a crashed run leaves no CSV
        # and is redone on the next invocation instead of being skipped as done
        tmp_paths = {lens: csv_paths[lens].with_suffix(".csv.tmp") for lens in active}
        nan_count = 0
        total_count = 0

        with ExitStack() as stack:
            handles = {
                lens: stack.enter_context(open(tmp_paths[lens], "w", newline="", encoding="utf-8"))
                for lens in active
            }
            writers = {lens: csv.writer(fh) for lens, fh in handles.items()}
            for w in writers.values():
                w.writerow(header)

            with prof("build"):
                pending = build_pending(trajectory, step_idxs, n_prefix, n_suffix)

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
                    if save_in_pass_one:
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
                    # buffered per (lens, layer) so each CSV keeps its layer-major row order
                    # while each lens' transport runs as one bmm across its layers
                    rows_by_lens: dict[str, dict[int, list]] = {
                        lens: {layer: [] for layer in layers_by_lens[lens]} for lens in active
                    }
                    row_prefix = [name["size"], name["comp"], name["run"], item["si"]]
                    for i in range(0, n_pos, bs):
                        chunk = positions[i : i + bs]
                        with torch.no_grad():
                            # One forward pass feeds every lens: the residual streams in
                            # `blocks` are the same, only the transport before the unembed
                            # differs. This is what makes --lens both cost one extra unembed
                            # rather than a second pass.
                            for lens_name in active:
                                lens_layer_list = layers_by_lens[lens_name]
                                J_stack, J_rows = transport_by_lens[lens_name]
                                with prof("lens"):
                                    # [L, b, d] straight off the device, no host round-trip
                                    h = torch.stack(
                                        [blocks[layer][rows][i : i + bs] for layer in lens_layer_list]
                                    ).float()
                                    h = apply_lens_transport(h, J_stack, J_rows, norm_w, eps)
                                for li, layer in enumerate(lens_layer_list):
                                    with prof("lens"):
                                        ranks, logprobs, topk = lens_predictions(h[li], lm_head, id_cols)
                                    with prof("rows"):
                                        rows_by_lens[lens_name][layer].extend(
                                            row_prefix + [rp, ap, token, layer, item["agent_action"]]
                                            + r
                                            + [round(x, 4) for x in lp]
                                            + [decode_id(t) for t in tk]
                                            for (rp, ap, token), r, lp, tk in zip(
                                                chunk, ranks, logprobs, topk, strict=True
                                            )
                                        )
                                del h
                    with prof("csv"):
                        for lens_name in active:
                            for layer in layers_by_lens[lens_name]:
                                writers[lens_name].writerows(rows_by_lens[lens_name][layer])
                del blocks

        # The CSVs are complete but still under their .tmp names; the selection is computed
        # from them and the activations gathered before they are renamed, so an interrupted
        # run leaves no CSV and is redone rather than skipped with a half-filled tree.
        if selective and args.save_activations:
            total_count = finalize_selection(
                args=args, stem=stem, traj_dir=traj_dir, trajectory=trajectory, pending=pending,
                new_methods=new_methods, tmp_paths=tmp_paths, always_layers=always_layers,
                existing_record=existing_record, model=model, output_base=output_base,
                sanitized_model=sanitized_model, writer_pool=writer_pool, prof=prof,
            )
            if total_count is None:  # dry run, or a refused record: nothing was committed
                for tmp in tmp_paths.values():
                    tmp.unlink(missing_ok=True)
                continue

        with prof("drain_pt"):
            nan_count += writer_pool.drain()
        for lens_name in active:
            os.replace(tmp_paths[lens_name], csv_paths[lens_name])
        written += 1
        print(f"  wrote {', '.join(str(csv_paths[x]) for x in active)} "
              f"({total_count} activations)", flush=True)
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
