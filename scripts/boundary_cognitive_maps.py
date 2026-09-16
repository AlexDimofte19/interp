#!/usr/bin/env python3
"""The six channel-boundary tokens, from extraction to the paired pre/post comparison.

THE QUESTION. Every probe in this repo so far reads a REASONING token. This one reads the
two template boundaries that bracket the reasoning chain, three tokens each:

    pre_reasoning   `<|end|><|start|>assistant`  ending the user turn, before any reasoning
    post_reasoning  `<|end|><|start|>assistant`  ending the analysis channel, before `final`

Identical token strings, identical relative order, and the whole reasoning chain in between.
Whatever the grid-cell probe reads off the pre triple, the model had the grid but had not yet
reasoned about it; at the post triple it has. So the pre->post difference, on the same three
token identities and the same grids, is what the chain did to the decodable map.

WHY AN ADAPTER RATHER THAN THE EXISTING COMMANDS. `prepare_activations_for_probing`'s
token-major modes and `build_loudness_tables.py`'s selection both address the OUTPUT
category and score reasoning tokens; neither can name a `prompt_suffix` token or the
analysis->final triple inside `output`. Everything downstream of "which tokens" is the
repo's own machinery and is called, not copied: `parse_grid_state` + `_sample_triples` for
the cells, `extract_activations_batched` + `ActivationWriter` for the gather,
`train_cognitive_map_probe` (from the wrapper) for the probes, `apply_lens_transport` +
`lens_predictions` for the J-lens mass, `GridTileProbeType` for the row shape,
`loudness_analysis.stats` for every statistic and `plotting._style` for the figures.

COORDINATES, BOTH OF WHICH THIS FILE WRITES. `abs_pos` is prompt-inclusive -- position in
`prefix + grid + suffix + output`. `token_idx` is CATEGORY-RELATIVE and is what names the
`.pt` file, so the pre triple is `prompt_suffix/{0,1,2}.pt` and the post triple is
`output/{i,i+1,i+2}.pt`. Joining the two without converting yields an empty join, never an
error, so every row carries both.

THREE INDEPENDENT SAMPLES, NEVER A CONCATENATION. A triple contributes three `(D,)` rows,
one per token. Concatenating them would build a `3D` feature no probe in this repo reads and
would make the pre/post comparison a comparison of feature widths.

Stages (each idempotent; `wrappers/boundary_cognitive_maps.sh` drives them in order):

    extract   locate the six positions, gather layer-15 residuals into a .pt tree
    prepare   write the six token-major grid_tile manifests (3 partitions x 2 boundaries)
    loudness  J-lens grid log-mass at the same six positions, from those activations
    evaluate  every probe of one boundary over one partition -> per-token CSV
    compare   pair post against pre per architecture, aggregate, draw the paired figures
    manifest  identities, hashes, parameters and coverage

The per-table figures -- decile curves, per-class panels, ruler gaps -- are NOT drawn here.
`loudness_analysis/plotting/figures.py` already registers them against a counts-mode `grid`
table, which is exactly what `evaluate` writes, so the wrapper points that CLI at each
boundary table rather than this file growing a second set. What `compare` draws is only the
part that registry cannot: a contrast BETWEEN two tables.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# scripts/ sits beside telos_interp/; build_loudness_tables does the same insert so that
# `scripts.jlens_action_ranks` (which it imports lazily) resolves however it was entered.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------- constants

#: The three tokens of a boundary, in order. Matched as DECODED STRINGS against the
#: trajectory's own `token` field, never as fixed offsets into `output_tokens` -- the
#: analysis chain has no fixed length, so an offset would be right only by accident.
BOUNDARY_TOKENS = ("<|end|>", "<|start|>", "assistant")
#: Short names for the three, used in columns, filenames and group-bys.
BOUNDARY_ROLES = ("end", "start", "assistant")
#: The two boundaries, in the order pre -> post. `post - pre` is the experiment's contrast.
BOUNDARIES = ("pre_reasoning", "post_reasoning")
#: Where each boundary's tokens live in the activation tree's category axis.
BOUNDARY_CATEGORY = {"pre_reasoning": "prompt_suffix", "post_reasoning": "output"}

LAYER = 15
MODEL_FOLDER = "openai__gpt-oss-20b"
#: Cells drawn per (trajectory, step) for TRAINING. Evaluation scores every native cell.
TRAIN_MAX_CELLS = 25
PAD_TO_SIZE = 15
CELL_SEED = 42
LENS = "jlens"
SIGNAL = "grid"


class BoundaryError(ValueError):
    """A trajectory whose boundaries cannot be located unambiguously.

    Raised, never swallowed into a silent substitution: the caller reports the trajectory
    by name and leaves it out, because a boundary guessed from a fixed offset would produce
    rows that look exactly like the real ones.
    """


@dataclass(frozen=True)
class BoundaryToken:
    """One of the six positions, in both coordinate systems.

    `token_idx` is category-relative and names the `.pt` file; `abs_pos` is
    prompt-inclusive and is what a lens CSV would call the same token.
    """

    boundary: str
    role: str
    category: str
    token_idx: int
    abs_pos: int
    token: str
    token_id: int


def _token_texts(tokens: list[dict]) -> list[str]:
    return [t["token"] for t in tokens]


def _find_triple(texts: list[str]) -> list[int]:
    """Every start index at which `BOUNDARY_TOKENS` appears, in order.

    >>> _find_triple(["a", "<|end|>", "<|start|>", "assistant", "b"])
    [1]
    >>> _find_triple(["<|end|>", "<|start|>", "b"])
    []
    """
    n = len(BOUNDARY_TOKENS)
    return [i for i in range(len(texts) - n + 1) if tuple(texts[i : i + n]) == BOUNDARY_TOKENS]


def locate_boundaries(trajectory: dict, step_index: int) -> list[BoundaryToken]:
    """The six boundary tokens of one step, pre triple first.

    The pre triple is the prompt suffix itself -- the template tokens that close the user
    turn -- so it is checked to BE `<|end|><|start|>assistant` rather than searched for.
    The post triple is the analysis->final transition inside `output_tokens`, found by
    scanning for the same three strings; a step with none or several is ambiguous and
    raises rather than taking the first.

    Raises:
        BoundaryError: the suffix is not the expected triple, or the output has anything
            other than exactly one transition.
    """
    prompt = trajectory["prompt"]
    step = trajectory["steps"][step_index]
    prefix_tokens = prompt["prompt_prefix_tokens"]
    suffix_tokens = prompt["prompt_suffix_tokens"]
    grid_tokens = step["grid_state_tokens"]
    output_tokens = step["output_tokens"]

    suffix_texts = _token_texts(suffix_tokens)
    if tuple(suffix_texts) != BOUNDARY_TOKENS:
        raise BoundaryError(f"prompt suffix is {suffix_texts!r}, expected {list(BOUNDARY_TOKENS)!r}")

    hits = _find_triple(_token_texts(output_tokens))
    if len(hits) != 1:
        raise BoundaryError(
            f"step {step_index}: {len(hits)} analysis->final transitions in output_tokens "
            f"(expected exactly 1){'' if not hits else f' at {hits}'}"
        )
    post_start = hits[0]

    suffix_start = len(prefix_tokens) + len(grid_tokens)
    output_start = suffix_start + len(suffix_tokens)

    out: list[BoundaryToken] = []
    for i, role in enumerate(BOUNDARY_ROLES):
        tok = suffix_tokens[i]
        out.append(
            BoundaryToken(
                boundary="pre_reasoning",
                role=role,
                category="prompt_suffix",
                token_idx=i,
                abs_pos=suffix_start + i,
                token=tok["token"],
                token_id=int(tok["token_id"]),
            )
        )
    for i, role in enumerate(BOUNDARY_ROLES):
        idx = post_start + i
        tok = output_tokens[idx]
        out.append(
            BoundaryToken(
                boundary="post_reasoning",
                role=role,
                category="output",
                token_idx=idx,
                abs_pos=output_start + idx,
                token=tok["token"],
                token_id=int(tok["token_id"]),
            )
        )
    return out


def source_fingerprint(trajectory: dict, step_index: int) -> str:
    """A digest of the token ids this step's activations were (or would be) read from.

    `extract` writes it beside the tensors and `extract --resume` re-derives it. Matching
    FILENAMES are not enough to trust a cached `.pt`: the tree is keyed by trajectory name
    and category-relative index, both of which survive a trajectory being re-generated with
    a different chain. The ids do not.
    """
    prompt = trajectory["prompt"]
    step = trajectory["steps"][step_index]
    ids: list[int] = []
    for group in (
        prompt["prompt_prefix_tokens"],
        step["grid_state_tokens"],
        prompt["prompt_suffix_tokens"],
        step["output_tokens"],
    ):
        ids.extend(int(t["token_id"]) for t in group)
    digest = hashlib.sha256()
    digest.update(json.dumps(ids, separators=(",", ":")).encode())
    return digest.hexdigest()


def file_sha256(path: Path, limit: int | None = None) -> str:
    """sha256 of a file, or of its first `limit` bytes for a multi-hundred-MB one.

    The Jacobian lens is 380 MB and is hashed in full only when asked; the signal
    vocabulary is small and is always hashed whole, because its contents silently decide
    what every loudness number means.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        read = 0
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            if limit is not None and read + len(chunk) > limit:
                digest.update(chunk[: limit - read])
                break
            digest.update(chunk)
            read += len(chunk)
    return digest.hexdigest()


# -------------------------------------------------------------- partitions and lookup


@dataclass(frozen=True)
class Partition:
    """One of the three membership lists, and where its trajectory JSONs live.

    The lists are taken EXACTLY as they are on disk and never re-split: `mass_eval_720.txt`
    is byte-for-byte the eval set every probe already on disk was scored against, and the
    2,880 is its complement. A fresh draw here would quietly make this experiment
    incomparable with every other arm in the repo.
    """

    name: str
    names_file: Path
    trajectories_dir: Path

    def names(self) -> list[str]:
        return sorted(n for n in self.names_file.read_text().split() if n)


def default_partitions(workspace: Path) -> dict[str, Partition]:
    return {
        "train_2880": Partition(
            "train_2880",
            workspace / "splits/mass_train_2880.txt",
            workspace / "activations/mass_train2880_view/trajectories",
        ),
        "eval_720": Partition(
            "eval_720",
            workspace / "splits/mass_eval_720.txt",
            workspace / "activations/mass_eval720_view/trajectories",
        ),
        "heldout_360": Partition(
            "heldout_360",
            workspace / "trajectories/heldout360_names.txt",
            workspace / "trajectories/heldout360",
        ),
    }


def find_trajectory(trajectories_dir: Path, name: str) -> Path:
    """The JSON for `name` under a sizeN-nested or flat directory.

    The mass views are directories of SYMLINKS into the shared trajectory store, which is
    the point: pointing a stage at one view makes the other partition unreachable rather
    than merely unselected.
    """
    hits = sorted(trajectories_dir.glob(f"size*/{name}.json")) + sorted(trajectories_dir.glob(f"{name}.json"))
    if not hits:
        raise FileNotFoundError(f"no trajectory JSON for {name} under {trajectories_dir}")
    if len(hits) > 1:
        raise FileNotFoundError(f"{name} resolves to {len(hits)} files under {trajectories_dir}: {hits}")
    return hits[0]


def size_folder(trajectory: dict) -> str:
    """`size{N}` for the tree's first level, from the trajectory's own grid params."""
    return f"size{int(trajectory['grid_params']['grid_width'])}"


def trajectory_base(tree_root: Path, size_name: str, name: str) -> Path:
    """`{tree}/{sizeN}/{name}` -- the folder holding one trajectory's model tree and sidecar."""
    return tree_root / size_name / name


def activation_path(
    tree_root: Path, size_name: str, name: str, step: int, bt: BoundaryToken, layer: int = LAYER
) -> Path:
    """The `.pt` for one boundary token, in the gather's own layout.

    `{tree}/{sizeN}/{name}/{model}/layer_{L}/step_{M}/{category}/{token_idx}.pt` -- the same
    contract `gather_activations` and `build_loudness_tables.py` write, so anything that
    walks an activation tree reads this one unchanged.
    """
    return (
        trajectory_base(tree_root, size_name, name)
        / MODEL_FOLDER
        / f"layer_{layer}"
        / f"step_{step}"
        / bt.category
        / f"{bt.token_idx}.pt"
    )


def index_path(tree_root: Path, size_name: str, name: str) -> Path:
    """The per-trajectory sidecar recording which positions were gathered, and from what."""
    return trajectory_base(tree_root, size_name, name) / f"{name}_boundary_index.json"


def read_index(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


# ------------------------------------------------------------------------------- cells


def boundary_cell_payload(
    trajectory: dict,
    *,
    name: str,
    step_index: int,
    pad_to_size: int | None,
    max_cells: int | None,
    drop_padding: bool,
    seed: int,
) -> dict:
    """One step's `{positions, labels}` for the grid_tile label.

    Mirrors `prepare_activations_for_probing._grid_cell_payload` -- same parser, same
    sampler, same per-(trajectory, step) seeding formula -- with one deliberate difference:
    `drop_padding` removes the cells that lie OUTSIDE the native grid before the draw.
    Padded to 15 a size-5 grid is 25 real cells in 225, so a 25-cell uniform draw over the
    padded pool would spend ~22 of them on the padding class, which the headline metric
    (native-cell balanced accuracy) then discards. Every size has at least 25 native cells,
    so the draw still returns a constant 25 and the compact dataset's `(T, C, 2)` shape
    holds.

    The seed is a function of (trajectory, step) and nothing else, for the reason
    `_grid_cell_payload` records: two arms consume different numbers of draws, so a shared
    global stream would hand them different cells and the arms would stop differing only in
    their tokens.
    """
    from telos_interp.commands.prepare_activations_for_probing.prepare_activations_for_probing_fn import (
        _sample_triples,
    )
    from telos_interp.commands.prepare_activations_for_probing.prepare_activations_for_probing_utils import (
        parse_grid_state_from_trajectory,
    )
    from telos_interp.grid_utils import CELL_SYMBOL_TO_ID

    triples = parse_grid_state_from_trajectory(trajectory, step_idx=step_index, pad_to_size=pad_to_size)
    if drop_padding:
        pad_id = CELL_SYMBOL_TO_ID["+"]
        triples = [t for t in triples if t[2] != pad_id]
    if not triples:
        raise BoundaryError(f"{name} step {step_index}: no grid cells after parsing")
    selected = _sample_triples(
        triples,
        balance_classes=False,
        max_positions=max_cells,
        rng=random.Random(f"{name}:{step_index}:{seed}"),
    )
    return {
        "positions": [[int(t[0]), int(t[1])] for t in selected],
        "labels": [int(t[2]) for t in selected],
    }


# ------------------------------------------------------------------- stage: extract


def _plan_trajectory(traj_path: Path, tree_root: Path, layer: int) -> dict:
    """Everything `extract` needs for one trajectory: positions, ids, and what is cached.

    Returns a job dict even when nothing is missing -- the caller reports coverage from it.
    """
    trajectory = json.loads(traj_path.read_text())
    name = traj_path.stem
    size_name = size_folder(trajectory)
    prompt = trajectory["prompt"]
    prefix_ids = [int(t["token_id"]) for t in prompt["prompt_prefix_tokens"]]
    suffix_ids = [int(t["token_id"]) for t in prompt["prompt_suffix_tokens"]]

    cached = read_index(index_path(tree_root, size_name, name)) or {}
    cached_steps = {int(s["step"]): s for s in cached.get("steps", [])}

    steps: list[dict] = []
    for step_index, step in enumerate(trajectory["steps"]):
        step_id = int(step.get("step_id", step_index))
        tokens = locate_boundaries(trajectory, step_index)
        fingerprint = source_fingerprint(trajectory, step_index)

        # A cached step is trusted only when the ids it was read from still hash the same
        # AND every file it claims is on disk. Filenames alone survive a regenerated chain.
        previous = cached_steps.get(step_id)
        complete = (
            previous is not None
            and previous.get("source_fingerprint") == fingerprint
            and previous.get("layer") == layer
            and [t["abs_pos"] for t in previous.get("tokens", [])] == [t.abs_pos for t in tokens]
            and all(activation_path(tree_root, size_name, name, step_id, t, layer).exists() for t in tokens)
        )

        ids = prefix_ids + [int(t["token_id"]) for t in step["grid_state_tokens"]] + suffix_ids
        ids = ids + [int(t["token_id"]) for t in step["output_tokens"]]
        last = max(t.abs_pos for t in tokens)
        steps.append(
            {
                "step": step_id,
                "step_index": step_index,
                "tokens": tokens,
                "source_fingerprint": fingerprint,
                "ids": ids[: last + 1],
                "complete": complete,
            }
        )

    return {
        "name": name,
        "size_name": size_name,
        "path": traj_path,
        "grid_width": int(trajectory["grid_params"]["grid_width"]),
        "grid_complexity": trajectory["grid_params"].get("grid_complexity"),
        "steps": steps,
    }


def _write_index(tree_root: Path, job: dict, layer: int) -> None:
    """The sidecar: which positions this trajectory contributed, and from which token ids."""
    path = index_path(tree_root, job["size_name"], job["name"])
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": job["name"],
        "size": job["grid_width"],
        "complexity": job["grid_complexity"],
        "layer": layer,
        "model": MODEL_FOLDER,
        "source": str(job["path"]),
        "steps": [
            {
                "step": s["step"],
                "step_index": s["step_index"],
                "layer": layer,
                "source_fingerprint": s["source_fingerprint"],
                "tokens": [asdict(t) for t in s["tokens"]],
            }
            for s in job["steps"]
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def stage_extract(args) -> int:
    """Gather the layer-15 residual stream at the six boundary positions.

    One forward pass per (trajectory, step), truncated at the LAST boundary position: the
    post triple sits near the end of the sequence and the pre triple is a prefix of it, so
    both boundaries come out of the same pass. Sequences are packed into padded batches by
    the same `group_consecutive` budget the lens gather uses, and right padding is safe
    under causal attention (a real token at position i attends only to <= i).
    """
    import torch
    from telos_interp.commands.gather_activations.gather_activations_utils import (
        ActivationWriter,
        extract_activations_batched,
    )
    from telos_interp.loudness_analysis.build_loudness_tables import group_consecutive
    from transformers import AutoModelForCausalLM

    partitions = default_partitions(args.workspace)
    tree_root = args.tree
    tree_root.mkdir(parents=True, exist_ok=True)

    jobs: list[dict] = []
    failures: list[tuple[str, str]] = []
    wanted = args.partitions or list(partitions)
    for pname in wanted:
        part = partitions[pname]
        names = part.names()
        if args.limit:
            names = names[: args.limit]
        print(f"{pname}: {len(names)} trajectories from {part.names_file}", flush=True)
        for name in names:
            try:
                jobs.append(_plan_trajectory(find_trajectory(part.trajectories_dir, name), tree_root, args.layer))
            except (BoundaryError, FileNotFoundError, KeyError) as exc:
                failures.append((name, f"{type(exc).__name__}: {exc}"))

    pending = [
        {"job": job, "step": step} for job in jobs for step in job["steps"] if not step["complete"] or args.overwrite
    ]
    total_steps = sum(len(job["steps"]) for job in jobs)
    print(
        f"\n{len(jobs)} trajectories, {total_steps} steps, {total_steps * 6} boundary tokens; "
        f"{len(pending)} step(s) to extract, {total_steps - len(pending)} already cached",
        flush=True,
    )
    if failures:
        print(f"!! {len(failures)} trajectory/ies could not be located:", flush=True)
        for name, why in failures[: args.report_failures]:
            print(f"   {name}: {why}", flush=True)
    if args.dry_run:
        return 1 if failures else 0
    if not pending:
        for job in jobs:
            _write_index(tree_root, job, args.layer)
        return 1 if failures else 0

    print(f"loading model on {args.device} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map=args.device, dtype=torch.bfloat16)
    model.eval()

    # Longest first so a padded batch wastes as little as possible; groups are consecutive
    # by construction, and every job writes into its own folder, so order is free here.
    pending.sort(key=lambda p: len(p["step"]["ids"]), reverse=True)
    groups = group_consecutive(
        [len(p["step"]["ids"]) for p in pending],
        max_items=max(1, args.forward_batch_size),
        max_tokens=max(1, args.forward_batch_tokens),
    )

    writer = ActivationWriter(max_workers=args.io_workers)
    saved = nan_total = 0
    done_steps = 0

    def run_batch(batch: list[dict]) -> int:
        """One padded forward pass; queues every boundary tensor it produced.

        Returns the number of tensors queued. Raises whatever the forward raised -- the
        caller catches `torch.OutOfMemoryError` and splits the batch.
        """
        width = max(len(item["step"]["ids"]) for item in batch)
        input_ids = torch.zeros((len(batch), width), dtype=torch.long)
        attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
        padded = False
        for row, item in enumerate(batch):
            ids = item["step"]["ids"]
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row, : len(ids)] = 1
            padded |= len(ids) < width
        blocks = extract_activations_batched(
            model,
            input_ids,
            None if not padded else attention_mask,
            [[t.abs_pos for t in item["step"]["tokens"]] for item in batch],
            [args.layer],
            keep_on_device=True,
        )
        block = blocks[args.layer].to("cpu")
        queued = 0
        offset = 0
        for item in batch:
            job, step = item["job"], item["step"]
            tokens = step["tokens"]
            base = trajectory_base(tree_root, job["size_name"], job["name"]) / MODEL_FOLDER
            # One submit per category: the writer keys its folders by (layer, step,
            # category), and the two boundaries live in different categories.
            for category in (BOUNDARY_CATEGORY[b] for b in BOUNDARIES):
                payload = {
                    t.token_idx: block[offset + j].clone() for j, t in enumerate(tokens) if t.category == category
                }
                writer.submit({args.layer: payload}, base, step_idx=step["step"], category=category)
                queued += len(payload)
            offset += len(tokens)
        del blocks, block
        return queued

    try:
        for gi, group in enumerate(groups, 1):
            batch = [pending[i] for i in group]
            try:
                saved += run_batch(batch)
                done_steps += len(batch)
            except torch.OutOfMemoryError:
                # Attention here is EAGER -- gpt-oss carries sinks -- so one layer
                # materialises a [batch, heads, width, width] matrix and the cost is
                # quadratic in the PADDED width, not linear in the token count. A batch of
                # long chains can therefore exceed the card even inside the token budget.
                # Retry its members one at a time rather than losing the run: the budget is
                # a throughput knob, and one sequence is the smallest thing askable.
                torch.cuda.empty_cache()
                print(
                    f"  [{gi}/{len(groups)}] out of memory on {len(batch)} sequence(s) of width "
                    f"{max(len(i['step']['ids']) for i in batch)}; retrying one at a time",
                    flush=True,
                )
                for item in batch:
                    try:
                        saved += run_batch([item])
                        done_steps += 1
                    except torch.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        failures.append(
                            (
                                item["job"]["name"],
                                f"out of memory on a single {len(item['step']['ids'])}-token sequence",
                            )
                        )
            if gi % 25 == 0 or gi == len(groups):
                print(f"  [{gi}/{len(groups)} batches] {done_steps}/{len(pending)} steps, {saved} tensors", flush=True)
    finally:
        nan_total += writer.close()

    for job in jobs:
        _write_index(tree_root, job, args.layer)

    print(f"\nwrote {saved} activation file(s) under {tree_root}")
    if nan_total:
        print(f"!! {nan_total} activation(s) contained NaN -- a multi-GPU device_map does this to this MoE model")
    if failures:
        print(f"!! {len(failures)} trajectory/ies were skipped; see the list above")
    return 1 if (failures or nan_total) else 0


# ------------------------------------------------------------------- stage: prepare


def build_manifest(
    *,
    partition: Partition,
    boundary: str,
    tree_root: Path,
    layer: int,
    max_cells: int,
    pad_to_size: int,
    drop_padding: bool,
    cell_seed: int,
    limit: int | None,
    names: list[str] | None = None,
) -> tuple[dict, list[tuple[str, str]]]:
    """A token-major `grid_tile` v3 manifest for one (partition, boundary).

    One entry is one (token, layer) and copies nothing: `act_path` is relative to the
    absolute `activations_root`, exactly as the `next_action` and selected-`grid_tile`
    manifests do. Three entries per (trajectory, step) -- the boundary's three tokens --
    which is why the per-cell payload lives once per (trajectory, step) under `cells`,
    keyed by each entry's `cells_key`, rather than three times over.

    The trajectory's grid is a property of the STEP, shared by all three of its tokens, so a
    row-level train/eval split would leak. It never happens here: the partitions ARE the
    split, and `train_cognitive_map_probe` refuses an internal `--eval-split` on a
    token-major manifest anyway.
    """
    import torch

    entries: list[dict] = []
    cells: dict[str, dict] = {}
    per_size: dict[str, dict] = {}
    skipped: list[tuple[str, str]] = []
    activation_dim: int | None = None
    cells_len: int | None = None

    selected = names if names is not None else partition.names()
    if limit:
        selected = selected[:limit]

    for name in selected:
        try:
            traj_path = find_trajectory(partition.trajectories_dir, name)
            trajectory = json.loads(traj_path.read_text())
            size_name = size_folder(trajectory)
        except (FileNotFoundError, KeyError) as exc:
            skipped.append((name, f"{type(exc).__name__}: {exc}"))
            continue

        index = read_index(index_path(tree_root, size_name, name))
        if index is None:
            skipped.append((name, "no boundary index sidecar; run the extract stage"))
            continue
        by_step = {int(s["step"]): s for s in index.get("steps", [])}

        for step_index, step in enumerate(trajectory["steps"]):
            step_id = int(step.get("step_id", step_index))
            try:
                tokens = [t for t in locate_boundaries(trajectory, step_index) if t.boundary == boundary]
            except BoundaryError as exc:
                skipped.append((name, str(exc)))
                continue

            record = by_step.get(step_id)
            if record is None or record.get("source_fingerprint") != source_fingerprint(trajectory, step_index):
                skipped.append((name, f"step {step_id}: sidecar missing or stale against the trajectory"))
                continue

            paths = [activation_path(tree_root, size_name, name, step_id, t, layer) for t in tokens]
            missing = [p for p in paths if not p.exists()]
            if missing:
                skipped.append((name, f"step {step_id}: {len(missing)} boundary activation(s) missing"))
                continue

            try:
                payload = boundary_cell_payload(
                    trajectory,
                    name=name,
                    step_index=step_index,
                    pad_to_size=pad_to_size,
                    max_cells=max_cells,
                    drop_padding=drop_padding,
                    seed=cell_seed,
                )
            except BoundaryError as exc:
                skipped.append((name, str(exc)))
                continue

            if cells_len is None:
                cells_len = len(payload["labels"])
            if len(payload["labels"]) != cells_len:
                # (T, C, 2) is a rectangle: a step with a different cell count cannot go in.
                skipped.append((name, f"step {step_id}: {len(payload['labels'])} cells, manifest holds {cells_len}"))
                continue

            if activation_dim is None:
                activation_dim = int(torch.load(paths[0], map_location="cpu", weights_only=True).numel())

            cells_key = f"{name}|{step_id}"
            cells[cells_key] = payload
            size_int = int(trajectory["grid_params"]["grid_width"])
            for token, path in zip(tokens, paths, strict=True):
                entries.append(
                    {
                        "name": name,
                        "act_path": path.relative_to(tree_root).as_posix(),
                        "layer": layer,
                        "step": step_id,
                        "token_id": token.token_idx,
                        "category": token.category,
                        "size": size_int,
                        "cells_key": cells_key,
                        "boundary": token.boundary,
                        "role": token.role,
                        "token": token.token,
                        "abs_pos": token.abs_pos,
                    }
                )
            info = per_size.setdefault(size_name, {"num_trajectories": 0, "num_cells_per_trajectory": cells_len})
            info["num_trajectories"] += len(tokens)

    if not entries:
        raise SystemExit(f"{partition.name}/{boundary}: no usable entries; nothing to write")

    manifest = {
        "format_version": 3,
        "probe_type": "grid_tile",
        "activation_dim": activation_dim,
        "sizes": sorted(per_size, key=lambda s: int(s.replace("size", ""))),
        "per_size_info": per_size,
        "loading_spec": {
            "layers": str(layer),
            "steps": "all",
            "prompt_prefix_indices": None,
            "prompt_suffix_indices": "0,1,2" if boundary == "pre_reasoning" else None,
            "grid_state_indices": None,
            "output_indices": None if boundary == "pre_reasoning" else "analysis->final transition",
        },
        "config": {
            "probe_type": "grid_tile",
            "layers": str(layer),
            "steps": "all",
            "grid_step_idx": 0,
            "pad_to_size": pad_to_size,
            "max_positions_per_trajectory": max_cells,
            "balance_classes_per_trajectory": False,
            "drop_padding_cells": drop_padding,
            "seed": cell_seed,
            "prompt_prefix_indices": None,
            "prompt_suffix_indices": "0,1,2" if boundary == "pre_reasoning" else None,
            "grid_state_indices": None,
            "output_indices": None,
        },
        "activations_root": str(tree_root.resolve()),
        "selection": {
            "token_selection": f"boundary_{boundary}",
            "layer_selection": "spec",
            "method": "boundary",
            "boundary": boundary,
            "boundary_tokens": list(BOUNDARY_TOKENS),
            "category": BOUNDARY_CATEGORY[boundary],
            "seed": cell_seed,
        },
        "trajectories": entries,
        "num_cells_per_trajectory": cells_len,
        "cells": cells,
        "split": {
            "source_names": str(partition.names_file),
            "partition": partition.name,
            "num_samples": len(entries),
            "num_trajectories": len({e["name"] for e in entries}),
            "tokens_per_trajectory": len(BOUNDARY_ROLES),
            "layers_per_token": 1,
            "single_layer": layer,
            "thin_mode": "uniform",
        },
    }
    return manifest, skipped


def manifest_dir(prepared_root: Path, boundary: str, partition_name: str) -> Path:
    return prepared_root / f"{boundary}_{partition_name}"


def stage_prepare(args) -> int:
    partitions = default_partitions(args.workspace)
    wanted = args.partitions or list(partitions)
    boundaries = args.boundaries or list(BOUNDARIES)
    status = 0

    for pname in wanted:
        for boundary in boundaries:
            out = manifest_dir(args.prepared, boundary, pname)
            target = out / "manifest.json"
            if target.exists() and not args.overwrite:
                # Configuration-checked resumability: a manifest is reused only when the
                # knobs that decide its CONTENT are the ones this run was given. The cells
                # are a seeded draw, so a changed seed or cell budget would otherwise be
                # silently ignored and the arm would carry the previous run's cells.
                existing = json.loads(target.read_text())
                wanted = {
                    "pad_to_size": args.pad_to_size,
                    "max_positions_per_trajectory": args.max_cells,
                    "drop_padding_cells": not args.include_padding_cells,
                    "seed": args.cell_seed,
                    "layers": str(args.layer),
                }
                differs = {
                    k: (existing["config"].get(k), v) for k, v in wanted.items() if existing["config"].get(k) != v
                }
                if differs:
                    raise SystemExit(
                        f"{target} was built with {differs} (on disk, requested) -- "
                        "rebuild it with --overwrite or give this run its own --prepared."
                    )
                print(
                    f"- have {target} ({len(existing['trajectories'])} entries, "
                    f"{existing['split']['num_trajectories']} trajectories), reusing",
                    flush=True,
                )
                continue
            manifest, skipped = build_manifest(
                partition=partitions[pname],
                boundary=boundary,
                tree_root=args.tree,
                layer=args.layer,
                max_cells=args.max_cells,
                pad_to_size=args.pad_to_size,
                drop_padding=not args.include_padding_cells,
                cell_seed=args.cell_seed,
                limit=args.limit,
            )
            out.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(manifest) + "\n")
            print(
                f"- {pname}/{boundary}: {len(manifest['trajectories'])} entries over "
                f"{manifest['split']['num_trajectories']} trajectories, "
                f"{manifest['num_cells_per_trajectory']} cells each -> {target}",
                flush=True,
            )
            if skipped:
                status = 1
                print(f"  !! {len(skipped)} trajectory/step(s) left out:", flush=True)
                for name, why in skipped[: args.report_failures]:
                    print(f"     {name}: {why}", flush=True)
                (out / "skipped.json").write_text(json.dumps(skipped, indent=2) + "\n")
    return status


# ------------------------------------------------------------------ stage: loudness


def loudness_csv_path(prepared: Path, partition_name: str, lens: str = LENS, signal: str = SIGNAL) -> Path:
    return prepared / "loudness" / f"{partition_name}_{lens}_{signal}_boundary_mass.csv"


def loudness_key(name: str, step: int, boundary: str, token_idx: int) -> tuple[str, int, str, int]:
    """The join key every loudness row and every probe row carries.

    `token_idx` and not `abs_pos`: the two differ by the prompt length and joining on the
    wrong one produces an empty join in silence. `boundary` is in the key because the two
    boundaries live in different CATEGORIES, so their `token_idx` values are drawn from
    different origins and 0 means two different tokens.
    """
    return (name, int(step), boundary, int(token_idx))


def stage_loudness(args) -> int:
    """J-lens grid log-mass at the six boundary positions, from the gathered activations.

    Not a second forward pass: the residual streams are already on disk, and the lens is
    `transport -> final RMS norm -> unembed -> logsumexp over the vocabulary`, which is
    exactly `apply_lens_transport` followed by `lens_predictions`. Both are imported from
    the builder that produced every mass table in this repo, so a boundary token's loudness
    is measured with the same instrument as a reasoning token's.

    The vocabulary is `grid_tokens_full.json` as committed -- not regenerated, not
    substituted -- and its content hash goes in the sidecar beside the table. A mass table
    is not self-describing, and this repo points two vocabularies at the same trees.
    """
    import torch
    from scripts.jlens_action_ranks import action_token_ids, ensure_unembed_assets
    from telos_interp.jlens_utils import write_mass_meta
    from telos_interp.loudness_analysis import columns as cols
    from telos_interp.loudness_analysis import provenance, signals
    from telos_interp.loudness_analysis.build_loudness_tables import (
        ACTIONS,
        apply_lens_transport,
        build_lens_transports,
        lens_predictions,
        resolve_direction_ids,
    )

    partitions = default_partitions(args.workspace)
    wanted = args.partitions or list(partitions)
    dev = torch.device(args.device)

    signal = signals.resolve(args.signal_name, args.signal_json)
    vocabulary = signal.load(args.signal_json, args.signal_classes)
    ids, tok = action_token_ids()
    id_cols = [ids[a] for a in ACTIONS]
    resolved, dropped = resolve_direction_ids(
        vocabulary, lambda t: tok.encode(t, add_special_tokens=False), lambda i: tok.decode([i])
    )
    if not resolved:
        raise SystemExit(f"{args.signal_json} resolved to no usable token ids")
    print(f"{signal.name} vocabulary: {len(resolved)} token id(s), {len(dropped)} dropped", flush=True)
    signal_ids = torch.tensor(resolved, device=dev)

    assets = ensure_unembed_assets(args.jlens_dir)
    lm_head = assets["lm_head"].to(dev)
    norm_w = assets["norm_weight"].float().to(dev)
    eps = assets["rms_eps"]
    layers_by_lens, transport_by_lens = build_lens_transports([args.lens], [args.layer], args.jlens_dir, dev)
    if args.layer not in layers_by_lens[args.lens]:
        raise SystemExit(f"{args.lens} cannot score layer {args.layer}: no fitted matrix")
    J_stack, J_rows = transport_by_lens[args.lens]

    mass_col = cols.loudness_column(args.lens, signal.name, args.layer)
    prob_col = cols.prob_column(args.lens, signal.name, args.layer)
    header = [
        "name",
        "size",
        "complexity",
        "step",
        "boundary",
        "role",
        "category",
        "token_idx",
        "abs_pos",
        "token",
        mass_col,
        prob_col,
    ]

    status = 0
    for pname in wanted:
        part = partitions[pname]
        out = loudness_csv_path(args.prepared, pname, args.lens, signal.name)
        if out.exists() and not args.overwrite:
            print(f"- have {out}, reusing", flush=True)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)

        names = part.names()
        if args.limit:
            names = names[: args.limit]

        rows: list[list] = []
        pending: list[tuple[list, Path]] = []
        skipped: list[tuple[str, str]] = []

        def flush(pending: list[tuple[list, Path]], rows: list[list] = rows) -> None:
            """Unembed one batch of boundary activations and append their rows.

            `rows` is bound as a default argument rather than captured: the enclosing loop
            rebinds it once per partition, and a late-binding closure would append every
            partition's rows to the last one's list.
            """
            if not pending:
                return
            acts = torch.stack([torch.load(p, map_location="cpu", weights_only=True).float() for _, p in pending]).to(
                dev
            )
            h = acts.unsqueeze(0)  # [L=1, b, d], the layer-major stack the transport expects
            h = apply_lens_transport(h, J_stack, J_rows, norm_w, eps)
            _, _, _, _, mass = lens_predictions(h[0], lm_head, id_cols, signal_ids)
            for (row, _), value in zip(pending, mass, strict=True):
                rows.append(row + [f"{value:.6f}", f"{math.exp(value):.8g}"])
            pending.clear()

        for name in names:
            try:
                traj_path = find_trajectory(part.trajectories_dir, name)
                trajectory = json.loads(traj_path.read_text())
                size_name = size_folder(trajectory)
            except (FileNotFoundError, KeyError) as exc:
                skipped.append((name, f"{type(exc).__name__}: {exc}"))
                continue
            for step_index, step in enumerate(trajectory["steps"]):
                step_id = int(step.get("step_id", step_index))
                try:
                    tokens = locate_boundaries(trajectory, step_index)
                except BoundaryError as exc:
                    skipped.append((name, str(exc)))
                    continue
                for token in tokens:
                    path = activation_path(args.tree, size_name, name, step_id, token, args.layer)
                    if not path.exists():
                        skipped.append((name, f"step {step_id} {token.boundary}/{token.role}: no activation"))
                        continue
                    pending.append(
                        (
                            [
                                name,
                                int(trajectory["grid_params"]["grid_width"]),
                                trajectory["grid_params"].get("grid_complexity", ""),
                                step_id,
                                token.boundary,
                                token.role,
                                token.category,
                                token.token_idx,
                                token.abs_pos,
                                token.token,
                            ],
                            path,
                        )
                    )
                    if len(pending) >= args.batch_size:
                        flush(pending)
        flush(pending)

        with open(out, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            writer.writerows(rows)

        write_mass_meta(
            out,
            {
                "signal_json": str(args.signal_json),
                "signal_name": signal.name,
                "signal_fingerprint": signal.fingerprint(args.signal_json),
                "signal_sha256": file_sha256(Path(args.signal_json)),
                "signal_classes": args.signal_classes,
                "num_signal_tokens": len(resolved),
                "num_dropped": len(dropped),
                "dropped": dropped,
                "lens": args.lens,
                "lens_file": str(args.jlens_dir / "gpt-oss-20b_jacobian_lens.pt"),
                "lens_sha256": file_sha256(args.jlens_dir / "gpt-oss-20b_jacobian_lens.pt"),
                "unembed_file": str(args.jlens_dir / "gpt-oss-20b_unembed.pt"),
                "layer": args.layer,
                "model": args.model_id,
                "boundary_tokens": list(BOUNDARY_TOKENS),
                "rows": len(rows),
            },
        )
        print(f"- {pname}: {len(rows)} boundary row(s) -> {out}", flush=True)
        if skipped:
            status = 1
            print(f"  !! {len(skipped)} position(s) skipped:", flush=True)
            for name, why in skipped[: args.report_failures]:
                print(f"     {name}: {why}", flush=True)

        cfg = provenance.RunConfig("scripts/boundary_cognitive_maps.py loudness")
        cfg.measurement(lens=args.lens, signal=signal.name, layer=args.layer, signal_json=args.signal_json)
        cfg.input("activations", args.tree)
        cfg.input("trajectories", part.trajectories_dir)
        cfg.input("names", part.names_file)
        cfg.params.update({"partition": pname, "boundaries": list(BOUNDARIES), "signal_classes": args.signal_classes})
        cfg.rows("boundary_rows", len(rows))
        cfg.rows("skipped", len(skipped))
        cfg.outputs.append(out)
        cfg.write(out.parent)
    return status


def read_loudness(path: Path, lens: str, signal: str, layer: int) -> dict[tuple, float]:
    """`{(name, step, boundary, token_idx): logmass}` from a boundary mass table.

    `csv.DictReader`, never `pandas.read_csv`: decoded tokens include the literal string
    "NA", empty strings, embedded commas and newlines, all of which pandas corrupts.
    """
    from telos_interp.loudness_analysis import columns as cols

    out: dict[tuple, float] = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        column = cols.resolve(reader.fieldnames or [], lens, signal, layer)
        for row in reader:
            value = row[column]
            if value in (None, ""):
                continue
            out[loudness_key(row["name"], int(row["step"]), row["boundary"], int(row["token_idx"]))] = float(value)
    return out


# ------------------------------------------------------------------ stage: evaluate


class _CellArgs:
    """The three attributes `GridTileProbeType.step_state` reads off its args object.

    Evaluation deliberately does NOT reuse training's cell draw. Training samples 25 cells
    per step because ~17k token samples x 225 cells x 50 epochs is not affordable;
    evaluation scores every cell of the NATIVE grid (`pad_to_size=None`), because the
    headline metric is native-cell balanced accuracy and there is no reason to estimate it
    from a sample when the whole grid costs one extra forward.
    """

    def __init__(self, pad_to_size: int | None, max_cells: int | None, seed: int) -> None:
        self.pad_to_size = pad_to_size
        self.max_cells = max_cells
        self.seed = seed


def _boot_counts_bal_acc(df, probe: str, classes, n_boot: int, seed: int) -> tuple[float, float, float]:
    """Counts-pooled balanced accuracy with a trajectory-clustered 95% CI.

    `stats.boot_bal_acc` is the ROWS form -- one prediction per row -- and a grid row is a
    whole step's cells, so it cannot be used here (the two balanced accuracies are not the
    same number; see `loudness_analysis.stats`). The resampling rule is the same one: draw
    trajectory NAMES with replacement, never rows, because the cells of one step share a
    grid and a row-level bootstrap reports a band several times too narrow.
    """
    import numpy as np
    from telos_interp.loudness_analysis import stats

    point, _ = stats.bal_acc_from_counts(df, probe, classes)
    names = df["name"].to_numpy()
    uniq = np.unique(names)
    if len(uniq) < 2 or n_boot == 0:
        return point, float("nan"), float("nan")
    idx_of = {u: np.flatnonzero(names == u) for u in uniq}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        rows = np.concatenate([idx_of[uniq[k]] for k in pick])
        value, _ = stats.bal_acc_from_counts(df.iloc[rows], probe, classes)
        draws.append(value)
    return point, float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))


def summarise(df, probe: str, classes, n_boot: int, seed: int) -> dict:
    """Balanced accuracy (counts-pooled), per-class recalls and the three breakdowns."""
    from telos_interp.loudness_analysis import stats

    point, lo, hi = _boot_counts_bal_acc(df, probe, classes, n_boot, seed)
    _, recalls = stats.bal_acc_from_counts(df, probe, classes)
    out = {
        "n_rows": int(len(df)),
        "n_trajectories": int(df["name"].nunique()),
        "n_cells": int(df["n_cells"].sum()),
        "balanced_accuracy": point,
        "balanced_accuracy_ci95": [lo, hi],
        "plain_accuracy": stats.plain_accuracy(df, probe),
        "per_class_recall": {str(k): v for k, v in recalls.items()},
        "by": {},
    }
    for key in ("role", "size", "complexity"):
        if key not in df.columns:
            continue
        block = {}
        for value, group in df.groupby(key, observed=True):
            ba, rec = stats.bal_acc_from_counts(group, probe, classes)
            block[str(value)] = {
                "n_rows": int(len(group)),
                "balanced_accuracy": ba,
                "plain_accuracy": stats.plain_accuracy(group, probe),
                "per_class_recall": {str(k): v for k, v in rec.items()},
            }
        out["by"][key] = block
    return out


def stage_evaluate(args) -> int:
    """Every probe of ONE boundary, over one partition -> a per-token CSV and its summary.

    `--probe` is repeatable, as it is on `score_probes_per_token.py`, and for the same
    reason: the probes of one boundary are scored on IDENTICAL rows -- same tokens, same
    per-step cell draw, so the same `n_true_{class}` denominators -- and a counts-mode grid
    table is built to carry several probes against one set of denominators. Scoring them
    into separate tables would make `bal_acc_from_counts` pool one probe's hits against
    another's cells the moment the two were concatenated, and would cost a second pass over
    the activations for nothing.

    A probe is still scored ONLY on the boundary it was trained on. The two boundaries are
    different token positions with different context in front of them, and a pre-trained
    probe read at the post boundary would answer a third question that this experiment is
    not asking; the pre/post contrast is between each boundary's OWN probes.
    """
    import pandas as pd
    import torch
    from telos_interp.loudness_analysis import provenance, signals
    from telos_interp.loudness_analysis.probes import get_probe_type

    partitions = default_partitions(args.workspace)
    part = partitions[args.partition]
    ptype = get_probe_type("grid_tile")
    signal = signals.resolve(args.signal_name, args.signal_json)
    probes = {ptype.probe_key(path): ptype.load_probe(path) for path in args.probe}
    if len(probes) != len(args.probe):
        raise SystemExit(f"two probes share a key: {[str(p) for p in args.probe]}")
    print(f"{len(probes)} probe(s): {', '.join(probes)}", flush=True)
    cell_args = _CellArgs(args.eval_pad_to_size, args.eval_max_cells, args.cell_seed)

    loud_path = loudness_csv_path(args.prepared, args.partition, args.lens, signal.name)
    loudness = read_loudness(loud_path, args.lens, signal.name, args.layer) if loud_path.exists() else {}
    if not loudness:
        print(f"!! no loudness table at {loud_path}; the loudness columns will be empty", flush=True)

    from telos_interp.loudness_analysis import columns as cols

    mass_col = cols.loudness_column(args.lens, signal.name, args.layer)
    extra_cols = ["boundary", "role", "category", mass_col]
    header = ptype.prefix_columns() + extra_cols + ptype.result_columns(list(probes), args.full_probs)

    names = part.names()
    if args.limit:
        names = names[: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "per_token.csv"
    written = 0
    skipped: list[tuple[str, str]] = []
    missing_loudness = 0

    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        rows: list[list] = []
        acts: list = []
        states: list = []

        def flush() -> int:
            if not rows:
                return 0
            results = ptype.score(probes, acts, states, args.batch_size, args.device, args.full_probs)
            for row, result in zip(rows, results, strict=True):
                writer.writerow(row + result)
            n = len(rows)
            rows.clear()
            acts.clear()
            states.clear()
            return n

        for name in names:
            try:
                traj_path = find_trajectory(part.trajectories_dir, name)
                trajectory = json.loads(traj_path.read_text())
                size_name = size_folder(trajectory)
            except (FileNotFoundError, KeyError) as exc:
                skipped.append((name, f"{type(exc).__name__}: {exc}"))
                continue

            for step_index, step in enumerate(trajectory["steps"]):
                step_id = int(step.get("step_id", step_index))
                try:
                    tokens = [t for t in locate_boundaries(trajectory, step_index) if t.boundary == args.boundary]
                except BoundaryError as exc:
                    skipped.append((name, str(exc)))
                    continue
                state = ptype.step_state(trajectory, step_index, name, cell_args)
                if state is None:
                    skipped.append((name, f"step {step_id}: no grid cells"))
                    continue
                for token in tokens:
                    path = activation_path(args.tree, size_name, name, step_id, token, args.layer)
                    if not path.exists():
                        skipped.append((name, f"step {step_id} {token.role}: no activation"))
                        continue
                    key = loudness_key(name, step_id, token.boundary, token.token_idx)
                    value = loudness.get(key)
                    if value is None:
                        missing_loudness += 1
                    # `step_id`, not `step_index`: the row's step column has to agree with
                    # the loudness table's and with the `.pt` tree's `step_M` folder, all of
                    # which are named by `step_id`. `step_state` above takes the INDEX,
                    # because that is what indexes `traj["steps"]`.
                    prefix = ptype.row_prefix(
                        name, trajectory, step_id, token.abs_pos, token.token_idx, token.token, state
                    )
                    rows.append(
                        prefix + [token.boundary, token.role, token.category, "" if value is None else f"{value:.6f}"]
                    )
                    acts.append(torch.load(path, map_location="cpu", weights_only=True).float())
                    states.append(state)
            if len(rows) >= args.flush_tokens:
                written += flush()
        written += flush()

    df = pd.read_csv(csv_path, keep_default_na=False, na_values=[""])
    summary = {
        "partition": args.partition,
        "boundary": args.boundary,
        "layer": args.layer,
        "loudness_column": mass_col,
        "classes_scored": list(ptype.analysis_classes),
        "padding_class_excluded": True,
        "missing_loudness_rows": missing_loudness,
        "skipped": len(skipped),
        "probes": {str(path): ptype.probe_key(path) for path in args.probe},
        "by_probe": {key: summarise(df, key, ptype.analysis_classes, args.n_boot, args.seed) for key in probes},
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    cfg = provenance.RunConfig("scripts/boundary_cognitive_maps.py evaluate")
    cfg.measurement(lens=args.lens, signal=signal.name, layer=args.layer, signal_json=args.signal_json)
    for path in args.probe:
        cfg.input(f"probe:{ptype.probe_key(path)}", path)
    cfg.input("activations", args.tree)
    cfg.input("names", part.names_file)
    cfg.input("loudness", loud_path)
    cfg.params.update(
        {
            "probe_type": "grid_tile",
            "probes": sorted(probes),
            "partition": args.partition,
            "boundary": args.boundary,
            "eval_pad_to_size": args.eval_pad_to_size,
            "eval_max_cells": args.eval_max_cells,
            "cell_seed": args.cell_seed,
        }
    )
    cfg.aggregation(balanced_accuracy="counts", bootstrap="trajectory-clustered", n_boot=args.n_boot, seed=args.seed)
    cfg.rows("token_rows", written)
    cfg.rows("missing_loudness", missing_loudness)
    cfg.rows("skipped", len(skipped))
    cfg.outputs.extend([csv_path, args.out / "summary.json"])
    cfg.write(args.out)

    print(f"{written} row(s) -> {csv_path}", flush=True)
    for key, block in summary["by_probe"].items():
        print(
            f"  {key}: native-cell balanced accuracy (padding excluded) "
            f"{block['balanced_accuracy']:.4f} "
            f"[{block['balanced_accuracy_ci95'][0]:.4f}, {block['balanced_accuracy_ci95'][1]:.4f}]",
            flush=True,
        )
    if skipped:
        print(f"  !! {len(skipped)} step/token(s) skipped:", flush=True)
        for name, why in skipped[: args.report_failures]:
            print(f"     {name}: {why}", flush=True)
    if missing_loudness:
        print(f"  !! {missing_loudness} row(s) had no loudness value", flush=True)
    return 1 if (skipped or missing_loudness) else 0


# ------------------------------------------------------------------- stage: compare


def probe_keys_of(df) -> list[str]:
    """Every probe key a counts-mode table carries, read back off its `_n_correct` columns.

    Same derivation as `plotting.figures.grid_probes`, so this module and the figures agree
    about what a table holds. A boundary table carries one key per architecture.
    """
    return sorted({c[: -len("_n_correct")] for c in df.columns if c.endswith("_n_correct")})


def architecture_of(probe_key: str) -> str:
    """The architecture a probe key names -- its last `_`-separated field.

    Keys are `"<parent dir>.<stem minus the trainer's prefix>"`, and the trainers end a stem
    with the model type, so `...l15_mlp` is the mlp. Pairing pre with post has to match
    architectures: the two boundaries' keys differ everywhere except this field.

    >>> architecture_of("pre_reasoning.boundary_pre_reasoning_l15_mlp")
    'mlp'
    >>> architecture_of("post_reasoning.boundary_post_reasoning_l15_lr")
    'lr'
    """
    return probe_key.rsplit("_", 1)[-1]


def _load_side(path: Path) -> tuple:
    """A boundary's per-token table, plus `{architecture: probe key}` for what it carries."""
    import pandas as pd

    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    keys = probe_keys_of(df)
    if not keys:
        raise SystemExit(f"{path} carries no probe columns")
    by_arch: dict[str, str] = {}
    for key in keys:
        arch = architecture_of(key)
        if arch in by_arch:
            raise SystemExit(f"{path} carries two {arch} probes: {by_arch[arch]} and {key}")
        by_arch[arch] = key
    return df, by_arch


def paired_frame(pre, post, pre_probe: str, post_probe: str, mass_col: str):
    """One row per (trajectory, step, token identity), carrying both sides.

    Pairing is on `role`, not on position: the two boundaries are the SAME three token
    strings, so `<|end|>` pre pairs with `<|end|>` post. Rows missing on either side are
    dropped rather than filled, and the count is reported.
    """
    keys = ["name", "step", "role"]
    carry = ["size", "complexity"]
    left = pre[keys + carry + ["accuracy", mass_col, "n_cells"]].rename(
        columns={"accuracy": "pre_accuracy", mass_col: "pre_loudness", "n_cells": "pre_n_cells"}
    )
    right = post[keys + ["accuracy", mass_col, "n_cells"]].rename(
        columns={"accuracy": "post_accuracy", mass_col: "post_loudness", "n_cells": "post_n_cells"}
    )
    out = left.merge(right, on=keys, how="inner")
    out["delta_accuracy"] = out["post_accuracy"] - out["pre_accuracy"]
    out["delta_loudness"] = out["post_loudness"] - out["pre_loudness"]
    return out


def _paired_bal_acc_delta(pre, post, pre_probe: str, post_probe: str, classes, n_boot: int, seed: int) -> dict:
    """post-minus-pre counts-pooled balanced accuracy, with a clustered CI on the DIFFERENCE.

    The bootstrap resamples trajectory names ONCE per draw and recomputes both sides from
    the same draw, so the interval is on the paired difference rather than on two
    independent numbers whose intervals happen to overlap.
    """
    import numpy as np
    from telos_interp.loudness_analysis import stats

    shared = sorted(set(pre["name"]) & set(post["name"]))
    pre_s = pre[pre["name"].isin(shared)]
    post_s = post[post["name"].isin(shared)]
    pre_ba, pre_rec = stats.bal_acc_from_counts(pre_s, pre_probe, classes)
    post_ba, post_rec = stats.bal_acc_from_counts(post_s, post_probe, classes)

    lo = hi = float("nan")
    if len(shared) >= 2 and n_boot:
        rng = np.random.default_rng(seed)
        pre_idx = {n: np.flatnonzero(pre_s["name"].to_numpy() == n) for n in shared}
        post_idx = {n: np.flatnonzero(post_s["name"].to_numpy() == n) for n in shared}
        draws = []
        for _ in range(n_boot):
            pick = [shared[i] for i in rng.integers(0, len(shared), len(shared))]
            a, _ = stats.bal_acc_from_counts(
                pre_s.iloc[np.concatenate([pre_idx[n] for n in pick])], pre_probe, classes
            )
            b, _ = stats.bal_acc_from_counts(
                post_s.iloc[np.concatenate([post_idx[n] for n in pick])], post_probe, classes
            )
            draws.append(b - a)
        lo, hi = (float(v) for v in np.nanpercentile(draws, [2.5, 97.5]))

    return {
        "n_trajectories": len(shared),
        "pre_balanced_accuracy": pre_ba,
        "post_balanced_accuracy": post_ba,
        "delta_balanced_accuracy": post_ba - pre_ba,
        "delta_ci95": [lo, hi],
        "pre_per_class_recall": {str(k): v for k, v in pre_rec.items()},
        "post_per_class_recall": {str(k): v for k, v in post_rec.items()},
    }


def _by_loudness_decile(df, probe: str, classes, mass_col: str, n_bins: int) -> list[dict]:
    """Counts-pooled balanced accuracy per loudness decile, within one boundary.

    `stats.qbin` and not `pd.qcut`: loudness is heavy-tailed with a repeated floor, so
    duplicate bin edges are the normal case rather than an error.
    """
    from telos_interp.loudness_analysis import stats

    usable = df[df[mass_col].notna()]
    if usable.empty:
        return []
    bins = stats.qbin(usable[mass_col], n_bins, labels=False)
    out = []
    for b, group in usable.groupby(bins, observed=True):
        ba, rec = stats.bal_acc_from_counts(group, probe, classes)
        out.append(
            {
                "bin": int(b),
                "n_rows": int(len(group)),
                "n_trajectories": int(group["name"].nunique()),
                "loudness_mean": float(group[mass_col].mean()),
                "balanced_accuracy": ba,
                "plain_accuracy": stats.plain_accuracy(group, probe),
                "per_class_recall": {str(k): v for k, v in rec.items()},
            }
        )
    return out


def _spearman_or_none(df, x: str, y: str, min_n: int = 3) -> dict:
    """Rank correlation, or an explicit `null` with the reason it is undefined.

    A constant column has no ranks to correlate; reporting 0 there would read as "no
    relationship" when the truth is "not asked".
    """
    from telos_interp.loudness_analysis import stats

    sub = df[[x, y]].dropna()
    if len(sub) < min_n:
        return {"n": int(len(sub)), "spearman": None, "undefined": "insufficient samples"}
    if sub[x].nunique() < 2 or sub[y].nunique() < 2:
        return {"n": int(len(sub)), "spearman": None, "undefined": "constant column"}
    value = stats.spearman(sub, x, y)
    if math.isnan(value):
        return {"n": int(len(sub)), "spearman": None, "undefined": "nan"}
    return {"n": int(len(sub)), "spearman": value, "undefined": None}


def stage_compare(args) -> int:
    """Pair the two boundaries per ARCHITECTURE, aggregate, and draw the paired figures.

    One boundary table carries every architecture, so the loop is over architectures inside
    one dataset rather than over separate directories. Pairing matches `lr` with `lr` and
    `mlp` with `mlp`: the two boundaries' probe keys differ everywhere except that field, and
    crossing them would compare a linear probe's pre with an MLP's post.

    This stage draws only what the shared plotting registry cannot: the registry's three
    `grid` figures answer "where in the loudness range is this table decodable", which is a
    question about ONE table, while the pre/post contrast is a question about two. The
    decile curves, per-class panels and ruler gaps come from `plotting/figures.py` instead
    -- see the `plots` stage -- and are not redrawn here.

    Everything here is an ASSOCIATION. The pre and post triples differ by the whole
    reasoning chain, so a difference between them is not attributable to any one thing the
    chain did, and a correlation between loudness and accuracy is not evidence that either
    causes the other. The direction of the expected drop is not assumed: the interval is
    two-sided and the sign is read off the result.
    """

    from telos_interp.loudness_analysis import columns as cols
    from telos_interp.loudness_analysis import provenance, signals
    from telos_interp.loudness_analysis.probes import get_probe_type

    ptype = get_probe_type("grid_tile")
    classes = ptype.analysis_classes
    signal = signals.resolve(args.signal_name, args.signal_json)
    mass_col = cols.loudness_column(args.lens, signal.name, args.layer)

    pre_csv = args.dataset_dir / "pre_reasoning" / "per_token.csv"
    post_csv = args.dataset_dir / "post_reasoning" / "per_token.csv"
    for path in (pre_csv, post_csv):
        if not path.exists():
            raise SystemExit(f"missing {path}; run the evaluate stage for both boundaries first")

    pre_all, pre_by_arch = _load_side(pre_csv)
    post_all, post_by_arch = _load_side(post_csv)
    shared_archs = sorted(set(pre_by_arch) & set(post_by_arch))
    if not shared_archs:
        raise SystemExit(f"no architecture is present on both sides: {sorted(pre_by_arch)} vs {sorted(post_by_arch)}")
    only_one_side = sorted(set(pre_by_arch) ^ set(post_by_arch))
    if only_one_side:
        print(f"!! {only_one_side} appear on one boundary only and are not paired", flush=True)

    for arch in shared_archs:
        pre_probe, post_probe = pre_by_arch[arch], post_by_arch[arch]
        pre = pre_all.copy()
        post = post_all.copy()
        pre["accuracy"] = pre[f"{pre_probe}_acc"].astype(float)
        post["accuracy"] = post[f"{post_probe}_acc"].astype(float)

        out = args.dataset_dir / "comparison" / arch
        out.mkdir(parents=True, exist_ok=True)

        paired = paired_frame(pre, post, pre_probe, post_probe, mass_col)
        paired.to_csv(out / "paired.csv", index=False)

        aggregates = {
            "architecture": arch,
            "probe_keys": {"pre_reasoning": pre_probe, "post_reasoning": post_probe},
            "loudness_column": mass_col,
            "classes_scored": list(classes),
            "padding_class_excluded": True,
            "aggregation": "counts-pooled balanced accuracy; bootstrap resamples trajectories",
            "paired_rows": int(len(paired)),
            "unpaired_pre_rows": int(len(pre) - len(paired)),
            "unpaired_post_rows": int(len(post) - len(paired)),
            "overall": _paired_bal_acc_delta(pre, post, pre_probe, post_probe, classes, args.n_boot, args.seed),
            "by_role": {},
            "by_loudness_decile": {},
            "correlations": {"within_group": {}, "paired_delta": {}},
            "interpretation": (
                "Associations only. The two boundaries are separated by the whole reasoning "
                "chain, so a pre/post difference names no mechanism."
            ),
        }

        for role in BOUNDARY_ROLES:
            aggregates["by_role"][role] = _paired_bal_acc_delta(
                pre[pre["role"] == role],
                post[post["role"] == role],
                pre_probe,
                post_probe,
                classes,
                args.n_boot,
                args.seed,
            )

        for label, df, probe in (("pre_reasoning", pre, pre_probe), ("post_reasoning", post, post_probe)):
            aggregates["by_loudness_decile"][label] = _by_loudness_decile(df, probe, classes, mass_col, args.n_bins)
            for role in BOUNDARY_ROLES:
                group = df[df["role"] == role]
                aggregates["correlations"]["within_group"][f"{label}/{role}"] = _spearman_or_none(
                    group, mass_col, "accuracy"
                )
            aggregates["correlations"]["within_group"][f"{label}/all"] = _spearman_or_none(df, mass_col, "accuracy")

        aggregates["correlations"]["paired_delta"]["all"] = _spearman_or_none(
            paired, "delta_loudness", "delta_accuracy"
        )
        for role in BOUNDARY_ROLES:
            aggregates["correlations"]["paired_delta"][role] = _spearman_or_none(
                paired[paired["role"] == role], "delta_loudness", "delta_accuracy"
            )

        (out / "aggregates.json").write_text(json.dumps(aggregates, indent=2, default=float) + "\n")

        figures = draw_aggregate_figures(
            pre=pre,
            post=post,
            paired=paired,
            aggregates=aggregates,
            mass_col=mass_col,
            signal=signal.name,
            lens=args.lens,
            layer=args.layer,
            out=out,
        )

        cfg = provenance.RunConfig("scripts/boundary_cognitive_maps.py compare")
        cfg.measurement(lens=args.lens, signal=signal.name, layer=args.layer, signal_json=args.signal_json)
        cfg.input("pre_per_token", pre_csv)
        cfg.input("post_per_token", post_csv)
        cfg.params.update({"architecture": arch, "pre_probe": pre_probe, "post_probe": post_probe})
        cfg.aggregation(
            balanced_accuracy="counts",
            bootstrap="trajectory-clustered",
            n_boot=args.n_boot,
            seed=args.seed,
            loudness_bins=args.n_bins,
            pairing="(name, step, role), architectures matched",
        )
        cfg.rows("paired_rows", len(paired))
        cfg.outputs.extend([out / "paired.csv", out / "aggregates.json", *figures])
        cfg.write(out)

        overall = aggregates["overall"]
        print(
            f"[{arch}] pre {overall['pre_balanced_accuracy']:.4f}  post {overall['post_balanced_accuracy']:.4f}  "
            f"delta {overall['delta_balanced_accuracy']:+.4f} "
            f"[{overall['delta_ci95'][0]:+.4f}, {overall['delta_ci95'][1]:+.4f}]  "
            f"over {overall['n_trajectories']} trajectories",
            flush=True,
        )
    return 0


# --------------------------------------------------------------------------- figures

#: One colour per boundary and one marker per token identity, fixed here so every paired
#: figure reads the same way. Borrowed from `plotting._style`'s two-series palette rather
#: than invented, so these sit beside the registry's figures without restyling them.
BOUNDARY_COLOR = {"pre_reasoning": "#2a78d6", "post_reasoning": "#eb6834"}
BOUNDARY_LABEL = {"pre_reasoning": "pre-reasoning", "post_reasoning": "post-reasoning"}
ROLE_MARKER = {"end": "o", "start": "s", "assistant": "^"}


def draw_aggregate_figures(*, pre, post, paired, aggregates, mass_col, signal, lens, layer, out: Path) -> list[Path]:
    """The three aggregate figures, drawn with `plotting._style`'s scaffolding.

    `_style` is imported rather than re-implemented so these sit beside the published
    loudness figures without restyling them, and `axis_label` comes from `columns` so a
    figure cannot name its ruler differently from the table it was drawn from.
    """
    import numpy as np
    from matplotlib import pyplot as plt
    from telos_interp.loudness_analysis.plotting._style import INK_MUTED, axis_label, finish, style_axis

    paths: list[Path] = []
    xlabel = axis_label(lens, signal, layer)

    # 1. balanced accuracy against loudness decile, one line per boundary
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for boundary in BOUNDARIES:
        rows = aggregates["by_loudness_decile"].get(boundary) or []
        if not rows:
            continue
        ax.plot(
            [r["loudness_mean"] for r in rows],
            [r["balanced_accuracy"] for r in rows],
            "-o",
            ms=4,
            lw=1.6,
            color=BOUNDARY_COLOR[boundary],
            label=BOUNDARY_LABEL[boundary],
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("native-cell balanced accuracy\n(padding excluded, counts pooled)")
    ax.set_title("grid-cell decodability against boundary loudness")
    ax.legend(frameon=False)
    style_axis(ax)
    path = out / "balanced_accuracy_by_loudness_decile.png"
    finish(fig, path)
    paths.append(path)

    # 2. the paired difference, overall and per token identity
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    labels = ["all"] + list(BOUNDARY_ROLES)
    blocks = [aggregates["overall"]] + [aggregates["by_role"][r] for r in BOUNDARY_ROLES]
    y = np.arange(len(labels))
    deltas = [b["delta_balanced_accuracy"] for b in blocks]
    lo = [b["delta_balanced_accuracy"] - b["delta_ci95"][0] for b in blocks]
    hi = [b["delta_ci95"][1] - b["delta_balanced_accuracy"] for b in blocks]
    axes[0].errorbar(deltas, y, xerr=[lo, hi], fmt="o", color=BOUNDARY_COLOR["post_reasoning"], capsize=3, lw=1.4)
    axes[0].axvline(0, color=INK_MUTED, ls=":", lw=1.2)
    axes[0].set_yticks(y, labels)
    axes[0].set_xlabel("post - pre balanced accuracy (95% CI, trajectories resampled)")
    axes[0].set_title("paired difference by token identity")
    style_axis(axes[0])

    for role in BOUNDARY_ROLES:
        values = paired[paired["role"] == role]["delta_accuracy"].dropna()
        if values.empty:
            continue
        axes[1].hist(values, bins=40, histtype="step", lw=1.5, label=f"`{role}`")
    axes[1].axvline(0, color=INK_MUTED, ls=":", lw=1.2)
    axes[1].set_xlabel("post - pre per-token native accuracy")
    axes[1].set_ylabel("boundary pairs")
    axes[1].set_title("per-pair difference")
    axes[1].legend(frameon=False)
    style_axis(axes[1])
    path = out / "paired_delta.png"
    finish(fig, path)
    paths.append(path)

    # 3. loudness against accuracy, and the paired deltas against each other
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for boundary, df in (("pre_reasoning", pre), ("post_reasoning", post)):
        sub = df[[mass_col, "accuracy", "role"]].dropna()
        for role in BOUNDARY_ROLES:
            block = sub[sub["role"] == role]
            if block.empty:
                continue
            axes[0].scatter(
                block[mass_col],
                block["accuracy"],
                s=5,
                alpha=0.15,
                lw=0,
                color=BOUNDARY_COLOR[boundary],
                marker=ROLE_MARKER[role],
                label=f"{BOUNDARY_LABEL[boundary]} `{role}`",
            )
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("per-token native accuracy")
    axes[0].set_title("one point per boundary token")
    axes[0].legend(frameon=False, fontsize=7, markerscale=2)
    style_axis(axes[0])

    sub = paired[["delta_loudness", "delta_accuracy", "role"]].dropna()
    for role in BOUNDARY_ROLES:
        block = sub[sub["role"] == role]
        if block.empty:
            continue
        axes[1].scatter(
            block["delta_loudness"],
            block["delta_accuracy"],
            s=5,
            alpha=0.15,
            lw=0,
            marker=ROLE_MARKER[role],
            label=f"`{role}`",
        )
    axes[1].axhline(0, color=INK_MUTED, ls=":", lw=1.0)
    axes[1].axvline(0, color=INK_MUTED, ls=":", lw=1.0)
    axes[1].set_xlabel(f"post - pre {xlabel}")
    axes[1].set_ylabel("post - pre native accuracy")
    axes[1].set_title("paired change in loudness against paired change in accuracy")
    axes[1].legend(frameon=False, fontsize=8, markerscale=2)
    style_axis(axes[1])
    path = out / "loudness_vs_accuracy.png"
    finish(fig, path)
    paths.append(path)

    plt.close("all")
    return paths


# ------------------------------------------------------------------ stage: manifest


def stage_manifest(args) -> int:
    """The run manifest: what was used, what it hashed to, and how much of it landed.

    Written last because it reports COVERAGE as well as configuration -- how many of each
    partition's trajectories actually produced six boundary tokens, and what every
    evaluation folder found. It also re-checks the three membership lists against each
    other and against the pinned eval names every probe already on disk was scored on: the
    lists are used exactly as they are, so the manifest is where that claim is verified
    rather than assumed.
    """
    from telos_interp.loudness_analysis import provenance, signals

    partitions = default_partitions(args.workspace)
    signal = signals.resolve(args.signal_name, args.signal_json)

    names_by_partition = {k: set(v.names()) for k, v in partitions.items()}
    pinned = args.workspace / "prepared/next_action_mass_l15_eval_names.txt"
    splits = {}
    for key, part in partitions.items():
        splits[key] = {
            "names_file": str(part.names_file),
            "sha256": file_sha256(part.names_file),
            "n": len(names_by_partition[key]),
            "trajectories_dir": str(part.trajectories_dir),
        }
    overlaps = {
        f"{a}&{b}": sorted(names_by_partition[a] & names_by_partition[b])
        for a, b in (("train_2880", "eval_720"), ("train_2880", "heldout_360"), ("eval_720", "heldout_360"))
    }

    coverage = {}
    for pname in partitions:
        for boundary in BOUNDARIES:
            path = manifest_dir(args.prepared, boundary, pname) / "manifest.json"
            if not path.exists():
                continue
            manifest = json.loads(path.read_text())
            coverage[f"{pname}/{boundary}"] = {
                "manifest": str(path),
                "entries": len(manifest["trajectories"]),
                "trajectories": manifest["split"]["num_trajectories"],
                "expected_trajectories": len(names_by_partition[pname]),
                "cells_per_sample": manifest["num_cells_per_trajectory"],
                "skipped": len(json.loads((path.parent / "skipped.json").read_text()))
                if (path.parent / "skipped.json").exists()
                else 0,
            }

    checkpoints = {}
    for arch in sorted(p.name for p in args.experiment_root.glob("general_l15_*") if p.is_dir()):
        for boundary in BOUNDARIES:
            folder = args.experiment_root / arch / boundary
            hits = sorted(folder.glob("*.pt"))
            checkpoints[f"{arch}/{boundary}"] = {
                "folder": str(folder),
                "checkpoints": [{"path": str(h), "sha256": file_sha256(h), "bytes": h.stat().st_size} for h in hits],
            }

    # `{dataset}/{boundary}/summary.json` and `{dataset}/comparison/{arch}/aggregates.json`.
    # A glob that matches nothing is an empty `results` block that still LOOKS like a
    # finished manifest, so a run that found no results at all is an error rather than a
    # quiet omission -- this exact glob went stale once when the layout became dataset-major.
    results = {}
    for summary in sorted(args.evaluation_root.glob("*/*/summary.json")):
        results[str(summary.relative_to(args.evaluation_root))] = json.loads(summary.read_text())
    for aggregate in sorted(args.evaluation_root.glob("*/comparison/*/aggregates.json")):
        results[str(aggregate.relative_to(args.evaluation_root))] = json.loads(aggregate.read_text())["overall"]
    if not results and any(args.evaluation_root.glob("*/*/per_token.csv")):
        raise SystemExit(
            f"{args.evaluation_root} holds per-token tables but no summary.json or "
            "aggregates.json matched: the manifest's globs and the output layout disagree."
        )

    lens_file = args.jlens_dir / "gpt-oss-20b_jacobian_lens.pt"
    payload = {
        "experiment": "boundary grid-probe (pre/post reasoning channel boundaries)",
        "written_at": provenance.RunConfig("manifest").to_dict()["written_at"],
        "git_commit": provenance.git_commit(),
        "commands": json.loads(Path(args.commands).read_text()) if args.commands else None,
        "splits": splits,
        "split_overlaps": {k: {"n": len(v), "names": v[:20]} for k, v in overlaps.items()},
        "eval_720_matches_pinned_next_action_names": (
            names_by_partition["eval_720"] == set(pinned.read_text().split()) if pinned.exists() else None
        ),
        "pinned_names_file": str(pinned),
        "model": {"model_id": args.model_id, "layer": args.layer, "category_axis": BOUNDARY_CATEGORY},
        "lens": {
            "lens": args.lens,
            "file": str(lens_file),
            "sha256": file_sha256(lens_file) if lens_file.exists() else None,
            "unembed": str(args.jlens_dir / "gpt-oss-20b_unembed.pt"),
        },
        "vocabulary": {
            "signal": signal.name,
            "path": str(args.signal_json),
            "sha256": file_sha256(Path(args.signal_json)),
            "fingerprint": signal.fingerprint(args.signal_json),
            "classes": list(signal.classes(args.signal_json)),
        },
        "parameters": {
            "boundary_tokens": list(BOUNDARY_TOKENS),
            "roles": list(BOUNDARY_ROLES),
            "train_max_cells": args.max_cells,
            "pad_to_size": args.pad_to_size,
            "cell_seed": args.cell_seed,
            "include_padding_cells": args.include_padding_cells,
            "eval_cells": "every native cell (pad_to_size=None)",
            "training": json.loads(Path(args.training_params).read_text()) if args.training_params else None,
        },
        "activation_tree": str(args.tree),
        "coverage": coverage,
        "checkpoints": checkpoints,
        "results": results,
    }

    args.evaluation_root.mkdir(parents=True, exist_ok=True)
    out = args.evaluation_root / "provenance"
    out.mkdir(parents=True, exist_ok=True)
    target = out / "run_manifest.json"
    target.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"wrote {target}", flush=True)

    bad = [k for k, v in overlaps.items() if v]
    if bad:
        print(f"!! membership lists overlap: {bad}", flush=True)
    for key, block in coverage.items():
        if block["trajectories"] != block["expected_trajectories"]:
            print(
                f"!! {key}: {block['trajectories']} trajectories in the manifest, "
                f"{block['expected_trajectories']} in the list",
                flush=True,
            )
    return 1 if bad else 0


# ------------------------------------------------------------------------------- CLI


DEFAULT_WORKSPACE = Path("/workspace")
DEFAULT_EXPERIMENT = DEFAULT_WORKSPACE / "probes/2880_trajectory_trained_cogn_maps"
DEFAULT_JLENS_DIR = Path("/workspace/jlens/gridenv")
DEFAULT_SIGNAL_JSON = Path(__file__).resolve().parents[1] / "data/jlens/grid_tokens_full.json"
MODEL_ID = "openai/gpt-oss-20b"


def _add_common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--workspace", type=Path, default=DEFAULT_WORKSPACE, help="Root holding splits/ and trajectories/."
    )
    ap.add_argument(
        "--experiment-root", type=Path, default=DEFAULT_EXPERIMENT, help="Where checkpoints and prepared/ live."
    )
    ap.add_argument("--prepared", type=Path, default=None, help="Default: {experiment-root}/prepared.")
    ap.add_argument(
        "--tree", type=Path, default=None, help="Boundary activation tree. Default: {prepared}/activations."
    )
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--limit", type=int, default=None, help="Process at most N trajectories per partition.")
    ap.add_argument("--report-failures", type=int, default=20, help="How many skipped names to print.")
    ap.add_argument("--overwrite", action="store_true", help="Redo work that is already on disk.")


def _add_lens(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--lens", default=LENS, choices=["jlens", "logitlens"])
    ap.add_argument("--jlens-dir", "--jlens_dir", dest="jlens_dir", type=Path, default=DEFAULT_JLENS_DIR)
    ap.add_argument(
        "--signal-json",
        type=Path,
        default=DEFAULT_SIGNAL_JSON,
        help="Signal vocabulary. Defaults to the committed grid vocabulary; not regenerated here.",
    )
    ap.add_argument("--signal-name", default=None, help="Inferred from the filename when omitted.")
    ap.add_argument("--signal-classes", "--direction-classes", dest="signal_classes", default="all")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)

    extract = sub.add_parser("extract", help="Gather the six boundary activations.")
    _add_common(extract)
    extract.add_argument("--partitions", nargs="*", default=None, help="Default: all three.")
    extract.add_argument("--device", default="cuda:0", help="Single device. device_map='auto' NaNs this MoE model.")
    extract.add_argument("--model-id", default=MODEL_ID)
    # gpt-oss carries attention sinks and so runs EAGER attention: one layer materialises a
    # [batch, heads, width, width] matrix, which makes memory quadratic in the PADDED width
    # rather than linear in the token count. build_loudness_tables.py's 16384 is sized for
    # an 80 GB card; on 32 GB, four ~1900-token chains in one batch ask for 6.6 GiB in a
    # single allocation and die. 4096 caps the widest pairing at two long chains, and a
    # batch that still runs out is retried one sequence at a time rather than lost.
    extract.add_argument("--forward-batch-size", type=int, default=4)
    extract.add_argument("--forward-batch-tokens", type=int, default=4096)
    extract.add_argument("--io-workers", type=int, default=8)
    extract.add_argument("--dry-run", action="store_true", help="Report coverage and exit without loading the model.")
    extract.set_defaults(fn=stage_extract)

    prepare = sub.add_parser("prepare", help="Write the token-major grid_tile manifests.")
    _add_common(prepare)
    prepare.add_argument("--partitions", nargs="*", default=None)
    prepare.add_argument("--boundaries", nargs="*", default=None, choices=list(BOUNDARIES))
    prepare.add_argument("--max-cells", type=int, default=TRAIN_MAX_CELLS)
    prepare.add_argument("--pad-to-size", type=int, default=PAD_TO_SIZE)
    prepare.add_argument("--cell-seed", type=int, default=CELL_SEED)
    prepare.add_argument(
        "--include-padding-cells",
        action="store_true",
        help="Draw training cells from the padded 15x15 frame instead of the native grid. "
        "Off by default: padded, a size-5 grid is 25 real cells in 225, so a 25-cell draw "
        "would spend ~22 on the class the headline metric discards.",
    )
    prepare.set_defaults(fn=stage_prepare)

    loudness = sub.add_parser("loudness", help="Lens signal log-mass at the six positions.")
    _add_common(loudness)
    _add_lens(loudness)
    loudness.add_argument("--partitions", nargs="*", default=None)
    loudness.add_argument("--device", default="cuda:0")
    loudness.add_argument("--model-id", default=MODEL_ID)
    loudness.add_argument("--batch-size", type=int, default=256, help="Boundary tokens per unembed.")
    loudness.set_defaults(fn=stage_loudness)

    evaluate = sub.add_parser("evaluate", help="Score one probe on one partition x boundary.")
    _add_common(evaluate)
    _add_lens(evaluate)
    evaluate.add_argument(
        "--probe",
        type=Path,
        action="append",
        required=True,
        help="Trained probe .pt (repeatable). Every probe of ONE boundary belongs in one "
        "table: they are scored on identical rows, which is what a counts-mode grid table "
        "and the shared grid figures expect.",
    )
    evaluate.add_argument("--partition", required=True, choices=["train_2880", "eval_720", "heldout_360"])
    evaluate.add_argument("--boundary", required=True, choices=list(BOUNDARIES))
    evaluate.add_argument("--out", type=Path, required=True)
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument("--batch-size", type=int, default=8192, help="Cell rows per probe forward.")
    evaluate.add_argument("--flush-tokens", type=int, default=512)
    evaluate.add_argument(
        "--eval-pad-to-size",
        type=int,
        default=None,
        help="Default: none, i.e. the NATIVE grid, which excludes the padding class structurally.",
    )
    evaluate.add_argument("--eval-max-cells", type=int, default=None, help="Default: every cell of the grid.")
    evaluate.add_argument("--cell-seed", type=int, default=CELL_SEED)
    evaluate.add_argument("--full-probs", action="store_true")
    evaluate.add_argument("--n-boot", type=int, default=500)
    evaluate.add_argument("--seed", type=int, default=CELL_SEED)
    evaluate.set_defaults(fn=stage_evaluate)

    compare = sub.add_parser("compare", help="Pair post against pre and aggregate.")
    _add_common(compare)
    _add_lens(compare)
    compare.add_argument(
        "--dataset-dir",
        "--arch-dir",
        dest="dataset_dir",
        type=Path,
        required=True,
        help="One dataset's directory, holding pre_reasoning/ and post_reasoning/.",
    )
    compare.add_argument("--n-boot", type=int, default=500)
    compare.add_argument("--n-bins", type=int, default=10)
    compare.add_argument("--seed", type=int, default=CELL_SEED)
    compare.set_defaults(fn=stage_compare)

    manifest = sub.add_parser("manifest", help="Write the run manifest: identities, parameters, coverage.")
    _add_common(manifest)
    _add_lens(manifest)
    manifest.add_argument("--evaluation-root", type=Path, required=True)
    manifest.add_argument("--model-id", default=MODEL_ID)
    manifest.add_argument("--max-cells", type=int, default=TRAIN_MAX_CELLS)
    manifest.add_argument("--pad-to-size", type=int, default=PAD_TO_SIZE)
    manifest.add_argument("--cell-seed", type=int, default=CELL_SEED)
    manifest.add_argument("--include-padding-cells", action="store_true")
    manifest.add_argument("--commands", type=Path, default=None, help="JSON list of the commands the wrapper ran.")
    manifest.add_argument("--training-params", type=Path, default=None, help="JSON of the trainer's hyperparameters.")
    manifest.set_defaults(fn=stage_manifest)

    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.prepared is None:
        args.prepared = args.experiment_root / "prepared"
    if args.tree is None:
        args.tree = args.prepared / "activations"
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
