"""Score a saved binary cognitive map probe against a prepared dataset, one row per cell.

`eval_cognitive_map_probe` cannot do this job. It walks the gathered tree and
*concatenates* the token indices it is given into one activation per (trajectory, step), so
a probe trained token-major -- `input_dim = D + 2`, one reasoning token per row -- can only
ever be handed a single token index. Training reads every selected token on its own, and an
evaluation that does anything else is not measuring the thing that was trained.

So this reads the same artifact the trainer reads: a v3 `grid_tile` manifest directory. The
720-trajectory eval half and the held-out 360 are then scored under identical conditions,
differing only in which trajectories and which tokens the manifest names.

Rows are grouped by grid size (from the manifest entry) and by complexity (parsed out of
the trajectory name, `..._comp0.4_123`), because a wall probe that only works on small
grids and one that works everywhere have the same global number.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import torch

from telos_interp.commands.prepare_activations_for_probing.manifest_loader import (
    detect_format,
    load_v3_manifest,
    resolve_manifest_path,
)
from telos_interp.commands.train_binary_cognitive_map_probe import (
    BinaryCognitiveMapProbe,
    binary_metrics,
)
from telos_interp.commands.train_cognitive_map_probe.train_cognitive_map_probe_fn import (
    _load_grid_tile_compact_cached,
)
from telos_interp.training import resolve_device

COMPLEXITY_RE = re.compile(r"_comp([0-9]*\.?[0-9]+)")


def _drop_nan_entries(compact: dict, verbose: bool) -> dict:
    """Drop entries whose activation holds a NaN, the same filter the trainer applies.

    Args:
        compact: The dict from `load_grid_tile_compact`.
        verbose: Print how many were dropped.

    Returns:
        The same dict, filtered in place.
    """
    nan_mask = torch.isnan(compact["base_act"]).any(dim=1)
    if not bool(nan_mask.any()):
        return compact

    keep = ~nan_mask
    compact["base_act"] = compact["base_act"][keep]
    compact["positions"] = compact["positions"][keep]
    compact["labels"] = compact["labels"][keep]
    kept = keep.tolist()
    compact["trajectory_names"] = [n for n, k in zip(compact["trajectory_names"], kept, strict=False) if k]
    if compact.get("sizes") is not None:
        compact["sizes"] = [s for s, k in zip(compact["sizes"], kept, strict=False) if k]
    if verbose:
        print(f"WARNING: dropped {int(nan_mask.sum().item())} entries with NaN activations")
    return compact


def _complexity_of(trajectory_name: str) -> float | None:
    """Grid complexity encoded in a trajectory name, or None if it carries none.

    Args:
        trajectory_name: e.g. "together_ai_openai_gpt-oss-20b_size9_comp0.4_123".

    Returns:
        The complexity as a float, or None.

    Example:
        >>> _complexity_of("together_ai_openai_gpt-oss-20b_size9_comp0.4_123")
        0.4
        >>> _complexity_of("no_complexity_here") is None
        True
    """
    match = COMPLEXITY_RE.search(trajectory_name)
    return float(match.group(1)) if match else None


@torch.no_grad()
def _score_all_cells(
    probe: BinaryCognitiveMapProbe,
    base_act: torch.Tensor,
    positions: torch.Tensor,
    batch_size: int,
    verbose: bool,
) -> torch.Tensor:
    """Positive-class probability for every (entry, cell), shaped (T, C).

    Builds the `(B*C, D+2)` block for a chunk of entries with tensor ops rather than
    walking a Dataset row by row: at C=25 that is ~25x fewer Python-level fetches, which is
    the difference between minutes and an hour on the held-out tree's 87k tokens.

    Args:
        probe: The loaded binary probe (applies its own normalization).
        base_act: (T, D) activations, un-normalized.
        positions: (T, C, 2) cell coordinates.
        batch_size: Target number of ROWS per forward pass.
        verbose: Print progress.

    Returns:
        (T, C) float tensor of positive-class probabilities, on CPU.
    """
    num_entries, activation_dim = base_act.shape
    cells_per_entry = positions.shape[1]
    entries_per_chunk = max(1, batch_size // max(1, cells_per_entry))

    out = torch.empty((num_entries, cells_per_entry), dtype=torch.float32)
    for start in range(0, num_entries, entries_per_chunk):
        stop = min(start + entries_per_chunk, num_entries)
        chunk = stop - start
        acts = base_act[start:stop].to(probe.device)
        coords = positions[start:stop].float().to(probe.device)
        rows = torch.cat(
            [acts.unsqueeze(1).expand(chunk, cells_per_entry, activation_dim), coords],
            dim=2,
        ).reshape(chunk * cells_per_entry, activation_dim + 2)
        out[start:stop] = probe.predict_positive_proba(rows).reshape(chunk, cells_per_entry).float().cpu()
        if verbose and (start // entries_per_chunk) % 200 == 0:
            print(f"  scored {stop}/{num_entries} entries", flush=True)
    return out


def _block(labels: torch.Tensor, scores: torch.Tensor, threshold: float) -> dict:
    """One metric block for a subset of rows.

    Args:
        labels: (N,) binary ground truth.
        scores: (N,) positive-class probabilities.
        threshold: Probability above which a cell is called positive.

    Returns:
        Dict from `binary_metrics`.
    """
    return binary_metrics(labels, (scores > threshold).long(), scores)


def _grouped(
    keys: list,
    labels: torch.Tensor,
    scores: torch.Tensor,
    threshold: float,
    cells_per_entry: int,
) -> dict:
    """Metric blocks keyed by a per-entry grouping key.

    Args:
        keys: One key per manifest entry (None entries are skipped).
        labels: (T*C,) binary ground truth.
        scores: (T*C,) positive-class probabilities.
        threshold: Decision threshold.
        cells_per_entry: C, used to expand an entry key over its cells.

    Returns:
        Dict of {str(key): metric block}, ordered by key.
    """
    by_key: dict = defaultdict(list)
    for entry_idx, key in enumerate(keys):
        if key is not None:
            by_key[key].append(entry_idx)

    out = {}
    for key in sorted(by_key):
        entry_idx = torch.tensor(by_key[key], dtype=torch.long)
        offsets = entry_idx.unsqueeze(1) * cells_per_entry + torch.arange(cells_per_entry).unsqueeze(0)
        flat = offsets.reshape(-1)
        out[str(key)] = _block(labels[flat], scores[flat], threshold)
    return out


def eval_binary_cognitive_map_probe(
    probe_path: str,
    data_path: str,
    output_path: str | None = None,
    threshold: float = 0.5,
    batch_size: int = 8192,
    cache_activations: bool = False,
    device: str | None = None,
    verbose: bool = True,
) -> dict:
    """Evaluate a saved binary cognitive map probe on a prepared v3 dataset.

    Every (token, cell) the manifest names becomes one scored row, which is what the
    trainer saw. Point it at the split-eval half for the matched comparison and at a
    held-out dataset's manifest for the generalisation number; nothing but the manifest
    changes between the two.

    Args:
        probe_path: Path to a .pt written by `train_binary_cognitive_map_probe`
        data_path: Path to a v3 prepared dataset directory (probe_type=grid_tile)
        output_path: Where to write the results JSON. Defaults to
            `eval_{probe stem}_{dataset dirname}.json` beside the probe.
        threshold: Positive-class probability above which a cell is called positive
        batch_size: Rows per forward pass
        cache_activations: Read/write `_packed_activations.pt` beside the manifest. Shared
            with the trainer's cache, so the eval half costs nothing extra after training.
        device: Device to use for inference (e.g. "cuda", "cpu")
        verbose: Print progress and the metric tables

    Returns:
        The results dict, also written to `output_path`.

    Raises:
        FileNotFoundError: If the probe or the dataset does not exist.
        ValueError: If the dataset is not a v3 prepared directory.
    """
    probe_file = Path(probe_path)
    dataset_dir = Path(data_path)
    if not probe_file.exists():
        raise FileNotFoundError(f"Probe not found: {probe_file}")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")
    if detect_format(dataset_dir) != 3:
        raise ValueError(
            f"{dataset_dir} is not a v3 prepared dataset directory. This command scores a "
            "probe on the same artifact the trainer reads, not on a gathered tree."
        )

    torch_device = resolve_device(device)
    probe = BinaryCognitiveMapProbe.load(probe_file, device=str(torch_device))
    if verbose:
        print(f"Using device: {torch_device}")
        print(f"Probe: {probe_file} ('{probe.positive_class}' vs rest, {probe.model_type})")

    # Read the manifest before the tensors: both checks below are cheap, and the held-out
    # dataset is ~1 GB of activations to load before finding out the probe cannot read it.
    manifest_path = resolve_manifest_path(dataset_dir)
    manifest = load_v3_manifest(manifest_path)
    if manifest.get("probe_type") != "grid_tile":
        raise ValueError(f"Expected probe_type='grid_tile' in {manifest_path}, got {manifest.get('probe_type')!r}.")
    if manifest["activation_dim"] + 2 != probe.input_dim:
        raise ValueError(
            f"Probe expects input_dim {probe.input_dim} (activation dim {probe.input_dim - 2}) "
            f"but {dataset_dir} holds activation dim {manifest['activation_dim']}. The probe and "
            "the dataset were built from different layers or token categories."
        )

    compact = _drop_nan_entries(
        _load_grid_tile_compact_cached(manifest, manifest_path, cache_activations, verbose), verbose
    )

    base_act = compact["base_act"]
    positions = compact["positions"]
    cells_per_entry = compact["C"]
    num_entries = base_act.shape[0]

    labels = (compact["labels"] == probe.positive_cell_id).long()
    if verbose:
        print(f"Scoring {num_entries} entries x {cells_per_entry} cells = {num_entries * cells_per_entry} rows")

    scores = _score_all_cells(probe, base_act, positions, batch_size, verbose)

    flat_labels = labels.reshape(-1)
    flat_scores = scores.reshape(-1)

    sizes = compact.get("sizes")
    size_keys = [int(s) for s in sizes] if sizes is not None else [None] * num_entries
    complexity_keys = [_complexity_of(name) for name in compact["trajectory_names"]]
    size_complexity_keys = [
        None if s is None or c is None else f"size{s}|comp{c}"
        for s, c in zip(size_keys, complexity_keys, strict=False)
    ]

    results = {
        "probe": {
            "path": str(probe_file.resolve()),
            "positive_class": probe.positive_class,
            "positive_cell_id": probe.positive_cell_id,
            "model_type": probe.model_type,
            "input_dim": probe.input_dim,
            "normalized": probe.normalized,
            "train_config": probe.config,
        },
        "data": {
            "path": str(dataset_dir.resolve()),
            "activations_root": manifest.get("activations_root"),
            "selection": manifest.get("selection"),
            "split": manifest.get("split"),
            "n_entries": int(num_entries),
            "n_trajectories": len(set(compact["trajectory_names"])),
            "n_cells_per_entry": int(cells_per_entry),
            "n_rows": int(flat_labels.numel()),
        },
        "config": {"threshold": threshold, "batch_size": batch_size, "device": str(torch_device)},
        "global": _block(flat_labels, flat_scores, threshold),
        "by_size": _grouped(size_keys, flat_labels, flat_scores, threshold, cells_per_entry),
        "by_complexity": _grouped(complexity_keys, flat_labels, flat_scores, threshold, cells_per_entry),
        "by_size_complexity": _grouped(size_complexity_keys, flat_labels, flat_scores, threshold, cells_per_entry),
    }

    if output_path is None:
        final_output_path = probe_file.parent / f"eval_{probe_file.stem}_{dataset_dir.name}.json"
    else:
        final_output_path = Path(output_path)
    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(final_output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    if verbose:
        _print_results(results)
    print(f"\nResults saved to {final_output_path}")
    return results


def _print_results(results: dict) -> None:
    """Print the global block and the per-size breakdown.

    Args:
        results: The dict returned by `eval_binary_cognitive_map_probe`.
    """
    symbol = results["probe"]["positive_class"]
    data = results["data"]
    print("\n" + "=" * 87)
    print(
        f"'{symbol}' vs rest — {data['n_rows']} rows, {data['n_entries']} tokens, {data['n_trajectories']} trajectories"
    )
    print("=" * 87)

    columns = ("n_rows", "positive_rate", "accuracy", "balanced_accuracy", "precision", "recall", "f1", "auroc")
    header = f"{'Group':<16}" + "".join(
        f"{c:>14}" for c in ("rows", "pos rate", "acc", "bal acc", "prec", "recall", "f1", "auroc")
    )
    print(header)
    print("-" * len(header))

    def row(label: str, block: dict) -> None:
        cells = [f"{block['n_rows']:>14}"]
        cells += [f"{block[c]:>14.4f}" for c in columns[1:]]
        print(f"{label:<16}" + "".join(cells))

    row("global", results["global"])
    for key, block in results["by_size"].items():
        row(f"  size {key}", block)
    for key, block in results["by_complexity"].items():
        row(f"  comp {key}", block)
    print("-" * len(header))
