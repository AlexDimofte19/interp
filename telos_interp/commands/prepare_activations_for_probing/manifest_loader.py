"""Loader for v3 prepared-activation manifests.

A v3 prepared dataset is a directory containing:
  manifest.json
  activations/{traj_name}.pt              # single-size mode
  activations/size{N}/{traj_name}.pt      # multi-size mode

Each per-trajectory .pt holds a single (D,) activation tensor. Per-cell
positions/labels (for grid_tile) and other probe-type metadata live in
manifest.json. The trainer loads activations into a (T, D) tensor and
materializes per-cell rows on the fly via a Dataset.

Two manifests copy no activations at all and instead reference the gathered tree in
place, through an absolute `activations_root`: `next_action`, and `grid_tile` prepared
with a token selection. Both are **token-major** -- one entry per (token, layer) rather
than per trajectory, so a trajectory name repeats across entries. A token-major
`grid_tile` manifest carries its per-cell payload once per (trajectory, step) under
`cells`, keyed by each entry's `cells_key`, rather than repeating ~225 positions on every
one of that trajectory's ~20 entries.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


def load_v3_manifest(manifest_path: Path) -> dict:
    """Read manifest.json and return the parsed dict (no tensor loading)."""
    with open(manifest_path) as f:
        return json.load(f)


def load_per_trajectory_activations(
    manifest: dict,
    manifest_path: Path,
) -> torch.Tensor:
    """Eager-load all per-trajectory activations into a single (T, D) tensor.

    Total size is T * D * 4 bytes. For T=5000, D=131072 that's ~2.6 GB —
    comfortably fits in RAM for typical workloads.

    `act_path` is relative to the manifest for a dataset that copied its activations, and
    relative to the absolute `activations_root` for one that references the gathered tree
    in place. A token-major manifest has one entry per (token, layer), so T counts entries,
    not trajectories.
    """
    root = Path(manifest["activations_root"]) if "activations_root" in manifest else manifest_path.parent
    entries = manifest["trajectories"]
    num_trajectories = len(entries)
    activation_dim = manifest["activation_dim"]
    out = torch.empty((num_trajectories, activation_dim), dtype=torch.float32)
    for i, entry in enumerate(entries):
        act_path = Path(entry["act_path"])
        act = torch.load(
            act_path if act_path.is_absolute() else root / act_path,
            map_location="cpu",
            weights_only=True,
        )
        out[i] = act.float()
    return out


def load_grid_tile_compact(manifest: dict, manifest_path: Path) -> dict:
    """Load grid_tile manifest into compact tensors.

    On a token-major manifest each entry is one (token, layer) and the per-cell payload
    lives once per (trajectory, step) under `cells`; T is then the number of token samples
    and `trajectory_names` repeats, which is what the trainer needs to refuse a row-level
    train/eval split.

    Returns:
        Dict with keys:
            base_act: (T, D) float32
            positions: (T, C, 2) int16 — row/col per cell
            labels: (T, C) int64 — grid_tile_id per cell
            trajectory_names: list[str]
            sizes: list[int] | None — per-trajectory size (multi-size only)
            C: int — cells per trajectory
            D: int — activation dimension
    """
    base_act = load_per_trajectory_activations(manifest, manifest_path)
    entries = manifest["trajectories"]
    cells = manifest.get("cells")
    num_trajectories = len(entries)
    cells_per_trajectory = manifest["num_cells_per_trajectory"]
    activation_dim = manifest["activation_dim"]
    positions = torch.empty((num_trajectories, cells_per_trajectory, 2), dtype=torch.int16)
    labels = torch.empty((num_trajectories, cells_per_trajectory), dtype=torch.int64)
    names: list[str] = []
    sizes: list[int] = []
    has_sizes = False
    for i, entry in enumerate(entries):
        payload = cells[entry["cells_key"]] if cells is not None else entry
        positions[i] = torch.tensor(payload["positions"], dtype=torch.int16)
        labels[i] = torch.tensor(payload["labels"], dtype=torch.int64)
        names.append(entry["name"])
        if "size" in entry:
            has_sizes = True
            sizes.append(int(entry["size"]))
    return {
        "base_act": base_act,
        "positions": positions,
        "labels": labels,
        "trajectory_names": names,
        "sizes": sizes if has_sizes else None,
        "C": cells_per_trajectory,
        "D": activation_dim,
    }


def load_distance_compact(manifest: dict, manifest_path: Path) -> dict:
    """Load distance manifest into compact tensors.

    Returns:
        Dict with keys:
            base_act: (T, D) float32
            labels: (T,) int64 — astar_distance per trajectory
            trajectory_names: list[str]
            sizes: list[int] | None
            D: int
    """
    base_act = load_per_trajectory_activations(manifest, manifest_path)
    entries = manifest["trajectories"]
    labels = torch.tensor(
        [int(e["astar_distance"]) for e in entries],
        dtype=torch.int64,
    )
    names = [e["name"] for e in entries]
    sizes_list = [int(e["size"]) for e in entries if "size" in e]
    return {
        "base_act": base_act,
        "labels": labels,
        "trajectory_names": names,
        "sizes": sizes_list if sizes_list else None,
        "D": manifest["activation_dim"],
    }


def load_action_sequence_compact(manifest: dict, manifest_path: Path) -> dict:
    """Load action_sequence manifest into compact tensors.

    Returns:
        Dict with keys:
            base_act: (T, D) float32
            labels: (T, max_seq_len) int64 — padded with -1
            sequence_lengths: (T,) int64
            trajectory_names: list[str]
            sizes: list[int] | None
            D: int
            max_seq_len: int
    """
    base_act = load_per_trajectory_activations(manifest, manifest_path)
    entries = manifest["trajectories"]
    num_trajectories = len(entries)
    max_seq_len = manifest["max_seq_len"]
    labels = torch.full((num_trajectories, max_seq_len), -1, dtype=torch.int64)
    sequence_lengths = torch.empty((num_trajectories,), dtype=torch.int64)
    names: list[str] = []
    sizes: list[int] = []
    has_sizes = False
    for i, entry in enumerate(entries):
        actions = entry["actions"]
        labels[i, : len(actions)] = torch.tensor(actions, dtype=torch.int64)
        sequence_lengths[i] = len(actions)
        names.append(entry["name"])
        if "size" in entry:
            has_sizes = True
            sizes.append(int(entry["size"]))
    return {
        "base_act": base_act,
        "labels": labels,
        "sequence_lengths": sequence_lengths,
        "trajectory_names": names,
        "sizes": sizes if has_sizes else None,
        "D": manifest["activation_dim"],
        "max_seq_len": max_seq_len,
    }


def load_next_action_compact(manifest: dict, manifest_path: Path) -> dict:
    """Load a next_action manifest into compact tensors.

    Unlike the other probe types, next_action manifests do not copy activations: each
    entry under "samples" references an existing gathered token .pt file (relative to
    `activations_root`) and carries a single action label. Each token is an i.i.d.
    sample, so this returns flat (N, D) activations and (N,) labels with no trajectory
    grouping.

    Returns:
        Dict with keys:
            base_act: (N, D) float32 — one row per token sample
            labels: (N,) int64 — action id per sample
            D: int — activation dimension
    """
    entries = manifest["samples"]
    num_samples = len(entries)
    activation_dim = manifest["activation_dim"]
    root = Path(manifest["activations_root"])

    base_act = torch.empty((num_samples, activation_dim), dtype=torch.float32)
    labels = torch.empty((num_samples,), dtype=torch.int64)
    for i, entry in enumerate(entries):
        act_path = Path(entry["act_path"])
        full_path = act_path if act_path.is_absolute() else root / act_path
        act = torch.load(full_path, map_location="cpu", weights_only=True)
        base_act[i] = act.float()
        labels[i] = int(entry["label"])

    return {
        "base_act": base_act,
        "labels": labels,
        "D": activation_dim,
    }


class GridTileCompactDataset(Dataset):
    """Materializes [activation, row, col] rows on the fly from compact tensors.

    Holds (T, D) activations + (T, C, 2) positions + (T, C) labels in RAM.
    `__getitem__(idx)` decodes (t, c) = divmod(idx, C) and concatenates a single
    activation with the cell's (row, col), returning a (D+2,) float tensor and
    its int64 label. No file I/O at item-fetch time, so it's safe with
    DataLoader's num_workers > 0.
    """

    def __init__(
        self,
        base_act: torch.Tensor,
        positions: torch.Tensor,
        labels: torch.Tensor,
    ):
        if base_act.shape[0] != positions.shape[0] or base_act.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Trajectory-axis size mismatch: base_act={base_act.shape[0]}, "
                f"positions={positions.shape[0]}, labels={labels.shape[0]}"
            )
        self.base_act = base_act
        self.positions = positions
        self.labels = labels
        self.num_trajectories, self.cells_per_trajectory = labels.shape

    def __len__(self) -> int:
        return self.num_trajectories * self.cells_per_trajectory

    def __getitem__(self, idx: int):
        t, c = divmod(idx, self.cells_per_trajectory)
        x = torch.cat(
            [self.base_act[t].float(), self.positions[t, c].float()],
            dim=0,
        )
        return x, self.labels[t, c]


class IndexedGridTileCompactDataset(Dataset):
    """Like GridTileCompactDataset but indexes into a precomputed list of (t, c) pairs.

    Used for the eval-side filter where some cells must be excluded (their labels
    aren't in the training class set).
    """

    def __init__(
        self,
        base_act: torch.Tensor,
        positions: torch.Tensor,
        labels: torch.Tensor,
        flat_indices: torch.Tensor,
    ):
        self.base_act = base_act
        self.positions = positions
        self.labels = labels
        self.flat_indices = flat_indices
        _, self.cells_per_trajectory = labels.shape

    def __len__(self) -> int:
        return self.flat_indices.numel()

    def __getitem__(self, idx: int):
        flat = int(self.flat_indices[idx].item())
        t, c = divmod(flat, self.cells_per_trajectory)
        x = torch.cat(
            [self.base_act[t].float(), self.positions[t, c].float()],
            dim=0,
        )
        return x, self.labels[t, c]


def build_flat_grid_tile(
    base_act: torch.Tensor,
    positions: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize the flat (T*C, D+2) tensor + (T*C,) labels from compact form.

    Used only on the class-balancing path, where the upsampler operates on
    flat tensors. Peak memory matches v1 (T*C*(D+2)*4 bytes), so callers
    should only invoke this after subset/split has reduced T.
    """
    num_trajectories, cells_per_trajectory = labels.shape
    activation_dim = base_act.shape[1]
    base_act_expanded = base_act.unsqueeze(1).expand(num_trajectories, cells_per_trajectory, activation_dim)
    positions_float = positions.float()
    flat_x = torch.cat([base_act_expanded, positions_float], dim=2).reshape(
        num_trajectories * cells_per_trajectory, activation_dim + 2
    )
    flat_y = labels.reshape(num_trajectories * cells_per_trajectory)
    return flat_x, flat_y


def detect_format(path: Path) -> int:
    """Detect prepared-data format.

    Returns:
        3 if `path` is a v3 manifest directory or manifest.json file.
        1 if `path` is a legacy single .pt file.
    """
    if path.is_file() and path.name == "manifest.json":
        return 3
    if path.is_dir() and (path / "manifest.json").exists():
        return 3
    return 1


def resolve_manifest_path(path: Path) -> Path:
    """Given a v3 input path (dir or file), return the manifest.json path."""
    if path.is_file() and path.name == "manifest.json":
        return path
    if path.is_dir():
        return path / "manifest.json"
    raise FileNotFoundError(f"No v3 manifest found at {path}")
