"""Train cognitive map probing classifiers on prepared activations."""

from pathlib import Path
from typing import Literal

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from telos_interp.commands.prepare_activations_for_probing.manifest_loader import (
    GridTileCompactDataset,
    IndexedGridTileCompactDataset,
    build_flat_grid_tile,
    detect_format,
    load_grid_tile_compact,
    load_v3_manifest,
    resolve_manifest_path,
)
from telos_interp.grid_utils import CELL_ID_TO_SYMBOL
from telos_interp.probe_models import (
    ModelType,
)
from telos_interp.probe_models import (
    create_classification_model as _create_model,
)
from telos_interp.training import (
    compute_normalization_params,
    normalize_activations,
    resolve_device,
    set_seed,
)

ClassWeight = Literal["balanced"] | None


class CognitiveMapProbe:
    """A trained cognitive map probe with built-in normalization and prediction.

    This class encapsulates a trained probe model along with normalization
    parameters and label mappings, providing a simple interface for inference.

    Example:
        # Training
        probe = train_cognitive_map_probe("train.pt", normalize=True)
        probe.save("probe.pt")

        # Inference
        probe = CognitiveMapProbe.load("probe.pt")
        probs = probe.predict_proba(activations)  # (N, num_classes)
        labels = probe.predict(activations)        # (N,) original label IDs
    """

    def __init__(
        self,
        model: nn.Module,
        model_type: ModelType,
        input_dim: int,
        label_to_idx: dict[int, int],
        idx_to_label: dict[int, int],
        hidden_dims: list[int] | None = None,
        dropout: float | None = None,
        scaler_mean: torch.Tensor | None = None,
        scaler_std: torch.Tensor | None = None,
        config: dict | None = None,
        results: dict | None = None,
        device: torch.device | str | None = None,
    ):
        """Initialize a CognitiveMapProbe.

        Args:
            model: The trained nn.Module (LogisticRegressionProbe or MLPProbe)
            model_type: Type of model ("lr" or "mlp")
            input_dim: Input dimension of the model
            label_to_idx: Maps original label ID -> model output index
            idx_to_label: Maps model output index -> original label ID
            hidden_dims: Hidden layer dimensions (for MLP)
            dropout: Dropout rate (for MLP)
            scaler_mean: Mean for input normalization (None if no normalization)
            scaler_std: Std for input normalization (None if no normalization)
            config: Training configuration (for provenance)
            results: Training results (accuracy, metrics, etc.)
            device: Device to use for inference
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.model = model.to(self.device)
        self.model.eval()

        self.model_type = model_type
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

        self.label_to_idx = label_to_idx
        self.idx_to_label = idx_to_label
        self.config = config or {}
        self.results = results or {}

        # Store normalization parameters
        if scaler_mean is not None:
            self.scaler_mean = scaler_mean.to(self.device)
            self.scaler_std = scaler_std.to(self.device)
        else:
            self.scaler_mean = None
            self.scaler_std = None

    @property
    def num_classes(self) -> int:
        """Number of classes the probe predicts."""
        return len(self.label_to_idx)

    @property
    def normalized(self) -> bool:
        """Whether this probe applies input normalization."""
        return self.scaler_mean is not None

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply normalization if enabled."""
        if self.scaler_mean is not None:
            return (x - self.scaler_mean) / self.scaler_std
        return x

    @torch.no_grad()
    def predict_proba(self, activations: torch.Tensor) -> torch.Tensor:
        """Get class probabilities for activations.

        Args:
            activations: Input tensor of shape (N, input_dim)

        Returns:
            Tensor of shape (N, num_classes) with class probabilities
        """
        self.model.eval()
        x = activations.float().to(self.device)
        x = self._normalize(x)
        logits = self.model(x)
        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict(self, activations: torch.Tensor) -> torch.Tensor:
        """Get predicted original label IDs for activations.

        Args:
            activations: Input tensor of shape (N, input_dim)

        Returns:
            Tensor of shape (N,) with original label IDs (not internal indices)
        """
        probs = self.predict_proba(activations)
        pred_indices = torch.argmax(probs, dim=-1)
        # Map indices back to original label IDs
        original_labels = torch.tensor(
            [self.idx_to_label[idx.item()] for idx in pred_indices],
            dtype=torch.long,
            device=activations.device,
        )
        return original_labels

    def save(self, path: str | Path) -> None:
        """Save the probe to a file."""
        save_data = {
            "model_state_dict": self.model.state_dict(),
            "model_type": self.model_type,
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "label_to_idx": self.label_to_idx,
            "idx_to_label": self.idx_to_label,
            "scaler_mean": self.scaler_mean.cpu() if self.scaler_mean is not None else None,
            "scaler_std": self.scaler_std.cpu() if self.scaler_std is not None else None,
            "config": self.config,
            "results": self.results,
        }
        torch.save(save_data, path)

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "CognitiveMapProbe":
        """Load a probe from a file."""
        device_str = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        torch_device = torch.device(device_str)

        data = torch.load(path, map_location=torch_device, weights_only=False)

        model = _create_model(
            model_type=data["model_type"],
            input_dim=data["input_dim"],
            num_classes=data["num_classes"],
            hidden_dims=data["hidden_dims"] or [],
            dropout=data["dropout"] or 0.0,
        )
        model.load_state_dict(data["model_state_dict"])

        return cls(
            model=model,
            model_type=data["model_type"],
            input_dim=data["input_dim"],
            label_to_idx=data["label_to_idx"],
            idx_to_label=data["idx_to_label"],
            hidden_dims=data["hidden_dims"],
            dropout=data["dropout"],
            scaler_mean=data.get("scaler_mean"),
            scaler_std=data.get("scaler_std"),
            config=data.get("config"),
            results=data.get("results"),
            device=torch_device,
        )


def _print_debug_grids(
    positions: torch.Tensor,
    true_labels: torch.Tensor,
    pred_labels: torch.Tensor,
    idx_to_label: dict[int, int],
) -> None:
    """Print debug grids showing observation vs prediction for one trajectory's grid.

    Args:
        positions: Tensor of shape (n, 2) with [row_id, col_id] for each cell (original, non-normalized)
        true_labels: Ground truth label indices (remapped)
        pred_labels: Predicted label indices (remapped)
        idx_to_label: Maps model output index -> original cell ID
    """
    n = len(positions)

    # Convert positions to int (should already be integer values)
    positions = positions.int()  # (n, 2) - row_id, col_id

    # Determine grid size from positions
    max_row = positions[:, 0].max().item() + 1
    max_col = positions[:, 1].max().item() + 1
    grid_size = max(max_row, max_col)

    # Create grids (filled with '·' for unseen positions)
    true_grid = [["·" for _ in range(grid_size)] for _ in range(grid_size)]
    pred_grid = [["·" for _ in range(grid_size)] for _ in range(grid_size)]

    # Fill in the grids
    correct_count = 0
    for i in range(n):
        row, col = positions[i].tolist()
        true_label_id = idx_to_label[true_labels[i].item()]
        pred_label_id = idx_to_label[pred_labels[i].item()]
        true_grid[row][col] = CELL_ID_TO_SYMBOL[true_label_id]
        pred_grid[row][col] = CELL_ID_TO_SYMBOL[pred_label_id]
        if true_label_id == pred_label_id:
            correct_count += 1

    # Print grids side by side
    print("\n" + "=" * 60)
    print("DEBUG: Single Trajectory Grid Comparison")
    print(f"(grid size {grid_size}x{grid_size}, {n} cells)")
    print("=" * 60)

    # Header
    header_width = grid_size * 2 - 1
    print(f"\n{'Observation (Ground Truth)':<{header_width + 5}}  {'Prediction'}")
    print("-" * header_width + "     " + "-" * header_width)

    # Print rows side by side
    for row_idx in range(grid_size):
        true_row = " ".join(true_grid[row_idx])
        pred_row = " ".join(pred_grid[row_idx])
        print(f"{true_row}     {pred_row}")

    print(f"\nGrid accuracy: {correct_count}/{n} ({100 * correct_count / n:.1f}%)")
    print("=" * 60 + "\n")


def _iter_index_batches(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    device: torch.device,
    shuffle: bool,
):
    """Yield (batch_x, batch_y) by slicing CPU tensors directly.

    Avoids DataLoader/TensorDataset collate overhead: a single vectorized
    gather per batch, then one transfer. Inputs are kept in their stored
    dtype (float16) and cast to float32 on-device per batch to halve the
    host->device transfer volume.
    """
    n = x.shape[0]
    if shuffle:
        # Build the permutation on the tensor's own device so that, when the
        # data is GPU-resident, the gather happens entirely on the GPU.
        order = torch.randperm(n, device=x.device)
    else:
        order = None

    for start in range(0, n, batch_size):
        if order is None:
            batch_x = x[start : start + batch_size]
            batch_y = y[start : start + batch_size]
        else:
            idx = order[start : start + batch_size]
            batch_x = x[idx]
            batch_y = y[idx]
        batch_x = batch_x.to(device, non_blocking=True).float()
        batch_y = batch_y.to(device, non_blocking=True)
        yield batch_x, batch_y


def _train_one_epoch(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    batch_size: int,
) -> float:
    """Train one epoch using direct index-batching. Returns average loss."""
    model.train()
    total_loss = torch.zeros((), device=device)
    num_batches = 0

    for batch_x, batch_y in _iter_index_batches(x, y, batch_size, device, shuffle=True):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach()
        num_batches += 1

    return (total_loss / num_batches).item() if num_batches > 0 else 0.0


def _evaluate(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    batch_size: int,
) -> dict:
    """Evaluate model and return metrics."""
    model.eval()
    total_loss = torch.zeros((), device=device)
    num_batches = 0

    # Confusion matrix accumulated on-device: rows = ground truth, cols = prediction.
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)

    with torch.no_grad():
        for batch_x, batch_y in _iter_index_batches(x, y, batch_size, device, shuffle=False):
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            total_loss += loss.detach()
            num_batches += 1

            predicted = torch.argmax(outputs, dim=1)
            flat = batch_y * num_classes + predicted
            confusion += torch.bincount(flat, minlength=num_classes * num_classes).reshape(
                num_classes, num_classes
            )

    confusion = confusion.cpu()
    class_tp = confusion.diag().to(torch.float64)
    class_gt_support = confusion.sum(dim=1).to(torch.float64)
    class_pred_count = confusion.sum(dim=0).to(torch.float64)
    total = int(confusion.sum().item())
    correct = int(class_tp.sum().item())

    # Compute per-class precision, recall, F1, accuracy
    per_class_metrics = {}
    for i in range(num_classes):
        tp = class_tp[i].item()
        gt_support = class_gt_support[i].item()
        pred_count = class_pred_count[i].item()

        precision = tp / pred_count if pred_count > 0 else 0.0
        recall = tp / gt_support if gt_support > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = tp / gt_support if gt_support > 0 else 0.0  # Per-class accuracy

        per_class_metrics[i] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "gt_support": int(gt_support),
            "predicted": int(pred_count),
        }

    # Compute balanced accuracy (mean of per-class accuracies)
    per_class_accuracies = [m["accuracy"] for m in per_class_metrics.values() if m["gt_support"] > 0]
    balanced_accuracy = sum(per_class_accuracies) / len(per_class_accuracies) if per_class_accuracies else 0.0

    return {
        "loss": (total_loss / num_batches).item() if num_batches > 0 else 0.0,
        "accuracy": correct / total if total > 0 else 0.0,
        "balanced_accuracy": balanced_accuracy,
        "per_class_metrics": per_class_metrics,
        "num_samples": total,
    }


def _get_balanced_indices(
    labels: torch.Tensor,
    seed: int = 42,
    per_class_max_count: int | None = None,
) -> torch.Tensor:
    """Return shuffled flat indices that balance classes, without materializing data."""
    torch.manual_seed(seed)
    unique_classes = torch.unique(labels)
    class_counts = {c.item(): (labels == c).sum().item() for c in unique_classes}
    target_count = max(class_counts.values())
    if per_class_max_count is not None:
        target_count = min(target_count, per_class_max_count)

    balanced_indices = []
    for class_id in unique_classes:
        class_indices = torch.where(labels == class_id)[0]
        current_count = len(class_indices)
        if current_count > target_count:
            perm = torch.randperm(current_count)[:target_count]
            balanced_indices.append(class_indices[perm])
        elif current_count < target_count:
            num_additional = target_count - current_count
            additional_indices = class_indices[torch.randint(0, current_count, (num_additional,))]
            balanced_indices.append(torch.cat([class_indices, additional_indices]))
        else:
            balanced_indices.append(class_indices)

    all_indices = torch.cat(balanced_indices)
    perm = torch.randperm(len(all_indices))
    return all_indices[perm]


def _balance_classes_by_upsampling(
    activations: torch.Tensor,
    labels: torch.Tensor,
    seed: int = 42,
    per_class_max_count: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Balance classes by upsampling minority classes to match the majority class.

    Args:
        activations: Input activations tensor
        labels: Label tensor
        seed: Random seed for reproducibility
        per_class_max_count: Maximum samples per class. Classes exceeding this will be
                   downsampled before balancing. If None, uses the largest class size.

    Returns:
        Tuple of (balanced_activations, balanced_labels)
    """
    torch.manual_seed(seed)

    unique_classes = torch.unique(labels)
    class_counts = {c.item(): (labels == c).sum().item() for c in unique_classes}

    # Determine target count: cap at per_class_max_count if provided
    target_count = max(class_counts.values())
    if per_class_max_count is not None:
        target_count = min(target_count, per_class_max_count)

    balanced_indices = []

    for class_id in unique_classes:
        class_indices = torch.where(labels == class_id)[0]
        current_count = len(class_indices)

        if current_count > target_count:
            # Downsample: randomly select target_count indices
            perm = torch.randperm(current_count)[:target_count]
            balanced_indices.append(class_indices[perm])
        elif current_count < target_count:
            # Upsample: repeat and sample additional indices
            num_additional = target_count - current_count
            additional_indices = class_indices[torch.randint(0, current_count, (num_additional,))]
            balanced_indices.append(torch.cat([class_indices, additional_indices]))
        else:
            balanced_indices.append(class_indices)

    all_indices = torch.cat(balanced_indices)

    # Shuffle
    perm = torch.randperm(len(all_indices))
    all_indices = all_indices[perm]

    return activations[all_indices], labels[all_indices]


def _compute_class_weights(
    labels: torch.Tensor,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute balanced class weights for CrossEntropyLoss.

    Weights are computed as: n_samples / (n_classes * class_count)
    Assumes labels are contiguous (0 to num_classes-1) with no missing classes.

    Args:
        labels: Label tensor (should be remapped to contiguous indices)
        num_classes: Total number of classes
        device: Device to put weights on

    Returns:
        Tensor of class weights
    """
    class_weights = torch.zeros(num_classes, dtype=torch.float32)
    unique_classes, class_counts = torch.unique(labels.cpu(), return_counts=True)
    n_samples = len(labels)

    for class_id, count in zip(unique_classes, class_counts, strict=False):
        class_weights[int(class_id)] = n_samples / (num_classes * count.float())

    return class_weights.to(device)


def _build_indices_for_trajectories(
    trajectory_ids: torch.Tensor,
    num_cells_per_trajectory: int,
) -> torch.Tensor:
    """Build flat sample indices from trajectory IDs.

    Given a set of trajectory IDs, returns the indices of all samples
    belonging to those trajectories. This keeps trajectory data together
    while allowing trajectory-level shuffling and splitting.

    Args:
        trajectory_ids: Tensor of trajectory indices to include
        num_cells_per_trajectory: Number of samples per trajectory

    Returns:
        1D tensor of sample indices (all samples from the specified trajectories)
    """
    indices_list = []
    for traj_id in trajectory_ids:
        start = traj_id.item() * num_cells_per_trajectory
        indices_list.append(torch.arange(start, start + num_cells_per_trajectory))
    return torch.cat(indices_list)


def _load_and_preprocess_train_data_v1(
    train_path: Path,
    verbose: bool,
) -> dict:
    """Load legacy v1 training data and filter NaN values.

    Returns a dict with keys: format=1, activations (N, D+2), labels (N,),
    train_data (raw loaded dict), num_cells_per_trajectory (int | None).
    """
    if verbose:
        print(f"Loading training data from {train_path}")

    train_data = torch.load(train_path, map_location="cpu", weights_only=False)

    # Validate probe type
    probe_type = train_data.get("probe_type", train_data.get("config", {}).get("probe_type"))
    if probe_type is not None and probe_type != "grid_tile":
        raise ValueError(
            f"Expected probe_type='grid_tile', got '{probe_type}'. This command only supports grid_tile probe data."
        )

    activations = train_data["activations"]
    labels = train_data["labels"]

    if verbose:
        print(f"Loaded {activations.shape[0]} samples")
        print(f"Activation dimension (with position): {activations.shape[1]}")
        print(f"Number of unique labels: {labels.unique().shape[0]}")

    # Filter out samples with NaN values
    nan_mask = torch.isnan(activations).any(dim=1)
    num_nan = nan_mask.sum().item()
    if num_nan > 0:
        if verbose:
            print(
                f"WARNING: Found {num_nan} samples with NaN values ({100 * num_nan / len(activations):.1f}%), filtering them out"
            )
        activations = activations[~nan_mask]
        labels = labels[~nan_mask]
        if verbose:
            print(f"Remaining samples: {activations.shape[0]}")

    num_cells_per_trajectory = train_data.get("num_cells_per_trajectory")
    return {
        "format": 1,
        "activations": activations,
        "labels": labels,
        "train_data": train_data,
        "num_cells_per_trajectory": num_cells_per_trajectory,
    }


def _load_and_preprocess_train_data_v3(
    train_path: Path,
    verbose: bool,
) -> dict:
    """Load v3 training data (manifest dir) and filter NaN trajectories.

    NaN filter operates per-trajectory: if a trajectory's activation has any
    NaN, drop the matching row of base_act/positions/labels. The trainer body
    consumes the compact dict throughout subset/split.

    Returns a dict with keys: format=3, compact, manifest, num_cells_per_trajectory.
    """
    if verbose:
        print(f"Loading training data from {train_path}")

    manifest_path = resolve_manifest_path(train_path)
    manifest = load_v3_manifest(manifest_path)

    probe_type = manifest.get("probe_type")
    if probe_type != "grid_tile":
        raise ValueError(
            f"Expected probe_type='grid_tile', got '{probe_type}'. This command only supports grid_tile probe data."
        )

    compact = load_grid_tile_compact(manifest, manifest_path)

    if verbose:
        num_trajectories = compact["base_act"].shape[0]
        print(f"Loaded {num_trajectories} trajectories ({num_trajectories * compact['C']} cells)")
        print(f"Activation dimension (without position): {compact['D']}")
        print(f"Number of unique labels: {compact['labels'].unique().shape[0]}")

    # Per-trajectory NaN filter: drop trajectories whose activation has any NaN.
    nan_mask = torch.isnan(compact["base_act"]).any(dim=1)
    num_nan = nan_mask.sum().item()
    if num_nan > 0:
        keep_mask = ~nan_mask
        compact["base_act"] = compact["base_act"][keep_mask]
        compact["positions"] = compact["positions"][keep_mask]
        compact["labels"] = compact["labels"][keep_mask]
        compact["trajectory_names"] = [n for n, k in zip(compact["trajectory_names"], keep_mask.tolist()) if k]
        if compact.get("sizes") is not None:
            compact["sizes"] = [s for s, k in zip(compact["sizes"], keep_mask.tolist()) if k]
        if verbose:
            num_trajectories = compact["base_act"].shape[0]
            print(
                f"WARNING: Found {num_nan} trajectories with NaN activations, filtering them out. "
                f"Remaining trajectories: {num_trajectories}"
            )

    return {
        "format": 3,
        "compact": compact,
        "manifest": manifest,
        "num_cells_per_trajectory": compact["C"],
    }


def _load_and_preprocess_train_data(
    train_path: Path,
    verbose: bool,
) -> dict:
    """Detect format (v1 .pt or v3 manifest dir) and dispatch."""
    fmt = detect_format(train_path)
    if fmt == 3:
        return _load_and_preprocess_train_data_v3(train_path, verbose)
    return _load_and_preprocess_train_data_v1(train_path, verbose)


def _remap_labels(
    labels: torch.Tensor,
    verbose: bool,
) -> tuple[torch.Tensor, int, dict[int, int], dict[int, int]]:
    """Remap labels to contiguous indices.

    Returns:
        Tuple of (remapped_labels, num_classes, label_to_idx, idx_to_label)
    """
    unique_labels = torch.unique(labels).tolist()
    num_classes = len(unique_labels)
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    if verbose:
        print(f"Classes present in data: {unique_labels}")
        if unique_labels != list(range(num_classes)):
            print(f"Remapping to contiguous indices: {label_to_idx}")

    remapped_labels = torch.tensor([label_to_idx[l.item()] for l in labels], dtype=labels.dtype)
    return remapped_labels, num_classes, label_to_idx, idx_to_label


def _load_separate_eval_data_v1(
    eval_path: Path,
    label_to_idx: dict[int, int],
    verbose: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load and process separate evaluation data from a legacy v1 .pt file.

    Returns:
        Tuple of (eval_activations, eval_labels) — flat (N, D+2) and (N,).
    """
    if verbose:
        print(f"Loading evaluation data from {eval_path}")

    eval_data = torch.load(eval_path, map_location="cpu", weights_only=False)
    eval_activations = eval_data["activations"]
    eval_labels_raw = eval_data["labels"]

    # Remap eval labels using the same mapping (skip labels not in training)
    eval_labels_list = []
    valid_eval_mask = []
    for l in eval_labels_raw:
        if l.item() in label_to_idx:
            eval_labels_list.append(label_to_idx[l.item()])
            valid_eval_mask.append(True)
        else:
            valid_eval_mask.append(False)

    valid_eval_mask_tensor = torch.tensor(valid_eval_mask)
    if not valid_eval_mask_tensor.all():
        num_skipped = (~valid_eval_mask_tensor).sum().item()
        if verbose:
            print(f"WARNING: Skipping {num_skipped} eval samples with labels not in training data")
        eval_activations = eval_activations[valid_eval_mask_tensor]

    eval_labels = torch.tensor(eval_labels_list, dtype=eval_labels_raw.dtype)

    if verbose:
        print(f"Loaded {eval_activations.shape[0]} evaluation samples")

    return eval_activations, eval_labels


def _load_separate_eval_data_v3(
    eval_path: Path,
    label_to_idx: dict[int, int],
    verbose: bool,
) -> dict:
    """Load and process separate evaluation data from a v3 manifest dir.

    Applies the same NaN filter as the train loader and remaps cell labels using
    the training label_to_idx. Cells whose original label isn't in label_to_idx
    are masked out via a per-cell valid mask (kept inside the compact dict).

    Returns a compact dict with the same shape as the train compact (sans
    `format` / `manifest`), augmented with `valid_cell_mask: (T, C) bool` so
    the dataset knows which cells to materialize.
    """
    if verbose:
        print(f"Loading evaluation data from {eval_path}")

    manifest_path = resolve_manifest_path(eval_path)
    manifest = load_v3_manifest(manifest_path)

    probe_type = manifest.get("probe_type")
    if probe_type != "grid_tile":
        raise ValueError(f"Expected probe_type='grid_tile' in eval manifest, got '{probe_type}'.")

    compact = load_grid_tile_compact(manifest, manifest_path)

    # NaN filter on trajectories
    nan_mask = torch.isnan(compact["base_act"]).any(dim=1)
    if nan_mask.any():
        keep_mask = ~nan_mask
        compact["base_act"] = compact["base_act"][keep_mask]
        compact["positions"] = compact["positions"][keep_mask]
        compact["labels"] = compact["labels"][keep_mask]
        compact["trajectory_names"] = [n for n, k in zip(compact["trajectory_names"], keep_mask.tolist()) if k]
        if compact.get("sizes") is not None:
            compact["sizes"] = [s for s, k in zip(compact["sizes"], keep_mask.tolist()) if k]

    # Remap labels using training label_to_idx; cells with unknown labels are masked.
    raw_labels = compact["labels"]
    remapped = torch.full_like(raw_labels, -1)
    valid_cell_mask = torch.zeros_like(raw_labels, dtype=torch.bool)
    for original, idx in label_to_idx.items():
        match = raw_labels == original
        remapped[match] = idx
        valid_cell_mask |= match
    compact["labels"] = remapped
    compact["valid_cell_mask"] = valid_cell_mask

    num_kept = int(valid_cell_mask.sum().item())
    num_total = int(valid_cell_mask.numel())
    if verbose:
        if num_kept < num_total:
            print(
                f"WARNING: {num_total - num_kept} eval cells have labels not in training; "
                f"they will be excluded from the eval set."
            )
        print(f"Loaded {num_kept} valid evaluation cells across {compact['base_act'].shape[0]} trajectories")

    return compact


def _prepare_train_eval_v1(
    load_result: dict,
    eval_data_path: str | None,
    eval_split: float,
    subset: float,
    balance_classes: bool,
    normalize: bool,
    per_class_max_count: int | None,
    seed: int,
    verbose: bool,
) -> dict:
    """V1 (legacy flat .pt) flow: subset, remap, split, balance, normalize, build TensorDatasets."""
    activations = load_result["activations"]
    labels = load_result["labels"]
    num_cells_per_trajectory = load_result["num_cells_per_trajectory"]

    # Subset
    if num_cells_per_trajectory is not None and num_cells_per_trajectory > 0:
        num_trajectories = len(activations) // num_cells_per_trajectory
        num_trajectories_to_keep = max(1, int(num_trajectories * subset))
        trajectory_perm = torch.randperm(num_trajectories)
        selected_trajectories = trajectory_perm[:num_trajectories_to_keep]
        selected_indices = _build_indices_for_trajectories(selected_trajectories, num_cells_per_trajectory)
        activations = activations[selected_indices]
        labels = labels[selected_indices]
        print(f"Subset: keeping {num_trajectories_to_keep}/{num_trajectories} trajectories ({subset * 100:.1f}%)")
        print(f"  Remaining samples: {len(activations)}")
    else:
        perm = torch.randperm(len(activations))
        num_samples_to_keep = max(1, int(len(activations) * subset))
        activations = activations[perm[:num_samples_to_keep]]
        labels = labels[perm[:num_samples_to_keep]]
        print(f"Subset: keeping {num_samples_to_keep} samples ({subset * 100:.1f}%)")

    # Remap labels
    remapped_labels, num_classes, label_to_idx, idx_to_label = _remap_labels(labels, verbose)

    # Debug snapshot
    debug_trajectory_activations = None
    debug_trajectory_positions = None
    debug_trajectory_labels = None
    if num_cells_per_trajectory is not None and num_cells_per_trajectory <= len(activations):
        debug_trajectory_activations = activations[:num_cells_per_trajectory].clone()
        debug_trajectory_positions = activations[:num_cells_per_trajectory, -2:].clone()
        debug_trajectory_labels = remapped_labels[:num_cells_per_trajectory].clone()

    # Train/eval split
    if eval_data_path is None:
        num_samples = activations.shape[0]
        if num_cells_per_trajectory is not None and num_cells_per_trajectory > 0:
            num_trajectories = num_samples // num_cells_per_trajectory
            trajectory_perm = torch.randperm(num_trajectories)
            num_train_trajectories = int(num_trajectories * (1 - eval_split))
            train_trajectory_ids = trajectory_perm[:num_train_trajectories]
            eval_trajectory_ids = trajectory_perm[num_train_trajectories:]
            train_indices = _build_indices_for_trajectories(train_trajectory_ids, num_cells_per_trajectory)
            eval_indices = _build_indices_for_trajectories(eval_trajectory_ids, num_cells_per_trajectory)
            print(f"Split by trajectories: {len(train_trajectory_ids)} train, {len(eval_trajectory_ids)} eval")
        else:
            indices = torch.randperm(num_samples)
            split_idx = int(num_samples * (1 - eval_split))
            train_indices = indices[:split_idx]
            eval_indices = indices[split_idx:]
        train_activations = activations[train_indices]
        train_labels = remapped_labels[train_indices]
        eval_activations = activations[eval_indices]
        eval_labels = remapped_labels[eval_indices]
        print(f"Split: {train_activations.shape[0]} train, {eval_activations.shape[0]} eval")
    else:
        train_activations = activations
        train_labels = remapped_labels
        eval_path = Path(eval_data_path)
        if not eval_path.exists():
            raise FileNotFoundError(f"Evaluation data not found: {eval_path}")
        if detect_format(eval_path) != 1:
            raise ValueError("Train data is v1 (legacy .pt) but eval data is v3 — formats must match.")
        eval_activations, eval_labels = _load_separate_eval_data_v1(eval_path, label_to_idx, verbose)

    # Class balancing
    if balance_classes:
        print("Balancing classes by upsampling...")
        unique, counts = torch.unique(train_labels, return_counts=True)
        print(f"  Before: {dict(zip(unique.tolist(), counts.tolist(), strict=False))}")
        train_activations, train_labels = _balance_classes_by_upsampling(
            train_activations, train_labels, seed=seed, per_class_max_count=per_class_max_count
        )
        unique, counts = torch.unique(train_labels, return_counts=True)
        print(f"  After: {dict(zip(unique.tolist(), counts.tolist(), strict=False))}")
        print(f"  New training set size: {train_activations.shape[0]}")

    # Normalization
    scaler_mean = None
    scaler_std = None
    if normalize:
        scaler_mean, scaler_std = compute_normalization_params(train_activations)
        train_activations = normalize_activations(train_activations, scaler_mean, scaler_std)
        eval_activations = normalize_activations(eval_activations, scaler_mean, scaler_std)
        print(f"Normalization enabled: computed mean/std from {train_activations.shape[0]} training samples")

    train_dataset = TensorDataset(train_activations.float(), train_labels.long())
    eval_dataset = TensorDataset(eval_activations.float(), eval_labels.long())

    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "num_classes": num_classes,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "scaler_mean": scaler_mean,
        "scaler_std": scaler_std,
        "debug_trajectory_activations": debug_trajectory_activations,
        "debug_trajectory_positions": debug_trajectory_positions,
        "debug_trajectory_labels": debug_trajectory_labels,
        "train_labels_for_weights": train_labels,
        "input_dim": train_activations.shape[1],
    }


def _prepare_train_eval_v3(
    load_result: dict,
    eval_data_path: str | None,
    eval_split: float,
    subset: float,
    balance_classes: bool,
    normalize: bool,
    per_class_max_count: int | None,
    seed: int,
    verbose: bool,
) -> dict:
    """V3 (manifest dir) flow: operate on compact (T, D) + (T, C, 2) + (T, C) tensors throughout.

    Default builds a `GridTileCompactDataset` that materializes (D+2,) rows on the fly
    so peak training-time RAM is ~T*D*4 bytes regardless of C. The class-balanced branch
    materializes the flat (T*C, D+2) tensor and reverts to v1 memory characteristics.
    """
    compact = load_result["compact"]
    cells_per_trajectory = compact["C"]
    activation_dim = compact["D"]
    num_trajectories = compact["base_act"].shape[0]

    # Subset by trajectory
    num_to_keep = max(1, int(num_trajectories * subset))
    perm = torch.randperm(num_trajectories)
    keep_idx = perm[:num_to_keep]
    base_act = compact["base_act"][keep_idx]
    positions = compact["positions"][keep_idx]
    raw_labels = compact["labels"][keep_idx]
    print(f"Subset: keeping {num_to_keep}/{num_trajectories} trajectories ({subset * 100:.1f}%)")
    print(f"  Remaining samples: {num_to_keep * cells_per_trajectory}")

    # Remap labels (flatten -> remap -> reshape)
    flat_labels_for_remap = raw_labels.reshape(-1)
    remapped_flat, num_classes, label_to_idx, idx_to_label = _remap_labels(flat_labels_for_remap, verbose)
    remapped_labels = remapped_flat.reshape(raw_labels.shape)

    # Debug snapshot from first trajectory (data already shuffled by subset)
    debug_trajectory_activations = None
    debug_trajectory_positions = None
    debug_trajectory_labels = None
    if num_to_keep > 0:
        first_act = base_act[0]                           # (D,)
        first_pos_float = positions[0].float()            # (C, 2)
        # Reconstruct (C, D+2) for the model. This stores positions un-normalized;
        # if normalization is on we'll compose the position-pad-normalized version
        # back in the main body before calling the model.
        debug_trajectory_activations = torch.cat(
            [first_act.unsqueeze(0).expand(cells_per_trajectory, activation_dim), first_pos_float],
            dim=1,
        ).clone()                                          # (C, D+2)
        debug_trajectory_positions = positions[0].clone()  # (C, 2) original ints
        debug_trajectory_labels = remapped_labels[0].clone()  # (C,)

    # Train/eval split (trajectory-level)
    if eval_data_path is None:
        T = base_act.shape[0]
        traj_perm = torch.randperm(T)
        num_train = int(T * (1 - eval_split))
        train_idx = traj_perm[:num_train]
        eval_idx = traj_perm[num_train:]

        train_base_act = base_act[train_idx]
        train_positions = positions[train_idx]
        train_labels_2d = remapped_labels[train_idx]
        eval_base_act = base_act[eval_idx]
        eval_positions = positions[eval_idx]
        eval_labels_2d = remapped_labels[eval_idx]
        eval_valid_cell_mask = None
        print(f"Split by trajectories: {len(train_idx)} train, {len(eval_idx)} eval")
    else:
        train_base_act = base_act
        train_positions = positions
        train_labels_2d = remapped_labels

        eval_path = Path(eval_data_path)
        if not eval_path.exists():
            raise FileNotFoundError(f"Evaluation data not found: {eval_path}")
        if detect_format(eval_path) != 3:
            raise ValueError("Train data is v3 (manifest dir) but eval data is v1 — formats must match.")
        eval_compact = _load_separate_eval_data_v3(eval_path, label_to_idx, verbose)
        eval_base_act = eval_compact["base_act"]
        eval_positions = eval_compact["positions"]
        eval_labels_2d = eval_compact["labels"]
        eval_valid_cell_mask = eval_compact.get("valid_cell_mask")

    # Class balancing — compute indices only, no data materialization.
    balanced_flat_indices: torch.Tensor | None = None
    if balance_classes:
        print("Balancing classes by upsampling...")
        flat_labels = train_labels_2d.reshape(-1)
        unique, counts = torch.unique(flat_labels, return_counts=True)
        print(f"  Before: {dict(zip(unique.tolist(), counts.tolist(), strict=False))}")
        balanced_flat_indices = _get_balanced_indices(flat_labels, seed=seed, per_class_max_count=per_class_max_count)
        unique, counts = torch.unique(flat_labels[balanced_flat_indices], return_counts=True)
        print(f"  After: {dict(zip(unique.tolist(), counts.tolist(), strict=False))}")
        print(f"  New training set size: {balanced_flat_indices.shape[0]}")

    # Normalization (D-dim only; positions stay un-normalized).
    # Always normalize train_base_act in-place so the lazy dataset sees normalized activations.
    scaler_mean = None
    scaler_std = None
    if normalize:
        scaler_mean_act, scaler_std_act = compute_normalization_params(train_base_act)
        train_base_act = normalize_activations(train_base_act, scaler_mean_act, scaler_std_act)
        n_normalized = train_base_act.shape[0]
        eval_base_act = normalize_activations(eval_base_act, scaler_mean_act, scaler_std_act)
        # Pad to (D+2,) so the saved probe applies an identity transform on the position cols
        # — and so the body's debug-display code, which normalizes (debug - scaler_mean) / scaler_std,
        # handles the v3 debug snapshot the same way as v1.
        scaler_mean = torch.cat([scaler_mean_act, torch.zeros(2, dtype=scaler_mean_act.dtype)])
        scaler_std = torch.cat([scaler_std_act, torch.ones(2, dtype=scaler_std_act.dtype)])
        print(f"Normalization enabled: computed mean/std from {n_normalized} training samples")

    # Build datasets
    if balance_classes:
        train_dataset: Dataset = IndexedGridTileCompactDataset(
            train_base_act, train_positions, train_labels_2d, balanced_flat_indices
        )
        train_labels_for_weights = train_labels_2d.reshape(-1)[balanced_flat_indices]
    else:
        train_dataset = GridTileCompactDataset(train_base_act, train_positions, train_labels_2d)
        train_labels_for_weights = train_labels_2d.reshape(-1)

    if eval_valid_cell_mask is not None:
        flat_indices = torch.nonzero(eval_valid_cell_mask.reshape(-1), as_tuple=True)[0]
        eval_dataset: Dataset = IndexedGridTileCompactDataset(
            eval_base_act, eval_positions, eval_labels_2d, flat_indices
        )
    else:
        eval_dataset = GridTileCompactDataset(eval_base_act, eval_positions, eval_labels_2d)

    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "num_classes": num_classes,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "scaler_mean": scaler_mean,
        "scaler_std": scaler_std,
        "debug_trajectory_activations": debug_trajectory_activations,
        "debug_trajectory_positions": debug_trajectory_positions,
        "debug_trajectory_labels": debug_trajectory_labels,
        "train_labels_for_weights": train_labels_for_weights,
        "input_dim": activation_dim + 2,
    }


def _print_final_results(
    final_results: dict,
    idx_to_label: dict[int, int],
    debug_trajectory_activations: torch.Tensor | None,
    debug_trajectory_positions: torch.Tensor | None,
    debug_trajectory_labels: torch.Tensor | None,
    model: nn.Module,
    torch_device: torch.device,
) -> None:
    """Print final evaluation results including per-class metrics and debug grid."""
    print(f"Accuracy: {final_results['accuracy']:.4f}")
    print(f"Balanced Accuracy: {final_results['balanced_accuracy']:.4f}")
    print(f"Loss: {final_results['loss']:.4f}")
    print(f"Number of samples: {final_results['num_samples']}")

    # Print per-class metrics table
    print("\nPer-class metrics:")
    print("-" * 87)
    print(
        f"{'Class':<10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'GT Support':>12} {'Predicted':>10}"
    )
    print("-" * 87)
    for class_idx in sorted(final_results["per_class_metrics"].keys()):
        metrics = final_results["per_class_metrics"][class_idx]
        original_label = idx_to_label[class_idx]
        symbol = CELL_ID_TO_SYMBOL[original_label]
        print(
            f"{symbol:<10} {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
            f"{metrics['f1']:>10.4f} {metrics['gt_support']:>12} {metrics['predicted']:>10}"
        )
    print("-" * 87)

    # Print debug grid comparison using the first trajectory (saved before shuffling)
    if debug_trajectory_activations is not None and debug_trajectory_positions is not None:
        debug_x = debug_trajectory_activations.float().to(torch_device)
        with torch.no_grad():
            debug_outputs = model(debug_x)
            _, debug_preds = torch.max(debug_outputs, 1)
        _print_debug_grids(
            positions=debug_trajectory_positions,
            true_labels=debug_trajectory_labels,
            pred_labels=debug_preds.cpu(),
            idx_to_label=idx_to_label,
        )
    else:
        print("\n(Debug grid not available: num_cells_per_trajectory not found in data)")


def train_cognitive_map_probe(
    train_data_path: str,
    model_type: ModelType = "lr",
    eval_data_path: str | None = None,
    output_path: str | None = None,
    num_epochs: int = 100,
    learning_rate: float = 0.01,
    batch_size: int = 256,
    weight_decay: float = 1e-4,
    hidden_dims: str = "512,256",
    dropout: float = 0.1,
    eval_split: float = 0.2,
    eval_every: int = 1,
    data_on_gpu: bool = False,
    subset: float = 1.0,
    class_weight: ClassWeight = None,
    balance_classes: bool = False,
    normalize: bool = False,
    device: str | None = None,
    seed: int = 42,
    verbose: bool = True,
    per_class_max_count: int | None = None,
) -> CognitiveMapProbe:
    """Train a cognitive map probing classifier on prepared activations.

    Takes a .pt file produced by prepare_activations_for_probing with
    probe_type=grid_tile and trains either a logistic regression or MLP
    classifier to predict grid tile identity from activations.

    Args:
        train_data_path: Path to .pt file containing training data (from
            prepare_activations_for_probing with probe_type=grid_tile)
        model_type: Type of classifier to train:
            - "lr": Logistic regression (linear classifier)
            - "mlp": Multi-layer perceptron
        eval_data_path: Optional path to separate .pt file for evaluation.
            If not provided, uses eval_split from training data.
        output_path: Path to save the trained model. If not provided,
            saves to the same directory as train_data_path.
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        weight_decay: L2 regularization weight decay
        hidden_dims: Comma-separated hidden layer dimensions for MLP
            (e.g., "512,256" for two hidden layers)
        dropout: Dropout rate for MLP (ignored for lr)
        eval_split: Fraction of training data to use for validation
            (only used if eval_data_path is not provided)
        eval_every: Run evaluation every N epochs (default 1). The final
            epoch is always evaluated. Larger values speed up training by
            skipping the eval pass on intermediate epochs.
        data_on_gpu: If True, load the entire training tensor onto the GPU
            once (eliminating per-batch PCIe transfer and moving the shuffle
            gather onto the GPU). The eval set is still streamed from CPU.
            Requires a CUDA device with enough free memory for the float16
            train tensor plus headroom; raises a clear error otherwise.
            Pin the GPU with CUDA_VISIBLE_DEVICES on a shared node.
        subset: Fraction of full trajectories to use (0.0 to 1.0, default 1.0).
            Applied before train/eval split. If the data has trajectory info,
            subsets by complete trajectories; otherwise subsets by samples.
        class_weight: How to weight classes in the loss function:
            - None: No class weighting (default)
            - "balanced": Weight inversely proportional to class frequency
        balance_classes: If True, upsample minority classes to match the
            majority class count before training
        normalize: If True, normalize activations using mean and std computed
            from training data. The normalization parameters are saved with
            the probe and applied automatically during inference.
        device: Device to use for training (e.g., "cuda", "cpu").
            If not provided, uses CUDA if available.
        seed: Random seed for reproducibility
        verbose: Print training progress

    Returns:
        Trained CognitiveMapProbe instance (also saved to output_path)
    """
    # Set random seed
    set_seed(seed)

    # Determine device
    torch_device = resolve_device(device)

    print(f"Using device: {torch_device}")

    # Load training data
    train_path = Path(train_data_path)
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")

    if not 0.0 < subset <= 1.0:
        raise ValueError(f"subset must be in (0.0, 1.0], got {subset}")

    load_result = _load_and_preprocess_train_data(train_path, verbose)

    if load_result["format"] == 3:
        bundle = _prepare_train_eval_v3(
            load_result=load_result,
            eval_data_path=eval_data_path,
            eval_split=eval_split,
            subset=subset,
            balance_classes=balance_classes,
            normalize=normalize,
            per_class_max_count=per_class_max_count,
            seed=seed,
            verbose=verbose,
        )
    else:
        bundle = _prepare_train_eval_v1(
            load_result=load_result,
            eval_data_path=eval_data_path,
            eval_split=eval_split,
            subset=subset,
            balance_classes=balance_classes,
            normalize=normalize,
            per_class_max_count=per_class_max_count,
            seed=seed,
            verbose=verbose,
        )

    train_dataset = bundle["train_dataset"]
    eval_dataset = bundle["eval_dataset"]
    num_classes = bundle["num_classes"]
    label_to_idx = bundle["label_to_idx"]
    idx_to_label = bundle["idx_to_label"]
    scaler_mean = bundle["scaler_mean"]
    scaler_std = bundle["scaler_std"]
    debug_trajectory_activations = bundle["debug_trajectory_activations"]
    debug_trajectory_positions = bundle["debug_trajectory_positions"]
    debug_trajectory_labels = bundle["debug_trajectory_labels"]
    train_labels = bundle["train_labels_for_weights"]
    input_dim = bundle["input_dim"]

    # Parse hidden dimensions
    hidden_dims_list = [int(d.strip()) for d in hidden_dims.split(",") if d.strip()]

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    model = _create_model(
        model_type=model_type,
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims_list,
        dropout=dropout,
    )
    model = model.to(torch_device)

    if verbose:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model type: {model_type}")
        print(f"Input dimension: {input_dim}")
        print(f"Number of classes: {num_classes}")
        print(f"Hidden dimensions: {hidden_dims_list}")
        print(f"Number of parameters: {num_params:,}")

    # Loss and optimizer
    if class_weight == "balanced":
        print("Using balanced class weights in loss function")
        weights = _compute_class_weights(train_labels, num_classes, torch_device)
        print(f"  Class weights: {weights.tolist()}")
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training loop
    best_eval_accuracy = 0.0
    best_balanced_accuracy = 0.0
    # best_model_state = None

    print(f"\nTraining for {num_epochs} epochs...")

    if eval_every < 1:
        raise ValueError(f"eval_every must be >= 1, got {eval_every}")

    epoch_iterator = tqdm(range(num_epochs), desc="Training", disable=not verbose)

    eval_results = None
    for epoch in epoch_iterator:
        train_loss = _train_one_epoch(
            model, train_activations, train_labels, criterion, optimizer, torch_device, batch_size
        )

        # Evaluate every eval_every epochs; always evaluate the final epoch.
        is_last_epoch = epoch == num_epochs - 1
        if (epoch + 1) % eval_every == 0 or is_last_epoch:
            eval_results = _evaluate(
                model, eval_activations, eval_labels, criterion, torch_device, num_classes, batch_size
            )
            best_eval_accuracy = max(best_eval_accuracy, eval_results["accuracy"])
            best_balanced_accuracy = max(best_balanced_accuracy, eval_results["balanced_accuracy"])

        # Update progress bar
        postfix = {"loss": f"{train_loss:.4f}"}
        if eval_results is not None:
            postfix.update(
                {
                    "eval_acc": f"{eval_results['accuracy']:.4f}",
                    "bal_acc": f"{eval_results['balanced_accuracy']:.4f}",
                    "best_acc": f"{best_eval_accuracy:.4f}",
                    "best_bal": f"{best_balanced_accuracy:.4f}",
                }
            )
        epoch_iterator.set_postfix(postfix)

    # Restore best model
    # if best_model_state is not None:
    # model.load_state_dict(best_model_state)

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION (best model)")
    print("=" * 60)

    final_results = _evaluate(
        model, eval_activations, eval_labels, criterion, torch_device, num_classes, batch_size
    )

    if verbose:
        # Normalize debug activations if normalization is enabled (model expects normalized input)
        debug_activations_for_display = debug_trajectory_activations
        if normalize and debug_trajectory_activations is not None:
            debug_activations_for_display = (debug_trajectory_activations - scaler_mean) / scaler_std

        _print_final_results(
            final_results=final_results,
            idx_to_label=idx_to_label,
            debug_trajectory_activations=debug_activations_for_display,
            debug_trajectory_positions=debug_trajectory_positions,
            debug_trajectory_labels=debug_trajectory_labels,
            model=model,
            torch_device=torch_device,
        )

    # Convert per_class_metrics keys back to original labels for interpretability
    per_class_metrics_original = {
        idx_to_label[idx]: metrics for idx, metrics in final_results["per_class_metrics"].items()
    }

    # Create the CognitiveMapProbe instance
    probe = CognitiveMapProbe(
        model=model,
        model_type=model_type,
        input_dim=input_dim,
        label_to_idx=label_to_idx,
        idx_to_label=idx_to_label,
        hidden_dims=hidden_dims_list if model_type == "mlp" else None,
        dropout=dropout if model_type == "mlp" else None,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
        config={
            "train_data_path": str(train_path),
            "eval_data_path": str(eval_data_path) if eval_data_path else None,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "eval_split": eval_split,
            "eval_every": eval_every,
            "data_on_gpu": data_on_gpu,
            "subset": subset,
            "class_weight": class_weight,
            "balance_classes": balance_classes,
            "normalize": normalize,
            "seed": seed,
        },
        results={
            "best_eval_accuracy": best_eval_accuracy,
            "best_balanced_accuracy": best_balanced_accuracy,
            "final_accuracy": final_results["accuracy"],
            "final_balanced_accuracy": final_results["balanced_accuracy"],
            "final_loss": final_results["loss"],
            "per_class_metrics": per_class_metrics_original,
        },
        device=torch_device,
    )

    # Save the probe
    if output_path is None:
        output_dir = train_path.parent
        output_filename = f"cognitive_map_probe_{model_type}.pt"
        final_output_path = output_dir / output_filename
    else:
        final_output_path = Path(output_path)

    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    probe.save(final_output_path)

    print(f"\nModel saved to {final_output_path}")
    print(f"Best evaluation accuracy: {best_eval_accuracy:.4f}")
    print(f"Best balanced accuracy: {best_balanced_accuracy:.4f}")

    return probe
