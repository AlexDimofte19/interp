"""Train cognitive map probing classifiers on prepared activations."""

from pathlib import Path
from typing import Literal

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from telos_interp.commands.prepare_activations_for_probing import CELL_ID_TO_SYMBOL

# Model type literal for type hints
ModelType = Literal["lr", "mlp"]
ClassWeight = Literal["balanced"] | None


class LogisticRegressionProbe(nn.Module):
    """Logistic regression classifier for probing."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLPProbe(nn.Module):
    """Multi-layer perceptron classifier for probing."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list[int],
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def _create_model(
    model_type: ModelType,
    input_dim: int,
    num_classes: int,
    hidden_dims: list[int],
    dropout: float,
) -> nn.Module:
    """Create the appropriate model based on model_type."""
    if model_type == "lr":
        return LogisticRegressionProbe(input_dim, num_classes)
    elif model_type == "mlp":
        return MLPProbe(input_dim, num_classes, hidden_dims, dropout)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


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


def _train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_x_raw, batch_y_raw in dataloader:
        batch_x = batch_x_raw.to(device)
        batch_y = batch_y_raw.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


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


def _evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> dict:
    """Evaluate model and return metrics."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    # Per-class metrics: true positives, ground truth count, predicted count
    class_tp = torch.zeros(num_classes)
    class_gt_support = torch.zeros(num_classes)
    class_pred_count = torch.zeros(num_classes)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x_raw, batch_y_raw in dataloader:
            batch_x = batch_x_raw.to(device)
            batch_y = batch_y_raw.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item()
            num_batches += 1

            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(batch_y.cpu().tolist())

            # Per-class metrics
            for i in range(num_classes):
                gt_mask = batch_y == i
                pred_mask = predicted == i
                class_gt_support[i] += gt_mask.sum().item()
                class_pred_count[i] += pred_mask.sum().item()
                class_tp[i] += (gt_mask & pred_mask).sum().item()

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

    return {
        "loss": total_loss / num_batches if num_batches > 0 else 0.0,
        "accuracy": correct / total if total > 0 else 0.0,
        "per_class_metrics": per_class_metrics,
        "num_samples": total,
        "predictions": all_preds,
        "labels": all_labels,
    }


def _balance_classes_by_upsampling(
    activations: torch.Tensor,
    labels: torch.Tensor,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Balance classes by upsampling minority classes to match the majority class.

    Args:
        activations: Input activations tensor
        labels: Label tensor
        seed: Random seed for reproducibility

    Returns:
        Tuple of (balanced_activations, balanced_labels)
    """
    torch.manual_seed(seed)

    unique_classes = torch.unique(labels)
    class_counts = {c.item(): (labels == c).sum().item() for c in unique_classes}
    max_count = max(class_counts.values())

    balanced_indices = []

    for class_id in unique_classes:
        class_indices = torch.where(labels == class_id)[0]
        current_count = len(class_indices)

        if current_count < max_count:
            # Upsample: repeat and sample additional indices
            num_additional = max_count - current_count
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
    unique_classes, class_counts = torch.unique(labels, return_counts=True)
    n_samples = len(labels)

    for class_id, count in zip(unique_classes, class_counts, strict=False):
        class_weights[class_id] = n_samples / (num_classes * count.float())

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


def _load_and_preprocess_train_data(
    train_path: Path,
    verbose: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict, int | None]:
    """Load training data and filter NaN values.

    Returns:
        Tuple of (activations, labels, train_data_dict, num_cells_per_trajectory)
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
    return activations, labels, train_data, num_cells_per_trajectory


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


def _load_separate_eval_data(
    eval_path: Path,
    label_to_idx: dict[int, int],
    verbose: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load and process separate evaluation data.

    Returns:
        Tuple of (eval_activations, eval_labels)
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
    subset: float = 1.0,
    class_weight: ClassWeight = None,
    balance_classes: bool = False,
    normalize: bool = False,
    device: str | None = None,
    seed: int = 42,
    verbose: bool = True,
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
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Determine device
    device_str = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    torch_device = torch.device(device_str)

    print(f"Using device: {torch_device}")

    # Load training data
    train_path = Path(train_data_path)
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")

    activations, labels, train_data, num_cells_per_trajectory = _load_and_preprocess_train_data(train_path, verbose)

    # Validate and apply subset parameter
    if not 0.0 < subset <= 1.0:
        raise ValueError(f"subset must be in (0.0, 1.0], got {subset}")

    # If subset is 1.0, no subsetting is applied. In any case, shuffle the trajectories.
    if num_cells_per_trajectory is not None and num_cells_per_trajectory > 0:
        # Subset by complete trajectories, randomly selected (not just first N)
        num_trajectories = len(activations) // num_cells_per_trajectory
        num_trajectories_to_keep = max(1, int(num_trajectories * subset))

        # Shuffle trajectory indices and select subset
        trajectory_perm = torch.randperm(num_trajectories)
        selected_trajectories = trajectory_perm[:num_trajectories_to_keep]

        # Build sample indices and reorder data by selected trajectories
        selected_indices = _build_indices_for_trajectories(selected_trajectories, num_cells_per_trajectory)
        activations = activations[selected_indices]
        labels = labels[selected_indices]

        print(f"Subset: keeping {num_trajectories_to_keep}/{num_trajectories} trajectories ({subset * 100:.1f}%)")
        print(f"  Remaining samples: {len(activations)}")
    else:
        # Fallback: shuffle then subset by samples if no trajectory info
        perm = torch.randperm(len(activations))
        num_samples_to_keep = max(1, int(len(activations) * subset))
        activations = activations[perm[:num_samples_to_keep]]
        labels = labels[perm[:num_samples_to_keep]]
        print(f"Subset: keeping {num_samples_to_keep} samples ({subset * 100:.1f}%)")

    # Remap labels to only include classes present in the data
    remapped_labels, num_classes, label_to_idx, idx_to_label = _remap_labels(labels, verbose)

    # Save first trajectory's data for debug visualization
    # Data is already shuffled at trajectory level, so first trajectory is random
    debug_trajectory_activations = None
    debug_trajectory_positions = None
    debug_trajectory_labels = None
    if num_cells_per_trajectory is not None and num_cells_per_trajectory <= len(activations):
        debug_idx_start = 0
        debug_idx_end = num_cells_per_trajectory
        debug_trajectory_activations = activations[debug_idx_start:debug_idx_end].clone()
        # Save positions separately (last 2 columns) before any normalization
        debug_trajectory_positions = activations[debug_idx_start:debug_idx_end, -2:].clone()
        debug_trajectory_labels = remapped_labels[debug_idx_start:debug_idx_end].clone()

    # Parse hidden dimensions
    hidden_dims_list = [int(d.strip()) for d in hidden_dims.split(",") if d.strip()]

    # Split data if no separate eval file provided
    if eval_data_path is None:
        num_samples = activations.shape[0]

        if num_cells_per_trajectory is not None and num_cells_per_trajectory > 0:
            # Split by complete trajectories - no trajectory appears in both sets
            num_trajectories = num_samples // num_cells_per_trajectory
            trajectory_perm = torch.randperm(num_trajectories)
            num_train_trajectories = int(num_trajectories * (1 - eval_split))

            train_trajectory_ids = trajectory_perm[:num_train_trajectories]
            eval_trajectory_ids = trajectory_perm[num_train_trajectories:]

            train_indices = _build_indices_for_trajectories(train_trajectory_ids, num_cells_per_trajectory)
            eval_indices = _build_indices_for_trajectories(eval_trajectory_ids, num_cells_per_trajectory)

            print(f"Split by trajectories: {len(train_trajectory_ids)} train, {len(eval_trajectory_ids)} eval")
        else:
            # Fallback: shuffle samples if no trajectory info
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

        # Load evaluation data
        eval_path = Path(eval_data_path)
        if not eval_path.exists():
            raise FileNotFoundError(f"Evaluation data not found: {eval_path}")

        eval_activations, eval_labels = _load_separate_eval_data(eval_path, label_to_idx, verbose)

    # Apply class balancing by upsampling if requested
    if balance_classes:
        print("Balancing classes by upsampling...")
        unique, counts = torch.unique(train_labels, return_counts=True)
        print(f"  Before: {dict(zip(unique.tolist(), counts.tolist(), strict=False))}")

        train_activations, train_labels = _balance_classes_by_upsampling(train_activations, train_labels, seed=seed)

        unique, counts = torch.unique(train_labels, return_counts=True)
        print(f"  After: {dict(zip(unique.tolist(), counts.tolist(), strict=False))}")
        print(f"  New training set size: {train_activations.shape[0]}")

    # Compute normalization parameters from training data (before creating loaders)
    scaler_mean = None
    scaler_std = None
    if normalize:
        scaler_mean = train_activations.mean(dim=0)
        scaler_std = train_activations.std(dim=0)
        # Avoid division by zero
        scaler_std = torch.where(scaler_std > 1e-8, scaler_std, torch.ones_like(scaler_std))
        # Normalize both train and eval using training statistics
        train_activations = (train_activations - scaler_mean) / scaler_std
        eval_activations = (eval_activations - scaler_mean) / scaler_std
        print(f"Normalization enabled: computed mean/std from {train_activations.shape[0]} training samples")

    # Create data loaders
    train_dataset = TensorDataset(train_activations.float(), train_labels.long())
    eval_dataset = TensorDataset(eval_activations.float(), eval_labels.long())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    input_dim = train_activations.shape[1]
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
    # best_model_state = None

    print(f"\nTraining for {num_epochs} epochs...")

    epoch_iterator = tqdm(range(num_epochs), desc="Training", disable=not verbose)

    for _ in epoch_iterator:
        train_loss = _train_epoch(model, train_loader, criterion, optimizer, torch_device)

        # Evaluate
        eval_results = _evaluate(model, eval_loader, criterion, torch_device, num_classes)

        # Track best model
        best_eval_accuracy = max(best_eval_accuracy, eval_results["accuracy"])
        # best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Update progress bar
        epoch_iterator.set_postfix(
            {
                "train_loss": f"{train_loss:.4f}",
                "eval_acc": f"{eval_results['accuracy']:.4f}",
                "best_acc": f"{best_eval_accuracy:.4f}",
            }
        )

    # Restore best model
    # if best_model_state is not None:
    # model.load_state_dict(best_model_state)

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION (best model)")
    print("=" * 60)

    final_results = _evaluate(model, eval_loader, criterion, torch_device, num_classes)

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
            "subset": subset,
            "class_weight": class_weight,
            "balance_classes": balance_classes,
            "normalize": normalize,
            "seed": seed,
        },
        results={
            "best_eval_accuracy": best_eval_accuracy,
            "final_accuracy": final_results["accuracy"],
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

    return probe
