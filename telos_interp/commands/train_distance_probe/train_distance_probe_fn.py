"""Train distance regression probes on prepared activations."""

from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from telos_interp.probe_models import (
    ModelType,
)
from telos_interp.probe_models import (
    create_regression_model as _create_model,
)
from telos_interp.training import (
    compute_normalization_params,
    normalize_activations,
    resolve_device,
    set_seed,
)
from telos_interp.training import (
    train_epoch as _train_epoch,
)


class DistanceProbe:
    """A trained distance regression probe with built-in normalization and prediction.

    This class encapsulates a trained probe model along with normalization
    parameters, providing a simple interface for inference.

    Example:
        # Training
        probe = train_distance_probe("train.pt", normalize=True)
        probe.save("probe.pt")

        # Inference
        probe = DistanceProbe.load("probe.pt")
        predictions = probe.predict(activations)  # (N,) predicted distances
    """

    def __init__(
        self,
        model: nn.Module,
        model_type: ModelType,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float | None = None,
        scaler_mean: torch.Tensor | None = None,
        scaler_std: torch.Tensor | None = None,
        label_mean: float | None = None,
        label_std: float | None = None,
        config: dict | None = None,
        results: dict | None = None,
        device: torch.device | str | None = None,
    ):
        """Initialize a DistanceProbe.

        Args:
            model: The trained nn.Module (LinearRegressionProbe or MLPRegressionProbe)
            model_type: Type of model ("lr" or "mlp")
            input_dim: Input dimension of the model
            hidden_dims: Hidden layer dimensions (for MLP)
            dropout: Dropout rate (for MLP)
            scaler_mean: Mean for input normalization (None if no normalization)
            scaler_std: Std for input normalization (None if no normalization)
            label_mean: Mean for label normalization (None if no normalization)
            label_std: Std for label normalization (None if no normalization)
            config: Training configuration (for provenance)
            results: Training results (metrics, etc.)
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

        self.config = config or {}
        self.results = results or {}

        # Store normalization parameters
        if scaler_mean is not None:
            self.scaler_mean = scaler_mean.to(self.device)
            self.scaler_std = scaler_std.to(self.device)
        else:
            self.scaler_mean = None
            self.scaler_std = None

        # Store label normalization parameters
        self.label_mean = label_mean
        self.label_std = label_std

    @property
    def normalized(self) -> bool:
        """Whether this probe applies input normalization."""
        return self.scaler_mean is not None

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply normalization if enabled."""
        if self.scaler_mean is not None:
            return (x - self.scaler_mean) / self.scaler_std
        return x

    def _denormalize_labels(self, y: torch.Tensor) -> torch.Tensor:
        """Denormalize predictions if label normalization was used."""
        if self.label_mean is not None and self.label_std is not None:
            return y * self.label_std + self.label_mean
        return y

    @torch.no_grad()
    def predict(self, activations: torch.Tensor) -> torch.Tensor:
        """Get predicted distances for activations.

        Args:
            activations: Input tensor of shape (N, input_dim)

        Returns:
            Tensor of shape (N,) with predicted distances
        """
        self.model.eval()
        x = activations.float().to(self.device)
        x = self._normalize(x)
        predictions = self.model(x)
        return self._denormalize_labels(predictions)

    def save(self, path: str | Path) -> None:
        """Save the probe to a file."""
        save_data = {
            "model_state_dict": self.model.state_dict(),
            "model_type": self.model_type,
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "scaler_mean": self.scaler_mean.cpu() if self.scaler_mean is not None else None,
            "scaler_std": self.scaler_std.cpu() if self.scaler_std is not None else None,
            "label_mean": self.label_mean,
            "label_std": self.label_std,
            "config": self.config,
            "results": self.results,
        }
        torch.save(save_data, path)

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "DistanceProbe":
        """Load a probe from a file."""
        device_str = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        torch_device = torch.device(device_str)

        data = torch.load(path, map_location=torch_device, weights_only=False)

        model = _create_model(
            model_type=data["model_type"],
            input_dim=data["input_dim"],
            hidden_dims=data["hidden_dims"] or [],
            dropout=data["dropout"] or 0.0,
        )
        model.load_state_dict(data["model_state_dict"])

        return cls(
            model=model,
            model_type=data["model_type"],
            input_dim=data["input_dim"],
            hidden_dims=data["hidden_dims"],
            dropout=data["dropout"],
            scaler_mean=data.get("scaler_mean"),
            scaler_std=data.get("scaler_std"),
            label_mean=data.get("label_mean"),
            label_std=data.get("label_std"),
            config=data.get("config"),
            results=data.get("results"),
            device=torch_device,
        )


def _evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    label_mean: float | None = None,
    label_std: float | None = None,
) -> dict:
    """Evaluate model and return regression metrics."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

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

            all_preds.extend(outputs.cpu().tolist())
            all_labels.extend(batch_y.cpu().tolist())

    preds = torch.tensor(all_preds)
    labels = torch.tensor(all_labels)

    # Denormalize for computing metrics in original scale
    if label_mean is not None and label_std is not None:
        preds_orig = preds * label_std + label_mean
        labels_orig = labels * label_std + label_mean
    else:
        preds_orig = preds
        labels_orig = labels

    # Compute metrics
    mse = ((preds_orig - labels_orig) ** 2).mean().item()
    mae = (preds_orig - labels_orig).abs().mean().item()
    rmse = mse**0.5

    # R² score
    ss_res = ((labels_orig - preds_orig) ** 2).sum().item()
    ss_tot = ((labels_orig - labels_orig.mean()) ** 2).sum().item()
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Compute metrics by distance bucket
    distance_buckets = {}
    unique_distances = labels_orig.unique().tolist()
    for dist in sorted(unique_distances):
        mask = labels_orig == dist
        if mask.sum() > 0:
            bucket_preds = preds_orig[mask]
            bucket_labels = labels_orig[mask]
            bucket_mae = (bucket_preds - bucket_labels).abs().mean().item()
            bucket_mse = ((bucket_preds - bucket_labels) ** 2).mean().item()
            distance_buckets[int(dist)] = {
                "mae": bucket_mae,
                "mse": bucket_mse,
                "rmse": bucket_mse**0.5,
                "count": int(mask.sum().item()),
            }

    return {
        "loss": total_loss / num_batches if num_batches > 0 else 0.0,
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "num_samples": len(preds),
        "predictions": all_preds,
        "labels": all_labels,
        "per_distance_metrics": distance_buckets,
    }


def _load_and_preprocess_train_data(
    train_path: Path,
    verbose: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Load training data and filter NaN values.

    Returns:
        Tuple of (activations, labels, train_data_dict)
    """
    if verbose:
        print(f"Loading training data from {train_path}")

    train_data = torch.load(train_path, map_location="cpu", weights_only=False)

    # Validate probe type
    probe_type = train_data.get("probe_type", train_data.get("config", {}).get("probe_type"))
    if probe_type is not None and probe_type != "distance":
        raise ValueError(
            f"Expected probe_type='distance', got '{probe_type}'. This command only supports distance probe data."
        )

    activations = train_data["activations"]
    labels = train_data["labels"].float()  # Convert to float for regression

    if verbose:
        print(f"Loaded {activations.shape[0]} samples")
        print(f"Activation dimension: {activations.shape[1]}")
        print(f"Label range: [{labels.min().item():.1f}, {labels.max().item():.1f}]")
        print(f"Label mean: {labels.mean().item():.2f}, std: {labels.std().item():.2f}")

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

    return activations, labels, train_data


def _print_final_results(
    final_results: dict,
) -> None:
    """Print final evaluation results."""
    print(f"MSE: {final_results['mse']:.4f}")
    print(f"MAE: {final_results['mae']:.4f}")
    print(f"RMSE: {final_results['rmse']:.4f}")
    print(f"R²: {final_results['r2']:.4f}")
    print(f"Number of samples: {final_results['num_samples']}")

    # Print per-distance metrics table
    if final_results.get("per_distance_metrics"):
        print("\nPer-distance metrics:")
        print("-" * 60)
        print(f"{'Distance':<10} {'MAE':>10} {'RMSE':>10} {'Count':>10}")
        print("-" * 60)
        for dist in sorted(final_results["per_distance_metrics"].keys()):
            metrics = final_results["per_distance_metrics"][dist]
            print(f"{dist:<10} {metrics['mae']:>10.4f} {metrics['rmse']:>10.4f} {metrics['count']:>10}")
        print("-" * 60)


def train_distance_probe(
    train_data_path: str,
    model_type: ModelType = "lr",
    eval_data_path: str | None = None,
    output_path: str | None = None,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    batch_size: int = 256,
    weight_decay: float = 1e-4,
    hidden_dims: str = "512,256",
    dropout: float = 0.1,
    eval_split: float = 0.2,
    subset: float = 1.0,
    normalize: bool = False,
    normalize_labels: bool = False,
    device: str | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> DistanceProbe:
    """Train a distance regression probe on prepared activations.

    Takes a .pt file produced by prepare_activations_for_probing with
    probe_type=distance and trains either a linear regression or MLP
    regressor to predict A* distance from activations.

    Args:
        train_data_path: Path to .pt file containing training data (from
            prepare_activations_for_probing with probe_type=distance)
        model_type: Type of regressor to train:
            - "lr": Linear regression
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
        subset: Fraction of samples to use (0.0 to 1.0, default 1.0)
        normalize: If True, normalize activations using mean and std computed
            from training data. The normalization parameters are saved with
            the probe and applied automatically during inference.
        normalize_labels: If True, normalize labels to zero mean and unit
            variance during training. Predictions are denormalized automatically.
        device: Device to use for training (e.g., "cuda", "cpu").
            If not provided, uses CUDA if available.
        seed: Random seed for reproducibility
        verbose: Print training progress

    Returns:
        Trained DistanceProbe instance (also saved to output_path)
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

    activations, labels, train_data = _load_and_preprocess_train_data(train_path, verbose)

    # Validate and apply subset parameter
    if not 0.0 < subset <= 1.0:
        raise ValueError(f"subset must be in (0.0, 1.0], got {subset}")

    if subset < 1.0:
        num_samples_to_keep = max(1, int(len(activations) * subset))
        perm = torch.randperm(len(activations))[:num_samples_to_keep]
        activations = activations[perm]
        labels = labels[perm]
        print(f"Subset: keeping {num_samples_to_keep} samples ({subset * 100:.1f}%)")

    # Parse hidden dimensions
    hidden_dims_list = [int(d.strip()) for d in hidden_dims.split(",") if d.strip()]

    # Split data if no separate eval file provided
    if eval_data_path is None:
        num_samples = activations.shape[0]
        indices = torch.randperm(num_samples)
        split_idx = int(num_samples * (1 - eval_split))
        train_indices = indices[:split_idx]
        eval_indices = indices[split_idx:]

        train_activations = activations[train_indices]
        train_labels = labels[train_indices]
        eval_activations = activations[eval_indices]
        eval_labels = labels[eval_indices]

        print(f"Split: {train_activations.shape[0]} train, {eval_activations.shape[0]} eval")
    else:
        train_activations = activations
        train_labels = labels

        # Load evaluation data
        eval_path = Path(eval_data_path)
        if not eval_path.exists():
            raise FileNotFoundError(f"Evaluation data not found: {eval_path}")

        eval_data = torch.load(eval_path, map_location="cpu", weights_only=False)
        eval_activations = eval_data["activations"]
        eval_labels = eval_data["labels"].float()

        if verbose:
            print(f"Loaded {eval_activations.shape[0]} evaluation samples")

    # Compute normalization parameters from training data
    scaler_mean = None
    scaler_std = None
    if normalize:
        scaler_mean, scaler_std = compute_normalization_params(train_activations)
        train_activations = normalize_activations(train_activations, scaler_mean, scaler_std)
        eval_activations = normalize_activations(eval_activations, scaler_mean, scaler_std)
        print(
            f"Activation normalization enabled: computed mean/std from {train_activations.shape[0]} training samples"
        )

    # Compute label normalization parameters
    label_mean = None
    label_std = None
    if normalize_labels:
        label_mean = train_labels.mean().item()
        label_std = train_labels.std().item()
        if label_std < 1e-8:
            label_std = 1.0
        train_labels = (train_labels - label_mean) / label_std
        eval_labels = (eval_labels - label_mean) / label_std
        print(f"Label normalization enabled: mean={label_mean:.2f}, std={label_std:.2f}")

    # Create data loaders
    train_dataset = TensorDataset(train_activations.float(), train_labels.float())
    eval_dataset = TensorDataset(eval_activations.float(), eval_labels.float())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    input_dim = train_activations.shape[1]
    model = _create_model(
        model_type=model_type,
        input_dim=input_dim,
        hidden_dims=hidden_dims_list,
        dropout=dropout,
    )
    model = model.to(torch_device)

    if verbose:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model type: {model_type}")
        print(f"Input dimension: {input_dim}")
        print(f"Hidden dimensions: {hidden_dims_list}")
        print(f"Number of parameters: {num_params:,}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training loop
    best_eval_mae = float("inf")
    best_eval_r2 = float("-inf")

    print(f"\nTraining for {num_epochs} epochs...")

    epoch_iterator = tqdm(range(num_epochs), desc="Training", disable=not verbose)

    for _ in epoch_iterator:
        train_loss = _train_epoch(model, train_loader, criterion, optimizer, torch_device)

        # Evaluate
        eval_results = _evaluate(model, eval_loader, criterion, torch_device, label_mean, label_std)

        # Track best model
        best_eval_mae = min(best_eval_mae, eval_results["mae"])
        best_eval_r2 = max(best_eval_r2, eval_results["r2"])

        # Update progress bar
        epoch_iterator.set_postfix(
            {
                "loss": f"{train_loss:.4f}",
                "mae": f"{eval_results['mae']:.4f}",
                "r2": f"{eval_results['r2']:.4f}",
                "best_mae": f"{best_eval_mae:.4f}",
                "best_r2": f"{best_eval_r2:.4f}",
            }
        )

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    final_results = _evaluate(model, eval_loader, criterion, torch_device, label_mean, label_std)

    if verbose:
        _print_final_results(final_results)

    # Create the DistanceProbe instance
    probe = DistanceProbe(
        model=model,
        model_type=model_type,
        input_dim=input_dim,
        hidden_dims=hidden_dims_list if model_type == "mlp" else None,
        dropout=dropout if model_type == "mlp" else None,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
        label_mean=label_mean,
        label_std=label_std,
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
            "normalize": normalize,
            "normalize_labels": normalize_labels,
            "seed": seed,
        },
        results={
            "best_eval_mae": best_eval_mae,
            "best_eval_r2": best_eval_r2,
            "final_mse": final_results["mse"],
            "final_mae": final_results["mae"],
            "final_rmse": final_results["rmse"],
            "final_r2": final_results["r2"],
            "per_distance_metrics": final_results["per_distance_metrics"],
        },
        device=torch_device,
    )

    # Save the probe
    if output_path is None:
        output_dir = train_path.parent
        output_filename = f"distance_regression_probe_{model_type}.pt"
        final_output_path = output_dir / output_filename
    else:
        final_output_path = Path(output_path)

    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    probe.save(final_output_path)

    print(f"\nModel saved to {final_output_path}")
    print(f"Best evaluation MAE: {best_eval_mae:.4f}")
    print(f"Best evaluation R²: {best_eval_r2:.4f}")

    return probe
