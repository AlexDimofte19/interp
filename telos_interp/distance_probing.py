"""Distance regression probes for predicting continuous values from activations."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from torch import nn, optim
from tqdm import tqdm

ArrayLike = np.ndarray | torch.Tensor


def _to_torch(x: ArrayLike, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert array-like to torch tensor."""
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(dtype)
    raise TypeError(f"Unsupported type {type(x)}; expected torch.Tensor or np.ndarray.")


def _build_model(
    n_features: int,
    n_outputs: int,
    use_mlp: bool,
    mlp_hidden_size: int,
    fit_intercept: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> nn.Module:
    """Build linear or MLP model for regression."""
    if use_mlp:
        model = nn.Sequential(
            nn.Linear(n_features, mlp_hidden_size, bias=fit_intercept),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, n_outputs, bias=fit_intercept),
        )
    else:
        model = nn.Linear(n_features, n_outputs, bias=fit_intercept)

    return model.to(dtype=dtype, device=device)


class DistanceProbe:
    """Regression probe for predicting continuous values (e.g., distances) from activations.

    Supports both linear regression and MLP architectures, trained with MSE loss.
    """

    def __init__(
        self,
        reg_coeff: float = 1e-4,
        normalize: bool = False,
        fit_intercept: bool = True,
        dtype: torch.dtype = torch.float32,
        verbose: int = 0,
        device: str | torch.device | None = None,
        learning_rate: float = 1e-3,
        num_epochs: int = 100,
        batch_size: int | None = None,
        use_mlp: bool = False,
        mlp_hidden_size: int = 256,
        preload_to_device: bool = True,
    ) -> None:
        """Initialize the distance probe.

        Args:
            reg_coeff: L2 regularization coefficient (weight decay).
            normalize: Whether to normalize input features.
            fit_intercept: Whether to include bias terms.
            dtype: Torch dtype for computations.
            verbose: Verbosity level (0=silent, 1+=print epoch losses).
            device: Device to use ('cuda', 'cpu', or None for auto-detect).
            learning_rate: Learning rate for Adam optimizer.
            num_epochs: Number of training epochs.
            batch_size: Batch size (None for auto: min(1024, n_samples)).
            use_mlp: Use MLP instead of linear regression.
            mlp_hidden_size: Hidden layer size for MLP.
            preload_to_device: If True, load all data to GPU upfront (faster for small datasets).
        """
        self.reg_coeff = float(reg_coeff)
        self.normalize = bool(normalize)
        self.fit_intercept = bool(fit_intercept)
        self.dtype = dtype
        self.verbose = int(verbose)
        self.learning_rate = float(learning_rate)
        self.num_epochs = int(num_epochs)
        self.batch_size = batch_size
        self.use_mlp = bool(use_mlp)
        self.mlp_hidden_size = int(mlp_hidden_size)
        self.preload_to_device = bool(preload_to_device)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model: nn.Module | None = None
        self.scaler_mean: torch.Tensor | None = None
        self.scaler_std: torch.Tensor | None = None

    # ----------------------------- training -------------------------------- #
    def fit(self, X: ArrayLike, y: ArrayLike) -> "DistanceProbe":
        """Fit the probe on activations and continuous targets.

        Args:
            X: Input activations of shape (n_samples, hidden_dim).
            y: Target values of shape (n_samples,) or (n_samples, 1).

        Returns:
            self
        """
        X_tensor = _to_torch(X, self.dtype)
        y_tensor = _to_torch(y, self.dtype)

        if X_tensor.ndim != 2:
            raise ValueError(f"X must be 2D (n_samples, hidden_dim), got shape {X_tensor.shape}")

        # Ensure y is 2D for consistency with model output
        if y_tensor.ndim == 1:
            y_tensor = y_tensor.unsqueeze(-1)

        n_samples, n_features = X_tensor.shape
        n_outputs = y_tensor.shape[1]

        # Normalization
        if self.normalize:
            scaler_mean = X_tensor.mean(dim=0)
            scaler_std = X_tensor.std(dim=0)
            scaler_std = torch.where(scaler_std > 1e-8, scaler_std, torch.ones_like(scaler_std))
            X_tensor = (X_tensor - scaler_mean) / scaler_std
            self.scaler_mean = scaler_mean.to(self.device)
            self.scaler_std = scaler_std.to(self.device)
        else:
            self.scaler_mean = None
            self.scaler_std = None

        # Build model
        self.model = _build_model(
            n_features=n_features,
            n_outputs=n_outputs,
            use_mlp=self.use_mlp,
            mlp_hidden_size=self.mlp_hidden_size,
            fit_intercept=self.fit_intercept,
            dtype=self.dtype,
            device=self.device,
        )

        # Preload data to device if requested
        if self.preload_to_device:
            X_tensor = X_tensor.to(self.device)
            y_tensor = y_tensor.to(self.device)

        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.reg_coeff,
        )

        batch_size = self.batch_size if self.batch_size is not None else min(1024, n_samples)
        batch_size = max(1, batch_size)

        # Training loop
        self.model.train()
        for epoch in range(self.num_epochs):
            permutation = torch.randperm(n_samples)
            epoch_loss = 0.0
            n_batches = 0

            iterator = range(0, n_samples, batch_size)
            if self.verbose == 0:
                iterator = tqdm(iterator, desc=f"Epoch {epoch + 1}/{self.num_epochs}", leave=False)

            for start_idx in iterator:
                idx = permutation[start_idx : start_idx + batch_size]

                if self.preload_to_device:
                    batch_X = X_tensor[idx]
                    batch_y = y_tensor[idx]
                else:
                    batch_X = X_tensor[idx].to(self.device, non_blocking=True)
                    batch_y = y_tensor[idx].to(self.device, non_blocking=True)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            if self.verbose > 0:
                avg_loss = epoch_loss / n_batches
                print(f"  Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_loss:.6f}")

        self.model.eval()
        return self

    # ------------------------------ prediction ------------------------------ #
    @torch.no_grad()
    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict continuous values for input activations.

        Args:
            X: Input activations of shape (n_samples, hidden_dim) or (hidden_dim,).

        Returns:
            Predictions of shape (n_samples,) or scalar if input was 1D.
        """
        if self.model is None:
            raise RuntimeError("Probe not trained – call .fit() first.")

        X_tensor = _to_torch(X, self.dtype).to(self.device)
        squeeze_output = X_tensor.ndim == 1

        if X_tensor.ndim == 1:
            X_tensor = X_tensor.unsqueeze(0)

        if X_tensor.ndim != 2:
            raise ValueError(f"X must be 1D or 2D, got shape {X_tensor.shape}")

        # Apply normalization if used during training
        if self.scaler_mean is not None and self.scaler_std is not None:
            X_tensor = (X_tensor - self.scaler_mean) / self.scaler_std

        predictions = self.model(X_tensor).squeeze(-1)

        if squeeze_output:
            return predictions.cpu().numpy().item()
        return predictions.cpu().numpy()

    # ------------------------------ evaluation ------------------------------ #
    def evaluate(self, X: ArrayLike, y: ArrayLike) -> dict[str, float]:
        """Evaluate the probe on test data.

        Args:
            X: Test activations of shape (n_samples, hidden_dim).
            y: True target values of shape (n_samples,).

        Returns:
            Dictionary with MSE, MAE, R², and Pearson correlation.
        """
        if self.model is None:
            raise RuntimeError("Probe not trained – call .fit() first.")

        y_true = _to_torch(y, self.dtype).cpu().numpy().flatten()
        y_pred = self.predict(X).flatten()

        # MSE
        mse = float(np.mean((y_pred - y_true) ** 2))

        # MAE
        mae = float(np.mean(np.abs(y_pred - y_true)))

        # R² (coefficient of determination)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Pearson correlation
        if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
            pearson_r, _ = pearsonr(y_true, y_pred)
            pearson_r = float(pearson_r)
        else:
            pearson_r = 0.0

        return {
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "pearson_r": pearson_r,
        }

    # ----------------------------- I/O helpers ----------------------------- #
    def save(self, path: str | Path) -> None:
        """Save the probe to a pickle file."""
        payload = {
            "model_state_dict": self.model.state_dict() if self.model is not None else None,
            "scaler_mean": self.scaler_mean.cpu() if self.scaler_mean is not None else None,
            "scaler_std": self.scaler_std.cpu() if self.scaler_std is not None else None,
            "reg_coeff": self.reg_coeff,
            "normalize": self.normalize,
            "fit_intercept": self.fit_intercept,
            "dtype": self.dtype,
            "verbose": self.verbose,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "use_mlp": self.use_mlp,
            "mlp_hidden_size": self.mlp_hidden_size,
            "preload_to_device": self.preload_to_device,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device | None = None) -> "DistanceProbe":
        """Load a probe from a pickle file."""
        with open(path, "rb") as f:
            payload = pickle.load(f)

        obj = cls(
            reg_coeff=payload["reg_coeff"],
            normalize=payload["normalize"],
            fit_intercept=payload["fit_intercept"],
            dtype=payload["dtype"],
            verbose=payload.get("verbose", 0),
            device=device,
            learning_rate=payload.get("learning_rate", 1e-3),
            num_epochs=payload.get("num_epochs", 100),
            batch_size=payload.get("batch_size"),
            use_mlp=payload.get("use_mlp", False),
            mlp_hidden_size=payload.get("mlp_hidden_size", 256),
            preload_to_device=payload.get("preload_to_device", True),
        )

        # Restore model from state dict
        if payload.get("model_state_dict") is not None:
            state_dict = payload["model_state_dict"]

            # Infer model dimensions from state dict
            if obj.use_mlp:
                first_weight = state_dict["0.weight"]
                last_weight = state_dict["2.weight"]
                n_features = first_weight.shape[1]
                n_outputs = last_weight.shape[0]
            else:
                weight = state_dict["weight"]
                n_outputs, n_features = weight.shape

            obj.model = _build_model(
                n_features=n_features,
                n_outputs=n_outputs,
                use_mlp=obj.use_mlp,
                mlp_hidden_size=obj.mlp_hidden_size,
                fit_intercept=obj.fit_intercept,
                dtype=obj.dtype,
                device=obj.device,
            )
            obj.model.load_state_dict(state_dict)
            obj.model.eval()

        # Restore scalers
        obj.scaler_mean = payload["scaler_mean"]
        obj.scaler_std = payload["scaler_std"]
        if obj.scaler_mean is not None:
            obj.scaler_mean = obj.scaler_mean.to(obj.device)
        if obj.scaler_std is not None:
            obj.scaler_std = obj.scaler_std.to(obj.device)

        return obj


# ----------------------------- Data loading ----------------------------- #
def load_activations_and_labels(
    activations_dir: Path,
    layer: int,
    label_column: str = "optimal_trajectory_length",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load activations and regression labels from a directory.

    Args:
        activations_dir: Directory containing activations and CSV file.
        layer: Layer index to extract from multi-layer activations.
        label_column: Column name in CSV to use as labels.

    Returns:
        Tuple of (activations, labels) tensors.
    """
    csv_files = list(activations_dir.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV file found in {activations_dir}")

    csv_file = csv_files[0]
    print(f"Using CSV file: {csv_file}")
    df = pd.read_csv(csv_file)

    # Load activations
    all_layer_path = activations_dir / "all_layer_acts.pt"
    single_layer_path = activations_dir / f"acts_layer_{layer}.pt"

    if all_layer_path.exists():
        all_layer_acts = torch.load(all_layer_path, weights_only=True)
        acts = all_layer_acts[:, layer, :].clone()
        del all_layer_acts
    elif single_layer_path.exists():
        one_layer_acts = torch.load(single_layer_path, weights_only=True)
        acts = one_layer_acts[:, 0, :] if one_layer_acts.ndim == 3 else one_layer_acts
    else:
        raise ValueError(f"Neither all_layer_acts.pt nor acts_layer_{layer}.pt found in {activations_dir}")

    if len(df) != acts.shape[0]:
        raise ValueError(f"Mismatch: CSV has {len(df)} rows but activations have {acts.shape[0]} samples")

    labels = torch.tensor(df[label_column].values, dtype=torch.float32)

    return acts, labels


def _print_regression_results(metrics: dict[str, float]) -> None:
    """Print regression evaluation results."""
    print("\n📊 Evaluation Results:")
    print(f"  MSE:       {metrics['mse']:.6f}")
    print(f"  MAE:       {metrics['mae']:.6f}")
    print(f"  R²:        {metrics['r2']:.6f}")
    print(f"  Pearson r: {metrics['pearson_r']:.6f}")


def train_and_save_distance_probe(
    activations_dir: str,
    layer: int,
    output_dir: str | None = None,
    label_column: str = "optimal_trajectory_length",
    eval_split: float = 0.2,
    reg_coeff: float = 1e-4,
    normalize: bool = False,
    fit_intercept: bool = True,
    verbose: int = 0,
    device: str | torch.device | None = None,
    learning_rate: float = 1e-3,
    num_epochs: int = 100,
    batch_size: int = 1024,
    use_mlp: bool = False,
    mlp_hidden_size: int = 256,
    seed: int = 42,
) -> str:
    """Train a distance regression probe and save it.

    Args:
        activations_dir: Directory containing activations and CSV with labels.
        layer: Layer index to use for activations.
        output_dir: Directory to save the probe (defaults to activations_dir/probes).
        label_column: CSV column name for regression targets.
        eval_split: Fraction of data for evaluation.
        reg_coeff: L2 regularization (weight decay).
        normalize: Whether to normalize features.
        fit_intercept: Whether to include bias terms.
        verbose: Verbosity level.
        device: Device for training.
        learning_rate: Learning rate for Adam.
        num_epochs: Number of training epochs.
        batch_size: Training batch size.
        use_mlp: Use MLP instead of linear.
        mlp_hidden_size: Hidden size for MLP.
        seed: Random seed for data shuffling.

    Returns:
        Path to the saved probe.
    """
    activations_dir = Path(activations_dir)
    acts, labels = load_activations_and_labels(activations_dir, layer, label_column)

    # Shuffle and split data
    generator = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(acts.shape[0], generator=generator)
    acts = acts[shuffled_indices]
    labels = labels[shuffled_indices]

    n_samples = acts.shape[0]
    n_eval = int(n_samples * eval_split)

    if n_eval > 0:
        train_acts, eval_acts = acts[:-n_eval], acts[-n_eval:]
        train_labels, eval_labels = labels[:-n_eval], labels[-n_eval:]
    else:
        train_acts, eval_acts = acts, acts[:0]
        train_labels, eval_labels = labels, labels[:0]

    print("\n🔢 Data split:")
    print(f"   Training:   {len(train_acts)} samples")
    print(f"   Evaluation: {len(eval_acts)} samples")
    print(f"   Label stats: min={labels.min():.2f}, max={labels.max():.2f}, mean={labels.mean():.2f}")

    # Train probe
    probe = DistanceProbe(
        reg_coeff=reg_coeff,
        normalize=normalize,
        fit_intercept=fit_intercept,
        verbose=verbose,
        device=device,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        use_mlp=use_mlp,
        mlp_hidden_size=mlp_hidden_size,
    )

    print(f"\n🚀 Training on device: {probe.device}")
    probe.fit(train_acts, train_labels)

    # Evaluate
    if len(eval_acts) > 0:
        eval_metrics = probe.evaluate(eval_acts, eval_labels)
        _print_regression_results(eval_metrics)
    else:
        print("\n⚠️  No evaluation data available (eval_split too small)")

    # Save
    if output_dir is None:
        output_dir = activations_dir / "probes"

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Build filename with configuration
    suffix_parts = []
    if use_mlp:
        suffix_parts.append(f"mlp{mlp_hidden_size}")
    if normalize:
        suffix_parts.append("normalized")

    suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
    output_path = output_dir / f"distance_probe_layer_{layer}{suffix}.pkl"

    probe.save(output_path)
    print(f"\n💾 Probe saved to {output_path}")

    return str(output_path)
