import pickle
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

from telos_interp.probing import (
    ArrayLike,
    _compute_metrics,
    _prepare_multiclass_data,
    _print_evaluation_results,
    _to_torch,
)


class MultiClassProbingClassifierGPU:
    """Multi-class linear probe trained with PyTorch for GPU acceleration."""

    def __init__(
        self,
        reg_coeff: float = 1e3,
        normalize: bool = False,
        fit_intercept: bool = True,
        dtype: torch.dtype = torch.float32,
        verbose: int = 0,
        device: str | torch.device | None = None,
        learning_rate: float = 0.1,
        max_iter: int = 1000,
        batch_size: int | None = None,
        class_weight: str | dict[str, float] | None = None,
    ) -> None:
        self.reg_coeff = float(reg_coeff)
        self.normalize = bool(normalize)
        self.fit_intercept = bool(fit_intercept)
        self.dtype = dtype
        self.verbose = int(verbose)
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.batch_size = batch_size
        self.class_weight = class_weight

        # Auto-detect device if not provided
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.linear: nn.Linear | None = None
        self.scaler_mean: torch.Tensor | None = None
        self.scaler_std: torch.Tensor | None = None
        self.class_names: list[str] | None = None
        self.class_name_to_class_idx: dict[str, int] = {}

    # ----------------------------- training -------------------------------- #
    def fit(self, activations_dict: dict[str, ArrayLike]) -> "MultiClassProbingClassifierGPU":
        """Fit the probe on multi-class activations.

        Parameters:
        activations_dict: Dictionary mapping class names to activations
                         e.g., {"wall": (n_wall, hidden_dim), "empty": (n_empty, hidden_dim), ...}

        Returns:
        self
        """
        # Prepare data using common utility
        X, y, self.class_names, self.class_name_to_class_idx = _prepare_multiclass_data(activations_dict, self.dtype)

        # Convert to torch tensors and move to device
        X_tensor = torch.from_numpy(X).to(dtype=self.dtype, device=self.device)
        y_tensor = torch.from_numpy(y).to(device=self.device)

        # Normalisation ---------------------------------------------------- #
        if self.normalize:
            self.scaler_mean = X_tensor.mean(dim=0)
            self.scaler_std = X_tensor.std(dim=0)
            # Avoid division by zero
            self.scaler_std = torch.where(self.scaler_std > 1e-8, self.scaler_std, torch.ones_like(self.scaler_std))
            X_tensor = (X_tensor - self.scaler_mean) / self.scaler_std
        else:
            self.scaler_mean = torch.zeros(X_tensor.shape[1], dtype=self.dtype, device=self.device)
            self.scaler_std = torch.ones(X_tensor.shape[1], dtype=self.dtype, device=self.device)

        # Initialize linear layer ------------------------------------------ #
        n_features = X_tensor.shape[1]
        n_classes = len(self.class_names)

        self.linear = nn.Linear(n_features, n_classes, bias=self.fit_intercept).to(
            dtype=self.dtype, device=self.device
        )

        # Compute class weights if requested ------------------------------ #
        class_weights_tensor = None
        if self.class_weight == "balanced":
            # Compute balanced class weights: n_samples / (n_classes * class_counts)
            unique_classes, class_counts = torch.unique(y_tensor, return_counts=True)
            n_samples = len(y_tensor)
            class_weights = n_samples / (n_classes * class_counts.float())
            class_weights_tensor = class_weights.to(dtype=self.dtype, device=self.device)

            if self.verbose > 0:
                print("  Using balanced class weights:")
                for i, class_name in enumerate(self.class_names):
                    print(f"    {class_name}: {class_weights[i].item():.4f}")
        elif isinstance(self.class_weight, dict):
            # Use provided class weights dictionary
            class_weights = torch.zeros(n_classes, dtype=self.dtype, device=self.device)
            for class_name, weight in self.class_weight.items():
                if class_name in self.class_name_to_class_idx:
                    class_idx = self.class_name_to_class_idx[class_name]
                    class_weights[class_idx] = weight
            class_weights_tensor = class_weights

        # Training loop ---------------------------------------------------- #
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        # L2 regularization strength (sklearn's C=1/reg_coeff maps to weight_decay=reg_coeff in PyTorch)
        optimizer = optim.LBFGS(
            self.linear.parameters(),
            lr=self.learning_rate,
            max_iter=20,
        )

        # For LBFGS optimizer
        def closure():
            optimizer.zero_grad()
            outputs = self.linear(X_tensor)
            loss = criterion(outputs, y_tensor)

            # L2 regularization
            if self.reg_coeff > 0:
                l2_reg = torch.tensor(0.0, device=self.device)
                for param in self.linear.parameters():
                    l2_reg = l2_reg + torch.norm(param, 2) ** 2
                loss = loss + 0.5 * self.reg_coeff * l2_reg

            loss.backward()
            return loss

        # Training
        self.linear.train()
        for i in range(self.max_iter // 20):  # LBFGS takes ~20 iterations per step
            loss = optimizer.step(closure)

            if self.verbose > 0 and i % 10 == 0:
                print(f"  Iteration {i * 20}, Loss: {loss.item():.4f}")

        self.linear.eval()

        return self

    # ------------------------------ scoring -------------------------------- #
    @torch.no_grad()
    def predict_proba(self, hidden_states: ArrayLike) -> torch.Tensor:
        """
        Get class probabilities for a single sequence.

        Parameters:
        hidden_states : (seq_len, hidden_dim)

        Returns:
        torch.Tensor
            2-D tensor of shape (seq_len, n_classes) with class probabilities.
        """
        if self.linear is None:
            raise RuntimeError("Probe not trained – call .fit(...) first.")

        hs = _to_torch(hidden_states, self.dtype).to(self.device)

        if hs.ndim != 2:
            raise ValueError(f"hidden_states must be (seq_len, hidden_dim); got {hs.shape}.")

        if self.normalize:
            hs = (hs - self.scaler_mean) / self.scaler_std

        logits = self.linear(hs)
        probabilities = torch.softmax(logits, dim=-1)

        return probabilities.cpu()

    @torch.no_grad()
    def predict(self, hidden_states: ArrayLike) -> np.ndarray:
        """
        Get predicted class labels for a single sequence.

        Parameters:
        hidden_states : (seq_len, hidden_dim)

        Returns:
        np.ndarray
            1-D tensor of shape (seq_len,) with predicted class indices.
        """
        if self.linear is None:
            raise RuntimeError("Probe not trained – call .fit(...) first.")

        probabilities = self.predict_proba(hidden_states)
        predictions = torch.argmax(probabilities, dim=-1)

        return predictions.cpu().numpy()

    def evaluate(self, activations_dict: dict[str, ArrayLike]) -> dict[str, float]:
        """
        Evaluate the probe on test data.

        Parameters:
        activations_dict: Dictionary mapping class names to test activations

        Returns:
        Dict[str, float]
            Dictionary containing evaluation metrics
        """
        if self.linear is None:
            raise RuntimeError("Probe not trained – call .fit(...) first.")

        # Prepare test data using common utility
        # We pass the class names in the same order they were learned
        test_dict_ordered = {name: activations_dict[name] for name in self.class_names}
        X_test, y_test, _, _ = _prepare_multiclass_data(test_dict_ordered, self.dtype)

        # Get predictions
        y_pred = self.predict(X_test)

        # Compute metrics using common utility
        return _compute_metrics(y_test, y_pred, self.class_names, self.class_name_to_class_idx)

    # ----------------------------- I/O helpers ---------------------------- #
    def save(self, file: str | Path) -> None:
        """Pickle the probe to *file*."""
        payload = {
            "linear_state_dict": self.linear.state_dict() if self.linear is not None else None,
            "scaler_mean": self.scaler_mean.cpu() if self.scaler_mean is not None else None,
            "scaler_std": self.scaler_std.cpu() if self.scaler_std is not None else None,
            "reg_coeff": self.reg_coeff,
            "normalize": self.normalize,
            "fit_intercept": self.fit_intercept,
            "dtype": self.dtype,
            "verbose": self.verbose,
            "class_names": self.class_names,
            "class_name_to_class_idx": self.class_name_to_class_idx,
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "batch_size": self.batch_size,
            "class_weight": self.class_weight,
        }
        with open(file, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, file: str | Path, device: str | torch.device | None = None) -> "MultiClassProbingClassifierGPU":
        """Load a probe previously saved with :pymeth:`save`."""
        with open(file, "rb") as f:
            payload = pickle.load(f)

        obj = cls(
            reg_coeff=payload["reg_coeff"],
            normalize=payload["normalize"],
            fit_intercept=payload["fit_intercept"],
            dtype=payload["dtype"],
            verbose=payload.get("verbose", 0),
            device=device,
            learning_rate=payload.get("learning_rate", 0.1),
            max_iter=payload.get("max_iter", 1000),
            batch_size=payload.get("batch_size", None),
            class_weight=payload.get("class_weight", None),
        )

        # Restore linear layer
        if payload["linear_state_dict"] is not None:
            # Infer dimensions from state dict
            weight_shape = payload["linear_state_dict"]["weight"].shape
            n_classes, n_features = weight_shape

            obj.linear = nn.Linear(n_features, n_classes, bias=obj.fit_intercept).to(
                dtype=obj.dtype, device=obj.device
            )
            obj.linear.load_state_dict(payload["linear_state_dict"])
            obj.linear.eval()

        obj.scaler_mean = payload["scaler_mean"]
        obj.scaler_std = payload["scaler_std"]
        obj.class_names = payload["class_names"]
        obj.class_name_to_class_idx = payload["class_name_to_class_idx"]

        # Move scaler to device if they exist
        if obj.scaler_mean is not None:
            obj.scaler_mean = obj.scaler_mean.to(obj.device)
        if obj.scaler_std is not None:
            obj.scaler_std = obj.scaler_std.to(obj.device)

        return obj


def train_and_save_multiclass_probe_gpu(
    activations_dir: str,
    layer: int,
    output_dir: str = None,
    eval_split: float = 0.2,
    reg_coeff: float = 1e3,
    normalize: bool = False,
    fit_intercept: bool = True,
    verbose: int = 0,
    device: str | torch.device | None = None,
    learning_rate: float = 0.1,
    max_iter: int = 1000,
    class_weight: str | dict[str, float] | None = None,
) -> str:
    """Train a multi-class probing classifier on grid cell activations using GPU.

    Args:
        activations_dir: Directory containing activation files (acts_wall.pt, acts_empty.pt, etc.)
        layer: Layer to train the probe on (not used for single-layer activations)
        output_dir: Directory to save the probe
        eval_split: Fraction of data to use for evaluation
        reg_coeff: Regularization coefficient
        normalize: Whether to normalize activations
        fit_intercept: Whether to fit an intercept term in the logistic regression
        verbose: Verbosity level for the logistic regression solver (0 = silent, 1+ = verbose)
        device: Device to use for training ('cuda', 'cpu', or None for auto-detect)
        learning_rate: Learning rate for optimization
        max_iter: Maximum number of iterations
        class_weight: Class weights for handling class imbalance ('balanced' or dict mapping class names to weights)

    Returns:
        Path to the saved probe
    """
    from telos_interp.probing import load_activations

    activations_dir = Path(activations_dir)

    # Find all activation files
    activation_files = {}
    for file_path in activations_dir.glob("acts_*.pt"):
        class_name = file_path.stem.replace("acts_", "")
        activation_files[class_name] = file_path

    if not activation_files:
        raise ValueError(f"No activation files found in {activations_dir}")

    print(f"🔍 Found activation files for classes: {list(activation_files.keys())}")

    # Load activations for each class
    activations_dict = {}
    for class_name, file_path in activation_files.items():
        acts = load_activations(file_path)
        print(f"   {class_name}: {acts.shape}")
        activations_dict[class_name] = acts

    # Split data for training and evaluation
    train_dict = {}
    eval_dict = {}

    for class_name, acts in activations_dict.items():
        n_samples = acts.shape[0]
        n_eval = int(n_samples * eval_split)

        train_dict[class_name] = acts[:-n_eval] if n_eval > 0 else acts
        eval_dict[class_name] = acts[-n_eval:] if n_eval > 0 else acts[:0]

    print("\n🔢 Data split:")
    for class_name in activations_dict.keys():
        train_count = train_dict[class_name].shape[0]
        eval_count = eval_dict[class_name].shape[0]
        print(f"   {class_name}: {train_count} train, {eval_count} eval")

    # Train the probe
    probe = MultiClassProbingClassifierGPU(
        reg_coeff=reg_coeff,
        normalize=normalize,
        fit_intercept=fit_intercept,
        verbose=verbose,
        device=device,
        learning_rate=learning_rate,
        max_iter=max_iter,
        class_weight=class_weight,
    )

    print(f"\n🚀 Training on device: {probe.device}")
    probe.fit(train_dict)

    # Evaluate the probe
    if any(eval_dict[class_name].shape[0] > 0 for class_name in eval_dict.keys()):
        eval_metrics = probe.evaluate(eval_dict)
        _print_evaluation_results(eval_metrics, probe.class_names)
    else:
        print("\n⚠️  No evaluation data available (eval_split too small)")

    # Save the probe
    if output_dir is None:
        output_dir = activations_dir / "probes"

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Only add suffix if not using defaults (no intercept or normalizing)
    suffix_parts = ["gpu"]
    if not fit_intercept:
        suffix_parts.append("no_intercept")
    if normalize:
        suffix_parts.append("normalized")

    suffix = "_" + "_".join(suffix_parts)
    output_path = output_dir / f"multiclass_probe_layer_{layer}{suffix}.pkl"
    probe.save(output_path)

    print(f"\n💾 Probe saved to {output_path}")
    return str(output_path)
