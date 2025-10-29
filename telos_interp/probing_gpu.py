import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn, optim

from telos_interp.activations import Observability, ObservationType
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
        interaction_features: bool = False,
        fit_intercept: bool = True,
        dtype: torch.dtype = torch.float32,
        verbose: int = 0,
        device: str | torch.device | None = None,
        learning_rate: float = 0.1,
        max_iter: int = 1000,
        batch_size: int | None = None,
        class_weight: str | dict[str, float] | None = None,
        use_mlp: bool = False,
        mlp_hidden_size: int = 1024,
    ) -> None:
        self.reg_coeff = float(reg_coeff)
        self.normalize = bool(normalize)
        self.fit_intercept = bool(fit_intercept)
        self.interaction_features = bool(interaction_features)
        self.dtype = dtype
        self.verbose = int(verbose)
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.batch_size = batch_size
        self.class_weight = class_weight
        self.use_mlp = bool(use_mlp)
        self.mlp_hidden_size = int(mlp_hidden_size)

        # Auto-detect device if not provided
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.linear: nn.Linear | None = None
        self.model: nn.Module | None = None  # For MLP architecture
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

        if self.interaction_features:
            H = X_tensor.shape[1] - 2
            h = X_tensor[:, :H]
            xy = X_tensor[:, H:]  # (N, 2)

            # Build interaction features (N, 2H): [h * x, h * y]
            hx = h * xy[:, 0:1]  # broadcast
            hy = h * xy[:, 1:2]
            X_tensor = torch.cat([h, xy, hx, hy], dim=1)

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

        # Initialize linear layer or MLP ----------------------------------- #
        n_features = X_tensor.shape[1]
        n_classes = len(self.class_names)

        if self.use_mlp:
            # Build MLP: input -> hidden -> ReLU -> output
            self.model = nn.Sequential(
                nn.Linear(n_features, self.mlp_hidden_size, bias=self.fit_intercept),
                nn.ReLU(),
                nn.Linear(self.mlp_hidden_size, n_classes, bias=self.fit_intercept),
            ).to(dtype=self.dtype, device=self.device)
            # Keep linear as None for MLP mode
            self.linear = None
        else:
            self.linear = nn.Linear(n_features, n_classes, bias=self.fit_intercept).to(
                dtype=self.dtype, device=self.device
            )
            # Keep model as None for linear mode
            self.model = None

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
        # Get the appropriate model for training
        train_model = self.model if self.use_mlp else self.linear

        # L2 regularization strength (sklearn's C=1/reg_coeff maps to weight_decay=reg_coeff in PyTorch)
        # optimizer = optim.LBFGS(
        #     train_model.parameters(),
        #     lr=self.learning_rate,
        #     max_iter=20,
        # )
        optimizer = optim.Adam(train_model.parameters(), lr=self.learning_rate)

        # For LBFGS optimizer
        def closure():
            optimizer.zero_grad()
            outputs = train_model(X_tensor)
            loss = criterion(outputs, y_tensor)

            # L2 regularization
            if self.reg_coeff > 0:
                l2_reg = torch.tensor(0.0, device=self.device)
                for name, param in train_model.named_parameters():
                    if "weight" in name:  # do not penalize bias
                        l2_reg = l2_reg + torch.norm(param, 2) ** 2
                loss = loss + 0.5 * self.reg_coeff * l2_reg

            loss.backward()
            return loss

        # Training
        train_model.train()
        for i in range(self.max_iter // 20):  # LBFGS takes ~20 iterations per step
            loss = optimizer.step(closure)

            if self.verbose > 0 and i % 10 == 0:
                print(f"  Iteration {i * 20}, Loss: {loss.item():.4f}")

        train_model.eval()

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
        if self.linear is None and self.model is None:
            raise RuntimeError("Probe not trained – call .fit(...) first.")

        hs = _to_torch(hidden_states, self.dtype).to(self.device)

        if hs.ndim != 2:
            raise ValueError(f"hidden_states must be (seq_len, hidden_dim); got {hs.shape}.")

        if self.interaction_features:
            H = hs.shape[1] - 2
            h = hs[:, :H]
            xy = hs[:, H:]  # (N, 2)

            # Build interaction features (N, 2H): [h * x, h * y]
            hx = h * xy[:, 0:1]  # broadcast
            hy = h * xy[:, 1:2]
            hs = torch.cat([h, xy, hx, hy], dim=1)

        if self.normalize:
            hs = (hs - self.scaler_mean) / self.scaler_std

        # Use the appropriate model
        predict_model = self.model if self.use_mlp else self.linear
        logits = predict_model(hs)
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
        if self.linear is None and self.model is None:
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
        if self.linear is None and self.model is None:
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
            "model_state_dict": self.model.state_dict() if self.model is not None else None,
            "scaler_mean": self.scaler_mean.cpu() if self.scaler_mean is not None else None,
            "scaler_std": self.scaler_std.cpu() if self.scaler_std is not None else None,
            "reg_coeff": self.reg_coeff,
            "normalize": self.normalize,
            "interaction_features": self.interaction_features,
            "fit_intercept": self.fit_intercept,
            "dtype": self.dtype,
            "verbose": self.verbose,
            "class_names": self.class_names,
            "class_name_to_class_idx": self.class_name_to_class_idx,
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "batch_size": self.batch_size,
            "class_weight": self.class_weight,
            "use_mlp": self.use_mlp,
            "mlp_hidden_size": self.mlp_hidden_size,
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
            interaction_features=payload.get("interaction_features", False),
            fit_intercept=payload["fit_intercept"],
            dtype=payload["dtype"],
            verbose=payload.get("verbose", 0),
            device=device,
            learning_rate=payload.get("learning_rate", 0.1),
            max_iter=payload.get("max_iter", 1000),
            batch_size=payload.get("batch_size", None),
            class_weight=payload.get("class_weight", None),
            use_mlp=payload.get("use_mlp", False),
            mlp_hidden_size=payload.get("mlp_hidden_size", 1024),
        )

        # Restore linear layer or MLP model
        if payload.get("model_state_dict") is not None:
            # Load MLP model
            # Infer dimensions from state dict
            first_layer_weight = payload["model_state_dict"]["0.weight"]
            last_layer_weight = payload["model_state_dict"]["2.weight"]
            n_classes, hidden_size = last_layer_weight.shape
            hidden_size_in, n_features = first_layer_weight.shape

            obj.model = nn.Sequential(
                nn.Linear(n_features, hidden_size_in, bias=obj.fit_intercept),
                nn.ReLU(),
                nn.Linear(hidden_size_in, n_classes, bias=obj.fit_intercept),
            ).to(dtype=obj.dtype, device=obj.device)
            obj.model.load_state_dict(payload["model_state_dict"])
            obj.model.eval()
        elif payload.get("linear_state_dict") is not None:
            # Load linear layer
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


def balance_classes_by_downsampling(train_dict: dict[str, ArrayLike], seed: int = 42) -> dict[str, ArrayLike]:
    """Balance classes by downsampling the majority classes."""
    min_samples = min(acts.shape[0] for acts in train_dict.values())
    balanced_train_dict = {}
    for class_name, acts in train_dict.items():
        shuffled_indices = torch.randperm(acts.shape[0], generator=torch.Generator().manual_seed(seed))
        balanced_acts = acts[shuffled_indices[:min_samples]]
        balanced_train_dict[class_name] = balanced_acts

    return balanced_train_dict


def balance_classes_by_upsampling(train_dict: dict[str, ArrayLike], seed: int = 42) -> dict[str, ArrayLike]:
    """Balance classes by upsampling the minority classes."""
    max_samples = max(acts.shape[0] for acts in train_dict.values())
    balanced_train_dict = {}
    for class_name, acts in train_dict.items():
        generator = torch.Generator().manual_seed(seed)
        shuffled_indices = torch.randint(0, acts.shape[0], (max_samples,), generator=generator)
        upsampled_acts = acts[shuffled_indices]
        balanced_train_dict[class_name] = upsampled_acts

    return balanced_train_dict


def train_and_save_multiclass_probe_gpu(
    activations_dir: str,
    layer: int,
    output_dir: str = None,
    eval_split: float = 0.2,
    reg_coeff: float = 1e3,
    normalize: bool = False,
    interaction_features: bool = False,
    fit_intercept: bool = True,
    verbose: int = 0,
    device: str | torch.device | None = None,
    learning_rate: float = 0.1,
    max_iter: int = 1000,
    class_weight: str | dict[str, float] | None = None,
    balance_classes: bool = False,
    seed: int = 42,
    use_mlp: bool = False,
    mlp_hidden_size: int = 1024,
) -> str:
    """Train a multi-class probing classifier on grid cell activations using GPU.

    Args:
        activations_dir: Directory containing activation files (acts_wall.pt, acts_empty.pt, etc.)
        layer: Layer to train the probe on (not used for single-layer activations)
        output_dir: Directory to save the probe
        eval_split: Fraction of data to use for evaluation
        reg_coeff: Regularization coefficient
        normalize: Whether to normalize activations
        interaction_features: Whether to use interaction features
        fit_intercept: Whether to fit an intercept term in the logistic regression
        verbose: Verbosity level for the logistic regression solver (0 = silent, 1+ = verbose)
        device: Device to use for training ('cuda', 'cpu', or None for auto-detect)
        learning_rate: Learning rate for optimization
        max_iter: Maximum number of iterations
        class_weight: Class weights for handling class imbalance ('balanced' or dict mapping class names to weights)
        balance_classes: Whether to balance classes by downsampling the majority classes
        use_mlp: Whether to use an MLP architecture instead of linear
        mlp_hidden_size: Hidden size for the MLP (only used if use_mlp=True)

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

    if balance_classes:
        activations_dict = balance_classes_by_downsampling(activations_dict, seed=seed)
        # activations_dict = balance_classes_by_upsampling(activations_dict, seed=seed)

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
        interaction_features=interaction_features,
        fit_intercept=fit_intercept,
        verbose=verbose,
        device=device,
        learning_rate=learning_rate,
        max_iter=max_iter,
        class_weight=class_weight,
        use_mlp=use_mlp,
        mlp_hidden_size=mlp_hidden_size,
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
    if use_mlp:
        suffix_parts.append(f"mlp{mlp_hidden_size}")
    if not fit_intercept:
        suffix_parts.append("no_intercept")
    if normalize:
        suffix_parts.append("normalized")

    suffix = "_" + "_".join(suffix_parts)
    output_path = output_dir / f"multiclass_probe_layer_{layer}{suffix}.pkl"
    probe.save(output_path)

    print(f"\n💾 Probe saved to {output_path}")
    return str(output_path)


def predict_from_csv(
    probe_path: str,
    activations_list: list[dict],
    input_csv_path: str,
    observability: Observability,
    observation_type: ObservationType,
) -> list[dict]:
    """Predict using the multiclass probe on a list of activations.
    Activations list has the following structure:
    [
        {
            "observation": str,
            "activations": {(x,y): torch.Tensor, ...},
        },
        ...
    ]
    Each entry = one grid, with the activations for each cell in the grid.
    Resulting list has the following structure:
    [
        {
            "observation": str,
            "predictions": {(x,y): {class_name: probability, class_name: probability, ...}, ...},
        },
        ...
    ]
    """
    probe = MultiClassProbingClassifierGPU.load(probe_path)

    df = pd.read_csv(input_csv_path)
    results = []
    for activations_dict, (_idx, row) in zip(activations_list, df.iterrows(), strict=False):
        saved_observation = activations_dict["observation"]
        if observability == Observability.full:
            prefix = "fo_"
        elif observability == Observability.partial:
            prefix = "po_"
        if observation_type == ObservationType.grid_only:
            row_observation = row[f"{prefix}observation"]
        elif observation_type == ObservationType.full_prompt:
            row_observation = row[f"{prefix}prompt"]

        assert saved_observation == row_observation, "Observations do not match"
        row_predictions = {}
        for coordinates, activation in activations_dict["activations"].items():
            x, y = coordinates

            cell_predictions = {}
            probabilities = probe.predict_proba(activation.unsqueeze(0))
            for class_name, class_idx in probe.class_name_to_class_idx.items():
                cell_predictions[class_name] = probabilities[0, class_idx].item()
            # Predictions are stored in (row, column) order.
            row_predictions[str((y, x))] = cell_predictions

        results.append(
            {
                "observation": saved_observation,
                "predictions": row_predictions,
            }
        )

    return results
