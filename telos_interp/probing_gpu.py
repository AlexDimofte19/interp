import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from tqdm import tqdm

from telos_interp import activations
from telos_interp.probing import (
    ArrayLike,
    _compute_metrics,
    _prepare_multiclass_data,
    _print_evaluation_results,
    _to_torch,
    load_activations,
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
        num_epochs: int = 100,
        batch_size: int | None = None,
        class_weight: str | dict[str, float] | None = None,
        use_mlp: bool = False,
        mlp_hidden_size: int = 1024,
    ) -> None:
        self.reg_coeff = float(reg_coeff)
        self.normalize = bool(normalize)
        self.fit_intercept = bool(fit_intercept)
        self.dtype = dtype
        self.verbose = int(verbose)
        self.learning_rate = float(learning_rate)
        self.num_epochs = int(num_epochs)
        self.batch_size = batch_size
        self.class_weight = class_weight
        self.use_mlp = bool(use_mlp)
        self.mlp_hidden_size = int(mlp_hidden_size)

        # Auto-detect device if not provided
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

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
        X_tensor, y_tensor, self.class_names, self.class_name_to_class_idx = _prepare_multiclass_data(
            activations_dict, self.dtype
        )

        # Normalisation ---------------------------------------------------- #
        if self.normalize:
            scaler_mean = X_tensor.mean(dim=0)
            scaler_std = X_tensor.std(dim=0)
            # Avoid division by zero
            scaler_std = torch.where(scaler_std > 1e-8, scaler_std, torch.ones_like(scaler_std))
            X_tensor = (X_tensor - scaler_mean) / scaler_std
            self.scaler_mean = scaler_mean.to(self.device)
            self.scaler_std = scaler_std.to(self.device)
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
        else:
            self.model = nn.Linear(n_features, n_classes, bias=self.fit_intercept).to(
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
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor).to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        n_samples = X_tensor.shape[0]
        if n_samples == 0:
            raise ValueError("Training data is empty.")

        if self.batch_size is None:
            batch_size = min(1024, n_samples)
        else:
            batch_size = int(self.batch_size)
        batch_size = max(1, batch_size)

        # Training
        self.model.train()
        step = 0
        last_loss = None
        for epoch in range(self.num_epochs):
            permutation = torch.randperm(n_samples)
            for start_idx in tqdm(range(0, n_samples, batch_size)):
                idx = permutation[start_idx : start_idx + batch_size]
                batch_X = X_tensor[idx].to(self.device, non_blocking=True)
                batch_y = y_tensor[idx].to(self.device, non_blocking=True)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                if self.reg_coeff > 0:
                    l2_reg = torch.tensor(0.0, device=self.device)
                    for name, param in self.model.named_parameters():
                        if "weight" in name:
                            l2_reg = l2_reg + torch.norm(param, 2) ** 2
                    loss = loss + 0.5 * self.reg_coeff * l2_reg

                loss.backward()
                optimizer.step()

                step += 1
                last_loss = loss.item()

            if self.verbose > 0:
                print(f"  Epoch {epoch}, Loss: {last_loss:.4f}")

        self.model.eval()

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
        if self.model is None:
            raise RuntimeError("Probe not trained – call .fit(...) first.")

        hs = _to_torch(hidden_states, self.dtype).to(self.device)

        if hs.ndim != 2:
            raise ValueError(f"hidden_states must be (seq_len, hidden_dim); got {hs.shape}.")

        if self.normalize:
            hs = (hs - self.scaler_mean) / self.scaler_std

        logits = self.model(hs)
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
        if self.model is None:
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
        if self.model is None:
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
            "model_state_dict": self.model.state_dict() if self.model is not None else None,
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
            "num_epochs": self.num_epochs,
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
            fit_intercept=payload["fit_intercept"],
            dtype=payload["dtype"],
            verbose=payload.get("verbose", 0),
            device=device,
            learning_rate=payload.get("learning_rate", 0.1),
            num_epochs=payload.get("num_epochs", 100),
            batch_size=payload.get("batch_size", None),
            class_weight=payload.get("class_weight", None),
            use_mlp=payload.get("use_mlp", False),
            mlp_hidden_size=payload.get("mlp_hidden_size", 1024),
        )

        # Restore linear layer or MLP model
        if payload.get("model_state_dict") is not None:
            if obj.use_mlp:
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
            else:
                # Load linear layer
                # Infer dimensions from state dict
                weight_shape = payload["model_state_dict"]["weight"].shape
                n_classes, n_features = weight_shape

                obj.model = nn.Linear(n_features, n_classes, bias=obj.fit_intercept).to(
                    dtype=obj.dtype, device=obj.device
                )
            obj.model.load_state_dict(payload["model_state_dict"])
            obj.model.eval()

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


def create_activations_dict(
    activations_dir: Path, layer: int, observability: activations.Observability
) -> dict[str, ArrayLike]:
    if "raw_acts" in activations_dir.name:
        # First find the csv file
        csv_file = activations_dir.glob("*.csv").__next__()
        if not csv_file:
            raise ValueError(f"No csv file found in {activations_dir}.")
        else:
            print(f"Using csv file: {csv_file}")
            df = pd.read_csv(csv_file)

        if os.path.exists(activations_dir / "all_layer_acts.pt"):
            all_layer_acts = torch.load(activations_dir / "all_layer_acts.pt")  # (num_samples, num_layers, hidden_dim)
            acts = all_layer_acts[:, layer, :].clone()
            del all_layer_acts
        elif os.path.exists(activations_dir / f"acts_layer_{layer}.pt"):  # (num_samples, 1, hidden_dim)
            one_layer_acts = torch.load(activations_dir / f"acts_layer_{layer}.pt")
            acts = one_layer_acts[:, 0, :]
        else:
            raise ValueError(f"Neither all_layer_acts.pt nor acts_layer_{layer}.pt found in {activations_dir}.")

        assert len(df) == acts.shape[0], (
            f"Number of rows in csv and activations must match. {len(df)} != {acts.shape[0]}"
        )

        activations_dict = activations.get_cell_type_activations_from_csv(df, acts, observability)
        return activations_dict

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

    for class_name, acts in activations_dict.items():
        if acts.ndim == 3:  # (num_samples, num_layers, hidden_dim)
            if acts.shape[1] == 1:
                # We only saved one layer
                activations_dict[class_name] = acts[:, 0, :]
            else:
                activations_dict[class_name] = acts[:, layer, :]
        elif acts.ndim == 2:  # (num_samples, hidden_dim)
            # Legacy format
            activations_dict[class_name] = acts
        else:
            raise ValueError(f"Unsupported activation shape: {acts.shape}")

    return activations_dict


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
    num_epochs: int = 100,
    class_weight: str | dict[str, float] | None = None,
    balance_classes: bool = False,
    seed: int = 42,
    use_mlp: bool = False,
    mlp_hidden_size: int = 1024,
    batch_size: int = 1024,
    observability: activations.Observability = activations.Observability.full,
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
        num_epochs: Number of epochs for training
        class_weight: Class weights for handling class imbalance ('balanced' or dict mapping class names to weights)
        balance_classes: Whether to balance classes by downsampling the majority classes
        use_mlp: Whether to use an MLP architecture instead of linear
        mlp_hidden_size: Hidden size for the MLP (only used if use_mlp=True)

    Returns:
        Path to the saved probe
    """
    activations_dir = Path(activations_dir)

    activations_dict = create_activations_dict(activations_dir, layer, observability)

    # Split data for training and evaluation
    train_dict = {}
    eval_dict = {}

    if balance_classes:
        # activations_dict = balance_classes_by_downsampling(activations_dict, seed=seed)
        activations_dict = balance_classes_by_upsampling(activations_dict, seed=seed)

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
        num_epochs=num_epochs,
        class_weight=class_weight,
        use_mlp=use_mlp,
        mlp_hidden_size=mlp_hidden_size,
        batch_size=batch_size,
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
    if balance_classes:
        suffix_parts.append("balanced")

    suffix = "_" + "_".join(suffix_parts)
    output_path = output_dir / f"multiclass_probe_layer_{layer}{suffix}.pkl"
    probe.save(output_path)

    print(f"\n💾 Probe saved to {output_path}")
    return str(output_path)


def predict_from_activations_list(
    probe_path: str,
    activations_list: list[dict],
) -> list[dict[str, Any]]:
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
            "predictions": {'(x,y)': {class_name: probability, class_name: probability, ...}, ...},
        },
        ...
    ]
    """
    probe = MultiClassProbingClassifierGPU.load(probe_path)

    results = []
    for activations_dict in activations_list:
        saved_observation = activations_dict["observation"]

        row_predictions = {}
        for coordinates, activation in activations_dict["activations"].items():
            x, y = coordinates

            cell_predictions = {}
            probabilities = probe.predict_proba(activation.unsqueeze(0))
            for class_name, class_idx in probe.class_name_to_class_idx.items():
                cell_predictions[class_name] = probabilities[0, class_idx].item()
            # Predictions are stored in (column, row) order.
            row_predictions[str((x, y))] = cell_predictions

        results.append(
            {
                "observation": saved_observation,
                "predictions": row_predictions,
            }
        )

    return results


def get_predicted_class(per_cell_predictions: dict[str, Any]) -> str:
    """Get the predicted class from the predictions dictionary."""
    max_probability = 0
    predicted_class = None
    for class_name, probability in per_cell_predictions.items():
        if probability > max_probability:
            max_probability = probability
            predicted_class = class_name
    return predicted_class, max_probability


def evaluate_predictions(csv_path: str, predictions: list[dict[str, Any]]) -> dict[str, float]:
    """Evaluate the predictions against the ground truth in the csv file.

    predictions has the following structure:
    [
        {
            "observation": str,
            "predictions": {'(x,y)': {class_name: probability, class_name: probability, ...}, ...},
        },
        ...
    ]
    """

    df = pd.read_csv(csv_path)

    if len(df) != len(predictions):
        print(f"Number of rows in csv and predictions must match. {len(df)} != {len(predictions)}")
        return {}

    per_row_accuracies = []
    per_class_accuracies = {}

    all_class_names = predictions[0]["predictions"]["(0, 0)"].keys()
    per_class_accuracies = {class_name: [] for class_name in all_class_names}

    for row_idx, row in df.iterrows():
        true_fo_cell_types = eval(row.fo_cell_types)
        correct_predictions = 0
        total_predictions = 0
        class_counts = dict.fromkeys(all_class_names, 0)
        class_correct_predictions = dict.fromkeys(all_class_names, 0)
        for x, y, true_cell_type in true_fo_cell_types:
            xy_key = f"({x}, {y})"
            pred_xy_probabilities = predictions[row_idx]["predictions"][xy_key]
            predicted_class, max_probability = get_predicted_class(pred_xy_probabilities)
            if predicted_class == true_cell_type:
                correct_predictions += 1
                class_correct_predictions[predicted_class] += 1

            total_predictions += 1
            class_counts[true_cell_type] += 1

        per_row_accuracies.append(correct_predictions / (total_predictions or 1))
        for class_name in all_class_names:
            per_class_accuracies[class_name].append(
                class_correct_predictions[class_name] / (class_counts[class_name] or 1)
            )

    total_accuracy = sum(per_row_accuracies) / len(per_row_accuracies)
    total_per_class_accuracies = {}
    for class_name in all_class_names:
        total_per_class_accuracies[class_name] = sum(per_class_accuracies[class_name]) / len(
            per_class_accuracies[class_name]
        )

    return {
        "total_accuracy": total_accuracy,
        "total_per_class_accuracies": total_per_class_accuracies,
        # "per_row_accuracies": per_row_accuracies,
        # "per_class_accuracies": per_class_accuracies,
    }
