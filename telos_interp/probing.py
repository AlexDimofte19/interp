import pickle
from pathlib import Path

import nnsight
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from telos_interp import activations as act_module

ArrayLike = np.ndarray | torch.Tensor


# ----------------------------- Common utilities ----------------------------- #
def _prepare_multiclass_data(
    activations_dict: dict[str, ArrayLike],
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, list[str], dict[str, int]]:
    """Prepare multi-class data from activations dictionary.

    Returns:
        X: (n_samples, hidden_dim) numpy array
        y: (n_samples,) numpy array of class indices
        class_names: list of class names
        class_name_to_class_idx: mapping from class name to index
    """
    X_list = []
    y_list = []
    class_names = list(activations_dict.keys())
    class_name_to_class_idx = {}

    for class_idx, (class_name, acts) in enumerate(activations_dict.items()):
        class_name_to_class_idx[class_name] = class_idx
        if isinstance(acts, np.ndarray):
            acts_tensor = torch.from_numpy(acts)
        else:
            acts_tensor = acts

        X_list.append(acts_tensor.to(dtype=dtype))
        y_list.append(torch.full((acts_tensor.shape[0],), class_idx, dtype=torch.long))

    X_tensor = torch.cat(X_list, dim=0)
    y_tensor = torch.cat(y_list, dim=0)

    del X_list, y_list

    return X_tensor, y_tensor, class_names, class_name_to_class_idx


def _compute_metrics(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    class_name_to_class_idx: dict[str, int],
) -> dict[str, float]:
    """Compute evaluation metrics for multi-class classification.

    Returns:
        Dictionary containing overall and per-class metrics
    """
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    # Calculate per-class metrics
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

    # Calculate predicted counts for each class
    predicted_counts = {}
    for class_name in class_names:
        class_idx = class_name_to_class_idx[class_name]
        predicted_counts[class_name] = int(np.sum(y_pred == class_idx))

    # Extract per-class metrics
    per_class_metrics = {}
    for class_name in class_names:
        if class_name in report:
            per_class_metrics[f"{class_name}_precision"] = report[class_name]["precision"]
            per_class_metrics[f"{class_name}_recall"] = report[class_name]["recall"]
            per_class_metrics[f"{class_name}_f1"] = report[class_name]["f1-score"]
            per_class_metrics[f"{class_name}_support"] = report[class_name]["support"]
            per_class_metrics[f"{class_name}_predicted"] = predicted_counts[class_name]

    # Calculate macro averages
    macro_precision = report["macro avg"]["precision"]
    macro_recall = report["macro avg"]["recall"]
    macro_f1 = report["macro avg"]["f1-score"]

    # Calculate weighted averages
    weighted_precision = report["weighted avg"]["precision"]
    weighted_recall = report["weighted avg"]["recall"]
    weighted_f1 = report["weighted avg"]["f1-score"]

    metrics = {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        **per_class_metrics,
    }

    return metrics


def _print_evaluation_results(eval_metrics: dict[str, float], class_names: list[str]) -> None:
    """Print evaluation results in a formatted way."""
    print("\n📊 Evaluation Results:")

    # Print overall metrics first
    print(f"  Accuracy: {eval_metrics['accuracy']:.4f}")
    print(
        f"  Macro avg    - Precision: {eval_metrics['macro_precision']:.4f}, Recall: {eval_metrics['macro_recall']:.4f}, F1: {eval_metrics['macro_f1']:.4f}"
    )
    print(
        f"  Weighted avg - Precision: {eval_metrics['weighted_precision']:.4f}, Recall: {eval_metrics['weighted_recall']:.4f}, F1: {eval_metrics['weighted_f1']:.4f}"
    )

    # Print per-class metrics as a DataFrame
    print("\n  Per-Class Metrics:")
    per_class_data = []
    for class_name in class_names:
        per_class_data.append(
            {
                "Class": class_name,
                "Precision": eval_metrics.get(f"{class_name}_precision", 0),
                "Recall": eval_metrics.get(f"{class_name}_recall", 0),
                "F1-Score": eval_metrics.get(f"{class_name}_f1", 0),
                "GT Support": int(eval_metrics.get(f"{class_name}_support", 0)),
                "Predicted": int(eval_metrics.get(f"{class_name}_predicted", 0)),
            }
        )

    df = pd.DataFrame(per_class_data)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


class ProbingClassifier:
    """Binary linear probe trained with logistic regression."""

    def __init__(
        self,
        reg_coeff: float = 1e3,
        normalize: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.reg_coeff = float(reg_coeff)
        self.normalize = bool(normalize)
        self.dtype = dtype

        self.direction: torch.Tensor | None = None  # shape (hidden_dim,)
        self.scaler_mean: torch.Tensor | None = None
        self.scaler_std: torch.Tensor | None = None

    # ----------------------------- training -------------------------------- #
    def fit(self, pos_acts: ArrayLike, neg_acts: ArrayLike) -> "ProbingClassifier":
        """Fit the probe on positive/negative token activations.

        Parameters:
        pos_acts: (n_pos, hidden_dim)
        neg_acts: (n_neg, hidden_dim)

        Returns:
        self
        """
        X_pos = _to_numpy(pos_acts, self.dtype)
        X_neg = _to_numpy(neg_acts, self.dtype)

        if X_pos.ndim != 2 or X_neg.ndim != 2:
            raise ValueError(f"Activations must be 2-D: (n_tokens, hidden_dim). Got {X_pos.shape} and {X_neg.shape}.")

        X = np.vstack([X_pos, X_neg])
        y = np.concatenate(
            [
                np.ones(X_pos.shape[0], dtype=np.int64),
                np.zeros(X_neg.shape[0], dtype=np.int64),
            ]
        )

        # Normalisation ---------------------------------------------------- #
        if self.normalize:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)  # type: ignore[arg-type]

            self.scaler_mean = torch.tensor(scaler.mean_, dtype=self.dtype)
            self.scaler_std = torch.tensor(scaler.scale_, dtype=self.dtype)
        else:
            # Identity transform
            self.scaler_mean = torch.zeros(X.shape[1], dtype=self.dtype)
            self.scaler_std = torch.ones(X.shape[1], dtype=self.dtype)

        # Logistic regression --------------------------------------------- #
        model = LogisticRegression(
            C=1.0 / self.reg_coeff,
            fit_intercept=False,
            random_state=42,
            solver="liblinear",
        )
        model.fit(X, y)  # type: ignore[arg-type]

        self.direction = torch.tensor(model.coef_.reshape(-1), dtype=self.dtype)
        return self

    # ------------------------------ scoring -------------------------------- #
    @torch.no_grad()
    def score_tokens(self, hidden_states: ArrayLike) -> torch.Tensor:
        """
        Get per-token *logit* scores for a single sequence.

        Parameters:
        hidden_states : (seq_len, hidden_dim)

        Returns:
        torch.Tensor
            1-D tensor of shape (seq_len,) where higher → more 'positive'.
        """
        if self.direction is None:
            raise RuntimeError("Probe not trained – call .fit(...) first.")

        hs = _to_torch(hidden_states, self.dtype)

        if hs.ndim != 2:
            raise ValueError(f"hidden_states must be (seq_len, hidden_dim); got {hs.shape}.")

        if self.normalize:
            hs = (hs - self.scaler_mean) / self.scaler_std  # type: ignore[arg-type]

        return torch.einsum("ih,h->i", hs, self.direction)  # (seq_len,)

    def evaluate(self, pos_acts: ArrayLike, neg_acts: ArrayLike) -> dict[str, float]:
        """
        Evaluate the probe on test data.

        Parameters:
        pos_acts: (n_pos, hidden_dim) - positive test activations
        neg_acts: (n_neg, hidden_dim) - negative test activations

        Returns:
        Dict[str, float]
            Dictionary containing evaluation metrics (accuracy, precision, recall, f1, roc_auc)
        """
        if self.direction is None:
            raise RuntimeError("Probe not trained – call .fit(...) first.")

        # Get scores for test data
        pos_scores = self.score_tokens(pos_acts).cpu().numpy()
        neg_scores = self.score_tokens(neg_acts).cpu().numpy()

        # Create labels and predictions
        y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
        y_scores = np.concatenate([pos_scores, neg_scores])
        y_pred = (y_scores > 0).astype(int)  # threshold at 0

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_scores),
        }

        return metrics

    # ----------------------------- I/O helpers ---------------------------- #
    def save(self, file: str | Path) -> None:
        """Pickle the probe to *file*."""
        payload = {
            "direction": self.direction.cpu() if self.direction is not None else None,
            "scaler_mean": self.scaler_mean.cpu() if self.scaler_mean is not None else None,
            "scaler_std": self.scaler_std.cpu() if self.scaler_std is not None else None,
            "reg_coeff": self.reg_coeff,
            "normalize": self.normalize,
            "dtype": self.dtype,
        }
        with open(file, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, file: str | Path) -> "ProbingClassifier":
        """Load a probe previously saved with :pymeth:`save`."""
        with open(file, "rb") as f:
            payload = pickle.load(f)

        obj = cls(
            reg_coeff=payload["reg_coeff"],
            normalize=payload["normalize"],
            dtype=payload["dtype"],
        )
        obj.direction = payload["direction"]
        obj.scaler_mean = payload["scaler_mean"]
        obj.scaler_std = payload["scaler_std"]
        return obj


class MultiClassProbingClassifier:
    """Multi-class linear probe trained with logistic regression."""

    def __init__(
        self,
        reg_coeff: float = 1e3,
        normalize: bool = False,
        fit_intercept: bool = True,
        dtype: torch.dtype = torch.float32,
        verbose: int = 0,
        class_weight: str | dict[str, float] | None = None,
    ) -> None:
        self.reg_coeff = float(reg_coeff)
        self.normalize = bool(normalize)
        self.fit_intercept = bool(fit_intercept)
        self.dtype = dtype
        self.verbose = int(verbose)
        self.class_weight = class_weight

        self.classifier: LogisticRegression | None = None
        self.scaler_mean: torch.Tensor | None = None
        self.scaler_std: torch.Tensor | None = None
        self.class_names: list[str] | None = None

        self.class_name_to_class_idx: dict[str, int] = {}

    # ----------------------------- training -------------------------------- #
    def fit(self, activations_dict: dict[str, ArrayLike]) -> "MultiClassProbingClassifier":
        """Fit the probe on multi-class activations.

        Parameters:
        activations_dict: Dictionary mapping class names to activations
                         e.g., {"wall": (n_wall, hidden_dim), "empty": (n_empty, hidden_dim), ...}

        Returns:
        self
        """
        # Prepare data using common utility
        X, y, self.class_names, self.class_name_to_class_idx = _prepare_multiclass_data(activations_dict, self.dtype)

        # Normalisation ---------------------------------------------------- #
        if self.normalize:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)  # type: ignore[arg-type]

            self.scaler_mean = torch.tensor(scaler.mean_, dtype=self.dtype)
            self.scaler_std = torch.tensor(scaler.scale_, dtype=self.dtype)
        else:
            # Identity transform
            self.scaler_mean = torch.zeros(X.shape[1], dtype=self.dtype)
            self.scaler_std = torch.ones(X.shape[1], dtype=self.dtype)

        # Multi-class logistic regression ---------------------------------- #
        self.classifier = LogisticRegression(
            C=1.0 / self.reg_coeff,
            fit_intercept=self.fit_intercept,
            random_state=42,
            solver="lbfgs",
            verbose=self.verbose,
            class_weight=self.class_weight,
        )
        self.classifier.fit(X, y)  # type: ignore[arg-type]

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
        if self.classifier is None:
            raise RuntimeError("Probe not trained – call .fit(...) first.")

        hs = _to_torch(hidden_states, self.dtype)

        if hs.ndim != 2:
            raise ValueError(f"hidden_states must be (seq_len, hidden_dim); got {hs.shape}.")

        if self.normalize:
            hs = (hs - self.scaler_mean) / self.scaler_std  # type: ignore[arg-type]

        # Convert to numpy for sklearn
        hs_np = hs.cpu().numpy()
        probabilities = self.classifier.predict_proba(hs_np)  # type: ignore[arg-type]

        return torch.tensor(probabilities, dtype=self.dtype)

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
        if self.classifier is None:
            raise RuntimeError("Probe not trained – call .fit(...) first.")

        hs = _to_torch(hidden_states, self.dtype)

        if hs.ndim != 2:
            raise ValueError(f"hidden_states must be (seq_len, hidden_dim); got {hs.shape}.")

        if self.normalize:
            hs = (hs - self.scaler_mean) / self.scaler_std  # type: ignore[arg-type]

        # Convert to numpy for sklearn
        hs_np = hs.cpu().numpy()
        predictions = self.classifier.predict(hs_np)  # type: ignore[arg-type]

        return predictions

    def evaluate(self, activations_dict: dict[str, ArrayLike]) -> dict[str, float]:
        """
        Evaluate the probe on test data.

        Parameters:
        activations_dict: Dictionary mapping class names to test activations

        Returns:
        Dict[str, float]
            Dictionary containing evaluation metrics
        """
        if self.classifier is None:
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
            "classifier": self.classifier,
            "scaler_mean": self.scaler_mean.cpu() if self.scaler_mean is not None else None,
            "scaler_std": self.scaler_std.cpu() if self.scaler_std is not None else None,
            "reg_coeff": self.reg_coeff,
            "normalize": self.normalize,
            "fit_intercept": self.fit_intercept,
            "dtype": self.dtype,
            "verbose": self.verbose,
            "class_weight": self.class_weight,
            "class_names": self.class_names,
            "class_name_to_class_idx": self.class_name_to_class_idx,
        }
        with open(file, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, file: str | Path) -> "MultiClassProbingClassifier":
        """Load a probe previously saved with :pymeth:`save`."""
        with open(file, "rb") as f:
            payload = pickle.load(f)

        obj = cls(
            reg_coeff=payload["reg_coeff"],
            normalize=payload["normalize"],
            fit_intercept=payload["fit_intercept"],
            dtype=payload["dtype"],
            verbose=payload.get("verbose", 0),  # Default to 0 for backward compatibility
            class_weight=payload.get("class_weight", None),  # Default to None for backward compatibility
        )
        obj.classifier = payload["classifier"]
        obj.scaler_mean = payload["scaler_mean"]
        obj.scaler_std = payload["scaler_std"]
        obj.class_names = payload["class_names"]
        obj.class_name_to_class_idx = payload["class_name_to_class_idx"]
        return obj


def load_activations(path) -> torch.Tensor:
    """
    Load activations from a .npy, .pt, or .pth file. Accepts str or Path.
    Returns the activations tensor, handling both old and new formats.
    """
    if not isinstance(path, Path):
        path = Path(path)
    if path.suffix == ".npy":
        return torch.from_numpy(np.load(path))
    if path.suffix in {".pt", ".pth"}:
        # First try loading with weights_only=False to handle dict format
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(data, dict) and "activations" in data:
                # New format with metadata
                return data["activations"]
            elif isinstance(data, torch.Tensor):
                # Old direct tensor format
                return data
            else:
                # Try with weights_only=True for safety
                return torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            # Fallback to weights_only=True
            return torch.load(path, map_location="cpu", weights_only=True)
    raise ValueError(f"Unsupported file format: {path}")


def _to_numpy(x: ArrayLike, dtype: torch.dtype = torch.float32) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().to(dtype).cpu().numpy()
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    raise TypeError(f"Unsupported type {type(x)}; expected torch.Tensor or np.ndarray.")


def _to_torch(x: ArrayLike, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype).cpu()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(dtype)
    raise TypeError(f"Unsupported type {type(x)}; expected torch.Tensor or np.ndarray.")


def train_and_save_probe(
    positive_acts: str,
    negative_acts: str,
    layer: int,
    output_dir: str,
    eval_split: float = 0.2,
    reg_coeff: float = 1e3,
    normalize: bool = True,
) -> str:
    """Train a probing classifier on a dataset of activations.

    Args:
        positive_acts: Path to the positive activations. Expecting a .pt, .pth or .npy file of shape (num_positive_samples, num_layers, hidden_dim)
        negative_acts: Path to the negative activations. Expecting a .pt, .pth or .npy file of shape (num_negative_samples, num_layers, hidden_dim)
        layer: Layer to train the probe on.
        output_dir: Directory to save the probe.

    Returns:
    """

    pos_acts_all_layers = load_activations(positive_acts)
    neg_acts_all_layers = load_activations(negative_acts)

    pos_acts = pos_acts_all_layers[:, layer, :]  # (num_positive_samples, hidden_dim)
    neg_acts = neg_acts_all_layers[:, layer, :]  # (num_negative_samples, hidden_dim)
    n_pos = pos_acts.shape[0]
    n_neg = neg_acts.shape[0]
    n_pos_eval = int(n_pos * eval_split)
    n_neg_eval = int(n_neg * eval_split)
    pos_train = pos_acts[:-n_pos_eval]
    pos_eval = pos_acts[-n_pos_eval:]
    neg_train = neg_acts[:-n_neg_eval]
    neg_eval = neg_acts[-n_neg_eval:]

    print("🔢 Data split:")
    print(f"   Training:   {len(pos_train)} positive, {len(neg_train)} negative")
    print(f"   Evaluation: {len(pos_eval)} positive, {len(neg_eval)} negative")

    probe = ProbingClassifier(
        reg_coeff=reg_coeff,
        normalize=normalize,
    ).fit(pos_train, neg_train)

    eval_metrics = probe.evaluate(pos_eval, neg_eval)
    print("\n📊 Evaluation Results:")
    for metric, value in eval_metrics.items():
        print(f"  {metric}: {value:.4f}")

    output_path = Path(output_dir) / f"probe_layer_{layer}.pkl"
    probe.save(output_path)
    return output_path


def apply_probe_on_prompt_response(
    model: nnsight.LanguageModel,
    probe: ProbingClassifier,
    prompt: str,
    response: str,
    layer: int,
    token_position: act_module.TokenPosition,
) -> float:
    activations = act_module.run_model_and_gather_activations_at_token_position(
        model, prompt, response, token_position
    )

    layer_activations = activations[layer, :].unsqueeze(0)  # (1, hidden_dim,)

    score = probe.score_tokens(layer_activations)

    return score.item()


def train_and_save_multiclass_probe(
    activations_dir: str,
    layer: int,
    output_dir: str = None,
    eval_split: float = 0.2,
    reg_coeff: float = 1e3,
    normalize: bool = False,
    fit_intercept: bool = True,
    verbose: int = 0,
    class_weight: str | dict[str, float] | None = None,
) -> str:
    """Train a multi-class probing classifier on grid cell activations.

    Args:
        activations_dir: Directory containing activation files (acts_wall.pt, acts_empty.pt, etc.)
        layer: Layer to train the probe on (not used for single-layer activations)
        output_dir: Directory to save the probe
        eval_split: Fraction of data to use for evaluation
        reg_coeff: Regularization coefficient
        normalize: Whether to normalize activations
        fit_intercept: Whether to fit an intercept term in the logistic regression
        verbose: Verbosity level for the logistic regression solver (0 = silent, 1+ = verbose)
        class_weight: Class weights for handling class imbalance ('balanced' or dict mapping class names to weights)

    Returns:
        Path to the saved probe
    """
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
        eval_dict[class_name] = acts[-n_eval:] if n_eval > 0 else acts[:0]  # Empty tensor if no eval data

    print("\n🔢 Data split:")
    for class_name in activations_dict.keys():
        train_count = train_dict[class_name].shape[0]
        eval_count = eval_dict[class_name].shape[0]
        print(f"   {class_name}: {train_count} train, {eval_count} eval")

    # Train the probe
    probe = MultiClassProbingClassifier(
        reg_coeff=reg_coeff,
        normalize=normalize,
        fit_intercept=fit_intercept,
        verbose=verbose,
        class_weight=class_weight,
    ).fit(train_dict)

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
    suffix_parts = []
    if not fit_intercept:
        suffix_parts.append("no_intercept")
    if normalize:
        suffix_parts.append("normalized")

    suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
    output_path = output_dir / f"multiclass_probe_layer_{layer}{suffix}.pkl"
    probe.save(output_path)

    print(f"\n💾 Probe saved to {output_path}")
    return str(output_path)
