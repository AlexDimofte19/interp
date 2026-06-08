"""Train next-action probing classifiers on prepared EOS-token activations.

Consumes a v3 manifest produced by `prepare_activations_for_probing` with
`probe_type="next_action"`: each sample is a single gathered EOS-token activation
labeled with its trajectory's agent_action. Samples are treated as i.i.d. (plain
random train/eval split, no trajectory grouping).
"""

from pathlib import Path
from typing import Literal

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from telos_interp.commands.prepare_activations_for_probing.manifest_loader import (
    detect_format,
    load_next_action_compact,
    load_v3_manifest,
    resolve_manifest_path,
)
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
from telos_interp.training import (
    train_epoch as _train_epoch,
)

ClassWeight = Literal["balanced"] | None

# Canonical action id -> name for human-readable metrics. Mirrors NEXT_ACTION_TO_ID
# in prepare_activations_for_probing_fn.py (UP is the canonical name for id 1).
ACTION_ID_TO_NAME = {0: "LEFT", 1: "UP", 2: "RIGHT", 3: "DOWN"}


class NextActionProbe:
    """A trained next-action probe with built-in normalization and prediction.

    Encapsulates a trained classifier (logistic regression or MLP) plus optional
    input-normalization parameters and the original-label <-> contiguous-index
    mappings, for simple inference. Analogous to CognitiveMapProbe but flat (one
    activation -> one action label; no grid positions).
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
        """Get class probabilities for activations of shape (N, input_dim)."""
        self.model.eval()
        x = activations.float().to(self.device)
        x = self._normalize(x)
        logits = self.model(x)
        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict(self, activations: torch.Tensor) -> torch.Tensor:
        """Get predicted original action IDs (not internal indices) for activations."""
        probs = self.predict_proba(activations)
        pred_indices = torch.argmax(probs, dim=-1)
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
    def load(cls, path: str | Path, device: str | None = None) -> "NextActionProbe":
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

    remapped_labels = torch.tensor([label_to_idx[label.item()] for label in labels], dtype=labels.dtype)
    return remapped_labels, num_classes, label_to_idx, idx_to_label


def _get_balanced_indices(
    labels: torch.Tensor,
    seed: int = 42,
    per_class_max_count: int | None = None,
) -> torch.Tensor:
    """Return shuffled flat indices that balance classes (v3 index-based balancer).

    Upsamples minority classes (and downsamples classes above per_class_max_count)
    to a common per-class count, without materializing the activations.
    """
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


def _compute_class_weights(
    labels: torch.Tensor,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute balanced class weights for CrossEntropyLoss: n / (n_classes * count)."""
    class_weights = torch.zeros(num_classes, dtype=torch.float32)
    unique_classes, class_counts = torch.unique(labels, return_counts=True)
    n_samples = len(labels)

    for class_id, count in zip(unique_classes, class_counts, strict=False):
        class_weights[class_id] = n_samples / (num_classes * count.float())

    return class_weights.to(device)


def _evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> dict:
    """Evaluate model and return metrics (loss, accuracy, balanced accuracy, per-class)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    class_tp = torch.zeros(num_classes)
    class_gt_support = torch.zeros(num_classes)
    class_pred_count = torch.zeros(num_classes)

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

            for i in range(num_classes):
                gt_mask = batch_y == i
                pred_mask = predicted == i
                class_gt_support[i] += gt_mask.sum().item()
                class_pred_count[i] += pred_mask.sum().item()
                class_tp[i] += (gt_mask & pred_mask).sum().item()

    per_class_metrics = {}
    for i in range(num_classes):
        tp = class_tp[i].item()
        gt_support = class_gt_support[i].item()
        pred_count = class_pred_count[i].item()

        precision = tp / pred_count if pred_count > 0 else 0.0
        recall = tp / gt_support if gt_support > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = tp / gt_support if gt_support > 0 else 0.0

        per_class_metrics[i] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "gt_support": int(gt_support),
            "predicted": int(pred_count),
        }

    per_class_accuracies = [m["accuracy"] for m in per_class_metrics.values() if m["gt_support"] > 0]
    balanced_accuracy = sum(per_class_accuracies) / len(per_class_accuracies) if per_class_accuracies else 0.0

    return {
        "loss": total_loss / num_batches if num_batches > 0 else 0.0,
        "accuracy": correct / total if total > 0 else 0.0,
        "balanced_accuracy": balanced_accuracy,
        "per_class_metrics": per_class_metrics,
        "num_samples": total,
    }


def _load_and_preprocess_train_data(train_path: Path, verbose: bool) -> dict:
    """Load a v3 next_action manifest and filter per-sample NaN activations.

    Returns a dict with keys: activations (N, D), labels (N,).
    """
    if detect_format(train_path) != 3:
        raise ValueError(
            f"Expected a v3 manifest directory for next_action, but {train_path} is not one."
        )

    manifest_path = resolve_manifest_path(train_path)
    manifest = load_v3_manifest(manifest_path)

    probe_type = manifest.get("probe_type")
    if probe_type != "next_action":
        raise ValueError(
            f"Expected probe_type='next_action', got '{probe_type}'. "
            "This command only supports next_action probe data."
        )

    if verbose:
        print(f"Loading training data from {train_path}")

    compact = load_next_action_compact(manifest, manifest_path)
    activations = compact["base_act"]
    labels = compact["labels"]

    if verbose:
        print(f"Loaded {activations.shape[0]} token samples")
        print(f"Activation dimension: {activations.shape[1]}")
        print(f"Number of unique labels: {labels.unique().shape[0]}")

    nan_mask = torch.isnan(activations).any(dim=1)
    num_nan = nan_mask.sum().item()
    if num_nan > 0:
        if verbose:
            print(
                f"WARNING: Found {num_nan} samples with NaN values "
                f"({100 * num_nan / len(activations):.1f}%), filtering them out"
            )
        activations = activations[~nan_mask]
        labels = labels[~nan_mask]

    return {"activations": activations, "labels": labels}


def _print_final_results(final_results: dict, idx_to_label: dict[int, int]) -> None:
    """Print final evaluation results including the per-class (per-action) metrics table."""
    print(f"Accuracy: {final_results['accuracy']:.4f}")
    print(f"Balanced Accuracy: {final_results['balanced_accuracy']:.4f}")
    print(f"Loss: {final_results['loss']:.4f}")
    print(f"Number of samples: {final_results['num_samples']}")

    print("\nPer-class metrics:")
    print("-" * 87)
    print(
        f"{'Action':<10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} "
        f"{'F1-Score':>10} {'GT Support':>12} {'Predicted':>10}"
    )
    print("-" * 87)
    for class_idx in sorted(final_results["per_class_metrics"].keys()):
        metrics = final_results["per_class_metrics"][class_idx]
        original_label = idx_to_label[class_idx]
        name = ACTION_ID_TO_NAME.get(original_label, str(original_label))
        print(
            f"{name:<10} {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
            f"{metrics['f1']:>10.4f} {metrics['gt_support']:>12} {metrics['predicted']:>10}"
        )
    print("-" * 87)


def train_next_action_probe(
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
    per_class_max_count: int | None = None,
) -> NextActionProbe:
    """Train a next-action probing classifier on prepared EOS-token activations.

    Takes a v3 manifest directory produced by prepare_activations_for_probing with
    probe_type=next_action and trains either a logistic regression or MLP classifier to
    predict the agent's next action from a single EOS-token activation. Each token is an
    i.i.d. sample, so subsetting and the train/eval split are plain random splits.

    Args:
        train_data_path: Path to the v3 manifest directory (from
            prepare_activations_for_probing with probe_type=next_action).
        model_type: "lr" (logistic regression) or "mlp".
        eval_data_path: Optional separate v3 next_action manifest dir for evaluation.
            If not provided, uses eval_split from the training data.
        output_path: Where to save the trained probe. Defaults to the train data dir.
        num_epochs: Number of training epochs.
        learning_rate: Learning rate for AdamW.
        batch_size: Batch size for training.
        weight_decay: L2 regularization weight decay.
        hidden_dims: Comma-separated hidden layer dims for MLP (e.g., "512,256").
        dropout: Dropout rate for MLP (ignored for lr).
        eval_split: Fraction of samples used for validation (if eval_data_path is None).
        subset: Fraction of samples to use (0.0 to 1.0), applied before the split.
        class_weight: None or "balanced" (inverse-frequency weights in the loss).
        balance_classes: If True, upsample minority classes (v3 index-based balancer).
        normalize: If True, normalize activations with train-set mean/std (saved with probe).
        device: "cuda"/"cpu"; defaults to CUDA if available.
        seed: Random seed.
        verbose: Print training progress.
        per_class_max_count: Cap per-class samples when balancing.

    Returns:
        Trained NextActionProbe instance (also saved to output_path).
    """
    set_seed(seed)
    torch_device = resolve_device(device)
    print(f"Using device: {torch_device}")

    train_path = Path(train_data_path)
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not 0.0 < subset <= 1.0:
        raise ValueError(f"subset must be in (0.0, 1.0], got {subset}")

    load_result = _load_and_preprocess_train_data(train_path, verbose)
    activations = load_result["activations"]
    labels = load_result["labels"]

    # Subset (i.i.d. random sample of rows).
    num_samples = activations.shape[0]
    num_to_keep = max(1, int(num_samples * subset))
    perm = torch.randperm(num_samples)[:num_to_keep]
    activations = activations[perm]
    labels = labels[perm]
    print(f"Subset: keeping {num_to_keep}/{num_samples} samples ({subset * 100:.1f}%)")

    # Remap labels to contiguous indices.
    remapped_labels, num_classes, label_to_idx, idx_to_label = _remap_labels(labels, verbose)

    # Train/eval split (plain random, i.i.d.).
    if eval_data_path is None:
        indices = torch.randperm(activations.shape[0])
        split_idx = int(activations.shape[0] * (1 - eval_split))
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
        eval_load = _load_and_preprocess_train_data(eval_path, verbose)
        eval_activations = eval_load["activations"]
        eval_labels_raw = eval_load["labels"]
        # Remap eval labels with the training mapping; drop labels unseen in training.
        keep = [label.item() in label_to_idx for label in eval_labels_raw]
        keep_tensor = torch.tensor(keep)
        if not keep_tensor.all() and verbose:
            print(f"WARNING: Skipping {(~keep_tensor).sum().item()} eval samples with unseen labels")
        eval_activations = eval_activations[keep_tensor]
        eval_labels = torch.tensor(
            [label_to_idx[label.item()] for label in eval_labels_raw if label.item() in label_to_idx],
            dtype=torch.int64,
        )

    # Normalization (computed from train set, applied to both).
    scaler_mean = None
    scaler_std = None
    if normalize:
        scaler_mean, scaler_std = compute_normalization_params(train_activations)
        train_activations = normalize_activations(train_activations, scaler_mean, scaler_std)
        eval_activations = normalize_activations(eval_activations, scaler_mean, scaler_std)
        print(f"Normalization enabled: computed mean/std from {train_activations.shape[0]} training samples")

    # Class balancing (v3 index-based balancer).
    if balance_classes:
        print("Balancing classes by upsampling...")
        unique, counts = torch.unique(train_labels, return_counts=True)
        print(f"  Before: {dict(zip(unique.tolist(), counts.tolist(), strict=False))}")
        balanced_indices = _get_balanced_indices(train_labels, seed=seed, per_class_max_count=per_class_max_count)
        train_activations = train_activations[balanced_indices]
        train_labels = train_labels[balanced_indices]
        unique, counts = torch.unique(train_labels, return_counts=True)
        print(f"  After: {dict(zip(unique.tolist(), counts.tolist(), strict=False))}")
        print(f"  New training set size: {train_activations.shape[0]}")

    input_dim = train_activations.shape[1]
    hidden_dims_list = [int(d.strip()) for d in hidden_dims.split(",") if d.strip()]

    train_dataset = TensorDataset(train_activations.float(), train_labels.long())
    eval_dataset = TensorDataset(eval_activations.float(), eval_labels.long())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

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

    if class_weight == "balanced":
        print("Using balanced class weights in loss function")
        weights = _compute_class_weights(train_labels, num_classes, torch_device)
        print(f"  Class weights: {weights.tolist()}")
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_eval_accuracy = 0.0
    best_balanced_accuracy = 0.0

    print(f"\nTraining for {num_epochs} epochs...")
    epoch_iterator = tqdm(range(num_epochs), desc="Training", disable=not verbose)
    for _ in epoch_iterator:
        train_loss = _train_epoch(model, train_loader, criterion, optimizer, torch_device)
        eval_results = _evaluate(model, eval_loader, criterion, torch_device, num_classes)
        best_eval_accuracy = max(best_eval_accuracy, eval_results["accuracy"])
        best_balanced_accuracy = max(best_balanced_accuracy, eval_results["balanced_accuracy"])
        epoch_iterator.set_postfix(
            {
                "loss": f"{train_loss:.4f}",
                "eval_acc": f"{eval_results['accuracy']:.4f}",
                "bal_acc": f"{eval_results['balanced_accuracy']:.4f}",
                "best_acc": f"{best_eval_accuracy:.4f}",
                "best_bal": f"{best_balanced_accuracy:.4f}",
            }
        )

    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    final_results = _evaluate(model, eval_loader, criterion, torch_device, num_classes)
    if verbose:
        _print_final_results(final_results, idx_to_label)

    per_class_metrics_original = {
        idx_to_label[idx]: metrics for idx, metrics in final_results["per_class_metrics"].items()
    }

    probe = NextActionProbe(
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
            "best_balanced_accuracy": best_balanced_accuracy,
            "final_accuracy": final_results["accuracy"],
            "final_balanced_accuracy": final_results["balanced_accuracy"],
            "final_loss": final_results["loss"],
            "per_class_metrics": per_class_metrics_original,
        },
        device=torch_device,
    )

    if output_path is None:
        final_output_path = train_path.parent / f"next_action_probe_{model_type}.pt"
    else:
        final_output_path = Path(output_path)
    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    probe.save(final_output_path)

    print(f"\nModel saved to {final_output_path}")
    print(f"Best evaluation accuracy: {best_eval_accuracy:.4f}")
    print(f"Best balanced accuracy: {best_balanced_accuracy:.4f}")

    return probe
