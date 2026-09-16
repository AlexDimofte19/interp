"""Train one-vs-rest binary cognitive map probes on prepared activations.

Branched from `train_cognitive_map_probe_fn.py`, which trains a single N-way softmax over
whatever cell ids a `grid_tile` manifest happens to contain. That answers "which symbol is
at (row, col)"; this answers "is (row, col) a wall", one probe per symbol, so per-class
readability is a number rather than a row of a confusion matrix.

Three things differ and nothing else does:

1. `_binarize_labels` replaces `_remap_labels`. The head is **always** 2-wide, so a class
   that happens to be absent from a split cannot silently shrink it -- which matters here,
   since `A` and `G` are one cell per grid and a small eval slice can hold none.
2. The saved probe carries the FULL cell-id map in `label_to_idx` (every id in
   `CELL_SYMBOL_TO_ID` -> 0 or 1), so any consumer that binarises ground truth by
   `label_to_idx.get(cell_id)` agrees with training. `idx_to_label` is `{0: -1, 1: pos_id}`,
   and `num_classes` reads off *that* -- reading it off `label_to_idx`, as the multiclass
   probe does, would say 8.
3. Metrics are the binary ones: balanced accuracy, positive-class precision/recall/F1, and
   AUROC. Raw accuracy is close to useless for `A`/`G` at ~0.9% positives.

Only v3 manifest directories are supported. The legacy v1 flat-`.pt` path is deliberately
not carried over -- every dataset this is meant for is v3, and the v1 branch would double
the surface for no user.
"""

from pathlib import Path
from typing import Literal

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from telos_interp.commands.prepare_activations_for_probing.manifest_loader import (
    GridTileCompactDataset,
    IndexedGridTileCompactDataset,
    detect_format,
)
from telos_interp.commands.train_cognitive_map_probe.train_cognitive_map_probe_fn import (
    _compute_class_weights,
    _get_balanced_indices,
    _load_and_preprocess_train_data_v3,
    _load_separate_eval_data_v3,
)
from telos_interp.grid_utils import CELL_SYMBOL_TO_ID
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

#: `idx_to_label[0]` for a binary probe. Not a cell id -- the negative class is "every
#: symbol but one", which no single id names. Chosen outside `CELL_SYMBOL_TO_ID`'s range so
#: a consumer that looks it up in `CELL_ID_TO_SYMBOL` fails loudly instead of printing `A`.
NEGATIVE_LABEL = -1

#: Readable aliases for the grid symbols, so a shell loop can say `wall` instead of
#: quoting `#` (a comment to the shell) or `_`.
POSITIVE_CLASS_ALIASES = {
    "agent": "A",
    "wall": "#",
    "goal": "G",
    "empty": "_",
    "door": "D",
    "key": "K",
    "unknown": "?",
    "padding": "+",
}

#: The inverse, for filenames: `#` and `?` are not usable in a path a shell will expand.
POSITIVE_CLASS_SLUGS = {symbol: alias for alias, symbol in POSITIVE_CLASS_ALIASES.items()}


def resolve_positive_class(positive_class: str) -> tuple[str, int]:
    """Resolve a `--positive-class` spec to its (grid symbol, cell id).

    Accepts either the symbol itself or its english alias.

    Args:
        positive_class: Grid symbol ("#") or alias ("wall"), case-insensitive for aliases.

    Returns:
        Tuple of (symbol, cell id).

    Raises:
        ValueError: If the spec names neither a known symbol nor a known alias.

    Example:
        >>> resolve_positive_class("wall")
        ('#', 1)
        >>> resolve_positive_class("_")
        ('_', 3)
    """
    key = positive_class.strip()
    symbol = POSITIVE_CLASS_ALIASES.get(key.lower(), key)
    if symbol not in CELL_SYMBOL_TO_ID:
        known = ", ".join(f"{alias}={sym}" for alias, sym in POSITIVE_CLASS_ALIASES.items())
        raise ValueError(f"Unknown positive class {positive_class!r}. Pass a grid symbol or one of: {known}")
    return symbol, CELL_SYMBOL_TO_ID[symbol]


def binary_label_maps(positive_cell_id: int) -> tuple[dict[int, int], dict[int, int]]:
    """The label maps a binary probe carries.

    `label_to_idx` covers **every** cell id rather than only the ones seen in training, so
    a consumer binarising ground truth off a probe never meets a missing key and never has
    to know which symbols the training split happened to contain.

    Args:
        positive_cell_id: The cell id that maps to class 1.

    Returns:
        Tuple of (label_to_idx, idx_to_label).
    """
    label_to_idx = {cell_id: int(cell_id == positive_cell_id) for cell_id in sorted(CELL_SYMBOL_TO_ID.values())}
    idx_to_label = {0: NEGATIVE_LABEL, 1: positive_cell_id}
    return label_to_idx, idx_to_label


def _binarize_labels(labels: torch.Tensor, positive_cell_id: int) -> torch.Tensor:
    """Collapse cell ids to {0, 1} against one positive id, preserving shape."""
    return (labels == positive_cell_id).to(labels.dtype)


def binary_auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Rank-based AUROC with tie correction, vectorised.

    The Mann-Whitney form, so it never sorts thresholds and never allocates a curve; ties
    get their average rank, which is what a probe with saturated probabilities produces a
    lot of. Returns NaN when one class is absent -- AUROC is undefined there, and 0.5 would
    read as "chance" when the truth is "unmeasurable".

    Args:
        scores: (N,) positive-class scores.
        labels: (N,) binary labels in {0, 1}.

    Returns:
        Area under the ROC curve, or NaN if either class has no rows.

    Example:
        >>> round(binary_auroc(torch.tensor([0.1, 0.4, 0.35, 0.8]), torch.tensor([0, 0, 1, 1])), 4)
        0.75
    """
    scores = scores.detach().double().flatten()
    labels = labels.detach().flatten()
    num_positive = int((labels == 1).sum().item())
    num_negative = int((labels == 0).sum().item())
    if num_positive == 0 or num_negative == 0:
        return float("nan")

    order = torch.argsort(scores)
    _, counts = torch.unique_consecutive(scores[order], return_counts=True)
    ends = torch.cumsum(counts, dim=0)
    starts = ends - counts + 1
    average_rank = (starts + ends).double() / 2.0
    ranks = torch.repeat_interleave(average_rank, counts)

    positive_rank_sum = ranks[labels[order] == 1].sum().item()
    return (positive_rank_sum - num_positive * (num_positive + 1) / 2.0) / (num_positive * num_negative)


def binary_metrics(labels: torch.Tensor, predictions: torch.Tensor, scores: torch.Tensor | None = None) -> dict:
    """Confusion counts and the derived rates for one binary readout.

    `balanced_accuracy` is the mean of recall and specificity, averaged only over the
    classes that actually have rows -- the same convention as the multiclass trainer, and
    the number to read when positives are 0.9% of the data.

    Args:
        labels: (N,) ground-truth labels in {0, 1}.
        predictions: (N,) predicted labels in {0, 1}.
        scores: Optional (N,) positive-class scores, for AUROC.

    Returns:
        Dict of counts and rates.
    """
    labels = labels.detach().flatten()
    predictions = predictions.detach().flatten()
    positive_truth = labels == 1
    positive_prediction = predictions == 1

    true_positive = int((positive_truth & positive_prediction).sum().item())
    false_positive = int((~positive_truth & positive_prediction).sum().item())
    true_negative = int((~positive_truth & ~positive_prediction).sum().item())
    false_negative = int((positive_truth & ~positive_prediction).sum().item())
    n_rows = labels.numel()

    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
    specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0.0
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    per_class = []
    if true_positive + false_negative > 0:
        per_class.append(recall)
    if true_negative + false_positive > 0:
        per_class.append(specificity)
    balanced_accuracy = sum(per_class) / len(per_class) if per_class else 0.0

    return {
        "n_rows": int(n_rows),
        "positive_support": true_positive + false_negative,
        "positive_rate": (true_positive + false_negative) / n_rows if n_rows > 0 else 0.0,
        "predicted_positive": true_positive + false_positive,
        "accuracy": (true_positive + true_negative) / n_rows if n_rows > 0 else 0.0,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "auroc": binary_auroc(scores, labels) if scores is not None else float("nan"),
        "tp": true_positive,
        "fp": false_positive,
        "tn": true_negative,
        "fn": false_negative,
    }


class BinaryCognitiveMapProbe:
    """A trained one-vs-rest cognitive map probe.

    Mirrors `CognitiveMapProbe`'s save/load dict so the two are interchangeable to anything
    that only reads weights and normalisation, and adds `positive_class` /
    `positive_cell_id` so a probe on disk names what it decides without a filename
    convention.
    """

    def __init__(
        self,
        model: nn.Module,
        model_type: ModelType,
        input_dim: int,
        positive_class: str,
        positive_cell_id: int,
        hidden_dims: list[int] | None = None,
        dropout: float | None = None,
        scaler_mean: torch.Tensor | None = None,
        scaler_std: torch.Tensor | None = None,
        config: dict | None = None,
        results: dict | None = None,
        device: torch.device | str | None = None,
    ):
        """Initialize a BinaryCognitiveMapProbe.

        Args:
            model: The trained nn.Module (LogisticRegressionProbe or MLPProbe, 2-wide head)
            model_type: Type of model ("lr" or "mlp")
            input_dim: Input dimension of the model (activation dim + 2 position columns)
            positive_class: The grid symbol this probe decides ("#", "_", "A", "G", ...)
            positive_cell_id: That symbol's id in `CELL_SYMBOL_TO_ID`
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

        self.positive_class = positive_class
        self.positive_cell_id = positive_cell_id
        self.label_to_idx, self.idx_to_label = binary_label_maps(positive_cell_id)

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
        """Always 2. Read off `idx_to_label`; `label_to_idx` covers every cell id."""
        return len(self.idx_to_label)

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
        """Get (N, 2) class probabilities, column 1 being the positive class.

        Args:
            activations: Input tensor of shape (N, input_dim)

        Returns:
            Tensor of shape (N, 2) with class probabilities
        """
        self.model.eval()
        x = activations.float().to(self.device)
        x = self._normalize(x)
        return torch.softmax(self.model(x), dim=-1)

    @torch.no_grad()
    def predict_positive_proba(self, activations: torch.Tensor) -> torch.Tensor:
        """Get (N,) probability of the positive class.

        Args:
            activations: Input tensor of shape (N, input_dim)

        Returns:
            Tensor of shape (N,)
        """
        return self.predict_proba(activations)[:, 1]

    @torch.no_grad()
    def predict(self, activations: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Get predicted original label IDs (`positive_cell_id` or `NEGATIVE_LABEL`).

        Args:
            activations: Input tensor of shape (N, input_dim)
            threshold: Positive-class probability above which a cell is called positive

        Returns:
            Tensor of shape (N,) with original label IDs
        """
        positive = self.predict_positive_proba(activations) > threshold
        return torch.where(
            positive,
            torch.full_like(positive, self.positive_cell_id, dtype=torch.long),
            torch.full_like(positive, NEGATIVE_LABEL, dtype=torch.long),
        )

    def save(self, path: str | Path) -> None:
        """Save the probe to a file.

        Args:
            path: Destination .pt path.
        """
        save_data = {
            "model_state_dict": self.model.state_dict(),
            "model_type": self.model_type,
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "positive_class": self.positive_class,
            "positive_cell_id": self.positive_cell_id,
            "label_to_idx": self.label_to_idx,
            "idx_to_label": self.idx_to_label,
            "scaler_mean": self.scaler_mean.cpu() if self.scaler_mean is not None else None,
            "scaler_std": self.scaler_std.cpu() if self.scaler_std is not None else None,
            "config": self.config,
            "results": self.results,
        }
        torch.save(save_data, path)

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "BinaryCognitiveMapProbe":
        """Load a probe from a file.

        Args:
            path: Path to a .pt written by `save`.
            device: Device to place the model on; CUDA if available when omitted.

        Returns:
            The loaded probe.

        Raises:
            ValueError: If the file is a multiclass probe (no `positive_class`).
        """
        device_str = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        torch_device = torch.device(device_str)

        data = torch.load(path, map_location=torch_device, weights_only=False)
        if "positive_class" not in data:
            raise ValueError(
                f"{path} has no 'positive_class': it is a multiclass cognitive map probe. "
                "Load it with CognitiveMapProbe instead."
            )

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
            positive_class=data["positive_class"],
            positive_cell_id=data["positive_cell_id"],
            hidden_dims=data["hidden_dims"],
            dropout=data["dropout"],
            scaler_mean=data.get("scaler_mean"),
            scaler_std=data.get("scaler_std"),
            config=data.get("config"),
            results=data.get("results"),
            device=torch_device,
        )


def _print_binary_debug_grid(
    positions: torch.Tensor,
    true_labels: torch.Tensor,
    pred_labels: torch.Tensor,
    symbol: str,
) -> None:
    """Print one trajectory's grid twice: where the class is, and where the probe says it is.

    The multiclass trainer's equivalent prints cell symbols; a binary probe only ever has
    two things to say, so positive cells show the symbol and negative cells a dot. Cells the
    manifest did not sample stay `·`.

    Args:
        positions: (n, 2) [row, col] per sampled cell, un-normalized.
        true_labels: (n,) ground truth in {0, 1}.
        pred_labels: (n,) predictions in {0, 1}.
        symbol: The grid symbol this probe decides.
    """
    positions = positions.int()
    grid_size = max(positions[:, 0].max().item(), positions[:, 1].max().item()) + 1

    true_grid = [["·" for _ in range(grid_size)] for _ in range(grid_size)]
    pred_grid = [["·" for _ in range(grid_size)] for _ in range(grid_size)]

    correct = 0
    for i in range(len(positions)):
        row, col = positions[i].tolist()
        is_true = bool(true_labels[i].item())
        is_pred = bool(pred_labels[i].item())
        true_grid[row][col] = symbol if is_true else "."
        pred_grid[row][col] = symbol if is_pred else "."
        correct += int(is_true == is_pred)

    width = grid_size * 2 - 1
    print("\n" + "=" * 60)
    print(f"DEBUG: Single Trajectory Grid Comparison ('{symbol}' vs rest)")
    print(f"(grid size {grid_size}x{grid_size}, {len(positions)} sampled cells)")
    print("=" * 60)
    print(f"\n{'Observation (Ground Truth)':<{width + 5}}  {'Prediction'}")
    print("-" * width + "     " + "-" * width)
    for row_idx in range(grid_size):
        print(f"{' '.join(true_grid[row_idx])}     {' '.join(pred_grid[row_idx])}")
    print(f"\nGrid accuracy: {correct}/{len(positions)} ({100 * correct / len(positions):.1f}%)")
    print("=" * 60 + "\n")


def _evaluate_binary(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Run the model over a loader and return loss plus the binary metric block.

    Args:
        model: The 2-wide classifier.
        dataloader: Yields (x, y) with y in {0, 1}.
        criterion: Loss used for reporting only.
        device: Device to run on.

    Returns:
        Dict with `loss` and everything `binary_metrics` returns.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    scores: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []

    with torch.no_grad():
        for batch_x_raw, batch_y_raw in dataloader:
            batch_x = batch_x_raw.to(device)
            batch_y = batch_y_raw.to(device)

            outputs = model(batch_x)
            total_loss += criterion(outputs, batch_y).item()
            num_batches += 1

            scores.append(torch.softmax(outputs, dim=-1)[:, 1].cpu())
            labels.append(batch_y.cpu())

    all_scores = torch.cat(scores) if scores else torch.zeros(0)
    all_labels = torch.cat(labels) if labels else torch.zeros(0, dtype=torch.long)
    predictions = (all_scores > 0.5).long()

    results = binary_metrics(all_labels, predictions, all_scores)
    results["loss"] = total_loss / num_batches if num_batches > 0 else 0.0
    results["num_samples"] = int(all_labels.numel())
    results["predictions"] = predictions
    results["labels"] = all_labels
    return results


def _prepare_train_eval_v3_binary(
    load_result: dict,
    positive_cell_id: int,
    eval_data_path: str | None,
    eval_split: float,
    subset: float,
    balance_classes: bool,
    normalize: bool,
    per_class_max_count: int | None,
    seed: int,
    verbose: bool,
    use_cache: bool = False,
) -> dict:
    """Build the train/eval datasets for one binary arm from a v3 compact bundle.

    Structurally the multiclass `_prepare_train_eval_v3` with `_remap_labels` swapped for
    `_binarize_labels`; the token-major guard, the trajectory-level split, the balancing
    and the position-preserving normalisation are carried over unchanged.

    Args:
        load_result: Output of `_load_and_preprocess_train_data_v3`.
        positive_cell_id: Cell id that becomes class 1.
        eval_data_path: Separate v3 manifest dir to evaluate on, or None for an internal split.
        eval_split: Fraction held out when `eval_data_path` is None.
        subset: Fraction of entries to keep.
        balance_classes: Upsample the minority class to the majority count.
        normalize: Standardize the D activation dims (positions pass through).
        per_class_max_count: Cap per class when balancing.
        seed: Seed for the balancing draw.
        verbose: Print progress.
        use_cache: Read/write `_packed_activations.pt` beside the eval manifest.

    Returns:
        Bundle consumed by `train_binary_cognitive_map_probe`.

    Raises:
        ValueError: On a token-major manifest given an internal split or `--subset < 1.0`,
            or when the eval data is not also v3.
        FileNotFoundError: If `eval_data_path` does not exist.
    """
    compact = load_result["compact"]
    cells_per_trajectory = compact["C"]
    activation_dim = compact["D"]
    num_trajectories = compact["base_act"].shape[0]

    # Both `subset` and the internal split permute ENTRIES. One entry is one trajectory in
    # a classic manifest, so both are trajectory-level there. In a token-major manifest a
    # trajectory owns ~20 entries that share its grid, and a permutation over entries puts
    # near-duplicates of every eval row into training. Refuse rather than silently inflate.
    token_major = len(set(compact["trajectory_names"])) < num_trajectories
    if token_major:
        if eval_data_path is None:
            raise ValueError(
                f"{num_trajectories} entries cover only {len(set(compact['trajectory_names']))} "
                "trajectories, so this manifest is token-major and an internal --eval-split "
                "would put the same trajectory in both halves. Split it by trajectory first "
                "(scripts/split_next_action_manifest.py ... --train-out X_train --eval-out "
                "X_eval) and pass --eval-data-path."
            )
        if subset < 1.0:
            raise ValueError(
                f"--subset {subset} drops individual (token, layer) entries, not whole "
                "trajectories, on a token-major manifest. Thin it at split time instead "
                "(--tokens-per-trajectory / --layers-per-token) and keep --subset 1.0."
            )

    num_to_keep = max(1, int(num_trajectories * subset))
    perm = torch.randperm(num_trajectories)
    keep_idx = perm[:num_to_keep]
    base_act = compact["base_act"][keep_idx]
    positions = compact["positions"][keep_idx]
    raw_labels = compact["labels"][keep_idx]
    unit = "token samples" if token_major else "trajectories"
    print(f"Subset: keeping {num_to_keep}/{num_trajectories} {unit} ({subset * 100:.1f}%)")
    print(f"  Remaining samples: {num_to_keep * cells_per_trajectory}")

    label_to_idx, idx_to_label = binary_label_maps(positive_cell_id)
    binary_labels = _binarize_labels(raw_labels, positive_cell_id)
    if verbose:
        num_positive = int(binary_labels.sum().item())
        total = int(binary_labels.numel())
        print(f"Positive cells: {num_positive}/{total} ({100 * num_positive / total:.3f}%)")

    # Debug snapshot from the first entry (order already shuffled by the subset permutation).
    debug_positions = positions[0].clone() if num_to_keep > 0 else None
    debug_labels = binary_labels[0].clone() if num_to_keep > 0 else None
    debug_activations = None
    if num_to_keep > 0:
        debug_activations = torch.cat(
            [base_act[0].unsqueeze(0).expand(cells_per_trajectory, activation_dim), positions[0].float()],
            dim=1,
        ).clone()

    if eval_data_path is None:
        total_entries = base_act.shape[0]
        entry_perm = torch.randperm(total_entries)
        num_train = int(total_entries * (1 - eval_split))
        train_idx = entry_perm[:num_train]
        eval_idx = entry_perm[num_train:]

        train_base_act = base_act[train_idx]
        train_positions = positions[train_idx]
        train_labels_2d = binary_labels[train_idx]
        eval_base_act = base_act[eval_idx]
        eval_positions = positions[eval_idx]
        eval_labels_2d = binary_labels[eval_idx]
        eval_valid_cell_mask = None
        print(f"Split by trajectories: {len(train_idx)} train, {len(eval_idx)} eval")
    else:
        train_base_act = base_act
        train_positions = positions
        train_labels_2d = binary_labels

        eval_path = Path(eval_data_path)
        if not eval_path.exists():
            raise FileNotFoundError(f"Evaluation data not found: {eval_path}")
        if detect_format(eval_path) != 3:
            raise ValueError("Binary grid probes read v3 manifest directories only; the eval path is not one.")
        # `label_to_idx` covers every cell id, so this binarises the eval manifest's raw
        # labels with the same rule as training and masks nothing out.
        eval_compact = _load_separate_eval_data_v3(eval_path, label_to_idx, verbose, use_cache)
        eval_base_act = eval_compact["base_act"]
        eval_positions = eval_compact["positions"]
        eval_labels_2d = eval_compact["labels"]
        eval_valid_cell_mask = eval_compact.get("valid_cell_mask")

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

    # Normalisation covers the D activation dims only; the two position columns stay raw,
    # and the saved scaler is padded with [0, 0] / [1, 1] so inference is an identity there.
    scaler_mean = None
    scaler_std = None
    if normalize:
        scaler_mean_act, scaler_std_act = compute_normalization_params(train_base_act)
        train_base_act = normalize_activations(train_base_act, scaler_mean_act, scaler_std_act)
        eval_base_act = normalize_activations(eval_base_act, scaler_mean_act, scaler_std_act)
        scaler_mean = torch.cat([scaler_mean_act, torch.zeros(2, dtype=scaler_mean_act.dtype)])
        scaler_std = torch.cat([scaler_std_act, torch.ones(2, dtype=scaler_std_act.dtype)])
        print(f"Normalization enabled: computed mean/std from {train_base_act.shape[0]} training samples")

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
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "scaler_mean": scaler_mean,
        "scaler_std": scaler_std,
        "debug_trajectory_activations": debug_activations,
        "debug_trajectory_positions": debug_positions,
        "debug_trajectory_labels": debug_labels,
        "train_labels_for_weights": train_labels_for_weights,
        "input_dim": activation_dim + 2,
    }


def _print_final_results(results: dict, symbol: str) -> None:
    """Print the final metric block for one binary arm.

    Args:
        results: Output of `_evaluate_binary`.
        symbol: The grid symbol this probe decides.
    """
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}")
    print(f"AUROC: {results['auroc']:.4f}")
    print(f"Loss: {results['loss']:.4f}")
    print(f"Number of samples: {results['num_samples']}")
    print(f"Positive rate: {results['positive_rate']:.5f} ({results['positive_support']} cells)")

    print("\nPer-class metrics:")
    print("-" * 87)
    header = f"{'Class':<10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'GT Support':>12}"
    print(header)
    print("-" * 87)
    negative_support = results["n_rows"] - results["positive_support"]
    negative_precision = (
        results["tn"] / (results["tn"] + results["fn"]) if (results["tn"] + results["fn"]) > 0 else 0.0
    )
    negative_f1 = (
        2 * negative_precision * results["specificity"] / (negative_precision + results["specificity"])
        if (negative_precision + results["specificity"]) > 0
        else 0.0
    )
    print(
        f"{'¬' + symbol:<10} {results['specificity']:>10.4f} {negative_precision:>10.4f} "
        f"{results['specificity']:>10.4f} {negative_f1:>10.4f} {negative_support:>12}"
    )
    print(
        f"{symbol:<10} {results['recall']:>10.4f} {results['precision']:>10.4f} "
        f"{results['recall']:>10.4f} {results['f1']:>10.4f} {results['positive_support']:>12}"
    )
    print("-" * 87)
    print(f"Confusion: tp={results['tp']} fp={results['fp']} tn={results['tn']} fn={results['fn']}")


def train_binary_cognitive_map_probe(
    train_data_path: str,
    positive_class: str,
    model_type: ModelType = "lr",
    eval_data_path: str | None = None,
    output_path: str | None = None,
    num_epochs: int = 50,
    learning_rate: float = 3e-4,
    batch_size: int = 2048,
    weight_decay: float = 1e-3,
    hidden_dims: str = "1024",
    dropout: float = 0.0,
    eval_split: float = 0.2,
    subset: float = 1.0,
    class_weight: ClassWeight = "balanced",
    balance_classes: bool = False,
    normalize: bool = True,
    device: str | None = None,
    seed: int = 42,
    verbose: bool = True,
    per_class_max_count: int | None = None,
    cache_activations: bool = False,
) -> BinaryCognitiveMapProbe:
    """Train a one-vs-rest binary cognitive map probe on prepared activations.

    Takes a v3 `grid_tile` manifest directory produced by
    `prepare_activations_for_probing` and trains a 2-class classifier to decide whether a
    grid cell holds one chosen symbol. Four runs over the same dataset -- `_`, `#`, `A`,
    `G` -- give four probes that differ by their label and nothing else.

    Args:
        positive_class: The grid symbol to decide, or its alias:
            `_`/empty, `#`/wall, `A`/agent, `G`/goal, `D`/door, `K`/key, `?`/unknown,
            `+`/padding.
        train_data_path: Path to a v3 prepared dataset directory (probe_type=grid_tile)
        model_type: Type of classifier to train:
            - "lr": Logistic regression (linear classifier)
            - "mlp": Multi-layer perceptron
        eval_data_path: Optional path to a separate v3 dataset for evaluation.
            **Required** for a token-major manifest, where an internal split would leak.
        output_path: Path to save the trained model. If not provided, saves to the same
            directory as train_data_path.
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        weight_decay: L2 regularization weight decay
        hidden_dims: Comma-separated hidden layer dimensions for MLP
        dropout: Dropout rate for MLP (ignored for lr)
        eval_split: Fraction of training data to use for validation
            (only used if eval_data_path is not provided)
        subset: Fraction of entries to use (0.0 to 1.0). Refused below 1.0 on a
            token-major manifest -- thin at split time instead.
        class_weight: How to weight classes in the loss function. Defaults to "balanced"
            here, unlike the multiclass trainer: `A` and `G` are one cell per grid, so an
            unweighted loss on those arms converges to predicting the negative class.
        balance_classes: If True, upsample the minority class to the majority count
        normalize: If True, standardize the activation dims using training mean/std.
            The parameters are saved with the probe and applied at inference.
        device: Device to use for training (e.g., "cuda", "cpu")
        seed: Random seed for reproducibility
        verbose: Print training progress
        per_class_max_count: Cap per class when balance_classes upsamples
        cache_activations: Pack (activations, positions, labels) into
            `_packed_activations.pt` beside the manifest on first load and reuse it.
            Worth it for a token-major manifest, and shared by every arm over the same
            dataset -- the first of the eight runs pays for it and the rest do not.

    Returns:
        Trained BinaryCognitiveMapProbe instance (also saved to output_path)

    Raises:
        FileNotFoundError: If `train_data_path` does not exist.
        ValueError: If `subset` is outside (0, 1] or the dataset is not a v3 manifest dir.
    """
    symbol, positive_cell_id = resolve_positive_class(positive_class)

    set_seed(seed)
    torch_device = resolve_device(device)
    print(f"Using device: {torch_device}")
    print(f"Positive class: '{symbol}' (cell id {positive_cell_id})")

    train_path = Path(train_data_path)
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not 0.0 < subset <= 1.0:
        raise ValueError(f"subset must be in (0.0, 1.0], got {subset}")
    if detect_format(train_path) != 3:
        raise ValueError(
            f"{train_path} is not a v3 prepared dataset directory. Binary grid probes read "
            "v3 manifests only; re-run prepare_activations_for_probing to produce one."
        )

    load_result = _load_and_preprocess_train_data_v3(train_path, verbose, cache_activations)
    bundle = _prepare_train_eval_v3_binary(
        load_result=load_result,
        positive_cell_id=positive_cell_id,
        eval_data_path=eval_data_path,
        eval_split=eval_split,
        subset=subset,
        balance_classes=balance_classes,
        normalize=normalize,
        per_class_max_count=per_class_max_count,
        seed=seed,
        verbose=verbose,
        use_cache=cache_activations,
    )

    train_dataset = bundle["train_dataset"]
    eval_dataset = bundle["eval_dataset"]
    scaler_mean = bundle["scaler_mean"]
    scaler_std = bundle["scaler_std"]
    train_labels = bundle["train_labels_for_weights"]
    input_dim = bundle["input_dim"]

    hidden_dims_list = [int(d.strip()) for d in hidden_dims.split(",") if d.strip()]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    model = _create_model(
        model_type=model_type,
        input_dim=input_dim,
        num_classes=2,
        hidden_dims=hidden_dims_list,
        dropout=dropout,
    ).to(torch_device)

    if verbose:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model type: {model_type}")
        print(f"Input dimension: {input_dim}")
        print(f"Hidden dimensions: {hidden_dims_list}")
        print(f"Number of parameters: {num_params:,}")

    if class_weight == "balanced":
        print("Using balanced class weights in loss function")
        weights = _compute_class_weights(train_labels, 2, torch_device)
        print(f"  Class weights: {weights.tolist()}")
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_eval_accuracy = 0.0
    best_balanced_accuracy = 0.0
    best_auroc = 0.0

    print(f"\nTraining for {num_epochs} epochs...")
    epoch_iterator = tqdm(range(num_epochs), desc="Training", disable=not verbose)
    for _ in epoch_iterator:
        train_loss = _train_epoch(model, train_loader, criterion, optimizer, torch_device)
        eval_results = _evaluate_binary(model, eval_loader, criterion, torch_device)

        best_eval_accuracy = max(best_eval_accuracy, eval_results["accuracy"])
        best_balanced_accuracy = max(best_balanced_accuracy, eval_results["balanced_accuracy"])
        if eval_results["auroc"] == eval_results["auroc"]:  # not NaN
            best_auroc = max(best_auroc, eval_results["auroc"])

        epoch_iterator.set_postfix(
            {
                "loss": f"{train_loss:.4f}",
                "bal_acc": f"{eval_results['balanced_accuracy']:.4f}",
                "auroc": f"{eval_results['auroc']:.4f}",
                "recall": f"{eval_results['recall']:.4f}",
                "best_bal": f"{best_balanced_accuracy:.4f}",
            }
        )

    print("\n" + "=" * 60)
    print(f"FINAL EVALUATION ('{symbol}' vs rest)")
    print("=" * 60)
    final_results = _evaluate_binary(model, eval_loader, criterion, torch_device)

    if verbose:
        _print_final_results(final_results, symbol)
        debug_activations = bundle["debug_trajectory_activations"]
        if debug_activations is not None:
            if normalize:
                debug_activations = (debug_activations - scaler_mean) / scaler_std
            with torch.no_grad():
                debug_outputs = model(debug_activations.float().to(torch_device))
                debug_preds = torch.argmax(debug_outputs, dim=1).cpu()
            _print_binary_debug_grid(
                positions=bundle["debug_trajectory_positions"],
                true_labels=bundle["debug_trajectory_labels"],
                pred_labels=debug_preds,
                symbol=symbol,
            )

    probe = BinaryCognitiveMapProbe(
        model=model,
        model_type=model_type,
        input_dim=input_dim,
        positive_class=symbol,
        positive_cell_id=positive_cell_id,
        hidden_dims=hidden_dims_list if model_type == "mlp" else None,
        dropout=dropout if model_type == "mlp" else None,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
        config={
            "train_data_path": str(train_path),
            "eval_data_path": str(eval_data_path) if eval_data_path else None,
            "positive_class": symbol,
            "positive_cell_id": positive_cell_id,
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
            "best_auroc": best_auroc,
            "final_accuracy": final_results["accuracy"],
            "final_balanced_accuracy": final_results["balanced_accuracy"],
            "final_auroc": final_results["auroc"],
            "final_precision": final_results["precision"],
            "final_recall": final_results["recall"],
            "final_f1": final_results["f1"],
            "final_loss": final_results["loss"],
            "positive_rate": final_results["positive_rate"],
            "confusion": {k: final_results[k] for k in ("tp", "fp", "tn", "fn")},
        },
        device=torch_device,
    )

    if output_path is None:
        slug = POSITIVE_CLASS_SLUGS[symbol]
        final_output_path = train_path.parent / f"binary_cognitive_map_probe_{slug}_{model_type}.pt"
    else:
        final_output_path = Path(output_path)

    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    probe.save(final_output_path)

    print(f"\nModel saved to {final_output_path}")
    print(f"Best evaluation accuracy: {best_eval_accuracy:.4f}")
    print(f"Best balanced accuracy: {best_balanced_accuracy:.4f}")
    print(f"Best AUROC: {best_auroc:.4f}")

    return probe
