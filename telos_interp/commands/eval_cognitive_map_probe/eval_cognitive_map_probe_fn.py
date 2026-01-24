"""Evaluate cognitive map probes on test trajectories with detailed metrics."""

import json
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from telos_interp.commands.prepare_activations_for_probing import (
    CELL_ID_TO_SYMBOL,
    CELL_SYMBOL_TO_ID,
    parse_grid_state,
)
from telos_interp.commands.prepare_activations_for_probing.prepare_activations_for_probing_utils import (
    discover_available_layers,
    discover_available_steps,
    discover_available_token_indices,
    discover_model_folder,
    load_activation,
    parse_index_specification,
)
from telos_interp.commands.train_cognitive_map_probe import CognitiveMapProbe


class MetricsAccumulator:
    """Accumulates per-class metrics for evaluation."""

    def __init__(self, num_classes: int, class_names: list[str]):
        self.num_classes = num_classes
        self.class_names = class_names
        # Per-class counts
        self.true_positives = defaultdict(int)
        self.ground_truth_count = defaultdict(int)
        self.predicted_count = defaultdict(int)
        self.total_correct = 0
        self.total_samples = 0

    def update(self, predictions: list[int], ground_truth: list[int]) -> None:
        """Update metrics with a batch of predictions and ground truth."""
        for pred, gt in zip(predictions, ground_truth, strict=False):
            self.total_samples += 1
            self.ground_truth_count[gt] += 1
            self.predicted_count[pred] += 1

            if pred == gt:
                self.total_correct += 1
                self.true_positives[gt] += 1

    def compute_metrics(self) -> dict:
        """Compute final metrics including baseline accuracy."""
        if self.total_samples == 0:
            return {"accuracy": 0.0, "baseline_accuracy": 0.0, "per_class": {}, "total_samples": 0}

        accuracy = self.total_correct / self.total_samples

        # Compute baseline accuracy (majority class baseline)
        # This is the accuracy achieved by always predicting the most common class
        if self.ground_truth_count:
            majority_class_count = max(self.ground_truth_count.values())
            baseline_accuracy = majority_class_count / self.total_samples
        else:
            baseline_accuracy = 0.0

        per_class = {}
        for class_id in range(self.num_classes):
            tp = self.true_positives[class_id]
            gt_count = self.ground_truth_count[class_id]
            pred_count = self.predicted_count[class_id]

            precision = tp / pred_count if pred_count > 0 else 0.0
            recall = tp / gt_count if gt_count > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            class_accuracy = tp / gt_count if gt_count > 0 else 0.0

            per_class[class_id] = {
                "accuracy": class_accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "gt_support": gt_count,
                "predicted": pred_count,
            }

        return {
            "accuracy": accuracy,
            "baseline_accuracy": baseline_accuracy,
            "per_class": per_class,
            "total_samples": self.total_samples,
        }


def _print_metrics_table(
    metrics: dict,
    class_names: list[str],
    idx_to_label: dict[int, int],
    title: str | None = None,
) -> None:
    """Print a formatted metrics table."""
    if title:
        print(f"\n{'=' * 87}")
        print(title)
        print("=" * 87)

    baseline_acc = metrics.get("baseline_accuracy", 0.0)
    print(f"Overall Accuracy: {metrics['accuracy']:.4f} (Baseline: {baseline_acc:.4f}, {metrics['total_samples']} samples)")
    print("\nPer-class metrics:")
    print("-" * 87)
    print(
        f"{'Class':<10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'GT Support':>12} {'Predicted':>10}"
    )
    print("-" * 87)

    for class_idx in sorted(metrics["per_class"].keys()):
        class_metrics = metrics["per_class"][class_idx]
        # Map back to original label ID, then to symbol
        original_label = idx_to_label.get(class_idx, class_idx)
        symbol = CELL_ID_TO_SYMBOL.get(original_label, str(original_label))
        print(
            f"{symbol:<10} {class_metrics['accuracy']:>10.4f} {class_metrics['precision']:>10.4f} "
            f"{class_metrics['recall']:>10.4f} {class_metrics['f1']:>10.4f} "
            f"{class_metrics['gt_support']:>12} {class_metrics['predicted']:>10}"
        )
    print("-" * 87)


def _load_concatenated_activations(
    trajectory_folder: Path,
    layer_idx: int,
    step_idx: int,
    category_specs: dict[str, str | None],
) -> torch.Tensor | None:
    """Load and concatenate activations from multiple tokens for a layer/step."""
    model_folder = discover_model_folder(trajectory_folder)
    if model_folder is None:
        return None

    step_folder = model_folder / f"layer_{layer_idx}" / f"step_{step_idx}"
    if not step_folder.exists():
        return None

    all_activations = []

    # Process categories in consistent order
    for category in ["prompt_prefix", "prompt_suffix", "grid_state", "output"]:
        index_spec = category_specs.get(category)
        if index_spec is None:
            continue

        category_folder = step_folder / category
        if not category_folder.exists():
            continue

        available_tokens = discover_available_token_indices(category_folder)
        if not available_tokens:
            continue

        selected_tokens = parse_index_specification(index_spec, available_tokens)
        if not selected_tokens:
            continue

        for token_idx in selected_tokens:
            file_path = category_folder / f"{token_idx}.pt"
            if file_path.exists():
                activation = load_activation(file_path)
                all_activations.append(activation)

    if not all_activations:
        return None

    return torch.cat(all_activations, dim=0)


def _apply_probe_to_all_positions(
    probe: CognitiveMapProbe,
    activation: torch.Tensor,
    grid_size: int,
) -> dict[tuple[int, int], int]:
    """Apply probe to an activation for all grid positions.

    Returns:
        Dict mapping (row, col) to predicted class index
    """
    # Generate all positions
    positions = []
    for row in range(grid_size):
        for col in range(grid_size):
            positions.append([row, col])

    # Create batch: replicate activation for all positions
    positions_tensor = torch.tensor(positions, dtype=activation.dtype)
    activation_expanded = activation.unsqueeze(0).expand(len(positions), -1)
    activations_with_positions = torch.cat([activation_expanded, positions_tensor], dim=1)

    # Get predictions
    predictions = probe.predict(activations_with_positions)

    # Build result dictionary
    results = {}
    for idx, (row, col) in enumerate(positions):
        # Convert from original label ID to class index for comparison
        original_label = predictions[idx].item()
        class_idx = probe.label_to_idx.get(original_label, original_label)
        results[(row, col)] = class_idx

    return results


def _get_ground_truth_for_step(
    grid_state: list[str],
    grid_size: int,
    label_to_idx: dict[int, int],
) -> dict[tuple[int, int], int]:
    """Parse grid state and return ground truth labels for each position.

    Returns:
        Dict mapping (row, col) to class index
    """
    parsed = parse_grid_state(grid_state, pad_to_size=grid_size)

    results = {}
    for row_id, col_id, cell_id in parsed:
        # Map cell_id to class index
        class_idx = label_to_idx.get(cell_id, cell_id)
        results[(row_id, col_id)] = class_idx

    return results


def _extract_complexity_from_trajectory(trajectory_data: dict) -> float:
    """Extract complexity value from trajectory data."""
    return trajectory_data.get("grid_params", {}).get("grid_complexity", 0.0)


def _extract_size_from_trajectory(trajectory_data: dict) -> int:
    """Extract grid size from trajectory data."""
    width = trajectory_data.get("grid_params", {}).get("grid_width", 0)
    height = trajectory_data.get("grid_params", {}).get("grid_height", 0)
    return max(width, height)


def eval_cognitive_map_probe(
    probe_path: str,
    trajectories_dir: str,
    activations_dir: str,
    layers: str = "all",
    steps: str = "all",
    prompt_prefix_indices: str | None = None,
    prompt_suffix_indices: str | None = None,
    grid_state_indices: str | None = None,
    output_indices: str | None = None,
    pad_to_size: int = 15,
    output_path: str | None = None,
    device: str | None = None,
    verbose: bool = True,
) -> dict:
    """Evaluate a cognitive map probe on test trajectories.

    Args:
        probe_path: Path to the trained probe (.pt file)
        trajectories_dir: Directory containing trajectory JSON files, organized by size
            (e.g., trajectories_dir/size7/, trajectories_dir/size9/, etc.)
        activations_dir: Directory containing gathered activations, organized by size
            (e.g., activations_dir/size7/, activations_dir/size9/, etc.)
        layers: Layer specification (e.g., "15", "10,15,20", "all")
        steps: Step specification (e.g., "0", "all", "0:5")
        prompt_prefix_indices: Token indices for prompt_prefix (e.g., "all", "-1")
        prompt_suffix_indices: Token indices for prompt_suffix (e.g., "-3:-1")
        grid_state_indices: Token indices for grid_state (e.g., "all")
        output_indices: Token indices for output (e.g., "all", "0:10")
        pad_to_size: Pad grid to this size for consistent evaluation
        output_path: Path to save JSON results. If not provided, auto-generates from
            probe name and trajectories directory name (e.g., "eval_<probe>_<trajectories>.json")
        device: Device to use for inference
        verbose: Print progress information

    Returns:
        Dictionary containing all evaluation metrics
    """
    # Determine device
    device_str = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    torch_device = torch.device(device_str)

    if verbose:
        print(f"Using device: {torch_device}")

    # Load probe
    probe_path_obj = Path(probe_path)
    if not probe_path_obj.exists():
        raise FileNotFoundError(f"Probe not found: {probe_path}")

    if verbose:
        print(f"Loading probe from {probe_path}")

    probe = CognitiveMapProbe.load(probe_path, device=device_str)

    if verbose:
        print(f"Loaded probe: {probe_path_obj.stem}")
        print(f"  Input dimension: {probe.input_dim}")
        print(f"  Number of classes: {probe.num_classes}")
        print(f"  Normalized: {probe.normalized}")

    # Build category specs
    category_specs = {
        "prompt_prefix": prompt_prefix_indices,
        "prompt_suffix": prompt_suffix_indices,
        "grid_state": grid_state_indices,
        "output": output_indices,
    }
    active_categories = {k: v for k, v in category_specs.items() if v is not None}

    if not active_categories:
        raise ValueError("At least one token category must be specified")

    if verbose:
        print(f"\nToken categories: {active_categories}")

    # Find all size folders
    trajectories_base = Path(trajectories_dir)
    activations_base = Path(activations_dir)

    size_folders = sorted([d for d in trajectories_base.iterdir() if d.is_dir() and d.name.startswith("size")])

    if not size_folders:
        raise ValueError(f"No size folders found in {trajectories_dir}")

    if verbose:
        print(f"\nFound {len(size_folders)} size folders: {[f.name for f in size_folders]}")

    # Get class info from probe
    num_classes = probe.num_classes
    class_names = [CELL_ID_TO_SYMBOL.get(probe.idx_to_label[i], str(i)) for i in range(num_classes)]

    # Initialize accumulators
    global_accumulator = MetricsAccumulator(num_classes, class_names)
    size_accumulators: dict[int, MetricsAccumulator] = defaultdict(
        lambda: MetricsAccumulator(num_classes, class_names)
    )
    complexity_accumulators: dict[float, MetricsAccumulator] = defaultdict(
        lambda: MetricsAccumulator(num_classes, class_names)
    )
    size_complexity_accumulators: dict[tuple[int, float], MetricsAccumulator] = defaultdict(
        lambda: MetricsAccumulator(num_classes, class_names)
    )

    # Process each size folder
    total_trajectories = 0
    total_steps_processed = 0

    for size_folder in size_folders:
        size_name = size_folder.name  # e.g., "size7"
        activations_size_folder = activations_base / size_name

        if not activations_size_folder.exists():
            if verbose:
                print(f"  Skipping {size_name}: no activations folder")
            continue

        # Find trajectory JSONs
        trajectory_files = sorted(size_folder.glob("*.json"))

        if verbose:
            print(f"\nProcessing {size_name}: {len(trajectory_files)} trajectories")

        for traj_file in tqdm(trajectory_files, desc=f"  {size_name}", disable=not verbose):
            # Load trajectory
            with open(traj_file) as f:
                trajectory_data = json.load(f)

            # Extract metadata
            grid_size = _extract_size_from_trajectory(trajectory_data)
            complexity = _extract_complexity_from_trajectory(trajectory_data)

            # Find corresponding activations folder
            traj_name = traj_file.stem
            traj_activations_folder = activations_size_folder / traj_name

            if not traj_activations_folder.exists():
                continue

            # Discover available layers
            model_folder = discover_model_folder(traj_activations_folder)
            if model_folder is None:
                continue

            available_layers = discover_available_layers(model_folder)
            selected_layers = parse_index_specification(layers, available_layers)

            if not selected_layers:
                continue

            total_trajectories += 1

            # Process each layer
            for layer_idx in selected_layers:
                layer_folder = model_folder / f"layer_{layer_idx}"
                if not layer_folder.exists():
                    continue

                available_steps = discover_available_steps(layer_folder)
                selected_steps = parse_index_specification(steps, available_steps)

                for step_idx in selected_steps:
                    # Get step data from trajectory
                    if step_idx >= len(trajectory_data.get("steps", [])):
                        continue

                    step_data = trajectory_data["steps"][step_idx]
                    grid_state = step_data.get("grid_state", [])

                    if not grid_state:
                        continue

                    # Load concatenated activations
                    activation = _load_concatenated_activations(
                        trajectory_folder=traj_activations_folder,
                        layer_idx=layer_idx,
                        step_idx=step_idx,
                        category_specs=category_specs,
                    )

                    if activation is None:
                        continue

                    # Get predictions for all positions
                    predictions = _apply_probe_to_all_positions(probe, activation, pad_to_size)

                    # Get ground truth
                    ground_truth = _get_ground_truth_for_step(
                        grid_state, pad_to_size, probe.label_to_idx
                    )

                    # Collect predictions and ground truth as lists
                    pred_list = []
                    gt_list = []

                    for pos in predictions:
                        if pos in ground_truth:
                            pred_list.append(predictions[pos])
                            gt_list.append(ground_truth[pos])

                    # Update all accumulators
                    global_accumulator.update(pred_list, gt_list)
                    size_accumulators[grid_size].update(pred_list, gt_list)
                    complexity_accumulators[complexity].update(pred_list, gt_list)
                    size_complexity_accumulators[(grid_size, complexity)].update(pred_list, gt_list)

                    total_steps_processed += 1

    # Compute and print results
    if verbose:
        print(f"\n{'=' * 87}")
        print("EVALUATION COMPLETE")
        print("=" * 87)
        print(f"Processed {total_trajectories} trajectories, {total_steps_processed} steps")

    # Global metrics
    global_metrics = global_accumulator.compute_metrics()
    if verbose:
        _print_metrics_table(global_metrics, class_names, probe.idx_to_label, "GLOBAL METRICS")

    # Per-size metrics
    size_metrics = {}
    if verbose:
        print(f"\n{'=' * 87}")
        print("METRICS BY SIZE")
        print("=" * 87)

    for grid_size in sorted(size_accumulators.keys()):
        metrics = size_accumulators[grid_size].compute_metrics()
        size_metrics[grid_size] = metrics
        if verbose:
            _print_metrics_table(metrics, class_names, probe.idx_to_label, f"Size {grid_size}")

    # Per-complexity metrics
    complexity_metrics = {}
    if verbose:
        print(f"\n{'=' * 87}")
        print("METRICS BY COMPLEXITY")
        print("=" * 87)

    for complexity in sorted(complexity_accumulators.keys()):
        metrics = complexity_accumulators[complexity].compute_metrics()
        complexity_metrics[complexity] = metrics
        if verbose:
            _print_metrics_table(
                metrics, class_names, probe.idx_to_label, f"Complexity {complexity:.2f}"
            )

    # Per-size-complexity metrics
    size_complexity_metrics = {}
    if verbose:
        print(f"\n{'=' * 87}")
        print("METRICS BY SIZE-COMPLEXITY COMBINATION")
        print("=" * 87)

    for (grid_size, complexity) in sorted(size_complexity_accumulators.keys()):
        metrics = size_complexity_accumulators[(grid_size, complexity)].compute_metrics()
        size_complexity_metrics[(grid_size, complexity)] = metrics
        if verbose:
            _print_metrics_table(
                metrics,
                class_names,
                probe.idx_to_label,
                f"Size {grid_size}, Complexity {complexity:.2f}",
            )

    # Build results dictionary
    results = {
        "global": global_metrics,
        "by_size": {str(k): v for k, v in size_metrics.items()},
        "by_complexity": {str(k): v for k, v in complexity_metrics.items()},
        "by_size_complexity": {f"{s}_{c}": m for (s, c), m in size_complexity_metrics.items()},
        "total_trajectories": total_trajectories,
        "total_steps": total_steps_processed,
        "config": {
            "probe_path": str(probe_path_obj),
            "trajectories_dir": str(trajectories_base),
            "activations_dir": str(activations_base),
            "layers": layers,
            "steps": steps,
            "token_categories": active_categories,
            "pad_to_size": pad_to_size,
        },
    }

    # Save JSON results
    if output_path is None:
        # Auto-generate filename from probe name and trajectories dir name
        probe_name = probe_path_obj.stem
        trajectories_name = trajectories_base.name
        output_path = f"eval_{probe_name}_{trajectories_name}.json"

    output_path_obj = Path(output_path)
    with open(output_path_obj, "w") as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"\nResults saved to: {output_path_obj}")

    return results
