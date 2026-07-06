"""Evaluate cognitive map probes with metrics stratified by distance to goal/agent.

This is a fork of ``eval_cognitive_map_probe`` that adds two extra aggregation
axes on top of the standard class-based metrics:

- **accuracy per distance to goal**: for every grid cell, the (square / Chebyshev)
  distance from that cell to the goal (``G``) is computed, and metrics are
  accumulated into buckets keyed by that distance.
- **accuracy per distance to agent**: same, but distance to the agent (``A``).

"Distance" is the square-based (Chebyshev) distance in the grid, i.e. the number
of king-moves between two cells: ``max(|row_a - row_b|, |col_a - col_b|)``. For
example, in a grid with goal ``G``::

    1  2  3  4
    5  G  6  7
    8  9  10 11
    12 13 14 15

cells 1,2,3,5,6,8,9,10 are at distance 1 and cells 4,7,11,12,13,14,15 are at
distance 2.

Everything else (probe loading, activation loading, token categories, per-size
and per-complexity breakdowns) is identical to ``eval_cognitive_map_probe``.
"""

import json
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from telos_interp.activation_loading import (
    discover_available_layers,
    discover_available_steps,
    discover_model_folder,
    load_concatenated_activations,
    parse_category_specs,
    parse_index_specification,
)
from telos_interp.commands.train_cognitive_map_probe import CognitiveMapProbe
from telos_interp.grid_utils import CELL_ID_TO_SYMBOL, CELL_SYMBOL_TO_ID, parse_grid_state


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
        """Compute final metrics including baseline and balanced accuracy."""
        if self.total_samples == 0:
            return {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "baseline_accuracy": 0.0,
                "per_class": {},
                "total_samples": 0,
            }

        accuracy = self.total_correct / self.total_samples

        # Compute baseline accuracy (majority class baseline)
        # This is the accuracy achieved by always predicting the most common class
        if self.ground_truth_count:
            majority_class_count = max(self.ground_truth_count.values())
            baseline_accuracy = majority_class_count / self.total_samples
        else:
            baseline_accuracy = 0.0

        per_class = {}
        recalls_for_balanced = []
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

            # Balanced accuracy = mean recall over classes that are actually present
            if gt_count > 0:
                recalls_for_balanced.append(recall)

        balanced_accuracy = (
            sum(recalls_for_balanced) / len(recalls_for_balanced) if recalls_for_balanced else 0.0
        )

        return {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "baseline_accuracy": baseline_accuracy,
            "per_class": per_class,
            "total_samples": self.total_samples,
        }


def _pivot_distance_by_class(distance_metrics: dict) -> dict:
    """Transpose ``distance -> {..., per_class}`` into ``class -> {distance: metrics}``.

    Args:
        distance_metrics: mapping of distance-key (str) to a metrics dict, i.e.
            the output of ``MetricsAccumulator.compute_metrics()``.

    Returns:
        Mapping of class index (int) to ``{distance_key: per_class_metrics}``,
        where ``per_class_metrics`` is that class's entry (accuracy, precision,
        recall, f1, gt_support, predicted) at that distance.
    """
    per_class_out: dict[int, dict[str, dict]] = defaultdict(dict)
    for dist_key, metrics in distance_metrics.items():
        for class_id, class_metrics in metrics.get("per_class", {}).items():
            per_class_out[class_id][dist_key] = class_metrics
    return {class_id: dict(dist_map) for class_id, dist_map in per_class_out.items()}


class DistanceAccumulatorSet:
    """Holds per-distance MetricsAccumulators for both goal- and agent-distance.

    Distance buckets are created lazily as distances are observed.
    """

    def __init__(self, num_classes: int, class_names: list[str]):
        self.num_classes = num_classes
        self.class_names = class_names
        self.by_goal_distance: dict[int, MetricsAccumulator] = defaultdict(
            lambda: MetricsAccumulator(num_classes, class_names)
        )
        self.by_agent_distance: dict[int, MetricsAccumulator] = defaultdict(
            lambda: MetricsAccumulator(num_classes, class_names)
        )

    def update(self, cell_records: list[tuple[int, int, int | None, int | None]]) -> None:
        """Update from per-cell records.

        Args:
            cell_records: list of ``(pred, gt, dist_to_goal, dist_to_agent)``. A
                distance of ``None`` means it could not be computed (goal/agent
                not present in the grid) and is skipped for that axis.
        """
        # Group by distance to avoid many tiny accumulator calls
        goal_groups: dict[int, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
        agent_groups: dict[int, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))

        for pred, gt, dist_goal, dist_agent in cell_records:
            if dist_goal is not None:
                goal_groups[dist_goal][0].append(pred)
                goal_groups[dist_goal][1].append(gt)
            if dist_agent is not None:
                agent_groups[dist_agent][0].append(pred)
                agent_groups[dist_agent][1].append(gt)

        for dist, (preds, gts) in goal_groups.items():
            self.by_goal_distance[dist].update(preds, gts)
        for dist, (preds, gts) in agent_groups.items():
            self.by_agent_distance[dist].update(preds, gts)

    def compute(self) -> dict:
        """Compute metrics for every distance bucket on both axes.

        In addition to the distance-keyed metrics (``distance -> class``), this
        also returns the transpose (``class -> distance``) so that per-class
        tables can be built directly. Per-class keys are class indices (probe
        index space); use the top-level ``class_labels`` mapping to decode them
        to symbols.
        """
        by_goal = {
            str(dist): self.by_goal_distance[dist].compute_metrics()
            for dist in sorted(self.by_goal_distance.keys())
        }
        by_agent = {
            str(dist): self.by_agent_distance[dist].compute_metrics()
            for dist in sorted(self.by_agent_distance.keys())
        }
        return {
            "by_goal_distance": by_goal,
            "by_agent_distance": by_agent,
            "by_goal_distance_per_class": _pivot_distance_by_class(by_goal),
            "by_agent_distance_per_class": _pivot_distance_by_class(by_agent),
        }


def _chebyshev_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Square-based (Chebyshev) distance between two cells."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


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
    balanced_acc = metrics.get("balanced_accuracy", 0.0)
    print(
        f"Accuracy: {metrics['accuracy']:.4f} | Balanced: {balanced_acc:.4f} | "
        f"Baseline: {baseline_acc:.4f} | {metrics['total_samples']} samples"
    )
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


def _print_distance_table(distance_metrics: dict, axis_name: str, title: str | None = None) -> None:
    """Print a compact table of accuracy per distance bucket."""
    if title:
        print(f"\n{'-' * 70}")
        print(title)
        print("-" * 70)
    print(
        f"{'Distance':>10} {'Accuracy':>10} {'Balanced':>10} {'Baseline':>10} {'Samples':>12}"
    )
    print("-" * 70)
    for dist_key in sorted(distance_metrics.keys(), key=lambda k: int(k)):
        m = distance_metrics[dist_key]
        print(
            f"{dist_key:>10} {m['accuracy']:>10.4f} {m['balanced_accuracy']:>10.4f} "
            f"{m['baseline_accuracy']:>10.4f} {m['total_samples']:>12}"
        )
    print("-" * 70)


def _print_per_class_distance_tables(
    per_class_pivot: dict,
    axis_name: str,
    idx_to_label: dict[int, int],
    title_prefix: str | None = None,
) -> None:
    """Print one table per class: rows are distances, columns are the metrics.

    Classes with no ground-truth support at any distance are skipped.
    """
    for class_id in sorted(per_class_pivot.keys()):
        dist_map = per_class_pivot[class_id]
        if all(cm["gt_support"] == 0 for cm in dist_map.values()):
            continue

        symbol = CELL_ID_TO_SYMBOL.get(idx_to_label.get(class_id, class_id), str(class_id))
        header = f"class '{symbol}' | distance to {axis_name}"
        if title_prefix:
            header = f"{title_prefix} | {header}"

        print(f"\n{'-' * 78}")
        print(header)
        print("-" * 78)
        print(
            f"{'Distance':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} "
            f"{'F1-Score':>10} {'Support':>10} {'Predicted':>10}"
        )
        print("-" * 78)
        for dist_key in sorted(dist_map.keys(), key=lambda k: int(k)):
            cm = dist_map[dist_key]
            print(
                f"{dist_key:>10} {cm['accuracy']:>10.4f} {cm['precision']:>10.4f} "
                f"{cm['recall']:>10.4f} {cm['f1']:>10.4f} {cm['gt_support']:>10} {cm['predicted']:>10}"
            )
        print("-" * 78)


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


def _find_special_positions(
    ground_truth: dict[tuple[int, int], int],
    goal_idx: int,
    agent_idx: int,
) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    """Locate the goal (G) and agent (A) positions from ground truth labels."""
    goal_pos = None
    agent_pos = None
    for pos, class_idx in ground_truth.items():
        if class_idx == goal_idx:
            goal_pos = pos
        elif class_idx == agent_idx:
            agent_pos = pos
    return goal_pos, agent_pos


def _build_distance_records(
    predictions: dict[tuple[int, int], int],
    ground_truth: dict[tuple[int, int], int],
    goal_pos: tuple[int, int] | None,
    agent_pos: tuple[int, int] | None,
    padding_idx: int,
) -> list[tuple[int, int, int | None, int | None]]:
    """Build per-cell ``(pred, gt, dist_to_goal, dist_to_agent)`` records.

    Padding cells are skipped entirely: they are not part of the real grid, so
    their distance to goal/agent is meaningless.
    """
    records = []
    for pos, gt in ground_truth.items():
        if gt == padding_idx:
            continue
        if pos not in predictions:
            continue
        pred = predictions[pos]
        dist_goal = _chebyshev_distance(pos, goal_pos) if goal_pos is not None else None
        dist_agent = _chebyshev_distance(pos, agent_pos) if agent_pos is not None else None
        records.append((pred, gt, dist_goal, dist_agent))
    return records


def _extract_complexity_from_trajectory(trajectory_data: dict) -> float:
    """Extract complexity value from trajectory data."""
    return trajectory_data.get("grid_params", {}).get("grid_complexity", 0.0)


def _extract_size_from_trajectory(trajectory_data: dict) -> int:
    """Extract grid size from trajectory data."""
    width = trajectory_data.get("grid_params", {}).get("grid_width", 0)
    height = trajectory_data.get("grid_params", {}).get("grid_height", 0)
    return max(width, height)


def eval_cognitive_map_probe_per_distance(  # noqa: PLR0912, PLR0915
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
    """Evaluate a cognitive map probe, stratifying metrics by distance to goal/agent.

    Same as ``eval_cognitive_map_probe`` but additionally reports, for each
    metrics block (global / per-size / per-complexity), the accuracy broken down
    by square-based (Chebyshev) distance from each cell to the goal and to the
    agent.

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
            probe name and trajectories directory name
            (e.g., "eval_per_distance_<probe>_<trajectories>.json")
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
    category_specs = parse_category_specs(
        prompt_prefix_indices=prompt_prefix_indices,
        prompt_suffix_indices=prompt_suffix_indices,
        grid_state_indices=grid_state_indices,
        output_indices=output_indices,
    )
    active_categories = {k: v for k, v in category_specs.items() if v is not None}

    if not active_categories:
        raise ValueError("At least one token category must be specified")

    if verbose:
        print(f"\nToken categories: {active_categories}")

    # Find all size folders
    trajectories_base = Path(trajectories_dir)
    activations_base = Path(activations_dir)

    size_folders = sorted([d for d in trajectories_base.iterdir() if d.is_dir() and d.name.startswith("size")])

    # Check if we have size folders or if trajectories are directly in the base folder
    single_size_mode = False
    if not size_folders:
        # Check if there are trajectory JSONs directly in the base folder
        direct_trajectories = list(trajectories_base.glob("*.json"))
        if direct_trajectories:
            single_size_mode = True
            if verbose:
                print(
                    f"\nNo size folders found. Running in single-size mode with {len(direct_trajectories)} trajectories"
                )
        else:
            raise ValueError(f"No size folders or trajectory files found in {trajectories_dir}")
    elif verbose:
        print(f"\nFound {len(size_folders)} size folders: {[f.name for f in size_folders]}")

    # Get class info from probe
    num_classes = probe.num_classes
    class_names = [CELL_ID_TO_SYMBOL.get(probe.idx_to_label[i], str(i)) for i in range(num_classes)]

    # Resolve special class indices (goal / agent / padding) in probe index space
    goal_idx = probe.label_to_idx.get(CELL_SYMBOL_TO_ID["G"], CELL_SYMBOL_TO_ID["G"])
    agent_idx = probe.label_to_idx.get(CELL_SYMBOL_TO_ID["A"], CELL_SYMBOL_TO_ID["A"])
    padding_idx = probe.label_to_idx.get(CELL_SYMBOL_TO_ID["+"], CELL_SYMBOL_TO_ID["+"])

    # Initialize class-based accumulators (same as the original command)
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

    # Initialize distance-based accumulators (the new metric)
    global_distance = DistanceAccumulatorSet(num_classes, class_names)
    size_distance: dict[int, DistanceAccumulatorSet] = defaultdict(
        lambda: DistanceAccumulatorSet(num_classes, class_names)
    )
    complexity_distance: dict[float, DistanceAccumulatorSet] = defaultdict(
        lambda: DistanceAccumulatorSet(num_classes, class_names)
    )

    # Process trajectories
    total_trajectories = 0
    total_steps_processed = 0

    def _process_step(activation: torch.Tensor, grid_state: list[str]):
        """Run probe on a step and return (pred_list, gt_list, cell_records)."""
        predictions = _apply_probe_to_all_positions(probe, activation, pad_to_size)
        ground_truth = _get_ground_truth_for_step(grid_state, pad_to_size, probe.label_to_idx)

        pred_list = []
        gt_list = []
        for pos in predictions:
            if pos in ground_truth:
                pred_list.append(predictions[pos])
                gt_list.append(ground_truth[pos])

        goal_pos, agent_pos = _find_special_positions(ground_truth, goal_idx, agent_idx)
        cell_records = _build_distance_records(
            predictions, ground_truth, goal_pos, agent_pos, padding_idx
        )
        return pred_list, gt_list, cell_records

    if single_size_mode:
        # Process trajectories directly from base folder (no size subfolders)
        trajectory_files = sorted(trajectories_base.glob("*.json"))

        if verbose:
            print(f"\nProcessing {len(trajectory_files)} trajectories from {trajectories_base}")

        for traj_file in tqdm(trajectory_files, desc="  trajectories", disable=not verbose):
            with open(traj_file, encoding="utf-8") as f:
                trajectory_data = json.load(f)

            complexity = _extract_complexity_from_trajectory(trajectory_data)

            traj_name = traj_file.stem
            traj_activations_folder = activations_base / traj_name

            if not traj_activations_folder.exists():
                continue

            model_folder = discover_model_folder(traj_activations_folder)
            if model_folder is None:
                continue

            available_layers = discover_available_layers(model_folder)
            selected_layers = parse_index_specification(layers, available_layers)

            if not selected_layers:
                continue

            total_trajectories += 1

            for layer_idx in selected_layers:
                layer_folder = model_folder / f"layer_{layer_idx}"
                if not layer_folder.exists():
                    continue

                available_steps = discover_available_steps(layer_folder)
                selected_steps = parse_index_specification(steps, available_steps)

                for step_idx in selected_steps:
                    if step_idx >= len(trajectory_data.get("steps", [])):
                        continue

                    step_data = trajectory_data["steps"][step_idx]
                    grid_state = step_data.get("grid_state", [])

                    if not grid_state:
                        continue

                    activation = load_concatenated_activations(
                        trajectory_folder=traj_activations_folder,
                        layer_idx=layer_idx,
                        step_idx=step_idx,
                        category_specs=category_specs,
                    )

                    if activation is None:
                        continue

                    pred_list, gt_list, cell_records = _process_step(activation, grid_state)

                    # Class-based accumulators (no size-based metrics in single_size_mode)
                    global_accumulator.update(pred_list, gt_list)
                    complexity_accumulators[complexity].update(pred_list, gt_list)

                    # Distance-based accumulators
                    global_distance.update(cell_records)
                    complexity_distance[complexity].update(cell_records)

                    total_steps_processed += 1
    else:
        # Process each size folder
        for size_folder in size_folders:
            size_name = size_folder.name  # e.g., "size7"
            activations_size_folder = activations_base / size_name

            if not activations_size_folder.exists():
                if verbose:
                    print(f"  Skipping {size_name}: no activations folder")
                continue

            trajectory_files = sorted(size_folder.glob("*.json"))

            if verbose:
                print(f"\nProcessing {size_name}: {len(trajectory_files)} trajectories")

            for traj_file in tqdm(trajectory_files, desc=f"  {size_name}", disable=not verbose):
                with open(traj_file, encoding="utf-8") as f:
                    trajectory_data = json.load(f)

                grid_size = _extract_size_from_trajectory(trajectory_data)
                complexity = _extract_complexity_from_trajectory(trajectory_data)

                traj_name = traj_file.stem
                traj_activations_folder = activations_size_folder / traj_name

                if not traj_activations_folder.exists():
                    continue

                model_folder = discover_model_folder(traj_activations_folder)
                if model_folder is None:
                    continue

                available_layers = discover_available_layers(model_folder)
                selected_layers = parse_index_specification(layers, available_layers)

                if not selected_layers:
                    continue

                total_trajectories += 1

                for layer_idx in selected_layers:
                    layer_folder = model_folder / f"layer_{layer_idx}"
                    if not layer_folder.exists():
                        continue

                    available_steps = discover_available_steps(layer_folder)
                    selected_steps = parse_index_specification(steps, available_steps)

                    for step_idx in selected_steps:
                        if step_idx >= len(trajectory_data.get("steps", [])):
                            continue

                        step_data = trajectory_data["steps"][step_idx]
                        grid_state = step_data.get("grid_state", [])

                        if not grid_state:
                            continue

                        activation = load_concatenated_activations(
                            trajectory_folder=traj_activations_folder,
                            layer_idx=layer_idx,
                            step_idx=step_idx,
                            category_specs=category_specs,
                        )

                        if activation is None:
                            continue

                        pred_list, gt_list, cell_records = _process_step(activation, grid_state)

                        # Class-based accumulators
                        global_accumulator.update(pred_list, gt_list)
                        size_accumulators[grid_size].update(pred_list, gt_list)
                        complexity_accumulators[complexity].update(pred_list, gt_list)
                        size_complexity_accumulators[(grid_size, complexity)].update(pred_list, gt_list)

                        # Distance-based accumulators
                        global_distance.update(cell_records)
                        size_distance[grid_size].update(cell_records)
                        complexity_distance[complexity].update(cell_records)

                        total_steps_processed += 1

    # Compute and print results
    if verbose:
        print(f"\n{'=' * 87}")
        print("EVALUATION COMPLETE")
        print("=" * 87)
        print(f"Processed {total_trajectories} trajectories, {total_steps_processed} steps")
        if single_size_mode:
            print("(Single-size mode: size-based metrics skipped)")

    # Global metrics
    global_metrics = global_accumulator.compute_metrics()
    global_distance_metrics = global_distance.compute()
    if verbose:
        _print_metrics_table(global_metrics, class_names, probe.idx_to_label, "GLOBAL METRICS")
        _print_distance_table(
            global_distance_metrics["by_goal_distance"],
            "goal",
            "GLOBAL: accuracy per distance to GOAL",
        )
        _print_distance_table(
            global_distance_metrics["by_agent_distance"],
            "agent",
            "GLOBAL: accuracy per distance to AGENT",
        )
        print(f"\n{'=' * 87}")
        print("GLOBAL: per-class metrics by distance")
        print("=" * 87)
        _print_per_class_distance_tables(
            global_distance_metrics["by_goal_distance_per_class"],
            "goal",
            probe.idx_to_label,
            "GLOBAL",
        )
        _print_per_class_distance_tables(
            global_distance_metrics["by_agent_distance_per_class"],
            "agent",
            probe.idx_to_label,
            "GLOBAL",
        )

    # Per-size metrics (skip in single_size_mode)
    size_metrics = {}
    size_distance_metrics = {}
    if not single_size_mode:
        if verbose:
            print(f"\n{'=' * 87}")
            print("METRICS BY SIZE")
            print("=" * 87)

        for grid_size in sorted(size_accumulators.keys()):
            metrics = size_accumulators[grid_size].compute_metrics()
            size_metrics[grid_size] = metrics
            dist_metrics = size_distance[grid_size].compute()
            size_distance_metrics[grid_size] = dist_metrics
            if verbose:
                _print_metrics_table(metrics, class_names, probe.idx_to_label, f"Size {grid_size}")
                _print_distance_table(
                    dist_metrics["by_goal_distance"], "goal", f"Size {grid_size}: per distance to GOAL"
                )
                _print_distance_table(
                    dist_metrics["by_agent_distance"], "agent", f"Size {grid_size}: per distance to AGENT"
                )

    # Per-complexity metrics
    complexity_metrics = {}
    complexity_distance_metrics = {}
    if verbose:
        print(f"\n{'=' * 87}")
        print("METRICS BY COMPLEXITY")
        print("=" * 87)

    for complexity in sorted(complexity_accumulators.keys()):
        metrics = complexity_accumulators[complexity].compute_metrics()
        complexity_metrics[complexity] = metrics
        dist_metrics = complexity_distance[complexity].compute()
        complexity_distance_metrics[complexity] = dist_metrics
        if verbose:
            _print_metrics_table(metrics, class_names, probe.idx_to_label, f"Complexity {complexity:.2f}")
            _print_distance_table(
                dist_metrics["by_goal_distance"], "goal", f"Complexity {complexity:.2f}: per distance to GOAL"
            )
            _print_distance_table(
                dist_metrics["by_agent_distance"], "agent", f"Complexity {complexity:.2f}: per distance to AGENT"
            )

    # Build results dictionary
    # class_labels maps every class index used as a key in `per_class` (and in
    # the *_per_class distance pivots) to its grid symbol, so the JSON is
    # self-describing (the probe may cover only a subset of all cell classes).
    class_labels = {str(i): class_names[i] for i in range(num_classes)}

    results = {
        "class_labels": class_labels,
        "global": global_metrics,
        "global_by_distance": global_distance_metrics,
        "by_complexity": {str(k): v for k, v in complexity_metrics.items()},
        "by_complexity_distance": {str(k): v for k, v in complexity_distance_metrics.items()},
        "total_trajectories": total_trajectories,
        "total_steps": total_steps_processed,
        "single_size_mode": single_size_mode,
        "config": {
            "probe_path": str(probe_path_obj),
            "trajectories_dir": str(trajectories_base),
            "activations_dir": str(activations_base),
            "layers": layers,
            "steps": steps,
            "token_categories": active_categories,
            "pad_to_size": pad_to_size,
            "distance_metric": "chebyshev",
        },
    }

    # Add size-based metrics only if not in single_size_mode
    if not single_size_mode:
        results["by_size"] = {str(k): v for k, v in size_metrics.items()}
        results["by_size_distance"] = {str(k): v for k, v in size_distance_metrics.items()}

    # Save JSON results
    if output_path is None:
        probe_name = probe_path_obj.stem
        trajectories_name = trajectories_base.name
        output_path = f"eval_per_distance_{probe_name}_{trajectories_name}.json"

    output_path_obj = Path(output_path)
    with open(output_path_obj, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"\nResults saved to: {output_path_obj}")

    return results
