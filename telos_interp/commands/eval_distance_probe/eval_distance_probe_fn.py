"""Evaluate distance regression probes on test trajectories with detailed metrics."""

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
from telos_interp.commands.train_distance_probe import DistanceProbe


class RegressionMetricsAccumulator:
    """Accumulates regression predictions and labels for computing metrics."""

    def __init__(self):
        self.predictions = []
        self.labels = []

    def update(self, preds: list[float], labels: list[float]) -> None:
        """Add a batch of predictions and labels."""
        self.predictions.extend(preds)
        self.labels.extend(labels)

    def compute_metrics(self) -> dict:
        """Compute regression metrics from accumulated data."""
        if not self.predictions:
            return {
                "mse": 0.0,
                "mae": 0.0,
                "rmse": 0.0,
                "r2": 0.0,
                "num_samples": 0,
            }

        preds = torch.tensor(self.predictions)
        labels = torch.tensor(self.labels)

        mse = ((preds - labels) ** 2).mean().item()
        mae = (preds - labels).abs().mean().item()
        rmse = mse**0.5

        # R² score
        ss_res = ((labels - preds) ** 2).sum().item()
        ss_tot = ((labels - labels.mean()) ** 2).sum().item()
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "num_samples": len(preds),
        }


def _extract_distance_from_trajectory(trajectory_data: dict) -> int | None:
    """Extract astar_distance from trajectory data."""
    grid_params = trajectory_data.get("grid_params", {})
    return grid_params.get("astar_distance")


def _extract_size_from_trajectory(trajectory_data: dict) -> int:
    """Extract grid size from trajectory data."""
    grid_params = trajectory_data.get("grid_params", {})
    return grid_params.get("size", 0)


def _extract_complexity_from_trajectory(trajectory_data: dict) -> float:
    """Extract complexity from trajectory data."""
    grid_params = trajectory_data.get("grid_params", {})
    return grid_params.get("complexity", 0.0)


def _print_metrics_table(metrics: dict, title: str) -> None:
    """Print a formatted metrics table."""
    print(f"\n{title}")
    print("-" * 60)
    print(f"{'Metric':<20} {'Value':>15}")
    print("-" * 60)
    print(f"{'MSE':<20} {metrics['mse']:>15.4f}")
    print(f"{'MAE':<20} {metrics['mae']:>15.4f}")
    print(f"{'RMSE':<20} {metrics['rmse']:>15.4f}")
    print(f"{'R²':<20} {metrics['r2']:>15.4f}")
    print(f"{'Num Samples':<20} {metrics['num_samples']:>15}")
    print("-" * 60)


def eval_distance_probe(  # noqa: PLR0912
    trajectories_dir: str,
    activations_dir: str,
    probe_path: str,
    layers: str = "all",
    steps: str = "all",
    prompt_prefix_indices: str | None = None,
    prompt_suffix_indices: str | None = None,
    grid_state_indices: str | None = None,
    output_indices: str | None = None,
    output_path: str | None = None,
    verbose: bool = True,
) -> dict:
    """Evaluate a distance regression probe on test trajectories.

    Loads a trained distance regression probe and evaluates it on test trajectories,
    computing regression metrics (MSE, MAE, RMSE, R²) globally and broken down by
    grid size and complexity.

    Args:
        trajectories_dir: Directory containing test trajectory JSON files
            (with size subfolders like size7/, size9/, etc.)
        activations_dir: Directory containing activation folders
            (with matching size subfolders)
        probe_path: Path to trained DistanceProbe .pt file
        layers: Layer indices to use (e.g., "all", "7", "7,15")
        steps: Step indices to use (e.g., "all", "0", "0:5")
        prompt_prefix_indices: Token indices for prompt_prefix
        prompt_suffix_indices: Token indices for prompt_suffix
        grid_state_indices: Token indices for grid_state
        output_indices: Token indices for output
        output_path: Path to save results JSON (optional)
        verbose: Print detailed progress

    Returns:
        Dictionary containing evaluation results
    """
    probe_path_obj = Path(probe_path)
    if not probe_path_obj.exists():
        raise FileNotFoundError(f"Probe not found: {probe_path}")

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # For distance probes, enforce steps="0" since astar_distance is only valid for step 0
    if steps != "0":
        print(
            "WARNING: Distance probe evaluation requires steps='0' (astar_distance is only valid for initial state)."
        )
        print(f"         Overriding steps='{steps}' to steps='0'")
        steps = "0"

    # Load probe
    print(f"Loading probe from {probe_path}")
    probe = DistanceProbe.load(probe_path, device=device)
    print(f"Loaded probe: {probe_path_obj.stem}")
    print(f"  Input dimension: {probe.input_dim}")
    print(f"  Normalized: {probe.normalized}")

    # Parse category specifications
    category_specs = parse_category_specs(
        prompt_prefix_indices=prompt_prefix_indices,
        prompt_suffix_indices=prompt_suffix_indices,
        grid_state_indices=grid_state_indices,
        output_indices=output_indices,
    )

    active_categories = {k: v for k, v in category_specs.items() if v is not None}
    if verbose:
        print(f"\nToken categories: {active_categories}")

    # Find all size folders or use direct folder
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

    # Initialize accumulators
    global_accumulator = RegressionMetricsAccumulator()
    size_accumulators: dict[int, RegressionMetricsAccumulator] = defaultdict(RegressionMetricsAccumulator)
    complexity_accumulators: dict[float, RegressionMetricsAccumulator] = defaultdict(RegressionMetricsAccumulator)
    distance_accumulators: dict[int, RegressionMetricsAccumulator] = defaultdict(RegressionMetricsAccumulator)

    # Process trajectories
    total_trajectories = 0
    skipped_trajectories = 0

    if single_size_mode:
        # Process trajectories directly from base folder
        trajectory_files = sorted(trajectories_base.glob("*.json"))

        if verbose:
            print(f"\nProcessing {len(trajectory_files)} trajectories from {trajectories_base}")

        for traj_file in tqdm(trajectory_files, desc="  trajectories", disable=not verbose):
            # Load trajectory
            with open(traj_file) as f:
                trajectory_data = json.load(f)

            # Extract metadata
            grid_size = _extract_size_from_trajectory(trajectory_data)
            complexity = _extract_complexity_from_trajectory(trajectory_data)
            true_distance = _extract_distance_from_trajectory(trajectory_data)

            if true_distance is None:
                skipped_trajectories += 1
                continue

            # Find corresponding activations folder
            traj_name = traj_file.stem
            traj_activations_folder = activations_base / traj_name

            if not traj_activations_folder.exists():
                skipped_trajectories += 1
                continue

            # Discover available layers
            model_folder = discover_model_folder(traj_activations_folder)
            if model_folder is None:
                skipped_trajectories += 1
                continue

            available_layers = discover_available_layers(model_folder)
            selected_layers = parse_index_specification(layers, available_layers)

            if not selected_layers:
                skipped_trajectories += 1
                continue

            # Process each layer and step
            for layer_idx in selected_layers:
                layer_folder = model_folder / f"layer_{layer_idx}"
                if not layer_folder.exists():
                    continue

                available_steps = discover_available_steps(layer_folder)
                selected_steps = parse_index_specification(steps, available_steps)

                for step_idx in selected_steps:
                    # Load concatenated activations
                    activation = load_concatenated_activations(
                        trajectory_folder=traj_activations_folder,
                        layer_idx=layer_idx,
                        step_idx=step_idx,
                        category_specs=category_specs,
                    )

                    if activation is None:
                        continue

                    # Get prediction
                    pred = probe.predict(activation.unsqueeze(0)).item()

                    # Update accumulators
                    global_accumulator.update([pred], [float(true_distance)])
                    complexity_accumulators[complexity].update([pred], [float(true_distance)])
                    distance_accumulators[true_distance].update([pred], [float(true_distance)])

            total_trajectories += 1
    else:
        # Process each size folder
        for size_folder in size_folders:
            size_name = size_folder.name
            activations_size_folder = activations_base / size_name

            if not activations_size_folder.exists():
                if verbose:
                    print(f"  Skipping {size_name}: no activations folder")
                continue

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
                true_distance = _extract_distance_from_trajectory(trajectory_data)

                if true_distance is None:
                    skipped_trajectories += 1
                    continue

                # Find corresponding activations folder
                traj_name = traj_file.stem
                traj_activations_folder = activations_size_folder / traj_name

                if not traj_activations_folder.exists():
                    skipped_trajectories += 1
                    continue

                # Discover available layers
                model_folder = discover_model_folder(traj_activations_folder)
                if model_folder is None:
                    skipped_trajectories += 1
                    continue

                available_layers = discover_available_layers(model_folder)
                selected_layers = parse_index_specification(layers, available_layers)

                if not selected_layers:
                    skipped_trajectories += 1
                    continue

                # Process each layer and step
                for layer_idx in selected_layers:
                    layer_folder = model_folder / f"layer_{layer_idx}"
                    if not layer_folder.exists():
                        continue

                    available_steps = discover_available_steps(layer_folder)
                    selected_steps = parse_index_specification(steps, available_steps)

                    for step_idx in selected_steps:
                        # Load concatenated activations
                        activation = load_concatenated_activations(
                            trajectory_folder=traj_activations_folder,
                            layer_idx=layer_idx,
                            step_idx=step_idx,
                            category_specs=category_specs,
                        )

                        if activation is None:
                            continue

                        # Get prediction
                        pred = probe.predict(activation.unsqueeze(0)).item()

                        # Update accumulators
                        global_accumulator.update([pred], [float(true_distance)])
                        size_accumulators[grid_size].update([pred], [float(true_distance)])
                        complexity_accumulators[complexity].update([pred], [float(true_distance)])
                        distance_accumulators[true_distance].update([pred], [float(true_distance)])

                total_trajectories += 1

    # Compute and print results
    if verbose:
        print(f"\n{'=' * 60}")
        print("EVALUATION COMPLETE")
        print("=" * 60)
        print(f"Processed {total_trajectories} trajectories, skipped {skipped_trajectories}")
        if single_size_mode:
            print("(Single-size mode: size-based metrics skipped)")

    # Global metrics
    global_metrics = global_accumulator.compute_metrics()
    if verbose:
        _print_metrics_table(global_metrics, "GLOBAL METRICS")

    # Per-size metrics (skip in single_size_mode)
    size_metrics = {}
    if not single_size_mode:
        if verbose:
            print(f"\n{'=' * 60}")
            print("METRICS BY SIZE")
            print("=" * 60)

        for grid_size in sorted(size_accumulators.keys()):
            metrics = size_accumulators[grid_size].compute_metrics()
            size_metrics[grid_size] = metrics
            if verbose:
                _print_metrics_table(metrics, f"Size {grid_size}")

    # Per-complexity metrics
    complexity_metrics = {}
    if verbose:
        print(f"\n{'=' * 60}")
        print("METRICS BY COMPLEXITY")
        print("=" * 60)

    for complexity in sorted(complexity_accumulators.keys()):
        metrics = complexity_accumulators[complexity].compute_metrics()
        complexity_metrics[complexity] = metrics
        if verbose:
            _print_metrics_table(metrics, f"Complexity {complexity:.2f}")

    # Per-distance metrics
    distance_metrics = {}
    if verbose:
        print(f"\n{'=' * 60}")
        print("METRICS BY TRUE DISTANCE")
        print("=" * 60)
        print(f"{'Distance':<10} {'MAE':>10} {'RMSE':>10} {'Count':>10}")
        print("-" * 40)

    for distance in sorted(distance_accumulators.keys()):
        metrics = distance_accumulators[distance].compute_metrics()
        distance_metrics[distance] = metrics
        if verbose:
            print(f"{distance:<10} {metrics['mae']:>10.4f} {metrics['rmse']:>10.4f} {metrics['num_samples']:>10}")

    if verbose:
        print("-" * 40)

    # Build results dictionary
    results = {
        "global": global_metrics,
        "by_complexity": {str(k): v for k, v in complexity_metrics.items()},
        "by_true_distance": {str(k): v for k, v in distance_metrics.items()},
        "total_trajectories": total_trajectories,
        "skipped_trajectories": skipped_trajectories,
        "single_size_mode": single_size_mode,
        "config": {
            "probe_path": str(probe_path_obj),
            "trajectories_dir": str(trajectories_base),
            "activations_dir": str(activations_base),
            "layers": layers,
            "steps": steps,
            "token_categories": active_categories,
        },
    }

    # Add size-based metrics only if not in single_size_mode
    if not single_size_mode:
        results["by_size"] = {str(k): v for k, v in size_metrics.items()}

    # Save results if output path specified
    if output_path:
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path_obj, "w") as f:
            json.dump(results, f, indent=2)
        if verbose:
            print(f"\nResults saved to {output_path_obj}")

    return results
