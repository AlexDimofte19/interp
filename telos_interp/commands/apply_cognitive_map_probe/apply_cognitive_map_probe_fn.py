"""Apply trained cognitive map probes to trajectory activations and store predictions in trajectory JSON."""

import json
from pathlib import Path

import torch
from tqdm import tqdm

from telos_interp.commands.prepare_activations_for_probing import (
    CELL_ID_TO_SYMBOL,
)
from telos_interp.commands.prepare_activations_for_probing.prepare_activations_for_probing_utils import (
    discover_available_categories,
    discover_available_layers,
    discover_available_steps,
    discover_available_token_indices,
    discover_model_folder,
    discover_trajectory_folders,
    load_activation,
    parse_index_specification,
)
from telos_interp.commands.train_cognitive_map_probe import CognitiveMapProbe


def _load_concatenated_activations(
    trajectory_folder: Path,
    layer_idx: int,
    step_idx: int,
    category_specs: dict[str, str | None],
) -> tuple[torch.Tensor | None, list[tuple[str, int]]]:
    """Load and concatenate activations from multiple tokens for a layer/step.

    This matches the concatenation order used in prepare_activations_for_probing:
    activations are concatenated in order of categories (prompt_prefix, prompt_suffix,
    grid_state, output), and within each category by token index.

    Args:
        trajectory_folder: Path to trajectory folder in activations directory
        layer_idx: Layer index
        step_idx: Step index
        category_specs: Dict mapping category name to index specification string

    Returns:
        Tuple of:
        - Concatenated activation tensor, or None if no activations found
        - List of (category, token_idx) tuples for tokens that were concatenated
    """
    model_folder = discover_model_folder(trajectory_folder)
    if model_folder is None:
        return None, []

    step_folder = model_folder / f"layer_{layer_idx}" / f"step_{step_idx}"
    if not step_folder.exists():
        return None, []

    all_activations = []
    contributing_tokens: list[tuple[str, int]] = []

    # Process categories in consistent order (matching training data preparation)
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
                contributing_tokens.append((category, token_idx))

    if not all_activations:
        return None, []

    return torch.cat(all_activations, dim=0), contributing_tokens


def _load_probe(probe_path: Path) -> tuple[str, CognitiveMapProbe]:
    """Load a probe from a .pt file.

    Args:
        probe_path: Path to the probe .pt file

    Returns:
        Tuple of (probe_name, CognitiveMapProbe instance)

    Raises:
        ValueError: If probe cannot be loaded
    """
    if not probe_path.exists():
        raise ValueError(f"Probe file does not exist: {probe_path}")
    if not probe_path.suffix == ".pt":
        raise ValueError(f"Probe file must be a .pt file: {probe_path}")

    try:
        probe = CognitiveMapProbe.load(probe_path, device="cpu")
        return probe_path.stem, probe
    except Exception as e:
        raise ValueError(f"Failed to load probe from {probe_path}: {e}") from e


def _get_class_labels_from_probe(probe: CognitiveMapProbe) -> list[str]:
    """Get class labels from probe in order of output indices.

    Args:
        probe: Loaded CognitiveMapProbe instance

    Returns:
        List of class label strings in output order
    """
    # Map output indices to original label IDs, then to symbols
    labels = []
    for idx in range(probe.num_classes):
        original_label_id = probe.idx_to_label[idx]
        symbol = CELL_ID_TO_SYMBOL.get(original_label_id, str(original_label_id))
        # Map symbols to readable names
        symbol_to_name = {
            "A": "agent",
            "#": "wall",
            "G": "goal",
            "_": "empty",
            "D": "door",
            "K": "key",
            "?": "unknown",
            "+": "padding",
        }
        labels.append(symbol_to_name.get(symbol, symbol))
    return labels


def _apply_probe_for_all_positions(
    probe: CognitiveMapProbe,
    probe_name: str,
    activation: torch.Tensor,
    grid_size: int,
    layer_name: str,
) -> dict[str, dict[str, dict[str, float]]]:
    """Apply a probe to an activation for all grid positions.

    The probe expects input with position: [activation..., row_id, col_id].
    Normalization (if the probe was trained with it) is handled internally.

    Args:
        probe: Loaded CognitiveMapProbe instance
        probe_name: Base name of the probe (e.g., "cognitive_map_probe_mlp")
        activation: Activation tensor for a single token (activation_dim,)
        grid_size: Size of the grid (assumes square grid)
        layer_name: Name of the layer for output formatting

    Returns:
        Dictionary: {"{probe_name}_r{row}_c{col}": {layer_name: {class_label: probability, ...}}}
    """
    # Get class labels once
    class_labels = _get_class_labels_from_probe(probe)

    # Generate all positions and create batch input
    num_positions = grid_size * grid_size
    positions = []
    for row in range(grid_size):
        for col in range(grid_size):
            positions.append([row, col])

    # Create batch: replicate activation for all positions
    positions_tensor = torch.tensor(positions, dtype=activation.dtype)  # (N, 2)
    activation_expanded = activation.unsqueeze(0).expand(num_positions, -1)  # (N, activation_dim)
    activations_with_positions = torch.cat([activation_expanded, positions_tensor], dim=1)  # (N, activation_dim + 2)

    # Get class probabilities for all positions at once
    probs = probe.predict_proba(activations_with_positions)  # (N, num_classes)
    probs = probs.cpu()

    # Build result dictionary
    results = {}
    for idx, (row, col) in enumerate(positions):
        position_probs = probs[idx].tolist()
        class_probs = {label: round(prob, 4) for label, prob in zip(class_labels, position_probs, strict=False)}
        position_key = f"{probe_name}_r{row}_c{col}"
        results[position_key] = {layer_name: class_probs}

    return results


def _add_probes_to_token(token: dict, probe_results: dict[str, dict[str, dict[str, float]]]) -> None:
    """Add probe results to a token dictionary.

    Merges new probe results with existing ones if present.

    Args:
        token: Token dictionary to modify
        probe_results: Probe results to add
    """
    if "probes" not in token:
        token["probes"] = {}

    for probe_name, layer_results in probe_results.items():
        if probe_name not in token["probes"]:
            token["probes"][probe_name] = {}

        for layer_name, class_probs in layer_results.items():
            token["probes"][probe_name][layer_name] = class_probs


def _process_trajectory(
    trajectory_name: str,
    activations_dir: Path,
    trajectories_dir: Path,
    probe: CognitiveMapProbe,
    probe_name: str,
    grid_size: int,
    selected_layers: list[int],
    selected_steps: list[int],
    prompt_prefix_indices: str | None,
    prompt_suffix_indices: str | None,
    grid_state_indices: str | None,
    output_indices: str | None,
    verbose: bool,
) -> dict | None:
    """Process a single trajectory, applying probe to selected tokens.

    Args:
        trajectory_name: Name of the trajectory
        activations_dir: Directory containing activation folders
        trajectories_dir: Directory containing trajectory JSONs
        probe: Loaded CognitiveMapProbe instance
        probe_name: Name of the probe (used in output keys)
        grid_size: Size of the grid
        selected_layers: List of layer indices to process
        selected_steps: List of step indices to process
        prompt_prefix_indices: Token indices for prompt_prefix
        prompt_suffix_indices: Token indices for prompt_suffix
        grid_state_indices: Token indices for grid_state
        output_indices: Token indices for output
        verbose: Print progress information

    Returns:
        Modified trajectory dictionary or None if failed
    """
    # Load trajectory JSON
    trajectory_json_path = trajectories_dir / f"{trajectory_name}.json"
    if not trajectory_json_path.exists():
        if verbose:
            print(f"  Skipped: trajectory JSON not found at {trajectory_json_path}")
        return None

    with open(trajectory_json_path) as f:
        trajectory_data = json.load(f)

    trajectory_folder = activations_dir / trajectory_name

    # Build category to index spec mapping
    category_specs = {
        "prompt_prefix": prompt_prefix_indices,
        "prompt_suffix": prompt_suffix_indices,
        "grid_state": grid_state_indices,
        "output": output_indices,
    }
    active_categories = {k: v for k, v in category_specs.items() if v is not None}

    if not active_categories:
        if verbose:
            print("  Warning: No token categories specified")
        return None

    # Process each layer
    for layer_idx in selected_layers:
        model_folder = discover_model_folder(trajectory_folder)
        if model_folder is None:
            continue

        layer_folder = model_folder / f"layer_{layer_idx}"
        if not layer_folder.exists():
            continue

        available_steps = discover_available_steps(layer_folder)
        steps_to_process = [s for s in selected_steps if s in available_steps]

        for step_idx in steps_to_process:
            # Load and concatenate activations from all specified tokens
            concatenated_activation, contributing_tokens = _load_concatenated_activations(
                trajectory_folder=trajectory_folder,
                layer_idx=layer_idx,
                step_idx=step_idx,
                category_specs=category_specs,
            )

            if concatenated_activation is None or not contributing_tokens:
                continue

            # Apply probe once to concatenated activation
            layer_name = f"model.layers.{layer_idx}.output"
            probe_results = _apply_probe_for_all_positions(
                probe=probe,
                probe_name=probe_name,
                activation=concatenated_activation,
                grid_size=grid_size,
                layer_name=layer_name,
            )

            if not probe_results:
                continue

            # Store results on ALL tokens that contributed to the concatenated activation
            for category, token_idx in contributing_tokens:
                _add_probes_to_trajectory_token(
                    trajectory_data=trajectory_data,
                    step_idx=step_idx,
                    category=category,
                    token_idx=token_idx,
                    probe_results=probe_results,
                )

    return trajectory_data


def _add_probes_to_trajectory_token(
    trajectory_data: dict,
    step_idx: int,
    category: str,
    token_idx: int,
    probe_results: dict[str, dict[str, dict[str, float]]],
) -> None:
    """Find and update the correct token in the trajectory data with probe results.

    Args:
        trajectory_data: Full trajectory dictionary
        step_idx: Step index
        category: Token category
        token_idx: Token index within the category
        probe_results: Probe results to add
    """
    # Handle prompt-level tokens (only prompt_prefix lives at prompt level)
    if category == "prompt_prefix":
        tokens = trajectory_data.get("prompt", {}).get("prompt_prefix_tokens", [])
        for token in tokens:
            if token.get("id") == token_idx:
                _add_probes_to_token(token, probe_results)
                return

    # Handle step-level tokens (prompt_suffix uses step-level since activations differ per step)
    steps = trajectory_data.get("steps", [])
    for step in steps:
        if step.get("step_id") != step_idx:
            continue

        # Map category to token key in step
        if category == "grid_state":
            token_key = "grid_state_tokens"
        elif category == "output":
            token_key = "output_tokens"
        elif category == "prompt_suffix":
            token_key = "prompt_suffix_tokens"
        else:
            continue

        tokens = step.get(token_key, [])
        for token in tokens:
            if token.get("id") == token_idx:
                _add_probes_to_token(token, probe_results)
                return


def _infer_grid_size_from_trajectory(trajectory_data: dict) -> int | None:
    """Infer grid size from trajectory JSON.

    Args:
        trajectory_data: Trajectory dictionary

    Returns:
        Grid size (assumes square grid) or None if not found
    """
    grid_params = trajectory_data.get("grid_params", {})
    width = grid_params.get("grid_width")
    height = grid_params.get("grid_height")

    if width is not None and height is not None:
        if width != height:
            print(f"Warning: Non-square grid ({width}x{height}), using max dimension")
        return max(width, height)

    return None


def apply_cognitive_map_probe(
    activations_dir: str,
    trajectories_dir: str,
    probe_path: str,
    grid_size: int | None = None,
    output_dir: str | None = None,
    layers: str = "all",
    steps: str = "all",
    prompt_prefix_indices: str | None = None,
    prompt_suffix_indices: str | None = None,
    grid_state_indices: str | None = None,
    output_indices: str | None = None,
    verbose: bool = False,
) -> None:
    """Apply trained cognitive map probe to trajectory activations.

    Loads a probe from the specified path and applies it to the specified
    tokens in each trajectory. For each token and each grid position, the probe
    is applied with position (row, col) concatenated to the activation vector.
    Results are stored in the trajectory JSON files as a "probes" field in each token.

    The probe results are stored in the following format:
    {
        "probes": {
            "{probe_name}_r{row}_c{col}": {
                "model.layers.{layer}.output": {
                    "class_label_1": probability,
                    "class_label_2": probability,
                    ...
                }
            }
        }
    }

    Args:
        activations_dir: Directory containing trajectory activation folders.
            Expected structure: {activations_dir}/{trajectory_name}/{model_name}/layer_{N}/step_{M}/{category}/{token_id}.pt
        trajectories_dir: Directory containing trajectory JSON files.
            Expected structure: {trajectories_dir}/{trajectory_name}.json
        probe_path: Path to the trained probe .pt file.
        grid_size: Size of the grid (assumes square). If None, inferred from trajectory JSON.
        output_dir: Directory to save modified trajectory JSONs. If None, saves to
            trajectories_dir (overwriting original files).
        layers: Layer indices to extract (e.g., "all", "7", "7,15", "0:10", "-1")
        steps: Step indices to process (e.g., "all", "0", "0:5")
        prompt_prefix_indices: Token indices for prompt_prefix (None = skip)
        prompt_suffix_indices: Token indices for prompt_suffix (None = skip)
        grid_state_indices: Token indices for grid_state (None = skip)
        output_indices: Token indices for output (None = skip, "all" = all available)
        verbose: Print detailed progress information
    """
    activations_dir_path = Path(activations_dir)
    trajectories_dir_path = Path(trajectories_dir)
    probe_file_path = Path(probe_path)

    if not activations_dir_path.exists():
        raise ValueError(f"Activations directory does not exist: {activations_dir_path}")
    if not trajectories_dir_path.exists():
        raise ValueError(f"Trajectories directory does not exist: {trajectories_dir_path}")

    # Set output directory
    if output_dir is not None:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
    else:
        output_dir_path = trajectories_dir_path

    # Check that at least one category is specified
    if all(
        x is None
        for x in [
            prompt_prefix_indices,
            prompt_suffix_indices,
            grid_state_indices,
            output_indices,
        ]
    ):
        raise ValueError(
            "At least one of prompt_prefix_indices, prompt_suffix_indices, "
            "grid_state_indices, or output_indices must be specified"
        )

    # Load probe
    print(f"Loading probe from {probe_file_path}")
    probe_name, probe = _load_probe(probe_file_path)
    print(f"Loaded probe: {probe_name} (input_dim={probe.input_dim}, num_classes={probe.num_classes})")

    # Discover trajectory folders
    trajectory_folders = discover_trajectory_folders(activations_dir_path)
    if not trajectory_folders:
        raise ValueError(f"No trajectory folders found in {activations_dir_path}")

    print(f"Found {len(trajectory_folders)} trajectory folders")

    # Print extraction parameters
    print("\nExtraction parameters:")
    print(f"  layers: {layers}")
    print(f"  steps: {steps}")
    if grid_size is not None:
        print(f"  grid_size: {grid_size}")
    else:
        print("  grid_size: auto (inferred from trajectory)")
    if prompt_prefix_indices:
        print(f"  prompt_prefix_indices: {prompt_prefix_indices}")
    if prompt_suffix_indices:
        print(f"  prompt_suffix_indices: {prompt_suffix_indices}")
    if grid_state_indices:
        print(f"  grid_state_indices: {grid_state_indices}")
    if output_indices:
        print(f"  output_indices: {output_indices}")
    print(f"  output_dir: {output_dir_path}")

    # Process each trajectory
    processed_count = 0
    skipped_count = 0

    for trajectory_folder in tqdm(trajectory_folders, desc="Processing trajectories"):
        trajectory_name = trajectory_folder.name

        if verbose:
            print(f"\nProcessing {trajectory_name}")

        # Determine grid size for this trajectory
        effective_grid_size = grid_size
        if effective_grid_size is None:
            # Try to infer from trajectory JSON
            trajectory_json_path = trajectories_dir_path / f"{trajectory_name}.json"
            if trajectory_json_path.exists():
                with open(trajectory_json_path) as f:
                    trajectory_data = json.load(f)
                effective_grid_size = _infer_grid_size_from_trajectory(trajectory_data)

        if effective_grid_size is None:
            skipped_count += 1
            if verbose:
                print("  Skipped: could not determine grid size")
            continue

        # Discover available layers and steps for this trajectory
        model_folder = discover_model_folder(trajectory_folder)
        if model_folder is None:
            skipped_count += 1
            if verbose:
                print("  Skipped: no model folder found")
            continue

        available_layers = discover_available_layers(model_folder)
        if not available_layers:
            skipped_count += 1
            if verbose:
                print("  Skipped: no layers found")
            continue

        selected_layers = parse_index_specification(layers, available_layers)
        if not selected_layers:
            skipped_count += 1
            if verbose:
                print("  Skipped: no matching layers")
            continue

        # For steps, we need to check each layer's available steps
        # Use the first layer to determine available steps
        first_layer_folder = model_folder / f"layer_{selected_layers[0]}"
        available_steps = discover_available_steps(first_layer_folder)
        if not available_steps:
            skipped_count += 1
            if verbose:
                print("  Skipped: no steps found")
            continue

        selected_steps = parse_index_specification(steps, available_steps)
        if not selected_steps:
            skipped_count += 1
            if verbose:
                print("  Skipped: no matching steps")
            continue

        # Process the trajectory
        modified_trajectory = _process_trajectory(
            trajectory_name=trajectory_name,
            activations_dir=activations_dir_path,
            trajectories_dir=trajectories_dir_path,
            probe=probe,
            probe_name=probe_name,
            grid_size=effective_grid_size,
            selected_layers=selected_layers,
            selected_steps=selected_steps,
            prompt_prefix_indices=prompt_prefix_indices,
            prompt_suffix_indices=prompt_suffix_indices,
            grid_state_indices=grid_state_indices,
            output_indices=output_indices,
            verbose=verbose,
        )

        if modified_trajectory is None:
            skipped_count += 1
            continue

        # Save modified trajectory
        output_json_path = output_dir_path / f"{trajectory_name}.json"
        with open(output_json_path, "w") as f:
            json.dump(modified_trajectory, f, indent=4)

        processed_count += 1

    print(f"\nProcessed: {processed_count}, Skipped: {skipped_count}")
    print(f"Output saved to: {output_dir_path}")
