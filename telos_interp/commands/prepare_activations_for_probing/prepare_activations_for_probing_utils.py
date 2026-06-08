"""Utility functions for loading activations from nested folder structure."""

from pathlib import Path

import torch

from telos_interp.activation_loading import (  # noqa: F401
    discover_available_categories,
    discover_available_layers,
    discover_available_steps,
    discover_available_token_indices,
    discover_model_folder,
    discover_trajectory_folders,
    load_activation,
    parse_index_specification,
)
from telos_interp.grid_utils import (  # noqa: F401
    CELL_ID_TO_SYMBOL,
    CELL_SYMBOL_TO_ID,
    parse_grid_state,
)


def parse_grid_state_from_trajectory(
    trajectory: dict,
    step_idx: int = 0,
    pad_to_size: int | None = None,
) -> list[list[int]]:
    """Parse grid_state from a trajectory dictionary at a specific step.

    Args:
        trajectory: Parsed trajectory JSON dictionary
        step_idx: Index of the step to extract grid state from
        pad_to_size: If specified, pad the grid to this size

    Returns:
        List of [row_id, column_id, cell_identity_id] triples
    """
    if "steps" not in trajectory:
        raise ValueError("Trajectory does not contain 'steps' key")

    if step_idx >= len(trajectory["steps"]):
        raise ValueError(f"Step index {step_idx} out of range (max: {len(trajectory['steps']) - 1})")

    step = trajectory["steps"][step_idx]

    if "grid_state" not in step:
        raise ValueError(f"Step {step_idx} does not contain 'grid_state' key")

    return parse_grid_state(step["grid_state"], pad_to_size=pad_to_size)


def load_activations_for_trajectory(
    trajectory_folder: Path,
    layers: str,
    steps: str,
    prompt_prefix_indices: str | None,
    prompt_suffix_indices: str | None,
    grid_state_indices: str | None,
    output_indices: str | None,
    verbose: bool = False,
) -> torch.Tensor | None:
    """Load and concatenate activations for a single trajectory.

    Args:
        trajectory_folder: Path to the trajectory folder
        layers: Layer indices specification
        steps: Step indices specification
        prompt_prefix_indices: Indices for prompt_prefix tokens (None = skip)
        prompt_suffix_indices: Indices for prompt_suffix tokens (None = skip)
        grid_state_indices: Indices for grid_state tokens (None = skip)
        output_indices: Indices for output tokens (None = skip)
        verbose: Print progress information

    Returns:
        Concatenated activation tensor, or None if no activations found
    """
    model_folder = discover_model_folder(trajectory_folder)
    if model_folder is None:
        if verbose:
            print(f"  Warning: No model folder found in {trajectory_folder}")
        return None

    available_layers = discover_available_layers(model_folder)
    if not available_layers:
        if verbose:
            print(f"  Warning: No layers found in {model_folder}")
        return None

    selected_layers = parse_index_specification(layers, available_layers)
    if not selected_layers:
        if verbose:
            print(f"  Warning: No matching layers for spec '{layers}'")
        return None

    # Mapping from category name to index specification
    category_specs = {
        "prompt_prefix": prompt_prefix_indices,
        "prompt_suffix": prompt_suffix_indices,
        "grid_state": grid_state_indices,
        "output": output_indices,
    }

    # Filter out None specifications
    active_categories = {k: v for k, v in category_specs.items() if v is not None}

    if not active_categories:
        if verbose:
            print("  Warning: No token categories specified")
        return None

    all_activations = []

    for layer_idx in selected_layers:
        layer_folder = model_folder / f"layer_{layer_idx}"
        if not layer_folder.exists():
            continue

        available_steps = discover_available_steps(layer_folder)
        if not available_steps:
            continue

        selected_steps = parse_index_specification(steps, available_steps)
        if not selected_steps:
            continue

        for step_idx in selected_steps:
            step_folder = layer_folder / f"step_{step_idx}"
            if not step_folder.exists():
                continue

            for category, index_spec in active_categories.items():
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

    # Concatenate all activations into a single vector
    return torch.cat(all_activations, dim=0)


def generate_output_filename(
    layers: str,
    steps: str,
    prompt_prefix_indices: str | None,
    prompt_suffix_indices: str | None,
    grid_state_indices: str | None,
    output_indices: str | None,
) -> str:
    """Generate a descriptive output filename based on the extraction parameters.

    Args:
        layers: Layer indices specification
        steps: Step indices specification
        prompt_prefix_indices: Indices for prompt_prefix tokens
        prompt_suffix_indices: Indices for prompt_suffix tokens
        grid_state_indices: Indices for grid_state tokens
        output_indices: Indices for output tokens

    Returns:
        Filename string (without path)
    """
    parts = ["cognitive_map_activations"]

    # Add layer info
    layers_clean = layers.replace(":", "-").replace(",", "_")
    parts.append(f"l{layers_clean}")

    # Add step info
    steps_clean = steps.replace(":", "-").replace(",", "_")
    parts.append(f"s{steps_clean}")

    # Add category info
    if prompt_prefix_indices:
        idx_clean = prompt_prefix_indices.replace(":", "-").replace(",", "_")
        parts.append(f"prefix_{idx_clean}")
    if prompt_suffix_indices:
        idx_clean = prompt_suffix_indices.replace(":", "-").replace(",", "_")
        parts.append(f"suffix_{idx_clean}")
    if grid_state_indices:
        idx_clean = grid_state_indices.replace(":", "-").replace(",", "_")
        parts.append(f"grid_{idx_clean}")
    if output_indices:
        idx_clean = output_indices.replace(":", "-").replace(",", "_")
        parts.append(f"output_{idx_clean}")

    return "_".join(parts) + ".pt"


def discover_output_token_files(
    trajectory_folder: Path,
    layers: str,
    steps: str,
    output_indices: str,
    verbose: bool = False,
) -> list[tuple[int, int, int, Path]]:
    """Enumerate gathered `output`-category token activation files for a trajectory.

    Used by the `next_action` probe mode, where each gathered token activation is a
    separate training sample (no concatenation). Only paths are enumerated here — the
    caller is responsible for loading tensors. The `output` category is assumed to hold
    the reasoning-chain EOS tokens (gather with `output_indices="eos"`).

    Args:
        trajectory_folder: Path to the trajectory folder.
        layers: Layer index specification (e.g., "all", "7"). A single layer is expected.
        steps: Step index specification (e.g., "0").
        output_indices: Token index specification within the `output` category (e.g., "all").
        verbose: Print progress information.

    Returns:
        List of (layer_idx, step_idx, token_idx, abs_path) tuples, sorted by
        (layer, step, token). Empty if nothing is found.
    """
    model_folder = discover_model_folder(trajectory_folder)
    if model_folder is None:
        if verbose:
            print(f"  Warning: No model folder found in {trajectory_folder}")
        return []

    available_layers = discover_available_layers(model_folder)
    if not available_layers:
        if verbose:
            print(f"  Warning: No layers found in {model_folder}")
        return []

    selected_layers = parse_index_specification(layers, available_layers)
    if not selected_layers:
        if verbose:
            print(f"  Warning: No matching layers for spec '{layers}'")
        return []

    results: list[tuple[int, int, int, Path]] = []

    for layer_idx in selected_layers:
        layer_folder = model_folder / f"layer_{layer_idx}"
        if not layer_folder.exists():
            continue

        available_steps = discover_available_steps(layer_folder)
        if not available_steps:
            continue

        selected_steps = parse_index_specification(steps, available_steps)
        if not selected_steps:
            continue

        for step_idx in selected_steps:
            category_folder = layer_folder / f"step_{step_idx}" / "output"
            if not category_folder.exists():
                continue

            available_tokens = discover_available_token_indices(category_folder)
            if not available_tokens:
                continue

            selected_tokens = parse_index_specification(output_indices, available_tokens)
            for token_idx in selected_tokens:
                file_path = category_folder / f"{token_idx}.pt"
                if file_path.exists():
                    results.append((layer_idx, step_idx, token_idx, file_path))

    return sorted(results, key=lambda r: (r[0], r[1], r[2]))
