"""Utility functions for loading activations from nested folder structure."""

import os
import re
from pathlib import Path

import torch


# Cell identity symbol to ID mapping
CELL_SYMBOL_TO_ID = {
    'A': 0,  # Agent
    '#': 1,  # Wall
    'G': 2,  # Goal
    '_': 3,  # Empty
    'D': 4,  # Door
    'K': 5,  # Key
    '?': 6,  # Unknown
    '+': 7,  # Padding
}

# Reverse mapping for convenience
CELL_ID_TO_SYMBOL = {v: k for k, v in CELL_SYMBOL_TO_ID.items()}


def parse_grid_state(
    grid_state: list[str],
    pad_to_size: int | None = None,
) -> list[list[int]]:
    """Parse grid_state strings into [row_id, column_id, cell_identity_id] triples.

    Handles both single-digit and double-digit row/column indices in the grid format.

    Args:
        grid_state: List of strings representing the grid, where the first line
            is the column header and subsequent lines are rows with format
            "row_id cell cell cell ..."
        pad_to_size: If specified, pad the grid to this size using the padding
            symbol '+' (id 7). Padding is added to the right and bottom.

    Returns:
        List of [row_id, column_id, cell_identity_id] triples, ordered row by row,
        column by column.

    Example:
        >>> grid = [
        ...     "  0 1 2 3 4 ",
        ...     "0 # # # # # ",
        ...     "1 # _ _ G # ",
        ...     "2 # _ # # # ",
        ...     "3 # _ A _ # ",
        ...     "4 # # # # # "
        ... ]
        >>> triples = parse_grid_state(grid)
        >>> triples[0]  # First cell (0, 0) is a wall
        [0, 0, 1]
    """
    if not grid_state:
        raise ValueError("grid_state is empty")

    # Skip the header line (column indices)
    data_lines = grid_state[1:]

    # Parse each row
    parsed_cells: list[list[int]] = []
    actual_grid_size = 0

    for line in data_lines:
        line = line.rstrip()
        if not line:
            continue

        # Extract row index and cells
        # The row starts with the row number, followed by cells
        # Use regex to find the row number at the start
        match = re.match(r'^(\d+)\s+', line)
        if not match:
            continue

        row_id = int(match.group(1))
        actual_grid_size = max(actual_grid_size, row_id + 1)

        # Get the rest of the line after the row number
        rest = line[match.end():]

        # Extract cell symbols - they are single characters separated by spaces
        # Split by whitespace and filter to get only valid cell symbols
        tokens = rest.split()
        col_id = 0
        for token in tokens:
            # Each token should be a single character cell symbol
            if len(token) == 1 and token in CELL_SYMBOL_TO_ID:
                cell_id = CELL_SYMBOL_TO_ID[token]
                parsed_cells.append([row_id, col_id, cell_id])
                col_id += 1
                actual_grid_size = max(actual_grid_size, col_id)

    # Determine the target size
    target_size = pad_to_size if pad_to_size is not None else actual_grid_size

    if target_size < actual_grid_size:
        raise ValueError(
            f"pad_to_size ({target_size}) is smaller than actual grid size ({actual_grid_size})"
        )

    # Build a complete grid with padding
    result: list[list[int]] = []
    padding_id = CELL_SYMBOL_TO_ID['+']

    # Create a lookup for existing cells
    cell_lookup = {(cell[0], cell[1]): cell[2] for cell in parsed_cells}

    for row_id in range(target_size):
        for col_id in range(target_size):
            if (row_id, col_id) in cell_lookup:
                cell_id = cell_lookup[(row_id, col_id)]
            else:
                cell_id = padding_id
            result.append([row_id, col_id, cell_id])

    return result


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


def discover_trajectory_folders(activations_dir: Path) -> list[Path]:
    """Discover all trajectory folders in the activations directory.

    Trajectory folders are direct children of activations_dir that contain
    a model subfolder.

    Args:
        activations_dir: Base directory containing trajectory folders

    Returns:
        Sorted list of trajectory folder paths
    """
    trajectory_folders = []
    for item in activations_dir.iterdir():
        if item.is_dir():
            # Check if it has a model subfolder (contains __)
            subfolders = [f for f in item.iterdir() if f.is_dir()]
            if subfolders:
                trajectory_folders.append(item)
    return sorted(trajectory_folders)


def discover_model_folder(trajectory_folder: Path) -> Path | None:
    """Discover the model folder inside a trajectory folder.

    Model folders are identified by containing '__' (e.g., 'openai__gpt-oss-20b').

    Args:
        trajectory_folder: Path to a trajectory folder

    Returns:
        Path to the model folder, or None if not found
    """
    for item in trajectory_folder.iterdir():
        if item.is_dir() and "__" in item.name:
            return item
    # Fallback: return first directory
    for item in trajectory_folder.iterdir():
        if item.is_dir():
            return item
    return None


def discover_available_layers(model_folder: Path) -> list[int]:
    """Discover available layer indices in a model folder.

    Args:
        model_folder: Path to the model folder

    Returns:
        Sorted list of available layer indices
    """
    layers = []
    for item in model_folder.iterdir():
        if item.is_dir() and item.name.startswith("layer_"):
            try:
                layer_idx = int(item.name.replace("layer_", ""))
                layers.append(layer_idx)
            except ValueError:
                continue
    return sorted(layers)


def discover_available_steps(layer_folder: Path) -> list[int]:
    """Discover available step indices in a layer folder.

    Args:
        layer_folder: Path to the layer folder

    Returns:
        Sorted list of available step indices
    """
    steps = []
    for item in layer_folder.iterdir():
        if item.is_dir() and item.name.startswith("step_"):
            try:
                step_idx = int(item.name.replace("step_", ""))
                steps.append(step_idx)
            except ValueError:
                continue
    return sorted(steps)


def discover_available_categories(step_folder: Path) -> list[str]:
    """Discover available categories in a step folder.

    Args:
        step_folder: Path to the step folder

    Returns:
        List of available category names
    """
    return [item.name for item in step_folder.iterdir() if item.is_dir()]


def discover_available_token_indices(category_folder: Path) -> list[int]:
    """Discover available token indices in a category folder.

    Args:
        category_folder: Path to the category folder

    Returns:
        Sorted list of available token indices
    """
    indices = []
    for item in category_folder.iterdir():
        if item.is_file() and item.suffix == ".pt":
            try:
                idx = int(item.stem)
                indices.append(idx)
            except ValueError:
                continue
    return sorted(indices)


def parse_index_specification(spec: str, available_indices: list[int]) -> list[int]:
    """Parse index specification string and filter by available indices.

    Supports:
        - "all" -> all available indices
        - "0,1,5" -> specific indices (filtered by availability)
        - "0:10" -> range (inclusive, filtered by availability)
        - "-1" -> last available index
        - "-3:-1" -> last 3 available indices

    Args:
        spec: Index specification string
        available_indices: List of indices that are actually available

    Returns:
        List of indices to use (subset of available_indices)
    """
    if not available_indices:
        return []

    max_index = max(available_indices) + 1  # For resolving negative indices
    available_set = set(available_indices)

    if spec.strip().lower() == "all":
        return available_indices

    requested_indices = set()

    parts = spec.split(",")
    for part in parts:
        part = part.strip()
        if not part:
            continue

        if ":" in part:
            range_parts = part.split(":")
            if len(range_parts) != 2:
                raise ValueError(f"Invalid range specification: '{part}'")

            start_str, end_str = range_parts
            start_idx = int(start_str.strip())
            end_idx = int(end_str.strip())

            # Resolve negative indices
            if start_idx < 0:
                start_idx = max_index + start_idx
            if end_idx < 0:
                end_idx = max_index + end_idx

            # Add range (inclusive)
            if start_idx <= end_idx:
                for i in range(start_idx, end_idx + 1):
                    if i in available_set:
                        requested_indices.add(i)
            else:
                for i in range(end_idx, start_idx + 1):
                    if i in available_set:
                        requested_indices.add(i)
        else:
            try:
                idx = int(part)
            except ValueError:
                raise ValueError(f"Invalid index specification: '{part}'")

            if idx < 0:
                idx = max_index + idx

            if idx in available_set:
                requested_indices.add(idx)

    return sorted(requested_indices)


def load_activation(file_path: Path) -> torch.Tensor:
    """Load a single activation tensor from a .pt file.

    Args:
        file_path: Path to the .pt file

    Returns:
        Activation tensor
    """
    return torch.load(file_path, map_location="cpu", weights_only=True)


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
