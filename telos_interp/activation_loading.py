"""Activation discovery and loading from nested folder structure."""

from pathlib import Path
from typing import Literal, overload

import torch


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
    for part_raw in parts:
        part = part_raw.strip()
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
            except ValueError as e:
                raise ValueError(f"Invalid index specification: '{part}'") from e

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


def parse_category_specs(
    prompt_prefix_indices: str | None = None,
    prompt_suffix_indices: str | None = None,
    grid_state_indices: str | None = None,
    output_indices: str | None = None,
) -> dict[str, str | None]:
    """Build category specs dict from individual index parameters.

    Args:
        prompt_prefix_indices: Token indices for prompt_prefix (e.g., "all", "-1")
        prompt_suffix_indices: Token indices for prompt_suffix (e.g., "-3:-1")
        grid_state_indices: Token indices for grid_state (e.g., "all")
        output_indices: Token indices for output (e.g., "all", "0:10")

    Returns:
        Dict mapping category names to their index specifications
    """
    return {
        "prompt_prefix": prompt_prefix_indices,
        "prompt_suffix": prompt_suffix_indices,
        "grid_state": grid_state_indices,
        "output": output_indices,
    }


@overload
def load_concatenated_activations(
    trajectory_folder: Path,
    layer_idx: int,
    step_idx: int,
    category_specs: dict[str, str | None],
    track_contributing_tokens: Literal[False] = ...,
) -> torch.Tensor | None: ...


@overload
def load_concatenated_activations(
    trajectory_folder: Path,
    layer_idx: int,
    step_idx: int,
    category_specs: dict[str, str | None],
    track_contributing_tokens: Literal[True] = ...,
) -> tuple[torch.Tensor | None, list[tuple[str, int]]]: ...


def load_concatenated_activations(
    trajectory_folder: Path,
    layer_idx: int,
    step_idx: int,
    category_specs: dict[str, str | None],
    track_contributing_tokens: bool = False,
) -> torch.Tensor | None | tuple[torch.Tensor | None, list[tuple[str, int]]]:
    """Load and concatenate activations from multiple tokens for a layer/step.

    Args:
        trajectory_folder: Path to trajectory folder in activations directory
        layer_idx: Layer index
        step_idx: Step index
        category_specs: Dict mapping category name to index specification string
        track_contributing_tokens: If True, return a tuple of (tensor, contributing_tokens)
            where contributing_tokens is a list of (category, token_idx) tuples.
            If False, return just the tensor (or None).

    Returns:
        If track_contributing_tokens is False:
            Concatenated activation tensor, or None if no activations found.
        If track_contributing_tokens is True:
            Tuple of (tensor_or_None, list_of_contributing_tokens).
    """
    model_folder = discover_model_folder(trajectory_folder)
    if model_folder is None:
        return (None, []) if track_contributing_tokens else None

    step_folder = model_folder / f"layer_{layer_idx}" / f"step_{step_idx}"
    if not step_folder.exists():
        return (None, []) if track_contributing_tokens else None

    all_activations = []
    contributing_tokens: list[tuple[str, int]] = []

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
                if track_contributing_tokens:
                    contributing_tokens.append((category, token_idx))

    if not all_activations:
        return (None, []) if track_contributing_tokens else None

    result_tensor = torch.cat(all_activations, dim=0)

    if track_contributing_tokens:
        return result_tensor, contributing_tokens
    return result_tensor
