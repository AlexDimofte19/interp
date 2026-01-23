"""Extract and concatenate activations from nested folder structure for probing."""

import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Literal

import torch
from tqdm import tqdm

from .prepare_activations_for_probing_utils import (
    discover_trajectory_folders,
    generate_output_filename,
    load_activations_for_trajectory,
    parse_grid_state_from_trajectory,
)

# Action name to ID mapping for action_sequence probe type
ACTION_TO_ID = {
    "LEFT": 0,
    "TOP": 1,
    "RIGHT": 2,
    "DOWN": 3,
}

# Probe type literal for type hints
ProbeType = Literal["grid_tile", "distance", "action_sequence"]


def _sample_triples(
    triples: list[list[int]],
    balance_classes: bool,
    max_positions: int | None,
) -> list[list[int]]:
    """Sample triples with optional class balancing and max position limits.

    Args:
        triples: List of [row_id, col_id, cell_identity_id] triples
        balance_classes: If True, ensure equal representation of each cell_identity_id
        max_positions: Maximum number of triples to return. If combined with
            balance_classes, will be adjusted to nearest divisor of num_classes.

    Returns:
        Sampled list of triples
    """
    if not balance_classes and max_positions is None:
        return triples

    if balance_classes:
        # Group triples by cell_identity_id (index 2)
        groups: dict[int, list[list[int]]] = defaultdict(list)
        for triple in triples:
            cell_id = triple[2]
            groups[cell_id].append(triple)

        num_classes = len(groups)
        if num_classes == 0:
            return []

        # Find minimum count across all classes
        min_count = min(len(g) for g in groups.values())

        # Determine samples per class
        if max_positions is not None:
            # Adjust max_positions to be divisible by num_classes
            # Find the largest divisor of num_classes that is <= max_positions
            adjusted_max = (max_positions // num_classes) * num_classes
            if adjusted_max == 0:
                adjusted_max = num_classes  # At least 1 per class
            samples_per_class = min(adjusted_max // num_classes, min_count)
        else:
            samples_per_class = min_count

        # Sample from each class
        result = []
        for cell_id in sorted(groups.keys()):
            group = groups[cell_id]
            if len(group) <= samples_per_class:
                result.extend(group)
            else:
                result.extend(random.sample(group, samples_per_class))

        return result

    else:
        # Only max_positions is specified, no balancing
        if max_positions is not None and len(triples) > max_positions:
            return random.sample(triples, max_positions)
        return triples


def _discover_size_subfolders(base_dir: Path) -> list[Path]:
    """Discover size subfolders (e.g., size5, size7) in the base directory.

    Args:
        base_dir: Base directory to search

    Returns:
        Sorted list of size subfolder paths, or empty list if none found
    """
    size_pattern = re.compile(r"^size\d+$")
    subfolders = []
    for item in base_dir.iterdir():
        if item.is_dir() and size_pattern.match(item.name):
            subfolders.append(item)
    return sorted(subfolders, key=lambda p: int(p.name.replace("size", "")))


def _is_multi_size_mode(activations_dir: Path, trajectories_dir: Path) -> bool:
    """Check if we're in multi-size mode (directories contain size subfolders).

    Args:
        activations_dir: Activations directory
        trajectories_dir: Trajectories directory

    Returns:
        True if both directories contain matching size subfolders
    """
    act_sizes = _discover_size_subfolders(activations_dir)
    traj_sizes = _discover_size_subfolders(trajectories_dir)

    if not act_sizes or not traj_sizes:
        return False

    # Check that they have matching size folders
    act_names = {p.name for p in act_sizes}
    traj_names = {p.name for p in traj_sizes}
    return bool(act_names & traj_names)


def _process_single_folder(
    activations_dir_path: Path,
    trajectories_dir_path: Path,
    layers: str,
    steps: str,
    probe_type: ProbeType,
    grid_step_idx: int,
    pad_to_size: int | None,
    max_positions_per_trajectory: int | None,
    balance_classes_per_trajectory: bool,
    prompt_prefix_indices: str | None,
    prompt_suffix_indices: str | None,
    grid_state_indices: str | None,
    output_indices: str | None,
    verbose: bool,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[str], int, int | None]:
    """Process a single activations/trajectories folder pair.

    Args:
        activations_dir_path: Path to activations folder
        trajectories_dir_path: Path to trajectories folder
        layers: Layer specification
        steps: Steps specification
        probe_type: Type of probe ("grid_tile", "distance", or "action_sequence")
        grid_step_idx: Step index for grid parsing (used only for grid_tile)
        pad_to_size: Padding size for grid (used only for grid_tile)
        max_positions_per_trajectory: Max positions to sample (used only for grid_tile)
        balance_classes_per_trajectory: Whether to balance classes (used only for grid_tile)
        prompt_prefix_indices: Indices for prompt_prefix tokens
        prompt_suffix_indices: Indices for prompt_suffix tokens
        grid_state_indices: Indices for grid_state tokens
        output_indices: Indices for output tokens
        verbose: Print progress info

    Returns:
        Tuple of (all_vectors, all_labels, trajectory_names, skipped_count, extra_info)
        extra_info is num_cells_per_trajectory for grid_tile, None for others
    """
    trajectory_folders = discover_trajectory_folders(activations_dir_path)
    if not trajectory_folders:
        return [], [], [], 0, None

    all_vectors = []
    all_labels = []
    trajectory_names = []
    skipped = 0
    num_cells_first = None  # Only used for grid_tile probe type

    for trajectory_folder in tqdm(trajectory_folders, desc=f"Processing {activations_dir_path.name}"):
        if verbose:
            print(f"\nProcessing {trajectory_folder.name}")

        # Load activation vector
        activation = load_activations_for_trajectory(
            trajectory_folder=trajectory_folder,
            layers=layers,
            steps=steps,
            prompt_prefix_indices=prompt_prefix_indices,
            prompt_suffix_indices=prompt_suffix_indices,
            grid_state_indices=grid_state_indices,
            output_indices=output_indices,
            verbose=verbose,
        )

        if activation is None:
            skipped += 1
            if verbose:
                print("  Skipped: no activations found")
            continue

        # Load corresponding trajectory JSON
        trajectory_json_path = trajectories_dir_path / f"{trajectory_folder.name}.json"
        if not trajectory_json_path.exists():
            skipped += 1
            if verbose:
                print(f"  Skipped: trajectory JSON not found at {trajectory_json_path}")
            continue

        with open(trajectory_json_path) as f:
            trajectory_data = json.load(f)

        # Process based on probe type
        if probe_type == "grid_tile":
            result = _process_grid_tile_trajectory(
                activation=activation,
                trajectory_data=trajectory_data,
                grid_step_idx=grid_step_idx,
                pad_to_size=pad_to_size,
                max_positions_per_trajectory=max_positions_per_trajectory,
                balance_classes_per_trajectory=balance_classes_per_trajectory,
                num_cells_first=num_cells_first,
                verbose=verbose,
            )
            if result is None:
                skipped += 1
                continue
            vectors, labels, num_cells_first = result

        elif probe_type == "distance":
            result = _process_distance_trajectory(
                activation=activation,
                trajectory_data=trajectory_data,
                verbose=verbose,
            )
            if result is None:
                skipped += 1
                continue
            vectors, labels = result

        elif probe_type == "action_sequence":
            result = _process_action_sequence_trajectory(
                activation=activation,
                trajectory_data=trajectory_data,
                verbose=verbose,
            )
            if result is None:
                skipped += 1
                continue
            vectors, labels = result

        else:
            raise ValueError(f"Unknown probe_type: {probe_type}")

        all_vectors.append(vectors)
        all_labels.append(labels)
        trajectory_names.append(trajectory_folder.name)

    return all_vectors, all_labels, trajectory_names, skipped, num_cells_first


def _process_grid_tile_trajectory(
    activation: torch.Tensor,
    trajectory_data: dict,
    grid_step_idx: int,
    pad_to_size: int | None,
    max_positions_per_trajectory: int | None,
    balance_classes_per_trajectory: bool,
    num_cells_first: int | None,
    verbose: bool,
) -> tuple[torch.Tensor, torch.Tensor, int] | None:
    """Process a trajectory for grid_tile probe type.

    Returns:
        Tuple of (vectors, labels, num_cells) or None if should be skipped
    """
    # Parse grid state
    try:
        grid_triples = parse_grid_state_from_trajectory(
            trajectory_data,
            step_idx=grid_step_idx,
            pad_to_size=pad_to_size,
        )
    except (ValueError, KeyError) as e:
        if verbose:
            print(f"  Skipped: failed to parse grid state: {e}")
        return None

    # Apply sampling and/or class balancing
    selected_triples = _sample_triples(
        grid_triples,
        balance_classes=balance_classes_per_trajectory,
        max_positions=max_positions_per_trajectory,
    )

    num_cells = len(selected_triples)

    if num_cells == 0:
        if verbose:
            print("  Skipped: no triples after sampling")
        return None

    # Check consistency of cell counts across trajectories
    if num_cells_first is not None and num_cells != num_cells_first:
        if verbose:
            print(f"  Skipped: inconsistent sample count ({num_cells} cells, expected {num_cells_first})")
        return None

    # Convert triples to tensor: [row_id, col_id, cell_id]
    triples_tensor = torch.tensor(selected_triples, dtype=activation.dtype)  # (N, 3)

    # Split into position (row_id, col_id) and label (cell_id)
    positions_tensor = triples_tensor[:, :2]  # (N, 2) - row_id, col_id
    labels_tensor = triples_tensor[:, 2].to(torch.int64)  # (N,) - cell_id as label

    # Replicate activation N times and concatenate positions
    # Final vector: [activation..., row_id, col_id]
    activation_expanded = activation.unsqueeze(0).expand(num_cells, -1)  # (N, activation_dim)
    vectors_with_positions = torch.cat([activation_expanded, positions_tensor], dim=1)  # (N, activation_dim + 2)

    return vectors_with_positions, labels_tensor, num_cells


def _process_distance_trajectory(
    activation: torch.Tensor,
    trajectory_data: dict,
    verbose: bool,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Process a trajectory for distance probe type.

    Returns:
        Tuple of (vectors, labels) or None if should be skipped
    """
    # Get astar_distance from grid_params
    try:
        grid_params = trajectory_data.get("grid_params", {})
        astar_distance = grid_params.get("astar_distance")
        if astar_distance is None:
            if verbose:
                print("  Skipped: astar_distance not found in grid_params")
            return None
    except (KeyError, TypeError) as e:
        if verbose:
            print(f"  Skipped: failed to get astar_distance: {e}")
        return None

    # Activation stays as-is (single vector per trajectory)
    vectors = activation.unsqueeze(0)  # (1, activation_dim)
    labels = torch.tensor([int(astar_distance)], dtype=torch.int64)  # (1,)

    return vectors, labels


def _process_action_sequence_trajectory(
    activation: torch.Tensor,
    trajectory_data: dict,
    verbose: bool,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Process a trajectory for action_sequence probe type.

    Returns:
        Tuple of (vectors, labels) or None if should be skipped
    """
    # Get action sequence from steps
    try:
        steps = trajectory_data.get("steps", [])
        if not steps:
            if verbose:
                print("  Skipped: no steps found in trajectory")
            return None

        action_ids = []
        for step in steps:
            agent_action = step.get("agent_action")
            if agent_action is None:
                continue
            # Convert action name to ID
            action_id = ACTION_TO_ID.get(agent_action.upper())
            if action_id is None:
                if verbose:
                    print(f"  Warning: unknown action '{agent_action}', skipping")
                continue
            action_ids.append(action_id)

        if not action_ids:
            if verbose:
                print("  Skipped: no valid actions found")
            return None

    except (KeyError, TypeError) as e:
        if verbose:
            print(f"  Skipped: failed to get action sequence: {e}")
        return None

    # Activation stays as-is (single vector per trajectory)
    vectors = activation.unsqueeze(0)  # (1, activation_dim)
    labels = torch.tensor(action_ids, dtype=torch.int64)  # (seq_len,)

    return vectors, labels


def _generate_filename(
    probe_type: ProbeType,
    layers: str,
    steps: str,
    pad_to_size: int | None,
    num_cells: int | None,
    balance_classes_per_trajectory: bool,
    max_positions_per_trajectory: int | None,
    prompt_prefix_indices: str | None,
    prompt_suffix_indices: str | None,
    grid_state_indices: str | None,
    output_indices: str | None,
) -> str:
    """Generate output filename based on extraction parameters."""
    filename = generate_output_filename(
        layers,
        steps,
        prompt_prefix_indices,
        prompt_suffix_indices,
        grid_state_indices,
        output_indices,
    )

    # Add probe type to filename
    filename = filename.replace(".pt", f"_{probe_type}.pt")

    # Add pad_to_size or grid size to filename (only relevant for grid_tile)
    if probe_type == "grid_tile":
        if pad_to_size is not None:
            filename = filename.replace(".pt", f"_pad{pad_to_size}.pt")
        elif num_cells is not None:
            inferred_size = int(num_cells**0.5)
            filename = filename.replace(".pt", f"_grid{inferred_size}.pt")

        # Add sampling info to filename (only relevant for grid_tile)
        if balance_classes_per_trajectory:
            filename = filename.replace(".pt", "_balanced.pt")
        if max_positions_per_trajectory is not None:
            filename = filename.replace(".pt", f"_max{max_positions_per_trajectory}.pt")

    return filename


def prepare_activations_for_probing(
    activations_dir: str,
    trajectories_dir: str,
    probe_type: ProbeType,
    layers: str = "all",
    steps: str = "0",
    pad_to_size: int | None = None,
    max_positions_per_trajectory: int | None = None,
    balance_classes_per_trajectory: bool = False,
    prompt_prefix_indices: str | None = None,
    prompt_suffix_indices: str | None = None,
    grid_state_indices: str | None = None,
    output_indices: str | None = None,
    output_path: str | None = None,
    verbose: bool = False,
    seed: int | None = 42,
) -> None:
    """Extract and concatenate activations from stored trajectory activations.

    Loads activations from the nested folder structure produced by gather_activations,
    then processes them based on the specified probe_type.

    Probe types:
    - "grid_tile": Replicates each activation for every grid cell, concatenates
      [row_id, col_id] to the activation vector, and uses grid_tile_id as label.
      Output: activations (N, activation_dim + 2), labels (N,)
    - "distance": Uses activations as-is (one per trajectory), with the
      astar_distance from grid_params as label.
      Output: activations (num_trajectories, activation_dim), labels (num_trajectories,)
    - "action_sequence": Uses activations as-is (one per trajectory), with the
      sequence of action_ids from agent_action fields as label.
      Actions mapped as: LEFT=0, TOP=1, RIGHT=2, DOWN=3.
      Output: activations (num_trajectories, activation_dim), labels variable-length sequences

    Supports two modes:
    1. Single-folder mode: activations_dir contains trajectory folders directly
    2. Multi-size mode: activations_dir contains size subfolders (size5, size7, etc.)
       In this mode, each size folder is processed and merged into a single output
       file saved in the root activations_dir.

    Folder structure expected (single-folder mode):
        {activations_dir}/{trajectory_name}/{model_name}/layer_{N}/step_{M}/{category}/{token_id}.pt
        {trajectories_dir}/{trajectory_name}.json

    Folder structure expected (multi-size mode):
        {activations_dir}/{sizeN}/{trajectory_name}/{model_name}/layer_{N}/step_{M}/{category}/{token_id}.pt
        {trajectories_dir}/{sizeN}/{trajectory_name}.json

    Args:
        activations_dir: Directory containing trajectory activation folders or size subfolders
        trajectories_dir: Directory containing trajectory JSON files or size subfolders
        probe_type: Type of probe to prepare data for. One of:
            - "grid_tile": Grid cell identity prediction (row_id, col_id concatenated to activation)
            - "distance": A* distance prediction (astar_distance as label)
            - "action_sequence": Action sequence prediction (sequence of action_ids as label)
        layers: Layer indices to extract (e.g., "all", "7", "7,15", "0:10")
        steps: Step index to use for grid state (default: "0"). Only the first step in
            the specification is used for grid parsing (grid_tile mode only).
        pad_to_size: Size to pad the grid to (e.g., 15 for 15x15 grid). If None, uses
            the actual grid size from each trajectory. Only used for grid_tile mode.
        max_positions_per_trajectory: Maximum number of cell positions to sample per
            trajectory. If combined with balance_classes_per_trajectory, will be
            adjusted to the nearest divisor of the number of classes. Only used for grid_tile mode.
        balance_classes_per_trajectory: If True, ensures equal representation of each
            cell_identity_id by finding the minimum count M across all classes and
            selecting M samples per class. Only used for grid_tile mode.
        prompt_prefix_indices: Indices for prompt_prefix tokens (None = skip)
        prompt_suffix_indices: Indices for prompt_suffix tokens (None = skip)
        grid_state_indices: Indices for grid_state tokens (None = skip)
        output_indices: Indices for output tokens (None = skip, "all" = all available)
        output_path: Path for output .pt file (default: auto-generated in activations_dir).
            In multi-size mode, this is used for the merged file only.
        verbose: Print detailed progress information
        seed: Random seed for reproducible sampling (only used for grid_tile mode)
    """
    if seed is not None:
        random.seed(seed)

    activations_dir_path = Path(activations_dir)
    trajectories_dir_path = Path(trajectories_dir)

    if not activations_dir_path.exists():
        raise ValueError(f"Activations directory does not exist: {activations_dir_path}")
    if not trajectories_dir_path.exists():
        raise ValueError(f"Trajectories directory does not exist: {trajectories_dir_path}")

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

    # Parse step index for grid state (use first step in specification)
    grid_step_idx = _parse_first_step_index(steps)

    # Check if we're in multi-size mode
    multi_size = _is_multi_size_mode(activations_dir_path, trajectories_dir_path)

    print("Extraction parameters:")
    print(f"  probe_type: {probe_type}")
    print(f"  layers: {layers}")
    print(f"  steps: {steps}")
    if probe_type == "grid_tile":
        print(f"  grid_step_idx (for grid parsing): {grid_step_idx}")
        print(f"  pad_to_size: {pad_to_size if pad_to_size else 'auto (use actual grid size)'}")
        print(
            f"  max_positions_per_trajectory: {max_positions_per_trajectory if max_positions_per_trajectory else 'all'}"
        )
        print(f"  balance_classes_per_trajectory: {balance_classes_per_trajectory}")
    if seed is not None:
        print(f"  seed: {seed}")
    if prompt_prefix_indices:
        print(f"  prompt_prefix_indices: {prompt_prefix_indices}")
    if prompt_suffix_indices:
        print(f"  prompt_suffix_indices: {prompt_suffix_indices}")
    if grid_state_indices:
        print(f"  grid_state_indices: {grid_state_indices}")
    if output_indices:
        print(f"  output_indices: {output_indices}")
    print(f"  mode: {'multi-size' if multi_size else 'single-folder'}")

    if multi_size:
        _process_multi_size_mode(
            activations_dir_path=activations_dir_path,
            trajectories_dir_path=trajectories_dir_path,
            probe_type=probe_type,
            layers=layers,
            steps=steps,
            grid_step_idx=grid_step_idx,
            pad_to_size=pad_to_size,
            max_positions_per_trajectory=max_positions_per_trajectory,
            balance_classes_per_trajectory=balance_classes_per_trajectory,
            prompt_prefix_indices=prompt_prefix_indices,
            prompt_suffix_indices=prompt_suffix_indices,
            grid_state_indices=grid_state_indices,
            output_indices=output_indices,
            output_path=output_path,
            verbose=verbose,
            seed=seed,
        )
    else:
        _process_single_size_mode(
            activations_dir_path=activations_dir_path,
            trajectories_dir_path=trajectories_dir_path,
            probe_type=probe_type,
            layers=layers,
            steps=steps,
            grid_step_idx=grid_step_idx,
            pad_to_size=pad_to_size,
            max_positions_per_trajectory=max_positions_per_trajectory,
            balance_classes_per_trajectory=balance_classes_per_trajectory,
            prompt_prefix_indices=prompt_prefix_indices,
            prompt_suffix_indices=prompt_suffix_indices,
            grid_state_indices=grid_state_indices,
            output_indices=output_indices,
            output_path=output_path,
            verbose=verbose,
            seed=seed,
        )


def _process_single_size_mode(
    activations_dir_path: Path,
    trajectories_dir_path: Path,
    probe_type: ProbeType,
    layers: str,
    steps: str,
    grid_step_idx: int,
    pad_to_size: int | None,
    max_positions_per_trajectory: int | None,
    balance_classes_per_trajectory: bool,
    prompt_prefix_indices: str | None,
    prompt_suffix_indices: str | None,
    grid_state_indices: str | None,
    output_indices: str | None,
    output_path: str | None,
    verbose: bool,
    seed: int | None,
) -> None:
    """Process a single folder."""
    trajectory_folders = discover_trajectory_folders(activations_dir_path)
    if not trajectory_folders:
        raise ValueError(f"No trajectory folders found in {activations_dir_path}")

    print(f"Found {len(trajectory_folders)} trajectory folders in {activations_dir_path}")

    all_vectors, all_labels, trajectory_names, skipped, num_cells_first = _process_single_folder(
        activations_dir_path=activations_dir_path,
        trajectories_dir_path=trajectories_dir_path,
        layers=layers,
        steps=steps,
        probe_type=probe_type,
        grid_step_idx=grid_step_idx,
        pad_to_size=pad_to_size,
        max_positions_per_trajectory=max_positions_per_trajectory,
        balance_classes_per_trajectory=balance_classes_per_trajectory,
        prompt_prefix_indices=prompt_prefix_indices,
        prompt_suffix_indices=prompt_suffix_indices,
        grid_state_indices=grid_state_indices,
        output_indices=output_indices,
        verbose=verbose,
    )

    if not all_vectors:
        raise ValueError("No activations were extracted from any trajectory")

    # Stack vectors and labels based on probe type
    stacked_vectors = torch.cat(all_vectors, dim=0)

    if probe_type == "action_sequence":
        # For action_sequence, labels are variable-length sequences
        # Pad them to the same length with -1 as padding value
        max_seq_len = max(label.shape[0] for label in all_labels)
        padded_labels = []
        sequence_lengths = []
        for label in all_labels:
            seq_len = label.shape[0]
            sequence_lengths.append(seq_len)
            if seq_len < max_seq_len:
                padding = torch.full((max_seq_len - seq_len,), -1, dtype=torch.int64)
                padded_label = torch.cat([label, padding], dim=0)
            else:
                padded_label = label
            padded_labels.append(padded_label)
        stacked_labels = torch.stack(padded_labels, dim=0)  # (num_trajectories, max_seq_len)
        sequence_lengths_tensor = torch.tensor(sequence_lengths, dtype=torch.int64)
    else:
        stacked_labels = torch.cat(all_labels, dim=0)
        sequence_lengths_tensor = None

    # Compute activation dimension based on probe type
    if probe_type == "grid_tile":
        activation_dim = stacked_vectors.shape[1] - 2  # Subtract 2 for row_id, col_id
    else:
        activation_dim = stacked_vectors.shape[1]

    # Print results
    print(f"\nProcessed {len(all_vectors)} trajectories")
    if probe_type == "grid_tile":
        print(f"Cells per trajectory: {num_cells_first}")
        print(f"Activation dimension: {activation_dim}")
        print(f"Output tensor shape: {stacked_vectors.shape} (activation_dim + row_id + col_id)")
        print(f"Labels tensor shape: {stacked_labels.shape} (grid_tile_id)")
    elif probe_type == "distance":
        print(f"Activation dimension: {activation_dim}")
        print(f"Output tensor shape: {stacked_vectors.shape}")
        print(f"Labels tensor shape: {stacked_labels.shape} (astar_distance)")
    elif probe_type == "action_sequence":
        print(f"Activation dimension: {activation_dim}")
        print(f"Output tensor shape: {stacked_vectors.shape}")
        print(f"Labels tensor shape: {stacked_labels.shape} (action_sequence, padded with -1)")
        print(f"Max sequence length: {stacked_labels.shape[1]}")

    # Determine output path
    if output_path:
        final_output_path = Path(output_path)
    else:
        filename = _generate_filename(
            probe_type=probe_type,
            layers=layers,
            steps=steps,
            pad_to_size=pad_to_size,
            num_cells=num_cells_first,
            balance_classes_per_trajectory=balance_classes_per_trajectory,
            max_positions_per_trajectory=max_positions_per_trajectory,
            prompt_prefix_indices=prompt_prefix_indices,
            prompt_suffix_indices=prompt_suffix_indices,
            grid_state_indices=grid_state_indices,
            output_indices=output_indices,
        )
        final_output_path = activations_dir_path / filename

    # Save
    output_data = {
        "activations": stacked_vectors,
        "labels": stacked_labels,
        "trajectory_names": trajectory_names,
        "activation_dim": activation_dim,
        "probe_type": probe_type,
        "config": {
            "probe_type": probe_type,
            "layers": layers,
            "steps": steps,
            "grid_step_idx": grid_step_idx,
            "pad_to_size": pad_to_size,
            "max_positions_per_trajectory": max_positions_per_trajectory,
            "balance_classes_per_trajectory": balance_classes_per_trajectory,
            "seed": seed,
            "prompt_prefix_indices": prompt_prefix_indices,
            "prompt_suffix_indices": prompt_suffix_indices,
            "grid_state_indices": grid_state_indices,
            "output_indices": output_indices,
        },
    }

    # Add probe-type-specific data
    if probe_type == "grid_tile":
        output_data["num_cells_per_trajectory"] = num_cells_first
    elif probe_type == "action_sequence":
        output_data["sequence_lengths"] = sequence_lengths_tensor
        output_data["action_to_id"] = ACTION_TO_ID

    torch.save(output_data, final_output_path)

    print(f"\nSaved to {final_output_path}")
    print(f"Successfully processed: {len(all_vectors)}, skipped: {skipped}")


def _process_multi_size_mode(
    activations_dir_path: Path,
    trajectories_dir_path: Path,
    probe_type: ProbeType,
    layers: str,
    steps: str,
    grid_step_idx: int,
    pad_to_size: int | None,
    max_positions_per_trajectory: int | None,
    balance_classes_per_trajectory: bool,
    prompt_prefix_indices: str | None,
    prompt_suffix_indices: str | None,
    grid_state_indices: str | None,
    output_indices: str | None,
    output_path: str | None,
    verbose: bool,
    seed: int | None,
) -> None:
    """Process multiple size folders and merge results."""
    act_sizes = _discover_size_subfolders(activations_dir_path)
    traj_sizes = _discover_size_subfolders(trajectories_dir_path)

    # Find matching size folders
    act_names = {p.name: p for p in act_sizes}
    traj_names = {p.name: p for p in traj_sizes}
    common_sizes = sorted(set(act_names.keys()) & set(traj_names.keys()), key=lambda x: int(x.replace("size", "")))

    if not common_sizes:
        raise ValueError("No matching size folders found between activations and trajectories directories")

    # In multi-size mode, automatically pad to max size to ensure all outputs can be merged (grid_tile only)
    if probe_type == "grid_tile" and pad_to_size is None:
        max_size = max(int(s.replace("size", "")) for s in common_sizes)
        pad_to_size = max_size
        print(f"\nAuto-setting pad_to_size={pad_to_size} (max size across folders) for consistent merging")

    print(f"Found {len(common_sizes)} matching size folders: {common_sizes}")

    # Process each size folder
    all_size_results: dict[str, dict] = {}
    merged_vectors = []
    merged_labels = []  # For action_sequence, this will be a list of label tensors per size
    merged_trajectory_names = []
    merged_size_labels = []  # Track which size each entry came from
    total_skipped = 0
    activation_dim = None

    for size_name in common_sizes:
        print(f"\n{'=' * 60}")
        print(f"Processing {size_name}")
        print(f"{'=' * 60}")

        act_path = act_names[size_name]
        traj_path = traj_names[size_name]

        trajectory_folders = discover_trajectory_folders(act_path)
        if not trajectory_folders:
            print(f"  Warning: No trajectory folders found in {act_path}, skipping")
            continue

        print(f"Found {len(trajectory_folders)} trajectory folders")

        all_vectors, all_labels, trajectory_names, skipped, num_cells_first = _process_single_folder(
            activations_dir_path=act_path,
            trajectories_dir_path=traj_path,
            layers=layers,
            steps=steps,
            probe_type=probe_type,
            grid_step_idx=grid_step_idx,
            pad_to_size=pad_to_size,
            max_positions_per_trajectory=max_positions_per_trajectory,
            balance_classes_per_trajectory=balance_classes_per_trajectory,
            prompt_prefix_indices=prompt_prefix_indices,
            prompt_suffix_indices=prompt_suffix_indices,
            grid_state_indices=grid_state_indices,
            output_indices=output_indices,
            verbose=verbose,
        )

        if not all_vectors:
            print(f"  Warning: No activations extracted from {size_name}, skipping")
            total_skipped += skipped
            continue

        # Stack vectors for this size
        stacked_vectors = torch.cat(all_vectors, dim=0)

        # Compute activation dimension based on probe type
        if probe_type == "grid_tile":
            current_activation_dim = stacked_vectors.shape[1] - 2  # Subtract 2 for row_id, col_id
        else:
            current_activation_dim = stacked_vectors.shape[1]

        # Check activation dimension consistency
        if activation_dim is None:
            activation_dim = current_activation_dim
        elif current_activation_dim != activation_dim:
            print(
                f"  Warning: Activation dimension mismatch ({current_activation_dim} vs {activation_dim}), skipping {size_name}"
            )
            total_skipped += skipped + len(all_vectors)
            continue

        print(f"  Processed {len(all_vectors)} trajectories, {stacked_vectors.shape[0]} total samples")
        if probe_type == "grid_tile":
            print(f"  Cells per trajectory: {num_cells_first}")
        print(f"  Skipped: {skipped}")

        # Accumulate for merged output
        all_size_results[size_name] = {
            "num_trajectories": len(all_vectors),
            "num_samples": stacked_vectors.shape[0],
        }
        if probe_type == "grid_tile":
            all_size_results[size_name]["num_cells_per_trajectory"] = num_cells_first

        merged_vectors.append(stacked_vectors)
        merged_labels.extend(all_labels)  # Keep as list for later processing
        merged_trajectory_names.extend([f"{size_name}/{name}" for name in trajectory_names])

        # Create size labels tensor
        size_idx = int(size_name.replace("size", ""))
        size_labels_tensor = torch.full((stacked_vectors.shape[0],), size_idx, dtype=torch.int32)
        merged_size_labels.append(size_labels_tensor)

        total_skipped += skipped

    if not merged_vectors:
        raise ValueError("No activations were extracted from any size folder")

    # Create merged output
    final_vectors = torch.cat(merged_vectors, dim=0)
    final_size_labels = torch.cat(merged_size_labels, dim=0)

    # Handle labels based on probe type
    if probe_type == "action_sequence":
        # For action_sequence, labels are variable-length sequences
        # Pad them to the same length with -1 as padding value
        max_seq_len = max(label.shape[0] for label in merged_labels)
        padded_labels = []
        sequence_lengths = []
        for label in merged_labels:
            seq_len = label.shape[0]
            sequence_lengths.append(seq_len)
            if seq_len < max_seq_len:
                padding = torch.full((max_seq_len - seq_len,), -1, dtype=torch.int64)
                padded_label = torch.cat([label, padding], dim=0)
            else:
                padded_label = label
            padded_labels.append(padded_label)
        final_labels = torch.stack(padded_labels, dim=0)  # (num_trajectories, max_seq_len)
        sequence_lengths_tensor = torch.tensor(sequence_lengths, dtype=torch.int64)
    else:
        final_labels = torch.cat(merged_labels, dim=0)
        sequence_lengths_tensor = None

    print(f"\n{'=' * 60}")
    print("MERGED RESULTS")
    print(f"{'=' * 60}")
    print(f"Total samples: {final_vectors.shape[0]}")
    print(f"Activation dimension: {activation_dim}")
    print(f"Output tensor shape: {final_vectors.shape}")
    print(f"Labels tensor shape: {final_labels.shape}")
    print(f"Size labels tensor shape: {final_size_labels.shape}")
    if probe_type == "action_sequence":
        print(f"Max sequence length: {final_labels.shape[1]}")
    print(f"Total skipped: {total_skipped}")
    print("Per-size breakdown:")
    for size_name, info in all_size_results.items():
        print(f"  {size_name}: {info['num_trajectories']} trajectories, {info['num_samples']} samples")

    # Determine merged output path
    if output_path:
        final_output_path = Path(output_path)
    else:
        filename = _generate_filename(
            probe_type=probe_type,
            layers=layers,
            steps=steps,
            pad_to_size=pad_to_size,
            num_cells=None,  # Don't include grid size for merged file
            balance_classes_per_trajectory=balance_classes_per_trajectory,
            max_positions_per_trajectory=max_positions_per_trajectory,
            prompt_prefix_indices=prompt_prefix_indices,
            prompt_suffix_indices=prompt_suffix_indices,
            grid_state_indices=grid_state_indices,
            output_indices=output_indices,
        )
        # Add "merged" to filename
        filename = filename.replace(".pt", "_merged.pt")
        final_output_path = activations_dir_path / filename

    # Save merged output
    merged_data = {
        "activations": final_vectors,
        "labels": final_labels,
        "size_labels": final_size_labels,
        "trajectory_names": merged_trajectory_names,
        "activation_dim": activation_dim,
        "probe_type": probe_type,
        "sizes": list(all_size_results.keys()),
        "per_size_info": all_size_results,
        "config": {
            "probe_type": probe_type,
            "layers": layers,
            "steps": steps,
            "grid_step_idx": grid_step_idx,
            "pad_to_size": pad_to_size,
            "max_positions_per_trajectory": max_positions_per_trajectory,
            "balance_classes_per_trajectory": balance_classes_per_trajectory,
            "seed": seed,
            "prompt_prefix_indices": prompt_prefix_indices,
            "prompt_suffix_indices": prompt_suffix_indices,
            "grid_state_indices": grid_state_indices,
            "output_indices": output_indices,
        },
    }

    # Add probe-type-specific data
    if probe_type == "action_sequence":
        merged_data["sequence_lengths"] = sequence_lengths_tensor
        merged_data["action_to_id"] = ACTION_TO_ID

    torch.save(merged_data, final_output_path)

    print(f"\nSaved merged output to {final_output_path}")


def _parse_first_step_index(steps: str) -> int:
    """Parse the first step index from a step specification string.

    Args:
        steps: Step specification (e.g., "0", "all", "0:5", "0,3,5")

    Returns:
        The first step index (0 if "all" is specified)
    """
    steps = steps.strip().lower()
    if steps == "all":
        return 0

    # Handle comma-separated list
    if "," in steps:
        first_part = steps.split(",")[0].strip()
        return int(first_part)

    # Handle range
    if ":" in steps:
        start = steps.split(":")[0].strip()
        return int(start)

    # Single value
    return int(steps)
