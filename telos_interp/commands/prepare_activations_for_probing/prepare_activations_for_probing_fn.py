"""Extract and concatenate activations from nested folder structure for probing.

Output format (v3): a directory containing
  manifest.json
  activations/{trajectory_name}.pt              # single-size mode
  activations/size{N}/{trajectory_name}.pt      # multi-size mode

Each per-trajectory .pt holds a single (D,) activation tensor — there is no
per-cell replication on disk. Per-cell positions/labels (for grid_tile) and
other probe-type metadata live in manifest.json. Trainers consume this format
through `manifest_loader.py`.
"""

import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Literal

import torch
from tqdm import tqdm

from .prepare_activations_for_probing_utils import (
    discover_output_token_files,
    discover_trajectory_folders,
    generate_output_filename,
    load_activation,
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

# Action name to ID mapping for the next_action probe type. The trajectory data uses
# "UP" (not "TOP"); "TOP" is kept as an alias mapping to the same id. Canonical id->name
# is {0: LEFT, 1: UP, 2: RIGHT, 3: DOWN}.
NEXT_ACTION_TO_ID = {
    "LEFT": 0,
    "UP": 1,
    "TOP": 1,
    "RIGHT": 2,
    "DOWN": 3,
}

# Probe type literal for type hints
ProbeType = Literal["grid_tile", "distance", "action_sequence", "next_action"]

# Manifest format version. v1 = legacy single .pt; v3 = manifest dir.
PREPARED_FORMAT_VERSION = 3


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
    output_acts_dir: Path,
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
    size_name: str | None,
    manifest_root: Path,
    activations_root: Path,
    verbose: bool,
) -> tuple[list[dict], int, int | None, int | None, int | None]:
    """Process trajectories in a single folder; write per-trajectory .pt files.

    Args:
        activations_dir_path: Folder containing trajectory activation subfolders.
        trajectories_dir_path: Folder containing trajectory JSON files.
        output_acts_dir: Where to write per-trajectory .pt files (the
            `{output_dir}/activations[/size_name]/` directory).
        size_name: Multi-size folder name (e.g., "size5"); None in single-size mode.
        manifest_root: The output directory containing manifest.json. Used to
            compute `act_path` relative to the manifest's parent.
        activations_root: The top-level activations directory. For the `next_action`
            mode, token `act_path`s are stored relative to this (no files are copied).
        Other args: see `prepare_activations_for_probing`.

    Returns:
        Tuple of (manifest_entries, skipped_count, activation_dim,
                  num_cells_per_trajectory_or_None, max_seq_len_or_None)
    """
    # next_action references existing token files in place, so it writes no activations.
    if probe_type != "next_action":
        output_acts_dir.mkdir(parents=True, exist_ok=True)

    trajectory_folders = discover_trajectory_folders(activations_dir_path)
    if not trajectory_folders:
        return [], 0, None, None, None

    manifest_entries: list[dict] = []
    skipped = 0
    activation_dim: int | None = None
    num_cells_first: int | None = None       # grid_tile only
    max_seq_len_seen: int = 0                # action_sequence only

    desc = f"Processing {activations_dir_path.name}"
    for trajectory_folder in tqdm(trajectory_folders, desc=desc):
        if verbose:
            print(f"\nProcessing {trajectory_folder.name}")

        # 1. Load and validate trajectory JSON.
        trajectory_json_path = trajectories_dir_path / f"{trajectory_folder.name}.json"
        if not trajectory_json_path.exists():
            skipped += 1
            if verbose:
                print(f"  Skipped: trajectory JSON not found at {trajectory_json_path}")
            continue
        with open(trajectory_json_path, encoding="utf-8") as f:
            trajectory_data = json.load(f)

        # next_action mode: emit one manifest entry per gathered EOS token, each
        # referencing the existing token .pt file (no new files written). All tokens
        # from this trajectory share the trajectory's agent_action label.
        if probe_type == "next_action":
            steps_list = trajectory_data.get("steps", [])
            if grid_step_idx >= len(steps_list):
                skipped += 1
                if verbose:
                    print(f"  Skipped: step {grid_step_idx} out of range ({len(steps_list)} steps)")
                continue
            agent_action = steps_list[grid_step_idx].get("agent_action")
            action_id = NEXT_ACTION_TO_ID.get(agent_action.upper()) if agent_action else None
            if action_id is None:
                skipped += 1
                if verbose:
                    print(f"  Skipped: missing/unknown agent_action '{agent_action}'")
                continue

            token_files = discover_output_token_files(
                trajectory_folder=trajectory_folder,
                layers=layers,
                steps=steps,
                output_indices=output_indices,
                verbose=verbose,
            )
            if not token_files:
                skipped += 1
                if verbose:
                    print("  Skipped: no output token activations found")
                continue

            for layer_idx, step_idx, token_idx, abs_path in token_files:
                if activation_dim is None:
                    activation_dim = int(load_activation(abs_path).shape[0])
                entry = {
                    "name": trajectory_folder.name,
                    "act_path": abs_path.relative_to(activations_root).as_posix(),
                    "label": action_id,
                    "layer": layer_idx,
                    "step": step_idx,
                    "token_id": token_idx,
                    "category": "output",
                }
                if size_name is not None:
                    entry["size"] = int(size_name.replace("size", ""))
                manifest_entries.append(entry)
            continue

        # 2. Extract probe-type-specific labels.
        per_type_fields: dict
        num_cells_this: int | None = None

        if probe_type == "grid_tile":
            try:
                grid_triples = parse_grid_state_from_trajectory(
                    trajectory_data,
                    step_idx=grid_step_idx,
                    pad_to_size=pad_to_size,
                )
            except (ValueError, KeyError) as e:
                skipped += 1
                if verbose:
                    print(f"  Skipped: failed to parse grid state: {e}")
                continue

            selected_triples = _sample_triples(
                grid_triples,
                balance_classes=balance_classes_per_trajectory,
                max_positions=max_positions_per_trajectory,
            )
            num_cells_this = len(selected_triples)

            if num_cells_this == 0:
                skipped += 1
                if verbose:
                    print("  Skipped: no triples after sampling")
                continue

            if num_cells_first is None:
                num_cells_first = num_cells_this
            elif num_cells_this != num_cells_first:
                skipped += 1
                if verbose:
                    print(
                        f"  Skipped: inconsistent sample count "
                        f"({num_cells_this} cells, expected {num_cells_first})"
                    )
                continue

            positions = [[int(t[0]), int(t[1])] for t in selected_triples]
            labels = [int(t[2]) for t in selected_triples]
            per_type_fields = {"positions": positions, "labels": labels}

        elif probe_type == "distance":
            grid_params = trajectory_data.get("grid_params", {})
            astar_distance = grid_params.get("astar_distance")
            if astar_distance is None:
                skipped += 1
                if verbose:
                    print("  Skipped: astar_distance not found in grid_params")
                continue
            per_type_fields = {"astar_distance": int(astar_distance)}

        elif probe_type == "action_sequence":
            steps_list = trajectory_data.get("steps", [])
            if not steps_list:
                skipped += 1
                if verbose:
                    print("  Skipped: no steps found in trajectory")
                continue

            action_ids: list[int] = []
            for step in steps_list:
                agent_action = step.get("agent_action")
                if agent_action is None:
                    continue
                action_id = ACTION_TO_ID.get(agent_action.upper())
                if action_id is None:
                    if verbose:
                        print(f"  Warning: unknown action '{agent_action}', skipping")
                    continue
                action_ids.append(action_id)

            if not action_ids:
                skipped += 1
                if verbose:
                    print("  Skipped: no valid actions found")
                continue

            max_seq_len_seen = max(max_seq_len_seen, len(action_ids))
            per_type_fields = {"actions": action_ids}

        else:
            raise ValueError(f"Unknown probe_type: {probe_type}")

        # 3. Load the (D,) activation tensor.
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

        # 4. Activation dim consistency check.
        if activation_dim is None:
            activation_dim = activation.shape[0]
        elif activation.shape[0] != activation_dim:
            skipped += 1
            if verbose:
                print(
                    f"  Skipped: activation dim mismatch "
                    f"(got {activation.shape[0]}, expected {activation_dim})"
                )
            continue

        # 5. Save per-trajectory activation file.
        if size_name is not None:
            act_rel = Path("activations") / size_name / f"{trajectory_folder.name}.pt"
        else:
            act_rel = Path("activations") / f"{trajectory_folder.name}.pt"
        act_abs = manifest_root / act_rel
        act_abs.parent.mkdir(parents=True, exist_ok=True)
        torch.save(activation, act_abs)

        # 6. Build manifest entry.
        entry: dict = {
            "name": trajectory_folder.name,
            "act_path": act_rel.as_posix(),
            **per_type_fields,
        }
        if size_name is not None:
            entry["size"] = int(size_name.replace("size", ""))
        manifest_entries.append(entry)

    return (
        manifest_entries,
        skipped,
        activation_dim,
        num_cells_first if probe_type == "grid_tile" else None,
        max_seq_len_seen if probe_type == "action_sequence" else None,
    )


def _generate_dirname(
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
    """Generate output directory name based on extraction parameters."""
    # Use existing helper, then strip the .pt extension since v3 outputs a directory.
    filename = generate_output_filename(
        layers,
        steps,
        prompt_prefix_indices,
        prompt_suffix_indices,
        grid_state_indices,
        output_indices,
    )

    # Add probe type
    filename = filename.replace(".pt", f"_{probe_type}.pt")

    # Grid-specific suffixes
    if probe_type == "grid_tile":
        if pad_to_size is not None:
            filename = filename.replace(".pt", f"_pad{pad_to_size}.pt")
        elif num_cells is not None:
            inferred_size = int(num_cells**0.5)
            filename = filename.replace(".pt", f"_grid{inferred_size}.pt")

        if balance_classes_per_trajectory:
            filename = filename.replace(".pt", "_balanced.pt")
        if max_positions_per_trajectory is not None:
            filename = filename.replace(".pt", f"_max{max_positions_per_trajectory}.pt")

    # Strip the .pt extension — v3 outputs a directory.
    if filename.endswith(".pt"):
        filename = filename[:-3]

    return filename


def _resolve_output_dir(output_path: str | None, default_parent: Path, default_name: str) -> Path:
    """Resolve the output directory. Strips .pt suffix with a warning if user passes one."""
    if output_path is None:
        return default_parent / default_name

    candidate = Path(output_path)
    if candidate.suffix == ".pt":
        print(
            f"WARNING: output_path '{candidate}' ends with .pt, but v3 prepared data is a "
            f"directory. Stripping the .pt suffix; output will be written to a directory."
        )
        candidate = candidate.with_suffix("")
    return candidate


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
    """Extract activations and build a v3 prepared-data manifest directory.

    Loads activations from the nested folder structure produced by gather_activations,
    writes one (D,) activation file per trajectory, and emits a manifest.json describing
    the per-cell labels (grid_tile), astar_distance (distance), or action sequence
    (action_sequence) for each trajectory.

    Probe types:
    - "grid_tile": For each trajectory, manifest stores per-cell positions and labels;
      the trainer assembles [activation, row, col] -> grid_tile_id training rows lazily.
    - "distance": One activation per trajectory; label is astar_distance from grid_params.
    - "action_sequence": One activation per trajectory; label is the agent action sequence
      mapped via {LEFT=0, TOP=1, RIGHT=2, DOWN=3}.
    - "next_action": One sample per gathered EOS token (in the `output` category), each
      referencing the existing token .pt file (no files are copied) and labeled with the
      trajectory's agent_action mapped via {LEFT=0, UP=1, RIGHT=2, DOWN=3}. Requires
      output_indices to be set and assumes a single layer. Gather with output_indices="eos".

    Modes:
    1. Single-folder mode: activations_dir contains trajectory folders directly.
    2. Multi-size mode: activations_dir contains size subfolders (size5, size7, etc.).
       In this mode each size folder is processed and entries are merged into a single
       manifest, with activations namespaced under activations/sizeN/.

    Output directory layout (single-size mode):
        {output_dir}/manifest.json
        {output_dir}/activations/{trajectory_name}.pt

    Output directory layout (multi-size mode):
        {output_dir}/manifest.json
        {output_dir}/activations/sizeN/{trajectory_name}.pt

    Args:
        activations_dir: Directory containing trajectory activation folders or size subfolders.
        trajectories_dir: Directory containing trajectory JSON files or size subfolders.
        probe_type: One of "grid_tile", "distance", "action_sequence".
        layers: Layer indices to extract (e.g., "all", "7", "7,15", "0:10").
        steps: Step index to use (default "0"); only the first step in the spec is used
            for grid parsing in grid_tile mode.
        pad_to_size: Pad grid to this size (e.g., 15). grid_tile only.
        max_positions_per_trajectory: Cap cells sampled per trajectory. grid_tile only.
        balance_classes_per_trajectory: Equal-class sampling per trajectory. grid_tile only.
        prompt_prefix_indices, prompt_suffix_indices, grid_state_indices, output_indices:
            Token category index specs (None to skip; "all" for all available on output).
        output_path: Output directory. Defaults to an auto-named directory under
            `activations_dir`. If a path ending in .pt is passed, the .pt is stripped
            with a warning (v3 outputs a directory).
        verbose: Print per-trajectory progress.
        seed: Random seed for sampling determinism.
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

    # next_action reads only the EOS reasoning tokens in the `output` category.
    if probe_type == "next_action":
        if output_indices is None:
            raise ValueError(
                "probe_type='next_action' requires output_indices to be set (e.g. 'all'); "
                "the reasoning-chain EOS tokens live in the 'output' category."
            )
        if any(x is not None for x in [prompt_prefix_indices, prompt_suffix_indices, grid_state_indices]):
            print(
                "WARNING: probe_type='next_action' only uses output_indices; "
                "prompt_prefix/prompt_suffix/grid_state indices are ignored."
            )

    # For distance probes, enforce steps="0" since astar_distance is only valid for step 0
    if probe_type == "distance" and steps != "0":
        print("WARNING: probe_type='distance' requires steps='0' (astar_distance is only valid for initial state).")
        print(f"         Overriding steps='{steps}' to steps='0'")
        steps = "0"

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


def _build_loading_spec(
    layers: str,
    steps: str,
    prompt_prefix_indices: str | None,
    prompt_suffix_indices: str | None,
    grid_state_indices: str | None,
    output_indices: str | None,
) -> dict:
    """Echo the activation-loading spec into the manifest for reproducibility."""
    return {
        "layers": layers,
        "steps": steps,
        "prompt_prefix_indices": prompt_prefix_indices,
        "prompt_suffix_indices": prompt_suffix_indices,
        "grid_state_indices": grid_state_indices,
        "output_indices": output_indices,
    }


def _build_config(
    probe_type: ProbeType,
    layers: str,
    steps: str,
    grid_step_idx: int,
    pad_to_size: int | None,
    max_positions_per_trajectory: int | None,
    balance_classes_per_trajectory: bool,
    seed: int | None,
    prompt_prefix_indices: str | None,
    prompt_suffix_indices: str | None,
    grid_state_indices: str | None,
    output_indices: str | None,
) -> dict:
    return {
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
    }


def _write_manifest(manifest: dict, manifest_path: Path) -> None:
    """Write manifest.json. Uses no indentation to keep the file small for large T."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)


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
    """Process a single folder; emit manifest.json + per-trajectory .pt files."""
    trajectory_folders = discover_trajectory_folders(activations_dir_path)
    if not trajectory_folders:
        raise ValueError(f"No trajectory folders found in {activations_dir_path}")

    print(f"Found {len(trajectory_folders)} trajectory folders in {activations_dir_path}")

    # Determine output directory.
    default_name = _generate_dirname(
        probe_type=probe_type,
        layers=layers,
        steps=steps,
        pad_to_size=pad_to_size,
        num_cells=None,
        balance_classes_per_trajectory=balance_classes_per_trajectory,
        max_positions_per_trajectory=max_positions_per_trajectory,
        prompt_prefix_indices=prompt_prefix_indices,
        prompt_suffix_indices=prompt_suffix_indices,
        grid_state_indices=grid_state_indices,
        output_indices=output_indices,
    )
    output_dir = _resolve_output_dir(output_path, activations_dir_path, default_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_acts_dir = output_dir / "activations"

    manifest_entries, skipped, activation_dim, num_cells_per_trajectory, max_seq_len = _process_single_folder(
        activations_dir_path=activations_dir_path,
        trajectories_dir_path=trajectories_dir_path,
        output_acts_dir=output_acts_dir,
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
        size_name=None,
        manifest_root=output_dir,
        activations_root=activations_dir_path,
        verbose=verbose,
    )

    if not manifest_entries:
        raise ValueError("No activations were extracted from any trajectory")

    # Build manifest dict.
    manifest: dict = {
        "format_version": PREPARED_FORMAT_VERSION,
        "probe_type": probe_type,
        "activation_dim": activation_dim,
        "loading_spec": _build_loading_spec(
            layers, steps, prompt_prefix_indices, prompt_suffix_indices, grid_state_indices, output_indices
        ),
        "config": _build_config(
            probe_type=probe_type,
            layers=layers,
            steps=steps,
            grid_step_idx=grid_step_idx,
            pad_to_size=pad_to_size,
            max_positions_per_trajectory=max_positions_per_trajectory,
            balance_classes_per_trajectory=balance_classes_per_trajectory,
            seed=seed,
            prompt_prefix_indices=prompt_prefix_indices,
            prompt_suffix_indices=prompt_suffix_indices,
            grid_state_indices=grid_state_indices,
            output_indices=output_indices,
        ),
    }
    if probe_type == "next_action":
        # Token-level samples referencing existing files; nothing is copied.
        manifest["activations_root"] = str(activations_dir_path.resolve())
        manifest["action_to_id"] = NEXT_ACTION_TO_ID
        manifest["samples"] = manifest_entries
    else:
        manifest["trajectories"] = manifest_entries
        if probe_type == "grid_tile":
            manifest["num_cells_per_trajectory"] = num_cells_per_trajectory
        elif probe_type == "action_sequence":
            manifest["max_seq_len"] = max_seq_len
            manifest["action_to_id"] = ACTION_TO_ID

    # Write manifest.
    manifest_path = output_dir / "manifest.json"
    _write_manifest(manifest, manifest_path)

    # Print summary.
    if probe_type == "next_action":
        print(f"\nCollected {len(manifest_entries)} token samples")
    else:
        print(f"\nProcessed {len(manifest_entries)} trajectories")
    print(f"Activation dimension: {activation_dim}")
    if probe_type == "grid_tile":
        print(f"Cells per trajectory: {num_cells_per_trajectory}")
    elif probe_type == "action_sequence":
        print(f"Max sequence length: {max_seq_len}")
    print(f"\nSaved to {output_dir}")
    print(f"  manifest: {manifest_path}")
    if probe_type != "next_action":
        print(f"  per-trajectory activations: {output_acts_dir}")
    print(f"Successfully processed: {len(manifest_entries)}, skipped: {skipped}")


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
    """Process multiple size folders and merge results into one manifest."""
    act_sizes = _discover_size_subfolders(activations_dir_path)
    traj_sizes = _discover_size_subfolders(trajectories_dir_path)

    act_names = {p.name: p for p in act_sizes}
    traj_names = {p.name: p for p in traj_sizes}
    common_sizes = sorted(set(act_names.keys()) & set(traj_names.keys()), key=lambda x: int(x.replace("size", "")))

    if not common_sizes:
        raise ValueError("No matching size folders found between activations and trajectories directories")

    # Auto-pad to max size for grid_tile so per-cell counts agree across sizes.
    if probe_type == "grid_tile" and pad_to_size is None:
        max_size = max(int(s.replace("size", "")) for s in common_sizes)
        pad_to_size = max_size
        print(f"\nAuto-setting pad_to_size={pad_to_size} (max size across folders) for consistent merging")

    print(f"Found {len(common_sizes)} matching size folders: {common_sizes}")

    # Determine output directory (merged).
    default_name = _generate_dirname(
        probe_type=probe_type,
        layers=layers,
        steps=steps,
        pad_to_size=pad_to_size,
        num_cells=None,
        balance_classes_per_trajectory=balance_classes_per_trajectory,
        max_positions_per_trajectory=max_positions_per_trajectory,
        prompt_prefix_indices=prompt_prefix_indices,
        prompt_suffix_indices=prompt_suffix_indices,
        grid_state_indices=grid_state_indices,
        output_indices=output_indices,
    ) + "_merged"
    output_dir = _resolve_output_dir(output_path, activations_dir_path, default_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_size_results: dict[str, dict] = {}
    merged_entries: list[dict] = []
    total_skipped = 0
    activation_dim: int | None = None
    num_cells_per_trajectory: int | None = None
    max_seq_len_seen: int = 0

    for size_name in common_sizes:
        print(f"\n{'=' * 60}")
        print(f"Processing {size_name}")
        print(f"{'=' * 60}")

        act_path = act_names[size_name]
        traj_path = traj_names[size_name]
        output_acts_dir = output_dir / "activations" / size_name

        size_entries, skipped, size_act_dim, size_num_cells, size_max_seq_len = _process_single_folder(
            activations_dir_path=act_path,
            trajectories_dir_path=traj_path,
            output_acts_dir=output_acts_dir,
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
            size_name=size_name,
            manifest_root=output_dir,
            activations_root=activations_dir_path,
            verbose=verbose,
        )

        if not size_entries:
            print(f"  Warning: No activations extracted from {size_name}, skipping")
            total_skipped += skipped
            continue

        # Activation-dim consistency.
        if activation_dim is None:
            activation_dim = size_act_dim
        elif size_act_dim != activation_dim:
            print(
                f"  Warning: Activation dimension mismatch ({size_act_dim} vs {activation_dim}); "
                f"dropping {size_name}"
            )
            total_skipped += skipped + len(size_entries)
            # Best-effort cleanup: drop the per-trajectory files we already wrote.
            for entry in size_entries:
                act_file = output_dir / entry["act_path"]
                if act_file.exists():
                    act_file.unlink()
            continue

        # Cells-per-trajectory consistency (grid_tile only).
        if probe_type == "grid_tile":
            if num_cells_per_trajectory is None:
                num_cells_per_trajectory = size_num_cells
            elif size_num_cells != num_cells_per_trajectory:
                print(
                    f"  Warning: num_cells_per_trajectory mismatch "
                    f"({size_num_cells} vs {num_cells_per_trajectory}); dropping {size_name}"
                )
                total_skipped += skipped + len(size_entries)
                for entry in size_entries:
                    act_file = output_dir / entry["act_path"]
                    if act_file.exists():
                        act_file.unlink()
                continue

        if probe_type == "action_sequence" and size_max_seq_len is not None:
            max_seq_len_seen = max(max_seq_len_seen, size_max_seq_len)

        size_summary: dict = {"num_trajectories": len(size_entries)}
        if probe_type == "grid_tile":
            size_summary["num_cells_per_trajectory"] = size_num_cells
        all_size_results[size_name] = size_summary

        merged_entries.extend(size_entries)
        total_skipped += skipped

        print(f"  Processed {len(size_entries)} trajectories")
        if probe_type == "grid_tile":
            print(f"  Cells per trajectory: {size_num_cells}")
        print(f"  Skipped: {skipped}")

    if not merged_entries:
        raise ValueError("No activations were extracted from any size folder")

    # Build the merged manifest.
    manifest: dict = {
        "format_version": PREPARED_FORMAT_VERSION,
        "probe_type": probe_type,
        "activation_dim": activation_dim,
        "sizes": list(all_size_results.keys()),
        "per_size_info": all_size_results,
        "loading_spec": _build_loading_spec(
            layers, steps, prompt_prefix_indices, prompt_suffix_indices, grid_state_indices, output_indices
        ),
        "config": _build_config(
            probe_type=probe_type,
            layers=layers,
            steps=steps,
            grid_step_idx=grid_step_idx,
            pad_to_size=pad_to_size,
            max_positions_per_trajectory=max_positions_per_trajectory,
            balance_classes_per_trajectory=balance_classes_per_trajectory,
            seed=seed,
            prompt_prefix_indices=prompt_prefix_indices,
            prompt_suffix_indices=prompt_suffix_indices,
            grid_state_indices=grid_state_indices,
            output_indices=output_indices,
        ),
    }
    if probe_type == "next_action":
        manifest["activations_root"] = str(activations_dir_path.resolve())
        manifest["action_to_id"] = NEXT_ACTION_TO_ID
        manifest["samples"] = merged_entries
    else:
        manifest["trajectories"] = merged_entries
        if probe_type == "grid_tile":
            manifest["num_cells_per_trajectory"] = num_cells_per_trajectory
        elif probe_type == "action_sequence":
            manifest["max_seq_len"] = max_seq_len_seen
            manifest["action_to_id"] = ACTION_TO_ID

    manifest_path = output_dir / "manifest.json"
    _write_manifest(manifest, manifest_path)

    print(f"\n{'=' * 60}")
    print("MERGED RESULTS")
    print(f"{'=' * 60}")
    print(f"Total trajectories: {len(merged_entries)}")
    print(f"Activation dimension: {activation_dim}")
    if probe_type == "grid_tile":
        print(f"Cells per trajectory: {num_cells_per_trajectory}")
    elif probe_type == "action_sequence":
        print(f"Max sequence length: {max_seq_len_seen}")
    print(f"Total skipped: {total_skipped}")
    print("Per-size breakdown:")
    for size_name, info in all_size_results.items():
        print(f"  {size_name}: {info['num_trajectories']} trajectories")
    print(f"\nSaved merged output to {output_dir}")
    print(f"  manifest: {manifest_path}")


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
