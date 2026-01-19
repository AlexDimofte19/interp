"""Extract activations from trajectory JSON files produced by get_trajectory."""

import json
from glob import glob
from pathlib import Path

import torch
from nnterp import StandardizedTransformer
from tqdm import tqdm

from .gather_activations_utils import (
    parse_index_specification,
    sanitize_model_id,
    extract_activations_single_pass,
    build_truncated_input,
    save_activations_to_files,
    resolve_token_indices
)


def extract_activations_from_trajectories(
    trajectory_paths: list[str],
    output_dir: str = "out",
    layers: str = "all",
    steps: str = "all",
    prompt_prefix_indices: str | None = None,
    prompt_suffix_indices: str | None = None,
    grid_state_indices: str | None = None,
    output_indices: str | None = None,
    device_map: str = "auto",
    torch_dtype: str = "auto",
    debug: bool = False,
) -> None:
    """Main extraction function for trajectory JSON files.

    Processes trajectory JSON files and extracts activations for specified
    token categories, indices, and layers. Uses efficient extraction:
    - prompt_prefix is extracted once per trajectory (activations are step-independent)
    - Step-dependent categories (grid_state, prompt_suffix, output) are extracted
      in a single forward pass per step

    Args:
        trajectory_paths: Paths to trajectory JSON files (glob patterns supported)
        output_dir: Base output directory
        layers: Layer indices to extract (e.g., "all", "0:10", "-1")
        steps: Step indices to extract (e.g., "all", "0:10", "-1"). Uses same format as layers.
            Out-of-bounds indices are clamped, so "0:10" selects up to 10 steps if available.
        prompt_prefix_indices: Indices for prompt_prefix tokens (None = skip)
        prompt_suffix_indices: Indices for prompt_suffix tokens (None = skip)
        grid_state_indices: Indices for grid_state tokens (None = skip)
        output_indices: Indices for output tokens (None = skip)
        device_map: Device mapping for model loading
        torch_dtype: Torch dtype for model ("auto", "bfloat16", "float16")
        debug: If True, print the first truncated input text to verify format
    """
    # Expand glob patterns
    all_paths = []
    for pattern in trajectory_paths:
        expanded = glob(pattern)
        if expanded:
            all_paths.extend(expanded)
        elif Path(pattern).exists():
            all_paths.append(pattern)

    if not all_paths:
        raise ValueError(f"No files found matching patterns: {trajectory_paths}")

    print(f"Found {len(all_paths)} trajectory file(s) to process")

    # Load first trajectory to get model info
    with open(all_paths[0]) as f:
        first_traj = json.load(f)

    model_id = first_traj["model_params"]["model_id"]
    print(f"Loading model: {model_id}")

    # Determine torch_dtype
    resolved_dtype = torch_dtype
    if resolved_dtype == "auto":
        if "20b" in model_id.lower() or "70b" in model_id.lower():
            resolved_dtype = torch.bfloat16
        else:
            resolved_dtype = None
    elif resolved_dtype == "bfloat16":
        resolved_dtype = torch.bfloat16
    elif resolved_dtype == "float16":
        resolved_dtype = torch.float16
    else:
        resolved_dtype = None

    # Load model using nnterp
    model = StandardizedTransformer(model_id, device_map=device_map, torch_dtype=resolved_dtype)

    num_layers = model.config.num_hidden_layers
    print(f"Model has {num_layers} layers")

    layer_indices = parse_index_specification(layers, num_layers)
    print(f"Extracting from layers: {layer_indices}")

    # Check which categories are requested
    has_prefix = prompt_prefix_indices is not None
    has_suffix = prompt_suffix_indices is not None
    has_grid = grid_state_indices is not None
    has_output = output_indices is not None

    if not any([has_prefix, has_suffix, has_grid, has_output]):
        print("Warning: No token categories specified. Nothing to extract.")
        return

    # Process each trajectory file
    for traj_path in tqdm(all_paths, desc="Processing trajectory files"):
        with open(traj_path) as f:
            trajectory = json.load(f)

        # Verify model_id matches
        traj_model_id = trajectory["model_params"]["model_id"]
        if traj_model_id != model_id:
            raise ValueError(
                f"Model mismatch in {traj_path}. Expected {model_id}, got {traj_model_id}. "
                "All trajectory files must use the same model."
            )

        # Create output directory structure
        file_stem = Path(traj_path).stem
        sanitized_model = sanitize_model_id(model_id)
        output_base = Path(output_dir) / file_stem / sanitized_model

        # Skip if output folder already exists
        if output_base.exists():
            continue

        # Get shared token data
        prefix_tokens = trajectory["prompt"]["prompt_prefix_tokens"]
        suffix_tokens = trajectory["prompt"]["prompt_suffix_tokens"]
        n_prefix = len(prefix_tokens)
        n_suffix = len(suffix_tokens)
        num_steps = len(trajectory["steps"])

        # Parse which steps to extract (clamp=True allows "0:10" to work even if fewer steps exist)
        step_indices = parse_index_specification(steps, num_steps, clamp=True)
        steps_to_process = [trajectory["steps"][i] for i in step_indices]

        print(f"\n  Processing {file_stem} ({len(steps_to_process)}/{num_steps} steps)")

        # Track if we've printed debug output
        debug_printed = False

        # Track NaN activations for this trajectory
        trajectory_nan_count = 0
        trajectory_total_count = 0

        # Extract prompt_prefix once (step-independent due to causal attention)
        prefix_activations = None
        if has_prefix:
            print("    Extracting prompt_prefix activations (once for all steps)...")
            prefix_indices = resolve_token_indices(
                prompt_prefix_indices, prefix_tokens, "prompt_prefix"
            )
            max_prefix_pos = max(prefix_indices)
            # Only need tokens up to max requested prefix position
            prefix_ids = torch.tensor([[t["token_id"] for t in prefix_tokens[: max_prefix_pos + 1]]])

            if debug and not debug_printed:
                decoded_text = model.tokenizer.decode(prefix_ids[0].tolist(), skip_special_tokens=False)
                print(f"\n[DEBUG] First prefix input ({prefix_ids.shape[1]} tokens):\n{decoded_text}\n")
                debug_printed = True

            prefix_activations = extract_activations_single_pass(
                model, prefix_ids, prefix_indices, layer_indices
            )

        # Check if we have any step-dependent categories to extract
        has_step_dependent = has_grid or has_suffix or has_output

        # Process each step
        for step in tqdm(steps_to_process, desc="    Steps", leave=False):
            step_idx = step["step_id"]
            grid_tokens = step["grid_state_tokens"]
            output_tokens = step["output_tokens"]
            n_grid = len(grid_tokens)
            n_output = len(output_tokens)

            # Save prefix activations for this step (same for all steps)
            if prefix_activations is not None:
                nan_count = save_activations_to_files(
                    prefix_activations, output_base, step_idx=step_idx, category="prompt_prefix"
                )
                trajectory_nan_count += nan_count
                trajectory_total_count += sum(len(v) for v in prefix_activations.values())

            # Skip step-dependent extraction if not needed
            if not has_step_dependent:
                continue

            # Compute segment boundaries (absolute positions)
            grid_start = n_prefix
            suffix_start = n_prefix + n_grid
            output_start = n_prefix + n_grid + n_suffix

            # Resolve indices for step-dependent categories
            # Store as: (category_name, relative_indices, absolute_indices)
            extraction_tasks: list[tuple[str, list[int], list[int]]] = []

            if has_grid:
                grid_indices = resolve_token_indices(
                    grid_state_indices, grid_tokens, "grid_state"
                )
                absolute_grid = [grid_start + idx for idx in grid_indices]
                extraction_tasks.append(("grid_state", grid_indices, absolute_grid))

            if has_suffix:
                suffix_indices = resolve_token_indices(
                    prompt_suffix_indices, suffix_tokens, "prompt_suffix"
                )
                absolute_suffix = [suffix_start + idx for idx in suffix_indices]
                extraction_tasks.append(("prompt_suffix", suffix_indices, absolute_suffix))

            if has_output:
                out_indices = resolve_token_indices(
                    output_indices, output_tokens, "output"
                )
                absolute_output = [output_start + idx for idx in out_indices]
                extraction_tasks.append(("output", out_indices, absolute_output))

            if not extraction_tasks:
                continue

            # Find the maximum absolute position needed
            all_absolute_indices = []
            for _, _, abs_indices in extraction_tasks:
                all_absolute_indices.extend(abs_indices)

            max_position = max(all_absolute_indices)

            # Build truncated input up to max_position
            input_ids = build_truncated_input(trajectory, step, max_position)

            if debug and not debug_printed:
                decoded_text = model.tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False)
                categories = [cat for cat, _, _ in extraction_tasks]
                print(f"\n[DEBUG] First step-dependent input ({input_ids.shape[1]} tokens):")
                print(f"[DEBUG] Categories: {categories}")
                print(f"[DEBUG] Decoded text:\n{decoded_text}\n")
                debug_printed = True

            # Single forward pass to extract all step-dependent activations
            activations = extract_activations_single_pass(
                model, input_ids, all_absolute_indices, layer_indices
            )

            # Save activations for each category with relative indices
            for category, rel_indices, abs_indices in extraction_tasks:
                remapped: dict[int, dict[int, torch.Tensor]] = {
                    layer_idx: {} for layer_idx in layer_indices
                }
                for layer_idx in layer_indices:
                    for rel_idx, abs_idx in zip(rel_indices, abs_indices):
                        if abs_idx in activations[layer_idx]:
                            remapped[layer_idx][rel_idx] = activations[layer_idx][abs_idx]

                nan_count = save_activations_to_files(
                    remapped, output_base, step_idx=step_idx, category=category
                )
                trajectory_nan_count += nan_count
                trajectory_total_count += sum(len(v) for v in remapped.values())

        # Report NaN summary for this trajectory
        if trajectory_nan_count > 0:
            print(f"  WARNING: {trajectory_nan_count}/{trajectory_total_count} activations contain NaN values!")
            print(f"  This is often caused by multi-GPU setups. Try using CUDA_VISIBLE_DEVICES=0")

        # Clear CUDA cache after each trajectory to prevent OOM from memory fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nActivations saved to {output_dir}/")

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Re-export for CLI usage
gather_activations = extract_activations_from_trajectories