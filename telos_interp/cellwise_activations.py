import os

import nnsight
import pandas as pd
import torch
from tqdm import tqdm

from telos_interp.activations import TokenPosition, run_model_and_gather_activations_at_token_position


def gather_activations_from_grid_at_cell_token_positions(
    model_name_or_path: str, csv_path: str, layer: int = 12
) -> str:
    """Gather activations at cell token positions from grid CSV data, organized by cell type.

    Args:
        model_name_or_path: The name or path of the model to run.
        csv_path: Path to the grid CSV file with columns: env_idx, observation, x, y, cell_type, symbol, classes_map, optimal_trajectory_length
        layer: Which layer to extract activations from

    Returns:
        The path to the directory where the activations were saved.
    """
    print(f"Loading grid data from {csv_path}")
    data = pd.read_csv(csv_path)

    # Group by environment to process each grid separately
    env_groups = data.groupby("env_idx")

    model = nnsight.LanguageModel(model_name_or_path, device_map="auto")

    # Dictionary to store activations by cell type
    activations_by_type = {"wall": [], "empty": [], "agent": [], "goal": []}

    print(f"Processing {len(env_groups)} environments...")

    for _env_idx, env_data in tqdm(env_groups, desc="Processing environments"):
        # Get the full grid observation (same for all cells in this env)
        grid_text = env_data["observation"].iloc[0]

        # Extract activations for each cell in this environment
        cell_activations = extract_activations_at_grid_cell_token_positions(model, grid_text, env_data, layer)

        # Group activations by cell type
        for _, row in env_data.iterrows():
            x, y = row["x"], row["y"]
            cell_type = row["cell_type"]

            if (x, y) in cell_activations:
                activations_by_type[cell_type].append(cell_activations[(x, y)])

    # Convert lists to tensors and save
    csv_name = os.path.basename(csv_path)
    output_dir_name = csv_name.replace(".csv", "")
    short_model_name = model_name_or_path.split("/")[-1]
    output_dir = f"data/activations/{short_model_name}/{output_dir_name}/grid_cellwise_layer_{layer}"

    os.makedirs(output_dir, exist_ok=True)

    for cell_type, activations_list in activations_by_type.items():
        if activations_list:  # Only save if we have activations for this type
            activations_tensor = torch.stack(activations_list)
            output_path = f"{output_dir}/acts_{cell_type}.pt"
            torch.save(activations_tensor, output_path)
            print(f"Saved {len(activations_list)} {cell_type} activations to {output_path}")
        else:
            print(f"No activations found for cell type: {cell_type}")

    return output_dir


def extract_activations_at_grid_cell_token_positions(
    model, grid_text: str, env_data: pd.DataFrame, layer: int
) -> dict:
    """Extract activations for each grid cell in an environment.

    Args:
        model: The nnsight model
        grid_text: The full grid observation text
        env_data: DataFrame with cell information for this environment
        token_position: Which token position to extract from
        layer: Which layer to extract from

    Returns:
        Dictionary mapping (x, y) coordinates to activations
    """
    # Get activations for the entire grid using the new function that returns all tokens
    dummy_response = ""  # Empty response since we're only interested in the input
    all_layer_activations = run_model_and_gather_activations_at_token_position(
        model, grid_text, dummy_response, TokenPosition.all_tokens
    )

    # Extract the specific layer we want
    if layer >= len(all_layer_activations):
        print(f"Warning: Layer {layer} not available. Model has {len(all_layer_activations)} layers.")
        return {}

    layer_activations = all_layer_activations[layer]  # Shape: (seq_len, hidden_dim)

    cell_activations = {}

    # For each cell in this environment, find its token position and extract activation
    for _, row in env_data.iterrows():
        x, y = row["x"], row["y"]

        # Find token position for this cell
        token_pos = find_token_position_for_grid_cell(grid_text, x, y, model.tokenizer)

        if token_pos is not None and token_pos < layer_activations.shape[0]:
            # Extract activation for this token position
            cell_activation = layer_activations[token_pos]  # Shape: (hidden_dim,)
            cell_activations[(x, y)] = cell_activation

    return cell_activations


def find_token_position_for_grid_cell(grid_text: str, x: int, y: int, tokenizer) -> int:
    """Find the token position for a specific grid cell at (x, y).

    Args:
        grid_text: The full grid observation text
        x, y: Grid cell coordinates
        tokenizer: The model's tokenizer

    Returns:
        Token position for the cell, or None if not found
    """
    lines = grid_text.strip().split("\n")

    # Find grid start (skip mission, legend, etc.)
    grid_start = None
    for i, line in enumerate(lines):
        if line.startswith("#"):
            grid_start = i
            break

    if grid_start is None:
        return None

    # Calculate character position for (x, y)
    # First, count characters before the grid starts
    char_position = 0
    for line_idx in range(grid_start):
        char_position += len(lines[line_idx]) + 1  # +1 for newline

    # Now add characters within the grid
    grid_lines = lines[grid_start:]

    # Count characters before the target line within the grid
    for line_idx in range(y):
        if line_idx < len(grid_lines):
            char_position += len(grid_lines[line_idx]) + 1  # +1 for newline

    # Add x position within the target line
    if y < len(grid_lines):
        char_position += x

    # Use a different approach: tokenize and find the token that contains this character
    try:
        # Get token-to-character mapping
        tokens = tokenizer.encode(grid_text)
        token_ranges = []

        # Decode each token to find its character range
        for i, token in enumerate(tokens):
            decoded = tokenizer.decode([token])
            if i == 0:
                start = 0
            else:
                start = token_ranges[i - 1][1] if token_ranges else 0

            end = start + len(decoded)
            token_ranges.append((start, end))

        # Find which token contains our character position
        for i, (start, end) in enumerate(token_ranges):
            if start <= char_position < end:
                return i

        return None
    except Exception:
        return None
