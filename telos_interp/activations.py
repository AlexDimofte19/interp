import enum
import os

import nnsight
import pandas as pd
import torch
from tqdm import tqdm


class TokenPosition(str, enum.Enum):
    response_avg = "response_avg"
    response_last = "response_last"
    prompt_avg = "prompt_avg"
    prompt_last = "prompt_last"


def gather_activations_from_csv_data(model_name_or_path: str, csv_path: str, token_position: TokenPosition) -> str:
    """Run the model on a number of prompts and gather activations.

    Args:
        model_name_or_path: The name or path of the model to run.
        csv_path: The path to the CSV file containing the fields 'prompt' and 'response'.

    Returns:
        The path to the file where the activations were saved.
    """
    print(f"Loading the data from {csv_path}")
    data = pd.read_csv(csv_path)

    positive_examples = data[data["label"] == 1]

    negative_examples = data[data["label"] == 0]

    model = nnsight.LanguageModel(model_name_or_path, device_map="auto")

    positive_activations = []
    negative_activations = []

    for _idx, example in tqdm(
        positive_examples.iterrows(), desc="Gathering positive activations", total=len(positive_examples)
    ):
        pos_acts = run_model_and_gather_activations_at_token_position(
            model, example["prompt"], example["response"], token_position
        )
        positive_activations.append(pos_acts)

    for _idx, example in tqdm(
        negative_examples.iterrows(), desc="Gathering negative activations", total=len(negative_examples)
    ):
        neg_acts = run_model_and_gather_activations_at_token_position(
            model, example["prompt"], example["response"], token_position
        )
        negative_activations.append(neg_acts)

    positive_activations = torch.stack(positive_activations)
    negative_activations = torch.stack(negative_activations)

    csv_name = os.path.basename(csv_path)
    output_dir_name = csv_name.replace(".csv", "")
    output_dir = f"data/activations/{output_dir_name}/{token_position.value}"

    os.makedirs(output_dir, exist_ok=True)
    torch.save(positive_activations, f"{output_dir}/positive_activations.pt")
    torch.save(negative_activations, f"{output_dir}/negative_activations.pt")

    return output_dir


def run_model_and_gather_activations_at_token_position(
    nnsight_model, prompt, response, token_position: TokenPosition
) -> torch.Tensor:
    """Input the prompt and response into the model and gather activations at the token position.

    Returns:
        A torch.Tensor of shape (num_layers, hidden_size)

    """
    tokenized_prompt = nnsight_model.tokenizer(prompt, return_tensors="pt").input_ids
    prompt_length = tokenized_prompt.shape[1]

    prompt_and_response = prompt + response
    input_ids = nnsight_model.tokenizer(prompt_and_response, return_tensors="pt").input_ids

    # For GPT-2/DialoGPT, the model structure is different
    if hasattr(nnsight_model, "model") and hasattr(nnsight_model.model, "config"):
        num_layers = nnsight_model.model.config.num_hidden_layers
    elif hasattr(nnsight_model, "transformer") and hasattr(nnsight_model.transformer, "h"):
        num_layers = len(nnsight_model.transformer.h)
    else:
        raise ValueError("Cannot determine number of layers from model structure")
    all_layer_outputs = []

    with torch.no_grad():
        with nnsight_model.trace(input_ids):
            for layer in range(num_layers):
                # For GPT-2/DialoGPT, layers are in transformer.h
                if hasattr(nnsight_model, "model") and hasattr(nnsight_model.model, "layers"):
                    full_layer_output = nnsight_model.model.layers[layer].output[0]  # Get the tensor, not the tuple
                elif hasattr(nnsight_model, "transformer") and hasattr(nnsight_model.transformer, "h"):
                    full_layer_output = nnsight_model.transformer.h[layer].output[0]  # Get the tensor, not the tuple
                else:
                    raise ValueError("Cannot access model layers")
                if token_position == TokenPosition.prompt_last:
                    layer_output = full_layer_output[0, prompt_length - 1, :]  # -1 because indices are 0-based
                elif token_position == TokenPosition.prompt_avg:
                    layer_output = full_layer_output[0, :prompt_length, :].mean(dim=0)
                elif token_position == TokenPosition.response_avg:
                    layer_output = full_layer_output[0, prompt_length:, :].mean(dim=0)
                elif token_position == TokenPosition.response_last:
                    layer_output = full_layer_output[0, -1, :]
                else:
                    raise ValueError(f"Invalid token position: {token_position}")

                all_layer_outputs.append(layer_output)

    all_layer_outputs = torch.stack(all_layer_outputs).detach().clone().cpu()  # (num_layers, hidden_size)
    return all_layer_outputs


def run_model_and_gather_activations_all_tokens(nnsight_model, prompt, response) -> torch.Tensor:
    """Input the prompt and response into the model and gather activations for all tokens.

    Returns:
        A torch.Tensor of shape (num_layers, seq_len, hidden_size)
    """
    prompt_and_response = prompt + response
    input_ids = nnsight_model.tokenizer(prompt_and_response, return_tensors="pt").input_ids

    # For GPT-2/DialoGPT, the model structure is different
    if hasattr(nnsight_model, "model") and hasattr(nnsight_model.model, "config"):
        num_layers = nnsight_model.model.config.num_hidden_layers
    elif hasattr(nnsight_model, "transformer") and hasattr(nnsight_model.transformer, "h"):
        num_layers = len(nnsight_model.transformer.h)
    else:
        raise ValueError("Cannot determine number of layers from model structure")
    all_layer_outputs = []

    with torch.no_grad():
        with nnsight_model.trace(input_ids):
            for layer in range(num_layers):
                # For GPT-2/DialoGPT, layers are in transformer.h
                if hasattr(nnsight_model, "model") and hasattr(nnsight_model.model, "layers"):
                    full_layer_output = nnsight_model.model.layers[layer].output[0]  # Get the tensor, not the tuple
                elif hasattr(nnsight_model, "transformer") and hasattr(nnsight_model.transformer, "h"):
                    full_layer_output = nnsight_model.transformer.h[layer].output[0]  # Get the tensor, not the tuple
                else:
                    raise ValueError("Cannot access model layers")

                # Return all token activations: (seq_len, hidden_size)
                layer_output = full_layer_output[0, :, :]  # Remove batch dimension
                all_layer_outputs.append(layer_output)

    all_layer_outputs = torch.stack(all_layer_outputs).detach().clone().cpu()  # (num_layers, seq_len, hidden_size)
    return all_layer_outputs


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
    output_dir = f"data/activations/{output_dir_name}/grid_cellwise_layer_{layer}"

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


def gather_activations_from_grid_at_last_prompt_token(model_name_or_path: str, csv_path: str, layer: int = 12) -> str:
    """Gather activations using only the last prompt token for all cell types from grid CSV data.

    This function extracts activations only at the last prompt token position and groups
    them by cell type. Each activation represents the model's "summary" of the entire grid.

    Args:
        model_name_or_path: The name or path of the model to use
        csv_path: Path to the CSV file containing grid data
        layer: Which layer to extract activations from

    Returns:
        Path to the output directory containing saved activations
    """
    print(f"Loading grid data from {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"Loading model: {model_name_or_path}")
    model = nnsight.LanguageModel(model_name_or_path)

    # Group by environment
    activations_by_type = {}
    unique_cell_types = df["cell_type"].unique()

    for cell_type in unique_cell_types:
        activations_by_type[cell_type] = []

    print(f"Processing {df['env_idx'].nunique()} environments...")

    for env_idx in df["env_idx"].unique():
        env_data = df[df["env_idx"] == env_idx]
        grid_text = env_data.iloc[0]["observation"]

        print(f"Processing environment {env_idx}...")

        # Get activation at last prompt token for this environment
        last_prompt_activation = extract_last_prompt_activation_for_grid(model, grid_text, layer)

        if last_prompt_activation is not None:
            # For each cell type in this environment, add the same activation
            for cell_type in unique_cell_types:
                if cell_type in env_data["cell_type"].values:
                    activations_by_type[cell_type].append(last_prompt_activation)

    # Save activations by type
    csv_name = os.path.basename(csv_path)
    output_dir_name = csv_name.replace(".csv", "")
    output_dir = f"data/activations/{output_dir_name}/grid_last_prompt_layer_{layer}"

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


def extract_last_prompt_activation_for_grid(model, grid_text: str, layer: int) -> torch.Tensor | None:
    """Extract activation at the last prompt token for a grid.

    Args:
        model: The nnsight model
        grid_text: The full grid observation text
        layer: Which layer to extract from

    Returns:
        Activation tensor of shape (hidden_dim,) or None if extraction fails
    """
    try:
        # Tokenize the prompt to get its length
        tokenized_prompt = model.tokenizer(grid_text, return_tensors="pt").input_ids
        prompt_length = tokenized_prompt.shape[1]

        # Get activations for all tokens
        dummy_response = ""  # Empty response since we're only interested in the input
        all_layer_activations = run_model_and_gather_activations_all_tokens(model, grid_text, dummy_response)

        # Extract the specific layer we want
        if layer >= len(all_layer_activations):
            print(f"Warning: Layer {layer} not available. Model has {len(all_layer_activations)} layers.")
            return None

        layer_activations = all_layer_activations[layer]  # Shape: (seq_len, hidden_dim)

        # Get the last prompt token activation (0-based indexing)
        last_prompt_idx = prompt_length - 1
        if last_prompt_idx < layer_activations.shape[0]:
            last_prompt_activation = layer_activations[last_prompt_idx]  # Shape: (hidden_dim,)
            return last_prompt_activation
        else:
            print(f"Warning: Last prompt token index {last_prompt_idx} out of bounds")
            return None

    except Exception as e:
        print(f"Error extracting last prompt activation: {e}")
        return None


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
    all_layer_activations = run_model_and_gather_activations_all_tokens(model, grid_text, dummy_response)

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
