import enum
import os
from collections import defaultdict

import nnsight
import pandas as pd
import torch
from tqdm import tqdm


class TokenPosition(str, enum.Enum):
    response_avg = "response_avg"
    response_last = "response_last"
    prompt_avg = "prompt_avg"
    prompt_last = "prompt_last"
    all_tokens = "all_tokens"


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
        if token_position == TokenPosition.all_tokens:
            A torch.Tensor of shape (num_layers, seq_len, hidden_size)
        else:
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
                # TODO Use nnterp to unify code across models!!!
                if hasattr(nnsight_model, "model") and hasattr(nnsight_model.model, "layers"):
                    full_layer_output = nnsight_model.model.layers[layer].output
                # For GPT-2/DialoGPT, layers are in transformer.h
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
                elif token_position == TokenPosition.all_tokens:
                    layer_output = full_layer_output[0, :, :]
                else:
                    raise ValueError(f"Invalid token position: {token_position}")

                all_layer_outputs.append(layer_output)

    # Move to CPU first, then stack
    all_layer_outputs = [output.detach().clone().cpu() for output in all_layer_outputs]
    all_layer_outputs = torch.stack(all_layer_outputs)
    return all_layer_outputs


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
    activations_by_type = defaultdict(list)

    print(f"Processing {df['env_idx'].nunique()} environments...")

    for env_idx in df["env_idx"].unique():
        env_data = df[df["env_idx"] == env_idx]
        grid_text = env_data.iloc[0]["observation"]

        print(f"Processing environment {env_idx}...")

        # Get activation at last prompt token for this environment
        empty_response = ""  # Empty response since we're only interested in the input
        all_layer_activations = run_model_and_gather_activations_at_token_position(
            model, grid_text, empty_response, TokenPosition.prompt_last
        )  # (num_layers, hidden_dim)
        last_prompt_activation = all_layer_activations[layer]  # Shape: (hidden_dim,)
        hidden_size = last_prompt_activation.shape[0]

        for _, env_row in env_data.iterrows():
            # Each row in env_data corresponds to a single cell. We save the activation together with x,y coordinates to the class specific data.
            x = env_row["x"]
            y = env_row["y"]
            cell_type = env_row["cell_type"]
            full_probe_input = torch.zeros(hidden_size + 2)
            full_probe_input[:hidden_size] = last_prompt_activation
            full_probe_input[hidden_size] = x
            full_probe_input[hidden_size + 1] = y
            activations_by_type[cell_type].append(full_probe_input)

    # Save activations by type
    csv_name = os.path.basename(csv_path)
    output_dir_name = csv_name.replace(".csv", "")
    short_model_name = model_name_or_path.split("/")[-1]
    output_dir = f"data/activations/{short_model_name}/{output_dir_name}/grid_last_prompt_layer_{layer}"

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
