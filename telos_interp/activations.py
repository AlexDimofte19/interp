import enum
import os

import nnsight
import pandas as pd
import torch
from tqdm import tqdm


class TokenPosition(str, enum.Enum):
    response_avg = "response_avg"
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
    output_dir = f"data/activations/{output_dir_name}"

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

    num_layers = nnsight_model.model.config.num_hidden_layers
    all_layer_outputs = []

    with torch.no_grad():
        with nnsight_model.trace(input_ids):
            for layer in range(num_layers):
                full_layer_output = nnsight_model.model.layers[
                    layer
                ].output  # Layer output has shape (batch_size, seq_len, hidden_size)
                if token_position == TokenPosition.prompt_last:
                    layer_output = full_layer_output[0, prompt_length, :]
                elif token_position == TokenPosition.response_avg:
                    layer_output = full_layer_output[0, prompt_length:, :].mean(dim=0)
                elif token_position == TokenPosition.prompt_avg:
                    layer_output = full_layer_output[0, :prompt_length, :].mean(dim=0)
                else:
                    raise ValueError(f"Invalid token position: {token_position}")

                all_layer_outputs.append(layer_output)

    all_layer_outputs = torch.stack(all_layer_outputs).detach().clone().cpu()  # (num_layers, hidden_size)
    return all_layer_outputs
