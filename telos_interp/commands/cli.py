import json
import os
from typing import Annotated

import nnsight
import pandas as pd
import torch
import typer
from telos_interp import activations, cellwise_activations, data_generation, probing, probing_gpu, steering

app = typer.Typer(no_args_is_help=True)


@app.command("version", help="Get the version of the application")
def get_version():
    try:
        from telos_interp import __version__ as version
    except ImportError:
        version = "unknown"
    typer.echo(f"telos_interp version: {version}")


@app.command("generate-data", help="Generate a dataset of prompts and responses on which we later train probes.")
def generate_data(
    model_name_or_path: str,
    num_rollouts: int = 10,
    max_new_tokens: int = 512,
    # TODO: Add arguments to generate different types of datasets
):
    data_generation.generate_data(model_name_or_path, num_rollouts, max_new_tokens)


@app.command("gather-activations", help="Gather model activations on a text dataset")
def gather_activations(
    model_name_or_path: str,
    csv_path: Annotated[str, typer.Argument(..., help="Path to a CSV file containing columns 'text' and 'label'")],
    token_position: Annotated[
        activations.TokenPosition, typer.Argument(..., help="Position of the token to gather activations from")
    ] = activations.TokenPosition.prompt_last,
):
    results_path = activations.gather_activations_from_csv_data(model_name_or_path, csv_path, token_position)
    typer.echo(f"Activations saved to {results_path}")


@app.command(
    "gather-grid-activations-cellwise",
    help="Gather model activations at cell token positions from grid CSV data, organized by cell type.",
)
def gather_grid_activations_cellwise(
    model_name_or_path: str,
    csv_path: Annotated[
        str,
        typer.Argument(
            ...,
            help="Path to a grid CSV file with columns: env_idx, observation, x, y, cell_type, symbol, classes_map, optimal_trajectory_length",
        ),
    ],
    layer: Annotated[int, typer.Option(..., help="Which layer to extract activations from")],
):
    results_path = cellwise_activations.gather_activations_from_grid_at_cell_token_positions(
        model_name_or_path, csv_path, layer
    )
    typer.echo(f"Grid activations saved to {results_path}")


@app.command(
    "gather-grid-activations-last-prompt",
    help="Gather model activations using only the last prompt token for all cell types.",
)
def gather_grid_activations_last_prompt(
    model_name_or_path: str,
    csv_path: Annotated[
        str,
        typer.Argument(
            ...,
            help="Path to a grid CSV file with columns: env_idx, observation, x, y, cell_type, symbol, classes_map, optimal_trajectory_length",
        ),
    ],
    layer: Annotated[int, typer.Option(help="Which layer to extract activations from")],
    observability: Annotated[
        activations.Observability, typer.Option(help="Observability of the grid")
    ] = activations.Observability.full,
    observation_type: Annotated[
        activations.ObservationType, typer.Option(help="Type of observation")
    ] = activations.ObservationType.full_prompt,
    row_column: Annotated[bool, typer.Option(help="Whether to use row-column coordinates")] = False,
):
    results_path = activations.gather_activations_from_grid_at_last_prompt_token(
        model_name_or_path, csv_path, layer, observability, observation_type, row_column
    )
    typer.echo(f"Grid activations (last prompt) saved to {results_path}")


def parse_list_of_integers(value: str) -> list[int]:
    return [int(x) for x in value.split(",")]


@app.command("gather-full-acts-jsonl", help="Gather full activations from a JSONL file")
def gather_full_activations_from_jsonl(
    model_name_or_path: str,
    jsonl_path: str,
    layers: Annotated[str, typer.Option(help="Comma-separated list of layer numbers for the probe")],
    hf_directory: Annotated[str, typer.Option(help="Directory to save the HF path")] = None,
):
    layers = parse_list_of_integers(layers)
    if hf_directory is None:
        hf_directory = os.path.basename(jsonl_path)
    results_path, hf_path = activations.gather_full_activations_from_jsonl(
        model_name_or_path, jsonl_path, layers, hf_directory
    )

    typer.echo(f"Full activations saved to {results_path}")
    typer.echo(f"HF path: {hf_path}")


@app.command("train-multiclass-probe", help="Train a multi-class probing classifier on grid cell activations")
def train_multiclass_probe(
    activations_dir: Annotated[
        str, typer.Argument(..., help="Directory containing activation files (acts_wall.pt, acts_empty.pt, etc.)")
    ],
    layer: Annotated[int, typer.Option(..., help="Layer number for the probe")],
    output_dir: Annotated[str, typer.Option(help="Directory to save the probe")] = None,
    eval_split: Annotated[float, typer.Option(help="Fraction of data to use for evaluation")] = 0.2,
    reg_coeff: Annotated[float, typer.Option(help="Regularization coefficient")] = 1e3,
    normalize: Annotated[bool, typer.Option(help="Whether to normalize activations")] = False,
    interaction_features: Annotated[bool, typer.Option(help="Whether to use interaction features")] = False,
    fit_intercept: Annotated[bool, typer.Option(help="Whether to fit an intercept term")] = True,
    verbose: Annotated[int, typer.Option(help="Verbosity level (0=silent, 1+=show progress)")] = 0,
    use_gpu: Annotated[bool, typer.Option("--gpu/--cpu", help="Use GPU-accelerated PyTorch implementation")] = False,
    learning_rate: Annotated[float, typer.Option(help="Learning rate for GPU training")] = 0.1,
    num_epochs: Annotated[int, typer.Option(help="Number of epochs for GPU training")] = 100,
    balanced_weights: Annotated[
        bool,
        typer.Option(
            "--balanced-weights/--no-balanced-weights", help="Use balanced class weights to handle class imbalance"
        ),
    ] = False,
    balance_classes: Annotated[
        bool,
        typer.Option(
            "--balance-classes/--no-balance-classes", help="Balance classes by downsampling the majority classes"
        ),
    ] = False,
    use_mlp: Annotated[
        bool, typer.Option("--use-mlp/--no-mlp", help="Use MLP architecture instead of linear (GPU only)")
    ] = False,
    mlp_hidden_size: Annotated[int, typer.Option(help="Hidden size for MLP architecture (GPU only)")] = 1024,
    batch_size: Annotated[int, typer.Option(help="Batch size for GPU training")] = 16384,
):
    class_weight = "balanced" if balanced_weights else None

    assert layer is not None, "Layer must be provided"

    if use_gpu:
        saved_probe_path = probing_gpu.train_and_save_multiclass_probe_gpu(
            activations_dir,
            layer,
            output_dir,
            eval_split,
            reg_coeff,
            normalize,
            interaction_features,
            fit_intercept,
            verbose,
            device=None,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            class_weight=class_weight,
            balance_classes=balance_classes,
            use_mlp=use_mlp,
            mlp_hidden_size=mlp_hidden_size,
            batch_size=batch_size,
        )
    else:
        if use_mlp:
            typer.echo("Warning: --use-mlp is only supported with --gpu flag. Ignoring MLP option.")
        saved_probe_path = probing.train_and_save_multiclass_probe(
            activations_dir,
            layer,
            output_dir,
            eval_split,
            reg_coeff,
            normalize,
            fit_intercept,
            verbose,
            class_weight=class_weight,
        )
    typer.echo(f"Multi-class probe saved to {saved_probe_path}")


@app.command("probe-predict", help="Predict using the multiclass probe on a csv dataset of trajectories.")
def probe_predict(
    model_name_or_path: str,
    probe_path: str,
    csv_path: str,
    layer: int,
    results_path: Annotated[str, typer.Option(help="Path to save the resulting json file")] = None,
    observability: Annotated[
        activations.Observability, typer.Option(help="Observability of the grid")
    ] = activations.Observability.full,
    observation_type: Annotated[
        activations.ObservationType, typer.Option(help="Type of observation")
    ] = activations.ObservationType.full_prompt,
):
    assert os.path.exists(probe_path), f"Probe path {probe_path} does not exist"
    assert os.path.exists(csv_path), f"CSV path {csv_path} does not exist"

    if results_path is None:
        output_dir = csv_path.replace(".csv", "")
        os.makedirs(output_dir, exist_ok=True)
        results_path = os.path.join(
            output_dir, f"predictions_l{layer}_{observability.value}_{observation_type.value}.json"
        )

    # 1. For each row in the csv file, get the corresponding activations for the probe
    activations_list = activations.get_activations_for_each_row_in_csv(
        model_name_or_path, csv_path, layer, observability, observation_type
    )

    predictions = probing_gpu.predict_from_activations_list(probe_path, activations_list)

    evaluation_results = probing_gpu.evaluate_predictions(csv_path, predictions)

    print(f"Total accuracy: {evaluation_results['total_accuracy']}")
    print(f"Total per class accuracies: {evaluation_results['total_per_class_accuracies']}")

    with open(results_path, "w") as f:
        json.dump(predictions, f, indent=4)

    print(f"Predictions saved to {results_path}")

    eval_results_path = results_path.replace(".json", "_evaluation.json")
    with open(eval_results_path, "w") as f:
        json.dump(evaluation_results, f, indent=4)


@app.command("probe-evaluate-predictions", help="Evaluate probe predictions saved in probe-predict output file")
def probe_evaluate_predictions(
    predictions_path: str,
    original_csv_path: str,
):
    predictions = json.load(open(predictions_path))
    evaluation_results = probing_gpu.evaluate_predictions(original_csv_path, predictions)
    print(f"Total accuracy: {evaluation_results['total_accuracy']}")
    print(f"Total per class accuracies: {evaluation_results['total_per_class_accuracies']}")

    eval_results_path = predictions_path.replace(".json", "_evaluation.json")
    with open(eval_results_path, "w") as f:
        json.dump(evaluation_results, f, indent=4)


@app.command("probe-eval-on-jsonl", help="Evaluate a probe on a JSONL file")
def probe_eval_on_jsonl(
    model_name_or_path: str,
    probe_path: str,
    layer: int,
    jsonl_path: str,
    grid_size: int,
    observability: Annotated[
        activations.Observability, typer.Option(help="Observability of the grid")
    ] = activations.Observability.full,
    observation_type: Annotated[
        activations.ObservationType, typer.Option(help="Type of observation")
    ] = activations.ObservationType.full_prompt,
    results_path: Annotated[str, typer.Option(help="Path to save the resulting json file")] = None,
):
    if results_path is None:
        output_dir = jsonl_path.replace(".jsonl", "")
        os.makedirs(output_dir, exist_ok=True)
        results_path = os.path.join(
            output_dir,
            f"predictions_grid_size_{grid_size}_l{layer}_{observability.value}_{observation_type.value}.jsonl",
        )

    # 1. For each observation of the corresponding size in the jsonl file, get the activations
    activations_list = activations.get_activations_for_each_row_in_jsonl(
        model_name_or_path, jsonl_path, grid_size, layer, observability, observation_type
    )

    predictions = probing_gpu.predict_from_activations_list(probe_path, activations_list)

    df = pd.read_json(jsonl_path, lines=True)
    grid_size_idx = df[df["size"] == grid_size].index.tolist()

    assert len(grid_size_idx) == len(activations_list), "Number of grids and activations must match"
    for predictions_dict, idx in zip(predictions, grid_size_idx, strict=False):
        df.at[idx, "metadata"]["predictions"] = predictions_dict

    df.to_json(results_path, lines=True, orient="records", indent=4)
    print(f"Predictions saved to {results_path}")


@app.command("train-probe", help="Train a probing classifier on a dataset of activations.")
def train_probe(
    positive_acts: str,
    negative_acts: str,
    layer: int,
    output_dir: str = None,
    eval_split: float = 0.2,
    reg_coeff: float = 1e3,
    normalize: bool = True,
):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(positive_acts), "probes")
        os.makedirs(output_dir, exist_ok=True)

    saved_probe_path = probing.train_and_save_probe(
        positive_acts, negative_acts, layer, output_dir, eval_split, reg_coeff, normalize
    )
    typer.echo(f"Probe saved to {saved_probe_path}")


@app.command("apply-probe", help="Apply a probe to a dataset of activations")
def apply_probe(
    model_name_or_path: str,
    probe_path: str,
    layer: int,
    prompt: str,
    response: str,
    token_position: Annotated[
        activations.TokenPosition, typer.Argument(..., help="Position of the token to gather activations from")
    ] = activations.TokenPosition.response_avg,
):
    # Initialize model and controller
    model = nnsight.LanguageModel(model_name_or_path, device_map="auto", dispatch=True)

    # Load probe
    probe = probing.ProbingClassifier.load(probe_path)

    # Apply probe
    score = probing.apply_probe_on_prompt_response(model, probe, prompt, response, layer, token_position)
    typer.echo(f"Score: {score}")


@app.command("compute-steering", help="Compute steering vector from successful and failed activations")
def compute_steering(
    successful_activations_path: str,
    failed_activations_path: str,
    goal_name: str,
    method: Annotated[str, typer.Option(help="Method for computing steering vector")] = "mean_difference",
    output_dir: Annotated[str, typer.Option(help="Directory to save steering vector")] = "steering_vectors",
):
    """Compute steering vector from pre-computed activations."""
    # Load activations
    typer.echo("Loading activations...")
    successful_activations = torch.load(successful_activations_path)
    failed_activations = torch.load(failed_activations_path)

    typer.echo(f"Loaded activations: {successful_activations.shape} successful, {failed_activations.shape} failed")

    # Compute steering vector
    typer.echo("Computing steering vector...")
    steering_vector = steering.compute_steering_vector(successful_activations, failed_activations, method)

    # Save steering vector
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{goal_name}_steering_vector.pt")
    steering_vector.save(path)
    typer.echo(f"Steering vector saved to {path}")


@app.command("apply-steering", help="Apply steering to generate a response")
def apply_steering(
    model_name_or_path: str,
    goal_name: str,
    prompt: str,
    layer: int,
    steering_dir: Annotated[str, typer.Option(help="Directory containing steering vectors")] = "steering_vectors",
    strength: Annotated[float, typer.Option(help="Steering strength")] = 1.0,
    max_new_tokens: int = 100,
):
    """Apply steering to generate a response."""
    # Initialize model and controller
    model = nnsight.LanguageModel(model_name_or_path, device_map="auto", dispatch=True)
    controller = steering.SteeringController(model)

    # Load steering vectors
    controller.load_steering_vectors(steering_dir)

    # Apply steering
    typer.echo(f"Generating response for prompt: {prompt}")
    response = controller.apply_steering(prompt, goal_name, layer, strength, max_new_tokens)

    typer.echo(f"Steered response: {response}")


@app.command("steer-interactive", help="Interactive steering session")
def steer_interactive(
    model_name_or_path: str,
    goal_name: str,
    steering_dir: Annotated[str, typer.Option(help="Directory containing steering vectors")] = "steering_vectors",
    strength: Annotated[float, typer.Option(help="Steering strength")] = 1.0,
    max_new_tokens: int = 100,
    token_position: Annotated[
        activations.TokenPosition, typer.Option(help="Position of the token to gather activations from")
    ] = activations.TokenPosition.response_avg,
):
    """Start an interactive steering session."""
    # Initialize model and controller
    model = nnsight.LanguageModel(model_name_or_path, device_map="auto")
    controller = steering.SteeringController(model)

    # Load steering vectors
    controller.load_steering_vectors(steering_dir)

    typer.echo(f"Interactive steering session for goal: {goal_name}")
    typer.echo("Enter prompts (type 'quit' to exit):")

    while True:
        prompt = typer.prompt("Prompt")
        if prompt.lower() == "quit":
            break

        try:
            response = controller.apply_steering(prompt, goal_name, strength, max_new_tokens)
            typer.echo(f"Response: {response}\n")
        except Exception as e:
            typer.echo(f"Error: {e}\n")


if __name__ == "__main__":
    app()
