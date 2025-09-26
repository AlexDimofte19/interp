import os
from typing import Annotated

import nnsight
import torch
import typer
from telos_interp import activations, data_generation, probing, steering

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
