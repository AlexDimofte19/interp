from typing import Annotated

import torch
import typer
from nnsight import LanguageModel
from telos_interp import activations, data_generation

app = typer.Typer(no_args_is_help=True)


@app.command("version", help="Get the version of the application")
def get_version():
    try:
        from telos_interp import __version__ as version
    except ImportError:
        version = "unknown"
    typer.echo(f"telos_interp version: {version}")


@app.command("count-params", help="Count the number of parameters in a model")
def count_params(model_name_or_path: str):
    model = LanguageModel(model_name_or_path, device_map="auto", dtype=torch.bfloat16)
    num_params = sum(p.numel() for p in model.parameters())
    typer.echo(f"Model '{model_name_or_path}' has {num_params:,} parameters.")


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


@app.command("generate-data", help="Generate a dataset of prompts and responses on which we later train probes.")
def generate_data(
    model_name_or_path: str,
    num_rollouts: int = 10,
    max_new_tokens: int = 512,
    # TODO: Add arguments to generate different types of datasets
):
    data_generation.generate_data(model_name_or_path, num_rollouts, max_new_tokens)


if __name__ == "__main__":
    app()
