import torch
import typer
from nnsight import LanguageModel

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


if __name__ == "__main__":
    app()
