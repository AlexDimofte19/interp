"""Command-line interface for the telos_interp package."""

import logging

import tyro

from telos_interp.commands.gather_activations import gather_activations
from telos_interp.commands.generate_data import generate_data
from telos_interp.commands.steering import steering_cmd
from telos_interp.commands.train_distance_probe import train_distance_probe
from telos_interp.commands.train_probe import train_probe


def main():
    """Main entry point for the telos-interp command-line tool.

    Configures logging and sets up the CLI with available subcommands using tyro.
    Currently supports the following subcommands:
    - gather_activations: Gather model activations from various data formats
    - generate_data: Generate datasets of prompts and responses for training probes
    - train_probe: Train and apply probing classifiers on model activations
    - train_distance_probe: Train and apply distance regression probes
    - steering: Compute and apply steering vectors for model generation
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    tyro.extras.subcommand_cli_from_dict(
        {
            "gather_activations": gather_activations,
            "generate_data": generate_data,
            "train_probe": train_probe,
            "train_distance_probe": train_distance_probe,
            "steering": steering_cmd,
        }
    )


if __name__ == "__main__":
    main()
