"""Command-line interface for the telos_interp package."""

import logging

import tyro

from telos_interp.commands.apply_cognitive_map_probe import apply_cognitive_map_probe
from telos_interp.commands.eval_cognitive_map_probe import eval_cognitive_map_probe
from telos_interp.commands.eval_distance_probe import eval_distance_probe
from telos_interp.commands.gather_activations import gather_activations
from telos_interp.commands.prepare_activations_for_probing import prepare_activations_for_probing
from telos_interp.commands.train_cognitive_map_probe import train_cognitive_map_probe
from telos_interp.commands.train_distance_probe import train_distance_probe


def main():
    """Main entry point for the telos-interp command-line tool.

    Configures logging and sets up the CLI with available subcommands using tyro.
    Currently supports the following subcommands:
    - gather_activations: Gather model activations from various data formats
    - train_distance_probe: Train and apply distance regression probes
    - prepare_activations_for_probing: Extract and concatenate activations for cognitive map probing
    - train_cognitive_map_probe: Train cognitive map probing classifiers (LR or MLP)
    - apply_cognitive_map_probe: Apply trained probes to trajectories and store predictions
    - eval_cognitive_map_probe: Evaluate probes on test trajectories with detailed metrics
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    tyro.extras.subcommand_cli_from_dict(
        {
            "gather_activations": gather_activations,
            "train_distance_probe": train_distance_probe,
            "eval_distance_probe": eval_distance_probe,
            "prepare_activations_for_probing": prepare_activations_for_probing,
            "train_cognitive_map_probe": train_cognitive_map_probe,
            "apply_cognitive_map_probe": apply_cognitive_map_probe,
            "eval_cognitive_map_probe": eval_cognitive_map_probe,
        }
    )


if __name__ == "__main__":
    main()
