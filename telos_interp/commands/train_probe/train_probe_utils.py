"""Utility functions for probe training and evaluation."""

import json
import os
from typing import Any


def validate_paths(*paths: str) -> None:
    """Validate that all provided paths exist.

    Args:
        *paths: Variable number of paths to validate

    Raises:
        AssertionError: If any path does not exist
    """
    for path in paths:
        assert os.path.exists(path), f"Path {path} does not exist"


def validate_layer(layer: int | None) -> None:
    """Validate that a layer is provided.

    Args:
        layer: Layer number to validate

    Raises:
        AssertionError: If layer is None
    """
    assert layer is not None, "Layer must be provided"


def get_predictions_output_path(
    csv_path: str,
    probe_path: str,
    layer: int,
    observability_value: str,
    observation_type_value: str,
) -> str:
    """Generate output path for probe predictions.

    Args:
        csv_path: Path to the input CSV file
        probe_path: Path to the probe file
        layer: Layer number
        observability_value: Observability enum value
        observation_type_value: Observation type enum value

    Returns:
        Path for saving predictions
    """
    probe_name = os.path.basename(probe_path)
    output_dir = csv_path.replace(".csv", "")
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(
        output_dir,
        f"predictions_{probe_name}_l{layer}_{observability_value}_{observation_type_value}.json",
    )


def get_jsonl_predictions_output_path(
    jsonl_path: str,
    grid_size: int,
    layer: int,
    observability_value: str,
    observation_type_value: str,
) -> str:
    """Generate output path for JSONL probe predictions.

    Args:
        jsonl_path: Path to the input JSONL file
        grid_size: Size of the grid
        layer: Layer number
        observability_value: Observability enum value
        observation_type_value: Observation type enum value

    Returns:
        Path for saving predictions
    """
    output_dir = jsonl_path.replace(".jsonl", "")
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(
        output_dir,
        f"predictions_grid_size_{grid_size}_l{layer}_{observability_value}_{observation_type_value}.jsonl",
    )


def get_default_probe_output_dir(positive_acts_path: str) -> str:
    """Get default output directory for probes based on activations path.

    Args:
        positive_acts_path: Path to positive activations file

    Returns:
        Path to probes directory
    """
    output_dir = os.path.join(os.path.dirname(positive_acts_path), "probes")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_predictions(predictions: Any, output_path: str) -> None:
    """Save predictions to a JSON file.

    Args:
        predictions: Predictions data to save
        output_path: Path to save the JSON file
    """
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=4)
    print(f"Predictions saved to {output_path}")


def save_evaluation_results(evaluation_results: dict, predictions_path: str) -> str:
    """Save evaluation results alongside predictions.

    Args:
        evaluation_results: Evaluation metrics dictionary
        predictions_path: Path to the predictions file

    Returns:
        Path to the saved evaluation file
    """
    eval_results_path = predictions_path.replace(".json", "_evaluation.json")
    with open(eval_results_path, "w") as f:
        json.dump(evaluation_results, f, indent=4)
    return eval_results_path


def print_evaluation_results(evaluation_results: dict) -> None:
    """Print evaluation results to stdout.

    Args:
        evaluation_results: Evaluation metrics dictionary
    """
    print(f"Total accuracy: {evaluation_results['total_accuracy']}")
    print(f"Total per class accuracies: {evaluation_results['total_per_class_accuracies']}")
