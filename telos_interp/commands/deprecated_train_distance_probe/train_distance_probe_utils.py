"""Utility functions for distance probe training and evaluation."""

import os


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
