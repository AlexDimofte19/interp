"""Utility functions for data generation."""


def validate_generation_params(num_rollouts: int, max_new_tokens: int) -> None:
    """Validate data generation parameters.

    Args:
        num_rollouts: Number of rollouts to generate
        max_new_tokens: Maximum new tokens per rollout

    Raises:
        ValueError: If parameters are invalid
    """
    if num_rollouts <= 0:
        raise ValueError(f"num_rollouts must be positive, got {num_rollouts}")
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")
