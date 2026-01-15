"""Deprecated gather_activations configurations.

These configs are kept for backwards compatibility but are no longer actively maintained.
Use the main gather_activations function for trajectory-based activation extraction.
"""

from .legacy_configs import (
    GatherActivationsConfig,
    GatherFullActivationsFromJsonlConfig,
    GatherGridActivationsCellwiseConfig,
    GatherGridActivationsLastPromptConfig,
    gather_activations_legacy,
)

__all__ = [
    "GatherActivationsConfig",
    "GatherGridActivationsCellwiseConfig",
    "GatherGridActivationsLastPromptConfig",
    "GatherFullActivationsFromJsonlConfig",
    "gather_activations_legacy",
]
