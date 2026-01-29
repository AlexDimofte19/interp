"""Legacy gather_activations configurations (deprecated).

These configs are kept for backwards compatibility but are no longer actively maintained.
"""

from dataclasses import dataclass
from typing import Union

from telos_interp import activations, cellwise_activations

from ..gather_activations_utils import get_default_hf_directory, parse_list_of_integers


@dataclass
class GatherActivationsConfig:
    """Gather model activations on a text dataset.

    Args:
        model_name_or_path: Model name or path to load
        csv_path: Path to a CSV file containing columns 'text' and 'label'
        token_position: Position of the token to gather activations from
    """

    model_name_or_path: str
    csv_path: str
    token_position: activations.TokenPosition = activations.TokenPosition.prompt_last

    def __call__(self):
        results_path = activations.gather_activations_from_csv_data(
            self.model_name_or_path, self.csv_path, self.token_position
        )
        print(f"Activations saved to {results_path}")


@dataclass
class GatherGridActivationsCellwiseConfig:
    """Gather model activations at cell token positions from grid CSV data, organized by cell type.

    Args:
        model_name_or_path: Model name or path to load
        csv_path: Path to a grid CSV file with columns: env_idx, observation, x, y, cell_type, symbol, classes_map, optimal_trajectory_length
        layer: Which layer to extract activations from
    """

    model_name_or_path: str
    csv_path: str
    layer: int

    def __call__(self):
        results_path = cellwise_activations.gather_activations_from_grid_at_cell_token_positions(
            self.model_name_or_path, self.csv_path, self.layer
        )
        print(f"Grid activations saved to {results_path}")


@dataclass
class GatherGridActivationsLastPromptConfig:
    """Gather model activations using only the last prompt token for all cell types.

    Args:
        model_name_or_path: Model name or path to load
        csv_path: Path to a grid CSV file with columns: env_idx, observation, x, y, cell_type, symbol, classes_map, optimal_trajectory_length
        layer: Which layer to extract activations from
        observability: Observability of the grid
        observation_type: Type of observation
        row_column: Whether to use row-column coordinates
        raw_acts: Whether to use raw activations
    """

    model_name_or_path: str
    csv_path: str
    layer: int
    observability: activations.Observability = activations.Observability.full
    observation_type: activations.ObservationType = activations.ObservationType.full_prompt
    row_column: bool = False
    raw_acts: bool = False

    def __call__(self):
        results_path = activations.gather_activations_from_grid_at_last_prompt_token(
            self.model_name_or_path,
            self.csv_path,
            self.layer,
            self.observability,
            self.observation_type,
            self.row_column,
            self.raw_acts,
        )
        print(f"Grid activations (last prompt) saved to {results_path}")


@dataclass
class GatherFullActivationsFromJsonlConfig:
    """Gather full activations from a JSONL file.

    Args:
        model_name_or_path: Model name or path to load
        jsonl_path: Path to the JSONL file
        layers: Comma-separated list of layer numbers for the probe
        hf_directory: Directory to save the HF path
    """

    model_name_or_path: str
    jsonl_path: str
    layers: str
    hf_directory: str | None = None

    def __call__(self):
        layers_list = parse_list_of_integers(self.layers)
        hf_dir = self.hf_directory if self.hf_directory is not None else get_default_hf_directory(self.jsonl_path)
        results_path, hf_path = activations.gather_full_activations_from_jsonl(
            self.model_name_or_path, self.jsonl_path, layers_list, hf_dir
        )
        print(f"Full activations saved to {results_path}")
        print(f"HF path: {hf_path}")


GatherActivationsLegacy = Union[
    GatherActivationsConfig,
    GatherGridActivationsCellwiseConfig,
    GatherGridActivationsLastPromptConfig,
    GatherFullActivationsFromJsonlConfig,
]


def gather_activations_legacy(config: GatherActivationsLegacy):
    """Gather model activations from various data formats (deprecated).

    Subcommands:
    - GatherActivationsConfig: Gather activations from a CSV with text and labels
    - GatherGridActivationsCellwiseConfig: Gather activations at cell token positions
    - GatherGridActivationsLastPromptConfig: Gather activations at last prompt token
    - GatherFullActivationsFromJsonlConfig: Gather full activations from JSONL
    """
    config()
