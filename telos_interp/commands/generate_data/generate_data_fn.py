"""Generate datasets of prompts and responses for training probes."""

from dataclasses import dataclass

from telos_interp import data_generation

from .generate_data_utils import validate_generation_params


@dataclass
class GenerateDataConfig:
    """Generate a dataset of prompts and responses on which we later train probes.

    Args:
        model_name_or_path: Model name or path to load
        num_rollouts: Number of rollouts to generate
        max_new_tokens: Maximum new tokens per rollout
    """

    model_name_or_path: str
    num_rollouts: int = 10
    max_new_tokens: int = 512

    def __call__(self):
        validate_generation_params(self.num_rollouts, self.max_new_tokens)
        data_generation.generate_data(self.model_name_or_path, self.num_rollouts, self.max_new_tokens)


def generate_data(config: GenerateDataConfig):
    """Generate a dataset of prompts and responses on which we later train probes."""
    config()
