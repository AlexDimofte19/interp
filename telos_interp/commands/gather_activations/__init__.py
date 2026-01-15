from .gather_activations_fn import gather_activations

# Legacy configs available in deprecated submodule
from . import deprecated

__all__ = ["gather_activations", "deprecated"]
