"""Compute and apply steering vectors for model generation."""

from dataclasses import dataclass
from typing import Union

from telos_interp import activations, steering

from .steering_utils import (
    create_steering_controller,
    load_activations,
    load_model,
    run_interactive_loop,
    save_steering_vector,
)


@dataclass
class ComputeSteeringConfig:
    """Compute steering vector from successful and failed activations.

    Args:
        successful_activations_path: Path to successful activations file
        failed_activations_path: Path to failed activations file
        goal_name: Name of the goal for the steering vector
        method: Method for computing steering vector
        output_dir: Directory to save steering vector
    """

    successful_activations_path: str
    failed_activations_path: str
    goal_name: str
    method: str = "mean_difference"
    output_dir: str = "steering_vectors"

    def __call__(self):
        print("Loading activations...")
        successful_activations = load_activations(self.successful_activations_path)
        failed_activations = load_activations(self.failed_activations_path)

        print(f"Loaded activations: {successful_activations.shape} successful, {failed_activations.shape} failed")

        print("Computing steering vector...")
        steering_vector = steering.compute_steering_vector(successful_activations, failed_activations, self.method)

        path = save_steering_vector(steering_vector, self.goal_name, self.output_dir)
        print(f"Steering vector saved to {path}")


@dataclass
class ApplySteeringConfig:
    """Apply steering to generate a response.

    Args:
        model_name_or_path: Model name or path to load
        goal_name: Name of the goal for steering
        prompt: Prompt text to generate from
        layer: Layer to apply steering at
        steering_dir: Directory containing steering vectors
        strength: Steering strength
        max_new_tokens: Maximum new tokens to generate
    """

    model_name_or_path: str
    goal_name: str
    prompt: str
    layer: int
    steering_dir: str = "steering_vectors"
    strength: float = 1.0
    max_new_tokens: int = 100

    def __call__(self):
        model = load_model(self.model_name_or_path, dispatch=True)
        controller = create_steering_controller(model)
        controller.load_steering_vectors(self.steering_dir)

        print(f"Generating response for prompt: {self.prompt}")
        response = controller.apply_steering(
            self.prompt, self.goal_name, self.layer, self.strength, self.max_new_tokens
        )

        print(f"Steered response: {response}")


@dataclass
class SteerInteractiveConfig:
    """Start an interactive steering session.

    Args:
        model_name_or_path: Model name or path to load
        goal_name: Name of the goal for steering
        steering_dir: Directory containing steering vectors
        strength: Steering strength
        max_new_tokens: Maximum new tokens to generate
        token_position: Position of the token to gather activations from
    """

    model_name_or_path: str
    goal_name: str
    steering_dir: str = "steering_vectors"
    strength: float = 1.0
    max_new_tokens: int = 100
    token_position: activations.TokenPosition = activations.TokenPosition.response_avg

    def __call__(self):
        model = load_model(self.model_name_or_path, dispatch=False)
        controller = create_steering_controller(model)
        controller.load_steering_vectors(self.steering_dir)

        run_interactive_loop(controller, self.goal_name, self.strength, self.max_new_tokens)


Steering = Union[ComputeSteeringConfig, ApplySteeringConfig, SteerInteractiveConfig]


def steering_cmd(config: Steering):
    """Compute and apply steering vectors for model generation.

    Subcommands:
    - ComputeSteeringConfig: Compute steering vector from activations
    - ApplySteeringConfig: Apply steering to generate a response
    - SteerInteractiveConfig: Start an interactive steering session
    """
    config()
