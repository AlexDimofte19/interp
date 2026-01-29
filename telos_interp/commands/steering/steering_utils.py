"""Utility functions for steering vector computation and application."""

import os

import nnsight
import torch

from telos_interp import steering


def load_activations(path: str) -> torch.Tensor:
    """Load activations from a file.

    Args:
        path: Path to the activations file

    Returns:
        Loaded activations tensor
    """
    return torch.load(path)


def save_steering_vector(
    steering_vector: steering.SteeringVector,
    goal_name: str,
    output_dir: str,
) -> str:
    """Save a steering vector to disk.

    Args:
        steering_vector: The steering vector to save
        goal_name: Name of the goal for the steering vector
        output_dir: Directory to save the steering vector

    Returns:
        Path where the steering vector was saved
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{goal_name}_steering_vector.pt")
    steering_vector.save(path)
    return path


def load_model(model_name_or_path: str, dispatch: bool = True) -> nnsight.LanguageModel:
    """Load a language model using nnsight.

    Args:
        model_name_or_path: Model name or path to load
        dispatch: Whether to use dispatch mode

    Returns:
        Loaded language model
    """
    return nnsight.LanguageModel(model_name_or_path, device_map="auto", dispatch=dispatch)


def create_steering_controller(model: nnsight.LanguageModel) -> steering.SteeringController:
    """Create a steering controller for a model.

    Args:
        model: The language model to control

    Returns:
        Steering controller instance
    """
    return steering.SteeringController(model)


def run_interactive_loop(
    controller: steering.SteeringController,
    goal_name: str,
    strength: float,
    max_new_tokens: int,
) -> None:
    """Run an interactive steering session loop.

    Args:
        controller: The steering controller to use
        goal_name: Name of the goal for steering
        strength: Steering strength
        max_new_tokens: Maximum new tokens to generate
    """
    print(f"Interactive steering session for goal: {goal_name}")
    print("Enter prompts (type 'quit' to exit):")

    while True:
        prompt = input("Prompt: ")
        if prompt.lower() == "quit":
            break

        try:
            response = controller.apply_steering(prompt, goal_name, strength, max_new_tokens)
            print(f"Response: {response}\n")
        except Exception as e:
            print(f"Error: {e}\n")
