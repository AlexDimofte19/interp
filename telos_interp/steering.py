"""
Simple steering skeleton using nnsight for activation patching/intervention.

This module provides the core steering functionality without trajectory generation,
goal definitions, or other upstream concerns. It focuses purely on applying
steering vectors to model activations using nnsight's intervention capabilities.

Based on nnsight activation patching methodology:
https://nnsight.net/notebooks/tutorials/causal_mediation_analysis/activation_patching/
"""

import os

import torch
from nnsight import LanguageModel

from .activations import TokenPosition


class SteeringVector:
    """Simple steering vector for applying to model activations."""

    def __init__(self, vector: torch.Tensor, layer_indices: list[int] | None = None):
        """
        Initialize steering vector.

        Args:
            vector: Steering vector of shape (num_layers, hidden_size)
            layer_indices: Which layers to apply steering to (None = all layers)
        """
        self.vector = vector
        self.layer_indices = layer_indices or list(range(vector.shape[0]))
        self.strength = 1.0

    def save(self, path: str):
        """Save steering vector to file."""
        torch.save({"vector": self.vector, "layer_indices": self.layer_indices, "strength": self.strength}, path)

    @classmethod
    def load(cls, path: str) -> "SteeringVector":
        """Load steering vector from file."""
        data = torch.load(path, map_location="cpu")
        vector = cls(data["vector"], data["layer_indices"])
        vector.strength = data.get("strength", 1.0)
        return vector


class SteeringController:
    """Core steering controller using nnsight for activation intervention."""

    def __init__(self, model: LanguageModel, token_position: TokenPosition = TokenPosition.response_avg):
        self.model = model
        self.token_position = token_position
        self.steering_vectors: dict[str, SteeringVector] = {}

    def add_steering_vector(self, name: str, vector: SteeringVector):
        """Add a steering vector for a specific goal."""
        self.steering_vectors[name] = vector

    def apply_steering(
        self, prompt: str, goal_name: str, strength: float | None = None, max_new_tokens: int = 100
    ) -> tuple[str, torch.Tensor]:
        """
        Apply steering to generate a response using nnsight intervention.

        Args:
            prompt: Input prompt
            goal_name: Name of goal to steer towards
            strength: Steering strength (None = use default)
            max_new_tokens: Maximum tokens to generate

        Returns:
            Tuple of (generated_response, final_activations)
        """
        if goal_name not in self.steering_vectors:
            raise ValueError(f"No steering vector found for goal: {goal_name}")

        steering_vector = self.steering_vectors[goal_name]
        if strength is not None:
            steering_vector.strength = strength

        # Tokenize input
        input_ids = self.model.tokenizer(prompt, return_tensors="pt").input_ids
        prompt_length = input_ids.shape[1]

        # Apply steering using nnsight intervention
        with torch.no_grad():
            with self.model.trace(input_ids):
                # Apply steering to each specified layer
                for layer_idx in steering_vector.layer_indices:
                    if layer_idx < self.model.model.config.num_hidden_layers:
                        # Get the layer output
                        layer_output = self.model.model.layers[layer_idx].output

                        # Apply steering based on token position
                        if self.token_position == TokenPosition.response_avg:
                            # Apply to all response tokens
                            steered_output = layer_output.clone()
                            steered_output[0, prompt_length:, :] += (
                                steering_vector.strength * steering_vector.vector[layer_idx]
                            )
                            self.model.model.layers[layer_idx].output = steered_output

                        elif self.token_position == TokenPosition.prompt_last:
                            # Apply to last prompt token
                            steered_output = layer_output.clone()
                            steered_output[0, prompt_length - 1, :] += (
                                steering_vector.strength * steering_vector.vector[layer_idx]
                            )
                            self.model.model.layers[layer_idx].output = steered_output

                        elif self.token_position == TokenPosition.prompt_avg:
                            # Apply to all prompt tokens
                            steered_output = layer_output.clone()
                            steered_output[0, :prompt_length, :] += (
                                steering_vector.strength * steering_vector.vector[layer_idx]
                            )
                            self.model.model.layers[layer_idx].output = steered_output

                # Generate response with steering applied
                output = self.model.model.generate(
                    input_ids, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.95
                )

        # Extract response
        response_tokens = output[0, prompt_length:]
        response = self.model.tokenizer.decode(response_tokens, skip_special_tokens=True)

        # Get final activations for the response
        final_activations = self._get_final_activations(prompt, response)

        return response, final_activations

    def _get_final_activations(self, prompt: str, response: str) -> torch.Tensor:
        """Get final activations for the prompt + response."""
        from .activations import run_model_and_gather_activations_at_token_position

        return run_model_and_gather_activations_at_token_position(self.model, prompt, response, self.token_position)

    def save_steering_vectors(self, directory: str):
        """Save all steering vectors to directory."""
        os.makedirs(directory, exist_ok=True)

        for goal_name, vector in self.steering_vectors.items():
            path = os.path.join(directory, f"{goal_name}_steering_vector.pt")
            vector.save(path)

    def load_steering_vectors(self, directory: str):
        """Load steering vectors from directory."""
        for goal_name in os.listdir(directory):
            if goal_name.endswith("_steering_vector.pt"):
                name = goal_name.replace("_steering_vector.pt", "")
                path = os.path.join(directory, goal_name)
                self.steering_vectors[name] = SteeringVector.load(path)


def compute_steering_vector(
    successful_activations: torch.Tensor, failed_activations: torch.Tensor, method: str = "mean_difference"
) -> SteeringVector:
    """
    Compute steering vector from successful and failed activations.

    Args:
        successful_activations: Shape (num_samples, num_layers, hidden_size)
        failed_activations: Shape (num_samples, num_layers, hidden_size)
        method: Method for computing steering vector

    Returns:
        SteeringVector object
    """
    if method == "mean_difference":
        successful_mean = successful_activations.mean(dim=0)  # (num_layers, hidden_size)
        failed_mean = failed_activations.mean(dim=0)
        steering_vector = successful_mean - failed_mean
        return SteeringVector(steering_vector)

    elif method == "pca":
        # Compute per-sample differences
        differences = successful_activations - failed_activations.mean(dim=0, keepdim=True)

        # Reshape for PCA: (num_samples * num_layers, hidden_size)
        differences_flat = differences.view(-1, differences.shape[-1])

        # Compute PCA
        from sklearn.decomposition import PCA

        pca = PCA(n_components=1)
        pca_vector = pca.fit_transform(differences_flat)

        # Reshape back to (num_layers, hidden_size)
        steering_vector = pca_vector.reshape(differences.shape[1], -1)

        return SteeringVector(steering_vector)

    else:
        raise ValueError(f"Unknown method: {method}")


# Example usage
def main():
    """Example usage of the steering skeleton."""
    # Load model
    model = LanguageModel("microsoft/DialoGPT-medium", device_map="auto")

    # Initialize steering controller
    controller = SteeringController(model, TokenPosition.response_avg)

    # Example: Load pre-computed steering vector
    # steering_vector = SteeringVector.load("path/to/steering_vector.pt")
    # controller.add_steering_vector("humorous", steering_vector)

    # Apply steering
    # response, activations = controller.apply_steering("How do computers work?", "humorous")
    # print(f"Steered response: {response}")

    # Print controller info to avoid unused variable warning
    print(f"Steering controller initialized with {len(controller.steering_vectors)} steering vectors")


if __name__ == "__main__":
    main()
