"""Probe model architectures and factories for classification and regression probing."""

from typing import Literal

import torch
from torch import nn

# Shared model type literal
ModelType = Literal["lr", "mlp"]


# --- Classification probes ---


class LogisticRegressionProbe(nn.Module):
    """Logistic regression classifier for probing."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLPProbe(nn.Module):
    """Multi-layer perceptron classifier for probing."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list[int],
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def create_classification_model(
    model_type: ModelType,
    input_dim: int,
    num_classes: int,
    hidden_dims: list[int],
    dropout: float,
) -> nn.Module:
    """Create a classification probe model based on model_type."""
    if model_type == "lr":
        return LogisticRegressionProbe(input_dim, num_classes)
    elif model_type == "mlp":
        return MLPProbe(input_dim, num_classes, hidden_dims, dropout)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# --- Regression probes ---


class LinearRegressionProbe(nn.Module):
    """Linear regression probe for distance prediction."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class MLPRegressionProbe(nn.Module):
    """Multi-layer perceptron regression probe for distance prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def create_regression_model(
    model_type: ModelType,
    input_dim: int,
    hidden_dims: list[int],
    dropout: float,
) -> nn.Module:
    """Create a regression probe model based on model_type."""
    if model_type == "lr":
        return LinearRegressionProbe(input_dim)
    elif model_type == "mlp":
        return MLPRegressionProbe(input_dim, hidden_dims, dropout)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
