"""Shared training utilities: training loop, normalization, device/seed setup."""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_x_raw, batch_y_raw in dataloader:
        batch_x = batch_x_raw.to(device)
        batch_y = batch_y_raw.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def resolve_device(device: str | None) -> torch.device:
    """Resolve device string to torch.device, defaulting to CUDA if available."""
    device_str = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_normalization_params(activations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute mean and std for normalization, clamping std to avoid division by zero.

    Args:
        activations: Input tensor of shape (N, D)

    Returns:
        Tuple of (mean, std) tensors, each of shape (D,)
    """
    mean = activations.mean(dim=0)
    std = activations.std(dim=0)
    std = torch.where(std > 1e-8, std, torch.ones_like(std))
    return mean, std


def normalize_activations(activations: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Apply normalization to activations.

    Args:
        activations: Input tensor of shape (N, D)
        mean: Mean tensor of shape (D,)
        std: Std tensor of shape (D,)

    Returns:
        Normalized activations
    """
    return (activations - mean) / std
