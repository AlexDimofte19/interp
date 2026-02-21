"""Tests for telos_interp.training module."""

import torch
from telos_interp.training import (
    compute_normalization_params,
    normalize_activations,
    resolve_device,
    set_seed,
    train_epoch,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TestResolveDevice:
    def test_explicit_cpu(self):
        assert resolve_device("cpu") == torch.device("cpu")

    def test_none_returns_device(self):
        device = resolve_device(None)
        assert isinstance(device, torch.device)


class TestSetSeed:
    def test_reproducibility(self):
        set_seed(42)
        a = torch.randn(5)
        set_seed(42)
        b = torch.randn(5)
        assert torch.allclose(a, b)


class TestComputeNormalizationParams:
    def test_shape(self):
        x = torch.randn(100, 64)
        mean, std = compute_normalization_params(x)
        assert mean.shape == (64,)
        assert std.shape == (64,)

    def test_std_clamped(self):
        """Constant columns should have std clamped to 1.0."""
        x = torch.zeros(10, 4)
        _, std = compute_normalization_params(x)
        assert (std >= 1.0).all()


class TestNormalizeActivations:
    def test_normalized_stats(self):
        x = torch.randn(200, 16) * 5 + 3
        mean, std = compute_normalization_params(x)
        normed = normalize_activations(x, mean, std)
        # After normalization, mean should be ~0, std ~1
        assert normed.mean(dim=0).abs().max().item() < 0.1
        assert (normed.std(dim=0) - 1.0).abs().max().item() < 0.2


class TestTrainEpoch:
    def test_returns_loss(self):
        """Train epoch returns a finite float loss."""
        model = nn.Linear(16, 3)
        x = torch.randn(32, 16)
        y = torch.randint(0, 3, (32,))
        loader = DataLoader(TensorDataset(x, y), batch_size=8)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss = train_epoch(model, loader, criterion, optimizer, torch.device("cpu"))
        assert isinstance(loss, float)
        assert loss > 0
