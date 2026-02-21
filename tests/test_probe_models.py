"""Tests for telos_interp.probe_models module."""

import pytest
import torch
from telos_interp.probe_models import (
    LinearRegressionProbe,
    LogisticRegressionProbe,
    MLPProbe,
    MLPRegressionProbe,
    create_classification_model,
    create_regression_model,
)


class TestLogisticRegressionProbe:
    def test_output_shape(self):
        model = LogisticRegressionProbe(input_dim=128, num_classes=5)
        x = torch.randn(4, 128)
        out = model(x)
        assert out.shape == (4, 5)

    def test_single_sample(self):
        model = LogisticRegressionProbe(input_dim=64, num_classes=3)
        x = torch.randn(1, 64)
        out = model(x)
        assert out.shape == (1, 3)


class TestMLPProbe:
    def test_output_shape(self):
        model = MLPProbe(input_dim=128, num_classes=5, hidden_dims=[64, 32])
        x = torch.randn(4, 128)
        out = model(x)
        assert out.shape == (4, 5)

    def test_single_hidden_layer(self):
        model = MLPProbe(input_dim=64, num_classes=3, hidden_dims=[32])
        x = torch.randn(2, 64)
        out = model(x)
        assert out.shape == (2, 3)

    def test_dropout(self):
        model = MLPProbe(input_dim=64, num_classes=3, hidden_dims=[32], dropout=0.5)
        x = torch.randn(8, 64)
        model.train()
        out = model(x)
        assert out.shape == (8, 3)


class TestLinearRegressionProbe:
    def test_output_shape(self):
        model = LinearRegressionProbe(input_dim=128)
        x = torch.randn(4, 128)
        out = model(x)
        assert out.shape == (4,)

    def test_single_sample(self):
        model = LinearRegressionProbe(input_dim=64)
        x = torch.randn(1, 64)
        out = model(x)
        assert out.shape == (1,)


class TestMLPRegressionProbe:
    def test_output_shape(self):
        model = MLPRegressionProbe(input_dim=128, hidden_dims=[64, 32])
        x = torch.randn(4, 128)
        out = model(x)
        assert out.shape == (4,)


class TestCreateClassificationModel:
    def test_lr(self):
        model = create_classification_model("lr", 128, 5, [64], 0.1)
        assert isinstance(model, LogisticRegressionProbe)

    def test_mlp(self):
        model = create_classification_model("mlp", 128, 5, [64], 0.1)
        assert isinstance(model, MLPProbe)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown model_type"):
            create_classification_model("xgb", 128, 5, [64], 0.1)


class TestCreateRegressionModel:
    def test_lr(self):
        model = create_regression_model("lr", 128, [64], 0.1)
        assert isinstance(model, LinearRegressionProbe)

    def test_mlp(self):
        model = create_regression_model("mlp", 128, [64], 0.1)
        assert isinstance(model, MLPRegressionProbe)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown model_type"):
            create_regression_model("xgb", 128, [64], 0.1)
