"""Tests for telos_interp.activation_loading module."""

import pytest
import torch
from telos_interp.activation_loading import (
    discover_available_categories,
    discover_available_layers,
    discover_available_steps,
    discover_available_token_indices,
    discover_model_folder,
    discover_trajectory_folders,
    load_activation,
    load_concatenated_activations,
    parse_category_specs,
    parse_index_specification,
)


class TestParseIndexSpecification:
    """Tests for parse_index_specification with available_indices filtering."""

    def test_all_keyword(self):
        result = parse_index_specification("all", [0, 1, 2, 5, 10])
        assert result == [0, 1, 2, 5, 10]

    def test_single_index(self):
        result = parse_index_specification("2", [0, 1, 2, 5])
        assert result == [2]

    def test_missing_index_filtered(self):
        result = parse_index_specification("3", [0, 1, 2, 5])
        assert result == []

    def test_range_inclusive(self):
        result = parse_index_specification("1:5", [0, 1, 2, 3, 5, 10])
        assert result == [1, 2, 3, 5]

    def test_negative_index(self):
        result = parse_index_specification("-1", [0, 1, 2, 5, 10])
        assert result == [10]

    def test_negative_range(self):
        result = parse_index_specification("-3:-1", [0, 1, 2, 3, 4])
        assert result == [2, 3, 4]

    def test_comma_separated(self):
        result = parse_index_specification("0,2,5", [0, 1, 2, 5, 10])
        assert result == [0, 2, 5]

    def test_empty_available(self):
        result = parse_index_specification("all", [])
        assert result == []


class TestParseCategorySpecs:
    def test_all_none(self):
        result = parse_category_specs()
        assert all(v is None for v in result.values())

    def test_some_specified(self):
        result = parse_category_specs(grid_state_indices="all", output_indices="-1")
        assert result["grid_state"] == "all"
        assert result["output"] == "-1"
        assert result["prompt_prefix"] is None
        assert result["prompt_suffix"] is None


class TestDiscoverFunctions:
    """Tests for filesystem discovery functions using temp dirs."""

    @pytest.fixture
    def activation_tree(self, tmp_path):
        """Create a minimal activation directory tree."""
        traj = tmp_path / "traj_0"
        model = traj / "org__model"
        for layer in [0, 5]:
            for step in [0, 1]:
                for cat in ["grid_state", "output"]:
                    folder = model / f"layer_{layer}" / f"step_{step}" / cat
                    folder.mkdir(parents=True)
                    for tok in [0, 1, 2]:
                        torch.save(torch.randn(64), folder / f"{tok}.pt")
        return tmp_path

    def test_discover_trajectory_folders(self, activation_tree):
        folders = discover_trajectory_folders(activation_tree)
        assert len(folders) == 1
        assert folders[0].name == "traj_0"

    def test_discover_model_folder(self, activation_tree):
        traj = activation_tree / "traj_0"
        model = discover_model_folder(traj)
        assert model is not None
        assert "org__model" in model.name

    def test_discover_available_layers(self, activation_tree):
        model = activation_tree / "traj_0" / "org__model"
        layers = discover_available_layers(model)
        assert layers == [0, 5]

    def test_discover_available_steps(self, activation_tree):
        layer = activation_tree / "traj_0" / "org__model" / "layer_0"
        steps = discover_available_steps(layer)
        assert steps == [0, 1]

    def test_discover_available_categories(self, activation_tree):
        step = activation_tree / "traj_0" / "org__model" / "layer_0" / "step_0"
        cats = discover_available_categories(step)
        assert set(cats) == {"grid_state", "output"}

    def test_discover_available_token_indices(self, activation_tree):
        cat = activation_tree / "traj_0" / "org__model" / "layer_0" / "step_0" / "grid_state"
        indices = discover_available_token_indices(cat)
        assert indices == [0, 1, 2]


class TestLoadActivation:
    def test_roundtrip(self, tmp_path):
        t = torch.randn(128)
        path = tmp_path / "test.pt"
        torch.save(t, path)
        loaded = load_activation(path)
        assert torch.allclose(t, loaded)


class TestLoadConcatenatedActivations:
    @pytest.fixture
    def activation_tree(self, tmp_path):
        traj = tmp_path / "traj_0"
        model = traj / "org__model"
        cat_folder = model / "layer_0" / "step_0" / "grid_state"
        cat_folder.mkdir(parents=True)
        for tok in range(3):
            torch.save(torch.randn(64), cat_folder / f"{tok}.pt")
        return traj

    def test_basic_load(self, activation_tree):
        specs = parse_category_specs(grid_state_indices="all")
        result = load_concatenated_activations(activation_tree, 0, 0, specs)
        assert result is not None
        assert result.shape == (64 * 3,)

    def test_no_matching_category(self, activation_tree):
        specs = parse_category_specs(prompt_prefix_indices="all")
        result = load_concatenated_activations(activation_tree, 0, 0, specs)
        assert result is None

    def test_track_contributing_tokens(self, activation_tree):
        specs = parse_category_specs(grid_state_indices="all")
        result, tokens = load_concatenated_activations(activation_tree, 0, 0, specs, track_contributing_tokens=True)
        assert result is not None
        assert len(tokens) == 3
        assert all(cat == "grid_state" for cat, _ in tokens)
