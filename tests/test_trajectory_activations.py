"""Unit tests for trajectory activation extraction functionality."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from telos_interp.commands.gather_activations.gather_activations_utils import (
    parse_index_specification,
    sanitize_model_id,
)
from telos_interp.commands.gather_activations.trajectory_activations import (
    copy_activations_to_step,
    reconstruct_input_text,
    reconstruct_text_from_tokens,
    save_activations_to_files,
)


# =============================================================================
# Tests for parse_index_specification
# =============================================================================


class TestParseIndexSpecification:
    """Tests for the parse_index_specification function."""

    def test_all_keyword(self):
        """Test 'all' keyword returns all indices."""
        result = parse_index_specification("all", 10)
        assert result == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def test_all_keyword_case_insensitive(self):
        """Test 'all' keyword is case insensitive."""
        assert parse_index_specification("ALL", 5) == [0, 1, 2, 3, 4]
        assert parse_index_specification("All", 5) == [0, 1, 2, 3, 4]
        assert parse_index_specification("  all  ", 5) == [0, 1, 2, 3, 4]

    def test_single_index(self):
        """Test single index parsing."""
        assert parse_index_specification("0", 10) == [0]
        assert parse_index_specification("5", 10) == [5]
        assert parse_index_specification("9", 10) == [9]

    def test_multiple_indices(self):
        """Test comma-separated indices."""
        assert parse_index_specification("0,1,5", 10) == [0, 1, 5]
        assert parse_index_specification("0, 1, 5", 10) == [0, 1, 5]  # with spaces
        assert parse_index_specification("5,0,1", 10) == [0, 1, 5]  # sorted output

    def test_duplicate_indices(self):
        """Test duplicate indices are deduplicated."""
        assert parse_index_specification("1,1,1,2,2", 10) == [1, 2]

    def test_range_inclusive(self):
        """Test range specification (inclusive)."""
        assert parse_index_specification("0-5", 10) == [0, 1, 2, 3, 4, 5]
        assert parse_index_specification("3-7", 10) == [3, 4, 5, 6, 7]

    def test_range_single_element(self):
        """Test range with same start and end."""
        assert parse_index_specification("5-5", 10) == [5]

    def test_mixed_indices_and_ranges(self):
        """Test mixed format with indices and ranges."""
        assert parse_index_specification("0,5-7,9", 10) == [0, 5, 6, 7, 9]
        assert parse_index_specification("0-2,5,8-9", 10) == [0, 1, 2, 5, 8, 9]

    def test_negative_index_last(self):
        """Test negative index -1 returns last element."""
        assert parse_index_specification("-1", 10) == [9]
        assert parse_index_specification("-1", 5) == [4]

    def test_negative_index_from_end(self):
        """Test negative indices counting from end."""
        assert parse_index_specification("-2", 10) == [8]
        assert parse_index_specification("-3", 10) == [7]
        assert parse_index_specification("-10", 10) == [0]

    def test_negative_range(self):
        """Test range with negative indices."""
        # Last 3 elements of a 10-element list
        assert parse_index_specification("-3--1", 10) == [7, 8, 9]
        # Last 5 elements
        assert parse_index_specification("-5--1", 10) == [5, 6, 7, 8, 9]

    def test_mixed_positive_negative(self):
        """Test mixing positive and negative indices."""
        assert parse_index_specification("0,-1", 10) == [0, 9]
        assert parse_index_specification("0,5,-1", 10) == [0, 5, 9]

    def test_positive_to_negative_range(self):
        """Test range from positive to negative index."""
        # From index 5 to last element (index 9) in 10-element list
        assert parse_index_specification("5--1", 10) == [5, 6, 7, 8, 9]

    def test_negative_to_positive_range(self):
        """Test range from negative to positive index."""
        # From -5 (index 5) to index 7 in 10-element list
        assert parse_index_specification("-5-7", 10) == [5, 6, 7]

    def test_empty_list_max_index_zero(self):
        """Test with max_index=0 returns empty for 'all'."""
        assert parse_index_specification("all", 0) == []

    def test_index_out_of_bounds_positive(self):
        """Test positive index out of bounds raises error."""
        with pytest.raises(ValueError, match="out of bounds"):
            parse_index_specification("10", 10)
        with pytest.raises(ValueError, match="out of bounds"):
            parse_index_specification("15", 10)

    def test_index_out_of_bounds_negative(self):
        """Test negative index out of bounds raises error."""
        with pytest.raises(ValueError, match="out of bounds"):
            parse_index_specification("-11", 10)
        with pytest.raises(ValueError, match="out of bounds"):
            parse_index_specification("-15", 10)

    def test_range_out_of_bounds(self):
        """Test range with out of bounds values raises error."""
        with pytest.raises(ValueError, match="out of bounds"):
            parse_index_specification("0-15", 10)
        with pytest.raises(ValueError, match="out of bounds"):
            parse_index_specification("8-12", 10)

    def test_invalid_specification(self):
        """Test invalid specification raises error."""
        with pytest.raises(ValueError, match="Invalid index specification"):
            parse_index_specification("abc", 10)
        with pytest.raises(ValueError, match="Invalid index specification"):
            parse_index_specification("1.5", 10)

    def test_whitespace_handling(self):
        """Test whitespace is handled correctly."""
        assert parse_index_specification("  0  ,  1  ,  2  ", 10) == [0, 1, 2]
        assert parse_index_specification(" 0 - 5 ", 10) == [0, 1, 2, 3, 4, 5]


# =============================================================================
# Tests for sanitize_model_id
# =============================================================================


class TestSanitizeModelId:
    """Tests for the sanitize_model_id function."""

    def test_single_slash(self):
        """Test model ID with single slash."""
        assert sanitize_model_id("openai/gpt-oss-20b") == "openai__gpt-oss-20b"
        assert sanitize_model_id("meta-llama/Llama-2-7b") == "meta-llama__Llama-2-7b"

    def test_multiple_slashes(self):
        """Test model ID with multiple slashes."""
        assert sanitize_model_id("org/model/variant") == "org__model__variant"

    def test_no_slash(self):
        """Test model ID without slash."""
        assert sanitize_model_id("gpt2") == "gpt2"
        assert sanitize_model_id("bert-base-uncased") == "bert-base-uncased"

    def test_empty_string(self):
        """Test empty string."""
        assert sanitize_model_id("") == ""

    def test_preserves_other_characters(self):
        """Test other special characters are preserved."""
        assert sanitize_model_id("org/model-v1.0_beta") == "org__model-v1.0_beta"


# =============================================================================
# Tests for reconstruct_text_from_tokens
# =============================================================================


class TestReconstructTextFromTokens:
    """Tests for the reconstruct_text_from_tokens function."""

    def test_simple_tokens(self):
        """Test simple token reconstruction."""
        tokens = [
            {"token": "Hello", "id": 0},
            {"token": " ", "id": 1},
            {"token": "world", "id": 2},
        ]
        assert reconstruct_text_from_tokens(tokens) == "Hello world"

    def test_special_characters(self):
        """Test tokens with special characters (like GPT tokenizer output)."""
        tokens = [
            {"token": "Ġ", "id": 0},  # Space
            {"token": "Hello", "id": 1},
            {"token": "Ċ", "id": 2},  # Newline
        ]
        assert reconstruct_text_from_tokens(tokens) == "ĠHelloĊ"

    def test_empty_tokens(self):
        """Test empty token list."""
        assert reconstruct_text_from_tokens([]) == ""

    def test_single_token(self):
        """Test single token."""
        tokens = [{"token": "test", "id": 0}]
        assert reconstruct_text_from_tokens(tokens) == "test"


# =============================================================================
# Tests for reconstruct_input_text
# =============================================================================


class TestReconstructInputText:
    """Tests for the reconstruct_input_text function."""

    @pytest.fixture
    def sample_trajectory(self):
        """Create a sample trajectory structure for testing."""
        return {
            "prompt": {
                "prompt_prefix_tokens": [
                    {"token": "System:", "id": 0},
                    {"token": " ", "id": 1},
                ],
                "prompt_suffix_tokens": [
                    {"token": "\n", "id": 0},
                    {"token": "Answer:", "id": 1},
                ],
            },
            "steps": [
                {
                    "step_id": 0,
                    "grid_state_tokens": [
                        {"token": "Grid0", "id": 0},
                    ],
                },
                {
                    "step_id": 1,
                    "grid_state_tokens": [
                        {"token": "Grid1", "id": 0},
                    ],
                },
            ],
        }

    def test_step_0(self, sample_trajectory):
        """Test reconstruction for step 0."""
        result = reconstruct_input_text(sample_trajectory, step_idx=0)
        assert result == "System: Grid0\nAnswer:"

    def test_step_1(self, sample_trajectory):
        """Test reconstruction for step 1."""
        result = reconstruct_input_text(sample_trajectory, step_idx=1)
        assert result == "System: Grid1\nAnswer:"

    def test_structure_order(self, sample_trajectory):
        """Test that prefix + grid + suffix order is maintained."""
        result = reconstruct_input_text(sample_trajectory, step_idx=0)
        assert result.startswith("System:")
        assert result.endswith("Answer:")
        assert "Grid0" in result


# =============================================================================
# Tests for save_activations_to_files
# =============================================================================


class TestSaveActivationsToFiles:
    """Tests for the save_activations_to_files function."""

    def test_save_with_step(self):
        """Test saving activations with step index."""
        activations = {
            0: {0: torch.randn(768), 1: torch.randn(768)},
            5: {0: torch.randn(768), 1: torch.randn(768)},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_base = Path(tmpdir)
            save_activations_to_files(
                activations, output_base, step_idx=0, category="grid_state"
            )

            # Check directory structure
            assert (output_base / "layer_0" / "step_0" / "grid_state").exists()
            assert (output_base / "layer_5" / "step_0" / "grid_state").exists()

            # Check files exist
            assert (output_base / "layer_0" / "step_0" / "grid_state" / "0.pt").exists()
            assert (output_base / "layer_0" / "step_0" / "grid_state" / "1.pt").exists()
            assert (output_base / "layer_5" / "step_0" / "grid_state" / "0.pt").exists()
            assert (output_base / "layer_5" / "step_0" / "grid_state" / "1.pt").exists()

            # Check file contents
            loaded = torch.load(output_base / "layer_0" / "step_0" / "grid_state" / "0.pt")
            assert loaded.shape == (768,)

    def test_save_without_step(self):
        """Test saving activations without step index (for prompt_prefix)."""
        activations = {
            0: {0: torch.randn(768)},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_base = Path(tmpdir)
            save_activations_to_files(
                activations, output_base, step_idx=None, category="prompt_prefix"
            )

            # Check directory structure (no step folder)
            assert (output_base / "layer_0" / "prompt_prefix").exists()
            assert (output_base / "layer_0" / "prompt_prefix" / "0.pt").exists()

    def test_empty_activations(self):
        """Test saving empty activations dict."""
        activations: dict = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_base = Path(tmpdir)
            save_activations_to_files(
                activations, output_base, step_idx=0, category="output"
            )
            # Should not create any directories
            assert not (output_base / "layer_0").exists()


# =============================================================================
# Tests for copy_activations_to_step
# =============================================================================


class TestCopyActivationsToStep:
    """Tests for the copy_activations_to_step function."""

    def test_copy_to_step(self):
        """Test copying activations from source to step folder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            source_base = tmpdir / "source"
            output_base = tmpdir / "output"

            # Create source structure
            source_dir = source_base / "layer_0" / "prompt_prefix"
            source_dir.mkdir(parents=True)
            torch.save(torch.randn(768), source_dir / "0.pt")
            torch.save(torch.randn(768), source_dir / "1.pt")

            source_dir_l5 = source_base / "layer_5" / "prompt_prefix"
            source_dir_l5.mkdir(parents=True)
            torch.save(torch.randn(768), source_dir_l5 / "0.pt")

            # Copy to step 2
            copy_activations_to_step(
                source_base, output_base, step_idx=2, category="prompt_prefix", layer_indices=[0, 5]
            )

            # Check target structure
            assert (output_base / "layer_0" / "step_2" / "prompt_prefix" / "0.pt").exists()
            assert (output_base / "layer_0" / "step_2" / "prompt_prefix" / "1.pt").exists()
            assert (output_base / "layer_5" / "step_2" / "prompt_prefix" / "0.pt").exists()

    def test_copy_nonexistent_source(self):
        """Test copying from nonexistent source doesn't raise error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            source_base = tmpdir / "nonexistent"
            output_base = tmpdir / "output"

            # Should not raise error
            copy_activations_to_step(
                source_base, output_base, step_idx=0, category="prompt_prefix", layer_indices=[0]
            )

            # Nothing should be created
            assert not (output_base / "layer_0").exists()


# =============================================================================
# Integration-style tests using mock trajectory data
# =============================================================================


class TestTrajectoryIntegration:
    """Integration tests using realistic trajectory structures."""

    @pytest.fixture
    def realistic_trajectory(self):
        """Create a realistic trajectory structure mimicking actual output."""
        return {
            "grid_params": {
                "grid_width": 5,
                "grid_height": 5,
            },
            "model_params": {
                "model_id": "openai/gpt-oss-20b",
                "provider": "together",
            },
            "prompt": {
                "prompt_template": "...",
                "prompt_template_n_tokens": 365,
                "prompt_prefix_n_tokens": 357,
                "prompt_prefix_tokens": [
                    {"id": i, "token": f"prefix_{i}", "token_id": i, "token_groups": ["prompt"]}
                    for i in range(10)
                ],
                "prompt_suffix_n_tokens": 4,
                "prompt_suffix_tokens": [
                    {"id": i, "token": f"suffix_{i}", "token_id": i, "token_groups": ["prompt"]}
                    for i in range(4)
                ],
            },
            "steps": [
                {
                    "step_id": 0,
                    "grid_state": ["...", "..."],
                    "grid_state_n_tokens": 42,
                    "grid_state_tokens": [
                        {"id": i, "token": f"grid0_{i}", "token_id": i, "token_groups": ["prompt", "grid_state"]}
                        for i in range(42)
                    ],
                    "prompt_suffix_n_tokens": 4,
                    "prompt_suffix_tokens": [
                        {"id": i, "token": f"suffix_{i}", "token_id": i, "token_groups": ["prompt"]}
                        for i in range(4)
                    ],
                    "agent_action": "UP",
                    "output_text": "...",
                    "output_n_tokens": 100,
                    "output_tokens": [
                        {"id": i, "token": f"out0_{i}", "token_id": i, "token_groups": ["output"]}
                        for i in range(100)
                    ],
                },
                {
                    "step_id": 1,
                    "grid_state": ["...", "..."],
                    "grid_state_n_tokens": 42,
                    "grid_state_tokens": [
                        {"id": i, "token": f"grid1_{i}", "token_id": i, "token_groups": ["prompt", "grid_state"]}
                        for i in range(42)
                    ],
                    "prompt_suffix_n_tokens": 4,
                    "prompt_suffix_tokens": [
                        {"id": i, "token": f"suffix_{i}", "token_id": i, "token_groups": ["prompt"]}
                        for i in range(4)
                    ],
                    "agent_action": "RIGHT",
                    "output_text": "...",
                    "output_n_tokens": 80,
                    "output_tokens": [
                        {"id": i, "token": f"out1_{i}", "token_id": i, "token_groups": ["output"]}
                        for i in range(80)
                    ],
                },
            ],
        }

    def test_token_counts(self, realistic_trajectory):
        """Test that token counts match expected values."""
        assert len(realistic_trajectory["prompt"]["prompt_prefix_tokens"]) == 10
        assert len(realistic_trajectory["prompt"]["prompt_suffix_tokens"]) == 4
        assert len(realistic_trajectory["steps"][0]["grid_state_tokens"]) == 42
        assert len(realistic_trajectory["steps"][0]["output_tokens"]) == 100

    def test_reconstruct_different_steps(self, realistic_trajectory):
        """Test reconstruction produces different results for different steps."""
        text_step0 = reconstruct_input_text(realistic_trajectory, step_idx=0)
        text_step1 = reconstruct_input_text(realistic_trajectory, step_idx=1)

        # Should have same prefix and suffix
        assert text_step0.startswith("prefix_0")
        assert text_step1.startswith("prefix_0")
        assert text_step0.endswith("suffix_3")
        assert text_step1.endswith("suffix_3")

        # But different grid content
        assert "grid0_" in text_step0
        assert "grid1_" in text_step1
        assert "grid1_" not in text_step0
        assert "grid0_" not in text_step1

    def test_parse_indices_for_categories(self, realistic_trajectory):
        """Test index parsing with realistic token counts."""
        n_prefix = len(realistic_trajectory["prompt"]["prompt_prefix_tokens"])
        n_suffix = len(realistic_trajectory["prompt"]["prompt_suffix_tokens"])
        n_grid = len(realistic_trajectory["steps"][0]["grid_state_tokens"])
        n_output = len(realistic_trajectory["steps"][0]["output_tokens"])

        # Test "all" for each category
        assert len(parse_index_specification("all", n_prefix)) == 10
        assert len(parse_index_specification("all", n_suffix)) == 4
        assert len(parse_index_specification("all", n_grid)) == 42
        assert len(parse_index_specification("all", n_output)) == 100

        # Test last token for each category
        assert parse_index_specification("-1", n_prefix) == [9]
        assert parse_index_specification("-1", n_suffix) == [3]
        assert parse_index_specification("-1", n_grid) == [41]
        assert parse_index_specification("-1", n_output) == [99]

    def test_save_trajectory_json_roundtrip(self, realistic_trajectory):
        """Test that trajectory can be saved and loaded via JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(realistic_trajectory, f)
            f.flush()

            # Load it back
            with open(f.name) as rf:
                loaded = json.load(rf)

            assert loaded["model_params"]["model_id"] == "openai/gpt-oss-20b"
            assert len(loaded["steps"]) == 2


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
