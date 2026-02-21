"""Tests for telos_interp.grid_utils module."""

import pytest
from telos_interp.grid_utils import CELL_ID_TO_SYMBOL, CELL_SYMBOL_TO_ID, parse_grid_state


class TestCellConstants:
    """Tests for cell symbol/ID mappings."""

    def test_symbol_to_id_keys(self):
        """All expected symbols are present."""
        expected = {"A", "#", "G", "_", "D", "K", "?", "+"}
        assert set(CELL_SYMBOL_TO_ID.keys()) == expected

    def test_id_to_symbol_is_inverse(self):
        """CELL_ID_TO_SYMBOL is the inverse of CELL_SYMBOL_TO_ID."""
        for symbol, cell_id in CELL_SYMBOL_TO_ID.items():
            assert CELL_ID_TO_SYMBOL[cell_id] == symbol

    def test_ids_are_unique(self):
        """All IDs are distinct integers."""
        ids = list(CELL_SYMBOL_TO_ID.values())
        assert len(ids) == len(set(ids))


class TestParseGridState:
    """Tests for parse_grid_state."""

    @pytest.fixture
    def grid_5x5(self):
        return [
            "  0 1 2 3 4 ",
            "0 # # # # # ",
            "1 # _ _ G # ",
            "2 # _ # # # ",
            "3 # _ A _ # ",
            "4 # # # # # ",
        ]

    def test_basic_parse(self, grid_5x5):
        """Parse a 5x5 grid and check total cells."""
        result = parse_grid_state(grid_5x5)
        assert len(result) == 25

    def test_cell_identities(self, grid_5x5):
        """Check specific cell identities."""
        result = parse_grid_state(grid_5x5)
        lookup = {(c[0], c[1]): c[2] for c in result}
        # Corners are walls
        assert lookup[(0, 0)] == CELL_SYMBOL_TO_ID["#"]
        # Agent at (3, 2)
        assert lookup[(3, 2)] == CELL_SYMBOL_TO_ID["A"]
        # Goal at (1, 3)
        assert lookup[(1, 3)] == CELL_SYMBOL_TO_ID["G"]
        # Empty at (1, 1)
        assert lookup[(1, 1)] == CELL_SYMBOL_TO_ID["_"]

    def test_padding(self, grid_5x5):
        """Padding to larger size adds '+' cells."""
        result = parse_grid_state(grid_5x5, pad_to_size=7)
        assert len(result) == 49
        lookup = {(c[0], c[1]): c[2] for c in result}
        # Padded cells should be '+'
        assert lookup[(5, 0)] == CELL_SYMBOL_TO_ID["+"]
        assert lookup[(0, 6)] == CELL_SYMBOL_TO_ID["+"]

    def test_pad_to_size_too_small_raises(self, grid_5x5):
        """Padding to a size smaller than the grid raises ValueError."""
        with pytest.raises(ValueError, match="smaller than actual"):
            parse_grid_state(grid_5x5, pad_to_size=3)

    def test_empty_grid_raises(self):
        """Empty grid raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            parse_grid_state([])

    def test_ordered_row_col(self, grid_5x5):
        """Result is ordered by row then column."""
        result = parse_grid_state(grid_5x5)
        for i in range(len(result) - 1):
            r1, c1, _ = result[i]
            r2, c2, _ = result[i + 1]
            assert (r1, c1) < (r2, c2)
