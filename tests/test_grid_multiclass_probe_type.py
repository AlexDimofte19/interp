"""`grid_multiclass`: grid_tile's multiclass scoring on prepare's cells.

Two things are pinned. The cells are exactly `grid_binary`'s, which are prepare's
(`_grid_cell_payload`), so a multiclass table scored per token reproduces an evaluation on the
prepared manifest. And everything else -- the columns, the statistic -- is `grid_tile`'s, so the
analysis that reads a `grid_tile` table reads this one unchanged.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from telos_interp.loudness_analysis.probes import get_probe_type, probe_type_names  # noqa: E402

GRID = ["#######", "#A____#", "#_##__#", "#___#_#", "#_#__G#", "#_____#", "#######"]
TRAJ = {"steps": [{"grid_state": GRID}], "grid_params": {"grid_width": 7, "grid_complexity": 0.2}}
ARGS = SimpleNamespace(pad_to_size=15, max_cells=25, seed=42)


def test_it_is_registered():
    assert "grid_multiclass" in probe_type_names()


def test_the_cells_are_grid_binarys_which_are_prepares():
    multi = get_probe_type("grid_multiclass").step_state(TRAJ, 0, "traj_size7_comp0.2_1", ARGS)
    binary = get_probe_type("grid_binary").step_state(TRAJ, 0, "traj_size7_comp0.2_1", ARGS)
    assert torch.equal(multi[0], binary[0]) and torch.equal(multi[1], binary[1])
    assert multi[1].numel() == 25


def test_the_cells_are_not_grid_tiles_own_draw():
    """The reason the entry exists: grid_tile seeds its draw differently."""
    multi = get_probe_type("grid_multiclass").step_state(TRAJ, 0, "traj_size7_comp0.2_1", ARGS)
    tile = get_probe_type("grid_tile").step_state(TRAJ, 0, "traj_size7_comp0.2_1", ARGS)
    assert not (torch.equal(multi[0], tile[0]) and torch.equal(multi[1], tile[1]))


def test_columns_and_statistic_are_grid_tiles():
    multi, tile = get_probe_type("grid_multiclass"), get_probe_type("grid_tile")
    assert multi.result_columns(["p"], False) == tile.result_columns(["p"], False)
    assert multi.aggregation == tile.aggregation == "counts"
    assert multi.analysis_classes == tile.analysis_classes
