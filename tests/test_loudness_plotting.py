"""The plotting module must actually draw, not merely import.

The three plotters were merged into one registry plus one CLI, and the sixteen figure
functions have seven different signatures -- so "it imports" proves very little. These build a
small but structurally complete table and run a drawer end to end, asserting PNGs land.

Everything is under tmp_path; matplotlib is forced to Agg by _style.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import pytest

from telos_interp.loudness_analysis.plotting import figures as F


def test_every_figure_declares_a_table_that_has_a_drawer():
    """A figure whose table has no drawer would be listed and then be unreachable."""
    assert set(f.table for f in F.FIGURES.values()) <= set(F.DRAWERS)
    assert len(F.FIGURES) == 16


def test_get_figure_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="Unknown figure"):
        F.get_figure("no_such_figure")


def _rollout_table(n_traj: int = 10, per_traj: int = 20) -> pd.DataFrame:
    """Ten trajectories of twenty CONSECUTIVE tokens.

    fig_position_control cuts terciles of reasoning_pos / max(reasoning_pos) within each
    trajectory, so a fixture that interleaves trajectories leaves each with two distinct
    positions and an empty tercile.
    """
    rng = np.random.default_rng(0)
    n = n_traj * per_traj
    return pd.DataFrame(
        {
            "name": np.repeat([f"t{i}" for i in range(n_traj)], per_traj),
            "step": 0,
            "token_idx": np.tile(np.arange(per_traj), n_traj),
            "reasoning_pos": np.tile(np.arange(per_traj), n_traj),
            "jlens_logmass_L15": rng.normal(-3, 1, n),
            "logitlens_logmass_L15": rng.normal(-3, 1, n),
            "model_action": rng.choice(["UP", "DOWN", "LEFT", "RIGHT"], n),
            "answer_prob": rng.uniform(0.25, 1.0, n),
            "correct": rng.integers(0, 2, n),
        }
    )


def test_the_rollout_drawer_writes_its_figures(tmp_path):
    table = tmp_path / "rollout.csv"
    _rollout_table().to_csv(table, index=False)
    out = tmp_path / "figs"

    args = argparse.Namespace(
        out=out,
        out_dir=out,
        per_token=table,
        lenses="jlens,logitlens",  # draw_rollout splits this itself
        layer=15,
    )
    out.mkdir(parents=True)
    F.DRAWERS["rollout"](pd.read_csv(table, keep_default_na=False, na_values=[""]), args)

    pngs = sorted(p.name for p in out.glob("*.png"))
    assert pngs, "the rollout drawer produced no figures"
    assert len(pngs) == 3, f"expected the three rollout figures, got {pngs}"


def test_the_cli_lists_every_figure_with_its_table(capsys):
    assert F.main(["--list", "--out", "/tmp/unused"]) == 0
    printed = capsys.readouterr().out
    for name in F.figure_names():
        assert name in printed


def test_the_cli_refuses_a_figure_without_its_table(tmp_path):
    with pytest.raises(SystemExit, match="needs a rollout table"):
        F.main(["--figure", "rulers", "--out", str(tmp_path)])


def test_the_cli_refuses_when_no_table_is_given_at_all(tmp_path):
    with pytest.raises(SystemExit, match="no input tables given"):
        F.main(["--out", str(tmp_path)])


def test_the_cli_draws_and_records_provenance(tmp_path):
    table = tmp_path / "rollout.csv"
    _rollout_table().to_csv(table, index=False)
    out = tmp_path / "figs"

    F.main(["--rollout-table", str(table), "--out", str(out), "--lens", "jlens"])

    assert sorted(p.suffix for p in out.glob("*.png"))
    cfg = json.loads((out / "run_config.json").read_text())
    assert cfg["measurement"]["axis_label"] == "J-lens direction logmass"
    assert cfg["parameters"]["figures"]
