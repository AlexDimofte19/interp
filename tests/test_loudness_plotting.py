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
    assert len(F.FIGURES) == 19


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


# ---------------------------------------------------------------------------------------------
# The grid drawers: a counts-mode table, where one row summarises a whole step's CELLS.
# ---------------------------------------------------------------------------------------------


def _grid_table(n_traj: int = 12, per_traj: int = 25, n_cells: int = 8) -> pd.DataFrame:
    """A structurally complete `grid_tile` table carrying FOUR rulers.

    One row is one (trajectory, step, token) and owns `n_cells` cells, so it carries
    per-class counts rather than a verdict -- which is the whole reason these drawers exist
    separately from the `probe` ones. Two of the four rulers use the LEGACY spelling
    (`{lens}_mass_L15`, which is what every table on disk carries for the direction
    vocabulary) so the discovery path is exercised as well as the canonical one.
    """
    rng = np.random.default_rng(7)
    n = n_traj * per_traj
    classes = [0, 1, 2, 3]
    df = pd.DataFrame(
        {
            "name": np.repeat([f"t{i}" for i in range(n_traj)], per_traj),
            "step": 0,
            "abs_pos": np.tile(np.arange(per_traj), n_traj),
            "token_idx": np.tile(np.arange(per_traj), n_traj),
            "token": "x",
            "n_cells": n_cells,
            # legacy direction spelling, canonical grid spelling
            "jlens_mass_L15": rng.normal(-3, 1, n),
            "logitlens_mass_L15": rng.normal(-3, 1, n),
            "jlens_grid_logmass_L15": rng.normal(-2, 1, n),
            "logitlens_grid_logmass_L15": rng.normal(-2, 1, n),
        }
    )
    # Cells split evenly over the four real classes; classes 4-6 exist in the schema and get
    # no cells, which is what the drawer must narrow away.
    for c in classes + [4, 5, 6]:
        df[f"n_true_{c}"] = n_cells // len(classes) if c in classes else 0
    for probe in ("arm.p_lr", "arm.p_mlp"):
        df[f"{probe}_n_correct"] = rng.integers(0, n_cells + 1, n)
        for c in classes + [4, 5, 6]:
            df[f"{probe}_correct_{c}"] = rng.integers(0, n_cells // len(classes) + 1, n) if c in classes else 0
    return df


def _grid_args(out, deciles=4, boot=5):
    return argparse.Namespace(
        out=out, out_dir=out, layer=15, deciles=deciles, boot=boot, grid_probe=None, reference_gap=0.1559
    )


def test_the_grid_drawer_finds_every_ruler_including_legacy_spellings(tmp_path):
    df = _grid_table()
    found = F.grid_rulers(df, 15)
    assert [(lens, sig) for lens, sig, _ in found] == [
        ("jlens", "direction"),
        ("logitlens", "direction"),
        ("jlens", "grid"),
        ("logitlens", "grid"),
    ]
    # the legacy column is what a direction ruler resolves to
    assert dict(((l, s), c) for l, s, c in found)[("jlens", "direction")] == "jlens_mass_L15"


def test_the_grid_drawer_writes_its_figures_and_tables(tmp_path):
    out = tmp_path / "figs"
    out.mkdir(parents=True)
    F.DRAWERS["grid"](_grid_table(), _grid_args(out))

    pngs = sorted(p.name for p in out.glob("*.png"))
    assert pngs == [
        "grid_accuracy_by_loudness.png",
        "grid_per_class_by_loudness.png",
        "grid_ruler_gaps.png",
    ]
    # one decile table per (probe, ruler) -- 2 probes x 4 rulers
    assert len(list((out / "tables").glob("*_decile.csv"))) == 8


def test_the_grid_drawer_narrows_to_classes_that_have_cells(tmp_path):
    """Classes 4-6 exist in the schema with no cells. Scoring them would not change a
    balanced accuracy (`bal_acc_from_counts` drops an unsupported class) but it WOULD put
    the chance line at 1/7 instead of 1/4 and draw three empty panels."""
    out = tmp_path / "figs"
    out.mkdir(parents=True)
    args = _grid_args(out)
    F.DRAWERS["grid"](_grid_table(), args)
    assert args.classes == [0, 1, 2, 3]
    assert args.chance == pytest.approx(0.25)


def test_the_grid_drawer_refuses_a_table_with_no_ruler(tmp_path):
    df = _grid_table().drop(
        columns=["jlens_mass_L15", "logitlens_mass_L15", "jlens_grid_logmass_L15", "logitlens_grid_logmass_L15"]
    )
    out = tmp_path / "figs"
    out.mkdir(parents=True)
    with pytest.raises(SystemExit, match="no loudness column"):
        F.DRAWERS["grid"](df, _grid_args(out))


def test_the_grid_drawer_refuses_a_table_with_no_probe(tmp_path):
    df = _grid_table()
    df = df.drop(columns=[c for c in df.columns if "_n_correct" in c])
    out = tmp_path / "figs"
    out.mkdir(parents=True)
    with pytest.raises(SystemExit, match="no probe columns"):
        F.DRAWERS["grid"](df, _grid_args(out))


def test_the_grid_cli_draws_and_records_provenance(tmp_path):
    table = tmp_path / "grid.csv"
    _grid_table().to_csv(table, index=False)
    out = tmp_path / "figs"

    F.main(["--grid-table", str(table), "--out", str(out), "--signal-name", "grid", "--deciles", "4", "--boot", "5"])

    assert len(list(out.glob("*.png"))) == 3
    cfg = json.loads((out / "run_config.json").read_text())
    assert cfg["measurement"]["loudness_column"] == "jlens_grid_logmass_L15"
