"""Tests for folding the binary-probe result JSONs into one table.

The collector is the only place the eight arms meet, so the two things it must not get
wrong are which split a file scored -- that comes from the filename, not the JSON -- and
leaving a multiclass result that happens to share the directory out of a binary table.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "grid_cell_analysis" / "collect_binary_grid_results.py"

sys.path.insert(0, str(REPO / "grid_cell_analysis"))

from collect_binary_grid_results import _split_of  # noqa: E402


def _block(**overrides):
    block = {
        "n_rows": 100,
        "positive_support": 10,
        "positive_rate": 0.1,
        "predicted_positive": 12,
        "accuracy": 0.9,
        "balanced_accuracy": 0.75,
        "precision": 0.5,
        "recall": 0.6,
        "specificity": 0.9,
        "f1": 0.55,
        "auroc": 0.8,
        "tp": 6,
        "fp": 6,
        "tn": 84,
        "fn": 4,
    }
    block.update(overrides)
    return block


def _result(tmp_path, name, positive_class, model_type):
    payload = {
        "probe": {
            "path": str(tmp_path / f"{positive_class}.pt"),
            "positive_class": positive_class,
            "positive_cell_id": 1,
            "model_type": model_type,
        },
        "data": {"path": str(tmp_path / "ds"), "n_entries": 20, "n_trajectories": 5},
        "global": _block(),
        "by_size": {"5": _block(n_rows=40), "9": _block(n_rows=60)},
        "by_complexity": {"0.0": _block(n_rows=100)},
        "by_size_complexity": {"size5|comp0.0": _block(n_rows=40)},
    }
    (tmp_path / name).write_text(json.dumps(payload))


def _run(tmp_path, out):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--results-dir", str(tmp_path), "--out", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_split_comes_off_the_filename():
    assert _split_of(Path("eval_wall_mlp_heldout360.json")) == "heldout360"
    assert _split_of(Path("eval_empty_lr_eval720.json")) == "eval720"


def test_every_block_becomes_a_row(tmp_path):
    _result(tmp_path, "eval_wall_lr_eval720.json", "#", "lr")
    out = tmp_path / "summary.csv"
    assert _run(tmp_path, out).returncode == 0

    rows = list(csv.DictReader(out.open()))
    # global + two sizes + one complexity
    assert len(rows) == 4
    assert {r["group"] for r in rows} == {"global", "size5", "size9", "comp0.0"}
    assert {r["split"] for r in rows} == {"eval720"}
    assert {r["positive_class"] for r in rows} == {"#"}


def test_arms_are_ordered_and_global_comes_first(tmp_path):
    for cls, name in (("G", "goal"), ("_", "empty"), ("#", "wall"), ("A", "agent")):
        _result(tmp_path, f"eval_{name}_lr_eval720.json", cls, "lr")
    out = tmp_path / "summary.csv"
    assert _run(tmp_path, out).returncode == 0

    rows = list(csv.DictReader(out.open()))
    classes_in_order = [r["positive_class"] for r in rows if r["group"] == "global"]
    assert classes_in_order == ["_", "#", "A", "G"]
    assert rows[0]["group"] == "global"


def test_a_multiclass_result_in_the_directory_is_ignored(tmp_path):
    _result(tmp_path, "eval_wall_lr_eval720.json", "#", "lr")
    (tmp_path / "eval_cognitive_map_probe_something.json").write_text(
        json.dumps({"global": {"accuracy": 0.4}, "by_size": {}})
    )
    out = tmp_path / "summary.csv"
    assert _run(tmp_path, out).returncode == 0
    rows = list(csv.DictReader(out.open()))
    assert {r["positive_class"] for r in rows} == {"#"}


def test_an_empty_directory_fails_loudly(tmp_path):
    result = _run(tmp_path, tmp_path / "summary.csv")
    assert result.returncode == 1
    assert "No binary results" in result.stderr


@pytest.mark.parametrize("broken", ["not json at all", "[1, 2, 3]"])
def test_an_unreadable_file_does_not_take_the_run_down(tmp_path, broken):
    _result(tmp_path, "eval_wall_lr_eval720.json", "#", "lr")
    (tmp_path / "eval_broken_lr_eval720.json").write_text(broken)
    out = tmp_path / "summary.csv"
    assert _run(tmp_path, out).returncode == 0
    assert list(csv.DictReader(out.open()))
