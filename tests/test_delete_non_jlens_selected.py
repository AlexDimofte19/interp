"""Tests for scripts/delete_non_jlens_selected.py.

The load-bearing one is `test_pruning_equals_filtered_gathering`: a trajectory gathered in
full and then pruned must be byte-identical to one gathered through the filter in the first
place. That equivalence is the whole reason both paths call `jlens_top_filter` instead of
each deciding for itself, and it is what makes the pruning safe to run on the several
thousand trajectories that were gathered before the filter existed.

The rest pin the safety behaviour: a destructive script that has misread the layout must
delete nothing rather than delete the wrong thing.
"""

import importlib
import sys

import pytest
import torch
from tests.conftest import _run, _select_args

dnjs = importlib.import_module("scripts.delete_non_jlens_selected")


def _prune(out_dir, env, signal_json, *extra):
    """Run the pruner over `out_dir`, matching the gather script's default select args."""
    traj_root = env["traj_path"].parent.parent  # .../trajectories, holding size5/
    argv = [
        "delete_non_jlens_selected.py",
        "--activations-dir", str(out_dir),
        "--trajectories-dir", str(traj_root),
        "--signal-json", str(signal_json),
        "--select-num-tokens", "2",
        "--select-num-layers", "2",
        "--select-always-layers", "",
        "--select-random-tokens", "1",
        "--select-seed", "42",
        "--verbose",
        *extra,
    ]
    old = sys.argv
    sys.argv = argv
    try:
        dnjs.main()
    finally:
        sys.argv = old


def _pts(root):
    return sorted(p.relative_to(root) for p in root.rglob("*.pt"))


def test_pruning_equals_filtered_gathering(env, signal_json):
    """Prune(full gather) == filtered gather, file for file and byte for byte."""
    full = _run(env, "full")
    filtered = _run(env, "filtered", *_select_args(signal_json))

    _prune(env["tmp"] / "full", env, signal_json, "--apply")

    assert _pts(full) == _pts(filtered)
    assert _pts(full), "the selection cannot legitimately be empty"
    for rel in _pts(full):
        assert torch.equal(
            torch.load(full / rel, weights_only=True),
            torch.load(filtered / rel, weights_only=True),
        ), rel


def test_pruning_writes_the_same_record_as_gathering(env, signal_json):
    from telos_interp.jlens_utils import read_selection_record, record_path

    full = _run(env, "full")
    filtered = _run(env, "filtered", *_select_args(signal_json))
    _prune(env["tmp"] / "full", env, signal_json, "--apply")

    pruned_kept, _ = read_selection_record(record_path(full))
    gathered_kept, _ = read_selection_record(record_path(filtered))
    assert pruned_kept == gathered_kept


def test_the_csv_is_never_touched(env, signal_json):
    full = _run(env, "full")
    csv_path = full / f"{env['stem']}_jlens_analysis.csv"
    before = csv_path.read_bytes()
    _prune(env["tmp"] / "full", env, signal_json, "--apply")
    assert csv_path.read_bytes() == before, "the CSV is the analysis artefact and the done-marker"


def test_dry_run_deletes_nothing(env, signal_json, capsys):
    from telos_interp.jlens_utils import record_path

    full = _run(env, "full")
    before = _pts(full)
    _prune(env["tmp"] / "full", env, signal_json)

    assert _pts(full) == before
    assert not record_path(full).exists(), "a dry run must not leave a record either"
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "would prune" in out


def test_dry_run_reports_what_apply_would_do(env, signal_json, capsys):
    _run(env, "full")
    _prune(env["tmp"] / "full", env, signal_json)
    dry = capsys.readouterr().out

    _prune(env["tmp"] / "full", env, signal_json, "--apply")
    applied = capsys.readouterr().out

    def per_trajectory(text):
        return [line.strip() for line in text.splitlines() if line.startswith("  pruned")]

    assert per_trajectory(dry) == per_trajectory(applied)
    assert per_trajectory(dry), "the dry run has to report something to be useful"


def test_missing_trajectory_json_deletes_nothing(env, signal_json, capsys, tmp_path):
    """Without the JSON there is no output_start, so there is no safe mapping to act on."""
    full = _run(env, "full")
    before = _pts(full)

    empty = tmp_path / "no_trajectories"
    empty.mkdir()
    argv = [
        "delete_non_jlens_selected.py",
        "--activations-dir", str(env["tmp"] / "full"),
        "--trajectories-dir", str(empty),
        "--signal-json", str(signal_json),
        "--apply",
    ]
    old, sys.argv = sys.argv, argv
    try:
        dnjs.main()
    finally:
        sys.argv = old

    assert _pts(full) == before
    assert "no trajectory JSON" in capsys.readouterr().out


def test_missing_selected_files_deletes_nothing(env, signal_json, capsys):
    """A selection pointing at absent files means the mapping is wrong -- bail, don't guess."""
    full = _run(env, "full")
    # pruning twice with a *wider* selection asks for files the first pass removed
    _prune(env["tmp"] / "full", env, signal_json, "--apply")
    after_first = _pts(full)

    _prune(env["tmp"] / "full", env, signal_json, "--apply", "--select-num-tokens", "4")
    assert _pts(full) == after_first
    assert "selected files missing" in capsys.readouterr().out


def test_tolerate_missing_prunes_anyway(env, signal_json):
    full = _run(env, "full")
    _prune(env["tmp"] / "full", env, signal_json, "--apply")
    _prune(env["tmp"] / "full", env, signal_json, "--apply", "--select-num-tokens", "4",
           "--tolerate-missing")
    assert _pts(full), "still a valid subset, just a narrower one than asked for"


def test_pruning_is_idempotent(env, signal_json):
    full = _run(env, "full")
    _prune(env["tmp"] / "full", env, signal_json, "--apply")
    once = _pts(full)
    _prune(env["tmp"] / "full", env, signal_json, "--apply")
    assert _pts(full) == once


def test_empty_directories_are_cleaned_up(env, signal_json):
    full = _run(env, "full")
    _prune(env["tmp"] / "full", env, signal_json, "--apply")
    model = full / "stub__model"
    assert not [d for d in model.rglob("*") if d.is_dir() and not any(d.iterdir())]


def test_size_filter_leaves_other_sizes_alone(env, signal_json):
    """A filter that matches nothing exits rather than silently pruning everything."""
    full = _run(env, "full")
    before = _pts(full)
    with pytest.raises(SystemExit):
        _prune(env["tmp"] / "full", env, signal_json, "--apply", "--sizes", "99")
    assert _pts(full) == before


def test_size_filter_matching_prunes(env, signal_json):
    full = _run(env, "full")
    before = _pts(full)
    _prune(env["tmp"] / "full", env, signal_json, "--apply", "--sizes", "5")
    assert _pts(full) != before


def test_no_csvs_is_an_error(tmp_path, signal_json):
    empty = tmp_path / "nothing"
    empty.mkdir()
    argv = [
        "delete_non_jlens_selected.py",
        "--activations-dir", str(empty),
        "--trajectories-dir", str(tmp_path),
        "--signal-json", str(signal_json),
    ]
    old, sys.argv = sys.argv, argv
    try:
        with pytest.raises(SystemExit):
            dnjs.main()
    finally:
        sys.argv = old
