"""Tests for the two things train_cognitive_map_probe had to learn for token-major data.

A grid_tile manifest used to be one entry per trajectory, which made two shortcuts safe:
`torch.randperm` over entries *was* a split by trajectory, and loading meant one file read
per trajectory. Neither holds once a trajectory owns ~20 selected tokens that all share its
grid -- the split leaks and the load turns into tens of thousands of small reads. The guard
and the cache are what replace them, so both are pinned here.
"""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from telos_interp.commands.train_cognitive_map_probe.train_cognitive_map_probe_fn import (
    _load_and_preprocess_train_data_v3,
    _prepare_train_eval_v3,
)

CELLS = 6
DIM = 4


def _dataset(tmp_path, name, *, names, tokens_per_name):
    """A v3 grid_tile manifest referencing its activations in place.

    `tokens_per_name=1` is the classic shape (one entry per trajectory); anything more is
    token-major, the shape a lens selection produces.
    """
    acts = tmp_path / "acts"
    acts.mkdir(exist_ok=True)
    entries = []
    cells = {}
    for n_i, traj in enumerate(names):
        key = f"{traj}|0"
        cells[key] = {
            "positions": [[r, c] for r in range(2) for c in range(3)],
            "labels": [(n_i + i) % 3 for i in range(CELLS)],
        }
        for token in range(tokens_per_name):
            rel = f"{traj}/{token}.pt"
            path = acts / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(torch.full((DIM,), float(n_i * 10 + token)), path)
            entries.append(
                {
                    "name": traj,
                    "act_path": rel,
                    "layer": 15,
                    "step": 0,
                    "token_id": token,
                    "category": "output",
                    "cells_key": key,
                }
            )

    prepared = tmp_path / name
    prepared.mkdir()
    (prepared / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": 3,
                "probe_type": "grid_tile",
                "activation_dim": DIM,
                "activations_root": str(acts),
                "num_cells_per_trajectory": CELLS,
                "cells": cells,
                "trajectories": entries,
            }
        )
    )
    return prepared


def _prepare(prepared, **overrides):
    kwargs = {
        "eval_data_path": None,
        "eval_split": 0.5,
        "subset": 1.0,
        "balance_classes": False,
        "normalize": False,
        "per_class_max_count": None,
        "seed": 42,
        "verbose": False,
    }
    kwargs.update(overrides)
    load_result = _load_and_preprocess_train_data_v3(Path(prepared), verbose=False)
    return _prepare_train_eval_v3(load_result=load_result, **kwargs)


# --- the split guard ----------------------------------------------------------------------


def test_internal_split_is_refused_on_a_token_major_manifest(tmp_path):
    """randperm over entries would put the same trajectory in both halves."""
    prepared = _dataset(tmp_path, "tm", names=[f"t{i}" for i in range(6)], tokens_per_name=3)
    with pytest.raises(ValueError, match="token-major"):
        _prepare(prepared)


def test_subset_is_refused_on_a_token_major_manifest(tmp_path):
    """--subset drops individual tokens there, not whole trajectories."""
    prepared = _dataset(tmp_path, "tm", names=[f"t{i}" for i in range(6)], tokens_per_name=3)
    evaluation = _dataset(tmp_path, "ev", names=[f"e{i}" for i in range(4)], tokens_per_name=3)
    with pytest.raises(ValueError, match="subset"):
        _prepare(prepared, eval_data_path=str(evaluation), subset=0.5)


def test_a_pre_split_pair_is_accepted(tmp_path):
    prepared = _dataset(tmp_path, "tm", names=[f"t{i}" for i in range(6)], tokens_per_name=3)
    evaluation = _dataset(tmp_path, "ev", names=[f"e{i}" for i in range(4)], tokens_per_name=3)
    bundle = _prepare(prepared, eval_data_path=str(evaluation))
    assert len(bundle["train_dataset"]) == 6 * 3 * CELLS
    assert bundle["input_dim"] == DIM + 2


def test_the_classic_shape_still_splits_internally(tmp_path):
    """One entry per trajectory: the pre-existing behaviour must be untouched."""
    prepared = _dataset(tmp_path, "classic", names=[f"t{i}" for i in range(8)], tokens_per_name=1)
    bundle = _prepare(prepared)
    assert len(bundle["train_dataset"]) + len(bundle["eval_dataset"]) == 8 * CELLS


# --- the packed-activation cache ----------------------------------------------------------


def _compact(prepared, use_cache):
    return _load_and_preprocess_train_data_v3(Path(prepared), verbose=False, use_cache=use_cache)["compact"]


def test_cache_returns_identical_tensors(tmp_path):
    """A cached run and an uncached one must train on the same data, or the cache is a bug."""
    prepared = _dataset(tmp_path, "tm", names=[f"t{i}" for i in range(5)], tokens_per_name=3)
    plain = _compact(prepared, use_cache=False)
    built = _compact(prepared, use_cache=True)  # writes it
    assert (prepared / "_packed_activations.pt").exists()
    read = _compact(prepared, use_cache=True)  # reads it back

    for cached in (built, read):
        assert torch.equal(cached["base_act"], plain["base_act"])
        assert torch.equal(cached["labels"], plain["labels"])
        assert torch.equal(cached["positions"], plain["positions"])
        assert cached["trajectory_names"] == plain["trajectory_names"]


def test_cache_rebuilds_when_the_manifest_changes(tmp_path):
    prepared = _dataset(tmp_path, "tm", names=[f"t{i}" for i in range(5)], tokens_per_name=3)
    _compact(prepared, use_cache=True)

    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["trajectories"] = manifest["trajectories"][:6]
    manifest_path.write_text(json.dumps(manifest))

    assert _compact(prepared, use_cache=True)["base_act"].shape[0] == 6


def test_an_unreadable_cache_does_not_fail_the_run(tmp_path):
    prepared = _dataset(tmp_path, "tm", names=[f"t{i}" for i in range(5)], tokens_per_name=3)
    plain = _compact(prepared, use_cache=False)
    (prepared / "_packed_activations.pt").write_bytes(b"not a torch file")

    rebuilt = _compact(prepared, use_cache=True)
    assert torch.equal(rebuilt["base_act"], plain["base_act"])
