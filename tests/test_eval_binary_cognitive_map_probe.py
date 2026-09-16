"""Tests for scoring a saved binary grid probe against a prepared dataset.

The point of this command is that it reads the SAME artifact the trainer reads, so the two
cannot drift. The load-bearing test is exactly that: train, then score the probe on its own
eval half and demand the numbers match the trainer's final block. Everything else here
guards a mismatch the caller would otherwise not see -- a multiclass probe, a dataset built
from a different layer, a tree instead of a manifest.
"""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from telos_interp.commands.eval_binary_cognitive_map_probe.eval_binary_cognitive_map_probe_fn import (  # noqa: E402
    _complexity_of,
    eval_binary_cognitive_map_probe,
)
from telos_interp.commands.train_binary_cognitive_map_probe import (  # noqa: E402
    train_binary_cognitive_map_probe,
)

CELLS = 6
DIM = 8
LABELS = [1, 3, 3, 0, 2, 3]  # #  _  _  A  G  _


def _dataset(tmp_path, name, *, names, tokens_per_name, dim=DIM, size=5):
    """A v3 grid_tile manifest; `names` carry the size/complexity the breakdowns parse."""
    acts = tmp_path / f"acts_{name}"
    acts.mkdir(exist_ok=True)
    entries = []
    cells = {}
    for traj in names:
        key = f"{traj}|0"
        cells[key] = {"positions": [[r, c] for r in range(2) for c in range(3)], "labels": list(LABELS)}
        for token in range(tokens_per_name):
            rel = f"{traj}/{token}.pt"
            path = acts / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(torch.randn(dim), path)
            entries.append(
                {
                    "name": traj,
                    "act_path": rel,
                    "layer": 15,
                    "step": 0,
                    "token_id": token,
                    "category": "output",
                    "size": size,
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
                "activation_dim": dim,
                "activations_root": str(acts),
                "num_cells_per_trajectory": CELLS,
                "cells": cells,
                "trajectories": entries,
            }
        )
    )
    return prepared


def _trained(tmp_path, train, evaluation, positive_class="wall"):
    return train_binary_cognitive_map_probe(
        train_data_path=str(train),
        positive_class=positive_class,
        eval_data_path=str(evaluation),
        model_type="lr",
        num_epochs=2,
        batch_size=16,
        device="cpu",
        output_path=str(tmp_path / f"probe_{positive_class}.pt"),
        verbose=False,
    )


def test_complexity_is_parsed_out_of_the_trajectory_name():
    assert _complexity_of("together_ai_openai_gpt-oss-20b_size9_comp0.4_123") == 0.4
    assert _complexity_of("x_comp0_7") == 0.0
    assert _complexity_of("nothing_here") is None


def test_it_reproduces_the_trainers_own_final_numbers(tmp_path):
    """Same probe, same rows -- if these disagree the two paths have drifted apart."""
    train = _dataset(tmp_path, "tr", names=[f"a_size5_comp0.0_{i}" for i in range(8)], tokens_per_name=3)
    evaluation = _dataset(tmp_path, "ev", names=[f"b_size5_comp0.2_{i}" for i in range(4)], tokens_per_name=3)
    probe = _trained(tmp_path, train, evaluation)

    results = eval_binary_cognitive_map_probe(
        probe_path=str(tmp_path / "probe_wall.pt"),
        data_path=str(evaluation),
        output_path=str(tmp_path / "out.json"),
        device="cpu",
        verbose=False,
    )
    block = results["global"]
    assert block["accuracy"] == pytest.approx(probe.results["final_accuracy"])
    assert block["balanced_accuracy"] == pytest.approx(probe.results["final_balanced_accuracy"])
    assert block["recall"] == pytest.approx(probe.results["final_recall"])
    assert block["precision"] == pytest.approx(probe.results["final_precision"])
    assert block["auroc"] == pytest.approx(probe.results["final_auroc"], nan_ok=True)
    assert block["tp"] == probe.results["confusion"]["tp"]


def test_it_counts_every_token_and_cell(tmp_path):
    train = _dataset(tmp_path, "tr", names=[f"a_size5_comp0.0_{i}" for i in range(4)], tokens_per_name=2)
    held = _dataset(tmp_path, "held", names=[f"h_size5_comp0.4_{i}" for i in range(5)], tokens_per_name=7)
    _trained(tmp_path, train, held)

    results = eval_binary_cognitive_map_probe(
        probe_path=str(tmp_path / "probe_wall.pt"),
        data_path=str(held),
        output_path=str(tmp_path / "held.json"),
        device="cpu",
        verbose=False,
    )
    assert results["data"]["n_entries"] == 5 * 7
    assert results["data"]["n_trajectories"] == 5
    assert results["data"]["n_rows"] == 5 * 7 * CELLS
    assert results["global"]["n_rows"] == 5 * 7 * CELLS
    # Exactly one '#' per entry's cell list.
    assert results["global"]["positive_support"] == 5 * 7


def test_the_breakdowns_partition_the_rows(tmp_path):
    train = _dataset(tmp_path, "tr", names=[f"a_size5_comp0.0_{i}" for i in range(4)], tokens_per_name=2)
    mixed = _dataset(tmp_path, "mx", names=["b_size5_comp0.0_1", "b_size5_comp0.2_2"], tokens_per_name=3)
    _trained(tmp_path, train, mixed)

    results = eval_binary_cognitive_map_probe(
        probe_path=str(tmp_path / "probe_wall.pt"),
        data_path=str(mixed),
        output_path=str(tmp_path / "mixed.json"),
        device="cpu",
        verbose=False,
    )
    total = results["global"]["n_rows"]
    assert sum(b["n_rows"] for b in results["by_size"].values()) == total
    assert sum(b["n_rows"] for b in results["by_complexity"].values()) == total
    assert sorted(results["by_complexity"]) == ["0.0", "0.2"]
    assert sorted(results["by_size_complexity"]) == ["size5|comp0.0", "size5|comp0.2"]


def test_the_results_json_is_written_and_names_its_inputs(tmp_path):
    train = _dataset(tmp_path, "tr", names=[f"a_size5_comp0.0_{i}" for i in range(4)], tokens_per_name=2)
    evaluation = _dataset(tmp_path, "ev", names=[f"b_size5_comp0.2_{i}" for i in range(2)], tokens_per_name=2)
    _trained(tmp_path, train, evaluation, positive_class="empty")

    out = tmp_path / "nested" / "res.json"
    eval_binary_cognitive_map_probe(
        probe_path=str(tmp_path / "probe_empty.pt"),
        data_path=str(evaluation),
        output_path=str(out),
        device="cpu",
        verbose=False,
    )
    written = json.loads(out.read_text())
    assert written["probe"]["positive_class"] == "_"
    assert written["data"]["path"] == str(evaluation.resolve())
    assert written["config"]["threshold"] == 0.5


def test_a_dataset_from_a_different_layer_is_refused(tmp_path):
    """A probe reads one layer's residual; a manifest built from another has a different D."""
    train = _dataset(tmp_path, "tr", names=[f"a_size5_comp0.0_{i}" for i in range(4)], tokens_per_name=2)
    evaluation = _dataset(tmp_path, "ev", names=[f"b_size5_comp0.2_{i}" for i in range(2)], tokens_per_name=2)
    _trained(tmp_path, train, evaluation)

    wider = _dataset(tmp_path, "wide", names=["c_size5_comp0.0_1"], tokens_per_name=2, dim=DIM * 2)
    with pytest.raises(ValueError, match="different layers"):
        eval_binary_cognitive_map_probe(
            probe_path=str(tmp_path / "probe_wall.pt"),
            data_path=str(wider),
            output_path=str(tmp_path / "wide.json"),
            device="cpu",
            verbose=False,
        )


def test_a_gathered_tree_is_refused(tmp_path):
    train = _dataset(tmp_path, "tr", names=[f"a_size5_comp0.0_{i}" for i in range(4)], tokens_per_name=2)
    evaluation = _dataset(tmp_path, "ev", names=[f"b_size5_comp0.2_{i}" for i in range(2)], tokens_per_name=2)
    _trained(tmp_path, train, evaluation)

    tree = tmp_path / "tree"
    (tree / "size5").mkdir(parents=True)
    with pytest.raises(ValueError, match="v3 prepared dataset directory"):
        eval_binary_cognitive_map_probe(
            probe_path=str(tmp_path / "probe_wall.pt"),
            data_path=str(tree),
            device="cpu",
            verbose=False,
        )


def test_the_threshold_moves_the_operating_point(tmp_path):
    train = _dataset(tmp_path, "tr", names=[f"a_size5_comp0.0_{i}" for i in range(6)], tokens_per_name=3)
    evaluation = _dataset(tmp_path, "ev", names=[f"b_size5_comp0.2_{i}" for i in range(4)], tokens_per_name=3)
    _trained(tmp_path, train, evaluation)

    def at(threshold):
        return eval_binary_cognitive_map_probe(
            probe_path=str(tmp_path / "probe_wall.pt"),
            data_path=str(evaluation),
            output_path=str(tmp_path / f"t{threshold}.json"),
            threshold=threshold,
            device="cpu",
            verbose=False,
        )["global"]

    # A threshold of 0 calls everything positive, 1 calls nothing positive; AUROC is a
    # property of the scores and must not move between them.
    low, high = at(0.0), at(1.0)
    assert low["predicted_positive"] == low["n_rows"]
    assert high["predicted_positive"] == 0
    assert low["auroc"] == pytest.approx(high["auroc"], nan_ok=True)
