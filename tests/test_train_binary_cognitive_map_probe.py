"""Tests for the one-vs-rest grid probe: the label collapse and what it saves.

The multiclass trainer decides how many classes it has by looking at the data. A binary arm
must not: `A` and `G` are one cell per grid, so a split that happens to hold no agent would
silently train a 1-wide head and score 100%. The head is pinned at 2 here, and so is the
label map the probe carries, because a consumer binarising ground truth off `label_to_idx`
has to agree with training for symbols the training split never saw.
"""

import json
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from telos_interp.commands.train_binary_cognitive_map_probe.train_binary_cognitive_map_probe_fn import (  # noqa: E402
    NEGATIVE_LABEL,
    BinaryCognitiveMapProbe,
    _binarize_labels,
    _load_and_preprocess_train_data_v3,
    _prepare_train_eval_v3_binary,
    binary_auroc,
    binary_label_maps,
    binary_metrics,
    resolve_positive_class,
    train_binary_cognitive_map_probe,
)
from telos_interp.grid_utils import CELL_SYMBOL_TO_ID  # noqa: E402

CELLS = 6
DIM = 8
# One of each of the four symbols that actually occur in these grids, plus two empties.
LABELS = [1, 3, 3, 0, 2, 3]  # #  _  _  A  G  _


def _dataset(tmp_path, name, *, names, tokens_per_name, labels=None, size=5):
    """A v3 grid_tile manifest referencing its activations in place."""
    acts = tmp_path / "acts"
    acts.mkdir(exist_ok=True)
    entries = []
    cells = {}
    for traj in names:
        key = f"{traj}|0"
        cells[key] = {
            "positions": [[r, c] for r in range(2) for c in range(3)],
            "labels": list(labels if labels is not None else LABELS),
        }
        for token in range(tokens_per_name):
            rel = f"{traj}/{token}.pt"
            path = acts / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(torch.randn(DIM), path)
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
                "activation_dim": DIM,
                "activations_root": str(acts),
                "num_cells_per_trajectory": CELLS,
                "cells": cells,
                "trajectories": entries,
            }
        )
    )
    return prepared


def _prepare(prepared, positive_cell_id=1, **overrides):
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
    return _prepare_train_eval_v3_binary(load_result=load_result, positive_cell_id=positive_cell_id, **kwargs)


# --- the label collapse -------------------------------------------------------------------


def test_resolve_accepts_symbols_and_aliases():
    assert resolve_positive_class("wall") == ("#", 1)
    assert resolve_positive_class("#") == ("#", 1)
    assert resolve_positive_class("EMPTY") == ("_", 3)
    assert resolve_positive_class(" agent ") == ("A", 0)


def test_resolve_refuses_an_unknown_class():
    with pytest.raises(ValueError, match="Unknown positive class"):
        resolve_positive_class("lava")


def test_binarize_marks_exactly_the_positive_id():
    labels = torch.tensor([[0, 1, 2, 3, 7, 1]])
    assert _binarize_labels(labels, 1).tolist() == [[0, 1, 0, 0, 0, 1]]
    assert _binarize_labels(labels, 3).tolist() == [[0, 0, 0, 1, 0, 0]]
    # A symbol absent from the data is still a legal positive class; it just has no rows.
    assert _binarize_labels(labels, 5).sum().item() == 0


def test_label_maps_cover_every_cell_id_and_only_two_indices():
    label_to_idx, idx_to_label = binary_label_maps(CELL_SYMBOL_TO_ID["#"])
    assert set(label_to_idx) == set(CELL_SYMBOL_TO_ID.values())
    assert set(label_to_idx.values()) == {0, 1}
    assert label_to_idx[CELL_SYMBOL_TO_ID["#"]] == 1
    assert idx_to_label == {0: NEGATIVE_LABEL, 1: CELL_SYMBOL_TO_ID["#"]}
    # The negative label must not collide with a real cell id, or a lookup in
    # CELL_ID_TO_SYMBOL would quietly print the wrong symbol instead of failing.
    assert NEGATIVE_LABEL not in CELL_SYMBOL_TO_ID.values()


# --- metrics ------------------------------------------------------------------------------


def test_metrics_match_a_hand_computed_confusion_matrix():
    labels = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0])
    predictions = torch.tensor([1, 1, 0, 1, 0, 0, 0, 0])
    m = binary_metrics(labels, predictions)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (2, 1, 4, 1)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(4 / 5)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert m["balanced_accuracy"] == pytest.approx((2 / 3 + 4 / 5) / 2)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["positive_support"] == 3


def test_a_class_with_no_rows_reports_zero_not_nan():
    """An eval slice can hold no agent at all; recall is 0 there, and the run must go on."""
    labels = torch.zeros(8, dtype=torch.long)
    m = binary_metrics(labels, torch.zeros(8, dtype=torch.long))
    assert m["recall"] == 0.0
    assert m["positive_support"] == 0
    # balanced accuracy averages only over classes that have rows, so it is specificity.
    assert m["balanced_accuracy"] == pytest.approx(1.0)


def test_auroc_matches_the_textbook_value_and_handles_ties():
    assert binary_auroc(torch.tensor([0.1, 0.4, 0.35, 0.8]), torch.tensor([0, 0, 1, 1])) == pytest.approx(0.75)
    # Every score identical: average ranks put this at exactly chance, not at 0 or 1.
    assert binary_auroc(torch.tensor([0.5] * 4), torch.tensor([0, 0, 1, 1])) == pytest.approx(0.5)
    # Perfect separation.
    assert binary_auroc(torch.tensor([0.1, 0.2, 0.8, 0.9]), torch.tensor([0, 0, 1, 1])) == pytest.approx(1.0)


def test_auroc_is_nan_when_a_class_is_absent():
    """Undefined, not chance -- 0.5 would read as a measurement that never happened."""
    value = binary_auroc(torch.tensor([0.1, 0.9]), torch.tensor([0, 0]))
    assert math.isnan(value)


# --- the token-major guard, inherited verbatim --------------------------------------------


def test_internal_split_is_refused_on_a_token_major_manifest(tmp_path):
    prepared = _dataset(tmp_path, "tm", names=[f"t{i}" for i in range(6)], tokens_per_name=3)
    with pytest.raises(ValueError, match="token-major"):
        _prepare(prepared)


def test_subset_is_refused_on_a_token_major_manifest(tmp_path):
    prepared = _dataset(tmp_path, "tm", names=[f"t{i}" for i in range(6)], tokens_per_name=3)
    evaluation = _dataset(tmp_path, "ev", names=[f"e{i}" for i in range(4)], tokens_per_name=3)
    with pytest.raises(ValueError, match="subset"):
        _prepare(prepared, eval_data_path=str(evaluation), subset=0.5)


def test_a_pre_split_pair_is_accepted(tmp_path):
    prepared = _dataset(tmp_path, "tm", names=[f"t{i}" for i in range(6)], tokens_per_name=3)
    evaluation = _dataset(tmp_path, "ev", names=[f"e{i}" for i in range(4)], tokens_per_name=3)
    bundle = _prepare(prepared, eval_data_path=str(evaluation))
    assert len(bundle["train_dataset"]) == 6 * 3 * CELLS
    assert len(bundle["eval_dataset"]) == 4 * 3 * CELLS
    assert bundle["input_dim"] == DIM + 2


def test_the_eval_half_is_binarized_with_the_training_rule(tmp_path):
    """The eval manifest holds raw cell ids; nothing may be masked out as 'unseen'."""
    prepared = _dataset(tmp_path, "tm", names=[f"t{i}" for i in range(4)], tokens_per_name=2)
    evaluation = _dataset(tmp_path, "ev", names=[f"e{i}" for i in range(3)], tokens_per_name=2)
    bundle = _prepare(prepared, positive_cell_id=CELL_SYMBOL_TO_ID["#"], eval_data_path=str(evaluation))

    labels = torch.tensor([bundle["eval_dataset"][i][1] for i in range(len(bundle["eval_dataset"]))])
    assert set(labels.tolist()) == {0, 1}
    # LABELS holds exactly one '#' out of CELLS cells, on every entry.
    assert labels.sum().item() == 3 * 2


def test_normalization_leaves_the_position_columns_alone(tmp_path):
    prepared = _dataset(tmp_path, "tm", names=[f"t{i}" for i in range(4)], tokens_per_name=2)
    evaluation = _dataset(tmp_path, "ev", names=[f"e{i}" for i in range(2)], tokens_per_name=2)
    bundle = _prepare(prepared, eval_data_path=str(evaluation), normalize=True)
    assert bundle["scaler_mean"].shape == (DIM + 2,)
    assert torch.equal(bundle["scaler_mean"][-2:], torch.zeros(2))
    assert torch.equal(bundle["scaler_std"][-2:], torch.ones(2))


# --- what ends up on disk -----------------------------------------------------------------


def test_the_saved_probe_round_trips_and_is_two_wide(tmp_path):
    prepared = _dataset(tmp_path, "tm", names=[f"t{i}" for i in range(4)], tokens_per_name=2)
    evaluation = _dataset(tmp_path, "ev", names=[f"e{i}" for i in range(2)], tokens_per_name=2)
    out = tmp_path / "probe.pt"

    probe = train_binary_cognitive_map_probe(
        train_data_path=str(prepared),
        positive_class="wall",
        eval_data_path=str(evaluation),
        model_type="lr",
        num_epochs=1,
        batch_size=8,
        device="cpu",
        output_path=str(out),
        verbose=False,
    )
    assert probe.num_classes == 2
    assert probe.positive_class == "#"

    loaded = BinaryCognitiveMapProbe.load(out, device="cpu")
    assert loaded.num_classes == 2
    assert loaded.positive_class == "#"
    assert loaded.positive_cell_id == CELL_SYMBOL_TO_ID["#"]
    assert loaded.label_to_idx == probe.label_to_idx
    assert loaded.results["final_balanced_accuracy"] == probe.results["final_balanced_accuracy"]

    rows = torch.randn(5, DIM + 2)
    assert loaded.predict_proba(rows).shape == (5, 2)
    assert set(loaded.predict(rows).tolist()) <= {NEGATIVE_LABEL, CELL_SYMBOL_TO_ID["#"]}


def test_the_head_stays_two_wide_when_the_positive_class_is_absent(tmp_path):
    """A grid with no doors still yields a door probe -- a degenerate one, not a 1-way head."""
    prepared = _dataset(tmp_path, "tm", names=[f"t{i}" for i in range(4)], tokens_per_name=2)
    evaluation = _dataset(tmp_path, "ev", names=[f"e{i}" for i in range(2)], tokens_per_name=2)
    probe = train_binary_cognitive_map_probe(
        train_data_path=str(prepared),
        positive_class="door",
        eval_data_path=str(evaluation),
        model_type="lr",
        num_epochs=1,
        batch_size=8,
        device="cpu",
        output_path=str(tmp_path / "door.pt"),
        verbose=False,
    )
    assert probe.num_classes == 2
    assert probe.results["positive_rate"] == 0.0
    assert probe.results["final_auroc"] != probe.results["final_auroc"]  # NaN


def test_a_multiclass_probe_is_refused(tmp_path):
    path = tmp_path / "multiclass.pt"
    torch.save({"model_type": "lr", "input_dim": DIM + 2, "num_classes": 4}, path)
    with pytest.raises(ValueError, match="multiclass"):
        BinaryCognitiveMapProbe.load(path, device="cpu")
