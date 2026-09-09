"""Tests for scripts/inference_oss/gather_local_belief_activations.py's position filter.

The filter decides which of a rollout's cutoffs get a ``.pt`` saved, and it used to be a
module constant. Now it is ``--interior-kinds``, so the same script serves the
per-sentence-loudest arm, the sentence-end arm and the random-in-sentence control. Two
things are worth pinning:

* the default must resolve to exactly the old ``{"loudest_in_sentence"}``, or the tree that
  entry 45 published stops reproducing;
* ``--include-endpoints`` must UNION rather than replace, since the endpoint kinds are
  shared by every strategy while the interior kind is what distinguishes them.

``positions_by_step`` itself had no coverage at all before this file.
"""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("torch")
g = importlib.import_module("scripts.inference_oss.gather_local_belief_activations")


def test_the_default_is_the_old_hard_coded_kind():
    assert tuple(g.DEFAULT_INTERIOR_KINDS) == ("loudest_in_sentence",)
    assert g.resolve_kinds(g.DEFAULT_INTERIOR_KINDS, False) == {"loudest_in_sentence"}


def test_include_endpoints_adds_rather_than_replaces():
    assert g.resolve_kinds(["sentence_end"], True) == {"sentence_end", "no_reasoning", "end_of_reasoning"}


def test_asking_for_an_endpoint_kind_twice_is_idempotent():
    assert g.resolve_kinds(["end_of_reasoning"], True) == {"no_reasoning", "end_of_reasoning"}


def test_several_interior_kinds_can_be_gathered_at_once():
    assert g.resolve_kinds(["sentence_end", "random_in_sentence"], False) == {"sentence_end", "random_in_sentence"}


def _rollout_doc() -> dict:
    """Two steps, three kinds mixed, plus one entry with no position at all."""
    return {
        "steps": [
            {
                "step_id": 0,
                "sentence_evals": [
                    {"eos_token_pos": 2, "cutoff_kind": "no_reasoning"},
                    {"eos_token_pos": 9, "cutoff_kind": "random_in_sentence"},
                    {"eos_token_pos": 5, "cutoff_kind": "random_in_sentence"},
                    {"eos_token_pos": 6, "cutoff_kind": "sentence_end"},
                    {"eos_token_pos": 11, "cutoff_kind": "end_of_reasoning"},
                ],
            },
            {
                "step_id": 1,
                "sentence_evals": [
                    {"eos_token_pos": 4, "cutoff_kind": "sentence_end"},
                    {"eos_token_pos": None, "cutoff_kind": "sentence_end"},
                    {"eos_token_pos": 7, "cutoff_kind": "loudest_in_sentence"},
                ],
            },
        ]
    }


def test_positions_are_filtered_by_kind_and_returned_sorted():
    assert g.positions_by_step(_rollout_doc(), {"random_in_sentence"}) == {0: [5, 9]}


def test_positions_span_several_kinds_and_several_steps():
    got = g.positions_by_step(_rollout_doc(), {"sentence_end", "random_in_sentence"})
    assert got == {0: [5, 6, 9], 1: [4]}


def test_a_step_with_no_matching_kind_is_absent_rather_than_empty():
    """The caller iterates the returned dict, so an empty step must not appear at all."""
    assert g.positions_by_step(_rollout_doc(), {"loudest_in_sentence"}) == {1: [7]}


def test_a_null_position_is_dropped_not_kept_as_none():
    assert g.positions_by_step(_rollout_doc(), {"sentence_end"}) == {0: [6], 1: [4]}


def test_an_unknown_kind_matches_nothing():
    assert g.positions_by_step(_rollout_doc(), {"loud_top_k"}) == {}
