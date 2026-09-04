"""Tests for the PER-TOKEN commitment boundary in scripts/build_probe_loudness_heldout.py.

Every figure that reads "the model becomes convinced here" is downstream of one rule, and
until entry 50 that rule only ever ran over sentence ENDS. The rule itself is unchanged --
first cutoff from which every later truncated answer is correct -- but the grid it runs on
is now every reasoning token, and the difference is not cosmetic:

* a commitment that lands mid-sentence used to be rounded UP to the next sentence end;
* a relapse between two sentence ends was invisible, so the boundary came out TOO EARLY.

Both directions are pinned below, because neither shows up as a crash and both move the
cohort that entries 41/42 drop.
"""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

bl = importlib.import_module("scripts.build_probe_loudness_heldout")


def ev(pos, correct, kind="every_token"):
    return {"eos_token_pos": pos, "correct": correct, "cutoff_kind": kind}


def no_reasoning(correct):
    return ev(-999, correct, kind=bl.KIND_NO_REASONING)


def test_convinced_index_matches_run_inference_rule():
    assert bl.convinced_index([True, True, True]) == 0
    assert bl.convinced_index([False, False, True]) == 2
    assert bl.convinced_index([True, False, True, True]) == 2
    assert bl.convinced_index([True, True, False]) is None
    assert bl.convinced_index([]) is None


def test_boundary_is_a_token_position_not_a_sentence_end():
    """Correct from token 5 on: the boundary is 5, not the sentence end that follows it."""
    evals = [no_reasoning(False)] + [ev(p, p >= 5) for p in range(3, 10)]
    assert bl.convinced_token(evals) == 5


def test_sentinel_when_committed_before_writing_anything():
    evals = [no_reasoning(True)] + [ev(p, True) for p in range(3, 8)]
    assert bl.convinced_token(evals) == bl.NO_REASONING_POS


def test_no_boundary_when_the_full_chain_answers_wrong():
    evals = [no_reasoning(True)] + [ev(p, p < 7) for p in range(3, 8)]
    assert bl.convinced_token(evals) is None


def test_a_relapse_between_sentence_ends_pushes_the_boundary_later():
    """The failure the sentence-end grid cannot see.

    Sentence ends at 4 and 9 are both correct, so the old rule calls the step convinced at
    4. Token 6 is wrong, so the honest boundary is 7.
    """
    sentence_ends = {4, 9}
    corrects = {3: True, 4: True, 5: True, 6: False, 7: True, 8: True, 9: True}
    evals = [no_reasoning(False)] + [ev(p, corrects[p]) for p in sorted(corrects)]

    assert bl.convinced_token(evals) == 7
    ends_only = [e for e in evals if e["eos_token_pos"] in sentence_ends or e["cutoff_kind"] == bl.KIND_NO_REASONING]
    assert bl.convinced_token(ends_only) == 4


def test_evals_out_of_order_are_sorted_by_position():
    corrects = {3: False, 4: True, 5: True}
    evals = [ev(p, corrects[p]) for p in (5, 3, 4)] + [no_reasoning(False)]
    assert bl.convinced_token(evals) == 4


def test_ct_sentence_maps_the_boundary_to_its_sentence():
    toks = {3: {"sentence_idx": "0"}, 4: {"sentence_idx": "0"}, 5: {"sentence_idx": "1"}}
    assert bl.ct_sentence(5, toks) == 1
    assert bl.ct_sentence(bl.NO_REASONING_POS, toks) == -1
