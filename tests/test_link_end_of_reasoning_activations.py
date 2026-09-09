"""Tests for scripts/link_end_of_reasoning_activations.py.

The script repairs a silent data loss: a per-sentence arm whose pick lands on the final
reasoning token has that pick merged into the `end_of_reasoning` bookend by `_dedupe`, and
the sentence then leaves the dataset entirely because p1 datasets drop that kind. The rate is
a property of the rule (30/3600 for a jlens argmax, 776 for a uniform draw, all 3600 for
`eos`), so arms meant to differ only in *which token inside a span* they cut ended up
differing in *how many spans* they hold.

What has to hold, and what is pinned here:

  * only a step whose FINAL sentence lost its interior pick is repaired -- repairing a
    sentence that still has one would double-count it,
  * a step whose final sentence kept a distinct interior pick is left alone,
  * an interior pick landing on an EARLIER sentence's last token is not a collision and must
    not be touched.
"""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

link = importlib.import_module("scripts.link_end_of_reasoning_activations")


def ev(cut_sentence_idx, kind, pos, ordinal=None):
    """One rollout eval. `cut_sentence_idx` is the sentence the cut lands in; `sentence_idx`
    is only the cutoff's ordinal in the list, and is set to a DIFFERENT value on purpose so a
    reader of the wrong field fails these tests instead of passing them."""
    return {
        "cut_sentence_idx": cut_sentence_idx,
        "sentence_idx": ordinal if ordinal is not None else 99,
        "cutoff_kind": kind,
        "eos_token_pos": pos,
    }


def step(step_id, evals):
    return {"step_id": step_id, "sentence_evals": evals}


def test_collided_final_sentence_is_repaired():
    """The pick merged into the bookend, so the sentence has only an end_of_reasoning row."""
    doc = {
        "steps": [
            step(
                0,
                [
                    ev(None, "no_reasoning", 0),
                    ev(1, "loudest_in_sentence", 4),
                    ev(2, "end_of_reasoning", 9),
                ],
            )
        ]
    }
    assert link.steps_missing_final_sentence(doc) == {0: 9}


def test_intact_final_sentence_is_left_alone():
    """A distinct interior pick in the last sentence means nothing was lost."""
    doc = {
        "steps": [
            step(
                0,
                [
                    ev(1, "loudest_in_sentence", 4),
                    ev(2, "loudest_in_sentence", 7),
                    ev(2, "end_of_reasoning", 9),
                ],
            )
        ]
    }
    assert link.steps_missing_final_sentence(doc) == {}


def test_pick_on_an_earlier_sentences_last_token_is_not_a_collision():
    """Only the LAST sentence's end coincides with end_of_reasoning; sentence 1's does not.

    Measured on the real data, 825 jlens picks in 400 trajectories landed on an earlier
    sentence's final token and all 825 stayed interior cutoffs.
    """
    doc = {
        "steps": [
            step(
                0,
                [
                    ev(1, "loudest_in_sentence", 4),  # sentence 1 ENDS at 4: still interior
                    ev(2, "loudest_in_sentence", 7),  # sentence 2 keeps its own pick
                    ev(2, "end_of_reasoning", 9),  # bookend shares sentence 2 with it
                ],
            )
        ]
    }
    assert link.steps_missing_final_sentence(doc) == {}


def test_eos_arm_loses_its_final_sentence_in_every_step():
    """`eos` cuts at each sentence's last token, so the last sentence ALWAYS collides."""
    doc = {
        "steps": [
            step(0, [ev(1, "sentence_end", 4), ev(2, "sentence_end", 6), ev(3, "end_of_reasoning", 9)]),
            step(1, [ev(1, "sentence_end", 3), ev(2, "end_of_reasoning", 8)]),
        ]
    }
    assert link.steps_missing_final_sentence(doc) == {0: 9, 1: 8}


def test_at_most_one_row_per_step_is_added():
    """The repair cannot double-count: one final sentence, one position."""
    doc = {"steps": [step(0, [ev(1, "loudest_in_sentence", 2), ev(2, "end_of_reasoning", 5)])]}
    assert len(link.steps_missing_final_sentence(doc)) == 1


def test_rows_without_a_sentence_index_are_ignored():
    """`no_reasoning` sits at eos[0], outside every span, so its cut_sentence_idx is None."""
    doc = {"steps": [step(0, [ev(None, "no_reasoning", 0, ordinal=0)])]}
    assert link.steps_missing_final_sentence(doc) == {}


def test_step_dir_puts_size_above_the_trajectory_name():
    """A path built without the size level silently matches nothing rather than raising."""
    got = link.step_dir(Path("/tree"), "11", "traj_size11_x", "openai__gpt-oss-20b", 3)
    assert got == Path("/tree/size11/traj_size11_x/openai__gpt-oss-20b/layer_15/step_3/output")
