"""Tests for telos_interp/jlens_utils/commitment.py.

The load-bearing one is `test_label_covers_every_cohort_with_one_expression`: reasoning tokens
always land in `sentence_idx >= 1`, and that is the only reason `convinced_label` can handle a
never-convinced chain, an already-convinced-at-0 chain and an ordinary boundary without
branching. If sentence 0 ever owned a reasoning token, the `immediate` cohort would silently
start producing 0s.

The rest pin the index-space arithmetic that `scripts/build_probe_rollout_join.py` gets wrong by
hardcoding 3: the gap between a mass table's `reasoning_pos` and a rollout's `eos_token_pos` is
`eos[0] + 1`, read per trajectory.
"""

import pytest
from telos_interp.jlens_utils import (
    COHORT_IMMEDIATE,
    COHORT_NEVER,
    COHORT_NORMAL,
    cohort,
    convinced_label,
    place_token,
    reasoning_offset,
    rollout_path,
    sentence_of_token,
)

# Sentence 0 is the <|channel|>analysis<|message|> header, ending at output token 2.
EOS = [2, 5, 8, 9]


def test_sentence_spans_are_inclusive_of_their_eos():
    span = sentence_of_token(EOS)
    assert span == {3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3}
    # The header's own tokens belong to no reasoning sentence.
    assert 0 not in span and 1 not in span and 2 not in span


def test_reasoning_offset_is_read_not_assumed():
    assert reasoning_offset(EOS) == 3
    assert reasoning_offset([5, 9]) == 6  # a longer header shifts every reasoning_pos


@pytest.mark.parametrize(
    ("convinced", "sentence", "expected"),
    [
        (2, 1, 0),  # before the boundary
        (2, 2, 1),  # the convinced sentence itself is convinced
        (2, 3, 1),  # and everything after it
        (0, 1, 1),  # committed before writing anything -> every token positive
        (0, 3, 1),
        (None, 1, 0),  # never convinced -> every token negative
        (None, 3, 0),
    ],
)
def test_label_covers_every_cohort_with_one_expression(convinced, sentence, expected):
    assert convinced_label(convinced, sentence) == expected


def test_cohorts_name_the_three_regimes():
    assert cohort(None) == COHORT_NEVER
    assert cohort(0) == COHORT_IMMEDIATE
    assert cohort(1) == COHORT_NORMAL
    assert cohort(7) == COHORT_NORMAL


def test_immediate_cohort_is_all_positive_and_never_cohort_all_negative():
    span = sentence_of_token(EOS)
    for conv, expected in ((0, 1), (None, 0)):
        labels = {convinced_label(conv, si) for si in span.values()}
        assert labels == {expected}


def test_placement_coordinates():
    span = sentence_of_token(EOS)
    first = place_token(3, EOS, span, 2)
    assert (first.sentence_idx, first.pos_in_sentence, first.sentence_len) == (1, 0, 3)
    assert first.sentence_frac == 0.0 and first.is_sentence_end == 0
    # rel 0 IS the convinced sentence, so it occupies x in [-1, 0].
    assert first.rel_sentence == -1 and first.x_sentence == -2.0

    last = place_token(5, EOS, span, 2)
    assert last.sentence_frac == 1.0 and last.is_sentence_end == 1

    convinced_eos = place_token(8, EOS, span, 2)
    assert convinced_eos.rel_sentence == 0 and convinced_eos.x_sentence == 0.0


def test_single_token_sentence_has_no_interior_to_interpolate():
    span = sentence_of_token(EOS)
    only = place_token(9, EOS, span, 2)
    assert only.sentence_len == 1 and only.sentence_frac == 1.0 and only.is_sentence_end == 1


def test_never_convinced_leaves_relative_coordinates_undefined():
    span = sentence_of_token(EOS)
    place = place_token(3, EOS, span, None)
    assert place.rel_sentence is None and place.x_sentence is None
    assert place.sentence_idx == 1  # placement still works; only the boundary is unknown


def test_token_outside_every_sentence_is_none():
    span = sentence_of_token(EOS)
    assert place_token(99, EOS, span, 2) is None
    assert place_token(0, EOS, span, 2) is None  # a header token


def test_rollout_path_reads_size_from_the_stem():
    got = rollout_path("/p", "together_ai_openai_gpt-oss-20b_size11_comp0.0_1")
    assert got.as_posix() == "/p/size11/together_ai_openai_gpt-oss-20b_size11_comp0.0_1.json"
