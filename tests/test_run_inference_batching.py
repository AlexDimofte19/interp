"""Tests for the batch-size caps in scripts/inference_oss/run_inference.py.

These exist because of a real crash: an arm of the truncation-strategy rollout died 165
trajectories in with ``torch.OutOfMemoryError: Tried to allocate 4.80 GiB`` on a batch of
16 rows x 1587 tokens. That batch was INSIDE its padded-area budget (16 * 1587 = 25,392,
cap 49,152) -- the area cap is linear in the sequence length, and gpt-oss falls back to
eager attention, which materializes a ``[rows, heads, L, L]`` score tensor. The bound that
matters is therefore quadratic, and ``--max-attn-elems`` is it.

The arithmetic each test pins: ``rows * 64 heads * L**2 * 2`` bytes at bf16, with two or
three such tensors live at once inside ``eager_attention_forward``.
"""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("torch")
ri = importlib.import_module("scripts.inference_oss.run_inference")

DEFAULT_ATTN_ELEMS = 16_000_000
DEFAULT_AREA = 49_152
HEADS = 64
BF16 = 2


def batches_for(lengths, batch_size=16, area=DEFAULT_AREA, attn=DEFAULT_ATTN_ELEMS):
    """Batch a list of prompt lengths; return (rows, padded_len) per batch."""
    prompts = [[0] * n for n in lengths]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out = ri._length_sorted_batches(order, prompts, batch_size, area, attn)
    return [(len(b), max(len(prompts[i]) for i in b)) for b in out], out


def attn_gib(rows, length):
    return rows * HEADS * length * length * BF16 / 1024**3


def test_the_batch_that_crashed_the_run_is_no_longer_produced():
    shapes, _ = batches_for([1587] * 32)
    assert (16, 1587) not in shapes
    assert attn_gib(16, 1587) == pytest.approx(4.80, abs=0.02)  # what it asked for
    for rows, length in shapes:
        assert rows * length * length <= DEFAULT_ATTN_ELEMS
        assert attn_gib(rows, length) < 2.0


def test_every_batch_respects_all_three_caps():
    lengths = [120, 480, 500, 725, 900, 1000, 1125, 1600, 2400] * 6
    shapes, _ = batches_for(lengths)
    for rows, length in shapes:
        assert rows <= 16
        assert rows * length <= DEFAULT_AREA
        assert rows * length * length <= DEFAULT_ATTN_ELEMS


@pytest.mark.parametrize("length", [200, 500, 725, 1000])
def test_short_prompts_keep_the_full_row_count(length):
    """The cap must not cost throughput where memory was never the problem.

    16 x 1000 is exactly the default budget, so everything at or below the size-11 prompt
    lengths the run spends most of its time on batches at full width.
    """
    shapes, _ = batches_for([length] * 48)
    assert shapes[0] == (16, length)


def test_zero_disables_the_cap():
    shapes, _ = batches_for([1587] * 32, attn=0)
    assert shapes[0] == (16, 1587)


def test_a_single_oversized_prompt_still_runs_alone():
    """Work is never dropped: one prompt over the whole budget forms its own one-row batch."""
    shapes, _ = batches_for([9000])
    assert shapes == [(1, 9000)]


def test_no_prompt_is_dropped_or_duplicated():
    lengths = [137, 802, 1591, 244, 1000, 4096, 613, 1587, 55]
    _, groups = batches_for(lengths)
    flat = [i for g in groups for i in g]
    assert sorted(flat) == list(range(len(lengths)))


def test_area_cap_still_binds_on_its_own():
    """A long prompt whose quadratic term is fine can still be area-limited, and vice versa."""
    shapes, _ = batches_for([4000] * 24, attn=0)
    assert shapes[0] == (12, 4000)  # 12 * 4000 = 48,000 <= 49,152; a 13th would exceed it
