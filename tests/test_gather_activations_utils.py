"""Tests for the activation writer and the batched extractor.

CPU only — `extract_activations_batched` is exercised against a stub module list rather
than a real model, so the index arithmetic (which is the part that can silently misalign
tokens) is checked without needing a GPU or gpt-oss-20b.
"""

from pathlib import Path

import pytest
import torch
from telos_interp.commands.gather_activations.gather_activations_utils import (
    ActivationWriter,
    activation_layer_dir,
    extract_activations_batched,
    save_activations_to_files,
)


def _activations(layers, tokens, dim=4):
    """layer -> token -> deterministic (dim,) tensor, so files can be identified."""
    return {
        layer: {token: torch.full((dim,), float(layer * 100 + token)) for token in tokens}
        for layer in layers
    }


def test_activation_layer_dir_with_and_without_step():
    base = Path("/base")
    assert activation_layer_dir(base, 7, 3, "output") == base / "layer_7" / "step_3" / "output"
    assert activation_layer_dir(base, 7, None, "output") == base / "layer_7" / "output"


@pytest.mark.parametrize("workers", [0, 4])
def test_writer_matches_save_activations_to_files(tmp_path, workers):
    """The threaded writer must produce the identical tree, file for file, byte for byte."""
    activations = _activations([7, 8], [0, 1, 2])

    reference = tmp_path / "reference"
    save_activations_to_files(activations, reference, step_idx=3, category="output")

    threaded = tmp_path / "threaded"
    with ActivationWriter(max_workers=workers) as writer:
        writer.submit(activations, threaded, step_idx=3, category="output")

    ref_files = sorted(p.relative_to(reference) for p in reference.rglob("*.pt"))
    new_files = sorted(p.relative_to(threaded) for p in threaded.rglob("*.pt"))
    assert ref_files == new_files
    assert ref_files  # guard against both trees being empty
    for rel in ref_files:
        expected = torch.load(reference / rel, map_location="cpu", weights_only=True)
        actual = torch.load(threaded / rel, map_location="cpu", weights_only=True)
        assert torch.equal(expected, actual)


def test_writer_counts_nans_once_per_tensor(tmp_path):
    activations = _activations([7], [0, 1, 2])
    activations[7][1] = torch.tensor([float("nan"), 0.0, 0.0, 0.0])
    writer = ActivationWriter(max_workers=4)
    writer.submit(activations, tmp_path, step_idx=0, category="output")
    assert writer.close() == 1


def test_drain_resets_the_nan_count(tmp_path):
    """Counts are per-drain, so a caller can report them per trajectory."""
    writer = ActivationWriter(max_workers=2)
    writer.submit({7: {0: torch.tensor([float("nan")])}}, tmp_path, step_idx=0, category="output")
    assert writer.drain() == 1
    writer.submit({7: {1: torch.tensor([1.0])}}, tmp_path, step_idx=0, category="output")
    assert writer.close() == 0


def test_drain_reraises_worker_failures(tmp_path):
    """A failed write must never be mistaken for a completed trajectory."""
    writer = ActivationWriter(max_workers=2)
    # a directory where the .pt file should go makes torch.save fail in the worker
    layer_dir = activation_layer_dir(tmp_path, 7, 0, "output")
    layer_dir.mkdir(parents=True)
    (layer_dir / "0.pt").mkdir()
    writer.submit({7: {0: torch.tensor([1.0])}}, tmp_path, step_idx=0, category="output")
    with pytest.raises((OSError, RuntimeError)):
        writer.close()


def test_throttle_keeps_the_queue_bounded(tmp_path):
    """Queued writes hold their tensors alive, so submit() must block rather than buffer."""
    writer = ActivationWriter(max_workers=2, max_pending=8)
    writer.submit(_activations([7], list(range(50))), tmp_path, step_idx=0, category="output")
    assert len(writer._pending) <= 8
    assert writer.close() == 0
    assert len(list((tmp_path / "layer_7" / "step_0" / "output").glob("*.pt"))) == 50


def test_legacy_container_round_trips(tmp_path):
    """--pt-format legacy must still load under weights_only=True."""
    with ActivationWriter(max_workers=0, use_zipfile=False) as writer:
        writer.submit({7: {0: torch.arange(4.0)}}, tmp_path, step_idx=0, category="output")
    loaded = torch.load(tmp_path / "layer_7" / "step_0" / "output" / "0.pt", weights_only=True)
    assert torch.equal(loaded, torch.arange(4.0))


class _RecordingBlock(torch.nn.Module):
    """Stands in for a decoder block: emits a value that encodes (batch, position)."""

    def __init__(self, layer_idx, dim=3, as_tuple=False):
        super().__init__()
        self.layer_idx = layer_idx
        self.dim = dim
        self.as_tuple = as_tuple

    def forward(self, hidden):
        batch, seq = hidden.shape
        pos = torch.arange(seq).unsqueeze(0).expand(batch, seq)
        row = torch.arange(batch).unsqueeze(1).expand(batch, seq)
        # unique per (layer, batch row, position) so a misaligned gather cannot pass
        value = (self.layer_idx * 10_000 + row * 100 + pos).float()
        out = value.unsqueeze(-1).expand(batch, seq, self.dim).clone()
        return (out,) if self.as_tuple else out


class _StubModel(torch.nn.Module):
    def __init__(self, n_layers=3, as_tuple=False):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            _RecordingBlock(i, as_tuple=as_tuple) for i in range(n_layers)
        )
        # _get_decoder_layers looks for model.layers, and needs at least one parameter
        # to resolve the device
        self.marker = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        hidden = input_ids
        for block in self.layers:
            out = block(hidden)
            hidden = (out[0] if isinstance(out, tuple) else out)[:, :, 0]
        return hidden


class _Wrapper(torch.nn.Module):
    """Mirrors the real `model.model.layers` nesting."""

    def __init__(self, inner):
        super().__init__()
        self.model = inner

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


@pytest.mark.parametrize("as_tuple", [False, True])
def test_extract_activations_batched_gathers_the_right_rows(as_tuple):
    model = _Wrapper(_StubModel(n_layers=3, as_tuple=as_tuple))
    input_ids = torch.zeros((2, 5), dtype=torch.long)
    token_indices = [[1, 4], [0, 2, 3]]

    out = extract_activations_batched(model, input_ids, None, token_indices, [0, 2])

    assert sorted(out) == [0, 2]
    for layer, block in out.items():
        # rows follow token_indices in order: sequence 0's positions, then sequence 1's
        expected = torch.tensor(
            [layer * 10_000 + 0 * 100 + 1, layer * 10_000 + 0 * 100 + 4]
            + [layer * 10_000 + 1 * 100 + p for p in (0, 2, 3)],
            dtype=torch.float,
        )
        assert block.shape == (5, 3)
        assert torch.equal(block[:, 0], expected)


def test_extract_activations_batched_drops_out_of_range_positions():
    model = _Wrapper(_StubModel(n_layers=2))
    out = extract_activations_batched(model, torch.zeros((1, 3), dtype=torch.long), None, [[0, 9]], [1])
    assert out[1].shape[0] == 1


def test_extract_activations_batched_matches_single_pass_row_for_row():
    """The stacked result must agree with the per-token dict the older helper returns."""
    from telos_interp.commands.gather_activations.gather_activations_utils import (
        extract_activations_single_pass,
    )

    model = _Wrapper(_StubModel(n_layers=3))
    input_ids = torch.zeros((1, 6), dtype=torch.long)
    positions = [0, 3, 5]

    single = extract_activations_single_pass(model, input_ids, positions, [0, 1, 2])
    batched = extract_activations_batched(model, input_ids, None, [positions], [0, 1, 2])

    for layer in (0, 1, 2):
        stacked = torch.stack([single[layer][p] for p in positions])
        assert torch.equal(stacked, batched[layer])
