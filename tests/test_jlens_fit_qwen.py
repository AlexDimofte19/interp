"""Tests for the Qwen Jacobian-lens fit.

The load-bearing one is ``test_windowed_matches_full_graph``: the fit runs the instruction prefix
and the early chain under ``no_grad`` into a cache and only puts the window in the autograd graph,
which is worth ~9x in backward cost and graph memory but is only legitimate if it computes the
same ``J``. That is asserted here against a full-graph reference, on a randomly-initialised tiny
model, on CPU, with no download.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "jlens"))

jlens_fit_qwen = pytest.importorskip("jlens_fit_qwen", reason="needs the jlens package installed")
torch = pytest.importorskip("torch")

from jlens_fit_qwen import (  # noqa: E402
    FULL_ATTENTION_SOURCE_LAYERS,
    WindowTooShort,
    build_windowed_corpus,
    jacobian_for_windowed_prompt,
    resolve_window,
)

# --------------------------------------------------------------------------
# fixtures: a synthetic trajectory tree in the replayed-Qwen shape
# --------------------------------------------------------------------------


def _trajectory(grid_size: int, output_n: int) -> dict:
    """A trajectory JSON with only the fields the corpus builder reads."""
    grid = [f"row{r}" for r in range(grid_size)]
    return {
        "prompt": {
            "prompt_template": "INSTRUCTIONS {{grid_state}} SUFFIX",
            "prompt_prefix_n_tokens": 371,
            "prompt_suffix_n_tokens": 7,
        },
        "steps": [
            {
                "grid_state": grid,
                "grid_state_n_tokens": 47,
                "output_text": "REASONING",
                "output_n_tokens": output_n,
            }
        ],
    }


@pytest.fixture
def trajectory_tree(tmp_path: Path) -> Path:
    """Two grid sizes x two complexities x three files, named as the replay writes them."""
    root = tmp_path / "mass_train_576"
    for size in (5, 7):
        directory = root / f"size{size}"
        directory.mkdir(parents=True)
        for comp in ("0.0", "0.2"):
            for index in range(3):
                path = directory / f"together_ai_gsarti_qwen3_6-35b_size{size}_comp{comp}_{index}.json"
                # size 5 gets one short chain per stratum, so the skip path is exercised.
                output_n = 500 if (size == 5 and index == 0) else 8000
                path.write_text(json.dumps(_trajectory(size, output_n)), encoding="utf-8")
    return root


# --------------------------------------------------------------------------
# window arithmetic
# --------------------------------------------------------------------------


def test_window_ends_at_the_requested_fraction():
    trajectory = _trajectory(7, output_n=8000)
    reasoning_start, total_len, end = resolve_window(
        trajectory, trajectory["steps"][0], frac=0.5, window_size=1024, max_window_end=None
    )
    assert reasoning_start == 371 + 47 + 7
    assert total_len == reasoning_start + 8000
    assert end == reasoning_start + 4000
    assert end - 1024 >= reasoning_start  # the window never reaches back into the grid


def test_window_frac_zero_is_the_prefix_anchored_ablation():
    trajectory = _trajectory(7, output_n=8000)
    reasoning_start, _, end = resolve_window(
        trajectory, trajectory["steps"][0], frac=0.0, window_size=1024, max_window_end=None
    )
    assert end == reasoning_start + 1024


def test_max_window_end_caps_the_prefill():
    trajectory = _trajectory(7, output_n=30000)
    reasoning_start, _, end = resolve_window(
        trajectory, trajectory["steps"][0], frac=0.5, window_size=1024, max_window_end=4096
    )
    assert end == reasoning_start + 4096


def test_short_chain_is_refused_rather_than_half_windowed():
    trajectory = _trajectory(7, output_n=1500)  # 0.5 * 1500 = 750 < 1024
    with pytest.raises(WindowTooShort):
        resolve_window(trajectory, trajectory["steps"][0], frac=0.5, window_size=1024, max_window_end=None)


# --------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------


def test_corpus_is_balanced_and_skips_short_chains(trajectory_tree: Path):
    prompts, histogram, skipped = build_windowed_corpus(
        trajectory_tree, n_prompts=8, seed=0, sizes=None, frac=0.5, window_size=1024, max_window_end=None
    )
    assert len(prompts) == 8
    # Four strata, two of which lost one file to the short-chain filter; the round-robin still
    # spreads what is left evenly rather than draining one stratum first.
    assert set(histogram) == {(5, 0.0), (5, 0.2), (7, 0.0), (7, 0.2)}
    assert max(histogram.values()) - min(histogram.values()) <= 1
    assert sum(skipped.values()) == 2
    assert all(p.start >= p.reasoning_start for p in prompts)
    assert all(p.end <= p.total_len for p in prompts)


def test_corpus_honours_a_size_filter(trajectory_tree: Path):
    prompts, histogram, _ = build_windowed_corpus(
        trajectory_tree, n_prompts=4, seed=0, sizes=[7], frac=0.5, window_size=1024, max_window_end=None
    )
    assert {p.grid_size for p in prompts} == {7}
    assert set(histogram) == {(7, 0.0), (7, 0.2)}


def test_frac_achieved_reports_where_the_window_landed(trajectory_tree: Path):
    prompts, _, _ = build_windowed_corpus(
        trajectory_tree, n_prompts=4, seed=0, sizes=[7], frac=0.5, window_size=1024, max_window_end=None
    )
    assert all(math.isclose(p.frac_achieved, 0.5) for p in prompts)


# --------------------------------------------------------------------------
# the estimator
# --------------------------------------------------------------------------


def _tiny_text_model(architecture: str):
    """A randomly-initialised tiny decoder, or ``None`` if this transformers lacks it.

    ``qwen3_5_moe`` is the architecture actually being fitted -- 30 gated-DeltaNet layers whose
    cache is a recurrent state rather than a KV pair -- so it is the one that proves the prefill
    trick holds for a hybrid stack. ``qwen3`` is the plain-attention control and is present in
    every transformers the repo has used.
    """
    import transformers

    if architecture == "qwen3":
        config = transformers.Qwen3Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            tie_word_embeddings=False,
        )
        return transformers.Qwen3ForCausalLM(config).model

    config_class = getattr(transformers, "Qwen3_5MoeTextConfig", None)
    model_class = getattr(transformers, "Qwen3_5MoeTextModel", None)
    if config_class is None or model_class is None:
        return None
    config = config_class(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=4,
        layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        tie_word_embeddings=False,
    )
    return model_class(config)


class _StubLensModel:
    """The three members ``jacobian_for_windowed_prompt`` needs, over a bare text decoder.

    Mirrors what ``jlens.from_hf`` builds, without needing a ``*ForCausalLM`` wrapper or a
    tokenizer: parameters frozen so the recorded residual is the only graph leaf.
    """

    def __init__(self, text_module) -> None:
        text_module.eval()
        for param in text_module.parameters():
            param.requires_grad_(False)
        self._text_module = text_module
        self.layers = text_module.layers
        self.d_model = text_module.config.hidden_size
        self.n_layers = text_module.config.num_hidden_layers

    @property
    def input_device(self):
        return self._text_module.embed_tokens.weight.device


def _reference_jacobian(model, input_ids, *, window_start, source_layers, target_layer, dim_batch):
    """``J`` the way jlens computes it: the whole sequence in one graph, no cache.

    A transcription of ``jlens.fitting.jacobian_for_prompt`` taking ids instead of text, so the
    comparison does not depend on the tokenizer or on jlens being installed.
    """
    from jlens.hooks import ActivationRecorder

    d_model = model.d_model
    ids = input_ids.unsqueeze(0).expand(dim_batch, -1)
    seq_len = ids.shape[1]
    jacobians = {layer: torch.zeros(d_model, d_model, dtype=torch.float32) for layer in source_layers}
    n_passes = math.ceil(d_model / dim_batch)

    with (
        ActivationRecorder(
            model.layers, at=[*source_layers, target_layer], start_graph_at=min(source_layers)
        ) as recorder,
        torch.enable_grad(),
    ):
        model._text_module(input_ids=ids, use_cache=False)
        target_activation = recorder.activations[target_layer]
        source_activations = [recorder.activations[layer] for layer in source_layers]

        valid_positions = torch.arange(window_start, seq_len - 1)
        batch_indices = torch.arange(dim_batch)
        cotangent = torch.zeros_like(target_activation)
        for pass_idx, dim_start in enumerate(range(0, d_model, dim_batch)):
            n_dims = min(dim_batch, d_model - dim_start)
            cotangent.zero_()
            cotangent[
                batch_indices[:n_dims, None], valid_positions[None, :], dim_start + batch_indices[:n_dims, None]
            ] = 1.0
            grads = torch.autograd.grad(
                outputs=target_activation,
                inputs=source_activations,
                grad_outputs=cotangent,
                retain_graph=(pass_idx < n_passes - 1),
            )
            for layer, grad in zip(source_layers, grads, strict=True):
                rows = grad[:n_dims, valid_positions, :].float().mean(dim=1)
                jacobians[layer][dim_start : dim_start + n_dims, :] = rows
    return jacobians


@pytest.mark.parametrize("architecture", ["qwen3", "qwen3_5_moe"])
def test_windowed_matches_full_graph(architecture):
    """Prefilling the prefix instead of retaining it must not change a single row of J.

    This is what makes the window a cost saving rather than an approximation. By causality no path
    from a window position to a later window position leaves the window, so the prefix is a
    constant -- and if that reasoning were wrong for the DeltaNet's recurrent state or the
    convolution's lookback, this test is where it shows.
    """
    pytest.importorskip("jlens", reason="the estimator reuses jlens.hooks.ActivationRecorder")
    text_module = _tiny_text_model(architecture)
    if text_module is None:
        pytest.skip(f"this transformers has no {architecture}")

    torch.manual_seed(0)
    model = _StubLensModel(text_module)
    ids = torch.randint(0, 64, (48,))
    window_start, source_layers, target_layer = 32, [1], 3

    windowed, seq_len, n_valid = jacobian_for_windowed_prompt(
        model,
        ids,
        window_start=window_start,
        source_layers=source_layers,
        target_layer=target_layer,
        dim_batch=2,
    )
    reference = _reference_jacobian(
        model,
        ids,
        window_start=window_start,
        source_layers=source_layers,
        target_layer=target_layer,
        dim_batch=2,
    )

    assert seq_len == ids.shape[0]
    assert n_valid == ids.shape[0] - window_start - 1  # the last position has no next-token target
    for layer in source_layers:
        assert torch.allclose(windowed[layer], reference[layer], atol=1e-5, rtol=1e-4), (
            f"layer {layer}: max |diff| = {(windowed[layer] - reference[layer]).abs().max().item():.3e}"
        )


def test_window_of_one_position_is_refused():
    pytest.importorskip("jlens")
    text_module = _tiny_text_model("qwen3")
    model = _StubLensModel(text_module)
    with pytest.raises(ValueError, match="no valid positions"):
        jacobian_for_windowed_prompt(
            model, torch.randint(0, 64, (8,)), window_start=7, source_layers=[1], target_layer=3
        )


# --------------------------------------------------------------------------


def test_default_source_layers_are_full_attention_and_below_the_target():
    # config.text_config.layer_types puts a full_attention layer at every index == 3 (mod 4);
    # 39 is the tenth and is the target, so the nine below it are the sources.
    assert FULL_ATTENTION_SOURCE_LAYERS == [index for index in range(40) if index % 4 == 3 and index != 39]
