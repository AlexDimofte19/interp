"""Tests for scripts/inference_oss/truncation_strategies.py.

Two things are load-bearing and neither is visible from the output of a run:

* ``EosStrategy`` must keep cutting exactly where the old code cut, since every rollout on
  disk (and entry 42's whole loudness join) is indexed by those positions. The test pins its
  positions against ``reasoning_eos_positions`` itself.
* the loud strategies must read the mass table in *reasoning_pos* coordinates and hand back
  *output_tokens* coordinates. An off-by-one there does not crash, it silently truncates one
  token early for every trajectory in the run.

The mass table is built by hand rather than gathered: the point is the coordinate mapping
and the ranking, not the lens.
"""

import importlib
import json
import sys
from math import exp, isclose
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ts = importlib.import_module("scripts.inference_oss.truncation_strategies")

NAME = "together_ai_openai_gpt-oss-20b_size5_comp0.0_1"

# (token text, group) for one step. Sentences end at output index 6 and 9; the third
# "sentence" (10, 11) has no punctuation, so it is only closed by the forced end of reasoning.
TOKENS = [
    ("<|channel|>", "template"),
    ("analysis", "template"),
    ("<|message|>", "template"),
    ("We", "analysis"),
    ("Ġneed", "analysis"),
    ("Ġup", "analysis"),
    (".", "analysis"),
    ("ĠThen", "analysis"),
    ("Ġleft", "analysis"),
    (".", "analysis"),
    ("ĠSo", "analysis"),
    ("ĠUP", "analysis"),
    ("<|end|>", "template"),
    ("<|message|>", "template"),
    ("UP", "action"),
]
# reasoning_pos -> L15 log direction mass. Sentence argmaxes are output idx 5, 8 and 11.
MASS = {0: -5.0, 1: -4.0, 2: -1.0, 3: -6.0, 4: -3.0, 5: -2.0, 6: -7.0, 7: -8.0, 8: -0.5}


def _step(step_id: int = 0) -> dict:
    output_tokens = []
    for i, (text, group) in enumerate(TOKENS):
        groups = ["output", group] if group != "action" else ["output", "final", "action"]
        output_tokens.append({"id": i, "token": text, "token_id": 100 + i, "token_groups": groups})
    return {"step_id": step_id, "agent_action": "UP", "output_tokens": output_tokens}


def _trajectory() -> dict:
    return {"prompt": {"prompt_prefix_tokens": [], "prompt_suffix_tokens": []}, "steps": [_step()]}


@pytest.fixture
def lens_root(tmp_path: Path) -> Path:
    """A mass-table tree holding one trajectory, laid out the way the gather writes it."""
    d = tmp_path / "lens" / "size5" / NAME
    d.mkdir(parents=True)
    csv_path = d / f"{NAME}_jlens_direction_mass.csv"
    layers = [14, 15, 16]
    header = ["size", "complexity", "run", "step", "reasoning_pos", "abs_pos", "token"] + [f"L{n}" for n in layers]
    lines = [",".join(header)]
    for rp, v in MASS.items():
        # L14/L16 are deliberately wrong-but-plausible: only L15 may be read.
        lines.append(
            ",".join(["5", "0.0", "1", "0", str(rp), str(700 + rp), TOKENS[rp + 3][0], "-9.0", str(v), "-9.0"])
        )
    csv_path.write_text("\n".join(lines) + "\n")
    (d / f"{NAME}_jlens_direction_mass.csv.meta.json").write_text(
        json.dumps({"signal_json": "/x/direction_tokens_full.json", "direction_classes": "all", "layers": layers})
    )
    return tmp_path / "lens"


def _cuts(strategy) -> list[tuple[int, str]]:
    return [(c.pos, c.kind) for c in strategy.cutoffs(_trajectory(), _step(), NAME)]


def test_eos_strategy_cuts_where_the_old_code_cut():
    step = _step()
    positions = [c.pos for c in ts.EosStrategy().cutoffs(_trajectory(), step, NAME)]
    assert positions == ts.reasoning_eos_positions(step["output_tokens"]) == [2, 6, 9, 11]
    assert _cuts(ts.EosStrategy()) == [
        (2, ts.KIND_NO_REASONING),
        (6, ts.KIND_SENTENCE_END),
        (9, ts.KIND_SENTENCE_END),
        (11, ts.KIND_END_OF_REASONING),
    ]


def test_argmax_per_sentence_picks_the_loudest_token_of_each_sentence(lens_root):
    strategy = ts.build_strategy("jlens_argmax_per_sentence", lens_root=lens_root)
    # 11 is both the third sentence's argmax and the end of reasoning: one cutoff, not two.
    assert _cuts(strategy) == [
        (2, ts.KIND_NO_REASONING),
        (5, ts.KIND_LOUDEST_IN_SENTENCE),
        (8, ts.KIND_LOUDEST_IN_SENTENCE),
        (11, ts.KIND_END_OF_REASONING),
    ]


def test_cutoffs_carry_their_sentence_coordinates_and_mass(lens_root):
    strategy = ts.build_strategy("jlens_argmax_per_sentence", lens_root=lens_root)
    cut = next(c for c in strategy.cutoffs(_trajectory(), _step(), NAME) if c.pos == 5)
    assert (cut.sentence_idx, cut.pos_in_sentence, cut.sentence_len) == (1, 2, 4)
    assert cut.logmass == MASS[2] and isclose(cut.prob_mass, exp(MASS[2]))


def test_top_k_global_ranks_across_the_whole_chain(lens_root):
    strategy = ts.build_strategy("jlens_top_k_global", lens_root=lens_root, top_k=3)
    assert _cuts(strategy) == [
        (2, ts.KIND_NO_REASONING),
        (5, ts.KIND_LOUD_TOP_K),
        (8, ts.KIND_LOUD_TOP_K),
        (11, ts.KIND_END_OF_REASONING),
    ]


def test_top_k_larger_than_the_chain_keeps_every_reasoning_token(lens_root):
    strategy = ts.build_strategy("jlens_top_k_global", lens_root=lens_root, top_k=999)
    assert [c.pos for c in strategy.cutoffs(_trajectory(), _step(), NAME)] == [2] + list(range(3, 12))


def test_endpoints_can_be_dropped(lens_root):
    strategy = ts.build_strategy("jlens_top_k_global", lens_root=lens_root, top_k=3, include_endpoints=False)
    assert _cuts(strategy) == [(5, ts.KIND_LOUD_TOP_K), (8, ts.KIND_LOUD_TOP_K), (11, ts.KIND_LOUD_TOP_K)]


def test_ties_resolve_to_the_earlier_token(lens_root, monkeypatch):
    """Two tokens at the same mass must give the same cutoff whichever strategy asks."""
    flat = dict.fromkeys(MASS, -1.0)
    strategy = ts.build_strategy("jlens_argmax_per_sentence", lens_root=lens_root)
    monkeypatch.setattr(strategy.loudness, "step_scores", lambda *a, **k: {p + 3: v for p, v in flat.items()})
    assert [c.pos for c in strategy.cutoffs(_trajectory(), _step(), NAME)] == [2, 3, 7, 10, 11]


def test_a_mass_table_without_its_sidecar_is_refused(lens_root):
    (lens_root / "size5" / NAME / f"{NAME}_jlens_direction_mass.csv.meta.json").unlink()
    strategy = ts.build_strategy("jlens_argmax_per_sentence", lens_root=lens_root)
    with pytest.raises(ts.LoudnessUnavailable, match="sidecar"):
        strategy.cutoffs(_trajectory(), _step(), NAME)


def test_a_missing_mass_table_is_reported_not_guessed(lens_root):
    strategy = ts.build_strategy("jlens_argmax_per_sentence", lens_root=lens_root)
    with pytest.raises(ts.LoudnessUnavailable, match="no jlens direction-mass table"):
        strategy.cutoffs(_trajectory(), _step(), "together_ai_openai_gpt-oss-20b_size5_comp0.0_999")


def test_a_layer_the_table_does_not_cover_is_reported(lens_root):
    strategy = ts.build_strategy("jlens_argmax_per_sentence", lens_root=lens_root, layer=7)
    with pytest.raises(ts.LoudnessUnavailable, match="no column L7"):
        strategy.cutoffs(_trajectory(), _step(), NAME)


def test_every_token_cuts_at_every_reasoning_token(lens_root):
    """The dense grid: one cutoff per analysis token, plus the two shared endpoints.

    This is the property the held-out loudness join relies on -- the cutoff positions
    must be exactly ``analysis_positions``, so a per-token CSV built from the lens tree
    joins 1:1 against the rollout with nothing dropped on either side.
    """
    strategy = ts.build_strategy("every_token", lens_root=lens_root)
    step = _step()
    positions = [c.pos for c in strategy.cutoffs(_trajectory(), step, NAME)]
    assert positions == [2] + ts.analysis_positions(step["output_tokens"]) == [2] + list(range(3, 12))
    assert _cuts(strategy)[:3] == [
        (2, ts.KIND_NO_REASONING),
        (3, ts.KIND_EVERY_TOKEN),
        (4, ts.KIND_EVERY_TOKEN),
    ]
    # The last reasoning token keeps the endpoint label, not every_token: _dedupe lets the
    # endpoint win so this arm's final eval is the same labelled prompt as every other arm's.
    assert _cuts(strategy)[-1] == (11, ts.KIND_END_OF_REASONING)


def test_every_token_records_loudness_without_ranking_on_it(lens_root):
    strategy = ts.build_strategy("every_token", lens_root=lens_root)
    cuts = {c.pos: c for c in strategy.cutoffs(_trajectory(), _step(), NAME)}
    assert cuts[5].logmass == MASS[2] and isclose(cuts[5].prob_mass, exp(MASS[2]))
    assert (cuts[5].sentence_idx, cuts[5].pos_in_sentence, cuts[5].sentence_len) == (1, 2, 4)
    # MASS covers reasoning_pos 0..8, i.e. output 3..11; nothing is dropped for lacking a score.
    assert cuts[3].logmass == MASS[0]


def test_every_token_survives_a_missing_mass_table(lens_root):
    """Loudness is a covariate here, not a selector, so no table costs the column, not the step."""
    strategy = ts.build_strategy("every_token", lens_root=lens_root)
    other = "together_ai_openai_gpt-oss-20b_size5_comp0.0_999"
    cuts = strategy.cutoffs(_trajectory(), _step(), other)
    assert [c.pos for c in cuts] == [2] + list(range(3, 12))
    assert all(c.logmass is None for c in cuts)


def test_every_token_stride_thins_the_grid_and_keeps_the_endpoints(lens_root):
    strategy = ts.build_strategy("every_token", lens_root=lens_root, stride=2)
    positions = [c.pos for c in strategy.cutoffs(_trajectory(), _step(), NAME)]
    # Strided from the first reasoning token (3), so a stride-2 grid is a subset of the dense one.
    assert positions == [2, 3, 5, 7, 9, 11]
    assert set(positions[1:]) <= set(ts.analysis_positions(_step()["output_tokens"]))
    assert strategy.config()["stride"] == 2


def test_every_token_stride_must_be_positive(lens_root):
    with pytest.raises(ValueError, match="stride must be >= 1"):
        ts.build_strategy("every_token", lens_root=lens_root, stride=0)


def test_every_token_endpoints_can_be_dropped(lens_root):
    strategy = ts.build_strategy("every_token", lens_root=lens_root, include_endpoints=False)
    assert _cuts(strategy) == [(p, ts.KIND_EVERY_TOKEN) for p in range(3, 12)]


def test_unknown_strategy_names_are_rejected():
    with pytest.raises(ValueError, match="unknown strategy"):
        ts.build_strategy("loudest")
