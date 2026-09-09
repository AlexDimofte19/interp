"""Tests for --keep-kinds in scripts/inference_oss/relabel_manifest_from_rollout.py.

The flag exists because of one asymmetry: the arms built by
``gather_local_belief_activations.py`` write a tree holding exactly the positions they want,
but an arm built on a PRE-EXISTING tree does not get that. The sentence-end activation tree
carries every sentence end of every trajectory, and the last one's cutoff is
``end_of_reasoning`` -- a row whose local-belief label is the final action at p~1.0 by
construction. Keeping those would add one trivially decodable row per trajectory and break
comparability with the interior-only arms, so the eos arm filters them out here.

The default must stay "keep every kind", or every manifest relabelled before this flag
existed stops reproducing.
"""

import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

rl = importlib.import_module("scripts.inference_oss.relabel_manifest_from_rollout")

NAME = "together_ai_openai_gpt-oss-20b_size5_comp0.0_1"

# One trajectory, one step: two interior sentence ends and the end-of-reasoning bookend.
# The bookend agrees with the final action, as it always does; the interior ones do not.
EVALS = [
    {"eos_token_pos": 6, "cutoff_kind": "sentence_end", "model_action": "LEFT", "answer_prob": 0.5, "correct": False},
    {"eos_token_pos": 9, "cutoff_kind": "sentence_end", "model_action": "DOWN", "answer_prob": 0.6, "correct": False},
    {
        "eos_token_pos": 11,
        "cutoff_kind": "end_of_reasoning",
        "model_action": "UP",
        "answer_prob": 1.0,
        "correct": True,
    },
]


def _fixtures(tmp_path: Path) -> tuple[Path, Path]:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": 3,
                "probe_type": "next_action",
                "activation_dim": 2880,
                "activations_root": str(tmp_path / "acts"),
                "samples": [
                    {"name": NAME, "step": 0, "token_id": ev["eos_token_pos"], "label": 1, "act_path": "x.pt"}
                    for ev in EVALS
                ],
            }
        )
    )
    rollout = tmp_path / "rollout"
    rollout.mkdir()
    (rollout / f"{NAME}.json").write_text(json.dumps({"steps": [{"step_id": 0, "sentence_evals": EVALS}]}))
    return prepared, rollout


def _run(tmp_path: Path, extra: list[str]) -> dict:
    prepared, rollout = _fixtures(tmp_path)
    out = tmp_path / "out"
    sys.argv = ["relabel", str(prepared), str(rollout), str(out), *extra]
    assert rl.main() == 0
    return json.loads((out / "manifest.json").read_text())


def test_the_default_keeps_every_kind(tmp_path):
    m = _run(tmp_path, [])
    assert [s["cutoff_kind"] for s in m["samples"]] == ["sentence_end", "sentence_end", "end_of_reasoning"]
    assert m["label_source"]["keep_kinds"] is None
    assert m["label_source"]["n_dropped_wrong_kind"] == 0


def test_keep_kinds_drops_the_end_of_reasoning_row(tmp_path):
    m = _run(tmp_path, ["--keep-kinds", "sentence_end"])
    assert [s["token_id"] for s in m["samples"]] == [6, 9]
    assert all(s["cutoff_kind"] == "sentence_end" for s in m["samples"])
    # The dropped row is the one whose belief label was the final action by construction.
    assert m["label_source"]["n_dropped_wrong_kind"] == 1
    assert m["label_source"]["keep_kinds"] == ["sentence_end"]
    assert m["label_source"]["n_out"] == 2


def test_keep_kinds_accepts_several_kinds(tmp_path):
    m = _run(tmp_path, ["--keep-kinds", "sentence_end", "end_of_reasoning"])
    assert m["label_source"]["n_dropped_wrong_kind"] == 0
    assert m["label_source"]["n_out"] == 3


def test_a_kind_that_matches_nothing_drops_everything_rather_than_passing_it_through(tmp_path):
    m = _run(tmp_path, ["--keep-kinds", "loudest_in_sentence"])
    assert m["samples"] == []
    assert m["label_source"]["n_dropped_wrong_kind"] == 3


def test_the_kept_rows_still_carry_the_local_belief_label(tmp_path):
    """The filter must not disturb the relabel itself: label becomes the rollout's action."""
    m = _run(tmp_path, ["--keep-kinds", "sentence_end"])
    assert [s["label"] for s in m["samples"]] == [0, 3]  # LEFT, DOWN
    assert all(s["final_label"] == 1 for s in m["samples"])  # UP, the trajectory's own action
