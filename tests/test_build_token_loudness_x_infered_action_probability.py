"""Tests for telos_interp/loudness_analysis/join_rollout_answers.py.

The join is arithmetically trivial and silently wrong in exactly one way: the mass tables are
indexed by ``reasoning_pos``, which counts only the ANALYSIS-tagged output tokens, while the
rollout is indexed by the position in ``output_tokens``. The two differ by
``min(analysis_positions(...))``. Nothing crashes if that offset is wrong -- every row simply
describes a neighbouring token -- so the fixture here deliberately uses an offset of **4**
rather than the 3 that real harmony output happens to produce. A hardcoded 3 passes on the
real data and fails here.

The rest of the tests pin the guards, since a guard that never fires is indistinguishable
from a guard that cannot fire.
"""

import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

build = importlib.import_module("telos_interp.loudness_analysis.join_rollout_answers")

NAME = "together_ai_openai_gpt-oss-20b_size5_comp0.0_1"

# FOUR template tokens, so the analysis region starts at 4 and reasoning_pos != token_idx.
TOKENS = [
    ("<|start|>", "template"),
    ("<|channel|>", "template"),
    ("analysis", "template"),
    ("<|message|>", "template"),
    ("We", "analysis"),
    ("Ġneed", "analysis"),
    ("Ġup", "analysis"),
    (".", "analysis"),
    ("ĠSo", "analysis"),
    ("ĠUP", "analysis"),
    ("<|end|>", "template"),
    ("<|message|>", "template"),
    ("UP", "action"),
]
OFFSET = 4
N_REASONING = 6

# reasoning_pos -> L15 log direction mass, deliberately different per lens.
JLENS_MASS = {0: -5.0, 1: -4.0, 2: -1.0, 3: -6.0, 4: -3.0, 5: -0.5}
LOGITLENS_MASS = {0: -5.5, 1: -2.25, 2: -1.75, 3: -6.5, 4: -3.5, 5: -0.25}
LAYERS = [14, 15, 16]

# The action the model gives when cut at each reasoning token, and its probability.
ACTIONS = ["LEFT", "LEFT", "UP", "UP", "UP", "UP"]
PROBS = [0.31, 0.42, 0.55, 0.61, 0.77, 0.92]


def _step(step_id: int = 0) -> dict:
    output_tokens = []
    for i, (text, group) in enumerate(TOKENS):
        groups = ["output", group] if group != "action" else ["output", "final", "action"]
        output_tokens.append({"id": i, "token": text, "token_id": 100 + i, "token_groups": groups})
    return {"step_id": step_id, "agent_action": "UP", "output_tokens": output_tokens}


def _trajectory() -> dict:
    return {"prompt": {"prompt_prefix_tokens": [], "prompt_suffix_tokens": []}, "steps": [_step()]}


def _mass_csv(path: Path, mass: dict[int, float], *, token_shift: int = 0) -> None:
    """Write one lens's direction-mass table; ``token_shift`` corrupts only the token column."""
    header = ["size", "complexity", "run", "step", "reasoning_pos", "abs_pos", "token", "agent_action"]
    header += [f"L{n}" for n in LAYERS]
    lines = [",".join(header)]
    for rp, value in mass.items():
        token = TOKENS[rp + OFFSET + token_shift][0]
        # L14/L16 are wrong-but-plausible: only the requested layer may ever be read.
        row = ["5", "0.0", "1", "0", str(rp), str(700 + rp), token, "UP", "-9.0", str(value), "-9.0"]
        lines.append(",".join(row))
    path.write_text("\n".join(lines) + "\n")


def _sidecar(path: Path, lens: str, *, num_tokens: int = 446, layers: list[int] | None = None) -> None:
    path.with_suffix(path.suffix + ".meta.json").write_text(
        json.dumps(
            {
                "signal_json": "/x/direction_tokens_full.json",
                "direction_classes": "all",
                "num_direction_tokens": num_tokens,
                "lens": lens,
                "layers": LAYERS if layers is None else layers,
            }
        )
    )


def _rollout(*, dir_logmass: dict[int, float] | None = None) -> dict:
    """The every_token arm's result file: one eval per reasoning token, plus no_reasoning."""
    mass = JLENS_MASS if dir_logmass is None else dir_logmass
    evals = [
        {
            "sentence_idx": 0,
            "eos_token_pos": OFFSET - 1,
            "cutoff_kind": "no_reasoning",
            "dir_logmass": None,
            "n_prompt_tokens": 500,
            "model_action": "LEFT",
            "correct": False,
            "answer_token": "LEFT",
            "answer_prob": 0.2,
        }
    ]
    for rp in range(N_REASONING):
        evals.append(
            {
                "sentence_idx": rp + 1,
                "eos_token_pos": rp + OFFSET,
                "cutoff_kind": "end_of_reasoning" if rp == N_REASONING - 1 else "every_token",
                "dir_logmass": mass.get(rp),
                "n_prompt_tokens": 500 + rp,
                "model_action": ACTIONS[rp],
                "correct": ACTIONS[rp] == "UP",
                "answer_token": ACTIONS[rp],
                "answer_prob": PROBS[rp],
            }
        )
    return {
        "strategy": {"strategy": "every_token", "stride": 1, "lens": "jlens", "layer": 15},
        "steps": [{"step_id": 0, "ground_truth": "UP", "sentence_evals": evals}],
    }


@pytest.fixture
def tree(tmp_path: Path):
    """The three artifacts the join reads, laid out the way each producer writes them."""

    def build_tree(*, token_shift: int = 0, rollout: dict | None = None, logit_tokens: int = 446) -> dict:
        lens_dir = tmp_path / "lens" / "size5" / NAME
        lens_dir.mkdir(parents=True, exist_ok=True)
        for lens, mass, n_tokens in (
            ("jlens", JLENS_MASS, 446),
            ("logitlens", LOGITLENS_MASS, logit_tokens),
        ):
            path = lens_dir / f"{NAME}_{lens}_direction_mass.csv"
            _mass_csv(path, mass, token_shift=token_shift if lens == "jlens" else 0)
            _sidecar(path, lens, num_tokens=n_tokens)

        traj_dir = tmp_path / "trajectories" / "size5"
        traj_dir.mkdir(parents=True, exist_ok=True)
        (traj_dir / f"{NAME}.json").write_text(json.dumps(_trajectory()))

        roll_dir = tmp_path / "rollouts"
        roll_dir.mkdir(parents=True, exist_ok=True)
        (roll_dir / f"{NAME}.json").write_text(json.dumps(rollout or _rollout()))

        names = tmp_path / "names.txt"
        names.write_text(f"{NAME}\n")
        return {
            "lens_root": tmp_path / "lens",
            "trajectories": tmp_path / "trajectories",
            "rollout_dir": roll_dir,
            "names": names,
            "out": tmp_path / "out.csv",
        }

    return build_tree


def _run(paths: dict, monkeypatch, *extra: str) -> int:
    argv = [
        "build_token_loudness_x_infered_action_probability.py",
        "--rollout-dir",
        str(paths["rollout_dir"]),
        "--trajectories",
        str(paths["trajectories"]),
        "--lens-root",
        str(paths["lens_root"]),
        "--names-file",
        str(paths["names"]),
        "--out",
        str(paths["out"]),
        *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    return build.main()


def _rows(paths: dict) -> list[dict]:
    import csv

    with open(paths["out"], encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_one_row_per_reasoning_token_through_a_non_three_offset(tree, monkeypatch):
    paths = tree()
    assert _run(paths, monkeypatch) == 0
    rows = _rows(paths)
    assert len(rows) == N_REASONING
    for rp, row in enumerate(rows):
        assert int(row["reasoning_pos"]) == rp
        # The whole point: token_idx is reasoning_pos + 4 here, not + 3.
        assert int(row["token_idx"]) == rp + OFFSET
        assert row["token"] == TOKENS[rp + OFFSET][0]
        assert int(row["abs_pos"]) == 700 + rp


def test_both_lenses_land_in_their_own_columns(tree, monkeypatch):
    paths = tree()
    _run(paths, monkeypatch)
    for rp, row in enumerate(_rows(paths)):
        assert float(row["jlens_logmass_L15"]) == JLENS_MASS[rp]
        assert float(row["logitlens_logmass_L15"]) == LOGITLENS_MASS[rp]
        # exp of the log, so a plot can use either without redoing the arithmetic.
        assert float(row["jlens_prob_L15"]) == pytest.approx(2.718281828459045 ** JLENS_MASS[rp])


def test_the_truncated_answer_travels_with_the_token(tree, monkeypatch):
    paths = tree()
    _run(paths, monkeypatch)
    rows = _rows(paths)
    assert [r["model_action"] for r in rows] == ACTIONS
    assert [float(r["answer_prob"]) for r in rows] == PROBS
    assert [int(r["correct"]) for r in rows] == [int(a == "UP") for a in ACTIONS]
    assert {r["ground_truth"] for r in rows} == {"UP"}
    assert rows[-1]["cutoff_kind"] == "end_of_reasoning"


def test_layer_travels_in_the_column_name(tree, monkeypatch):
    paths = tree()
    _run(paths, monkeypatch, "--layer", "14")
    row = _rows(paths)[0]
    assert "jlens_logmass_L14" in row and "jlens_logmass_L15" not in row
    assert float(row["jlens_logmass_L14"]) == -9.0


def test_no_reasoning_cutoff_is_opt_in_and_carries_no_loudness(tree, monkeypatch):
    paths = tree()
    _run(paths, monkeypatch, "--include-no-reasoning")
    rows = _rows(paths)
    assert len(rows) == N_REASONING + 1
    sentinel = rows[0]
    assert int(sentinel["reasoning_pos"]) == -1
    assert int(sentinel["token_idx"]) == OFFSET - 1
    assert sentinel["cutoff_kind"] == "no_reasoning"
    assert sentinel["jlens_logmass_L15"] == "" and sentinel["logitlens_logmass_L15"] == ""
    assert float(sentinel["answer_prob"]) == 0.2


def test_a_shifted_mass_table_is_caught_by_the_token_column(tree, monkeypatch):
    """The offset trap: shift only the token column and nothing else looks wrong."""
    paths = tree(token_shift=1)
    with pytest.raises(SystemExit, match="reasoning_pos -> token_idx mapping is off"):
        _run(paths, monkeypatch)


def test_a_loudness_the_rollout_disagrees_with_is_caught(tree, monkeypatch):
    corrupted = {rp: value - 1.0 for rp, value in JLENS_MASS.items()}
    paths = tree(rollout=_rollout(dir_logmass=corrupted))
    with pytest.raises(SystemExit, match="indexed differently"):
        _run(paths, monkeypatch)


def test_lenses_scored_against_different_vocabularies_are_refused(tree, monkeypatch):
    paths = tree(logit_tokens=999)
    with pytest.raises(SystemExit, match="different vocabularies"):
        _run(paths, monkeypatch)


def test_a_layer_the_tables_do_not_cover_is_refused(tree, monkeypatch):
    paths = tree()
    with pytest.raises(SystemExit, match="does not cover layer 21"):
        _run(paths, monkeypatch, "--layer", "21")


def test_a_missing_rollout_names_the_command_that_builds_it(tree, monkeypatch):
    paths = tree()
    (paths["rollout_dir"] / f"{NAME}.json").unlink()
    with pytest.raises(SystemExit, match="run_inference_strategies.sh every_token"):
        _run(paths, monkeypatch)


def test_a_thinned_rollout_is_warned_about(tree, monkeypatch, capsys):
    rollout = _rollout()
    rollout["strategy"]["stride"] = 2
    paths = tree(rollout=rollout)
    _run(paths, monkeypatch)
    assert "stride is 2" in capsys.readouterr().out


def test_lens_columns_name_both_halves():
    assert build.lens_columns("logitlens", 7) == ("logitlens_logmass_L7", "logitlens_prob_L7")
    assert build.fieldnames(["jlens"], 15)[:2] == ["name", "size"]
