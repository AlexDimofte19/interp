"""Tests for scripts/intersect_belief_arms.py.

The per-sentence arms vary one thing -- which token inside a sentence the chain is cut at --
so a difference in *which sentences* they cover is a confound, not a result. This script
removes the residual asymmetries (a null `model_action` in one arm's rollout but not
another's, a tensor absent from one tree) so the arms are paired row for row rather than
merely equal in count.
"""

import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

intersect = importlib.import_module("scripts.intersect_belief_arms")


def row(name, step, cut_sentence_idx, token_id, label=0):
    """`sentence_idx` is set to a constant on purpose: it is the cutoff ORDINAL, not the
    sentence, so pairing on it would be wrong and these tests must fail if anything does."""
    return {
        "name": name,
        "step": step,
        "cut_sentence_idx": cut_sentence_idx,
        "sentence_idx": 99,
        "token_id": token_id,
        "label": label,
        "layer": 15,
        "act_path": f"{name}/{token_id}.pt",
    }


def write_arm(tmp_path, arm_name, rows):
    d = tmp_path / arm_name
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": 3,
                "probe_type": "next_action",
                "activations_root": "/tmp/acts",
                "samples": rows,
            }
        )
    )
    return d


def run(argv):
    old = sys.argv
    try:
        sys.argv = ["intersect_belief_arms.py", *argv]
        intersect.main()
    finally:
        sys.argv = old


def read(d):
    return json.loads((d / "manifest.json").read_text())["samples"]


def keys(rows):
    return {(r["name"], r["step"], r["cut_sentence_idx"]) for r in rows}


def test_arms_end_up_identical_in_size_and_coverage(tmp_path):
    """The point of the script: same count AND the same sentences, not just the same count."""
    a = write_arm(tmp_path, "a", [row("t1", 0, i, token_id=i) for i in (1, 2, 3)])
    b = write_arm(tmp_path, "b", [row("t1", 0, i, token_id=100 + i) for i in (1, 2)])
    run(["_eq", str(a), str(b)])
    ra, rb = read(tmp_path / "a_eq"), read(tmp_path / "b_eq")
    assert len(ra) == len(rb) == 2
    assert keys(ra) == keys(rb) == {("t1", 0, 1), ("t1", 0, 2)}


def test_the_differing_token_is_preserved(tmp_path):
    """Intersecting pairs the sentences; it must not touch which token each arm chose."""
    a = write_arm(tmp_path, "a", [row("t1", 0, 1, token_id=5)])
    b = write_arm(tmp_path, "b", [row("t1", 0, 1, token_id=9)])
    run(["_eq", str(a), str(b)])
    assert read(tmp_path / "a_eq")[0]["token_id"] == 5
    assert read(tmp_path / "b_eq")[0]["token_id"] == 9


def test_a_sentence_missing_from_one_arm_leaves_all_arms(tmp_path):
    a = write_arm(tmp_path, "a", [row("t1", 0, i, token_id=i) for i in (1, 2)])
    b = write_arm(tmp_path, "b", [row("t1", 0, 1, token_id=1)])
    c = write_arm(tmp_path, "c", [row("t1", 0, i, token_id=i) for i in (1, 2)])
    run(["_eq", str(a), str(b), str(c)])
    for name in ("a_eq", "b_eq", "c_eq"):
        assert keys(read(tmp_path / name)) == {("t1", 0, 1)}


def test_steps_are_kept_apart(tmp_path):
    """Sentence 1 of step 0 and sentence 1 of step 1 are different sentences."""
    a = write_arm(tmp_path, "a", [row("t1", 0, 1, 1), row("t1", 1, 1, 2)])
    b = write_arm(tmp_path, "b", [row("t1", 0, 1, 3)])
    run(["_eq", str(a), str(b)])
    assert keys(read(tmp_path / "a_eq")) == {("t1", 0, 1)}


def test_rows_without_a_sentence_index_are_dropped(tmp_path):
    """An unpairable row is exactly the asymmetry this removes, so it does not survive."""
    unpairable = {"name": "t1", "step": 0, "token_id": 7, "label": 0, "layer": 15, "act_path": "x.pt"}
    a = write_arm(tmp_path, "a", [row("t1", 0, 1, 1), unpairable])
    b = write_arm(tmp_path, "b", [row("t1", 0, 1, 2)])
    run(["_eq", str(a), str(b)])
    assert len(read(tmp_path / "a_eq")) == 1


def test_provenance_is_recorded(tmp_path):
    a = write_arm(tmp_path, "a", [row("t1", 0, i, i) for i in (1, 2)])
    b = write_arm(tmp_path, "b", [row("t1", 0, 1, 1)])
    run(["_eq", str(a), str(b)])
    meta = json.loads((tmp_path / "a_eq" / "manifest.json").read_text())["intersected_with"]
    assert meta["n_in"] == 2 and meta["n_out"] == 1 and meta["n_common_sentences"] == 1


def test_one_arm_is_an_error(tmp_path):
    a = write_arm(tmp_path, "a", [row("t1", 0, 1, 1)])
    with pytest.raises(SystemExit, match="at least two arms"):
        run(["_eq", str(a)])
