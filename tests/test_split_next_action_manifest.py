"""Tests for scripts/split_next_action_manifest.py.

This is the last thing that touches the data before a probe trains on it, and everything it
does is a narrowing: fewer tokens, fewer layers, a held-out split. Each of those has a way
of quietly ruining the experiment rather than failing --

  * ranking the `random` control by a count it never carries collapses it onto the lowest
    index, so "jlens vs random" becomes "jlens vs a different jlens",
  * a train/eval split that moves between K=1 and K=3 makes a top-K sweep incomparable,
  * a row-level split leaks a trajectory across both halves and inflates accuracy.

So those are what is pinned here.
"""

import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

split = importlib.import_module("scripts.split_next_action_manifest")

ACTIONS = {"LEFT": 0, "UP": 1, "RIGHT": 2, "DOWN": 3}


def sample(name, step, token_id, layer, label, direction_count=None, layer_direction_count=None):
    row = {
        "name": name,
        "act_path": f"{name}/l{layer}/s{step}/{token_id}.pt",
        "label": label,
        "layer": layer,
        "step": step,
        "token_id": token_id,
        "category": "output",
    }
    if direction_count is not None:
        row["token"] = f"t{token_id}"
        row["direction_count"] = direction_count
        row["layer_direction_count"] = layer_direction_count
    return row


def jlens_samples(num_trajectories=8, tokens=4, layers=(7, 15, 23)):
    """Token `t` scores `10 - t`, so the ranking within a trajectory is t=0 > 1 > 2 > 3."""
    rows = []
    for i in range(num_trajectories):
        name = f"traj_{i:03d}"
        label = i % 4
        for token_id in range(tokens):
            for layer in layers:
                rows.append(sample(
                    name, 0, token_id, layer, label,
                    direction_count=10 - token_id,
                    layer_direction_count={7: 1, 15: 5, 23: 3}[layer],
                ))
    return rows


def control_samples(num_trajectories=8, tokens=4, layers=(7, 15, 23)):
    """The same shape with no counts at all -- what a `recorded_random` manifest looks like."""
    return [
        sample(f"traj_{i:03d}", 0, token_id, layer, i % 4)
        for i in range(num_trajectories)
        for token_id in range(tokens)
        for layer in layers
    ]


def tokens_of(rows):
    return {(r["name"], r["step"], r["token_id"]) for r in rows}


# --- token thinning ---------------------------------------------------------------------


def test_keeps_the_top_ranked_tokens(tmp_path):
    kept = split.thin_tokens(jlens_samples(), tokens_per_trajectory=2, seed=42)
    assert {t for _, _, t in tokens_of(kept)} == {0, 1}, "ranked by direction_count, best first"
    assert len(tokens_of(kept)) == 8 * 2
    assert len(kept) == 8 * 2 * 3, "all of a kept token's layers survive"


def test_token_ranking_breaks_ties_on_step_then_index():
    """Same tie-break as select_token_layer_pairs, so this equals a --num-tokens prepare."""
    rows = [
        sample("t", step, token_id, 15, 0, direction_count=5, layer_direction_count=1)
        for step, token_id in [(1, 0), (0, 9), (0, 2)]
    ]
    kept = split.thin_tokens(rows, tokens_per_trajectory=2, seed=42)
    assert tokens_of(kept) == {("t", 0, 2), ("t", 0, 9)}


def test_control_arm_is_sampled_not_ranked():
    """With no counts every token ties; ranking would take t=0 from every trajectory."""
    kept = split.thin_tokens(control_samples(num_trajectories=40), tokens_per_trajectory=1, seed=42)
    assert len({t for _, _, t in tokens_of(kept)}) > 1, "a collapsed control is not a control"


def test_control_thinning_is_deterministic():
    rows = control_samples()
    assert tokens_of(split.thin_tokens(rows, 2, seed=7)) == tokens_of(split.thin_tokens(rows, 2, seed=7))
    assert tokens_of(split.thin_tokens(rows, 2, seed=7)) != tokens_of(split.thin_tokens(rows, 2, seed=8))


def test_asking_for_more_tokens_than_exist_keeps_all():
    rows = jlens_samples(num_trajectories=2, tokens=3)
    assert split.thin_tokens(rows, tokens_per_trajectory=99, seed=42) == rows


def test_thinning_is_per_trajectory():
    """Every trajectory contributes K, rather than K being taken globally."""
    kept = split.thin_tokens(jlens_samples(num_trajectories=5), tokens_per_trajectory=2, seed=42)
    per_name = {}
    for name, _, token_id in tokens_of(kept):
        per_name.setdefault(name, set()).add(token_id)
    assert len(per_name) == 5
    assert all(len(v) == 2 for v in per_name.values())


# --- the two knobs together -------------------------------------------------------------


def test_token_and_layer_thinning_commute():
    rows = jlens_samples()
    tokens_first = split.thin_layers(split.thin_tokens(rows, 2, 42), 1, 42)
    layers_first = split.thin_tokens(split.thin_layers(rows, 1, 42), 2, 42)
    key = lambda r: (r["name"], r["step"], r["token_id"], r["layer"])  # noqa: E731
    assert sorted(tokens_first, key=key) == sorted(layers_first, key=key)


def test_layer_thinning_ranks_jlens_and_samples_the_control():
    jlens = split.thin_layers(jlens_samples(), layers_per_token=1, seed=42)
    assert {r["layer"] for r in jlens} == {15}, "layer 15 carries the highest count here"

    control = split.thin_layers(control_samples(num_trajectories=40), layers_per_token=1, seed=42)
    assert len({r["layer"] for r in control}) > 1, "a collapsed control is not a control"


# --- the split --------------------------------------------------------------------------


def test_split_is_grouped_by_trajectory():
    rows = jlens_samples(num_trajectories=20)
    train, evaluation = split.split_names(rows, eval_split=0.25, seed=42)
    assert train and evaluation
    assert not (train & evaluation), "a trajectory in both halves leaks its label"


def test_split_keeps_every_class_in_both_halves():
    """train_next_action_probe silently drops eval samples whose label it never saw."""
    rows = jlens_samples(num_trajectories=20)
    train, evaluation = split.split_names(rows, eval_split=0.25, seed=42)
    for half in (train, evaluation):
        labels = {r["label"] for r in rows if r["name"] in half}
        assert labels == set(range(4)), half


def test_split_is_identical_across_top_k():
    """Otherwise a top-1 result and a top-3 result are not measured on the same data."""
    rows = jlens_samples(num_trajectories=20)
    strata = split.trajectory_strata(rows)
    splits = [
        split.split_names(split.thin_tokens(rows, k, 42), eval_split=0.25, seed=42, strata=strata)
        for k in (1, 2, 3)
    ]
    assert splits[0] == splits[1] == splits[2]


def test_strata_come_from_the_unthinned_samples():
    """A multi-step trajectory whose dominant label changes when thinned must not move."""
    rows = [
        sample("t", 0, 0, 15, ACTIONS["UP"], direction_count=9, layer_direction_count=1),
        sample("t", 1, 0, 15, ACTIONS["DOWN"], direction_count=1, layer_direction_count=1),
        sample("t", 2, 0, 15, ACTIONS["DOWN"], direction_count=1, layer_direction_count=1),
    ]
    assert split.trajectory_strata(rows)["t"] == ACTIONS["DOWN"]
    # thinning to the top token leaves only the UP row, which would restratify it
    assert split.trajectory_strata(split.thin_tokens(rows, 1, 42))["t"] == ACTIONS["UP"]


def test_empty_split_is_an_error():
    with pytest.raises(ValueError, match="empty split"):
        split.split_names(jlens_samples(num_trajectories=4), eval_split=1.0, seed=42)


# --- end to end -------------------------------------------------------------------------


def _write_manifest(tmp_path, rows):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "manifest.json").write_text(json.dumps({
        "format_version": 3,
        "probe_type": "next_action",
        "activation_dim": 8,
        "activations_root": str(tmp_path / "acts"),
        "action_to_id": ACTIONS,
        "samples": rows,
    }))
    return prepared


def _run(prepared, *extra):
    argv = ["split_next_action_manifest.py", str(prepared), "--seed", "42", *extra]
    old, sys.argv = sys.argv, argv
    try:
        split.main()
    finally:
        sys.argv = old
    return (
        json.loads((prepared.with_name(prepared.name + "_train") / "manifest.json").read_text()),
        json.loads((prepared.with_name(prepared.name + "_eval") / "manifest.json").read_text()),
    )


def test_end_to_end_writes_both_halves(tmp_path):
    prepared = _write_manifest(tmp_path, jlens_samples(num_trajectories=20))
    train, evaluation = _run(prepared, "--tokens-per-trajectory", "2", "--layers-per-token", "1")

    assert train["split"]["tokens_per_trajectory"] == 2
    assert train["split"]["layers_per_token"] == 1
    assert train["activations_root"] == str(tmp_path / "acts"), "paths still resolve"
    assert not ({s["name"] for s in train["samples"]} & {s["name"] for s in evaluation["samples"]})
    # 20 trajectories x 2 tokens x 1 layer, split between the two halves
    assert len(train["samples"]) + len(evaluation["samples"]) == 20 * 2


def test_end_to_end_rejects_zero(tmp_path):
    prepared = _write_manifest(tmp_path, jlens_samples())
    with pytest.raises(ValueError, match="tokens-per-trajectory must be >= 1"):
        _run(prepared, "--tokens-per-trajectory", "0")


def test_end_to_end_on_a_control_manifest(tmp_path):
    """The control has no counts anywhere; nothing may crash reaching for one."""
    prepared = _write_manifest(tmp_path, control_samples(num_trajectories=20))
    train, evaluation = _run(prepared, "--tokens-per-trajectory", "2", "--layers-per-token", "1")
    assert train["samples"] and evaluation["samples"]
    assert all("direction_count" not in s for s in train["samples"])
