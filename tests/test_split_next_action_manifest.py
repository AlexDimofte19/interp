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
                rows.append(
                    sample(
                        name,
                        0,
                        token_id,
                        layer,
                        label,
                        direction_count=10 - token_id,
                        layer_direction_count={7: 1, 15: 5, 23: 3}[layer],
                    )
                )
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
        split.split_names(split.thin_tokens(rows, k, 42), eval_split=0.25, seed=42, strata=strata) for k in (1, 2, 3)
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
    (prepared / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": 3,
                "probe_type": "next_action",
                "activation_dim": 8,
                "activations_root": str(tmp_path / "acts"),
                "action_to_id": ACTIONS,
                "samples": rows,
            }
        )
    )
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


# --- one layer for the whole dataset ------------------------------------------------------
#
# `--layers-per-token 1` still leaves the dataset spread over many layers, one per token,
# and a single weight vector cannot read them all. `--single-layer` pins the lot. Two things
# have to stay visible: which layer was chosen and why, and that any layer but 15 silently
# costs tokens, since 15 is the only one force-kept for every selected token.


def test_best_layer_is_the_highest_mean(tmp_path):
    assert split.best_layer(jlens_samples()) == 15, "layer_direction_count 5 beats 3 beats 1"
    assert split.mean_layer_scores(jlens_samples())[15] == (5.0, 8 * 4)


def test_best_layer_handles_negative_scores():
    """A logprob score is negative; 'highest' must not become 'closest to zero magnitude'."""
    rows = [
        sample("t", 0, 0, 7, 0, direction_count=-1.0, layer_direction_count=-9.0),
        sample("t", 0, 0, 15, 0, direction_count=-1.0, layer_direction_count=-0.5),
    ]
    assert split.best_layer(rows) == 15


def test_best_layer_is_none_without_scores():
    assert split.best_layer(control_samples()) is None


def test_single_layer_best_pins_the_dataset(tmp_path):
    prepared = _write_manifest(tmp_path, jlens_samples(num_trajectories=20))
    train, evaluation = _run(prepared, "--single-layer", "best")
    assert train["split"]["single_layer"] == 15
    assert {s["layer"] for s in train["samples"] + evaluation["samples"]} == {15}
    assert len(train["samples"]) + len(evaluation["samples"]) == 20 * 4


def test_single_layer_takes_an_explicit_layer(tmp_path):
    """How a control arm is matched to the lens arm's layer -- it cannot compute one."""
    prepared = _write_manifest(tmp_path, control_samples(num_trajectories=20))
    train, evaluation = _run(prepared, "--single-layer", "23")
    assert train["split"]["single_layer"] == 23
    assert {s["layer"] for s in train["samples"] + evaluation["samples"]} == {23}


def test_single_layer_best_refuses_an_unscored_manifest(tmp_path):
    prepared = _write_manifest(tmp_path, control_samples(num_trajectories=20))
    with pytest.raises(ValueError, match="needs per-layer direction scores"):
        _run(prepared, "--single-layer", "best")


def test_single_layer_drops_tokens_that_never_selected_it(tmp_path, capsys):
    """Only L15 is kept for every token; any other layer thins the token set too."""
    rows = jlens_samples(num_trajectories=20)
    # half the tokens were never selected at layer 23
    rows = [r for r in rows if not (r["layer"] == 23 and r["token_id"] % 2)]
    prepared = _write_manifest(tmp_path, rows)
    train, evaluation = _run(prepared, "--single-layer", "23")
    assert len(train["samples"]) + len(evaluation["samples"]) == 20 * 2
    assert "had no entry at L23" in capsys.readouterr().out


def test_single_layer_and_layers_per_token_are_exclusive(tmp_path):
    prepared = _write_manifest(tmp_path, jlens_samples())
    with pytest.raises(ValueError, match="pass one"):
        _run(prepared, "--single-layer", "best", "--layers-per-token", "1")


def test_single_layer_composes_with_token_thinning(tmp_path):
    prepared = _write_manifest(tmp_path, jlens_samples(num_trajectories=20))
    train, evaluation = _run(prepared, "--tokens-per-trajectory", "2", "--single-layer", "best")
    assert len(train["samples"]) + len(evaluation["samples"]) == 20 * 2
    assert {s["layer"] for s in train["samples"]} == {15}


# --- grid_tile manifests ------------------------------------------------------------------
#
# A grid arm is the same token selection with a different label, so it has to survive the
# same reshaping. Two things differ and both can fail quietly: the entries live under
# `trajectories` rather than `samples`, and there is no scalar label to stratify or count.


def grid_samples(num_trajectories=8, tokens=4, layers=(7, 15, 23), sizes=(5, 15)):
    """Token-major grid entries: no `label`, a `cells_key`, and a per-trajectory size."""
    rows = []
    for i in range(num_trajectories):
        name = f"traj_{i:03d}"
        for token_id in range(tokens):
            for layer in layers:
                row = sample(
                    name,
                    0,
                    token_id,
                    layer,
                    label=0,
                    direction_count=10 - token_id,
                    layer_direction_count={7: 1, 15: 5, 23: 3}[layer],
                )
                del row["label"]
                row["cells_key"] = f"{name}|0"
                row["size"] = sizes[i % len(sizes)]
                rows.append(row)
    return rows


def _write_grid_manifest(tmp_path, rows):
    prepared = tmp_path / "prepared_grid"
    prepared.mkdir()
    names = sorted({r["name"] for r in rows})
    (prepared / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": 3,
                "probe_type": "grid_tile",
                "activation_dim": 8,
                "activations_root": str(tmp_path / "acts"),
                "num_cells_per_trajectory": 4,
                "cells": {
                    f"{name}|0": {"positions": [[0, 0], [0, 1], [1, 0], [1, 1]], "labels": [0, 1, 2, 3]}
                    for name in names
                },
                "trajectories": rows,
            }
        )
    )
    return prepared


def _run_grid(prepared, *extra):
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


def test_grid_split_writes_under_the_trajectories_key(tmp_path):
    prepared = _write_grid_manifest(tmp_path, grid_samples(num_trajectories=20))
    train, evaluation = _run_grid(prepared, "--tokens-per-trajectory", "2", "--layers-per-token", "1")

    assert "samples" not in train, "grid entries keep the key the loader reads"
    assert len(train["trajectories"]) + len(evaluation["trajectories"]) == 20 * 2
    assert not ({s["name"] for s in train["trajectories"]} & {s["name"] for s in evaluation["trajectories"]})
    assert train["activations_root"] == str(tmp_path / "acts")


def test_grid_split_prunes_the_cells_map(tmp_path):
    """An eval manifest carrying the training grids around is dead weight in every run."""
    prepared = _write_grid_manifest(tmp_path, grid_samples(num_trajectories=20))
    train, evaluation = _run_grid(prepared, "--layers-per-token", "1")

    assert set(train["cells"]) == {s["cells_key"] for s in train["trajectories"]}
    assert set(evaluation["cells"]) == {s["cells_key"] for s in evaluation["trajectories"]}
    assert not (set(train["cells"]) & set(evaluation["cells"]))


def test_grid_strata_fall_back_to_size(tmp_path):
    """With no scalar label, the size mix is what has to match across the halves."""
    rows = grid_samples(num_trajectories=20)
    strata = split.trajectory_strata(rows)
    assert set(strata.values()) == {5, 15}

    prepared = _write_grid_manifest(tmp_path, rows)
    train, evaluation = _run_grid(prepared, "--layers-per-token", "1")
    for half in (train, evaluation):
        assert len({s["size"] for s in half["trajectories"]}) == 2


def test_grid_split_honours_eval_names(tmp_path):
    """The shared test set is the point of the grid rerun: same trajectories as every arm."""
    prepared = _write_grid_manifest(tmp_path, grid_samples(num_trajectories=10))
    names_file = tmp_path / "eval_names.txt"
    wanted = [f"traj_{i:03d}" for i in (1, 4, 7)]
    names_file.write_text("\n".join(wanted))

    train, evaluation = _run_grid(prepared, "--layers-per-token", "1", "--eval-names", str(names_file))
    assert {s["name"] for s in evaluation["trajectories"]} == set(wanted)
    assert not ({s["name"] for s in train["trajectories"]} & set(wanted))


def test_a_copied_manifest_is_refused(tmp_path):
    """A grid_tile dataset prepared without a selection is one entry per trajectory and
    carries none of the per-token fields the thinning ranks on."""
    prepared = tmp_path / "copied"
    prepared.mkdir()
    (prepared / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": 3,
                "probe_type": "grid_tile",
                "activation_dim": 8,
                "num_cells_per_trajectory": 4,
                "trajectories": [
                    {"name": "traj_000", "act_path": "activations/traj_000.pt", "positions": [[0, 0]], "labels": [0]}
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="token-selection"):
        split.load_manifest(prepared)


def test_thinning_ranks_negative_scores_the_right_way_up():
    """A logprob score is negative and 0 is its maximum; `-(x or 0)` would invert this."""
    rows = []
    for token_id, score in ((0, -9.0), (1, -0.5)):
        for layer in (7, 15):
            rows.append(sample("t", 0, token_id, layer, 0, direction_count=score, layer_direction_count=score))
    kept = split.thin_tokens(rows, tokens_per_trajectory=1, seed=42)
    assert {t for _, _, t in tokens_of(kept)} == {1}, "-0.5 is more direction mass than -9.0"


def test_a_row_missing_its_score_sorts_last_not_first():
    """0 is a count's floor but a logprob's ceiling, so absence cannot stand in for it."""
    rows = [
        sample("t", 0, 0, 7, 0, direction_count=-1.0, layer_direction_count=-1.0),
        sample("t", 0, 0, 15, 0, direction_count=-1.0),  # scored token, unscored layer row
    ]
    del rows[1]["layer_direction_count"]
    kept = split.thin_layers(rows, layers_per_token=1, seed=42)
    assert [r["layer"] for r in kept] == [7]
