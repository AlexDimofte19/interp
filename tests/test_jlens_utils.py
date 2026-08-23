"""Tests for the shared jlens selection logic.

The filter decides what lands on disk and what gets deleted, so the ranking, the tie-breaks
and the control arm's independence from the jlens arm are all pinned here rather than left
to the integration tests.
"""

import csv
import json
import subprocess
import sys

import pytest
from telos_interp.jlens_utils import (
    KeptTokens,
    TokenPick,
    build_record,
    csv_layers,
    jlens_top_filter,
    load_direction_tokens,
    rank_layers_by_direction,
    read_direction_counts,
    read_selection_record,
    record_path,
    to_disk_coords,
    write_selection_record,
)

DIRECTIONS = {"UP": [" up"], "DOWN": [" down"], "LEFT": [" left"], "RIGHT": [" right"]}
DIRECTION_WORDS = [" up", " down", " left", " right"]
TOP_K = 4
LAYERS = [7, 15, 23]

CSV_HEADER = ["step", "reasoning_pos", "abs_pos", "token", "layer", *[f"top_{i}" for i in range(1, TOP_K + 1)]]


def write_csv(path, rows):
    """Write a minimal jlens CSV. `rows` is {(step, abs_pos): {layer: direction hits}}."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for (step, abs_pos), per_layer in sorted(rows.items()):
            for layer, hits in sorted(per_layer.items()):
                # `hits` direction words, padded with a filler that is not one
                top = [DIRECTION_WORDS[i % len(DIRECTION_WORDS)] for i in range(hits)]
                top += ["_filler"] * (TOP_K - hits)
                writer.writerow([step, abs_pos, abs_pos, f"tok{abs_pos}", layer, *top])
    return path


@pytest.fixture
def signal(tmp_path):
    path = tmp_path / "directions.json"
    path.write_text(json.dumps(DIRECTIONS))
    return path


@pytest.fixture
def csv_path(tmp_path):
    """Five tokens, totals 8/6/4/2/0 over LAYERS -- a strict ranking on both axes.

    Per-layer hits stay <= TOP_K, since a layer cannot have more direction words than the
    CSV has top_i columns. Within every token, layer 23 outscores 7 outscores 15, so the
    layer ranking is unambiguous too.
    """
    per_token = {
        (0, 100): {23: 4, 7: 3, 15: 1},   # 8
        (0, 101): {23: 3, 7: 2, 15: 1},   # 6
        (0, 102): {23: 2, 7: 1, 15: 1},   # 4
        (0, 103): {23: 1, 7: 1, 15: 0},   # 2
        (0, 104): {23: 0, 7: 0, 15: 0},   # 0
    }
    return write_csv(tmp_path / "t_jlens_analysis.csv", per_token)


def trajectory(num_steps=1, n_prefix=2, n_grid=4, n_suffix=1):
    """output_start = n_prefix + n_grid + n_suffix = 7 by default."""
    return {
        "prompt": {
            "prompt_prefix_tokens": [{}] * n_prefix,
            "prompt_suffix_tokens": [{}] * n_suffix,
        },
        "steps": [{"step_id": s, "grid_state_tokens": [{}] * n_grid} for s in range(num_steps)],
    }


# --- CSV reading -----------------------------------------------------------------------


def test_reads_counts_and_layers(signal, csv_path):
    tokens = load_direction_tokens(signal)
    counts = read_direction_counts(csv_path, tokens, top_k=TOP_K)
    assert csv_layers(csv_path) == LAYERS
    assert counts[(0, 100)].total(LAYERS) == 8
    assert counts[(0, 104)].total(LAYERS) == 0
    assert counts[(0, 100)].token == "tok100"


def test_direction_classes_narrow_the_vocabulary(signal):
    assert load_direction_tokens(signal, "UP,DOWN") == {" up", " down"}
    with pytest.raises(ValueError, match="Unknown direction class"):
        load_direction_tokens(signal, "SIDEWAYS")


def test_token_strings_that_break_pandas_survive(tmp_path, signal):
    """Decoded tokens include "NA", empty strings and commas; DictReader must keep them."""
    path = tmp_path / "odd_jlens_analysis.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerow([0, 0, 100, "NA", 7, " up", "", "a,b", "NA"])
    counts = read_direction_counts(path, load_direction_tokens(signal), top_k=TOP_K)
    assert counts[(0, 100)].token == "NA"
    assert counts[(0, 100)].per_layer[7] == 1  # only " up" is a direction token


# --- ranking ---------------------------------------------------------------------------


def test_layer_ranking_breaks_ties_on_the_lower_layer():
    assert rank_layers_by_direction({7: 2, 15: 9, 23: 2}, [7, 15, 23]) == [15, 7, 23]
    assert rank_layers_by_direction({}, [23, 7, 15]) == [7, 15, 23]  # all tied -> ascending


def test_top_tokens_are_taken_in_score_order(signal, csv_path):
    kept = jlens_top_filter(signal, csv_path, num_tokens=3, num_layers=1, always_layers=())
    assert sorted(kept.jlens) == [(0, 100), (0, 101), (0, 102)]
    assert not kept.random


def test_top_layers_are_that_token_s_best(signal, csv_path):
    kept = jlens_top_filter(signal, csv_path, num_tokens=1, num_layers=2, always_layers=())
    # token 100 scores 4 at layer 23, 3 at layer 7, 1 at layer 15
    assert kept.jlens[(0, 100)].layers == (7, 23)
    assert kept.jlens[(0, 100)].direction_count == 8
    assert kept.jlens[(0, 100)].layer_direction_counts == {7: 3, 23: 4}


def test_always_layers_are_added_not_counted(signal, csv_path):
    """`num_layers=2` plus a forced layer 15 gives 3, since 15 was not in the top 2."""
    kept = jlens_top_filter(signal, csv_path, num_tokens=1, num_layers=2, always_layers=(15,))
    assert kept.jlens[(0, 100)].layers == (7, 15, 23)

    # ...but never duplicates: 23 is already the top layer
    kept = jlens_top_filter(signal, csv_path, num_tokens=1, num_layers=2, always_layers=(23,))
    assert kept.jlens[(0, 100)].layers == (7, 23)

    # a forced layer the CSV has no rows for is simply not available
    kept = jlens_top_filter(signal, csv_path, num_tokens=1, num_layers=1, always_layers=(99,))
    assert kept.jlens[(0, 100)].layers == (23,)


def test_candidate_layers_narrow_both_ranking_and_result(signal, csv_path):
    kept = jlens_top_filter(
        signal, csv_path, num_tokens=1, num_layers=1, always_layers=(), candidate_layers=[7, 15]
    )
    assert kept.jlens[(0, 100)].layers == (7,)
    assert kept.jlens[(0, 100)].direction_count == 4  # 3 + 1, layer 23 excluded


def test_asking_for_more_than_exists_keeps_everything(signal, csv_path):
    kept = jlens_top_filter(signal, csv_path, num_tokens=99, num_layers=99, always_layers=())
    assert len(kept.jlens) == 5
    assert kept.jlens[(0, 100)].layers == tuple(LAYERS)


# --- the control arm -------------------------------------------------------------------


def test_random_arm_is_seeded_and_carries_no_counts(signal, csv_path):
    a = jlens_top_filter(signal, csv_path, num_tokens=1, random_tokens=3, seed=1, seed_key="t")
    b = jlens_top_filter(signal, csv_path, num_tokens=1, random_tokens=3, seed=1, seed_key="t")
    assert list(a.random) == list(b.random)
    assert len(a.random) == 3
    for pick in a.random.values():
        assert pick.direction_count is None, "a control that records counts is not a control"
        assert pick.layer_direction_counts is None
        assert pick.token == ""


def test_random_arm_differs_per_trajectory(signal, csv_path):
    """Seeding on the trajectory name means each one gets its own draw from one --seed."""
    draws = {
        name: tuple(jlens_top_filter(
            signal, csv_path, num_tokens=1, random_tokens=2, seed=1, seed_key=name,
        ).random)
        for name in [f"traj_{i}" for i in range(8)]
    }
    assert len(set(draws.values())) > 1


def test_random_arm_draws_from_the_whole_chain(signal, csv_path):
    """Over many seeds every token must be reachable, including the top scorer.

    Drawing from what the jlens arm left over would make the control systematically
    low-scoring -- the opposite of a matched control.
    """
    seen = set()
    for seed in range(60):
        kept = jlens_top_filter(signal, csv_path, num_tokens=2, random_tokens=1, seed=seed, seed_key="t")
        seen.update(kept.random)
    assert seen == {(0, 100), (0, 101), (0, 102), (0, 103), (0, 104)}


def test_random_layers_spread_across_the_pool(signal, csv_path):
    """The control's layers must not collapse onto one layer the way a ranked draw would."""
    seen = set()
    for seed in range(40):
        kept = jlens_top_filter(
            signal, csv_path, num_tokens=1, random_tokens=1, random_layers=1,
            always_layers=(), seed=seed, seed_key="t",
        )
        seen.update(next(iter(kept.random.values())).layers)
    assert seen == set(LAYERS)


def test_always_layers_apply_to_the_control_too(signal, csv_path):
    kept = jlens_top_filter(
        signal, csv_path, num_tokens=1, random_tokens=2, random_layers=1, always_layers=(15,), seed_key="t"
    )
    assert all(15 in pick.layers for pick in kept.random.values())


# --- merging and coordinates -----------------------------------------------------------


def test_merged_unions_overlapping_arms():
    kept = KeptTokens(
        jlens={(0, 5): TokenPick(0, 5, (7, 15), direction_count=3)},
        random={(0, 5): TokenPick(0, 5, (15, 23)), (0, 9): TokenPick(0, 9, (7,))},
    )
    assert kept.merged() == {(0, 5): (7, 15, 23), (0, 9): (7,)}
    assert kept.num_files() == 4


def test_activation_paths_use_the_gather_layout(tmp_path):
    kept = KeptTokens(jlens={(2, 5): TokenPick(2, 5, (7,))})
    model = tmp_path / "openai__gpt-oss-20b"
    assert kept.activation_paths(model) == {model / "layer_7" / "step_2" / "output" / "5.pt"}


def test_to_disk_coords_subtracts_output_start(signal, csv_path):
    kept = jlens_top_filter(signal, csv_path, num_tokens=2, num_layers=1, always_layers=())
    disk = to_disk_coords(kept, trajectory())
    # output_start = 2 + 4 + 1 = 7, so abs_pos 100 is the .pt named 93
    assert sorted(disk.jlens) == [(0, 93), (0, 94)]
    assert disk.jlens[(0, 93)].direction_count == 8


def test_to_disk_coords_drops_steps_the_trajectory_lacks(signal, tmp_path):
    csv_file = write_csv(tmp_path / "x_jlens_analysis.csv", {(0, 100): {7: 2}, (5, 200): {7: 4}})
    kept = jlens_top_filter(signal, csv_file, num_tokens=9, num_layers=1, always_layers=())
    assert len(kept.jlens) == 2
    disk = to_disk_coords(kept, trajectory(num_steps=1))
    assert sorted(disk.jlens) == [(0, 93)]  # step 5 has no entry to resolve against


def test_step_folder_index_follows_step_id(signal, tmp_path):
    """Folders are named by step_id, which need not equal the CSV's list index."""
    csv_file = write_csv(tmp_path / "y_jlens_analysis.csv", {(1, 100): {7: 2}})
    traj = trajectory(num_steps=2)
    traj["steps"][1]["step_id"] = 42
    disk = to_disk_coords(jlens_top_filter(signal, csv_file, num_tokens=1, always_layers=()), traj)
    assert sorted(disk.jlens) == [(42, 93)]


# --- the selection record --------------------------------------------------------------


def test_record_round_trips(tmp_path, signal, csv_path):
    kept = to_disk_coords(
        jlens_top_filter(signal, csv_path, num_tokens=2, num_layers=1, random_tokens=2, seed_key="t"),
        trajectory(),
    )
    folder = tmp_path / "stub_size5_comp0.4_1"
    folder.mkdir()
    path = record_path(folder)
    write_selection_record(path, build_record(kept, stem=folder.name, model="m__n", config={"seed": 42}))

    loaded, config = read_selection_record(path)
    assert config == {"seed": 42}
    assert loaded == kept


def test_record_keeps_the_control_countless(tmp_path, signal, csv_path):
    kept = to_disk_coords(
        jlens_top_filter(signal, csv_path, num_tokens=1, random_tokens=2, seed_key="t"), trajectory()
    )
    record = build_record(kept, stem="s", model="m", config={}, output_starts={0: 7})
    assert all("direction_count" in e for e in record["jlens"])
    assert all("direction_count" not in e for e in record["random"])
    assert record["jlens"][0]["abs_pos"] == record["jlens"][0]["token_idx"] + 7


def test_record_rejects_an_unknown_version(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"format_version": 99, "jlens": [], "random": []}))
    with pytest.raises(ValueError, match="format_version"):
        read_selection_record(path)


# --- the package's own constraint ------------------------------------------------------


def test_package_does_not_import_torch():
    """The pruning script and these tests must not have to load the model stack."""
    code = "import sys, telos_interp.jlens_utils; assert 'torch' not in sys.modules"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
