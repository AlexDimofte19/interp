"""Tests for the shared, method-dispatched selection logic.

The filter decides what lands on disk and what gets deleted, so the ranking, the tie-breaks,
the control arm's independence from a lens arm, and the two lenses' independence from each
other are all pinned here rather than left to the integration tests.

`test_v1_record_still_reads` is the load-bearing one for the tree that already exists: every
trajectory pruned before methods were introduced has a v1 record, and it is the only
surviving trace of that trajectory's control arm.
"""

import csv
import json
import subprocess
import sys

import pytest
from telos_interp.jlens_utils import (
    METHODS,
    KeptTokens,
    TokenPick,
    TokenScore,
    arm_seed,
    build_record,
    csv_layers,
    load_direction_tokens,
    merge_records,
    parse_methods,
    rank_layers_by_direction,
    rank_tokens,
    read_direction_counts,
    read_selection_record,
    record_path,
    to_disk_coords,
    top_filter,
    write_selection_record,
)

DIRECTIONS = {"UP": [" up"], "DOWN": [" down"], "LEFT": [" left"], "RIGHT": [" right"]}
DIRECTION_WORDS = [" up", " down", " left", " right"]
TOP_K = 4
LAYERS = [7, 15, 23]

CSV_HEADER = ["step", "reasoning_pos", "abs_pos", "token", "layer", *[f"top_{i}" for i in range(1, TOP_K + 1)]]

# Five tokens, totals 8/6/4/2/0 over LAYERS -- a strict ranking on both axes. Per-layer hits
# stay <= TOP_K, since a layer cannot hold more direction words than the CSV has top_i
# columns. Within every token layer 23 outscores 7 outscores 15, so the layer ranking is
# unambiguous too.
PER_TOKEN = {
    (0, 100): {23: 4, 7: 3, 15: 1},   # 8
    (0, 101): {23: 3, 7: 2, 15: 1},   # 6
    (0, 102): {23: 2, 7: 1, 15: 1},   # 4
    (0, 103): {23: 1, 7: 1, 15: 0},   # 2
    (0, 104): {23: 0, 7: 0, 15: 0},   # 0
}


def write_csv(folder, rows, method="jlens"):
    """Write a minimal analysis CSV where that method's filter will look for it.

    `rows` is {(step, abs_pos): {layer: direction hits}}.
    """
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{folder.name}{METHODS[method].csv_suffix}"
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
def folder(tmp_path):
    """A trajectory folder holding a jlens CSV, which is what `top_filter` takes."""
    out = tmp_path / "t"
    write_csv(out, PER_TOKEN)
    return out


@pytest.fixture
def csv_path(folder):
    return folder / f"{folder.name}_jlens_analysis.csv"


def trajectory(num_steps=1, n_prefix=2, n_grid=4, n_suffix=1):
    """output_start = n_prefix + n_grid + n_suffix = 7 by default."""
    return {
        "prompt": {
            "prompt_prefix_tokens": [{}] * n_prefix,
            "prompt_suffix_tokens": [{}] * n_suffix,
        },
        "steps": [{"step_id": s, "grid_state_tokens": [{}] * n_grid} for s in range(num_steps)],
    }


# --- the registry ----------------------------------------------------------------------


def test_parse_methods_validates_eagerly():
    assert parse_methods("jlens,logitlens,random") == ["jlens", "logitlens", "random"]
    with pytest.raises(ValueError, match="Unknown selection method 'jlenz'"):
        parse_methods("jlenz")


def test_the_two_lenses_differ_only_in_their_csv():
    jlens, logitlens = METHODS["jlens"], METHODS["logitlens"]
    assert jlens.scored and logitlens.scored
    assert jlens.csv_suffix != logitlens.csv_suffix
    assert not METHODS["random"].scored
    assert METHODS["random"].csv_suffix is None


def test_control_arm_seed_formula_is_frozen():
    """The pruned tree's control arms were drawn with the bare form; it cannot change."""
    assert arm_seed(42, "traj_1", "random") == "42-traj_1"
    assert arm_seed(42, "traj_1", "future") == "42-traj_1-future"


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


def test_token_ranking_breaks_ties_on_step_then_position():
    scores = {
        (1, 5): TokenScore("a", {7: 2}),
        (0, 9): TokenScore("b", {7: 2}),
        (0, 3): TokenScore("c", {7: 2}),
        (0, 1): TokenScore("d", {7: 9}),
    }
    assert rank_tokens(scores, [7]) == [(0, 1), (0, 3), (0, 9), (1, 5)]


def test_top_tokens_are_taken_in_score_order(signal, folder):
    kept = top_filter(signal, folder, methods=["jlens"], num_tokens=3, num_layers=1, always_layers=())
    assert sorted(kept["jlens"]) == [(0, 100), (0, 101), (0, 102)]
    assert not kept["random"]


def test_top_layers_are_that_token_s_best(signal, folder):
    kept = top_filter(signal, folder, methods=["jlens"], num_tokens=1, num_layers=2, always_layers=())
    # token 100 scores 4 at layer 23, 3 at layer 7, 1 at layer 15
    assert kept["jlens"][(0, 100)].layers == (7, 23)
    assert kept["jlens"][(0, 100)].direction_count == 8
    assert kept["jlens"][(0, 100)].layer_direction_counts == {7: 3, 23: 4}


def test_always_layers_are_added_not_counted(signal, folder):
    """`num_layers=2` plus a forced layer 15 gives 3, since 15 was not in the top 2."""
    kept = top_filter(signal, folder, methods=["jlens"], num_tokens=1, num_layers=2, always_layers=(15,))
    assert kept["jlens"][(0, 100)].layers == (7, 15, 23)

    # ...but never duplicates: 23 is already the top layer
    kept = top_filter(signal, folder, methods=["jlens"], num_tokens=1, num_layers=2, always_layers=(23,))
    assert kept["jlens"][(0, 100)].layers == (7, 23)

    # a forced layer the CSV has no rows for is simply not available
    kept = top_filter(signal, folder, methods=["jlens"], num_tokens=1, num_layers=1, always_layers=(99,))
    assert kept["jlens"][(0, 100)].layers == (23,)


def test_candidate_layers_narrow_both_ranking_and_result(signal, folder):
    kept = top_filter(
        signal, folder, methods=["jlens"], num_tokens=1, num_layers=1,
        always_layers=(), candidate_layers=[7, 15],
    )
    assert kept["jlens"][(0, 100)].layers == (7,)
    assert kept["jlens"][(0, 100)].direction_count == 4  # 3 + 1, layer 23 excluded


def test_asking_for_more_than_exists_keeps_everything(signal, folder):
    kept = top_filter(signal, folder, methods=["jlens"], num_tokens=99, num_layers=99, always_layers=())
    assert len(kept["jlens"]) == 5
    assert kept["jlens"][(0, 100)].layers == tuple(LAYERS)


# --- several methods at once -----------------------------------------------------------


def test_each_lens_scores_its_own_csv(signal, folder):
    """The two lenses share every line of scoring code and disagree only via their CSVs."""
    # logitlens sees the ranking reversed, and one extra layer the jlens CSV never had
    write_csv(folder, {
        (0, 100): {23: 0, 7: 0, 15: 0, 19: 0},
        (0, 104): {23: 4, 7: 3, 15: 1, 19: 2},
    }, method="logitlens")

    kept = top_filter(signal, folder, methods=["jlens", "logitlens"], num_tokens=1,
                      num_layers=1, always_layers=())
    assert sorted(kept["jlens"]) == [(0, 100)]
    assert sorted(kept["logitlens"]) == [(0, 104)]
    # layer 19 exists only in the logitlens CSV, so only that arm can ever select it
    assert kept["logitlens"][(0, 104)].layers == (23,)
    assert kept.names == ["jlens", "logitlens"]


def test_a_lens_with_no_csv_is_absent_not_an_error(signal, folder):
    """A trajectory analysed by one lens only is a data state, not a bug."""
    kept = top_filter(signal, folder, methods=["jlens", "logitlens"], num_tokens=1, always_layers=())
    assert "jlens" in kept
    assert "logitlens" not in kept


def test_no_csv_at_all_yields_nothing(signal, tmp_path):
    empty = tmp_path / "unanalysed"
    empty.mkdir()
    assert top_filter(signal, empty, methods=["jlens", "random"]).arms == {}


def test_an_unscored_arm_alone_has_nothing_to_draw_from(signal, folder):
    """`random` needs a lens CSV to enumerate the chain -- that is what it samples over."""
    with pytest.raises(ValueError, match="no scored method to enumerate"):
        top_filter(signal, folder, methods=["random"], random_tokens=2)


def test_unknown_method_is_rejected_by_name(signal, folder):
    with pytest.raises(ValueError, match="Unknown selection method"):
        top_filter(signal, folder, methods=["jlenz"])


# --- the control arm -------------------------------------------------------------------


def test_random_arm_is_seeded_and_carries_no_counts(signal, folder):
    kw = {"methods": ["jlens", "random"], "num_tokens": 1, "random_tokens": 3, "seed": 1, "seed_key": "t"}
    a = top_filter(signal, folder, **kw)
    b = top_filter(signal, folder, **kw)
    assert list(a["random"]) == list(b["random"])
    assert len(a["random"]) == 3
    for pick in a["random"].values():
        assert pick.direction_count is None, "a control that records counts is not a control"
        assert pick.layer_direction_counts is None
        assert pick.token == ""


def test_random_arm_differs_per_trajectory(signal, folder):
    """Seeding on the trajectory name means each one gets its own draw from one --seed."""
    draws = {
        name: tuple(top_filter(
            signal, folder, methods=["jlens", "random"], num_tokens=1,
            random_tokens=2, seed=1, seed_key=name,
        )["random"])
        for name in [f"traj_{i}" for i in range(8)]
    }
    assert len(set(draws.values())) > 1


def test_random_arm_draws_from_the_whole_chain(signal, folder):
    """Over many seeds every token must be reachable, including the top scorer.

    Drawing from what a lens arm left over would make the control systematically
    low-scoring -- the opposite of a matched control.
    """
    seen = set()
    for seed in range(60):
        kept = top_filter(signal, folder, methods=["jlens", "random"], num_tokens=2,
                          random_tokens=1, seed=seed, seed_key="t")
        seen.update(kept["random"])
    assert seen == {(0, 100), (0, 101), (0, 102), (0, 103), (0, 104)}


def test_random_layers_spread_across_the_pool(signal, folder):
    """The control's layers must not collapse onto one layer the way a ranked draw would."""
    seen = set()
    for seed in range(40):
        kept = top_filter(
            signal, folder, methods=["jlens", "random"], num_tokens=1, random_tokens=1,
            random_layers=1, always_layers=(), seed=seed, seed_key="t",
        )
        seen.update(next(iter(kept["random"].values())).layers)
    assert seen == set(LAYERS)


def test_always_layers_apply_to_the_control_too(signal, folder):
    kept = top_filter(
        signal, folder, methods=["jlens", "random"], num_tokens=1, random_tokens=2,
        random_layers=1, always_layers=(15,), seed_key="t",
    )
    assert all(15 in pick.layers for pick in kept["random"].values())


def test_control_draws_over_the_union_of_the_lenses(signal, folder):
    """With two lenses read, the control can reach anything either of them could pick."""
    write_csv(folder, {(0, 200): {7: 1}, (0, 201): {7: 1}}, method="logitlens")
    seen = set()
    for seed in range(60):
        kept = top_filter(signal, folder, methods=["jlens", "logitlens", "random"],
                          num_tokens=1, random_tokens=1, seed=seed, seed_key="t")
        seen.update(kept["random"])
    assert {(0, 200), (0, 201)} <= seen, "the logitlens CSV's tokens must be drawable"
    assert (0, 100) in seen, "and the jlens CSV's too"


# --- merging and coordinates -----------------------------------------------------------


def test_merged_unions_overlapping_arms():
    kept = KeptTokens({
        "jlens": {(0, 5): TokenPick(0, 5, (7, 15), direction_count=3)},
        "random": {(0, 5): TokenPick(0, 5, (15, 23)), (0, 9): TokenPick(0, 9, (7,))},
    })
    assert kept.merged() == {(0, 5): (7, 15, 23), (0, 9): (7,)}
    assert kept.num_files() == 4
    # restricted to one arm, for reporting which arm a missing file belongs to
    assert kept.merged(["jlens"]) == {(0, 5): (7, 15)}


def test_activation_paths_use_the_gather_layout(tmp_path):
    kept = KeptTokens({"jlens": {(2, 5): TokenPick(2, 5, (7,))}})
    model = tmp_path / "openai__gpt-oss-20b"
    assert kept.activation_paths(model) == {model / "layer_7" / "step_2" / "output" / "5.pt"}


def test_to_disk_coords_subtracts_output_start(signal, folder):
    kept = top_filter(signal, folder, methods=["jlens"], num_tokens=2, num_layers=1, always_layers=())
    disk = to_disk_coords(kept, trajectory())
    # output_start = 2 + 4 + 1 = 7, so abs_pos 100 is the .pt named 93
    assert sorted(disk["jlens"]) == [(0, 93), (0, 94)]
    assert disk["jlens"][(0, 93)].direction_count == 8


def test_to_disk_coords_drops_steps_the_trajectory_lacks(signal, tmp_path):
    out = tmp_path / "x"
    write_csv(out, {(0, 100): {7: 2}, (5, 200): {7: 4}})
    kept = top_filter(signal, out, methods=["jlens"], num_tokens=9, num_layers=1, always_layers=())
    assert len(kept["jlens"]) == 2
    disk = to_disk_coords(kept, trajectory(num_steps=1))
    assert sorted(disk["jlens"]) == [(0, 93)]  # step 5 has no entry to resolve against


def test_step_folder_index_follows_step_id(signal, tmp_path):
    """Folders are named by step_id, which need not equal the CSV's list index."""
    out = tmp_path / "y"
    write_csv(out, {(1, 100): {7: 2}})
    traj = trajectory(num_steps=2)
    traj["steps"][1]["step_id"] = 42
    kept = top_filter(signal, out, methods=["jlens"], num_tokens=1, always_layers=())
    assert sorted(to_disk_coords(kept, traj)["jlens"]) == [(42, 93)]


# --- the selection record --------------------------------------------------------------


def test_record_round_trips(tmp_path, signal, folder):
    kept = to_disk_coords(
        top_filter(signal, folder, methods=["jlens", "random"], num_tokens=2,
                   num_layers=1, random_tokens=2, seed_key="t"),
        trajectory(),
    )
    out = tmp_path / "stub_size5_comp0.4_1"
    out.mkdir()
    path = record_path(out)
    write_selection_record(path, build_record(kept, stem=out.name, model="m__n", config={"seed": 42}))

    loaded, configs = read_selection_record(path)
    assert configs == {"jlens": {"seed": 42}, "random": {"seed": 42}}
    assert loaded == kept


def test_record_keeps_the_control_countless(signal, folder):
    kept = to_disk_coords(
        top_filter(signal, folder, methods=["jlens", "random"], num_tokens=1,
                   random_tokens=2, seed_key="t"),
        trajectory(),
    )
    record = build_record(kept, stem="s", model="m", config={}, output_starts={0: 7})
    jlens_picks = record["arms"]["jlens"]["picks"]
    assert all("direction_count" in e for e in jlens_picks)
    assert all("direction_count" not in e for e in record["arms"]["random"]["picks"])
    assert jlens_picks[0]["abs_pos"] == jlens_picks[0]["token_idx"] + 7


def test_v1_record_still_reads(tmp_path):
    """Every already-pruned trajectory has one of these, and it holds the only control draw.

    Losing the ability to read it would strand the control arm permanently: after pruning, a
    uniform draw over the reasoning chain cannot be made again.
    """
    path = tmp_path / "t_jlens_selection.json"
    path.write_text(json.dumps({
        "format_version": 1,
        "stem": "t", "model": "m__n",
        "config": {"seed": 42, "num_tokens": 20},
        "jlens": [{"step": 0, "token_idx": 93, "layers": [7, 15], "abs_pos": 100,
                   "token": "tok100", "direction_count": 8,
                   "layer_direction_counts": {"7": 3, "15": 1}}],
        "random": [{"step": 0, "token_idx": 5, "layers": [15, 23]}],
    }))

    kept, configs = read_selection_record(path)
    assert kept.names == ["jlens", "random"]
    assert kept["jlens"][(0, 93)].direction_count == 8
    assert kept["jlens"][(0, 93)].layer_direction_counts == {7: 3, 15: 1}
    # the control survives, still count-free -- which is what keeps it sampled, not ranked
    assert kept["random"][(0, 5)].layers == (15, 23)
    assert kept["random"][(0, 5)].direction_count is None
    assert configs["random"] == {"seed": 42, "num_tokens": 20}


def test_merging_preserves_arms_the_new_record_lacks(tmp_path):
    """Adding a logitlens arm must not disturb the two arms already on disk."""
    old = build_record(
        KeptTokens({"jlens": {(0, 1): TokenPick(0, 1, (7,), direction_count=3)},
                    "random": {(0, 2): TokenPick(0, 2, (15,))}}),
        stem="t", model="m", config={"seed": 42},
    )
    new = build_record(
        KeptTokens({"logitlens": {(0, 9): TokenPick(0, 9, (23,), direction_count=5)}}),
        stem="t", model="m", config={"seed": 7},
    )
    merged = merge_records(old, new)
    assert sorted(merged["arms"]) == ["jlens", "logitlens", "random"]
    assert merged["arms"]["random"]["picks"] == old["arms"]["random"]["picks"]
    # each arm keeps the config of the run that built it
    assert merged["arms"]["jlens"]["config"] == {"seed": 42}
    assert merged["arms"]["logitlens"]["config"] == {"seed": 7}


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
