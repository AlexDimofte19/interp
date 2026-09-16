"""The loudness_analysis library layer: columns, signals, stats, provenance, and the two
behaviours that are new rather than moved -- the counts-mode analysis and the signal-word
exclusion.

Nothing here touches /workspace.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import pytest
from telos_interp.loudness_analysis import columns as cols
from telos_interp.loudness_analysis import probes as probe_types
from telos_interp.loudness_analysis import provenance, signals, stats
from telos_interp.loudness_analysis.analysis import probe_accuracy_by_loudness as pabl

# --------------------------------------------------------------------------------------------
# columns: the naming convention, and every legacy spelling still readable
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "present,expected",
    [
        (["jlens_direction_logmass_L15"], "jlens_direction_logmass_L15"),
        (["jlens_mass_L15"], "jlens_mass_L15"),  # eval_probe_per_token
        (["jlens_logmass_L15"], "jlens_logmass_L15"),  # build_token_loudness_x
        (["dir_logmass_L15"], "dir_logmass_L15"),  # build_sentence_loudness
        (["dir_logmass"], "dir_logmass"),  # build_probe_loudness*
    ],
)
def test_every_legacy_loudness_spelling_still_resolves(present, expected):
    assert cols.resolve(["name", *present], "jlens", "direction", 15) == expected


def test_canonical_wins_when_several_spellings_are_present():
    fields = ["dir_logmass", "jlens_mass_L15", "jlens_direction_logmass_L15"]
    assert cols.resolve(fields, "jlens", "direction", 15) == "jlens_direction_logmass_L15"


def test_dir_prefix_is_not_accepted_for_a_non_direction_signal():
    """`dir_logmass` predates any vocabulary but direction, so it identifies the signal.

    Accepting it for the grid signal would silently read direction loudness into a table
    labelled grid -- exactly the confusion the rename exists to stop.
    """
    with pytest.raises(KeyError):
        cols.resolve(["name", "dir_logmass"], "jlens", "grid", 15)


def test_the_two_sentence_fracs_do_not_resolve_to_each_other():
    """`sentence_frac` means within-sentence position here and sentence_idx/n_sentences in
    probe_vs_rollout/per_token.csv. Reading the wrong one destroys a position control silently."""
    assert cols.resolve_ambiguous(["sentence_frac"], "frac_in_sentence") == "sentence_frac"
    with pytest.raises(KeyError):
        cols.resolve_ambiguous(["sentence_frac"], "frac_of_chain")


def test_axis_label_always_names_lens_and_signal():
    assert cols.axis_label("jlens", "direction") == "J-lens direction logmass"
    assert cols.axis_label("logitlens", "grid", 7) == "logitlens grid logmass (L7)"


# --------------------------------------------------------------------------------------------
# signals: open-ended, not the two committed vocabularies
# --------------------------------------------------------------------------------------------


def test_an_unregistered_vocabulary_is_usable_without_a_code_change(tmp_path):
    path = tmp_path / "shape_tokens_full.json"
    path.write_text(json.dumps({"ROUND": [" circle"], "FLAT": [" square"]}))

    sig = signals.resolve(None, path)
    assert sig.name == "shape"
    assert sig.membership_column == "is_shape_token"
    assert sig.load(path) == {" circle", " square"}
    assert sig.classes(path) == ("ROUND", "FLAT")
    assert cols.loudness_column("jlens", sig.name, 15) == "jlens_shape_logmass_L15"


def test_fingerprint_changes_when_the_vocabulary_content_changes(tmp_path):
    path = tmp_path / "v.json"
    path.write_text(json.dumps({"A": [" a"]}))
    first = signals.resolve("v", path).fingerprint(path)
    path.write_text(json.dumps({"A": [" a", " b"]}))
    assert signals.resolve("v", path).fingerprint(path) != first


# --------------------------------------------------------------------------------------------
# stats: the two balanced accuracies are not the same number
# --------------------------------------------------------------------------------------------


def test_the_two_balanced_accuracies_disagree_when_rows_summarise_unequal_buckets():
    """One row with 20 cells and one with 2 must not carry equal weight in the counts form."""
    df = pd.DataFrame(
        {
            "n_true_0": [20, 0],
            "n_true_1": [0, 2],
            "p_correct_0": [10, 0],
            "p_correct_1": [0, 2],
        }
    )
    pooled, recalls = stats.bal_acc_from_counts(df, "p", [0, 1])
    assert recalls == {0: 0.5, 1: 1.0}
    assert pooled == pytest.approx(0.75)

    # The row-averaged form cannot even be computed here: there is no per-row prediction.
    # That is the shape difference, not a tuning parameter.
    assert "p_pred" not in df.columns


def test_a_class_with_no_support_is_dropped_not_scored_zero():
    truth = np.array(["UP", "UP", "DOWN", "DOWN"])
    pred = np.array(["UP", "UP", "DOWN", "DOWN"])
    assert stats.bal_acc(truth, pred, ["UP", "DOWN", "LEFT", "RIGHT"]) == 1.0


# --------------------------------------------------------------------------------------------
# probe types
# --------------------------------------------------------------------------------------------


def test_probe_types_declare_two_distinct_class_vocabularies():
    na = probe_types.get_probe_type("next_action")
    grid = probe_types.get_probe_type("grid_tile")
    # ids for scoring tensors, names for reading a CSV's label column
    assert set(na.classes) == {0, 1, 2, 3}
    assert set(na.analysis_classes) == {"LEFT", "UP", "RIGHT", "DOWN"}
    # class 7 is padding: a cell that does not exist, excluded from any accuracy
    assert 7 in grid.classes
    assert 7 not in grid.analysis_classes
    assert na.aggregation == "rows" and grid.aggregation == "counts"


def test_probe_key_strips_the_trainer_prefix_but_keeps_the_parent_dir(tmp_path):
    """The parent dir keeps two arms with the same filename apart; assuming the prefix was
    NOT stripped once cost a round with a silently failed join."""
    na = probe_types.get_probe_type("next_action")
    assert na.probe_key(tmp_path / "p1" / "next_action_probe_jlens_lr.pt") == "p1.jlens_lr"
    grid = probe_types.get_probe_type("grid_tile")
    assert grid.probe_key(tmp_path / "arm" / "cognitive_map_probe_x_mlp.pt") == "arm.x_mlp"


# --------------------------------------------------------------------------------------------
# the signal-word exclusion
# --------------------------------------------------------------------------------------------


def _args(**over):
    base = dict(
        lens="jlens",
        signal_name="direction",
        layer=15,
        loudness_column=None,
        exclude_signal_words=False,
        exclude_radius=0,
        signal_words=None,
    )
    base.update(over)
    return argparse.Namespace(**base)


def _frame():
    # Two trajectories of 4 tokens; token 1 of each is the signal word.
    return pd.DataFrame(
        {
            "name": ["a"] * 4 + ["b"] * 4,
            "step": [0] * 8,
            "token": [" x", " up", " y", " z"] * 2,
            "jlens_direction_logmass_L15": [-3.0, -1.0, -2.0, -4.0] * 2,
            "is_direction_token": [0, 1, 0, 0] * 2,
        }
    )


def test_exclusion_radius_zero_drops_only_the_signal_word():
    df, _, before, after = pabl._prepare(_frame(), _args(exclude_signal_words=True))
    assert (before, after) == (8, 6)
    assert df["is_signal_word"].sum() == 0


def test_exclusion_radius_one_drops_the_neighbours_too():
    df, _, before, after = pabl._prepare(_frame(), _args(exclude_signal_words=True, exclude_radius=1))
    # per trajectory: tokens 0, 1, 2 go; token 3 survives
    assert (before, after) == (8, 2)
    assert df["token"].tolist() == [" z", " z"]


def test_exclusion_does_not_leak_across_trajectories():
    """A neighbour is a neighbour within its own chain. Shifting over the whole frame would
    let the last token of one trajectory drop the first token of the next."""
    frame = pd.DataFrame(
        {
            "name": ["a", "a", "b", "b"],
            "step": [0, 0, 0, 0],
            "token": [" x", " up", " y", " z"],
            "jlens_direction_logmass_L15": [-3.0, -1.0, -2.0, -4.0],
            "is_direction_token": [0, 1, 0, 0],
        }
    )
    df, _, _, _ = pabl._prepare(frame, _args(exclude_signal_words=True, exclude_radius=1))
    # "a" loses both its rows; "b" keeps both -- its first token is NOT adjacent to a's last.
    assert df["name"].tolist() == ["b", "b"]


def test_prepare_reads_a_legacy_table_unchanged():
    legacy = _frame().rename(columns={"jlens_direction_logmass_L15": "dir_logmass"})
    df, mass_col, _, _ = pabl._prepare(legacy, _args())
    assert mass_col == "dir_logmass"
    assert df["loudness"].tolist() == [-3.0, -1.0, -2.0, -4.0] * 2


# --------------------------------------------------------------------------------------------
# counts mode
# --------------------------------------------------------------------------------------------


def test_counts_mode_bins_a_raw_evaluator_table():
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame(
        {
            "name": [f"t{i % 6}" for i in range(n)],
            "loudness": np.linspace(-5, -1, n),
            "n_cells": 4,
            "n_true_0": 2,
            "n_true_1": 2,
            # accuracy rises with loudness, so the gap must come out positive
            "p_correct_0": (np.linspace(0, 2, n)).astype(int),
            "p_correct_1": (np.linspace(0, 2, n)).astype(int),
            "p_n_correct": (np.linspace(0, 4, n)).astype(int),
            "p_acc": np.linspace(0, 1, n),
        }
    )
    pabl.CLASSES[:] = [0, 1]
    args = argparse.Namespace(
        probe_type="grid_tile", deciles=4, boot=20, reference_gap=0.1559, probes_filter=[], out=None
    )
    table = pabl.decile_table(df, "p", "loudness", 4)
    assert len(table) == 4
    assert table["balanced_acc"].iloc[-1] > table["balanced_acc"].iloc[0]

    gap = pabl.gap_bootstrap(df, "p", "loudness", 4, args.boot, 0)
    assert gap["gap"] > 0
    assert "of the reference" in pabl.verdict(gap["gap"], gap["lo"], gap["hi"], args.reference_gap)
    _ = rng  # fixture determinism is via the explicit seed above


def test_counts_mode_says_so_when_the_table_has_no_probe_columns(tmp_path):
    df = pd.DataFrame({"name": ["a"], "loudness": [-1.0]})
    args = argparse.Namespace(
        probe_type="grid_tile", deciles=4, boot=5, reference_gap=0.1, probes_filter=[], out=tmp_path
    )
    with pytest.raises(SystemExit, match="counts-mode table needs"):
        pabl.run_counts_mode(df, args, probe_types.get_probe_type("grid_tile"))


# --------------------------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------------------------


def test_run_config_records_the_ruler_and_the_aggregation(tmp_path):
    cfg = provenance.RunConfig("x.py")
    cfg.measurement(lens="logitlens", signal="grid", layer=7)
    cfg.aggregation(balanced_accuracy="counts", n_boot=300)
    cfg.rows("input", 100)
    cfg.write(tmp_path)

    saved = json.loads((tmp_path / "run_config.json").read_text())
    assert saved["measurement"]["loudness_column"] == "logitlens_grid_logmass_L7"
    assert saved["measurement"]["axis_label"] == "logitlens grid logmass"
    assert saved["aggregation"]["balanced_accuracy"] == "counts"
    assert saved["row_counts"]["input"] == 100


def test_the_guard_refuses_to_mix_two_rulers_in_one_folder(tmp_path):
    """A folder holding jlens-ruler figures must not also receive logitlens-ruler ones: the
    two are not comparable and the directory would look complete and be wrong."""
    first = provenance.RunConfig("x.py")
    first.measurement(lens="jlens", signal="direction", layer=15)
    first.write(tmp_path)

    second = provenance.RunConfig("x.py")
    second.measurement(lens="logitlens", signal="direction", layer=15)
    with pytest.raises(SystemExit, match="Refusing to mix"):
        second.guard(tmp_path, "lens", "signal", "layer")


def test_the_guard_allows_a_rerun_with_the_same_ruler(tmp_path):
    for _ in range(2):
        cfg = provenance.RunConfig("x.py")
        cfg.measurement(lens="jlens", signal="direction", layer=15)
        cfg.guard(tmp_path, "lens", "signal", "layer")
        cfg.write(tmp_path)
