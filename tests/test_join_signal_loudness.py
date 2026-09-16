"""Adding a second signal's loudness to a per-token table that already exists.

The join replaces re-running `score_probes_per_token.py` -- hours of GPU -- to change four
columns per lens, so what it must prove is that it changes NOTHING ELSE: the probe verdicts
come through byte-for-byte, and the loudness cells equal the mass table they were read from.

The other half is the guard. Every mass table on disk is named `{stem}_{lens}_direction_mass.csv`
whatever vocabulary produced it, so the filename cannot say which question its numbers
answer; only the `.meta.json` sidecar can. A join that read the wrong tree would relabel one
question as another and nothing in the numbers would reveal it.
"""

from __future__ import annotations

import csv
import json

import pytest
from telos_interp.loudness_analysis import join_signal_loudness as J

LENSES = ("jlens", "logitlens")
LAYERS = (14, 15, 16)


def _write_tree(root, stem, size, signal_name, *, sidecar=True, values=None):
    """One trajectory folder holding an analysis CSV (so `trajectory_dirs` finds it) and one
    mass table per lens, each with the sidecar naming `signal_name`."""
    folder = root / f"size{size}" / stem
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{stem}_jlens_analysis.csv").write_text("size,complexity,run,step,abs_pos,token,layer\n")
    for lens in LENSES:
        path = folder / f"{stem}_{lens}_direction_mass.csv"
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["size", "complexity", "run", "step", "abs_pos", "token", *[f"L{n}" for n in LAYERS]])
            for step, abs_pos in [(0, 100), (0, 101), (1, 200)]:
                cells = values[(lens, step, abs_pos)] if values else [-3.0, -2.0, -4.0]
                w.writerow([size, 0.0, 1, step, abs_pos, "tok", *cells])
        if sidecar:
            (folder.parent / stem / f"{stem}_{lens}_direction_mass.csv.meta.json").write_text(
                json.dumps({"signal_name": signal_name, "signal_json": f"/x/{signal_name}.json", "lens": lens})
            )
    return folder


def _write_table(path, stem):
    """A minimal counts-mode probe table: the key columns plus one probe verdict per row."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "step", "abs_pos", "token", "n_cells", "n_true_1", "arm.p_mlp_n_correct"])
        for step, abs_pos in [(0, 100), (0, 101), (1, 200)]:
            w.writerow([stem, step, abs_pos, "tok", 5, 5, 3])
    return path


@pytest.fixture
def tree_and_table(tmp_path):
    stem = "together_ai_openai_gpt-oss-20b_size9_comp0.0_1"
    _write_tree(tmp_path / "grid_tree", stem, 9, "grid")
    table = _write_table(tmp_path / "table.csv", stem)
    return stem, tmp_path / "grid_tree", table


def test_it_adds_three_columns_per_lens_and_keeps_every_original(tree_and_table, tmp_path):
    _, tree, table = tree_and_table
    out = tmp_path / "out" / "joined.csv"
    assert J.main(["--table", str(table), "--lens-root", str(tree), "--signal-name", "grid", "--out", str(out)]) == 0

    before = list(csv.DictReader(open(table)))
    after = list(csv.DictReader(open(out)))
    assert len(after) == len(before) == 3
    for old, new in zip(before, after, strict=True):
        # every original cell survives untouched -- the probe verdict especially
        for key, value in old.items():
            assert new[key] == value
    for lens in LENSES:
        assert f"{lens}_grid_logmass_L15" in after[0]
        assert f"{lens}_grid_logmass_best_layer" in after[0]
        assert f"{lens}_grid_logmass_best" in after[0]


def test_the_cells_equal_the_mass_table_and_best_is_the_argmax_layer(tmp_path):
    stem = "together_ai_openai_gpt-oss-20b_size9_comp0.0_1"
    # L14=-3, L15=-2, L16=-4  ->  loudness -2 at layer 15, argmax layer 15
    # L14=-1, L15=-5, L16=-2  ->  loudness -5 at layer 15, argmax layer 14
    values = {}
    for lens in LENSES:
        values[(lens, 0, 100)] = [-3.0, -2.0, -4.0]
        values[(lens, 0, 101)] = [-1.0, -5.0, -2.0]
        values[(lens, 1, 200)] = [-3.0, -2.0, -4.0]
    _write_tree(tmp_path / "t", stem, 9, "grid", values=values)
    table = _write_table(tmp_path / "table.csv", stem)
    out = tmp_path / "joined.csv"
    J.main(["--table", str(table), "--lens-root", str(tmp_path / "t"), "--signal-name", "grid", "--out", str(out)])

    rows = list(csv.DictReader(open(out)))
    assert float(rows[0]["jlens_grid_logmass_L15"]) == -2.0
    assert rows[0]["jlens_grid_logmass_best_layer"] == "15"
    assert float(rows[0]["jlens_grid_logmass_best"]) == -2.0
    # the token that is quiet at 15 but loud at 14: the two columns are different questions
    assert float(rows[1]["jlens_grid_logmass_L15"]) == -5.0
    assert rows[1]["jlens_grid_logmass_best_layer"] == "14"
    assert float(rows[1]["jlens_grid_logmass_best"]) == -1.0


def test_it_refuses_a_tree_baked_against_a_different_vocabulary(tmp_path):
    """The filename is `_direction_mass.csv` either way; only the sidecar knows."""
    stem = "together_ai_openai_gpt-oss-20b_size9_comp0.0_1"
    _write_tree(tmp_path / "dir_tree", stem, 9, "direction")
    table = _write_table(tmp_path / "table.csv", stem)
    with pytest.raises(SystemExit, match="baked against signal 'direction', not 'grid'"):
        J.main(
            [
                "--table",
                str(table),
                "--lens-root",
                str(tmp_path / "dir_tree"),
                "--signal-name",
                "grid",
                "--out",
                str(tmp_path / "o.csv"),
            ]
        )


def test_a_table_with_no_sidecar_is_refused_unless_asserted(tmp_path):
    stem = "together_ai_openai_gpt-oss-20b_size9_comp0.0_1"
    _write_tree(tmp_path / "bare", stem, 9, "grid", sidecar=False)
    table = _write_table(tmp_path / "table.csv", stem)
    argv = ["--table", str(table), "--lens-root", str(tmp_path / "bare"), "--signal-name", "grid"]
    with pytest.raises(SystemExit, match="no .meta.json sidecar"):
        J.main([*argv, "--out", str(tmp_path / "o.csv")])
    assert J.main([*argv, "--allow-unlabelled", "--out", str(tmp_path / "o2.csv")]) == 0


def test_joining_the_same_signal_twice_is_refused(tree_and_table, tmp_path):
    """The second run would otherwise write duplicate column names into the header."""
    _, tree, table = tree_and_table
    out = tmp_path / "joined.csv"
    J.main(["--table", str(table), "--lens-root", str(tree), "--signal-name", "grid", "--out", str(out)])
    with pytest.raises(SystemExit, match="joined against 'grid' already"):
        J.main(
            ["--table", str(out), "--lens-root", str(tree), "--signal-name", "grid", "--out", str(tmp_path / "b.csv")]
        )


def test_a_token_the_tree_has_no_row_for_is_empty_not_floored(tmp_path):
    """Absent is not quiet: a floor value would put unmeasured tokens in the quietest
    decile, which is a silent bias in exactly the direction the analysis is looking."""
    stem = "together_ai_openai_gpt-oss-20b_size9_comp0.0_1"
    _write_tree(tmp_path / "t", stem, 9, "grid")
    table = tmp_path / "table.csv"
    with open(table, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "step", "abs_pos", "token"])
        w.writerow([stem, 0, 100, "tok"])  # in the tree
        w.writerow([stem, 0, 999, "tok"])  # not in the tree
    out = tmp_path / "joined.csv"
    J.main(["--table", str(table), "--lens-root", str(tmp_path / "t"), "--signal-name", "grid", "--out", str(out)])

    rows = list(csv.DictReader(open(out)))
    assert float(rows[0]["jlens_grid_logmass_L15"]) == -2.0
    assert rows[1]["jlens_grid_logmass_L15"] == ""
    assert rows[1]["jlens_grid_logmass_best_layer"] == ""


def test_it_refuses_when_nothing_joined_at_all(tmp_path):
    """The way this fails in real use is a table keyed on `token_idx` instead of the
    prompt-inclusive `abs_pos`, which joins to nothing without raising anywhere."""
    stem = "together_ai_openai_gpt-oss-20b_size9_comp0.0_1"
    _write_tree(tmp_path / "t", stem, 9, "grid")
    table = tmp_path / "table.csv"
    with open(table, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "step", "abs_pos", "token"])
        w.writerow([stem, 0, 3, "tok"])  # a token_idx, not an abs_pos
    with pytest.raises(SystemExit, match="nothing joined"):
        J.main(
            [
                "--table",
                str(table),
                "--lens-root",
                str(tmp_path / "t"),
                "--signal-name",
                "grid",
                "--out",
                str(tmp_path / "o.csv"),
            ]
        )


def test_it_writes_run_config_naming_the_ruler(tree_and_table, tmp_path):
    _, tree, table = tree_and_table
    out = tmp_path / "out" / "joined.csv"
    J.main(["--table", str(table), "--lens-root", str(tree), "--signal-name", "grid", "--out", str(out)])
    cfg = json.loads((out.parent / "run_config.json").read_text())
    assert cfg["measurement"]["loudness_column"] == "jlens_grid_logmass_L15"
    assert cfg["row_counts"]["joined"] == 3
