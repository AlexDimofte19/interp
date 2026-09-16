"""What the boundary experiment must not get wrong.

Seven things, one test each, all on synthetic trajectories so nothing here needs the GPU
host, the model, or the activation trees:

  * the six positions are found by CONTENT, not by a fixed offset into `output_tokens`
  * an incomplete or ambiguous boundary raises rather than being silently substituted
  * a triple contributes three independent `(D,)` samples, never one `3D` concatenation
  * every trajectory's samples stay in the partition its membership list puts it in
  * the cells carry the labels of that sample's own step
  * the padding class is excluded from the reported balanced accuracy
  * loudness joins to the probe rows on `token_idx`, not on `abs_pos`
  * counts-pooled balanced accuracy matches a hand-calculated example
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_module():
    """Import the adapter by path: `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location(
        "boundary_cognitive_maps", REPO / "scripts" / "boundary_cognitive_maps.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bcm = _load_module()


# --------------------------------------------------------------------------- fixtures


def _token(idx: int, text: str, token_id: int, groups: list[str]) -> dict:
    return {"id": idx, "token": text, "token_id": token_id, "token_groups": groups}


def _tokens(texts: list[str], groups: list[str]) -> list[dict]:
    ids = {"<|end|>": 200007, "<|start|>": 200006, "assistant": 173781}
    return [_token(i, t, ids.get(t, 1000 + i), groups) for i, t in enumerate(texts)]


GRID_5 = [
    "  0 1 2 3 4 ",
    "0 # # # # # ",
    "1 # _ _ G # ",
    "2 # _ # # # ",
    "3 # _ A _ # ",
    "4 # # # # # ",
]
GRID_5_ALT = [
    "  0 1 2 3 4 ",
    "0 # # # # # ",
    "1 # A _ _ # ",
    "2 # _ # # # ",
    "3 # _ G _ # ",
    "4 # # # # # ",
]


def make_trajectory(
    *,
    reasoning: list[str] | None = None,
    grid: list[str] | None = None,
    n_prefix: int = 4,
    n_grid_tokens: int = 6,
    extra_steps: list[list[str]] | None = None,
) -> dict:
    """A trajectory with the real shape: prefix, grid, the suffix triple, then the output.

    `reasoning` is the analysis chain; the `<|end|><|start|>assistant` triple and the final
    channel are appended after it, so the post boundary lands at a position that depends on
    the chain's length -- which is the whole point of locating it by content.
    """
    reasoning = ["We", " go", " left", "."] if reasoning is None else reasoning
    grids = [grid or GRID_5] + list(extra_steps or [])
    output = (
        ["<|channel|>", "analysis", "<|message|>"]
        + reasoning
        + ["<|end|>", "<|start|>", "assistant", "<|channel|>", "final", "<|message|>", "{", "LEFT", "}"]
    )
    steps = []
    for i, grid_state in enumerate(grids):
        steps.append(
            {
                "step_id": i,
                "grid_state": grid_state,
                "grid_state_tokens": _tokens([f"g{j}" for j in range(n_grid_tokens)], ["prompt", "grid_state"]),
                "output_tokens": _tokens(output, ["output"]),
                "agent_action": "LEFT",
            }
        )
    return {
        "grid_params": {"grid_width": 5, "grid_complexity": 0.0},
        "model_params": {"model_id": "openai/gpt-oss-20b"},
        "prompt": {
            "prompt_prefix_tokens": _tokens([f"p{j}" for j in range(n_prefix)], ["prompt"]),
            "prompt_suffix_tokens": _tokens(["<|end|>", "<|start|>", "assistant"], ["prompt", "template"]),
        },
        "steps": steps,
    }


# ------------------------------------------------------------------ boundary location


def test_boundaries_are_found_by_content_not_by_offset():
    """The post triple moves with the chain; both coordinate systems come out right."""
    short = make_trajectory(reasoning=["a", "."])
    long = make_trajectory(reasoning=["a", "b", "c", "d", "e", "."])

    for trajectory, chain in ((short, 2), (long, 6)):
        tokens = bcm.locate_boundaries(trajectory, 0)
        assert [t.boundary for t in tokens] == ["pre_reasoning"] * 3 + ["post_reasoning"] * 3
        assert [t.role for t in tokens] == list(bcm.BOUNDARY_ROLES) * 2
        assert [t.token for t in tokens] == list(bcm.BOUNDARY_TOKENS) * 2

        pre, post = tokens[:3], tokens[3:]
        # pre: the prompt suffix itself, category-relative 0,1,2
        assert [t.category for t in pre] == ["prompt_suffix"] * 3
        assert [t.token_idx for t in pre] == [0, 1, 2]
        # post: after `<|channel|>analysis<|message|>` plus the chain, inside `output`
        assert [t.category for t in post] == ["output"] * 3
        assert [t.token_idx for t in post] == [3 + chain, 4 + chain, 5 + chain]

        # abs_pos is prompt-inclusive: prefix + grid + (suffix for the output category)
        n_prefix = len(trajectory["prompt"]["prompt_prefix_tokens"])
        n_grid = len(trajectory["steps"][0]["grid_state_tokens"])
        assert [t.abs_pos for t in pre] == [n_prefix + n_grid + i for i in range(3)]
        assert [t.abs_pos for t in post] == [n_prefix + n_grid + 3 + t.token_idx for t in post]


def test_the_two_boundaries_are_the_same_three_token_ids():
    """Identical identities on both sides is what makes the pairing meaningful."""
    tokens = bcm.locate_boundaries(make_trajectory(), 0)
    assert [t.token_id for t in tokens[:3]] == [t.token_id for t in tokens[3:]]


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda t: t["prompt"]["prompt_suffix_tokens"].pop(), "prompt suffix"),
        (
            lambda t: t["steps"][0]["output_tokens"].__setitem__(7, _token(7, "nope", 42, ["output"])),
            "transitions",
        ),
    ],
)
def test_an_incomplete_or_ambiguous_boundary_raises(mutate, expected):
    """Never a silent substitution: a guessed boundary produces rows that look real."""
    trajectory = make_trajectory()
    mutate(trajectory)
    with pytest.raises(bcm.BoundaryError, match=expected):
        bcm.locate_boundaries(trajectory, 0)


def test_two_transitions_are_ambiguous_and_refused():
    trajectory = make_trajectory()
    trajectory["steps"][0]["output_tokens"] = _tokens(
        ["<|end|>", "<|start|>", "assistant", "x", "<|end|>", "<|start|>", "assistant"], ["output"]
    )
    with pytest.raises(bcm.BoundaryError, match="2 analysis->final transitions"):
        bcm.locate_boundaries(trajectory, 0)


def test_each_boundary_token_is_its_own_sample():
    """Three `(D,)` samples per triple, at three distinct positions -- not one `3D` row."""
    tokens = bcm.locate_boundaries(make_trajectory(), 0)
    for boundary in bcm.BOUNDARIES:
        side = [t for t in tokens if t.boundary == boundary]
        assert len(side) == 3
        assert len({t.token_idx for t in side}) == 3
        assert len({t.abs_pos for t in side}) == 3
    # and nothing in the record ties the three together
    assert len({(t.boundary, t.token_idx) for t in tokens}) == 6


# ---------------------------------------------------------------------------- cells


def test_cells_come_from_the_sample_s_own_step():
    """A two-step trajectory must not label step 1's tokens with step 0's grid."""
    trajectory = make_trajectory(grid=GRID_5, extra_steps=[GRID_5_ALT])
    first = bcm.boundary_cell_payload(
        trajectory, name="t", step_index=0, pad_to_size=15, max_cells=None, drop_padding=True, seed=42
    )
    second = bcm.boundary_cell_payload(
        trajectory, name="t", step_index=1, pad_to_size=15, max_cells=None, drop_padding=True, seed=42
    )
    agent_first = {tuple(p) for p, lab in zip(first["positions"], first["labels"], strict=True) if lab == 0}
    agent_second = {tuple(p) for p, lab in zip(second["positions"], second["labels"], strict=True) if lab == 0}
    assert agent_first == {(3, 2)}
    assert agent_second == {(1, 1)}


def test_padding_cells_are_dropped_before_the_draw():
    """`pad_to_size=15` on a 5-wide grid is 200 padding cells; none may be drawn."""
    trajectory = make_trajectory()
    padded = bcm.boundary_cell_payload(
        trajectory, name="t", step_index=0, pad_to_size=15, max_cells=None, drop_padding=False, seed=42
    )
    native = bcm.boundary_cell_payload(
        trajectory, name="t", step_index=0, pad_to_size=15, max_cells=None, drop_padding=True, seed=42
    )
    assert len(padded["labels"]) == 225
    assert 7 in padded["labels"]
    assert len(native["labels"]) == 25
    assert 7 not in native["labels"]
    assert max(max(p) for p in native["positions"]) < 5


def test_the_cell_draw_is_seeded_per_trajectory_and_step():
    """Two arms consume different numbers of draws; the cells must not depend on that."""
    trajectory = make_trajectory()
    kwargs = {"pad_to_size": 15, "max_cells": 10, "drop_padding": True, "seed": 42}
    a = bcm.boundary_cell_payload(trajectory, name="t", step_index=0, **kwargs)
    b = bcm.boundary_cell_payload(trajectory, name="t", step_index=0, **kwargs)
    other = bcm.boundary_cell_payload(trajectory, name="other", step_index=0, **kwargs)
    assert a == b
    assert a != other


# ----------------------------------------------------------------------- partitions


def test_partition_membership_is_taken_from_the_list(tmp_path):
    """A trajectory lands in the partition its list names, and nowhere else."""
    workspace = tmp_path
    (workspace / "splits").mkdir(parents=True)
    (workspace / "trajectories").mkdir(parents=True)
    (workspace / "splits/mass_train_2880.txt").write_text("a\nb\n")
    (workspace / "splits/mass_eval_720.txt").write_text("c\n")
    (workspace / "trajectories/heldout360_names.txt").write_text("d\n")
    parts = bcm.default_partitions(workspace)
    assert parts["train_2880"].names() == ["a", "b"]
    assert parts["eval_720"].names() == ["c"]
    assert parts["heldout_360"].names() == ["d"]
    everything = [n for p in parts.values() for n in p.names()]
    assert len(everything) == len(set(everything))


def test_find_trajectory_refuses_an_ambiguous_name(tmp_path):
    for size in ("size5", "size7"):
        (tmp_path / size).mkdir()
        (tmp_path / size / "t.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="resolves to 2 files"):
        bcm.find_trajectory(tmp_path, "t")


# ------------------------------------------------------------------------- loudness


def test_loudness_joins_on_token_idx_not_abs_pos():
    """The two coordinates differ by the prompt length; the wrong one joins to nothing."""
    tokens = bcm.locate_boundaries(make_trajectory(), 0)
    table = {bcm.loudness_key("t", 0, t.boundary, t.token_idx): float(i) for i, t in enumerate(tokens)}
    for i, t in enumerate(tokens):
        assert table[bcm.loudness_key("t", 0, t.boundary, t.token_idx)] == float(i)
        assert t.abs_pos != t.token_idx  # the prompt is non-empty, so they cannot coincide
        assert bcm.loudness_key("t", 0, t.boundary, t.abs_pos) not in table


def test_loudness_key_separates_the_two_boundaries():
    """`token_idx` 0 means a different token in each category, so the boundary is in the key."""
    assert bcm.loudness_key("t", 0, "pre_reasoning", 0) != bcm.loudness_key("t", 0, "post_reasoning", 0)


def test_read_loudness_round_trips_a_table(tmp_path):
    """Written by `csv.writer`, read back by `csv.DictReader` -- and a "NA" token survives."""
    from telos_interp.loudness_analysis import columns as cols

    column = cols.loudness_column("jlens", "grid", 15)
    path = tmp_path / "mass.csv"
    path.write_text(
        "name,size,complexity,step,boundary,role,category,token_idx,abs_pos,token,"
        f"{column},jlens_grid_prob_L15\n"
        "t,5,0.0,0,pre_reasoning,end,prompt_suffix,0,10,NA,-3.5,0.03\n"
        "t,5,0.0,0,post_reasoning,end,output,7,20,<|end|>,-1.25,0.28\n"
    )
    table = bcm.read_loudness(path, "jlens", "grid", 15)
    assert table[bcm.loudness_key("t", 0, "pre_reasoning", 0)] == -3.5
    assert table[bcm.loudness_key("t", 0, "post_reasoning", 7)] == -1.25


# ------------------------------------------------------------------------ statistics


def test_balanced_accuracy_pools_counts_and_excludes_padding():
    """Against a hand-calculated example.

    Two rows, classes 1 (wall) and 3 (empty) plus the padding class 7:

        row A: n_true_1 = 8, correct 4;  n_true_3 = 2, correct 2;  n_true_7 = 90, correct 90
        row B: n_true_1 = 2, correct 0;  n_true_3 = 8, correct 6;  n_true_7 = 90, correct 0

    Pooled over the two rows, ignoring padding:
        recall(1) = (4 + 0) / (8 + 2) = 0.4
        recall(3) = (2 + 6) / (2 + 8) = 0.8
        balanced accuracy = (0.4 + 0.8) / 2 = 0.6

    Averaging the two rows' own balanced accuracies instead gives
    ((0.5 + 1.0) / 2 + (0.0 + 0.75) / 2) / 2 = 0.5625 -- which is why the counts form
    exists and why `aggregation` is recorded. Including padding would give
    (0.4 + 0.8 + 0.5) / 3 = 0.5667, a third number again.
    """
    from telos_interp.loudness_analysis import stats
    from telos_interp.loudness_analysis.probes import get_probe_type

    ptype = get_probe_type("grid_tile")
    assert 7 not in ptype.analysis_classes  # the padding class is out of the reported metric

    df = pd.DataFrame(
        [
            {
                "name": "a",
                "n_cells": 100,
                "n_true_1": 8,
                "n_true_3": 2,
                "n_true_7": 90,
                "p_n_correct": 96,
                "p_correct_1": 4,
                "p_correct_3": 2,
                "p_correct_7": 90,
            },
            {
                "name": "b",
                "n_cells": 100,
                "n_true_1": 2,
                "n_true_3": 8,
                "n_true_7": 90,
                "p_n_correct": 6,
                "p_correct_1": 0,
                "p_correct_3": 6,
                "p_correct_7": 0,
            },
        ]
    )
    value, recalls = stats.bal_acc_from_counts(df, "p", ptype.analysis_classes)
    assert recalls[1] == pytest.approx(0.4)
    assert recalls[3] == pytest.approx(0.8)
    assert value == pytest.approx(0.6)
    assert 7 not in recalls

    with_padding, _ = stats.bal_acc_from_counts(df, "p", ptype.classes)
    assert with_padding == pytest.approx((0.4 + 0.8 + 0.5) / 3)
    assert stats.plain_accuracy(df, "p") == pytest.approx(0.51)


def test_summarise_reports_the_pooled_number_and_its_breakdowns():
    from telos_interp.loudness_analysis.probes import get_probe_type

    classes = get_probe_type("grid_tile").analysis_classes
    rows = []
    for i in range(6):
        rows.append(
            {
                "name": f"t{i}",
                "role": bcm.BOUNDARY_ROLES[i % 3],
                "size": 5,
                "complexity": 0.0,
                "n_cells": 10,
                "n_true_1": 5,
                "n_true_3": 5,
                "p_n_correct": 8,
                "p_correct_1": 4,
                "p_correct_3": 4,
            }
        )
    out = bcm.summarise(pd.DataFrame(rows), "p", classes, n_boot=20, seed=1)
    assert out["balanced_accuracy"] == pytest.approx(0.8)
    assert out["n_trajectories"] == 6
    assert set(out["by"]["role"]) == set(bcm.BOUNDARY_ROLES)
    assert out["per_class_recall"] == {"1": pytest.approx(0.8), "3": pytest.approx(0.8)}


def test_an_undefined_correlation_is_reported_as_undefined_not_as_zero():
    constant = pd.DataFrame({"x": [1.0, 1.0, 1.0, 1.0], "y": [0.1, 0.2, 0.3, 0.4]})
    assert bcm._spearman_or_none(constant, "x", "y")["undefined"] == "constant column"
    assert bcm._spearman_or_none(constant, "x", "y")["spearman"] is None

    tiny = pd.DataFrame({"x": [1.0, 2.0], "y": [0.1, 0.2]})
    assert bcm._spearman_or_none(tiny, "x", "y")["undefined"] == "insufficient samples"

    real = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, 3.0, 4.0]})
    assert bcm._spearman_or_none(real, "x", "y")["spearman"] == pytest.approx(1.0)


def test_a_boundary_table_is_read_back_one_probe_per_architecture(tmp_path):
    """Both architectures ride in one table; pairing has to match `lr` with `lr`."""
    import pandas as pd

    path = tmp_path / "per_token.csv"
    pd.DataFrame(
        [
            {
                "name": "t",
                "step": 0,
                "role": "end",
                "n_cells": 10,
                "n_true_1": 10,
                "pre_reasoning.boundary_pre_reasoning_l15_lr_n_correct": 4,
                "pre_reasoning.boundary_pre_reasoning_l15_lr_acc": 0.4,
                "pre_reasoning.boundary_pre_reasoning_l15_lr_correct_1": 4,
                "pre_reasoning.boundary_pre_reasoning_l15_mlp_n_correct": 7,
                "pre_reasoning.boundary_pre_reasoning_l15_mlp_acc": 0.7,
                "pre_reasoning.boundary_pre_reasoning_l15_mlp_correct_1": 7,
            }
        ]
    ).to_csv(path, index=False)

    df, by_arch = bcm._load_side(path)
    assert set(by_arch) == {"lr", "mlp"}
    assert by_arch["lr"].endswith("_lr")
    assert by_arch["mlp"].endswith("_mlp")
    # the denominators are shared -- which is the whole reason both probes sit in one table
    assert df["n_true_1"].tolist() == [10]
    assert bcm.probe_keys_of(df) == sorted(by_arch.values())


def test_architecture_is_read_off_the_probe_key():
    assert bcm.architecture_of("pre_reasoning.boundary_pre_reasoning_l15_lr") == "lr"
    assert bcm.architecture_of("post_reasoning.boundary_post_reasoning_l15_mlp") == "mlp"
    # the two boundaries' keys differ everywhere BUT the architecture field
    pre = "pre_reasoning.boundary_pre_reasoning_l15_mlp"
    post = "post_reasoning.boundary_post_reasoning_l15_mlp"
    assert pre != post
    assert bcm.architecture_of(pre) == bcm.architecture_of(post)


def test_a_table_with_two_probes_of_one_architecture_is_refused(tmp_path):
    """Silently keeping one of them would pair an arm against the wrong half."""
    import pandas as pd

    path = tmp_path / "per_token.csv"
    pd.DataFrame([{"name": "t", "a_l15_lr_n_correct": 1, "b_l15_lr_n_correct": 1}]).to_csv(path, index=False)
    with pytest.raises(SystemExit, match="two lr probes"):
        bcm._load_side(path)


def test_pairing_is_on_the_token_identity():
    """`<|end|>` pre pairs with `<|end|>` post, and an unpaired row is dropped, not filled."""
    column = "jlens_grid_logmass_L15"
    pre = pd.DataFrame(
        [
            {
                "name": "t",
                "step": 0,
                "role": r,
                "size": 5,
                "complexity": 0.0,
                "accuracy": 0.5,
                column: -3.0,
                "n_cells": 25,
            }
            for r in bcm.BOUNDARY_ROLES
        ]
    )
    post = pd.DataFrame(
        [{"name": "t", "step": 0, "role": r, "accuracy": 0.7, column: -1.0, "n_cells": 25} for r in ("end", "start")]
    )
    paired = bcm.paired_frame(pre, post, "p", "q", column)
    assert sorted(paired["role"]) == ["end", "start"]
    assert paired["delta_accuracy"].tolist() == pytest.approx([0.2, 0.2])
    assert paired["delta_loudness"].tolist() == pytest.approx([2.0, 2.0])


# -------------------------------------------------------------------------- manifest


def test_a_prepared_manifest_is_token_major_and_shares_one_cell_payload(tmp_path, monkeypatch):
    """Three entries per (trajectory, step), one `cells` payload, and no copied tensors."""
    import torch

    workspace = tmp_path / "ws"
    (workspace / "splits").mkdir(parents=True)
    (workspace / "trajectories").mkdir(parents=True)
    names = workspace / "splits/names.txt"
    names.write_text("t\n")
    trajectories = workspace / "trajectories/dir"
    (trajectories / "size5").mkdir(parents=True)
    trajectory = make_trajectory()
    (trajectories / "size5/t.json").write_text(json.dumps(trajectory))

    tree = tmp_path / "tree"
    tokens = bcm.locate_boundaries(trajectory, 0)
    for token in tokens:
        path = bcm.activation_path(tree, "size5", "t", 0, token, 15)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.zeros(8), path)
    index = bcm.index_path(tree, "size5", "t")
    index.write_text(
        json.dumps(
            {
                "name": "t",
                "steps": [
                    {
                        "step": 0,
                        "layer": 15,
                        "source_fingerprint": bcm.source_fingerprint(trajectory, 0),
                        "tokens": [t.__dict__ for t in tokens],
                    }
                ],
            }
        )
    )

    part = bcm.Partition("p", names, trajectories)
    manifest, skipped = bcm.build_manifest(
        partition=part,
        boundary="post_reasoning",
        tree_root=tree,
        layer=15,
        max_cells=10,
        pad_to_size=15,
        drop_padding=True,
        cell_seed=42,
        limit=None,
    )
    assert not skipped
    assert manifest["format_version"] == 3
    assert manifest["probe_type"] == "grid_tile"
    assert manifest["activation_dim"] == 8
    assert len(manifest["trajectories"]) == 3  # one entry per boundary token
    assert len(manifest["cells"]) == 1  # one payload per (trajectory, step)
    assert {e["cells_key"] for e in manifest["trajectories"]} == {"t|0"}
    assert {e["role"] for e in manifest["trajectories"]} == set(bcm.BOUNDARY_ROLES)
    assert {e["category"] for e in manifest["trajectories"]} == {"output"}
    assert manifest["activations_root"] == str(tree.resolve())
    assert not list(Path(manifest["activations_root"]).glob("*.pt"))  # nothing was copied
    assert manifest["split"]["num_trajectories"] == 1


def test_a_stale_sidecar_is_refused_rather_than_trusted(tmp_path):
    """Matching filenames are not enough: the ids the tensors came from must still hash the same."""
    import torch

    workspace = tmp_path / "ws"
    (workspace / "splits").mkdir(parents=True)
    names = workspace / "splits/names.txt"
    names.write_text("t\n")
    trajectories = workspace / "trajectories/dir"
    (trajectories / "size5").mkdir(parents=True)
    trajectory = make_trajectory()
    (trajectories / "size5/t.json").write_text(json.dumps(trajectory))

    tree = tmp_path / "tree"
    tokens = bcm.locate_boundaries(trajectory, 0)
    for token in tokens:
        path = bcm.activation_path(tree, "size5", "t", 0, token, 15)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.zeros(8), path)
    bcm.index_path(tree, "size5", "t").write_text(
        json.dumps({"name": "t", "steps": [{"step": 0, "layer": 15, "source_fingerprint": "stale", "tokens": []}]})
    )

    part = bcm.Partition("p", names, trajectories)
    with pytest.raises(SystemExit, match="no usable entries"):
        bcm.build_manifest(
            partition=part,
            boundary="post_reasoning",
            tree_root=tree,
            layer=15,
            max_cells=10,
            pad_to_size=15,
            drop_padding=True,
            cell_seed=42,
            limit=None,
        )
