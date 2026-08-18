"""Tests for prepare_activations_for_probing v3 manifest output."""

import json
from pathlib import Path

import pytest
import torch

from telos_interp.commands.prepare_activations_for_probing.manifest_loader import (
    GridTileCompactDataset,
    IndexedGridTileCompactDataset,
    build_flat_grid_tile,
    detect_format,
    load_distance_compact,
    load_grid_tile_compact,
    load_v3_manifest,
)
from telos_interp.commands.prepare_activations_for_probing.prepare_activations_for_probing_fn import (
    PREPARED_FORMAT_VERSION,
    prepare_activations_for_probing,
)


def _grid_3x3() -> list[str]:
    """A tiny 3x3 grid: walls, an empty cell, an agent, a goal."""
    return [
        "  0 1 2 ",
        "0 # # # ",
        "1 # A G ",
        "2 # _ # ",
    ]


def _make_trajectory_json(
    *,
    grid: list[str] | None = None,
    astar_distance: int | None = None,
    actions: list[str] | None = None,
) -> dict:
    """Build a minimal trajectory JSON. `actions` lists per-step agent_action strings."""
    grid = grid if grid is not None else _grid_3x3()
    if actions is None:
        actions = ["RIGHT"]
    steps = []
    for action in actions:
        steps.append({"grid_state": grid, "agent_action": action})
    grid_params = {}
    if astar_distance is not None:
        grid_params["astar_distance"] = astar_distance
    return {"steps": steps, "grid_params": grid_params}


def _make_fixture(
    tmp_path: Path,
    *,
    num_trajectories: int = 3,
    activation_dim: int = 8,
    astar_distance: int = 4,
    actions: list[str] | None = None,
    grid: list[str] | None = None,
    nan_indices: tuple[int, ...] = (),
) -> tuple[Path, Path]:
    """Build an activation tree + matching trajectory JSONs in tmp_path.

    Layout:
      activations/{traj_name}/{model}/layer_0/step_0/output/0.pt   # (D,) float32
      trajectories/{traj_name}.json
    """
    activations_dir = tmp_path / "activations"
    trajectories_dir = tmp_path / "trajectories"
    activations_dir.mkdir()
    trajectories_dir.mkdir()

    actions = actions if actions is not None else ["RIGHT", "DOWN"]
    for i in range(num_trajectories):
        traj_name = f"traj_{i:04d}"
        # Activation file: a deterministic per-traj activation so we can spot duplication.
        torch.manual_seed(i)
        activation = torch.randn(activation_dim, dtype=torch.float32)
        if i in nan_indices:
            activation[0] = float("nan")
        out_dir = activations_dir / traj_name / "org__model" / "layer_0" / "step_0" / "output"
        out_dir.mkdir(parents=True)
        torch.save(activation, out_dir / "0.pt")

        # Trajectory JSON
        traj_json = _make_trajectory_json(
            grid=grid,
            astar_distance=astar_distance,
            actions=actions,
        )
        with open(trajectories_dir / f"{traj_name}.json", "w") as f:
            json.dump(traj_json, f)

    return activations_dir, trajectories_dir


def _make_multi_size_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Build a multi-size activation tree with size3 and size5 subfolders."""
    activations_dir = tmp_path / "activations"
    trajectories_dir = tmp_path / "trajectories"

    grid_3x3 = _grid_3x3()
    grid_5x5 = [
        "  0 1 2 3 4 ",
        "0 # # # # # ",
        "1 # A _ G # ",
        "2 # _ # _ # ",
        "3 # _ _ _ # ",
        "4 # # # # # ",
    ]

    for size_name, grid, num_trajectories in [("size3", grid_3x3, 2), ("size5", grid_5x5, 3)]:
        size_acts = activations_dir / size_name
        size_trajs = trajectories_dir / size_name
        size_acts.mkdir(parents=True)
        size_trajs.mkdir(parents=True)
        for i in range(num_trajectories):
            traj_name = f"traj_{i:04d}"
            torch.manual_seed(hash(size_name) ^ i)
            activation = torch.randn(8, dtype=torch.float32)
            out_dir = size_acts / traj_name / "org__model" / "layer_0" / "step_0" / "output"
            out_dir.mkdir(parents=True)
            torch.save(activation, out_dir / "0.pt")
            traj_json = _make_trajectory_json(grid=grid, astar_distance=2 + i, actions=["RIGHT"])
            with open(size_trajs / f"{traj_name}.json", "w") as f:
                json.dump(traj_json, f)

    return activations_dir, trajectories_dir


class TestGridTileSingleSize:
    def test_manifest_schema(self, tmp_path):
        activations_dir, trajectories_dir = _make_fixture(tmp_path, num_trajectories=3)
        out_dir = tmp_path / "out"

        prepare_activations_for_probing(
            activations_dir=str(activations_dir),
            trajectories_dir=str(trajectories_dir),
            probe_type="grid_tile",
            output_indices="-1",
            pad_to_size=3,
            output_path=str(out_dir),
            verbose=False,
        )

        manifest_path = out_dir / "manifest.json"
        assert manifest_path.exists(), "manifest.json was not written"
        manifest = json.loads(manifest_path.read_text())

        assert manifest["format_version"] == PREPARED_FORMAT_VERSION == 3
        assert manifest["probe_type"] == "grid_tile"
        assert manifest["activation_dim"] == 8
        assert manifest["num_cells_per_trajectory"] == 9  # 3x3 grid
        assert len(manifest["trajectories"]) == 3
        assert "loading_spec" in manifest and "config" in manifest

    def test_per_trajectory_files(self, tmp_path):
        activations_dir, trajectories_dir = _make_fixture(tmp_path, num_trajectories=3)
        out_dir = tmp_path / "out"

        prepare_activations_for_probing(
            activations_dir=str(activations_dir),
            trajectories_dir=str(trajectories_dir),
            probe_type="grid_tile",
            output_indices="-1",
            pad_to_size=3,
            output_path=str(out_dir),
            verbose=False,
        )

        manifest_path = out_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest["trajectories"]:
            assert "act_path" in entry
            act_file = manifest_path.parent / entry["act_path"]
            assert act_file.exists()
            tensor = torch.load(act_file, weights_only=True)
            assert tensor.shape == (8,)
            # No replication: each file holds one (D,) tensor only.

    def test_positions_and_labels_per_cell(self, tmp_path):
        activations_dir, trajectories_dir = _make_fixture(tmp_path, num_trajectories=2)
        out_dir = tmp_path / "out"

        prepare_activations_for_probing(
            activations_dir=str(activations_dir),
            trajectories_dir=str(trajectories_dir),
            probe_type="grid_tile",
            output_indices="-1",
            pad_to_size=3,
            output_path=str(out_dir),
            verbose=False,
        )

        manifest_path = out_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest["trajectories"]:
            assert len(entry["positions"]) == 9
            assert len(entry["labels"]) == 9
            for pos in entry["positions"]:
                assert isinstance(pos, list) and len(pos) == 2
                assert all(isinstance(p, int) for p in pos)
            assert all(isinstance(label, int) for label in entry["labels"])

    def test_compact_loader_roundtrip(self, tmp_path):
        activations_dir, trajectories_dir = _make_fixture(tmp_path, num_trajectories=4)
        out_dir = tmp_path / "out"

        prepare_activations_for_probing(
            activations_dir=str(activations_dir),
            trajectories_dir=str(trajectories_dir),
            probe_type="grid_tile",
            output_indices="-1",
            pad_to_size=3,
            output_path=str(out_dir),
            verbose=False,
        )

        manifest_path = out_dir / "manifest.json"
        manifest = load_v3_manifest(manifest_path)
        compact = load_grid_tile_compact(manifest, manifest_path)

        assert compact["base_act"].shape == (4, 8)
        assert compact["positions"].shape == (4, 9, 2)
        assert compact["labels"].shape == (4, 9)
        assert compact["C"] == 9
        assert compact["D"] == 8

    def test_compact_dataset_yields_full_input(self, tmp_path):
        activations_dir, trajectories_dir = _make_fixture(tmp_path, num_trajectories=2)
        out_dir = tmp_path / "out"

        prepare_activations_for_probing(
            activations_dir=str(activations_dir),
            trajectories_dir=str(trajectories_dir),
            probe_type="grid_tile",
            output_indices="-1",
            pad_to_size=3,
            output_path=str(out_dir),
            verbose=False,
        )

        manifest_path = out_dir / "manifest.json"
        manifest = load_v3_manifest(manifest_path)
        compact = load_grid_tile_compact(manifest, manifest_path)
        dataset = GridTileCompactDataset(compact["base_act"], compact["positions"], compact["labels"])

        assert len(dataset) == 2 * 9  # T * C
        x, y = dataset[0]
        assert x.shape == (8 + 2,)  # D + (row, col)
        assert x.dtype == torch.float32
        assert y.dtype == torch.int64

        # Same trajectory -> same first-D slice across cells.
        x0, _ = dataset[0]
        x1, _ = dataset[1]
        assert torch.allclose(x0[:8], x1[:8])  # both from trajectory 0
        # Different cells -> different positions.
        assert not torch.equal(x0[8:], x1[8:])

    def test_default_output_dir_when_none(self, tmp_path):
        activations_dir, trajectories_dir = _make_fixture(tmp_path, num_trajectories=2)

        prepare_activations_for_probing(
            activations_dir=str(activations_dir),
            trajectories_dir=str(trajectories_dir),
            probe_type="grid_tile",
            output_indices="-1",
            pad_to_size=3,
            verbose=False,
        )

        # Auto-named directory should be a child of activations_dir.
        children = [p for p in activations_dir.iterdir() if p.is_dir() and (p / "manifest.json").exists()]
        assert len(children) == 1, f"expected exactly one auto-named output dir, got {children}"

    def test_pt_suffix_stripped_with_warning(self, tmp_path, capsys):
        activations_dir, trajectories_dir = _make_fixture(tmp_path, num_trajectories=1)
        legacy_path = tmp_path / "out.pt"

        prepare_activations_for_probing(
            activations_dir=str(activations_dir),
            trajectories_dir=str(trajectories_dir),
            probe_type="grid_tile",
            output_indices="-1",
            pad_to_size=3,
            output_path=str(legacy_path),
            verbose=False,
        )

        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        # Stripped path is a directory.
        actual_out = tmp_path / "out"
        assert actual_out.is_dir()
        assert (actual_out / "manifest.json").exists()


class TestDistance:
    def test_manifest_schema(self, tmp_path):
        activations_dir, trajectories_dir = _make_fixture(tmp_path, num_trajectories=3, astar_distance=7)
        out_dir = tmp_path / "out"

        prepare_activations_for_probing(
            activations_dir=str(activations_dir),
            trajectories_dir=str(trajectories_dir),
            probe_type="distance",
            output_indices="-1",
            output_path=str(out_dir),
            verbose=False,
        )

        manifest = json.loads((out_dir / "manifest.json").read_text())
        assert manifest["probe_type"] == "distance"
        assert manifest["format_version"] == 3
        assert "num_cells_per_trajectory" not in manifest
        assert all("astar_distance" in e for e in manifest["trajectories"])
        assert all(e["astar_distance"] == 7 for e in manifest["trajectories"])

    def test_compact_loader(self, tmp_path):
        activations_dir, trajectories_dir = _make_fixture(tmp_path, num_trajectories=4, astar_distance=12)
        out_dir = tmp_path / "out"

        prepare_activations_for_probing(
            activations_dir=str(activations_dir),
            trajectories_dir=str(trajectories_dir),
            probe_type="distance",
            output_indices="-1",
            output_path=str(out_dir),
            verbose=False,
        )

        manifest_path = out_dir / "manifest.json"
        manifest = load_v3_manifest(manifest_path)
        compact = load_distance_compact(manifest, manifest_path)
        assert compact["base_act"].shape == (4, 8)
        assert compact["labels"].shape == (4,)
        assert compact["labels"].tolist() == [12, 12, 12, 12]

    def test_skip_when_no_astar_distance(self, tmp_path):
        activations_dir = tmp_path / "activations"
        trajectories_dir = tmp_path / "trajectories"
        activations_dir.mkdir()
        trajectories_dir.mkdir()
        for i in range(2):
            traj_name = f"traj_{i:04d}"
            out_dir = activations_dir / traj_name / "org__model" / "layer_0" / "step_0" / "output"
            out_dir.mkdir(parents=True)
            torch.save(torch.randn(8), out_dir / "0.pt")
            # No astar_distance in grid_params
            traj_json = {"steps": [{"grid_state": _grid_3x3(), "agent_action": "RIGHT"}], "grid_params": {}}
            with open(trajectories_dir / f"{traj_name}.json", "w") as f:
                json.dump(traj_json, f)

        out = tmp_path / "out"
        with pytest.raises(ValueError, match="No activations were extracted"):
            prepare_activations_for_probing(
                activations_dir=str(activations_dir),
                trajectories_dir=str(trajectories_dir),
                probe_type="distance",
                output_indices="-1",
                output_path=str(out),
                verbose=False,
            )


class TestActionSequence:
    def test_manifest_schema(self, tmp_path):
        actions = ["RIGHT", "DOWN", "LEFT"]
        activations_dir, trajectories_dir = _make_fixture(tmp_path, num_trajectories=2, actions=actions)
        out_dir = tmp_path / "out"

        prepare_activations_for_probing(
            activations_dir=str(activations_dir),
            trajectories_dir=str(trajectories_dir),
            probe_type="action_sequence",
            output_indices="-1",
            output_path=str(out_dir),
            verbose=False,
        )

        manifest = json.loads((out_dir / "manifest.json").read_text())
        assert manifest["probe_type"] == "action_sequence"
        assert manifest["max_seq_len"] == 3
        assert manifest["action_to_id"] == {"LEFT": 0, "TOP": 1, "RIGHT": 2, "DOWN": 3}
        for entry in manifest["trajectories"]:
            assert entry["actions"] == [2, 3, 0]  # RIGHT, DOWN, LEFT


class TestMultiSize:
    def test_manifest_with_size_field(self, tmp_path):
        activations_dir, trajectories_dir = _make_multi_size_fixture(tmp_path)
        out_dir = tmp_path / "out"

        prepare_activations_for_probing(
            activations_dir=str(activations_dir),
            trajectories_dir=str(trajectories_dir),
            probe_type="grid_tile",
            output_indices="-1",
            output_path=str(out_dir),
            verbose=False,
        )

        manifest_path = out_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assert manifest["sizes"] == ["size3", "size5"]
        assert "per_size_info" in manifest
        # auto-pad to max size = 5: each trajectory has 5x5 = 25 cells.
        assert manifest["num_cells_per_trajectory"] == 25
        # Trajectories have a `size` field.
        sizes_seen = {entry["size"] for entry in manifest["trajectories"]}
        assert sizes_seen == {3, 5}

    def test_per_size_subdirs(self, tmp_path):
        activations_dir, trajectories_dir = _make_multi_size_fixture(tmp_path)
        out_dir = tmp_path / "out"

        prepare_activations_for_probing(
            activations_dir=str(activations_dir),
            trajectories_dir=str(trajectories_dir),
            probe_type="grid_tile",
            output_indices="-1",
            output_path=str(out_dir),
            verbose=False,
        )

        # size3 and size5 each have their own subdir under activations/.
        assert (out_dir / "activations" / "size3").is_dir()
        assert (out_dir / "activations" / "size5").is_dir()

        manifest_path = out_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest["trajectories"]:
            act_file = manifest_path.parent / entry["act_path"]
            assert act_file.exists()
            assert f"size{entry['size']}" in entry["act_path"]


class TestDetectFormat:
    def test_v3_directory(self, tmp_path):
        activations_dir, trajectories_dir = _make_fixture(tmp_path, num_trajectories=1)
        out_dir = tmp_path / "out"
        prepare_activations_for_probing(
            activations_dir=str(activations_dir),
            trajectories_dir=str(trajectories_dir),
            probe_type="grid_tile",
            output_indices="-1",
            pad_to_size=3,
            output_path=str(out_dir),
            verbose=False,
        )
        assert detect_format(out_dir) == 3
        assert detect_format(out_dir / "manifest.json") == 3

    def test_v1_pt_file(self, tmp_path):
        legacy = tmp_path / "legacy.pt"
        torch.save({"activations": torch.zeros(2, 4), "labels": torch.zeros(2)}, legacy)
        assert detect_format(legacy) == 1


class TestBuildFlatGridTile:
    def test_shape(self):
        T, C, D = 3, 5, 8
        base_act = torch.randn(T, D)
        positions = torch.randint(0, 4, (T, C, 2), dtype=torch.int16)
        labels = torch.randint(0, 8, (T, C), dtype=torch.int64)

        flat_x, flat_y = build_flat_grid_tile(base_act, positions, labels)
        assert flat_x.shape == (T * C, D + 2)
        assert flat_y.shape == (T * C,)

    def test_activation_replicated_per_cell(self):
        T, C, D = 2, 3, 4
        base_act = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        )
        positions = torch.tensor([[[0, 0], [0, 1], [0, 2]], [[1, 0], [1, 1], [1, 2]]], dtype=torch.int16)
        labels = torch.zeros((T, C), dtype=torch.int64)

        flat_x, _ = build_flat_grid_tile(base_act, positions, labels)
        # Cells 0..2 should share the trajectory-0 activation.
        for c in range(C):
            assert torch.allclose(flat_x[c, :D], base_act[0])
        # Cells 3..5 should share the trajectory-1 activation.
        for c in range(C, 2 * C):
            assert torch.allclose(flat_x[c, :D], base_act[1])


class TestIndexedDataset:
    def test_only_yields_selected_pairs(self):
        T, C, D = 2, 3, 4
        base_act = torch.randn(T, D)
        positions = torch.tensor([[[0, 0], [0, 1], [0, 2]], [[1, 0], [1, 1], [1, 2]]], dtype=torch.int16)
        labels = torch.tensor([[3, 1, -1], [3, -1, 7]], dtype=torch.int64)
        # Only mask in cells with non-(-1) labels.
        valid_mask = labels >= 0
        flat_indices = torch.nonzero(valid_mask.reshape(-1), as_tuple=True)[0]
        dataset = IndexedGridTileCompactDataset(base_act, positions, labels, flat_indices)
        assert len(dataset) == 4  # 2 + 2 valid cells
        for i in range(len(dataset)):
            x, y = dataset[i]
            assert y.item() != -1


# --- jlens-driven next_action selection -------------------------------------------------

# Direction counts baked into the synthetic jlens CSV: a token's count is
# base[token] + layer_bonus[layer] + step_bonus[step], so the expected ranking is known.
_JLENS_TOKEN_BASE = {0: 3, 1: 5, 2: 1, 3: 7}
_JLENS_LAYER_BONUS = {15: 2, 10: 1, 20: 0}
_JLENS_STEP_BONUS = {0: 10, 1: 0}
_JLENS_LAYERS = (10, 15, 20)
_JLENS_STEPS = (0, 1)
_JLENS_ACTIONS = ("RIGHT", "DOWN")  # per step
_JLENS_N_PREFIX, _JLENS_N_GRID, _JLENS_N_SUFFIX = 3, 4, 2
_JLENS_OUTPUT_START = _JLENS_N_PREFIX + _JLENS_N_GRID + _JLENS_N_SUFFIX  # 9
_JLENS_N_ANALYSIS = 4  # analysis tokens per step; a 5th output token is the action


def _jlens_count(step: int, token_idx: int, layer: int) -> int:
    return _JLENS_TOKEN_BASE[token_idx] + _JLENS_LAYER_BONUS[layer] + _JLENS_STEP_BONUS[step]


def _make_jlens_fixture(
    tmp_path: Path,
    *,
    num_trajectories: int = 2,
    activation_dim: int = 4,
    write_csv: bool = True,
) -> tuple[Path, Path, Path]:
    """Activations + trajectory JSONs + per-trajectory jlens CSVs + a direction-token JSON.

    Mirrors what scripts/jlens_reasoning_tokens.py writes:
      activations/{traj}/org__model/layer_{L}/step_{S}/output/{out_idx}.pt
      activations/{traj}/{traj}_jlens_analysis.csv
    """
    import csv as _csv

    activations_dir = tmp_path / "activations"
    trajectories_dir = tmp_path / "trajectories"
    activations_dir.mkdir()
    trajectories_dir.mkdir()

    direction_path = tmp_path / "direction_tokens.json"
    direction_path.write_text(
        json.dumps({"UP": [" up"], "DOWN": [" down"], "LEFT": [" left"], "RIGHT": [" right"]})
    )
    direction_words = [" up", " down", " left", " right"]
    filler = [f"w{i}" for i in range(20)]

    header = (
        ["size", "complexity", "run", "step", "reasoning_pos", "abs_pos", "token", "layer", "agent_action"]
        + [f"{a}_rank" for a in ("RIGHT", "LEFT", "UP", "DOWN")]
        + [f"{a}_logprob" for a in ("RIGHT", "LEFT", "UP", "DOWN")]
        + [f"top_{i}" for i in range(1, 21)]
    )

    def _token(i: int, groups: list[str]) -> dict:
        return {"id": i, "token": f"t{i}", "token_id": 100 + i, "token_groups": groups}

    for traj_i in range(num_trajectories):
        traj_name = f"traj_{traj_i:04d}"

        # Activations: value encodes (layer, step, token) so we can identify the file loaded.
        for layer in _JLENS_LAYERS:
            for step in _JLENS_STEPS:
                out_dir = activations_dir / traj_name / "org__model" / f"layer_{layer}" / f"step_{step}" / "output"
                out_dir.mkdir(parents=True)
                for token_idx in range(_JLENS_N_ANALYSIS):
                    value = float(layer * 1000 + step * 100 + token_idx)
                    torch.save(torch.full((activation_dim,), value), out_dir / f"{token_idx}.pt")

        # Trajectory JSON, with the prompt/step token lists the abs_pos math needs.
        trajectory = {
            "prompt": {
                "prompt_prefix_tokens": [_token(i, ["prompt"]) for i in range(_JLENS_N_PREFIX)],
                "prompt_suffix_tokens": [_token(i, ["prompt"]) for i in range(_JLENS_N_SUFFIX)],
            },
            "grid_params": {"astar_distance": 3},
            "steps": [
                {
                    "step_id": step,
                    "grid_state": _grid_3x3(),
                    "grid_state_tokens": [_token(i, ["grid_state"]) for i in range(_JLENS_N_GRID)],
                    "agent_action": _JLENS_ACTIONS[step],
                    "output_tokens": [
                        _token(i, ["output", "analysis"]) for i in range(_JLENS_N_ANALYSIS)
                    ]
                    + [_token(_JLENS_N_ANALYSIS, ["output", "final", "action"])],
                }
                for step in _JLENS_STEPS
            ],
        }
        (trajectories_dir / f"{traj_name}.json").write_text(json.dumps(trajectory))

        if not write_csv:
            continue

        csv_path = activations_dir / traj_name / f"{traj_name}_jlens_analysis.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = _csv.writer(f)
            writer.writerow(header)
            for step in _JLENS_STEPS:
                for token_idx in range(_JLENS_N_ANALYSIS):
                    for layer in _JLENS_LAYERS:
                        hits = _jlens_count(step, token_idx, layer)
                        top = [direction_words[i % len(direction_words)] for i in range(hits)]
                        top += filler[: 20 - hits]
                        writer.writerow(
                            ["3", "0.0", str(traj_i), step, token_idx, _JLENS_OUTPUT_START + token_idx,
                             f"t{token_idx}", layer, _JLENS_ACTIONS[step]]
                            + [0] * 8
                            + top
                        )

    return activations_dir, trajectories_dir, direction_path


def _prepare_next_action(activations_dir, trajectories_dir, out_dir, **kwargs):
    """Run prepare with next_action defaults, returning the parsed manifest."""
    params = {
        "probe_type": "next_action",
        "layers": "all",
        "steps": "all",
        "output_indices": "all",
        "verbose": False,
    }
    params.update(kwargs)
    prepare_activations_for_probing(
        activations_dir=str(activations_dir),
        trajectories_dir=str(trajectories_dir),
        output_path=str(out_dir) if out_dir is not None else None,
        **params,
    )
    manifest_path = (Path(out_dir) if out_dir is not None else Path(activations_dir)) / "manifest.json"
    return json.loads(manifest_path.read_text())


class TestJlensTokenSelection:
    def test_top_tokens_and_layers(self, tmp_path):
        acts, trajs, directions = _make_jlens_fixture(tmp_path, num_trajectories=2)
        manifest = _prepare_next_action(
            acts, trajs, tmp_path / "out",
            token_selection="jlens_direction",
            layer_selection="jlens_direction",
            num_tokens=3,
            num_layers=2,
            direction_tokens_path=str(directions),
        )

        samples = manifest["samples"]
        assert len(samples) == 2 * 3 * 2  # trajectories x tokens x layers

        for name in ("traj_0000", "traj_0001"):
            mine = [s for s in samples if s["name"] == name]
            # Step 0 outscores step 1, and within a step the base counts rank 3 > 1 > 0.
            assert {(s["step"], s["token_id"]) for s in mine} == {(0, 3), (0, 1), (0, 0)}
            # Per token, layers 15 and 10 carry the highest counts (20 is always lowest).
            assert {s["layer"] for s in mine} == {10, 15}
            # Every sample from step 0 is labeled with step 0's action (RIGHT -> 2).
            assert {s["label"] for s in mine} == {2}

        counts = [s["direction_count"] for s in samples if s["name"] == "traj_0000"]
        assert counts == sorted(counts, reverse=True)
        expected_top = sum(_jlens_count(0, 3, layer) for layer in _JLENS_LAYERS)
        assert max(counts) == expected_top

    def test_act_paths_resolve_to_the_selected_token(self, tmp_path):
        acts, trajs, directions = _make_jlens_fixture(tmp_path, num_trajectories=1)
        manifest = _prepare_next_action(
            acts, trajs, tmp_path / "out",
            token_selection="jlens_direction",
            layer_selection="jlens_direction",
            num_tokens=2,
            num_layers=1,
            direction_tokens_path=str(directions),
        )
        root = Path(manifest["activations_root"])
        for sample in manifest["samples"]:
            path = root / sample["act_path"]
            assert path.exists(), path
            tensor = torch.load(path, weights_only=True)
            # abs_pos -> out_idx mapping must land on the file for this (layer, step, token).
            expected = float(sample["layer"] * 1000 + sample["step"] * 100 + sample["token_id"])
            assert tensor[0].item() == expected

    def test_labels_follow_the_token_own_step(self, tmp_path):
        acts, trajs, directions = _make_jlens_fixture(tmp_path, num_trajectories=1)
        manifest = _prepare_next_action(
            acts, trajs, tmp_path / "out",
            layers="15",
            token_selection="jlens_direction",
            num_tokens=8,  # every analysis token of both steps
            direction_tokens_path=str(directions),
        )
        samples = manifest["samples"]
        assert len(samples) == 8
        assert {s["label"] for s in samples if s["step"] == 0} == {2}  # RIGHT
        assert {s["label"] for s in samples if s["step"] == 1} == {3}  # DOWN

    def test_manual_layer_with_jlens_tokens(self, tmp_path):
        acts, trajs, directions = _make_jlens_fixture(tmp_path, num_trajectories=1)
        manifest = _prepare_next_action(
            acts, trajs, tmp_path / "out",
            layers="15",
            token_selection="jlens_direction",
            num_tokens=2,
            direction_tokens_path=str(directions),
        )
        assert {s["layer"] for s in manifest["samples"]} == {15}
        assert len(manifest["samples"]) == 2

    def test_layers_spec_bounds_the_candidate_pool(self, tmp_path):
        acts, trajs, directions = _make_jlens_fixture(tmp_path, num_trajectories=1)
        manifest = _prepare_next_action(
            acts, trajs, tmp_path / "out",
            layers="10,20",
            token_selection="jlens_direction",
            layer_selection="jlens_direction",
            num_tokens=2,
            num_layers=1,
            direction_tokens_path=str(directions),
        )
        # 15 scores highest but is outside the spec, so 10 (the next best) wins.
        assert {s["layer"] for s in manifest["samples"]} == {10}


class TestRandomSelection:
    def test_sizes_and_reproducibility(self, tmp_path):
        acts, trajs, _ = _make_jlens_fixture(tmp_path, num_trajectories=2)
        kwargs = dict(
            token_selection="random",
            layer_selection="random",
            num_tokens=3,
            num_layers=2,
        )
        first = _prepare_next_action(acts, trajs, tmp_path / "out_a", seed=1, **kwargs)
        again = _prepare_next_action(acts, trajs, tmp_path / "out_b", seed=1, **kwargs)
        other = _prepare_next_action(acts, trajs, tmp_path / "out_c", seed=7, **kwargs)

        def keys(manifest):
            return [(s["name"], s["step"], s["token_id"], s["layer"]) for s in manifest["samples"]]

        assert len(first["samples"]) == 2 * 3 * 2
        assert keys(first) == keys(again)
        assert keys(first) != keys(other)
        # No jlens CSV was read, so no direction bookkeeping is attached.
        assert "direction_count" not in first["samples"][0]

    def test_random_tokens_with_jlens_layers(self, tmp_path):
        acts, trajs, directions = _make_jlens_fixture(tmp_path, num_trajectories=1)
        manifest = _prepare_next_action(
            acts, trajs, tmp_path / "out",
            token_selection="random",
            layer_selection="jlens_direction",
            num_tokens=4,
            num_layers=1,
            direction_tokens_path=str(directions),
        )
        # Layer 15 always carries the highest count, whichever tokens were drawn.
        assert {s["layer"] for s in manifest["samples"]} == {15}
        assert len(manifest["samples"]) == 4

    def test_more_requested_than_available(self, tmp_path):
        acts, trajs, _ = _make_jlens_fixture(tmp_path, num_trajectories=1)
        manifest = _prepare_next_action(
            acts, trajs, tmp_path / "out",
            token_selection="random",
            layer_selection="random",
            num_tokens=99,
            num_layers=99,
        )
        # Degrades to everything on disk: 2 steps x 4 tokens x 3 layers.
        assert len(manifest["samples"]) == 2 * 4 * 3


class TestSelectionValidation:
    def test_defaults_are_unchanged(self, tmp_path):
        acts, trajs, _ = _make_jlens_fixture(tmp_path, num_trajectories=1)
        manifest = _prepare_next_action(acts, trajs, tmp_path / "out")
        assert len(manifest["samples"]) == 2 * 4 * 3
        assert manifest["selection"]["token_selection"] == "all"
        assert manifest["selection"]["layer_selection"] == "spec"

    def test_missing_csv_skips_trajectory(self, tmp_path):
        acts, trajs, directions = _make_jlens_fixture(tmp_path, num_trajectories=1, write_csv=False)
        with pytest.raises(ValueError, match="No activations were extracted"):
            _prepare_next_action(
                acts, trajs, tmp_path / "out",
                token_selection="jlens_direction",
                num_tokens=2,
                direction_tokens_path=str(directions),
            )

    def test_requires_num_tokens(self, tmp_path):
        acts, trajs, directions = _make_jlens_fixture(tmp_path, num_trajectories=1)
        with pytest.raises(ValueError, match="requires num_tokens"):
            _prepare_next_action(
                acts, trajs, tmp_path / "out",
                token_selection="jlens_direction",
                direction_tokens_path=str(directions),
            )

    def test_requires_direction_tokens_path(self, tmp_path):
        acts, trajs, _ = _make_jlens_fixture(tmp_path, num_trajectories=1)
        with pytest.raises(ValueError, match="requires direction_tokens_path"):
            _prepare_next_action(
                acts, trajs, tmp_path / "out",
                token_selection="jlens_direction",
                num_tokens=2,
            )

    def test_rejected_for_other_probe_types(self, tmp_path):
        acts, trajs, directions = _make_jlens_fixture(tmp_path, num_trajectories=1)
        with pytest.raises(ValueError, match="next_action"):
            prepare_activations_for_probing(
                activations_dir=str(acts),
                trajectories_dir=str(trajs),
                probe_type="distance",
                output_indices="all",
                token_selection="random",
                num_tokens=2,
                output_path=str(tmp_path / "out"),
            )

    def test_output_dirname_encodes_the_selection(self, tmp_path):
        acts, trajs, directions = _make_jlens_fixture(tmp_path, num_trajectories=1)
        prepare_activations_for_probing(
            activations_dir=str(acts),
            trajectories_dir=str(trajs),
            probe_type="next_action",
            layers="all",
            steps="all",
            output_indices="all",
            token_selection="jlens_direction",
            layer_selection="jlens_direction",
            num_tokens=3,
            num_layers=2,
            direction_tokens_path=str(directions),
        )
        matches = [p.name for p in acts.iterdir() if p.is_dir() and "next_action" in p.name]
        assert len(matches) == 1
        assert "_tokjl3_layjl2" in matches[0]
