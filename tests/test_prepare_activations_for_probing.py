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
