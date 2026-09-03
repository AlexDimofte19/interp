"""Tests for prepare_activations_for_probing v3 manifest output."""

import json
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
        T, _C, D = 2, 3, 4
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
# log(0.01) per direction hit: 20 of them still sum to p=0.2, so a mass score stays a real
# (negative) log-probability the way it does on model output.
_HIT_LOGPROB = -4.6052


def _mass_of(hits: int) -> float:
    """The direction mass a row with `hits` equal hits carries: log(hits * 0.01).

    Deliberately the same number the top-k reader computes from those hits, so a fixture
    where the top-k window sees everything makes the two sources agree -- which is what
    isolates "the mass table is read correctly" from "the mass table sees more".
    """
    return math.log(hits * 0.01) if hits else -40.0


def _jlens_count(step: int, token_idx: int, layer: int) -> int:
    return _JLENS_TOKEN_BASE[token_idx] + _JLENS_LAYER_BONUS[layer] + _JLENS_STEP_BONUS[step]


def _make_jlens_fixture(
    tmp_path: Path,
    *,
    num_trajectories: int = 2,
    activation_dim: int = 4,
    write_csv: bool = True,
    lens: str = "jlens",
    logprobs: bool = False,
    mass_table: bool = False,
    step_grids: dict[int, list[str]] | None = None,
) -> tuple[Path, Path, Path]:
    """Activations + trajectory JSONs + per-trajectory jlens CSVs + a direction-token JSON.

    Mirrors what scripts/jlens_reasoning_tokens.py writes:
      activations/{traj}/org__model/layer_{L}/step_{S}/output/{out_idx}.pt
      activations/{traj}/{traj}_{lens}_analysis.csv

    `lens` names which method's CSV to write. The two lenses share a schema -- only the
    filename differs -- which is exactly why one scoring path serves both.

    `mass_table` additionally writes the wide direction-mass table, whose cells are the
    per-row mass the same `_jlens_count` hits imply -- so a `logprob_mass_full` selection
    lands on the same tokens a `logprob_mass` one does and the two can be compared.

    `logprobs` writes the current schema, with a `top_{i}_logprob` beside every `top_{i}`;
    left False it writes the pre-logprob schema, which is what every CSV on disk still is.
    Each hit gets the same logprob -- log(0.01), so the per-row total stays a real
    probability and a mass score stays negative and monotone in the hit count. The two modes
    therefore select the same tokens here: what is under test is the plumbing, not the
    ranking (tests/test_jlens_utils.py pins where the two diverge).
    """
    import csv as _csv

    activations_dir = tmp_path / "activations"
    trajectories_dir = tmp_path / "trajectories"
    activations_dir.mkdir()
    trajectories_dir.mkdir()

    direction_path = tmp_path / "direction_tokens.json"
    direction_path.write_text(json.dumps({"UP": [" up"], "DOWN": [" down"], "LEFT": [" left"], "RIGHT": [" right"]}))
    direction_words = [" up", " down", " left", " right"]
    filler = [f"w{i}" for i in range(20)]

    header = (
        ["size", "complexity", "run", "step", "reasoning_pos", "abs_pos", "token", "layer", "agent_action"]
        + [f"{a}_rank" for a in ("RIGHT", "LEFT", "UP", "DOWN")]
        + [f"{a}_logprob" for a in ("RIGHT", "LEFT", "UP", "DOWN")]
        + [f"top_{i}" for i in range(1, 21)]
        + ([f"top_{i}_logprob" for i in range(1, 21)] if logprobs else [])
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
                    "grid_state": (step_grids or {}).get(step, _grid_3x3()),
                    "grid_state_tokens": [_token(i, ["grid_state"]) for i in range(_JLENS_N_GRID)],
                    "agent_action": _JLENS_ACTIONS[step],
                    "output_tokens": [_token(i, ["output", "analysis"]) for i in range(_JLENS_N_ANALYSIS)]
                    + [_token(_JLENS_N_ANALYSIS, ["output", "final", "action"])],
                }
                for step in _JLENS_STEPS
            ],
        }
        (trajectories_dir / f"{traj_name}.json").write_text(json.dumps(trajectory))

        if not write_csv:
            continue

        from telos_interp.jlens_utils import METHODS

        csv_path = activations_dir / traj_name / f"{traj_name}{METHODS[lens].csv_suffix}"
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
                            [
                                "3",
                                "0.0",
                                str(traj_i),
                                step,
                                token_idx,
                                _JLENS_OUTPUT_START + token_idx,
                                f"t{token_idx}",
                                layer,
                                _JLENS_ACTIONS[step],
                            ]
                            + [0] * 8
                            + top
                            + ([_HIT_LOGPROB] * hits + [-25.0] * (20 - hits) if logprobs else [])
                        )

        if mass_table:
            from telos_interp.jlens_utils import mass_header

            mass_path = activations_dir / traj_name / f"{traj_name}{METHODS[lens].mass_suffix}"
            with open(mass_path, "w", newline="", encoding="utf-8") as f:
                writer = _csv.writer(f)
                writer.writerow(mass_header(list(_JLENS_LAYERS)))
                for step in _JLENS_STEPS:
                    for token_idx in range(_JLENS_N_ANALYSIS):
                        cells = [_mass_of(_jlens_count(step, token_idx, layer)) for layer in _JLENS_LAYERS]
                        writer.writerow(
                            [
                                3,
                                0.0,
                                traj_i,
                                step,
                                token_idx,
                                _JLENS_OUTPUT_START + token_idx,
                                f"t{token_idx}",
                                _JLENS_ACTIONS[step],
                            ]
                            + cells
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


class TestDirectionScoreModes:
    """The score is a registry now: `count`, or the lens' own logprobs weighted in.

    The failure that matters is silent -- a logprob mode reading a CSV with no logprob
    columns would score every row as "nothing matched" and rank every token by a tie, after
    a multi-hour prepare. It has to raise instead.
    """

    def test_logprob_mass_scores_and_records_itself(self, tmp_path):
        acts, trajs, directions = _make_jlens_fixture(tmp_path, num_trajectories=2, logprobs=True)
        manifest = _prepare_next_action(
            acts,
            trajs,
            tmp_path / "out",
            token_selection="jlens_direction",
            layer_selection="jlens_direction",
            num_tokens=2,
            num_layers=1,
            direction_tokens_path=str(directions),
            direction_score="logprob_mass",
        )
        assert manifest["selection"]["direction_score"] == "logprob_mass"
        scores = [s["direction_count"] for s in manifest["samples"]]
        assert scores and all(isinstance(x, float) and x < 0 for x in scores), "a logprob, not a count"
        assert all(isinstance(s["layer_direction_count"], float) for s in manifest["samples"])

    def test_the_two_modes_agree_when_the_logprobs_are_flat(self, tmp_path):
        """Same fixture, same picks: the plumbing changes the score, not the coordinates."""
        acts, trajs, directions = _make_jlens_fixture(tmp_path, num_trajectories=2, logprobs=True)
        common = {
            "token_selection": "jlens_direction",
            "layer_selection": "jlens_direction",
            "num_tokens": 2,
            "num_layers": 1,
            "direction_tokens_path": str(directions),
        }
        counted = _prepare_next_action(acts, trajs, tmp_path / "c", **common)
        massed = _prepare_next_action(acts, trajs, tmp_path / "m", direction_score="logprob_mass", **common)

        def picks(manifest):
            return sorted((s["name"], s["step"], s["token_id"], s["layer"]) for s in manifest["samples"])

        assert picks(counted) == picks(massed)

    def test_a_logprob_mode_refuses_a_pre_logprob_csv(self, tmp_path):
        acts, trajs, directions = _make_jlens_fixture(tmp_path, num_trajectories=1)
        with pytest.raises(ValueError, match="no top_i_logprob columns"):
            _prepare_next_action(
                acts,
                trajs,
                tmp_path / "out",
                token_selection="jlens_direction",
                num_tokens=2,
                direction_tokens_path=str(directions),
                direction_score="logprob_mass",
            )

    def test_a_recorded_selection_refuses_a_score_it_cannot_apply(self, tmp_path):
        acts, trajs, directions = _make_jlens_fixture(tmp_path, num_trajectories=1, logprobs=True)
        with pytest.raises(ValueError, match="reads scores from the selection record"):
            _prepare_next_action(
                acts,
                trajs,
                tmp_path / "out",
                token_selection="recorded_jlens",
                direction_tokens_path=str(directions),
                direction_score="logprob_mass",
            )

    def test_the_mass_table_selects_the_same_tokens_as_the_topk_mass(self, tmp_path):
        """Where the top-k window sees the whole vocabulary the two sources must agree, which
        is what separates 'the table is read right' from 'the table sees more'."""
        acts, trajs, directions = _make_jlens_fixture(tmp_path, num_trajectories=2, logprobs=True, mass_table=True)
        common = {
            "token_selection": "jlens_direction",
            "layer_selection": "jlens_direction",
            "num_tokens": 2,
            "num_layers": 1,
            "direction_tokens_path": str(directions),
        }
        topk = _prepare_next_action(acts, trajs, tmp_path / "t", direction_score="logprob_mass", **common)
        full = _prepare_next_action(acts, trajs, tmp_path / "f", direction_score="logprob_mass_full", **common)

        def picks(manifest):
            return sorted((s["name"], s["step"], s["token_id"], s["layer"]) for s in manifest["samples"])

        assert picks(topk) == picks(full)
        assert full["selection"]["direction_score"] == "logprob_mass_full"

    def test_a_mass_score_needs_no_logprob_columns(self, tmp_path):
        """It reads the table, so a pre-logprob analysis CSV beside it is irrelevant."""
        acts, trajs, directions = _make_jlens_fixture(tmp_path, num_trajectories=1, logprobs=False, mass_table=True)
        manifest = _prepare_next_action(
            acts,
            trajs,
            tmp_path / "out",
            token_selection="jlens_direction",
            num_tokens=2,
            direction_tokens_path=str(directions),
            direction_score="logprob_mass_full",
        )
        assert manifest["samples"]

    def test_a_mass_score_without_a_table_selects_nothing(self, tmp_path):
        """The analysis CSV cannot stand in for it -- the mass was never computed."""
        acts, trajs, directions = _make_jlens_fixture(tmp_path, num_trajectories=1, logprobs=True)
        with pytest.raises(ValueError, match="No activations were extracted"):
            _prepare_next_action(
                acts,
                trajs,
                tmp_path / "out",
                token_selection="jlens_direction",
                num_tokens=2,
                direction_tokens_path=str(directions),
                direction_score="logprob_mass_full",
            )

    def test_an_unknown_score_fails_before_any_work(self, tmp_path):
        acts, trajs, directions = _make_jlens_fixture(tmp_path, num_trajectories=1, logprobs=True)
        with pytest.raises(ValueError, match="Unknown direction_score"):
            _prepare_next_action(
                acts,
                trajs,
                tmp_path / "out",
                token_selection="jlens_direction",
                num_tokens=2,
                direction_tokens_path=str(directions),
                direction_score="logprob_maass",
            )


class TestJlensTokenSelection:
    def test_top_tokens_and_layers(self, tmp_path):
        acts, trajs, directions = _make_jlens_fixture(tmp_path, num_trajectories=2)
        manifest = _prepare_next_action(
            acts,
            trajs,
            tmp_path / "out",
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
            acts,
            trajs,
            tmp_path / "out",
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
            acts,
            trajs,
            tmp_path / "out",
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
            acts,
            trajs,
            tmp_path / "out",
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
            acts,
            trajs,
            tmp_path / "out",
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
        kwargs = {
            "token_selection": "random",
            "layer_selection": "random",
            "num_tokens": 3,
            "num_layers": 2,
        }
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
            acts,
            trajs,
            tmp_path / "out",
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
            acts,
            trajs,
            tmp_path / "out",
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
                acts,
                trajs,
                tmp_path / "out",
                token_selection="jlens_direction",
                num_tokens=2,
                direction_tokens_path=str(directions),
            )

    def test_requires_num_tokens(self, tmp_path):
        acts, trajs, directions = _make_jlens_fixture(tmp_path, num_trajectories=1)
        with pytest.raises(ValueError, match="requires num_tokens"):
            _prepare_next_action(
                acts,
                trajs,
                tmp_path / "out",
                token_selection="jlens_direction",
                direction_tokens_path=str(directions),
            )

    def test_requires_direction_tokens_path(self, tmp_path):
        acts, trajs, _ = _make_jlens_fixture(tmp_path, num_trajectories=1)
        with pytest.raises(ValueError, match="requires direction_tokens_path"):
            _prepare_next_action(
                acts,
                trajs,
                tmp_path / "out",
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


# --- reading a pruned tree's selection record -------------------------------------------
#
# Once delete_non_jlens_selected.py has run, re-scoring the CSV would still find the right
# jlens tokens but could no longer produce an honest control: a uniform draw over the
# survivors is not a uniform draw over the reasoning chain. Both arms therefore come from
# the record the pruning wrote.


def _prune_fixture(activations_dir, trajectories_dir, direction_path, **overrides):
    """Run the real pruner over a jlens fixture, leaving a selection record behind."""
    import importlib

    dnjs = importlib.import_module("scripts.delete_non_jlens_selected")
    opts = {
        "--select-num-tokens": "3",
        "--select-num-layers": "2",
        "--select-always-layers": "",
        "--select-random-tokens": "2",
        "--select-seed": "42",
    }
    opts.update(overrides)
    argv = [
        "delete_non_jlens_selected.py",
        "--activations-dir",
        str(activations_dir),
        "--trajectories-dir",
        str(trajectories_dir),
        "--signal-json",
        str(direction_path),
        "--apply",
        *[part for pair in opts.items() for part in pair],
    ]
    old, sys.argv = sys.argv, argv
    try:
        dnjs.main()
    finally:
        sys.argv = old


class TestRecordedSelection:
    def test_recorded_jlens_matches_an_unpruned_jlens_run(self, tmp_path):
        """The point of the record: pruning must not change which samples a probe gets."""
        (tmp_path / "full").mkdir()
        (tmp_path / "pruned").mkdir()
        full_acts, full_trajs, directions = _make_jlens_fixture(tmp_path / "full")
        acts, trajs, _ = _make_jlens_fixture(tmp_path / "pruned")

        reference = _prepare_next_action(
            full_acts,
            full_trajs,
            tmp_path / "ref",
            token_selection="jlens_direction",
            layer_selection="jlens_direction",
            num_tokens=3,
            num_layers=2,
            direction_tokens_path=str(directions),
        )
        _prune_fixture(acts, trajs, directions)
        recorded = _prepare_next_action(
            acts,
            trajs,
            tmp_path / "rec",
            token_selection="recorded_jlens",
        )

        def identity(manifest):
            return sorted(
                (
                    s["name"],
                    s["step"],
                    s["token_id"],
                    s["layer"],
                    s["label"],
                    s["direction_count"],
                    s["layer_direction_count"],
                )
                for s in manifest["samples"]
            )

        assert identity(recorded) == identity(reference)

    def test_recorded_random_is_countless(self, tmp_path):
        """No counts means split_next_action_manifest samples the control instead of ranking it."""
        acts, trajs, directions = _make_jlens_fixture(tmp_path)
        _prune_fixture(acts, trajs, directions)
        manifest = _prepare_next_action(acts, trajs, tmp_path / "out", token_selection="recorded_random")
        assert manifest["samples"]
        for sample in manifest["samples"]:
            assert "direction_count" not in sample
            assert "layer_direction_count" not in sample

    def test_recorded_jlens_keeps_counts(self, tmp_path):
        acts, trajs, directions = _make_jlens_fixture(tmp_path)
        _prune_fixture(acts, trajs, directions)
        manifest = _prepare_next_action(acts, trajs, tmp_path / "out", token_selection="recorded_jlens")
        assert all("direction_count" in s for s in manifest["samples"])
        assert all("layer_direction_count" in s for s in manifest["samples"])

    def test_num_tokens_caps_the_recorded_arm(self, tmp_path):
        acts, trajs, directions = _make_jlens_fixture(tmp_path, num_trajectories=1)
        _prune_fixture(acts, trajs, directions)
        full = _prepare_next_action(acts, trajs, tmp_path / "a", token_selection="recorded_jlens")
        capped = _prepare_next_action(acts, trajs, tmp_path / "b", token_selection="recorded_jlens", num_tokens=1)
        assert len({(s["step"], s["token_id"]) for s in full["samples"]}) == 3
        assert len({(s["step"], s["token_id"]) for s in capped["samples"]}) == 1
        # and it caps by rank, not arbitrarily
        assert max(s["direction_count"] for s in full["samples"]) == capped["samples"][0]["direction_count"]

    def test_layers_narrows_the_recorded_selection(self, tmp_path):
        """A single-layer dataset out of the same record, with no re-gather."""
        acts, trajs, directions = _make_jlens_fixture(tmp_path, num_trajectories=1)
        _prune_fixture(acts, trajs, directions)
        manifest = _prepare_next_action(acts, trajs, tmp_path / "out", token_selection="recorded_jlens", layers="15")
        assert manifest["samples"]
        assert {s["layer"] for s in manifest["samples"]} == {15}

    def test_missing_record_skips_the_trajectory(self, tmp_path):
        acts, trajs, _ = _make_jlens_fixture(tmp_path, num_trajectories=1)
        with pytest.raises(ValueError, match="No activations were extracted"):
            _prepare_next_action(acts, trajs, tmp_path / "out", token_selection="recorded_jlens")

    def test_layer_selection_must_stay_spec(self, tmp_path):
        acts, trajs, directions = _make_jlens_fixture(tmp_path, num_trajectories=1)
        _prune_fixture(acts, trajs, directions)
        with pytest.raises(ValueError, match="must stay 'spec'"):
            _prepare_next_action(
                acts,
                trajs,
                tmp_path / "out",
                token_selection="recorded_jlens",
                layer_selection="random",
                num_layers=1,
            )

    def test_recorded_modes_need_no_direction_tokens_path(self, tmp_path):
        """The scoring already happened; re-supplying the vocabulary would be misleading."""
        acts, trajs, directions = _make_jlens_fixture(tmp_path, num_trajectories=1)
        _prune_fixture(acts, trajs, directions)
        manifest = _prepare_next_action(acts, trajs, tmp_path / "out", token_selection="recorded_jlens")
        assert manifest["selection"]["token_selection"] == "recorded_jlens"
        assert manifest["selection"]["direction_tokens_path"] is None

    def test_the_two_arms_are_recorded_separately(self, tmp_path):
        acts, trajs, directions = _make_jlens_fixture(tmp_path, num_trajectories=1)
        _prune_fixture(acts, trajs, directions, **{"--select-random-tokens": "4"})
        jlens = _prepare_next_action(acts, trajs, tmp_path / "j", token_selection="recorded_jlens")
        control = _prepare_next_action(acts, trajs, tmp_path / "r", token_selection="recorded_random")
        assert len({(s["step"], s["token_id"]) for s in jlens["samples"]}) == 3
        assert len({(s["step"], s["token_id"]) for s in control["samples"]}) == 4


class TestLogitLensSelection:
    """The logit lens goes through the same code as the jlens, keyed only by method name.

    So what is worth pinning is not the ranking -- `tests/test_jlens_utils.py` covers that
    once for both -- but that the mode names resolve to the right CSV and the right arm.
    """

    def test_logitlens_direction_matches_jlens_direction_on_the_same_numbers(self, tmp_path):
        """Identical CSV contents under the other lens' filename must give identical samples."""
        (tmp_path / "j").mkdir()
        (tmp_path / "l").mkdir()
        j_acts, j_trajs, directions = _make_jlens_fixture(tmp_path / "j", lens="jlens")
        l_acts, l_trajs, _ = _make_jlens_fixture(tmp_path / "l", lens="logitlens")

        common = {
            "num_tokens": 3,
            "num_layers": 2,
            "direction_tokens_path": str(directions),
        }
        by_jlens = _prepare_next_action(
            j_acts,
            j_trajs,
            tmp_path / "out_j",
            token_selection="jlens_direction",
            layer_selection="jlens_direction",
            **common,
        )
        by_logit = _prepare_next_action(
            l_acts,
            l_trajs,
            tmp_path / "out_l",
            token_selection="logitlens_direction",
            layer_selection="logitlens_direction",
            **common,
        )

        def identity(manifest):
            return sorted(
                (s["name"], s["step"], s["token_id"], s["layer"], s["direction_count"]) for s in manifest["samples"]
            )

        assert identity(by_jlens) == identity(by_logit)
        assert identity(by_jlens), "the selection cannot legitimately be empty"
        assert by_logit["selection"]["method"] == "logitlens"

    def test_logitlens_direction_needs_the_logitlens_csv(self, tmp_path):
        """A jlens-only tree selects nothing for the other lens rather than scoring its CSV.

        Selecting nothing is prepare's existing "no activations were extracted" error, which
        is the right outcome: an empty probe dataset should stop the pipeline, not be written.
        """
        acts, trajs, directions = _make_jlens_fixture(tmp_path, lens="jlens")
        with pytest.raises(ValueError, match="No activations were extracted"):
            _prepare_next_action(
                acts,
                trajs,
                tmp_path / "out",
                token_selection="logitlens_direction",
                num_tokens=3,
                direction_tokens_path=str(directions),
            )

    def test_recorded_logitlens_reads_that_arm(self, tmp_path):
        from telos_interp.jlens_utils import (
            KeptTokens,
            TokenPick,
            build_record,
            read_selection_record,
            record_path,
            write_selection_record,
        )

        acts, trajs, directions = _make_jlens_fixture(tmp_path, num_trajectories=1)
        _prune_fixture(acts, trajs, directions)
        folder = acts / "traj_0000"

        # Add a logitlens arm the way jlens_reasoning_tokens.py --extend would.
        existing, _ = read_selection_record(record_path(folder))
        step, token_idx = 0, 1
        arm = {
            (step, token_idx): TokenPick(
                step,
                token_idx,
                (_JLENS_LAYERS[0],),
                token="t1",
                direction_count=9,
                layer_direction_counts={_JLENS_LAYERS[0]: 9},
            )
        }
        merged = KeptTokens({**existing.arms, "logitlens": arm})
        write_selection_record(
            record_path(folder),
            build_record(merged, stem=folder.name, model="org__model", config={"seed": 42}),
        )

        manifest = _prepare_next_action(
            acts,
            trajs,
            tmp_path / "out",
            token_selection="recorded_logitlens",
        )
        assert [(s["step"], s["token_id"], s["layer"]) for s in manifest["samples"]] == [
            (step, token_idx, _JLENS_LAYERS[0])
        ]
        assert manifest["samples"][0]["direction_count"] == 9

    def test_an_arm_the_record_lacks_yields_nothing(self, tmp_path, capsys):
        """A tree pruned before the logit lens existed cannot serve that arm.

        It has to be skipped rather than silently falling back to another arm -- the whole
        point of the recorded modes is that the arm you name is the arm you get.
        """
        acts, trajs, directions = _make_jlens_fixture(tmp_path, num_trajectories=1)
        _prune_fixture(acts, trajs, directions)

        with pytest.raises(ValueError, match="No activations were extracted"):
            _prepare_next_action(
                acts,
                trajs,
                tmp_path / "out",
                token_selection="recorded_logitlens",
                verbose=True,
            )
        assert "has no 'logitlens' arm" in capsys.readouterr().out

    def test_the_two_axes_cannot_name_different_lenses(self, tmp_path):
        acts, trajs, directions = _make_jlens_fixture(tmp_path, num_trajectories=1)
        with pytest.raises(ValueError, match="name different lenses"):
            _prepare_next_action(
                acts,
                trajs,
                tmp_path / "out",
                token_selection="jlens_direction",
                layer_selection="logitlens_direction",
                num_tokens=2,
                num_layers=1,
                direction_tokens_path=str(directions),
            )

    def test_unknown_mode_lists_the_registered_ones(self, tmp_path):
        acts, trajs, _ = _make_jlens_fixture(tmp_path, num_trajectories=1)
        with pytest.raises(ValueError, match="recorded_logitlens"):
            _prepare_next_action(
                acts,
                trajs,
                tmp_path / "out",
                token_selection="lensy_direction",
                num_tokens=2,
            )

    def test_each_mode_gets_its_own_auto_named_directory(self, tmp_path):
        """The abbreviation comes from the registry, so a new method cannot KeyError here.

        The previous hand-maintained lookup is exactly what produced a KeyError the last time
        a mode was added.
        """
        acts, trajs, directions = _make_jlens_fixture(tmp_path, num_trajectories=1, lens="logitlens")
        prepare_activations_for_probing(
            activations_dir=str(acts),
            trajectories_dir=str(trajs),
            output_path=None,
            probe_type="next_action",
            layers="all",
            steps="all",
            output_indices="all",
            verbose=False,
            token_selection="logitlens_direction",
            num_tokens=2,
            direction_tokens_path=str(directions),
        )
        # "ll" is METHODS["logitlens"].abbrev; "2" is num_tokens
        assert [d.name for d in acts.glob("*next_action_tokll2")], sorted(p.name for p in acts.iterdir())


# --- grid_tile on lens-selected tokens ---------------------------------------------------
#
# The grid arms exist to answer the same question as the next_action arms about a different
# label: does the lens pick tokens that carry more of the model's world model, not just more
# of the decision it already verbalized? That only works if the two probe types read exactly
# the same tokens at exactly the same layers, and are scored on exactly the same cells. Both
# of those are properties of prepare, so both are pinned here.


def _prepare_grid_tokens(activations_dir, trajectories_dir, out_dir, **kwargs):
    """Run prepare with the token-major grid_tile defaults, returning the parsed manifest."""
    params = {
        "probe_type": "grid_tile",
        "layers": "all",
        "steps": "all",
        "output_indices": "all",
        "verbose": False,
    }
    params.update(kwargs)
    prepare_activations_for_probing(
        activations_dir=str(activations_dir),
        trajectories_dir=str(trajectories_dir),
        output_path=str(out_dir),
        **params,
    )
    return json.loads((Path(out_dir) / "manifest.json").read_text())


_GRID_ARM = {
    "token_selection": "jlens_direction",
    "layer_selection": "jlens_direction",
    "num_tokens": 3,
    "num_layers": 2,
}


class TestGridTileTokenSelection:
    def test_selects_the_same_tokens_as_the_action_arm(self, tmp_path):
        """The whole comparison rests on this: same files, different label."""
        acts, trajs, directions = _make_jlens_fixture(tmp_path, num_trajectories=2)
        arm = dict(_GRID_ARM, direction_tokens_path=str(directions))
        action = _prepare_next_action(acts, trajs, tmp_path / "action", **arm)
        grid = _prepare_grid_tokens(acts, trajs, tmp_path / "grid", **arm)

        def refs(entries):
            return sorted((e["name"], e["step"], e["token_id"], e["layer"], e["act_path"]) for e in entries)

        assert refs(grid["trajectories"]) == refs(action["samples"])

    def test_is_token_major_and_copies_nothing(self, tmp_path):
        acts, trajs, directions = _make_jlens_fixture(tmp_path, num_trajectories=2)
        out = tmp_path / "grid"
        manifest = _prepare_grid_tokens(acts, trajs, out, direction_tokens_path=str(directions), **_GRID_ARM)

        entries = manifest["trajectories"]
        assert len(entries) == 2 * 3 * 2, "trajectories x tokens x layers"
        assert len({e["name"] for e in entries}) == 2
        assert manifest["activations_root"] == str(acts.resolve())
        assert not (out / "activations").exists(), "token-major mode copies nothing"
        assert manifest["num_cells_per_trajectory"] == 9
        assert manifest["selection"]["token_selection"] == "jlens_direction"

    def test_cells_are_stored_once_per_trajectory_step(self, tmp_path):
        """Repeating 225 positions on each of a trajectory's ~20 entries is a 20x manifest."""
        acts, trajs, directions = _make_jlens_fixture(tmp_path, num_trajectories=2)
        manifest = _prepare_grid_tokens(
            acts, trajs, tmp_path / "grid", direction_tokens_path=str(directions), **_GRID_ARM
        )
        entries = manifest["trajectories"]

        assert set(manifest["cells"]) == {f"{e['name']}|{e['step']}" for e in entries}
        assert len(manifest["cells"]) < len(entries)
        assert all(e["cells_key"] in manifest["cells"] for e in entries)

    def test_the_grid_follows_the_token_own_step(self, tmp_path):
        """A step-1 token labeled with step 0's grid would be silent and wrong."""
        moved = [
            "  0 1 2 ",
            "0 _ _ _ ",
            "1 _ A G ",
            "2 # # # ",
        ]
        acts, trajs, directions = _make_jlens_fixture(tmp_path, num_trajectories=1, step_grids={1: moved})
        manifest = _prepare_grid_tokens(
            acts,
            trajs,
            tmp_path / "grid",
            layers="15",
            token_selection="jlens_direction",
            num_tokens=8,  # every analysis token of both steps
            direction_tokens_path=str(directions),
        )
        labels_by_step = {e["step"]: manifest["cells"][e["cells_key"]]["labels"] for e in manifest["trajectories"]}
        assert set(labels_by_step) == {0, 1}
        assert labels_by_step[0] != labels_by_step[1]

    def test_cell_draw_does_not_depend_on_the_arm(self, tmp_path):
        """Arms must differ in tokens only. Cells drawn from a shared global RNG would not:
        each arm consumes a different number of draws, so the same seed hands them different
        cells and the arms stop being comparable."""
        acts, trajs, directions = _make_jlens_fixture(tmp_path, num_trajectories=2)
        common = {
            "direction_tokens_path": str(directions),
            "max_positions_per_trajectory": 4,
        }
        ranked = _prepare_grid_tokens(
            acts, trajs, tmp_path / "a", token_selection="jlens_direction", num_tokens=1, **common
        )
        control = _prepare_grid_tokens(acts, trajs, tmp_path / "b", token_selection="random", num_tokens=3, **common)

        shared = set(ranked["cells"]) & set(control["cells"])
        assert shared, "the two arms cover some of the same trajectories"
        for key in shared:
            assert ranked["cells"][key] == control["cells"][key]

    def test_loader_reads_it_back(self, tmp_path):
        acts, trajs, directions = _make_jlens_fixture(tmp_path, num_trajectories=2)
        out = tmp_path / "grid"
        manifest = _prepare_grid_tokens(acts, trajs, out, direction_tokens_path=str(directions), **_GRID_ARM)
        compact = load_grid_tile_compact(manifest, out / "manifest.json")
        entries = manifest["trajectories"]

        assert compact["base_act"].shape == (len(entries), 4)
        assert compact["C"] == 9
        assert compact["trajectory_names"] == [e["name"] for e in entries]
        for i, entry in enumerate(entries):
            # the fixture encodes (layer, step, token) in the value, so a mis-resolved
            # act_path shows up as the wrong number rather than as a missing file
            expected = float(entry["layer"] * 1000 + entry["step"] * 100 + entry["token_id"])
            assert compact["base_act"][i][0].item() == expected
        first = {e["name"]: i for i, e in reversed(list(enumerate(entries)))}
        for i, entry in enumerate(entries):
            assert torch.equal(compact["labels"][i], compact["labels"][first[entry["name"]]])

    def test_without_a_selection_it_stays_trajectory_major(self, tmp_path):
        """Every dataset prepared before selection existed must read back unchanged."""
        acts, trajs = _make_fixture(tmp_path, num_trajectories=3)
        out = tmp_path / "classic"
        prepare_activations_for_probing(
            activations_dir=str(acts),
            trajectories_dir=str(trajs),
            probe_type="grid_tile",
            layers="all",
            output_indices="all",
            output_path=str(out),
            verbose=False,
        )
        manifest = json.loads((out / "manifest.json").read_text())
        assert len(manifest["trajectories"]) == 3
        assert "cells" not in manifest
        assert "activations_root" not in manifest
        assert (out / "activations").is_dir(), "the classic path still copies"

    def test_still_rejected_for_probe_types_that_have_no_tokens(self, tmp_path):
        acts, trajs, directions = _make_jlens_fixture(tmp_path, num_trajectories=1)
        with pytest.raises(ValueError, match="next_action"):
            prepare_activations_for_probing(
                activations_dir=str(acts),
                trajectories_dir=str(trajs),
                probe_type="action_sequence",
                output_indices="all",
                token_selection="random",
                num_tokens=2,
                output_path=str(tmp_path / "out"),
            )


class TestGridTileTokenMajorFlag:
    """The EOS control has no selection record to read: its arm is 'every gathered token'."""

    def test_flag_makes_it_token_major_without_a_selection(self, tmp_path):
        acts, trajs, _ = _make_jlens_fixture(tmp_path, num_trajectories=2, write_csv=False)
        manifest = _prepare_grid_tokens(acts, trajs, tmp_path / "eos", layers="15", token_major=True)

        entries = manifest["trajectories"]
        # 2 trajectories x 2 steps x 4 analysis tokens, all at layer 15
        assert len(entries) == 2 * 2 * 4
        assert {e["layer"] for e in entries} == {15}
        assert manifest["activations_root"] == str(acts.resolve())
        assert all("cells_key" in e for e in entries)

    def test_off_by_default(self, tmp_path):
        acts, trajs, _ = _make_jlens_fixture(tmp_path, num_trajectories=2, write_csv=False)
        manifest = _prepare_grid_tokens(acts, trajs, tmp_path / "classic", layers="15")
        assert len(manifest["trajectories"]) == 2
        assert "cells" not in manifest

    def test_rejected_for_other_probe_types(self, tmp_path):
        acts, trajs, _ = _make_jlens_fixture(tmp_path, num_trajectories=1, write_csv=False)
        with pytest.raises(ValueError, match="token_major applies to"):
            prepare_activations_for_probing(
                activations_dir=str(acts),
                trajectories_dir=str(trajs),
                probe_type="distance",
                output_indices="all",
                token_major=True,
                output_path=str(tmp_path / "out"),
            )

    def test_needs_output_indices(self, tmp_path):
        acts, trajs, _ = _make_jlens_fixture(tmp_path, num_trajectories=1, write_csv=False)
        with pytest.raises(ValueError, match="output_indices"):
            prepare_activations_for_probing(
                activations_dir=str(acts),
                trajectories_dir=str(trajs),
                probe_type="grid_tile",
                layers="15",
                token_major=True,
                output_path=str(tmp_path / "out"),
            )
