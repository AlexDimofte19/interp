# prepare_activations_for_probing

Prepare activations from `gather_activations` output for training probing classifiers.

## Overview

This command loads activations from the nested folder structure produced by `gather_activations`, combines them with metadata from trajectory JSON files, and prepares them for different probing tasks.

## Probe Types

The trainer's logical view of each probe type is given below. The on-disk layout (v3 manifest dir) is described under **Output Format**.

### `grid_tile` (default)

Predicts grid cell identity from activations.

- **Per-trajectory activation**: `(D,)` — one shared activation per trajectory, written once.
- **Per-cell metadata** (in manifest): `positions` `(C, 2)` and `labels` `(C,)` for each trajectory.
- **Trainer input rows**: `(activation_dim + 2,)` — activation concatenated with `[row_id, col_id]`, materialized lazily by `GridTileCompactDataset`.

Cell type mapping:
| Symbol | ID | Meaning |
|--------|-----|---------|
| A | 0 | Agent |
| # | 1 | Wall |
| G | 2 | Goal |
| _ | 3 | Empty |
| D | 4 | Door |
| K | 5 | Key |
| ? | 6 | Unknown |
| + | 7 | Padding |

### `distance`

Predicts A* distance to goal from activations.

- **Per-trajectory activation**: `(D,)`.
- **Label**: int A* distance from `grid_params.astar_distance` (in manifest as `astar_distance`).
- **Trainer input rows**: `(D,)` activation, `(T,)` labels.

### `action_sequence`

Predicts the sequence of actions taken in a trajectory.

- **Per-trajectory activation**: `(D,)`.
- **Label**: variable-length action list (in manifest as `actions`); loader pads to `max_seq_len` with -1.
- **Trainer input rows**: `(D,)` activation, `(T, max_seq_len)` labels.

Action mapping: `{LEFT: 0, TOP: 1, RIGHT: 2, DOWN: 3}`

## Folder Modes

### Single-folder mode

```
activations_dir/
  {trajectory_name}/
    {model_name}/
      layer_{N}/step_{M}/{category}/{token_id}.pt

trajectories_dir/
  {trajectory_name}.json
```

### Multi-size mode

Automatically detected when directories contain `sizeN` subfolders:

```
activations_dir/
  size5/
    {trajectory_name}/...
  size7/
    {trajectory_name}/...

trajectories_dir/
  size5/
    {trajectory_name}.json
  size7/
    {trajectory_name}.json
```

In multi-size mode:
- Each size folder is processed
- Results are merged into a single manifest with per-size subdirs under `activations/`
- `pad_to_size` is auto-set to the maximum size for consistent merging
- Each manifest entry carries a `size: int` field

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `activations_dir` | str | required | Directory containing activation folders |
| `trajectories_dir` | str | required | Directory containing trajectory JSON files |
| `probe_type` | str | "grid_tile" | Type of probe: "grid_tile", "distance", or "action_sequence" |
| `layers` | str | "all" | Layer indices to extract |
| `steps` | str | "0" | Step indices (first step used for grid parsing) |
| `pad_to_size` | int \| None | None | Pad grid to this size (auto-detected if None) |
| `max_positions_per_trajectory` | int \| None | None | Max cell positions per trajectory (grid_tile only) |
| `balance_classes_per_trajectory` | bool | False | Balance cell type classes (grid_tile only) |
| `prompt_prefix_indices` | str \| None | None | Token indices for prompt_prefix |
| `prompt_suffix_indices` | str \| None | None | Token indices for prompt_suffix |
| `grid_state_indices` | str \| None | None | Token indices for grid_state |
| `output_indices` | str \| None | None | Token indices for output |
| `output_path` | str \| None | None | Output **directory** (auto-named under `activations_dir` if None). If you pass a path ending in `.pt`, the suffix is stripped with a warning. |
| `verbose` | bool | False | Print detailed progress |
| `seed` | int | 42 | Random seed for reproducibility |

## Examples

### Grid tile probing (single size)

```bash
interp-cli prepare_activations_for_probing \
    --activations-dir /path/to/activations/size5 \
    --trajectories-dir /path/to/trajectories/size5 \
    --probe-type grid_tile \
    --layers all \
    --steps 0 \
    --output-indices -1
```

### Grid tile with class balancing

```bash
interp-cli prepare_activations_for_probing \
    --activations-dir /path/to/activations/size7 \
    --trajectories-dir /path/to/trajectories/size7 \
    --probe-type grid_tile \
    --output-indices all \
    --balance-classes-per-trajectory \
    --max-positions-per-trajectory 100
```

### Distance probing

```bash
interp-cli prepare_activations_for_probing \
    --activations-dir /path/to/activations/size5 \
    --trajectories-dir /path/to/trajectories/size5 \
    --probe-type distance \
    --layers all \
    --output-indices -1
```

### Action sequence probing

```bash
interp-cli prepare_activations_for_probing \
    --activations-dir /path/to/activations/size5 \
    --trajectories-dir /path/to/trajectories/size5 \
    --probe-type action_sequence \
    --layers all \
    --output-indices -1
```

### Multi-size mode (automatic)

```bash
interp-cli prepare_activations_for_probing \
    --activations-dir /path/to/activations \
    --trajectories-dir /path/to/trajectories \
    --probe-type grid_tile \
    --output-indices all \
    --verbose
```

### Specific layers and custom output path

```bash
interp-cli prepare_activations_for_probing \
    --activations-dir /path/to/activations/size5 \
    --trajectories-dir /path/to/trajectories/size5 \
    --probe-type grid_tile \
    --layers "7,15,23,31" \
    --output-indices -1 \
    --output-path /path/to/output/my_activations
```

## Output Format

The command writes a **directory** (not a single `.pt`):

```
{output_dir}/
  manifest.json
  activations/
    {trajectory_name_1}.pt              # tensor of shape (D,)
    {trajectory_name_2}.pt
    ...
```

In multi-size mode the per-trajectory `.pt` files are namespaced by size:

```
{output_dir}/
  manifest.json
  activations/
    size5/{trajectory_name_a}.pt
    size7/{trajectory_name_b}.pt
    ...
```

Each per-trajectory `.pt` holds a single `(D,)` activation tensor — there is no per-cell replication on disk. For `grid_tile`, the trainer assembles `[activation, row, col]` rows lazily via `GridTileCompactDataset` (see `manifest_loader.py`).

### `manifest.json` schema

```jsonc
{
  "format_version": 3,
  "probe_type": "grid_tile",                 // or "distance" / "action_sequence"
  "activation_dim": 131072,
  "num_cells_per_trajectory": 225,           // grid_tile only
  "max_seq_len": 14,                         // action_sequence only
  "action_to_id": {"LEFT": 0, ...},          // action_sequence only
  "sizes": ["size5", "size7"],               // multi-size only
  "per_size_info": {                         // multi-size only
    "size5": {"num_trajectories": 100, "num_cells_per_trajectory": 225},
    "size7": {"num_trajectories": 80,  "num_cells_per_trajectory": 225}
  },
  "loading_spec": { /* echoes layers/steps/*_indices */ },
  "config":       { /* mirrors v1's "config" */ },
  "trajectories": [
    {
      "name": "traj_0001",
      "size": 5,                             // multi-size only
      "act_path": "activations/size5/traj_0001.pt",
      "positions": [[0,0],[0,1],...],        // grid_tile: list of [row, col]
      "labels":    [3, 1, 7, 3, ...],        // grid_tile: list of cell-ids
      "astar_distance": 12,                  // distance: int
      "actions":   [0, 2, 1, 3]              // action_sequence: list of action ids
    },
    ...
  ]
}
```

`act_path` is **always relative to `manifest.json`**, so the directory is portable: rename or move it without breaking references.

### Format versions

- **v3** (current): manifest dir + per-trajectory `(D,)` `.pt` files. Avoids the per-cell activation replication that made v1 prepare RAM scale as `T × C × D`.
- **v1** (legacy): single monolithic `.pt` containing a flat `(N, D+2)` activations tensor. Trainers still load v1 files via a backward-compatible dispatch.

## Notes

- At least one of `prompt_prefix_indices`, `prompt_suffix_indices`, `grid_state_indices`, or `output_indices` must be specified.
- The activation vector itself is stored once per trajectory; per-cell `[row_id, col_id]` is folded in at training time.
- Class balancing finds the minimum count across all cell types and samples equally from each.
- When `balance_classes_per_trajectory` and `max_positions_per_trajectory` are both set, `max_positions` is adjusted to be divisible by the number of classes.
- NaN filtering still happens on the trainer side (a trajectory whose activation contains a NaN is dropped at load time, not at prepare time).
