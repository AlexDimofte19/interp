# prepare_activations_for_probing

Prepare activations from `gather_activations` output for training probing classifiers.

## Overview

This command loads activations from the nested folder structure produced by `gather_activations`, combines them with metadata from trajectory JSON files, and prepares them for different probing tasks.

## Probe Types

### `grid_tile` (default)

Predicts grid cell identity from activations.

- **Activations**: `(N, activation_dim + 2)` — activation vector concatenated with `[row_id, col_id]`
- **Labels**: `(N,)` — grid tile identity (cell type)

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

- **Activations**: `(num_trajectories, activation_dim)` — raw activation vectors
- **Labels**: `(num_trajectories,)` — A* distance from `grid_params.astar_distance`

### `action_sequence`

Predicts the sequence of actions taken in a trajectory.

- **Activations**: `(num_trajectories, activation_dim)` — raw activation vectors
- **Labels**: `(num_trajectories, max_seq_len)` — action sequences, padded with -1

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
- Results are merged into a single output file
- `pad_to_size` is auto-set to the maximum size for consistent merging
- A `size_labels` tensor tracks which size each sample came from

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
| `output_path` | str \| None | None | Output file path (auto-generated if None) |
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
    --output-path /path/to/output/my_activations.pt
```

## Output Format

The output `.pt` file contains a dictionary with:

```python
{
    "activations": torch.Tensor,  # Shape depends on probe_type
    "labels": torch.Tensor,       # Shape depends on probe_type
    "trajectory_names": list[str],
    "activation_dim": int,
    "probe_type": str,
    "config": dict,               # All configuration parameters

    # For grid_tile:
    "num_cells_per_trajectory": int,

    # For action_sequence:
    "sequence_lengths": torch.Tensor,
    "action_to_id": dict,

    # For multi-size mode:
    "size_labels": torch.Tensor,
    "sizes": list[str],
    "per_size_info": dict,
}
```

## Notes

- At least one of `prompt_prefix_indices`, `prompt_suffix_indices`, `grid_state_indices`, or `output_indices` must be specified
- For `grid_tile` mode, the activation vector includes position information `[row_id, col_id]` at the end
- Class balancing finds the minimum count across all cell types and samples equally from each
- When `balance_classes_per_trajectory` and `max_positions_per_trajectory` are both set, `max_positions` is adjusted to be divisible by the number of classes
