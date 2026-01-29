# apply_cognitive_map_probe

Apply a trained cognitive map probe to trajectory activations and store predictions in the trajectory JSON files.

## Overview

This command loads a trained probe and applies it to specified tokens in each trajectory. For each token and each grid position, the probe is applied with position (row, col) concatenated to the activation vector. The predictions (class probabilities) are stored directly in the trajectory JSON files as a "probes" field in each token.

## Output Format

For each processed token, probe predictions are stored in the following format:

```json
{
    "id": 361,
    "token": "<|end|>",
    "token_id": 200007,
    "token_groups": ["prompt", "template"],
    "probes": {
        "cognitive_map_probe_mlp_r0_c0": {
            "model.layers.20.output": {
                "agent": 0.0005,
                "empty": 0.0019,
                "goal": 0.0005,
                "wall": 0.9971
            }
        },
        "cognitive_map_probe_mlp_r0_c1": {
            "model.layers.20.output": {
                "agent": 0.0005,
                "empty": 0.0119,
                "goal": 0.0005,
                "wall": 0.9871
            }
        }
    }
}
```

The probe name in the output is derived from the probe file name (e.g., `cognitive_map_probe_mlp.pt` → `cognitive_map_probe_mlp`), with position suffixed as `_r{row}_c{col}`.

## Parameters

| Parameter                 | Type         | Default | Description                                                  |
| ------------------------- | ------------ | ------- | ------------------------------------------------------------ |
| `activations_dir`         | str          | required | Directory containing trajectory activation folders          |
| `trajectories_dir`        | str          | required | Directory containing trajectory JSON files                  |
| `probe_path`              | str          | required | Path to the trained probe .pt file                          |
| `grid_size`               | int \| None  | None    | Size of the grid (default: inferred from trajectory JSON)   |
| `output_dir`              | str \| None  | None    | Output directory for modified JSONs (default: overwrite originals) |
| `layers`                  | str          | "all"   | Layer indices to process (e.g., "all", "20", "7,15", "0:10") |
| `steps`                   | str          | "all"   | Step indices to process (e.g., "all", "0", "0:5")           |
| `prompt_prefix_indices`   | str \| None  | None    | Token indices for prompt_prefix (None = skip)               |
| `prompt_suffix_indices`   | str \| None  | None    | Token indices for prompt_suffix (None = skip)               |
| `grid_state_indices`      | str \| None  | None    | Token indices for grid_state (None = skip)                  |
| `output_indices`          | str \| None  | None    | Token indices for output (None = skip, "all" = all available) |
| `verbose`                 | bool         | False   | Print detailed progress information                          |

### Index Specification Format

The `layers`, `steps`, and token index parameters support flexible specification:

- `"all"` - All available indices
- `"7"` - Single index
- `"7,15,23"` - Multiple specific indices
- `"0:10"` - Range (inclusive, 0 to 10)
- `"-1"` - Last index
- `"-3:-1"` - Last 3 indices

## Examples

### Apply probe to the last output token for all layers

```bash
interp-cli apply_cognitive_map_probe \
    --activations-dir /path/to/activations \
    --trajectories-dir /path/to/trajectories \
    --probe-path /path/to/cognitive_map_probe_mlp.pt \
    --layers all \
    --steps all \
    --output-indices -1
```

### Apply probe to prompt_suffix tokens for layer 20

```bash
interp-cli apply_cognitive_map_probe \
    --activations-dir /path/to/activations \
    --trajectories-dir /path/to/trajectories \
    --probe-path /path/to/cognitive_map_probe_mlp.pt \
    --layers 20 \
    --steps 0 \
    --prompt-suffix-indices all
```

### Apply probe with explicit grid size, saving to a new directory

```bash
interp-cli apply_cognitive_map_probe \
    --activations-dir /path/to/activations \
    --trajectories-dir /path/to/trajectories \
    --probe-path /path/to/cognitive_map_probe_mlp.pt \
    --output-dir /path/to/output \
    --grid-size 5 \
    --layers 20 \
    --steps all \
    --output-indices -1
```

## Notes

- At least one token category index must be specified (prompt_prefix, prompt_suffix, grid_state, or output)
- Existing probe predictions in tokens are preserved and merged with new predictions
- If `output_dir` is not specified, the original trajectory JSONs are overwritten
- The probe expects input with position: `[activation, row, col]`. Normalization (if the probe was trained with it) is handled internally by the probe.
- Grid size can be inferred automatically from the trajectory JSON's `grid_params.grid_width` and `grid_params.grid_height` fields.

## Workflow

This command fits into the cognitive map probing pipeline as follows:

```
[Trajectory JSONs] → gather_activations → [Activation .pt files]
                                                    ↓
                     prepare_activations_for_probing
                                                    ↓
                                         [Prepared dataset .pt]
                                                    ↓
                          train_cognitive_map_probe
                                                    ↓
                                         [Trained probe .pt]
                                                    ↓
                         apply_cognitive_map_probe ←───────────┐
                                                    ↓          │
                            [Updated Trajectory JSONs with     │
                             probe predictions in tokens]      │
                                         [Activation .pt files]─┘
```
