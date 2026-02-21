# eval_distance_probe

Evaluate distance regression probes on test trajectories with detailed metrics.

## Overview

This command evaluates a trained distance regression probe on a set of test trajectories and their corresponding activations. It provides regression metrics (MSE, MAE, RMSE, R²) globally and broken down by grid size, complexity, and true distance values.

## Features

- **Global metrics**: Overall MSE, MAE, RMSE, and R² score
- **Size-stratified metrics**: Performance breakdown by grid size (e.g., 5x5, 7x7, 9x9)
- **Complexity-stratified metrics**: Performance breakdown by grid complexity (0.0-1.0)
- **Distance-stratified metrics**: Performance breakdown by true A* distance
- **Single-size mode**: Automatic detection when trajectories are not organized by size
- **JSON export**: Save results for further analysis

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trajectories_dir` | str | required | Directory containing trajectory JSON files (with size subfolders or flat) |
| `activations_dir` | str | required | Directory containing gathered activations (matching structure) |
| `probe_path` | str | required | Path to trained DistanceProbe .pt file |
| `layers` | str | "all" | Layer specification (e.g., "15", "10,15,20", "all") |
| `steps` | str | "all" | Step specification (overridden to "0" for distance probes) |
| `prompt_prefix_indices` | str \| None | None | Token indices for prompt_prefix (e.g., "all", "-1") |
| `prompt_suffix_indices` | str \| None | None | Token indices for prompt_suffix (e.g., "-3:-1") |
| `grid_state_indices` | str \| None | None | Token indices for grid_state (e.g., "all") |
| `output_indices` | str \| None | None | Token indices for output (e.g., "all", "0:10") |
| `output_path` | str \| None | None | Path for JSON results (optional) |
| `verbose` | bool | True | Print progress and metrics tables |

## Directory Structure

The command expects trajectories and activations organized by size (or flat):

```
trajectories_dir/
├── size5/
│   ├── trajectory_001.json
│   └── ...
├── size7/
│   └── ...
└── size9/
    └── ...

activations_dir/
├── size5/
│   ├── trajectory_001/
│   │   └── <model_name>/
│   │       └── layer_15/
│   │           └── step_0/
│   │               ├── grid_state/
│   │               └── output/
│   └── ...
├── size7/
│   └── ...
└── size9/
    └── ...
```

Single-size mode is used automatically when trajectories are directly in the base folder without size subfolders.

## Examples

### Basic evaluation

```bash
interp-cli eval_distance_probe \
    --trajectories-dir /path/to/test_trajectories \
    --activations-dir /path/to/test_activations \
    --probe-path /path/to/distance_probe.pt \
    --output-indices -1
```

### Evaluate on specific layers

```bash
interp-cli eval_distance_probe \
    --trajectories-dir /path/to/test_trajectories \
    --activations-dir /path/to/test_activations \
    --probe-path /path/to/distance_probe.pt \
    --layers "15,20" \
    --output-indices -1
```

### Save results to JSON

```bash
interp-cli eval_distance_probe \
    --trajectories-dir /path/to/test_trajectories \
    --activations-dir /path/to/test_activations \
    --probe-path /path/to/distance_probe.pt \
    --output-indices -1 \
    --output-path /results/distance_eval.json
```

## Output Format

### Console Output

```
============================================================
GLOBAL METRICS
------------------------------------------------------------
Metric                           Value
------------------------------------------------------------
MSE                             2.3456
MAE                             1.1234
RMSE                            1.5318
R²                              0.8765
Num Samples                      5000
------------------------------------------------------------

============================================================
METRICS BY TRUE DISTANCE
============================================================
Distance        MAE       RMSE      Count
----------------------------------------
1              0.4523     0.5678        500
2              0.8901     1.0234        800
3              1.2345     1.4567        700
...
```

### JSON Output

```json
{
  "global": {
    "mse": 2.3456,
    "mae": 1.1234,
    "rmse": 1.5318,
    "r2": 0.8765,
    "num_samples": 5000
  },
  "by_size": {
    "5": { ... },
    "7": { ... }
  },
  "by_complexity": {
    "0.3": { ... },
    "0.5": { ... }
  },
  "by_true_distance": {
    "1": { ... },
    "2": { ... }
  },
  "total_trajectories": 200,
  "skipped_trajectories": 5,
  "config": {
    "probe_path": "/path/to/probe.pt",
    "layers": "all",
    "steps": "0",
    "token_categories": { "output": "-1" }
  }
}
```

## Metrics Explained

| Metric | Description |
|--------|-------------|
| **MSE** | Mean squared error between predicted and true distances |
| **MAE** | Mean absolute error between predicted and true distances |
| **RMSE** | Root mean squared error (square root of MSE) |
| **R²** | Coefficient of determination (1.0 = perfect, 0.0 = baseline) |

## Notes

- Distance probes use A* shortest-path distance, which is only valid for the initial grid state (step 0). The `steps` parameter is automatically overridden to `"0"`
- The probe must have been trained with the same token category configuration
- Trajectories without `astar_distance` in their `grid_params` are skipped
- Trajectories without corresponding activations are silently skipped
