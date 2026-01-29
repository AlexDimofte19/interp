# eval_cognitive_map_probe

Evaluate cognitive map probes on test trajectories with detailed metrics.

## Overview

This command evaluates a trained cognitive map probe on a set of test trajectories and their corresponding activations. It provides comprehensive metrics broken down by grid size, complexity, and their combinations, allowing for detailed analysis of probe performance across different conditions.

## Features

- **Global metrics**: Overall accuracy, balanced accuracy, and per-class precision/recall/F1
- **Size-stratified metrics**: Performance breakdown by grid size (e.g., 5x5, 7x7, 9x9)
- **Complexity-stratified metrics**: Performance breakdown by grid complexity (0.0-1.0)
- **Combined metrics**: Performance for each size-complexity combination
- **Baseline accuracy**: Majority class baseline for each metrics block
- **JSON export**: Automatic saving of results for further analysis

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `probe_path` | str | required | Path to the trained probe (.pt file) |
| `trajectories_dir` | str | required | Directory containing trajectory JSON files organized by size |
| `activations_dir` | str | required | Directory containing gathered activations organized by size |
| `layers` | str | "all" | Layer specification (e.g., "15", "10,15,20", "all") |
| `steps` | str | "all" | Step specification (e.g., "0", "all", "0:5") |
| `prompt_prefix_indices` | str \| None | None | Token indices for prompt_prefix (e.g., "all", "-1") |
| `prompt_suffix_indices` | str \| None | None | Token indices for prompt_suffix (e.g., "-3:-1") |
| `grid_state_indices` | str \| None | None | Token indices for grid_state (e.g., "all") |
| `output_indices` | str \| None | None | Token indices for output (e.g., "all", "0:10") |
| `pad_to_size` | int | 15 | Pad grid to this size for consistent evaluation |
| `output_path` | str \| None | None | Path for JSON results (auto-generates if not provided) |
| `device` | str \| None | None | Device ("cuda", "cpu"). Auto-detects if None |
| `verbose` | bool | True | Print progress and metrics tables |

## Directory Structure

The command expects trajectories and activations organized by size:

```
trajectories_dir/
├── size5/
│   ├── trajectory_001.json
│   ├── trajectory_002.json
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
│   │               ├── prompt_suffix/
│   │               ├── grid_state/
│   │               └── output/
│   └── ...
├── size7/
│   └── ...
└── size9/
    └── ...
```

## Examples

### Basic evaluation

```bash
telos-interp eval_cognitive_map_probe \
    --probe-path /path/to/cognitive_map_probe.pt \
    --trajectories-dir /path/to/test_trajectories \
    --activations-dir /path/to/test_activations \
    --prompt-suffix-indices "-3:-1"
```

### Evaluate on specific layers and steps

```bash
telos-interp eval_cognitive_map_probe \
    --probe-path /path/to/probe.pt \
    --trajectories-dir /path/to/trajectories \
    --activations-dir /path/to/activations \
    --layers "15,20" \
    --steps "0:5" \
    --prompt-suffix-indices "-3:-1" \
    --output-indices "0:10"
```

### Custom output path

```bash
telos-interp eval_cognitive_map_probe \
    --probe-path /path/to/probe.pt \
    --trajectories-dir /path/to/trajectories \
    --activations-dir /path/to/activations \
    --prompt-suffix-indices "-3:-1" \
    --output-path /results/my_evaluation.json
```

### Full configuration

```bash
telos-interp eval_cognitive_map_probe \
    --probe-path probes/cognitive_map_probe_l15_mlp.pt \
    --trajectories-dir data/test_trajectories \
    --activations-dir activations/test \
    --layers "15" \
    --steps "all" \
    --prompt-suffix-indices "-3:-1" \
    --pad-to-size 15 \
    --device cuda \
    --verbose
```

## Output Format

### Console Output

The command prints formatted tables for each metrics category:

```
=====================================================================================
GLOBAL METRICS
=====================================================================================
Overall Accuracy: 0.8542 (Baseline: 0.4231, 12500 samples)

Per-class metrics:
---------------------------------------------------------------------------------------
Class       Accuracy  Precision     Recall   F1-Score   GT Support  Predicted
---------------------------------------------------------------------------------------
.             0.9200     0.8800     0.9200     0.8996         5000       5227
#             0.8100     0.8500     0.8100     0.8295         3500       3335
A             0.7800     0.7200     0.7800     0.7488         2000       2167
G             0.8900     0.9100     0.8900     0.8999         2000       1771
---------------------------------------------------------------------------------------
```

### JSON Output

Results are automatically saved to JSON (default: `eval_<probe_name>_<trajectories_dir>.json`):

```json
{
  "global": {
    "accuracy": 0.8542,
    "baseline_accuracy": 0.4231,
    "per_class": {
      "0": {
        "accuracy": 0.92,
        "precision": 0.88,
        "recall": 0.92,
        "f1": 0.8996,
        "gt_support": 5000,
        "predicted": 5227
      }
    },
    "total_samples": 12500
  },
  "by_size": {
    "5": { ... },
    "7": { ... },
    "9": { ... }
  },
  "by_complexity": {
    "0.3": { ... },
    "0.5": { ... },
    "0.7": { ... }
  },
  "by_size_complexity": {
    "5_0.3": { ... },
    "7_0.5": { ... }
  },
  "total_trajectories": 150,
  "total_steps": 1200,
  "config": {
    "probe_path": "/path/to/probe.pt",
    "trajectories_dir": "/path/to/trajectories",
    "activations_dir": "/path/to/activations",
    "layers": "15",
    "steps": "all",
    "token_categories": {
      "prompt_suffix": "-3:-1"
    },
    "pad_to_size": 15
  }
}
```

## Metrics Explained

| Metric | Description |
|--------|-------------|
| **Accuracy** | Fraction of correct predictions |
| **Baseline Accuracy** | Accuracy achieved by always predicting the majority class |
| **Precision** | TP / (TP + FP) - fraction of predicted positives that are correct |
| **Recall** | TP / (TP + FN) - fraction of actual positives that are predicted |
| **F1-Score** | Harmonic mean of precision and recall |
| **GT Support** | Number of ground truth samples for each class |
| **Predicted** | Number of predictions for each class |

## Notes

- The probe must have been trained with the same token category configuration (concatenation order matters)
- Token indices must match those used during `prepare_activations_for_probing` and training
- Grid states are automatically padded to `pad_to_size` for consistent evaluation
- Trajectories without corresponding activations are silently skipped
- The baseline accuracy helps contextualize performance on imbalanced datasets
