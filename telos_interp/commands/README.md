# telos-interp Commands

This folder contains CLI commands for the telos-interp package. Commands are invoked via the `interp-cli` CLI.

## Command Pipeline

The typical workflow follows these steps:

```
[Trajectory JSONs] → gather_activations → [Activation .pt files]
                                                    ↓
                     prepare_activations_for_probing
                                                    ↓
                                         [Prepared dataset .pt]
                                        ╱                       ╲
          train_cognitive_map_probe                   train_distance_probe
                      ↓                                         ↓
           [Trained probe .pt]                       [Trained probe .pt]
                      ↓                                         ↓
         eval_cognitive_map_probe                   eval_distance_probe
                      ↓
        apply_cognitive_map_probe
                      ↓
           [Updated Trajectory JSONs
            with probe predictions]
```

## Commands

### gather_activations

Extract model activations from trajectory JSON files.

```bash
interp-cli gather_activations \
    --trajectory-paths "/path/to/trajectories/*.json" \
    --output-dir /path/to/activations \
    --layers all \
    --steps 0 \
    --output-indices -1
```

**Key options:**
| Option | Description |
|--------|-------------|
| `--trajectory-paths` | Paths to trajectory JSON files (glob patterns supported) |
| `--output-dir` | Output directory for activations |
| `--layers` | Layer indices ("all", "0:10", "7,15", "-1") |
| `--steps` | Step indices to extract |
| `--output-indices` | Token indices for output category |

See [`gather_activations/README.md`](gather_activations/README.md) for full documentation.

---

### prepare_activations_for_probing

Prepare extracted activations for training probing classifiers or regressors.

```bash
interp-cli prepare_activations_for_probing \
    --activations-dir /path/to/activations \
    --trajectories-dir /path/to/trajectories \
    --probe-type grid_tile \
    --output-indices -1
```

**Key options:**
| Option | Description |
|--------|-------------|
| `--activations-dir` | Directory with activation folders |
| `--trajectories-dir` | Directory with trajectory JSONs |
| `--probe-type` | "grid_tile", "distance", or "action_sequence" |
| `--layers` | Layer indices to include |
| `--balance-classes-per-trajectory` | Balance cell type classes |

**Probe types:**
- `grid_tile` — Predict cell identity (wall, empty, goal, etc.)
- `distance` — Predict A* distance to goal
- `action_sequence` — Predict action sequence

See [`prepare_activations_for_probing/README.md`](prepare_activations_for_probing/README.md) for full documentation.

---

### train_cognitive_map_probe

Train a cell identity classifier on prepared activations.

```bash
interp-cli train_cognitive_map_probe \
    --train-data-path /path/to/activations.pt \
    --model-type mlp \
    --hidden-dims "512,256" \
    --num-epochs 100
```

**Key options:**
| Option | Description |
|--------|-------------|
| `--train-data-path` | Path to prepared .pt file (probe_type=grid_tile) |
| `--model-type` | "lr" (logistic regression) or "mlp" |
| `--eval-data-path` | Optional separate evaluation file |
| `--hidden-dims` | MLP hidden layer sizes (e.g., "512,256") |
| `--num-epochs` | Number of training epochs |
| `--learning-rate` | Learning rate |

See [`train_cognitive_map_probe/README.md`](train_cognitive_map_probe/README.md) for full documentation.

---

### train_distance_probe

Train a distance regression probe on prepared activations.

```bash
interp-cli train_distance_probe \
    --train-data-path /path/to/activations_distance.pt \
    --model-type mlp \
    --hidden-dims "512,256" \
    --num-epochs 100
```

**Key options:**
| Option | Description |
|--------|-------------|
| `--train-data-path` | Path to prepared .pt file (probe_type=distance) |
| `--model-type` | "lr" (linear regression) or "mlp" |
| `--eval-data-path` | Optional separate evaluation file |
| `--hidden-dims` | MLP hidden layer sizes (e.g., "512,256") |
| `--num-epochs` | Number of training epochs |
| `--normalize-labels` | Normalize regression targets during training |

See [`train_distance_probe/README.md`](train_distance_probe/README.md) for full documentation.

---

### eval_cognitive_map_probe

Evaluate a trained cell identity probe on test trajectories.

```bash
interp-cli eval_cognitive_map_probe \
    --probe-path /path/to/cognitive_map_probe.pt \
    --trajectories-dir /path/to/test_trajectories \
    --activations-dir /path/to/test_activations \
    --output-indices -1
```

**Key options:**
| Option | Description |
|--------|-------------|
| `--probe-path` | Path to trained probe .pt file |
| `--trajectories-dir` | Directory with test trajectory JSONs (organized by size) |
| `--activations-dir` | Directory with test activations (matching structure) |
| `--layers` | Layer indices to evaluate |
| `--pad-to-size` | Pad grid to this size for consistent evaluation |

See [`eval_cognitive_map_probe/README.md`](eval_cognitive_map_probe/README.md) for full documentation.

---

### eval_distance_probe

Evaluate a trained distance regression probe on test trajectories.

```bash
interp-cli eval_distance_probe \
    --trajectories-dir /path/to/test_trajectories \
    --activations-dir /path/to/test_activations \
    --probe-path /path/to/distance_probe.pt \
    --output-indices -1
```

**Key options:**
| Option | Description |
|--------|-------------|
| `--probe-path` | Path to trained DistanceProbe .pt file |
| `--trajectories-dir` | Directory with test trajectory JSONs (organized by size) |
| `--activations-dir` | Directory with test activations (matching structure) |
| `--layers` | Layer indices to evaluate |
| `--output-path` | Path to save results JSON |

See [`eval_distance_probe/README.md`](eval_distance_probe/README.md) for full documentation.

---

### apply_cognitive_map_probe

Apply a trained probe to trajectory activations and store predictions in the trajectory JSON files.

```bash
interp-cli apply_cognitive_map_probe \
    --activations-dir /path/to/activations \
    --trajectories-dir /path/to/trajectories \
    --probe-path /path/to/probe.pt \
    --layers 20 \
    --steps all \
    --output-indices -1
```

**Key options:**
| Option | Description |
|--------|-------------|
| `--activations-dir` | Directory with activation folders |
| `--trajectories-dir` | Directory with trajectory JSONs |
| `--probe-path` | Path to trained probe .pt file |
| `--output-dir` | Output directory for modified JSONs (optional) |
| `--layers` | Layer indices to process |
| `--output-indices` | Token indices for output category |

See [`apply_cognitive_map_probe/README.md`](apply_cognitive_map_probe/README.md) for full documentation.

---

## Quick Start Example

```bash
# 1. Extract activations from trajectories
interp-cli gather_activations \
    --trajectory-paths "/data/trajectories/size5/*.json" \
    --output-dir /data/activations/size5 \
    --layers all \
    --steps 0 \
    --output-indices -1

# 2. Prepare activations for grid tile probing
interp-cli prepare_activations_for_probing \
    --activations-dir /data/activations/size5 \
    --trajectories-dir /data/trajectories/size5 \
    --probe-type grid_tile \
    --output-indices -1 \
    --balance-classes-per-trajectory

# 3. Train a probe
interp-cli train_cognitive_map_probe \
    --train-data-path /data/activations/size5/cognitive_map_activations_*.pt \
    --model-type mlp \
    --hidden-dims "512,256" \
    --num-epochs 100

# 4. Evaluate the probe
interp-cli eval_cognitive_map_probe \
    --probe-path /path/to/cognitive_map_probe_mlp.pt \
    --trajectories-dir /data/trajectories/size5_test \
    --activations-dir /data/activations/size5_test \
    --output-indices -1

# 5. Apply probe to trajectories
interp-cli apply_cognitive_map_probe \
    --activations-dir /data/activations/size5 \
    --trajectories-dir /data/trajectories/size5 \
    --probe-path /path/to/cognitive_map_probe_mlp.pt \
    --output-dir /data/trajectories_with_probes/size5 \
    --layers 20 \
    --steps all \
    --output-indices -1
```
