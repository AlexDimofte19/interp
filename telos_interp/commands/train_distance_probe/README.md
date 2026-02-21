# train_distance_probe

Train distance regression probes on prepared activations.

## Overview

This command trains either a linear regression or multi-layer perceptron (MLP) regressor on activations prepared by `prepare_activations_for_probing` with `probe_type=distance`. The trained model predicts A* shortest-path distance to the goal from activation vectors.

## Model Types

### Linear Regression (`lr`)

A simple linear regressor:
```
input (activation_dim) → Linear → output (1)
```

Best for:
- Baseline experiments
- Fast training
- Interpretable weights

### Multi-Layer Perceptron (`mlp`)

A neural network with configurable hidden layers:
```
input → [Linear → ReLU → Dropout] × N → Linear → output (1)
```

Best for:
- Higher capacity modeling
- Complex feature interactions
- Better accuracy on difficult tasks

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_data_path` | str | required | Path to .pt file from `prepare_activations_for_probing` (probe_type=distance) |
| `model_type` | str | "lr" | Model type: "lr" (linear regression) or "mlp" |
| `eval_data_path` | str \| None | None | Optional separate .pt file for evaluation |
| `output_path` | str \| None | None | Output path for trained model |
| `num_epochs` | int | 100 | Number of training epochs |
| `learning_rate` | float | 0.001 | Learning rate for optimizer |
| `batch_size` | int | 256 | Batch size for training |
| `weight_decay` | float | 1e-4 | L2 regularization weight decay |
| `hidden_dims` | str | "512,256" | Comma-separated hidden layer dimensions (MLP only) |
| `dropout` | float | 0.1 | Dropout rate (MLP only) |
| `eval_split` | float | 0.2 | Fraction for validation (if no eval_data_path) |
| `subset` | float | 1.0 | Fraction of training data to use (for quick experiments) |
| `normalize` | bool | False | Normalize input activations using training mean/std |
| `normalize_labels` | bool | False | Normalize regression targets (denormalized at inference) |
| `device` | str \| None | None | Device ("cuda", "cpu"). Auto-detects if None |
| `seed` | int | 42 | Random seed for reproducibility |
| `verbose` | bool | True | Print training progress |

## Examples

### Train linear regression

```bash
interp-cli train_distance_probe \
    --train-data-path /path/to/activations_distance.pt \
    --model-type lr \
    --num-epochs 100 \
    --learning-rate 0.01
```

### Train MLP with default architecture

```bash
interp-cli train_distance_probe \
    --train-data-path /path/to/activations_distance.pt \
    --model-type mlp \
    --num-epochs 200 \
    --learning-rate 0.001
```

### Train MLP with custom architecture and label normalization

```bash
interp-cli train_distance_probe \
    --train-data-path /path/to/activations_distance.pt \
    --model-type mlp \
    --hidden-dims "1024,512,256" \
    --dropout 0.2 \
    --num-epochs 300 \
    --normalize \
    --normalize-labels
```

### With separate evaluation file

```bash
interp-cli train_distance_probe \
    --train-data-path /path/to/train_activations.pt \
    --eval-data-path /path/to/eval_activations.pt \
    --model-type mlp \
    --hidden-dims "512,256" \
    --output-path /path/to/distance_probe.pt
```

## Output Format

The trained model is saved as a `.pt` file containing:

```python
{
    "model_state_dict": dict,       # PyTorch model weights
    "model_type": str,              # "lr" or "mlp"
    "input_dim": int,               # Input dimension
    "hidden_dims": list | None,     # Hidden dimensions (MLP only)
    "dropout": float | None,        # Dropout rate (MLP only)
    "scaler_mean": Tensor | None,   # Input normalization mean (if normalize=True)
    "scaler_std": Tensor | None,    # Input normalization std (if normalize=True)
    "label_mean": float | None,     # Label normalization mean (if normalize_labels=True)
    "label_std": float | None,      # Label normalization std (if normalize_labels=True)
    "config": { ... },              # Training configuration
    "results": { ... },             # Training results (MSE, MAE, R², etc.)
}
```

## Loading and Using a Trained Probe

```python
from telos_interp.commands.train_distance_probe import DistanceProbe

# Load probe
probe = DistanceProbe.load("/path/to/distance_probe.pt", device="cuda")

# Get predictions (returns denormalized distances)
predictions = probe.predict(activations)  # shape: (N,)

# Check properties
print(f"Input dim: {probe.input_dim}")
print(f"Normalized: {probe.normalized}")
```

## Training Details

- **Optimizer**: AdamW with configurable weight decay
- **Scheduler**: Cosine annealing learning rate schedule
- **Loss**: Mean squared error (MSE)
- **Best model tracking**: Saves the model with lowest validation loss
- **Progress**: tqdm progress bar showing train loss, eval MSE/MAE, and best loss
- **Label normalization**: When enabled, targets are z-scored during training and predictions are denormalized at inference

## Notes

- Only `probe_type=distance` data is supported
- Training automatically uses CUDA if available, unless `--device cpu` is specified
- The best model (lowest validation loss) is saved, not the final epoch model
- Samples with NaN values in activations are automatically filtered out with a warning
- Distance is the A* shortest-path distance from the agent to the goal at step 0
