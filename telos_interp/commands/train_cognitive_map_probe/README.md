# train_cognitive_map_probe

Train cognitive map probing classifiers on prepared activations.

## Overview

This command trains either a logistic regression or multi-layer perceptron (MLP) classifier on activations prepared by `prepare_activations_for_probing` with `probe_type=grid_tile`. The trained model predicts grid cell identity from activation vectors.

## Model Types

### Logistic Regression (`lr`)

A simple linear classifier:
```
input (activation_dim) → Linear → output (num_classes)
```

Best for:
- Baseline experiments
- Fast training
- Interpretable weights

### Multi-Layer Perceptron (`mlp`)

A neural network with configurable hidden layers:
```
input → [Linear → ReLU → Dropout] × N → Linear → output
```

Best for:
- Higher capacity modeling
- Complex feature interactions
- Better accuracy on difficult tasks

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_data_path` | str | required | Path to .pt file from `prepare_activations_for_probing` |
| `model_type` | str | "lr" | Model type: "lr" (logistic regression) or "mlp" |
| `eval_data_path` | str \| None | None | Optional separate .pt file for evaluation |
| `output_path` | str \| None | None | Output path for trained model |
| `num_epochs` | int | 100 | Number of training epochs |
| `learning_rate` | float | 0.01 | Learning rate for optimizer |
| `batch_size` | int | 256 | Batch size for training |
| `weight_decay` | float | 1e-4 | L2 regularization weight decay |
| `hidden_dims` | str | "512,256" | Comma-separated hidden layer dimensions (MLP only) |
| `dropout` | float | 0.1 | Dropout rate (MLP only) |
| `eval_split` | float | 0.2 | Fraction for validation (if no eval_data_path) |
| `class_weight` | str \| None | None | Class weighting: None or "balanced" |
| `balance_classes` | bool | False | Upsample minority classes to match majority |
| `normalize` | bool | False | Normalize activations using training mean/std |
| `device` | str \| None | None | Device ("cuda", "cpu"). Auto-detects if None |
| `seed` | int | 42 | Random seed for reproducibility |
| `verbose` | bool | True | Print training progress |

## Examples

### Train logistic regression

```bash
interp-cli train_cognitive_map_probe \
    --train-data-path /path/to/activations_grid_tile.pt \
    --model-type lr \
    --num-epochs 100 \
    --learning-rate 0.01
```

### Train MLP with default architecture

```bash
interp-cli train_cognitive_map_probe \
    --train-data-path /path/to/activations_grid_tile.pt \
    --model-type mlp \
    --num-epochs 200 \
    --learning-rate 0.001
```

### Train MLP with custom architecture

```bash
interp-cli train_cognitive_map_probe \
    --train-data-path /path/to/activations_grid_tile.pt \
    --model-type mlp \
    --hidden-dims "1024,512,256" \
    --dropout 0.2 \
    --num-epochs 300 \
    --learning-rate 0.0005
```

### With separate evaluation file

```bash
interp-cli train_cognitive_map_probe \
    --train-data-path /path/to/train_activations.pt \
    --eval-data-path /path/to/eval_activations.pt \
    --model-type mlp \
    --hidden-dims "512,256" \
    --output-path /path/to/model.pt
```

### Full configuration example

```bash
interp-cli train_cognitive_map_probe \
    --train-data-path /path/to/activations.pt \
    --model-type mlp \
    --hidden-dims "512,256,128" \
    --dropout 0.15 \
    --num-epochs 200 \
    --learning-rate 0.001 \
    --batch-size 512 \
    --weight-decay 1e-3 \
    --eval-split 0.15 \
    --device cuda \
    --seed 123 \
    --output-path /path/to/my_probe.pt

interp-cli train_cognitive_map_probe \
    --train-data-path activations/cognitive_map_activations_l7_s0_output_all_grid_tile_pad15_balanced_merged.pt \
    --model-type mlp \
    --hidden-dims "512,256" \
    --dropout 0.1 \
    --num-epochs 10 \
    --learning-rate 0.001 \
    --batch-size 512 \
    --weight-decay 1e-3 \
    --eval-split 0.05 \
    --device cuda \
    --seed 123 \
    --output-path cognitive_map_probe_test.pt
```

### Handling class imbalance

```bash
# Use balanced class weights (weights inversely proportional to frequency)
interp-cli train_cognitive_map_probe \
    --train-data-path /path/to/activations.pt \
    --model-type mlp \
    --class-weight balanced \
    --num-epochs 100

# Upsample minority classes to match majority
interp-cli train_cognitive_map_probe \
    --train-data-path /path/to/activations.pt \
    --model-type mlp \
    --balance-classes \
    --num-epochs 100

# Combine both approaches
interp-cli train_cognitive_map_probe \
    --train-data-path /path/to/activations.pt \
    --model-type mlp \
    --class-weight balanced \
    --balance-classes \
    --num-epochs 100
```

### Quick baseline with logistic regression

```bash
interp-cli train_cognitive_map_probe \
    --train-data-path /path/to/activations.pt \
    --model-type lr \
    --num-epochs 50 \
    --learning-rate 0.1 \
    --batch-size 1024
```

## Output Format

The trained model is saved as a `.pt` file containing:

```python
{
    "model_state_dict": dict,       # PyTorch model weights
    "model_type": str,              # "lr" or "mlp"
    "input_dim": int,               # Input dimension
    "num_classes": int,             # Number of output classes
    "hidden_dims": list | None,     # Hidden dimensions (MLP only)
    "dropout": float | None,        # Dropout rate (MLP only)
    "label_to_idx": dict,           # Original label -> model output index
    "idx_to_label": dict,           # Model output index -> original label
    "scaler_mean": Tensor | None,   # Normalization mean (if normalize=True)
    "scaler_std": Tensor | None,    # Normalization std (if normalize=True)
    "config": { ... },              # Training configuration
    "results": { ... },             # Training results (accuracy, loss, etc.)
}
```

## Loading and Using a Trained Probe

```python
from telos_interp.commands.train_cognitive_map_probe import CognitiveMapProbe

# Load probe
probe = CognitiveMapProbe.load("/path/to/model.pt", device="cuda")

# Get predictions (returns original label IDs, not internal indices)
predictions = probe.predict(activations)  # shape: (N,)

# Get class probabilities
probs = probe.predict_proba(activations)  # shape: (N, num_classes)

# Check if normalization is enabled (applied automatically in predict/predict_proba)
print(f"Normalized: {probe.normalized}")

# Access training results
print(f"Accuracy: {probe.results['final_accuracy']}")
```

## Training Details

- **Optimizer**: AdamW with configurable weight decay
- **Scheduler**: Cosine annealing learning rate schedule
- **Loss**: Cross-entropy loss
- **Best model tracking**: Saves the model with highest validation accuracy
- **Progress**: tqdm progress bar showing train loss, eval accuracy, and best accuracy

## Notes

- Only `probe_type=grid_tile` data is supported (activations with position info)
- The input dimension includes the 2 position features `[row_id, col_id]` appended to activations
- Training automatically uses CUDA if available, unless `--device cpu` is specified
- The best model (highest validation accuracy) is saved, not the final epoch model
- Samples with NaN values in activations are automatically filtered out with a warning
- **Label remapping**: Only classes present in the data are used. Labels are remapped to contiguous indices (e.g., `[0, 1, 2, 3, 7]` becomes `[0, 1, 2, 3, 4]`). The saved model includes `label_to_idx` and `idx_to_label` mappings to convert between original labels and model output indices
