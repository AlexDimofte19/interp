# train_next_action_probe

Train next-action probing classifiers on prepared EOS-token activations.

## Overview

This command trains either a logistic regression or multi-layer perceptron (MLP) classifier on
activations prepared by `prepare_activations_for_probing` with `probe_type=next_action`. The
input to each sample is a **single end-of-sentence (EOS) token activation** from the reasoning
chain, and the label is the trajectory's `agent_action` (`LEFT` / `UP` / `RIGHT` / `DOWN`).

Each EOS token is an independent (i.i.d.) sample — subsetting and the train/eval split are plain
random splits over tokens (no trajectory grouping). Every EOS token from a given trajectory
shares that trajectory's action label.

## Input data

A **v3 manifest directory** produced by `prepare_activations_for_probing --probe-type next_action`.
That manifest references the gathered token `.pt` files in place (relative to `activations_root`)
— make sure the original gather output is still available at that path when training. See the
[prepare README](../prepare_activations_for_probing/README.md) for the manifest schema.

## Model Types

### Logistic Regression (`lr`)

A simple linear classifier:
```
input (activation_dim) → Linear → output (num_classes)
```

### Multi-Layer Perceptron (`mlp`)

```
input → [Linear → ReLU → Dropout] × N → Linear → output
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_data_path` | str | required | Path to the v3 `next_action` manifest dir from `prepare_activations_for_probing` |
| `model_type` | str | "lr" | Model type: "lr" (logistic regression) or "mlp" |
| `eval_data_path` | str \| None | None | Optional separate `next_action` manifest dir for evaluation |
| `output_path` | str \| None | None | Output path for the trained probe (defaults to the train data dir) |
| `num_epochs` | int | 100 | Number of training epochs |
| `learning_rate` | float | 0.01 | Learning rate for AdamW |
| `batch_size` | int | 256 | Batch size for training |
| `weight_decay` | float | 1e-4 | L2 regularization weight decay |
| `hidden_dims` | str | "512,256" | Comma-separated hidden layer dimensions (MLP only) |
| `dropout` | float | 0.1 | Dropout rate (MLP only) |
| `eval_split` | float | 0.2 | Fraction for validation (if no eval_data_path) |
| `subset` | float | 1.0 | Fraction of samples to use (random), applied before the split |
| `class_weight` | str \| None | None | Class weighting: None or "balanced" |
| `balance_classes` | bool | False | Upsample minority classes to match majority (v3 index-based balancer) |
| `normalize` | bool | False | Normalize activations using training mean/std |
| `device` | str \| None | None | Device ("cuda", "cpu"). Auto-detects if None |
| `seed` | int | 42 | Random seed for reproducibility |
| `verbose` | bool | True | Print training progress |
| `per_class_max_count` | int \| None | None | Cap per-class samples when balancing |

## Examples

### Train logistic regression

```bash
interp-cli train_next_action_probe \
    --train-data-path /path/to/prepared_next_action \
    --model-type lr \
    --num-epochs 50 \
    --learning-rate 0.01
```

### Train MLP

```bash
interp-cli train_next_action_probe \
    --train-data-path /path/to/prepared_next_action \
    --model-type mlp \
    --hidden-dims "512,256" \
    --dropout 0.1 \
    --num-epochs 200 \
    --learning-rate 0.001
```

### Handling class imbalance

```bash
# Balanced class weights in the loss
interp-cli train_next_action_probe \
    --train-data-path /path/to/prepared_next_action \
    --class-weight balanced

# Upsample minority classes
interp-cli train_next_action_probe \
    --train-data-path /path/to/prepared_next_action \
    --balance-classes
```

## Output Format

The trained probe is saved as a `.pt` file (default `next_action_probe_{model_type}.pt`):

```python
{
    "model_state_dict": dict,       # PyTorch model weights
    "model_type": str,              # "lr" or "mlp"
    "input_dim": int,               # = hidden_dim (one token, one layer)
    "num_classes": int,             # Number of action classes present
    "hidden_dims": list | None,     # Hidden dimensions (MLP only)
    "dropout": float | None,        # Dropout rate (MLP only)
    "label_to_idx": dict,           # Original action id -> model output index
    "idx_to_label": dict,           # Model output index -> original action id
    "scaler_mean": Tensor | None,   # Normalization mean (if normalize=True)
    "scaler_std": Tensor | None,    # Normalization std (if normalize=True)
    "config": { ... },              # Training configuration
    "results": { ... },             # Training results (accuracy, loss, etc.)
}
```

## Loading and Using a Trained Probe

```python
from telos_interp.commands.train_next_action_probe import NextActionProbe

probe = NextActionProbe.load("/path/to/next_action_probe_lr.pt", device="cuda")

predictions = probe.predict(activations)        # (N,) original action ids
probs = probe.predict_proba(activations)        # (N, num_classes)
print(f"Accuracy: {probe.results['final_accuracy']}")
```

Action id mapping: `{0: LEFT, 1: UP, 2: RIGHT, 3: DOWN}`.

## Training Details

- **Optimizer**: AdamW with configurable weight decay
- **Loss**: Cross-entropy (optionally with balanced class weights)
- **Split**: plain random, i.i.d. over tokens (no trajectory grouping)
- **Progress**: tqdm bar showing train loss, eval accuracy, and best accuracy
- **Metrics**: per-action accuracy / precision / recall / F1, plus balanced accuracy

## Notes

- Only `probe_type=next_action` data is supported.
- The referenced gathered token `.pt` files must still exist at `activations_root` from the
  manifest — this mode does not copy activations.
- Samples with NaN activations are filtered out with a warning.
- **Label remapping**: only action classes present in the data are used; labels are remapped to
  contiguous indices, with `label_to_idx` / `idx_to_label` saved for conversion.
