# gather_activations

Extract model activations from trajectory JSON files produced by `get_trajectory`.

## Overview

This command processes trajectory JSON files and extracts activations for specified token categories, layers, and steps. It uses efficient extraction strategies:

- **prompt_prefix** is extracted once per trajectory (activations are step-independent due to causal attention)
- **Step-dependent categories** (grid_state, prompt_suffix, output) are extracted in a single forward pass per step

## Output Structure

```
{output_dir}/
  {trajectory_name}/
    {model_name}/
      layer_{N}/
        step_{M}/
          prompt_prefix/
            0.pt, 1.pt, ...
          prompt_suffix/
            0.pt, 1.pt, ...
          grid_state/
            0.pt, 1.pt, ...
          output/
            0.pt, 1.pt, ...
```

Each `.pt` file contains the activation tensor for a single token at the specified layer and step.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trajectory_paths` | list[str] | required | Paths to trajectory JSON files (glob patterns supported) |
| `output_dir` | str | "out" | Base output directory |
| `layers` | str | "all" | Layer indices to extract (e.g., "all", "0:10", "-1", "7,15") |
| `steps` | str | "all" | Step indices to extract (e.g., "all", "0:5", "-1") |
| `prompt_prefix_indices` | str \| None | None | Token indices for prompt_prefix (None = skip) |
| `prompt_suffix_indices` | str \| None | None | Token indices for prompt_suffix (None = skip) |
| `grid_state_indices` | str \| None | None | Token indices for grid_state (None = skip) |
| `output_indices` | str \| None | None | Token indices for output (None = skip, "all" = all available) |
| `device_map` | str | "auto" | Device mapping for model loading |
| `torch_dtype` | str | "auto" | Torch dtype ("auto", "bfloat16", "float16") |
| `debug` | bool | False | Print first truncated input text to verify format |

### Index Specification Format

The `layers`, `steps`, and token index parameters support flexible specification:

- `"all"` - All available indices
- `"7"` - Single index
- `"7,15,23"` - Multiple specific indices
- `"0:10"` - Range (inclusive, 0 to 10)
- `"-1"` - Last index
- `"-3:-1"` - Last 3 indices

## Examples

### Extract all layers for the last output token

```bash
interp-cli gather_activations \
    --trajectory-paths "/path/to/trajectories/*.json" \
    --output-dir /path/to/activations \
    --layers all \
    --steps 0 \
    --output-indices -1
```

### Extract specific layers for grid_state tokens

```bash
interp-cli gather_activations \
    --trajectory-paths "/path/to/trajectories/traj_*.json" \
    --output-dir /path/to/activations \
    --layers "7,15,23,31" \
    --steps all \
    --grid-state-indices all
```

### Extract prompt prefix and suffix for first 10 layers

```bash
interp-cli gather_activations \
    --trajectory-paths /path/to/traj1.json /path/to/traj2.json \
    --output-dir /path/to/activations \
    --layers "0:9" \
    --steps 0 \
    --prompt-prefix-indices all \
    --prompt-suffix-indices all
```

### Extract with specific dtype for large models

```bash
interp-cli gather_activations \
    --trajectory-paths "/path/to/trajectories/*.json" \
    --output-dir /path/to/activations \
    --layers all \
    --output-indices -1 \
    --torch-dtype bfloat16 \
    --device-map auto
```

## Notes

- All trajectory files must use the same model
- Existing output folders are skipped (allows resuming interrupted extractions)
- Out-of-bounds step indices are clamped, so "0:10" works even if fewer steps exist
- The command uses `nnterp.StandardizedTransformer` for model loading

### Multi-GPU Warning

**Important**: When using `device_map="auto"` with multiple GPUs, activations may contain NaN values due to issues with how nnterp/nnsight handles MoE (Mixture of Experts) models across devices.

**Recommended**: Always use a single GPU for activation extraction:

```bash
CUDA_VISIBLE_DEVICES=0 interp-cli gather_activations ...
```

If your model doesn't fit on a single GPU:
- The model will automatically use quantization or memory optimization when possible
- You may see warnings about MXFP4 falling back to bf16 - this is normal
- Memory warnings like "We will use 90% of the memory" are informational
