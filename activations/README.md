---
license: apache-2.0
---

# Single-step Activations for Probe Training

This folder contains pre-computed activations for the trajectories in [project-telos/trajectories_train_single_step](https://huggingface.co/datasets/project-telos/trajectories_train_single_step). Specifically, we extract activations for the last 3 positions of the prompt and the last three tokens of the reasoning block at layers 7, 15 and 23. This is done using the `interp-cli` from `github.com/SPAR-Telos/interp` as:

```shell
# Needed to fit GPT-OSS-20B in 48GB GPU RAM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SIZES=(5 7 9 11 13 15)

for size in "${SIZES[@]}"; do
    echo "Processing size ${size}..."

    CUDA_VISIBLE_DEVICES=0 interp-cli gather_activations \
        --trajectory-paths "trajectories_train_single_step/size${size}/*.json" \
        --prompt-suffix-indices '-3:-1' \
        --output-indices '-16:-14' \
        --layers '7,15,23' \
        --output-dir "activations_train_single_step/size${size}"

    echo "Finished size ${size}"
done
```

## How to Use

Once generated, activations can be prepared for probe training using `interp-cli prepare_activations_for_probing`. The resulting file can be used to train distance probes, action sequence probes or cognitive map probes (see CLI for details). For example, for the data format required by cognitive map probes:

```bash
interp-cli prepare_activations_for_probing \
    --activations-dir /path/to/activations_train_single_step \
    --trajectories-dir /path/to/trajectories_train_single_step \
    --probe-type grid_tile \
    # Use all output indices (last 3 reasoning tokens)
    --output-indices all \ 
    # Ensures an equal distribution of grid tile types in the data
    --balance-classes-per-trajectory 
```