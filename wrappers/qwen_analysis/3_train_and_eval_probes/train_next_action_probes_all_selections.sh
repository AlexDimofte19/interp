#!/usr/bin/env bash
# Qwen P2, step 5: train one next_action probe per arm. Training only -- no evaluation.
#
# The sibling of prepare_probe_datasets.sh -- it consumes the six manifests that script
# wrote, so keep OUT below equal to its OUT.
#
# WHAT EACH ARM REPORTS, AND WHAT IT DOES NOT. Every train command scores its probe on the
# VAL half of its OWN selection: the jlens probe on jlens-selected tokens, the control on
# uniformly drawn ones. Three arms therefore report three numbers measured on three
# different populations, which is a comparison of arms and NOT a statement about the
# model's tokens at large. Reading all three on one shared population -- every reasoning
# token of the held-out 72 -- is a separate script; on the gpt-oss line the ordering
# INVERTED between the two (ICLR entry 49), so neither substitutes for the other.
#
# WHY THE CONTROL IS HERE AT ALL. The loudest tokens are largely the direction words the
# model has already verbalized, so a lens arm on its own says nothing. `random` is what
# reports whether the lens found anything a uniform draw of the same size would not have.
#
# NO SPLIT, NO --eval-split. --eval-data-path is a separate gather (mass_eval_144), so the
# trainer's internal row-level split never runs and no trajectory can land in both halves.
#
# MODEL_TYPE is one variable on purpose: run it as lr, then flip to mlp and run again. The
# probe filenames carry the type, so the second pass adds files rather than overwriting.
set -euo pipefail

REPO=/workspace/repo/interp

OUT=/workspace/prepared/qwen_p2        # reads ${OUT}_${arm}_{train,val}
PROBES=/workspace/probes/qwen_p2

LAYER=27
MODEL_TYPE=lr          # flip to mlp and re-run for the second half of the sweep

# Probe hyperparameters: the general_probe_train.sh values, with the batch size lowered
# because a selection is ~60 rows per trajectory, not ~C cells per trajectory.
HIDDEN_DIMS=1024
LEARNING_RATE=3e-4
WEIGHT_DECAY=0.001
NUM_EPOCHS=50
BATCH_SIZE=512
SEED=42
DEVICE=cuda

mkdir -p "$PROBES"
cd "$REPO"

# ------------------------------------------------------------------- jlens
uv run interp-cli train_next_action_probe \
    --train-data-path "${OUT}_jlens_train" \
    --eval-data-path "${OUT}_jlens_val" \
    --output-path "${PROBES}/qwen_p2_jlens_l${LAYER}_${MODEL_TYPE}.pt" \
    --model-type "$MODEL_TYPE" \
    --hidden-dims "$HIDDEN_DIMS" \
    --learning-rate "$LEARNING_RATE" \
    --weight-decay "$WEIGHT_DECAY" \
    --num-epochs "$NUM_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --class-weight balanced \
    --normalize \
    --seed "$SEED" \
    --device "$DEVICE" \
    --verbose

# --------------------------------------------------------------- logitlens
uv run interp-cli train_next_action_probe \
    --train-data-path "${OUT}_logitlens_train" \
    --eval-data-path "${OUT}_logitlens_val" \
    --output-path "${PROBES}/qwen_p2_logitlens_l${LAYER}_${MODEL_TYPE}.pt" \
    --model-type "$MODEL_TYPE" \
    --hidden-dims "$HIDDEN_DIMS" \
    --learning-rate "$LEARNING_RATE" \
    --weight-decay "$WEIGHT_DECAY" \
    --num-epochs "$NUM_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --class-weight balanced \
    --normalize \
    --seed "$SEED" \
    --device "$DEVICE" \
    --verbose

# ------------------------------------------------------------------ random
uv run interp-cli train_next_action_probe \
    --train-data-path "${OUT}_random_train" \
    --eval-data-path "${OUT}_random_val" \
    --output-path "${PROBES}/qwen_p2_random_l${LAYER}_${MODEL_TYPE}.pt" \
    --model-type "$MODEL_TYPE" \
    --hidden-dims "$HIDDEN_DIMS" \
    --learning-rate "$LEARNING_RATE" \
    --weight-decay "$WEIGHT_DECAY" \
    --num-epochs "$NUM_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --class-weight balanced \
    --normalize \
    --seed "$SEED" \
    --device "$DEVICE" \
    --verbose
