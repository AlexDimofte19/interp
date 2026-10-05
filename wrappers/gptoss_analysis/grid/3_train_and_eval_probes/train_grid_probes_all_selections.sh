#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
DATA=/workspace/prepared/gptoss_p2_grid
PROBES=/workspace/probes/gptoss_p2_grid
NAME=gptoss_p2_grid
LOGS=$PROBES/logs
LAYER=14
ARMS="jlens logitlens random"
MODEL_TYPES="lr mlp"
HIDDEN_DIMS=1024
LEARNING_RATE=3e-4
WEIGHT_DECAY=0.001
NUM_EPOCHS=50
DROPOUT=0.0
BATCH_SIZE=2048
SEED=42
DEVICE=cuda
mkdir -p "$PROBES" "$LOGS"
cd "$REPO"
for arm in $ARMS; do
    for mt in $MODEL_TYPES; do
        uv run interp-cli train_cognitive_map_probe \
            --train-data-path "${DATA}_${arm}_train" \
            --eval-data-path "${DATA}_${arm}_val" \
            --output-path "${PROBES}/${NAME}_${arm}_multiclass_l${LAYER}_${mt}.pt" \
            --model-type "$mt" \
            --hidden-dims "$HIDDEN_DIMS" \
            --learning-rate "$LEARNING_RATE" \
            --weight-decay "$WEIGHT_DECAY" \
        --dropout "$DROPOUT" \
            --num-epochs "$NUM_EPOCHS" \
            --batch-size "$BATCH_SIZE" \
            --class-weight balanced \
            --normalize \
            --seed "$SEED" \
            --device "$DEVICE" \
            --cache-activations \
            --verbose \
            2>&1 | tee "$LOGS/${NAME}_${arm}_multiclass_l${LAYER}_${mt}.txt"
    done
done
