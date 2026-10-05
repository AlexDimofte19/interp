#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)
PREP=/workspace/prepared/gptoss_sentence
PROBES=/workspace/probes/gptoss_sentence
NAME=gptoss_sentence
LOGS=$PROBES/logs
LAYER=15
ARMS="jlens logitlens random eos"
MODEL_TYPES="lr mlp"
HIDDEN_DIMS=1024
LEARNING_RATE=3e-4
WEIGHT_DECAY=0.001
DROPOUT=0.0
NUM_EPOCHS=50
BATCH_SIZE=512
SEED=42
DEVICE=cuda
mkdir -p "$PROBES" "$LOGS"
cd "$REPO"
for arm in $ARMS; do
    for mt in $MODEL_TYPES; do
        uv run interp-cli train_next_action_probe \
            --train-data-path "${PREP}_${arm}_train_local_eq" \
            --eval-data-path "${PREP}_${arm}_val_local_eq" \
            --output-path "${PROBES}/${NAME}_${arm}_l${LAYER}_${mt}.pt" \
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
            --verbose \
            2>&1 | tee "$LOGS/${NAME}_${arm}_l${LAYER}_${mt}.txt"
    done
done
