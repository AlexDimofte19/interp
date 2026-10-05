#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)
PREP=/workspace/prepared/gptoss_sentence
PROBES=/workspace/probes/gptoss_sentence
NAME=gptoss_sentence
OUT=/workspace/results/gptoss_sentence/cross_selection_eval
SIGNAL_JSON=$REPO/data/jlens/direction_tokens_full.json
SIGNAL_NAME=direction
LAYER=15
ARMS="jlens logitlens random eos"
SLICES="jlens logitlens random eos"
MODEL_TYPES="lr mlp"
cd "$REPO"
for slice in $SLICES; do
    mkdir -p "$OUT/$slice"
    pids=""
    for arm in $ARMS; do
        for mt in $MODEL_TYPES; do
            uv run python telos_interp/loudness_analysis/rollouts/eval_local_belief.py \
                "${PROBES}/${NAME}_${arm}_l${LAYER}_${mt}.pt" "${PREP}_${slice}_val_local_eq" \
                --signal-json "$SIGNAL_JSON" \
                --signal-name "$SIGNAL_NAME" \
                > "$OUT/$slice/${arm}_${mt}.txt" 2>&1 &
            pids="$pids $!"
        done
    done
    for pid in $pids; do
        wait "$pid"
    done
done
