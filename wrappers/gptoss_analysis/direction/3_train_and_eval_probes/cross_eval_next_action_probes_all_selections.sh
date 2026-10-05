#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
PROBES=/workspace/probes/gptoss_p2_local_belief
NAME=gptoss_p2_local_belief
DATA=/workspace/prepared/gptoss_p2_local_belief
OUT=/workspace/results/gptoss_p2_local_belief/cross_selection_eval
SIGNAL_JSON=$REPO/data/jlens/direction_tokens_full.json
SIGNAL_NAME=direction
LAYER=15
ARMS="jlens logitlens random"
SLICES="jlens logitlens random"
MODEL_TYPES="lr mlp"
cd "$REPO"
for slice in $SLICES; do
    mkdir -p "$OUT/$slice"
    pids=""
    for arm in $ARMS; do
        for mt in $MODEL_TYPES; do
            uv run python telos_interp/loudness_analysis/rollouts/eval_local_belief.py \
                "${PROBES}/${NAME}_${arm}_l${LAYER}_${mt}.pt" "${DATA}_${slice}_val" \
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
