#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
PROBES=/workspace/probes/gptoss_p2_grid
NAME=gptoss_p2_grid
DATA=/workspace/prepared/gptoss_p2_grid
OUT=/workspace/results/gptoss_p2_grid/cross_selection_eval_multiclass
SIGNAL_JSON=$REPO/data/jlens/grid_tokens_pruned.json
SIGNAL_NAME=grid
LAYER=14
ARMS="jlens logitlens random"
SLICES="jlens logitlens random"
MODEL_TYPES="lr mlp"
cd "$REPO"
for slice in $SLICES; do
    mkdir -p "$OUT/$slice"
    pids=""
    for arm in $ARMS; do
        for mt in $MODEL_TYPES; do
            uv run python telos_interp/loudness_analysis/eval_grid_probe.py \
                "${PROBES}/${NAME}_${arm}_multiclass_l${LAYER}_${mt}.pt" "${DATA}_${slice}_val" \
                --signal-json "$SIGNAL_JSON" \
                --signal-name "$SIGNAL_NAME" \
            --cache-activations \
                > "$OUT/$slice/${arm}_${mt}.txt" 2>&1 &
            pids="$pids $!"
        done
    done
    for pid in $pids; do
        wait "$pid"
    done
done
