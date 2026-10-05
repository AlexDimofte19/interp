#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
PROBES=/workspace/probes/qwen_p2_grid
NAME=qwen_p2_grid
DATA=/workspace/prepared/qwen_p2_grid_heldout
OUT=/workspace/results/qwen_p2_grid/heldout_multiclass
SIGNAL_JSON=$REPO/data/jlens/qwen/grid_tokens_pruned_qwen3-6-35b-a3b.json
LAYER=27
mkdir -p "$OUT"
cd "$REPO"
for arm in jlens logitlens random; do
    for mt in lr mlp; do
        uv run python telos_interp/loudness_analysis/eval_grid_probe.py \
            "${PROBES}/${NAME}_${arm}_multiclass_l${LAYER}_${mt}.pt" "$DATA" \
            --signal-json "$SIGNAL_JSON" \
            --signal-name grid \
            --cache-activations \
            --device cuda \
            --out-json "$OUT/${arm}_multiclass_${mt}.json" \
            > "$OUT/${arm}_multiclass_${mt}.txt" 2>&1
    done
done
