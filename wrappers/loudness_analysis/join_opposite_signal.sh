#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUT=/workspace/loudness_probe_performance_analysis/tables
R=/workspace/results
A=/workspace/activations
J=$REPO/data/jlens
cd "$REPO"

join() {
    local run=$1 table=$2 tree=$3 name=$4 sig=$5 layer=$6
    mkdir -p "$OUT/$run"
    uv run python -m telos_interp.loudness_analysis.join_signal_loudness \
        --table "$table" \
        --lens-root "$tree" \
        --signal-name "$name" \
        --signal-json "$sig" \
        --layer "$layer" \
        --out "$OUT/$run/per_token_scores.csv"
}

join gptoss_direction "$R/gptoss_p2_local_belief/heldout/per_token_scores.csv" \
    "$A/heldout360_l14_grid" grid "$J/grid_tokens_pruned.json" 14
join gptoss_grid "$R/gptoss_p2_grid/heldout_multiclass/per_token_scores.csv" \
    "$A/heldout360_l15" direction "$J/direction_tokens_full.json" 15
join qwen_direction "$R/qwen_p2_local_belief/heldout/per_token_scores.csv" \
    "$A/qwen_p2_heldout_grid" grid "$J/qwen/grid_tokens_pruned_qwen3-6-35b-a3b.json" 27
join qwen_grid "$R/qwen_p2_grid/heldout_multiclass/per_token_scores.csv" \
    "$A/qwen_p2_heldout" direction "$J/qwen/direction_tokens_full_qwen3-6-35b-a3b.json" 27
