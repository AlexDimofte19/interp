#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
PROBES=/workspace/probes/qwen_p2_grid
NAME=qwen_p2_grid
ACT=/workspace/activations/qwen_p2_heldout_grid
TRAJ=/workspace/trajectories/qwen3.6-35b/replayed_single_step/heldout_72
OUT=/workspace/results/qwen_p2_grid/heldout_multiclass/per_token_scores.csv
SIGNAL_JSON=$REPO/data/jlens/qwen/grid_tokens_pruned_qwen3-6-35b-a3b.json
SIGNAL_NAME=grid
LAYER=27
PROBE_TYPE=grid_multiclass
probes=()
for arm in jlens logitlens random; do
    for mt in lr mlp; do
        probes+=(--probe "${PROBES}/${NAME}_${arm}_multiclass_l${LAYER}_${mt}.pt")
    done
done
mkdir -p "$(dirname "$OUT")"
cd "$REPO"
uv run --extra gpu python telos_interp/loudness_analysis/score_probes_per_token.py \
    --probe-type "$PROBE_TYPE" \
    "${probes[@]}" \
    --activations-dir "$ACT" \
    --trajectories-dir "$TRAJ" \
    --signal-json "$SIGNAL_JSON" \
    --signal-name "$SIGNAL_NAME" \
    --layer "$LAYER" \
    --pad-to-size 15 \
    --max-cells 25 \
    --seed 42 \
    --out "$OUT" \
    --read-threads 16 \
    --cache-activations \
    --device cuda
