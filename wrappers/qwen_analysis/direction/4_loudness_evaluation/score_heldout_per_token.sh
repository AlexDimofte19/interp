#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
PROBES=/workspace/probes/qwen_p2_local_belief
NAME=qwen_p2_local_belief
ACT=/workspace/activations/qwen_p2_heldout
TRAJ=/workspace/trajectories/qwen3.6-35b/replayed_single_step/heldout_72
OUT=/workspace/results/qwen_p2_local_belief/heldout/per_token_scores.csv
SIGNAL_JSON=$REPO/data/jlens/qwen/direction_tokens_full_qwen3-6-35b-a3b.json
SIGNAL_NAME=direction
LAYER=27
PROBE_TYPE=next_action
probes=()
for arm in jlens logitlens random; do
    for mt in lr mlp; do
        probes+=(--probe "${PROBES}/${NAME}_${arm}_l${LAYER}_${mt}.pt")
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
    --full-probs \
    --out "$OUT" \
    --read-threads 16 \
    --cache-activations \
    --device cuda
