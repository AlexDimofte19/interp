#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
PROBES=/workspace/probes/gptoss_p2_local_belief
NAME=gptoss_p2_local_belief
ACT=/workspace/activations/heldout360_l15
TRAJ=/workspace/trajectories/heldout360
OUT=/workspace/results/gptoss_p2_local_belief/heldout/per_token_scores.csv
SIGNAL_JSON=$REPO/data/jlens/direction_tokens_full.json
SIGNAL_NAME=direction
LAYER=15
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
