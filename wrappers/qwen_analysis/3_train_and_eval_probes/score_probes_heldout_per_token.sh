#!/usr/bin/env bash
# Qwen P2, step 7b: every probe on every reasoning token of the held-out 72.
#
# One row per (trajectory, step, token), carrying each probe's verdict and the loudness
# columns. Both lenses are written -- score_probes_per_token.py loops over jlens and
# logitlens with no flag -- so each row gets 4 columns per lens: count, mass at LAYER, the
# token's best layer, and the mass there.
#
# --full-probs is required by join_heldout_rollouts.sh, which reads the per-action
# probabilities. --lens-dir is omitted: the held-out gather pinned the .pt files and the
# CSVs to the same layer in one tree.
set -euo pipefail

REPO=/workspace/repo/interp

PROBES=/workspace/probes/qwen_p2_local_belief
ACT=/workspace/activations/qwen_p2_heldout
TRAJ=/workspace/trajectories/qwen3.6-35b/replayed_single_step/heldout_72
OUT=/workspace/results/qwen_p2_local_belief/heldout/per_token_scores.csv

SIGNAL_JSON=/workspace/repo/interp/data/jlens/qwen/direction_tokens_full_qwen3-6-35b-a3b.json
SIGNAL_NAME=direction

LAYER=27
PROBE_TYPE=next_action
DEVICE=cuda

mkdir -p "$(dirname "$OUT")"
cd "$REPO"

uv run --extra gpu python telos_interp/loudness_analysis/score_probes_per_token.py \
    --probe-type "$PROBE_TYPE" \
    --probe "${PROBES}/qwen_p2_local_belief_jlens_l${LAYER}_lr.pt" \
    --probe "${PROBES}/qwen_p2_local_belief_jlens_l${LAYER}_mlp.pt" \
    --probe "${PROBES}/qwen_p2_local_belief_logitlens_l${LAYER}_lr.pt" \
    --probe "${PROBES}/qwen_p2_local_belief_logitlens_l${LAYER}_mlp.pt" \
    --probe "${PROBES}/qwen_p2_local_belief_random_l${LAYER}_lr.pt" \
    --probe "${PROBES}/qwen_p2_local_belief_random_l${LAYER}_mlp.pt" \
    --activations-dir "$ACT" \
    --trajectories-dir "$TRAJ" \
    --signal-json "$SIGNAL_JSON" \
    --signal-name "$SIGNAL_NAME" \
    --layer "$LAYER" \
    --full-probs \
    --out "$OUT" \
    --device "$DEVICE"
