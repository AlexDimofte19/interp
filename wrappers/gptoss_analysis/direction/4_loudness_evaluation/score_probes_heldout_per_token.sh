#!/usr/bin/env bash
# gpt-oss P2, loudness evaluation step 1: every probe on every reasoning token of heldout360.
#
# The gpt-oss twin of wrappers/qwen_analysis/4_loudness_evaluation/score_probes_heldout_per_token.sh.
# One row per (trajectory, step, token), carrying each probe's verdict and both lenses'
# loudness columns.
#
# TWO TREES, NOT ONE. Qwen's held-out gather pinned the .pt files and the CSVs to one layer in
# one tree, so it omits --lens-dir. gpt-oss reuses two existing trees:
#   --activations-dir heldout360_l15    a layer-15 .pt for every reasoning token (87,221)
#   --lens-dir        heldout360_lens   both lenses' CSVs and direction-mass tables at 7:23
# heldout360_l15 carries only a jlens table, so pointing the lens side at it would leave
# every logitlens column blank.
#
# --full-probs is required by join_heldout_rollouts.sh, which reads the per-action probabilities.
#
# --read-threads issues the per-token .pt reads concurrently (MooseFS latency, not bandwidth).
# --cache-activations packs each trajectory into one file, so the next pass over the tree reads
# 360 files, not 87k. Neither changes a value (tests/test_score_probes_per_token.py).
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)   # repo root

PROBES=/workspace/probes/gptoss_p2_local_belief
ACT=/workspace/activations/heldout360_l15
LENS_DIR=/workspace/activations/heldout360_lens
TRAJ=/workspace/trajectories/heldout360
OUT=/workspace/results/gptoss_p2_local_belief/heldout/per_token_scores.csv

SIGNAL_JSON=/workspace/jlens/direction_tokens_full.json
SIGNAL_NAME=direction

LAYER=15
PROBE_TYPE=next_action
DEVICE=cuda
READ_THREADS=16        # latency-bound reads; see the header

mkdir -p "$(dirname "$OUT")"
cd "$REPO"

uv run --extra gpu python telos_interp/loudness_analysis/score_probes_per_token.py \
    --probe-type "$PROBE_TYPE" \
    --probe "${PROBES}/gptoss_p2_local_belief_jlens_l${LAYER}_lr.pt" \
    --probe "${PROBES}/gptoss_p2_local_belief_jlens_l${LAYER}_mlp.pt" \
    --probe "${PROBES}/gptoss_p2_local_belief_logitlens_l${LAYER}_lr.pt" \
    --probe "${PROBES}/gptoss_p2_local_belief_logitlens_l${LAYER}_mlp.pt" \
    --probe "${PROBES}/gptoss_p2_local_belief_random_l${LAYER}_lr.pt" \
    --probe "${PROBES}/gptoss_p2_local_belief_random_l${LAYER}_mlp.pt" \
    --activations-dir "$ACT" \
    --lens-dir "$LENS_DIR" \
    --trajectories-dir "$TRAJ" \
    --signal-json "$SIGNAL_JSON" \
    --signal-name "$SIGNAL_NAME" \
    --layer "$LAYER" \
    --full-probs \
    --out "$OUT" \
    --read-threads "$READ_THREADS" \
    --cache-activations \
    --device "$DEVICE"
