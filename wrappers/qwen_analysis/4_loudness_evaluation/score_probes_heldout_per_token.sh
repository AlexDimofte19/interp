#!/usr/bin/env bash
# Qwen P2, loudness evaluation step 1: every probe on every reasoning token of the held-out 72.
#
# One row per (trajectory, step, token), carrying each probe's verdict and the loudness
# columns. Both lenses are written -- score_probes_per_token.py loops over jlens and
# logitlens with no flag -- so each row gets 4 columns per lens: count, mass at LAYER, the
# token's best layer, and the mass there.
#
# --full-probs is required by join_heldout_rollouts.sh, which reads the per-action
# probabilities. --lens-dir is omitted: the held-out gather pinned the .pt files and the
# CSVs to the same layer in one tree.
#
# THIS TREE IS 1,169,734 SINGLE-TOKEN .pt ON MOOSEFS, and unthinned on purpose
# (heldout_sample.sh runs SAMPLE_PERCENT=1.0, because a token absent from the tree cannot be
# scored later without re-gathering). Each tensor is 5,588 bytes and the whole job reads
# 6.5 GB, so the cost is never bandwidth -- it is one network round trip per token, measured
# at 28 ms against ~0.1 ms for the same read on local NVMe. Serial and with a separate
# exists() stat, the first version of this run took 6.7 hours at 49 tokens/s.
#
# --read-threads issues those reads concurrently, which is what makes THIS run affordable;
# --cache-activations packs each trajectory into one file so the NEXT run over the same tree
# -- a second signal, another probe set, the grid_tile probe type -- reads 70 files instead of
# 1.17M. The cache is the one that does nothing for a first pass, so both are set.
#
# Neither changes a value: tests/test_score_probes_per_token.py asserts that the serial read,
# the threaded read, a cache-filling run and a cache-served run write byte-identical CSVs.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)   # repo root, as the stage-1/2 wrappers do

PROBES=/workspace/probes/qwen_p2_local_belief
ACT=/workspace/activations/qwen_p2_heldout
TRAJ=/workspace/trajectories/qwen3.6-35b/replayed_single_step/heldout_72
OUT=/workspace/results/qwen_p2_local_belief/heldout/per_token_scores.csv

SIGNAL_JSON=/workspace/repo/interp/data/jlens/qwen/direction_tokens_full_qwen3-6-35b-a3b.json
SIGNAL_NAME=direction

LAYER=27
PROBE_TYPE=next_action
DEVICE=cuda
READ_THREADS=16        # latency-bound reads; see the header

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
    --read-threads "$READ_THREADS" \
    --cache-activations \
    --device "$DEVICE"
