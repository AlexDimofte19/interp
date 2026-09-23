#!/usr/bin/env bash
# gpt-oss P2, loudness evaluation step 1b: the ROW SOURCE join_heldout_rollouts.sh iterates.
#
# The gpt-oss twin of wrappers/qwen_analysis/4_loudness_evaluation/build_commitment_per_token.sh.
# join_rollouts.py takes its rows from --commitment-csv: one row per (name, step, token_idx),
# carrying the sentence coordinates and the loudness. join_rollout_answers.py's output IS that
# row source. Its defaults are already the heldout360 set; every input is still named here,
# so this file records the run on its own.
#
# THE ROLLOUT IS DENSE HERE (stride 1, ../2_dataset_creation/3_rollouts/rollout_heldout_every_token.sh),
# so unlike Qwen's stride-64 arm the commitment boundary IS resolvable to the token. This join
# still writes `convinced_sentence_idx` blank, because join_rollout_answers.py never fills it.
# Do not fill it by copying the rollout's own `convinced_sentence_idx`: that is the cutoff's
# ORDINAL IN THE EVAL LIST, not a sentence index. A commitment analysis on gpt-oss needs the
# boundary computed on the eos grid first. Until then the consumer runs `--commitment off`.
#
# --layer 15 is the probes' layer, and --lenses writes both. The run cross-checks every joined
# loudness against the value the rollout recorded at the same cutoff and exits non-zero on any
# mismatch: a clean run is the indexing proof.
#
# CPU only -- no torch, no model.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)   # repo root

RESULTS=/workspace/results/gptoss_p2_local_belief/heldout
ROLLOUTS=/workspace/reasoning_theatre/rollout_strategies_heldout360/every_token   # the dense arm, reused
TRAJ=/workspace/trajectories/heldout360
LENS_ROOT=/workspace/activations/heldout360_lens
NAMES=/workspace/trajectories/heldout360_names.txt   # exactly the 360; the rollout dir also holds logs/
OUT=$RESULTS/commitment_per_token.csv

LAYER=15
LENSES=jlens,logitlens

mkdir -p "$RESULTS"
cd "$REPO"

uv run python telos_interp/loudness_analysis/join_rollout_answers.py \
    --rollout-dir "$ROLLOUTS" \
    --trajectories "$TRAJ" \
    --lens-root "$LENS_ROOT" \
    --names-file "$NAMES" \
    --lenses "$LENSES" \
    --layer "$LAYER" \
    --out "$OUT"
