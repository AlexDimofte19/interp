#!/usr/bin/env bash
# Qwen P2, loudness evaluation step 1b: the ROW SOURCE join_heldout_rollouts.sh iterates.
#
# join_rollouts.py takes its rows from --commitment-csv, not from the probe table: one row
# per (name, step, token_idx), carrying the sentence coordinates and the loudness. The
# gpt-oss line inherited that file from entries 39/40/41; no Qwen one existed, and this is
# the script that builds it.
#
# NOTHING HERE IS NEW MACHINERY. join_rollout_answers.py already joined this rollout, these
# mass tables and these trajectory JSONs; it simply did not carry the eos-grid columns
# through. They now travel in SENTENCE_FIELDS, so its output IS the row source.
#
# WHY THERE IS NO COMMITMENT BOUNDARY IN IT. `convinced_sentence_idx` is written blank on
# purpose. The boundary is a property of a whole step, and this arm is STRIDED (every 64th
# token), so it is resolved only to +-64 tokens and a relapse between two sampled cutoffs is
# invisible. Do not fill it by copying the rollout's own `convinced_sentence_idx`: that
# number is the cutoff's ORDINAL IN THE EVAL LIST, not a sentence index -- on the Qwen
# held-out set ordinal 191 is sentence 725 -- and join_rollouts.py subtracts it from a real
# `sentence_idx`. The consumer side of this decision is `--commitment off`.
#
# --layer 27 is the probes' layer, and --lenses writes both so the join can pick either with
# --lens without a re-run. The run cross-checks every joined loudness against the value the
# rollout recorded at the same cutoff and exits non-zero on any mismatch: that is the one
# thing this join cannot get wrong quietly, so a clean run is the indexing proof.
#
# CPU only -- no torch, no model. Minutes, not hours.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)   # repo root, as the stage-1/2 wrappers do

RESULTS=/workspace/results/qwen_p2_local_belief/heldout
ROLLOUTS=/workspace/rollouts/qwen_p2_heldout_every_token
TRAJ=/workspace/trajectories/qwen3.6-35b/replayed_single_step/heldout_72
LENS_ROOT=/workspace/activations/qwen_p2_heldout
OUT=$RESULTS/commitment_per_token.csv

LAYER=27
LENSES=jlens,logitlens

mkdir -p "$RESULTS"
cd "$REPO"

# --names-file "" takes every rollout in --rollout-dir, which is the 70 held-out trajectories.
uv run python telos_interp/loudness_analysis/join_rollout_answers.py \
    --rollout-dir "$ROLLOUTS" \
    --trajectories "$TRAJ" \
    --lens-root "$LENS_ROOT" \
    --names-file "" \
    --lenses "$LENSES" \
    --layer "$LAYER" \
    --out "$OUT"
