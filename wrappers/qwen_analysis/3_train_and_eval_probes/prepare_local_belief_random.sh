#!/usr/bin/env bash
# Qwen P2, step 4c: the LOCAL-BELIEF dataset for the RANDOM arm -- train and val.
#
# One of three arm scripts driven by prepare_local_belief_probe_datasets.sh, which is where
# the rationale for the three stages lives. Run this one alone to rebuild just this arm.
#
# THIS ARM IS THE MATCHED CONTROL, and it is the reason the other two mean anything: the
# loudest tokens are largely the direction words the model has already verbalized, so a lens
# arm on its own cannot say whether the lens found something a uniform draw of the same size
# would not have.
#
# ITS DRAW CANNOT BE RE-MADE. The 60 picks per trajectory were drawn uniformly over the whole
# reasoning chain at gather time, with the same N and seed as the lens arms, and recorded.
# Re-deriving them now would draw from the selection instead of the chain. That is why the
# prepare reads recorded_random rather than the `random` scoring mode, and why the rollout
# replays the record instead of sampling its own cut points.
#
# LENS IS A LABEL HERE, NOT A CHOOSER. The control has no ruler of its own; jlens is named
# only so dir_logmass is populated on the same scale as the jlens arm's, which is what makes
# the two comparable as a covariate. It moves no cutoff.
set -euo pipefail

REPO=/workspace/repo/interp

ARM=random

ACT_TRAIN=/workspace/activations/qwen_p2_selection
TRAJ_TRAIN=/workspace/trajectories/qwen3.6-35b/replayed_single_step/mass_train_576

ACT_VAL=/workspace/activations/qwen_p2_selection_eval
TRAJ_VAL=/workspace/trajectories/qwen3.6-35b/replayed_single_step/mass_eval_144

PREP=/workspace/prepared/qwen_p2              # stage 1 -> ${PREP}_${ARM}_{train,val}
ROLLOUTS=/workspace/rollouts/qwen_p2          # stage 2 -> ${ROLLOUTS}/${ARM}_{train,val}/
OUT=/workspace/prepared/qwen_p2_local         # stage 3 -> ${OUT}_${ARM}_{train,val}

MODEL=Qwen/Qwen3.6-35B-A3B
LAYER=27
PROBE_TYPE=next_action
LENS=jlens          # a scale for dir_logmass only; see the header
DEVICE_MAP=cuda:0   # NOT "auto": spreading this MoE over several GPUs produces NaNs
BATCH_SIZE=4
MAX_BATCH_TOKENS=49152

mkdir -p "$ROLLOUTS"
cd "$REPO"

# ------------------------------------------------------------------ 1. PREPARE
uv run interp-cli prepare_activations_for_probing \
    --activations-dir "$ACT_TRAIN" \
    --trajectories-dir "$TRAJ_TRAIN" \
    --probe-type "$PROBE_TYPE" \
    --layers "$LAYER" \
    --steps all \
    --output-indices all \
    --token-selection "recorded_${ARM}" \
    --output-path "${PREP}_${ARM}_train" \
    --verbose

uv run interp-cli prepare_activations_for_probing \
    --activations-dir "$ACT_VAL" \
    --trajectories-dir "$TRAJ_VAL" \
    --probe-type "$PROBE_TYPE" \
    --layers "$LAYER" \
    --steps all \
    --output-indices all \
    --token-selection "recorded_${ARM}" \
    --output-path "${PREP}_${ARM}_val" \
    --verbose

# ------------------------------------------------------------------ 2. ROLLOUT
uv run --extra gpu python telos_interp/loudness_analysis/rollouts/run_inference.py \
    --trajectory-paths "$TRAJ_TRAIN" \
    --output-dir "${ROLLOUTS}/${ARM}_train" \
    --model-id "$MODEL" \
    --strategy recorded_selection \
    --selection-arm "$ARM" \
    --selection-root "$ACT_TRAIN" \
    --lens-root "$ACT_TRAIN" \
    --lens "$LENS" \
    --loudness-layer "$LAYER" \
    --batch-size "$BATCH_SIZE" \
    --max-batch-tokens "$MAX_BATCH_TOKENS" \
    --device-map "$DEVICE_MAP" \
    --skip-existing

uv run --extra gpu python telos_interp/loudness_analysis/rollouts/run_inference.py \
    --trajectory-paths "$TRAJ_VAL" \
    --output-dir "${ROLLOUTS}/${ARM}_val" \
    --model-id "$MODEL" \
    --strategy recorded_selection \
    --selection-arm "$ARM" \
    --selection-root "$ACT_VAL" \
    --lens-root "$ACT_VAL" \
    --lens "$LENS" \
    --loudness-layer "$LAYER" \
    --batch-size "$BATCH_SIZE" \
    --max-batch-tokens "$MAX_BATCH_TOKENS" \
    --device-map "$DEVICE_MAP" \
    --skip-existing

# ------------------------------------------------------------------ 3. RELABEL
uv run python telos_interp/loudness_analysis/rollouts/relabel_manifest_from_rollout.py \
    "${PREP}_${ARM}_train" "${ROLLOUTS}/${ARM}_train" "${OUT}_${ARM}_train" \
    --keep-kinds recorded \
    --report-csv "${OUT}_${ARM}_train_relabel.csv"

uv run python telos_interp/loudness_analysis/rollouts/relabel_manifest_from_rollout.py \
    "${PREP}_${ARM}_val" "${ROLLOUTS}/${ARM}_val" "${OUT}_${ARM}_val" \
    --keep-kinds recorded \
    --report-csv "${OUT}_${ARM}_val_relabel.csv"
