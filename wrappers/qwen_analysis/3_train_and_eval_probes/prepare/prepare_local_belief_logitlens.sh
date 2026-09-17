#!/usr/bin/env bash
# Qwen P2, step 4b: the LOCAL-BELIEF dataset for the LOGITLENS arm -- train and val.
#
# One of three arm scripts driven by ../prepare_local_belief_probe_datasets.sh, which is where
# the rationale for the three stages lives. Run this one alone to rebuild just this arm.
#
# THE ARM IS THE ONLY THING THAT DIFFERS between this file and prepare_local_belief_jlens.sh (beside it):
# same tree, same trajectories, same layer, same rollout, same relabel. ARM below names which
# set of picks in {stem}_jlens_selection.json is read -- here, the tokens the LOGIT lens
# scored loudest at layer 27, which at that depth overlap the jlens arm's only about half.
#
# LENS FOLLOWS THE ARM. dir_logmass is a covariate, not a cut point, but an arm selected by
# one lens and annotated with the other's mass would be describing its tokens with a ruler
# that did not choose them.
set -euo pipefail

REPO=/workspace/repo/interp

ARM=logitlens

ACT_TRAIN=/workspace/activations/qwen_p2_selection
TRAJ_TRAIN=/workspace/trajectories/qwen3.6-35b/replayed_single_step/mass_train_576

ACT_VAL=/workspace/activations/qwen_p2_selection_eval
TRAJ_VAL=/workspace/trajectories/qwen3.6-35b/replayed_single_step/mass_eval_144

PREP=/workspace/prepared/qwen_p2              # stage 1 -> ${PREP}_${ARM}_{train,val}
ROLLOUTS=/workspace/rollouts/qwen_p2          # stage 2 -> ${ROLLOUTS}/${ARM}_{train,val}/
OUT=/workspace/prepared/qwen_p2_local_belief         # stage 3 -> ${OUT}_${ARM}_{train,val}

MODEL=Qwen/Qwen3.6-35B-A3B
LAYER=27
PROBE_TYPE=next_action
LENS=logitlens      # the ruler that chose this arm's tokens; see the header
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
