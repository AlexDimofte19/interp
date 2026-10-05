#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)
ARM=jlens
LENS=jlens
ACT_TRAIN=/workspace/activations/gptoss_direction_mass_l15
ACT_VAL=/workspace/activations/gptoss_direction_mass_l15_eval
TRAJ_TRAIN=/workspace/activations/mass_train2880_view/trajectories
TRAJ_VAL=/workspace/activations/mass_eval720_view/trajectories
PREP=/workspace/prepared/gptoss_p2
ROLLOUTS=/workspace/rollouts/gptoss_p2
OUT=/workspace/prepared/gptoss_p2_local_belief
MODEL=openai/gpt-oss-20b
LAYER=15
DEVICE_MAP=cuda:0
BATCH_SIZE=16
MAX_BATCH_TOKENS=49152
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$ROLLOUTS"
cd "$REPO"
uv run --extra gpu python telos_interp/loudness_analysis/rollouts/run_inference.py \
    --trajectory-paths "${TRAJ_TRAIN}/size*/*.json" \
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
    --branch-cache \
    --skip-existing
uv run --extra gpu python telos_interp/loudness_analysis/rollouts/run_inference.py \
    --trajectory-paths "${TRAJ_VAL}/size*/*.json" \
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
    --branch-cache \
    --skip-existing
uv run python telos_interp/loudness_analysis/rollouts/relabel_manifest_from_rollout.py \
    "${PREP}_${ARM}_train" "${ROLLOUTS}/${ARM}_train" "${OUT}_${ARM}_train" \
    --keep-kinds recorded \
    --report-csv "${OUT}_${ARM}_train_relabel.csv"
uv run python telos_interp/loudness_analysis/rollouts/relabel_manifest_from_rollout.py \
    "${PREP}_${ARM}_val" "${ROLLOUTS}/${ARM}_val" "${OUT}_${ARM}_val" \
    --keep-kinds recorded \
    --report-csv "${OUT}_${ARM}_val_relabel.csv"
