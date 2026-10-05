#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)
ARM=eos
STRATEGY=eos
LENS=jlens
TRAJ_TRAIN=/workspace/activations/mass_train2880_view/trajectories
TRAJ_VAL=/workspace/activations/mass_eval720_view/trajectories
NAMES_TRAIN=/workspace/splits/mass_train_2880.txt
NAMES_VAL=/workspace/splits/mass_eval_720.txt
LENS_TRAIN=/workspace/activations/gptoss_direction_mass_l15
LENS_VAL=/workspace/activations/gptoss_direction_mass_l15_eval
ROLLOUTS=/workspace/rollouts/gptoss_sentence
ACT=/workspace/activations/gptoss_sentence
PREP=/workspace/prepared/gptoss_sentence
MODEL=openai/gpt-oss-20b
LAYER=15
SEED=42
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
    --strategy "$STRATEGY" \
    --lens-root "$LENS_TRAIN" \
    --lens "$LENS" \
    --loudness-layer "$LAYER" \
    --seed "$SEED" \
    --batch-size "$BATCH_SIZE" \
    --max-batch-tokens "$MAX_BATCH_TOKENS" \
    --device-map "$DEVICE_MAP" \
    --skip-existing
uv run --extra gpu python telos_interp/loudness_analysis/rollouts/run_inference.py \
    --trajectory-paths "${TRAJ_VAL}/size*/*.json" \
    --output-dir "${ROLLOUTS}/${ARM}_val" \
    --model-id "$MODEL" \
    --strategy "$STRATEGY" \
    --lens-root "$LENS_VAL" \
    --lens "$LENS" \
    --loudness-layer "$LAYER" \
    --seed "$SEED" \
    --batch-size "$BATCH_SIZE" \
    --max-batch-tokens "$MAX_BATCH_TOKENS" \
    --device-map "$DEVICE_MAP" \
    --skip-existing
