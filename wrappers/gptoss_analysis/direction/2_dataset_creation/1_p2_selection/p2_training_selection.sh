#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)
MODEL=openai/gpt-oss-20b
TRAJ=/workspace/activations/mass_train2880_view/trajectories
JLENS_DIR=/workspace/jlens/gridenv
SIGNAL_JSON=$REPO/data/jlens/direction_tokens_full.json
SIGNAL_NAME=direction
OUT=/workspace/activations/gptoss_direction_mass_l15
LAYER=15
LAYERS=7:23
SAMPLE_PERCENT=1.0
SAMPLE_SEED=42
METHODS=jlens,logitlens,random
SCORE=logprob_mass_full
NUM_TOKENS=20
RANDOM_TOKENS=20
NUM_LAYERS=1
ALWAYS_LAYERS=""
SELECT_SEED=42
BATCH_SIZE=256
FORWARD_BATCH_SIZE=4
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
uv run --extra gpu python telos_interp/loudness_analysis/build_loudness_tables.py \
    --trajectory-paths "$TRAJ" \
    --activations-dir "$OUT" \
    --model-id "$MODEL" \
    --jlens_dir "$JLENS_DIR" \
    --signal-json "$SIGNAL_JSON" \
    --signal-name "$SIGNAL_NAME" \
    --direction-score "$SCORE" \
    --select-methods "$METHODS" \
    --select-num-tokens "$NUM_TOKENS" \
    --select-random-tokens "$RANDOM_TOKENS" \
    --select-num-layers "$NUM_LAYERS" \
    --select-always-layers "$ALWAYS_LAYERS" \
    --select-candidate-layers "$LAYER" \
    --select-seed "$SELECT_SEED" \
    --data_sample_p "$SAMPLE_PERCENT" \
    --data-sample-seed "$SAMPLE_SEED" \
    --lens both \
    --layers "$LAYERS" \
    --batch-size "$BATCH_SIZE" \
    --forward-batch-size "$FORWARD_BATCH_SIZE" \
    --device cuda
