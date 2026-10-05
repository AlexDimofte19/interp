#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)
MODEL=Qwen/Qwen3.6-35B-A3B
TRAJ=/workspace/trajectories/qwen3.6-35b/replayed_single_step/mass_eval_144
JLENS_DIR=/workspace/jlens/qwen3_6_35b
SIGNAL_JSON=$REPO/data/jlens/qwen/direction_tokens_full_qwen3-6-35b-a3b.json
SIGNAL_NAME=direction
OUT=/workspace/activations/qwen_p2_selection_eval
LAYER=27
LAYERS=27
SAMPLE_PERCENT=0.2
SAMPLE_SEED=42
METHODS=jlens,logitlens,random
SCORE=logprob_mass_full
NUM_TOKENS=60
RANDOM_TOKENS=60
NUM_LAYERS=1
ALWAYS_LAYERS=""
SELECT_SEED=42
BATCH_SIZE=256
FORWARD_BATCH_SIZE=1
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
