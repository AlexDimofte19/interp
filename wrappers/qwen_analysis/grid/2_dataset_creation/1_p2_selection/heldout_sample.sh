#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)
MODEL=Qwen/Qwen3.6-35B-A3B
TRAJ=/workspace/trajectories/qwen3.6-35b/replayed_single_step/heldout_72
JLENS_DIR=/workspace/jlens/qwen3_6_35b
SIGNAL_JSON=$REPO/data/jlens/qwen/grid_tokens_pruned_qwen3-6-35b-a3b.json
SIGNAL_NAME=grid
OUT=/workspace/activations/qwen_p2_heldout_grid
LAYER=27
SAMPLE_PERCENT=1.0
SAMPLE_SEED=42
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
    --direction-mass-json "$SIGNAL_JSON" \
    --signal-name "$SIGNAL_NAME" \
    --data_sample_p "$SAMPLE_PERCENT" \
    --data-sample-seed "$SAMPLE_SEED" \
    --lens both \
    --layers "$LAYER" \
    --batch-size "$BATCH_SIZE" \
    --forward-batch-size "$FORWARD_BATCH_SIZE" \
    --device cuda
