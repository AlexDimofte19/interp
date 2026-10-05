#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TRAJ=/workspace/trajectories/qwen3.6-35b/replayed_single_step/mass_train_576
OUT_DIR=/workspace/jlens/qwen3_6_35b
MODEL=Qwen/Qwen3.6-35B-A3B
N_PROMPTS=144
SEED=0
WINDOW_FRAC=0.5
WINDOW_SIZE=1024
SOURCE_LAYERS="3 7 11 15 19 23 27 31 35"
DIM_BATCH=1
EVAL_EVERY=8
STOP_AT_DELTA=0.002
MIN_PROMPTS=48
export DISABLE_KERNEL_MAPPING=1
cd "$REPO"
uv run --extra gpu python jlens/jlens_fit_qwen.py \
    --trajectories-dir "$TRAJ" \
    --out-dir "$OUT_DIR" \
    --model-id "$MODEL" \
    --n-prompts "$N_PROMPTS" \
    --seed "$SEED" \
    --window-frac "$WINDOW_FRAC" \
    --window-size "$WINDOW_SIZE" \
    --source-layers $SOURCE_LAYERS \
    --dim-batch "$DIM_BATCH" \
    --dtype bfloat16 \
    --device-map cuda \
    --eval-every "$EVAL_EVERY" \
    --stop-at-delta "$STOP_AT_DELTA" \
    --min-prompts "$MIN_PROMPTS"
