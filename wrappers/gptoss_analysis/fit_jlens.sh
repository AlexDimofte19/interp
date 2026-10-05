#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TRAJ=/workspace/activations/mass_train2880_view/trajectories
OUT_DIR=/workspace/jlens/gridenv
N_PROMPTS=2880
SEED=0
MAX_SEQ_LEN=1024
EVAL_EVERY=10
STOP_AT_DELTA=0.002
cd "$REPO"
uv run --extra gpu python jlens/jlens_fit_gpt_oss.py \
    --trajectories-dir "$TRAJ" \
    --out-dir "$OUT_DIR" \
    --n-prompts "$N_PROMPTS" \
    --seed "$SEED" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --eval-every "$EVAL_EVERY" \
    --stop-at-delta "$STOP_AT_DELTA"
