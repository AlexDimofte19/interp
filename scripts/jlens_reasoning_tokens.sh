#!/usr/bin/env bash
# Full j-space sweep over reasoning tokens on the GPU host.
# Top-20 lens predictions per (reasoning token, layer), with each token's
# position in the reasoning chain. See jlens_reasoning_tokens.py.
set -euo pipefail

TRAJECTORIES=${TRAJECTORIES:-/workspace/trajectories/trajectories_test_full}
JLENS_DIR=${JLENS_DIR:-/workspace/jlens}
OUT=${OUT:-/workspace/jlens/jlens_reasoning_tokens.csv}
MAX_TRAJ=${MAX_TRAJ:-20}   # sample this many trajectories; set empty for all
SEED=${SEED:-0}            # random-sample seed; set empty for first-N
SIZES=${SIZES:-"5,9,11,13"}           # e.g. "11,15"; empty for all grid sizes
COMPLEXITIES=${COMPLEXITIES:-"0.0,0.5"}  # e.g. "0.0,1.0"; empty for all complexities
BATCH_SIZE=${BATCH_SIZE:-}  # reasoning tokens per lens matmul; empty = whole step

python interp/scripts/jlens_reasoning_tokens.py \
    --trajectory-paths "$TRAJECTORIES" \
    --jlens_dir "$JLENS_DIR" \
    --layers "all" \
    --steps "all" \
    ${BATCH_SIZE:+--batch-size "$BATCH_SIZE"} \
    ${SIZES:+--sizes "$SIZES"} \
    ${COMPLEXITIES:+--complexities "$COMPLEXITIES"} \
    ${MAX_TRAJ:+--max-trajectories "$MAX_TRAJ"} \
    ${SEED:+--seed "$SEED"} \
    --out "$OUT"

echo "Done -> $OUT"
