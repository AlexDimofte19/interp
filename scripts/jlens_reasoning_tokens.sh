#!/usr/bin/env bash
# Full j-space sweep over reasoning tokens on the GPU host.
# Writes, per trajectory, the residual-stream activations (gather_activations
# layout) and a {stem}_jlens_analysis.csv of top-20 lens predictions per
# (reasoning token, layer). See jlens_reasoning_tokens.py.
set -euo pipefail

TRAJECTORIES=${TRAJECTORIES:-/workspace/trajectories/trajectories_test_full}
JLENS_DIR=${JLENS_DIR:-/workspace/jlens}
ACTIVATIONS_DIR=${ACTIVATIONS_DIR:-/workspace/activations/jlens_reasoning_tokens}
LAYERS=${LAYERS:-"7:23"}   # inclusive range; also the layers saved to disk
MAX_TRAJ=${MAX_TRAJ:-5}   # sample this many trajectories; set empty for all
SEED=${SEED:-0}            # random-sample seed; set empty for first-N
SIZES=${SIZES:-"5,9,11,13"}           # e.g. "11,15"; empty for all grid sizes
COMPLEXITIES=${COMPLEXITIES:-"0.0,0.5"}  # e.g. "0.0,1.0"; empty for all complexities
BATCH_SIZE=${BATCH_SIZE:-}  # reasoning tokens per lens matmul; empty = whole step
OVERWRITE=${OVERWRITE:-}    # non-empty to redo trajectories that already have a CSV

python scripts/jlens_reasoning_tokens.py \
    --trajectory-paths "$TRAJECTORIES" \
    --jlens_dir "$JLENS_DIR" \
    --layers "$LAYERS" \
    --steps "all" \
    ${BATCH_SIZE:+--batch-size "$BATCH_SIZE"} \
    ${SIZES:+--sizes "$SIZES"} \
    ${COMPLEXITIES:+--complexities "$COMPLEXITIES"} \
    ${MAX_TRAJ:+--max-trajectories "$MAX_TRAJ"} \
    ${SEED:+--seed "$SEED"} \
    ${OVERWRITE:+--overwrite} \
    --activations-dir "$ACTIVATIONS_DIR"

echo "Done -> $ACTIVATIONS_DIR"
