#!/usr/bin/env bash
# Full j-space sweep over reasoning tokens on the GPU host.
# Writes, per trajectory, the residual-stream activations (gather_activations
# layout) and a {stem}_jlens_analysis.csv of top-20 lens predictions per
# (reasoning token, layer). See jlens_reasoning_tokens.py.
set -euo pipefail

TRAJECTORIES=${TRAJECTORIES:-/workspace/trajectories/reveng/trajectories_train_single_step/}
JLENS_DIR=${JLENS_DIR:-/workspace/jlens/gridenv}
ACTIVATIONS_DIR=${ACTIVATIONS_DIR:-/workspace/activations/jlens_reasoning_tokens}
LAYERS=${LAYERS:-"7:23"}   # inclusive range; also the layers saved to disk
# Hard cap, spread evenly over the size x complexity grid: 200 per cell over the
# full 6x6 grid = 7200 trajectories. Set empty to fall back to MAX_TRAJ / all.
PER_COMBO=${PER_COMBO:-200}
MAX_TRAJ=${MAX_TRAJ:-}     # global cap instead of PER_COMBO (the two are mutually exclusive)
SEED=${SEED:-0}            # random-sample seed; empty = lowest run indices
SIZES=${SIZES:-}           # e.g. "11,15"; empty for all grid sizes
COMPLEXITIES=${COMPLEXITIES:-}  # e.g. "0.0,1.0"; empty for all complexities
BATCH_SIZE=${BATCH_SIZE:-}  # reasoning tokens per lens matmul; empty = script default (256)
OVERWRITE=${OVERWRITE:-}    # non-empty to redo trajectories that already have a CSV
# Throughput knobs; empty = script defaults. Long reasoning chains (comp 0.6-1.0) are
# bound by the per-token .pt writes, so IO_WORKERS matters more than the batching.
IO_WORKERS=${IO_WORKERS:-}          # threads writing .pt files (default 16, 0 = inline)
FWD_BATCH_SIZE=${FWD_BATCH_SIZE:-}  # steps per padded forward pass (default 4, 1 = off)
FWD_BATCH_TOKENS=${FWD_BATCH_TOKENS:-}  # padded-token budget per forward (default 16384)
PT_FORMAT=${PT_FORMAT:-}    # zip (default) or legacy; benchmark before switching
PROFILE=${PROFILE:-}        # non-empty to print the per-phase wall-time split

uv run python scripts/jlens_reasoning_tokens.py \
    --trajectory-paths "$TRAJECTORIES" \
    --jlens_dir "$JLENS_DIR" \
    --layers "$LAYERS" \
    --steps "all" \
    ${BATCH_SIZE:+--batch-size "$BATCH_SIZE"} \
    ${IO_WORKERS:+--io-workers "$IO_WORKERS"} \
    ${FWD_BATCH_SIZE:+--forward-batch-size "$FWD_BATCH_SIZE"} \
    ${FWD_BATCH_TOKENS:+--forward-batch-tokens "$FWD_BATCH_TOKENS"} \
    ${PT_FORMAT:+--pt-format "$PT_FORMAT"} \
    ${PROFILE:+--profile} \
    ${SIZES:+--sizes "$SIZES"} \
    ${COMPLEXITIES:+--complexities "$COMPLEXITIES"} \
    ${PER_COMBO:+--per-combo "$PER_COMBO"} \
    ${MAX_TRAJ:+--max-trajectories "$MAX_TRAJ"} \
    ${SEED:+--seed "$SEED"} \
    ${OVERWRITE:+--overwrite} \
    --activations-dir "$ACTIVATIONS_DIR"

echo "Done -> $ACTIVATIONS_DIR"
