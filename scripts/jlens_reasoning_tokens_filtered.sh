#!/usr/bin/env bash
# jlens sweep that keeps only the activations a probe will actually use.
#
# Same as jlens_reasoning_tokens.sh -- full CSV, same resume-by-CSV behaviour -- but the
# .pt tree is filtered down to the top NUM_TOKENS tokens x NUM_LAYERS layers (plus
# ALWAYS_LAYERS), which is ~75x less disk. See scripts/jlens_reasoning_tokens.py.
#
# RANDOM_TOKENS is not optional bookkeeping: it reserves a uniform-draw control arm. Once
# a trajectory holds only its top-scoring tokens, that draw can never be made again, and a
# jlens_direction number without a matched control means nothing.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)   # so uv finds pyproject.toml

TRAJECTORIES=${TRAJECTORIES:-/workspace/trajectories/reveng/trajectories_train_single_step/}
JLENS_DIR=${JLENS_DIR:-/workspace/jlens/gridenv}
ACTIVATIONS_DIR=${ACTIVATIONS_DIR:-/workspace/activations/jlens_reasoning_tokens}
SIGNAL_JSON=${SIGNAL_JSON:-$REPO/data/jlens/direction_tokens_full.json}
LAYERS=${LAYERS:-"7:23"}   # candidate pool; layers without a jlens matrix are unscorable
PER_COMBO=${PER_COMBO:-100}
MAX_TRAJ=${MAX_TRAJ:-}     # global cap instead of PER_COMBO (mutually exclusive)
SEED=${SEED:-0}            # trajectory-sampling seed; empty = lowest run indices
SIZES=${SIZES:-}           # e.g. "11,15"; empty for all grid sizes
COMPLEXITIES=${COMPLEXITIES:-}
OVERWRITE=${OVERWRITE:-}   # non-empty to redo trajectories that already have a CSV

# Selection. Keep these in step with delete_non_jlens_selected.sh -- the two must agree or
# a mixed tree ends up with two different definitions of "selected".
NUM_TOKENS=${NUM_TOKENS:-20}
NUM_LAYERS=${NUM_LAYERS:-3}
ALWAYS_LAYERS=${ALWAYS_LAYERS:-15}
RANDOM_TOKENS=${RANDOM_TOKENS:-20}
SELECT_SEED=${SELECT_SEED:-42}
DIRECTION_CLASSES=${DIRECTION_CLASSES:-all}

# Throughput knobs; empty = script defaults. Filtering removes ~99% of the .pt writes, so
# IO_WORKERS matters far less here than it does for an unfiltered run.
BATCH_SIZE=${BATCH_SIZE:-}
IO_WORKERS=${IO_WORKERS:-}
FWD_BATCH_SIZE=${FWD_BATCH_SIZE:-}
FWD_BATCH_TOKENS=${FWD_BATCH_TOKENS:-}
PT_FORMAT=${PT_FORMAT:-}
PROFILE=${PROFILE:-}

[ -f "$SIGNAL_JSON" ] || { echo "!! signal JSON not found: $SIGNAL_JSON" >&2; exit 1; }

uv run --project "$REPO" python "$REPO/scripts/jlens_reasoning_tokens.py" \
    --trajectory-paths "$TRAJECTORIES" \
    --jlens_dir "$JLENS_DIR" \
    --activations-dir "$ACTIVATIONS_DIR" \
    --layers "$LAYERS" \
    --steps "all" \
    --signal-json "$SIGNAL_JSON" \
    --select-num-tokens "$NUM_TOKENS" \
    --select-num-layers "$NUM_LAYERS" \
    --select-always-layers "$ALWAYS_LAYERS" \
    --select-random-tokens "$RANDOM_TOKENS" \
    --select-seed "$SELECT_SEED" \
    --direction-classes "$DIRECTION_CLASSES" \
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
    ${OVERWRITE:+--overwrite}

echo "Done -> $ACTIVATIONS_DIR"
echo "Prepare probe data with --token-selection recorded_jlens / recorded_random."
