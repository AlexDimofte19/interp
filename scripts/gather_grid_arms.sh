#!/usr/bin/env bash
# ROUND 2 gather: select reasoning tokens by GRID words instead of direction words.
#
# The first of the three grid-round scripts:
#   scripts/gather_grid_arms.sh    <- you are here: build the tree
#   scripts/prepare_grid_arms.sh      one grid_tile dataset per arm
#   scripts/train_grid_arms.sh        the seed sweep
#
# Identical machinery to scripts/jlens_reasoning_tokens_filtered.sh -- same script, same
# filter, same record format. Only two things differ, and they are the whole experiment:
# SIGNAL_JSON points at grid_tokens_full.json, and the output goes to its own tree.
#
# Why a separate tree rather than --extend on the direction one:
#   * that tree is PRUNED to the direction-scored selection, and a grid-scored arm ranks
#     the same CSVs differently, so it wants tokens whose .pt files were deleted;
#   * an arm's name IS its method name (`jlens_utils.METHODS`), and the signal JSON is a
#     caller argument, not part of a method's identity -- so a grid-scored arm written into
#     the old records would be called "jlens" and collide with the direction-scored one.
#     --extend leaves recorded arms alone, so it would silently do nothing.
# Here every arm is grid-scored by construction and the names carry no ambiguity.
#
# LENS=both emits the jlens AND logitlens CSVs from ONE forward pass -- the second lens
# costs an extra unembed per chunk, not a second pass -- so both lens arms and their shared
# control come out of a single sweep.
#
# The control arm is not optional bookkeeping: `top_filter` refuses an unscored arm with no
# scored arm to enumerate the reasoning chain, and a lens number without a matched control
# means nothing (see telos_interp/jlens_utils/README.md).
#
# PER_COMBO/SEED reproduce round 1's trajectory sample, so the two rounds cover the SAME
# trajectories and can be read against each other.
#
# Resumable: a trajectory that already has its CSV is skipped, so re-running after an
# interruption picks up where it stopped. Pass OVERWRITE=1 to redo them.
#
#   ./scripts/gather_grid_arms.sh
#   DRY_RUN=1 ./scripts/gather_grid_arms.sh            # print the invocation, run nothing
#   MAX_TRAJ=2 ACT=/tmp/smoke ./scripts/gather_grid_arms.sh   # smoke test
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)   # so uv finds pyproject.toml

TRAJECTORIES=${TRAJECTORIES:-/workspace/trajectories/reveng/trajectories_train_single_step/}
JLENS_DIR=${JLENS_DIR:-/workspace/jlens/gridenv}
ACT=${ACT:-/workspace/activations/grid_reasoning_tokens}
SIGNAL_JSON=${SIGNAL_JSON:-/workspace/jlens/grid_tokens_full.json}
LOG=${LOG:-/workspace/logs/grid_gather.log}
STATUS=${STATUS:-/workspace/logs/grid_gather_status.txt}

LENS=${LENS:-both}
METHODS=${METHODS:-jlens,logitlens,random}
LAYERS=${LAYERS:-"7:23"}   # candidate pool; layers without a jlens matrix are unscorable
STEPS=${STEPS:-all}

# Selection. NUM_TOKENS matches round 1 so the two rounds select the same budget. NUM_LAYERS
# is one value for ALL arms on purpose: round 1 had jlens at 5 and logitlens at 3, which
# entry 28 of the log identified as the layer-distribution confound that forced its
# L15-only comparison. One value removes that confound here by construction.
NUM_TOKENS=${NUM_TOKENS:-20}
NUM_LAYERS=${NUM_LAYERS:-3}
ALWAYS_LAYERS=${ALWAYS_LAYERS:-15}
RANDOM_TOKENS=${RANDOM_TOKENS:-20}
SELECT_SEED=${SELECT_SEED:-42}
SIGNAL_CLASSES=${SIGNAL_CLASSES:-all}      # subset of the signal JSON's top-level keys

# Trajectory sampling -- these two reproduce round 1's set.
PER_COMBO=${PER_COMBO:-100}
SEED=${SEED:-0}
MAX_TRAJ=${MAX_TRAJ:-}     # global cap instead of PER_COMBO (mutually exclusive)
SIZES=${SIZES:-}
COMPLEXITIES=${COMPLEXITIES:-}
OVERWRITE=${OVERWRITE:-}

# Throughput knobs; empty = script defaults.
BATCH_SIZE=${BATCH_SIZE:-}
IO_WORKERS=${IO_WORKERS:-}
FWD_BATCH_SIZE=${FWD_BATCH_SIZE:-}
FWD_BATCH_TOKENS=${FWD_BATCH_TOKENS:-}
DRY_RUN=${DRY_RUN:-}

[ -f "$SIGNAL_JSON" ] || { echo "!! signal JSON not found: $SIGNAL_JSON" >&2; exit 1; }
[ -d "$JLENS_DIR" ]   || { echo "!! jlens dir not found: $JLENS_DIR" >&2; exit 1; }
[ -e "$TRAJECTORIES" ] || { echo "!! trajectories not found: $TRAJECTORIES" >&2; exit 1; }

# device_map="auto" across multiple GPUs produces NaNs for this MoE model. One visible
# device is the supported configuration; say so rather than producing silent garbage.
gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
if [ "${gpus:-0}" -gt 1 ] && [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "!! $gpus GPUs visible and CUDA_VISIBLE_DEVICES unset." >&2
    echo "   device_map=auto sharded across GPUs gives NaNs for gpt-oss-20b." >&2
    echo "   Re-run with CUDA_VISIBLE_DEVICES=0 (see CLAUDE.md)." >&2
    exit 1
fi

mkdir -p "$(dirname "$LOG")" "$(dirname "$STATUS")"

cmd=(uv run --project "$REPO" python "$REPO/scripts/jlens_reasoning_tokens.py"
    --trajectory-paths "$TRAJECTORIES"
    --jlens_dir "$JLENS_DIR"
    --activations-dir "$ACT"
    --layers "$LAYERS"
    --steps "$STEPS"
    --lens "$LENS"
    --signal-json "$SIGNAL_JSON"
    --select-methods "$METHODS"
    --select-num-tokens "$NUM_TOKENS"
    --select-num-layers "$NUM_LAYERS"
    --select-always-layers "$ALWAYS_LAYERS"
    --select-random-tokens "$RANDOM_TOKENS"
    --select-seed "$SELECT_SEED"
    --direction-classes "$SIGNAL_CLASSES"
)
[ -z "$BATCH_SIZE" ]       || cmd+=(--batch-size "$BATCH_SIZE")
[ -z "$IO_WORKERS" ]       || cmd+=(--io-workers "$IO_WORKERS")
[ -z "$FWD_BATCH_SIZE" ]   || cmd+=(--forward-batch-size "$FWD_BATCH_SIZE")
[ -z "$FWD_BATCH_TOKENS" ] || cmd+=(--forward-batch-tokens "$FWD_BATCH_TOKENS")
[ -z "$SIZES" ]            || cmd+=(--sizes "$SIZES")
[ -z "$COMPLEXITIES" ]     || cmd+=(--complexities "$COMPLEXITIES")
[ -z "$MAX_TRAJ" ]         || cmd+=(--max-trajectories "$MAX_TRAJ")
[ -n "$MAX_TRAJ" ] || [ -z "$PER_COMBO" ] || cmd+=(--per-combo "$PER_COMBO")
[ -z "$SEED" ]             || cmd+=(--seed "$SEED")
[ -z "$OVERWRITE" ]        || cmd+=(--overwrite)

echo "signal:      $SIGNAL_JSON"
echo "tree:        $ACT"
echo "lens:        $LENS      arms: $METHODS"
echo "selection:   top $NUM_TOKENS token(s) x $NUM_LAYERS layer(s) + always [$ALWAYS_LAYERS], $RANDOM_TOKENS control"
echo "log:         $LOG"
echo ""

if [ -n "$DRY_RUN" ]; then
    printf '%q ' "${cmd[@]}"; echo
    echo "(DRY_RUN set; nothing run)"
    exit 0
fi

echo "GATHER_START $(date -Is)" >> "$STATUS"
"${cmd[@]}" >> "$LOG" 2>&1
rc=$?
echo "GATHER_END rc=$rc $(date -Is)" >> "$STATUS"

if [ $rc -eq 0 ]; then
    echo "Done -> $ACT"
    echo "Next: ARMS=\"jlens logitlens random\" LAYERS=15 OUT=/workspace/prepared/grid_l15 \\"
    echo "        ACT=$ACT ./scripts/prepare_grid_arms.sh"
else
    echo "!! gather failed (rc=$rc); see $LOG" >&2
fi
exit $rc
