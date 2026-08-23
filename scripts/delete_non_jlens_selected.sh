#!/usr/bin/env bash
# Prune trajectories that were gathered before the jlens filter existed.
#
# DRY RUN BY DEFAULT. It prints what it would delete and exits; set APPLY=1 to actually
# unlink. Read the dry-run output before doing that -- the deletion is not recoverable
# without re-running the whole GPU sweep.
#
# Before running:
#   * nothing else may be reading the tree. In particular
#     prepare_next_action_jlens_by_complexity.sh walks it and rm -rf's its own symlink
#     farm; running the two together will produce a dataset full of dangling paths.
#   * any already-prepared next_action manifest pointing into this tree dies here, since
#     its activations_root references files that are about to go. Re-run prepare after.
#
# Keep the selection knobs identical to jlens_reasoning_tokens_filtered.sh, or the tree
# ends up with two different definitions of "selected".
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)   # so uv finds pyproject.toml

ACTIVATIONS_DIR=${ACTIVATIONS_DIR:-/workspace/activations/jlens_reasoning_tokens}
TRAJECTORIES=${TRAJECTORIES:-/workspace/trajectories/reveng/trajectories_train_single_step}
SIGNAL_JSON=${SIGNAL_JSON:-$REPO/data/jlens/direction_tokens_full.json}

NUM_TOKENS=${NUM_TOKENS:-20}
NUM_LAYERS=${NUM_LAYERS:-3}
ALWAYS_LAYERS=${ALWAYS_LAYERS:-15}
RANDOM_TOKENS=${RANDOM_TOKENS:-20}
SELECT_SEED=${SELECT_SEED:-42}
DIRECTION_CLASSES=${DIRECTION_CLASSES:-all}

SIZES=${SIZES:-}                    # e.g. "11,15"; empty for all
COMPLEXITIES=${COMPLEXITIES:-}      # e.g. "0.0,0.2,0.4"; empty for all
APPLY=${APPLY:-}                    # 1 to actually delete
TOLERATE_MISSING=${TOLERATE_MISSING:-}
VERBOSE=${VERBOSE:-1}               # one line per trajectory

[ -f "$SIGNAL_JSON" ] || { echo "!! signal JSON not found: $SIGNAL_JSON" >&2; exit 1; }
[ -d "$ACTIVATIONS_DIR" ] || { echo "!! activations dir not found: $ACTIVATIONS_DIR" >&2; exit 1; }

if [ -n "$APPLY" ]; then
    echo "About to DELETE activations under $ACTIVATIONS_DIR."
    echo "Confirm nothing else is reading this tree, then press Enter (Ctrl-C to abort)."
    read -r _
fi

uv run --project "$REPO" python "$REPO/scripts/delete_non_jlens_selected.py" \
    --activations-dir "$ACTIVATIONS_DIR" \
    --trajectories-dir "$TRAJECTORIES" \
    --signal-json "$SIGNAL_JSON" \
    --select-num-tokens "$NUM_TOKENS" \
    --select-num-layers "$NUM_LAYERS" \
    --select-always-layers "$ALWAYS_LAYERS" \
    --select-random-tokens "$RANDOM_TOKENS" \
    --select-seed "$SELECT_SEED" \
    --direction-classes "$DIRECTION_CLASSES" \
    ${SIZES:+--sizes "$SIZES"} \
    ${COMPLEXITIES:+--complexities "$COMPLEXITIES"} \
    ${TOLERATE_MISSING:+--tolerate-missing} \
    ${VERBOSE:+--verbose} \
    ${APPLY:+--apply}
