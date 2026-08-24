#!/usr/bin/env bash
# Add a logit-lens arm to an activation tree that has already been pruned.
#
# The tree holds only the jlens ∪ random selection; every other reasoning token's
# activations are gone. So a logitlens arm cannot be recovered by re-filtering -- the tokens
# it would pick were deleted -- and it does not need a re-gather either. This does the
# minimum:
#
#   1. one forward pass per trajectory to write {stem}_logitlens_analysis.csv (the CSV comes
#      from the model, not from the activation tree, so a pruned tree is no obstacle),
#   2. the shared filter picks that arm's top tokens and layers,
#   3. only the .pt files not already on disk are written -- the jlens and logitlens arms
#      overlap heavily on genuinely direction-loaded tokens,
#   4. the arm is MERGED into {stem}_jlens_selection.json. The jlens and random arms keep
#      their recorded picks and config verbatim.
#
# The control arm is deliberately not redrawn. The one in the record is a uniform draw over
# the *full* reasoning chain, taken before anything was deleted; a fresh draw now could only
# sample the survivors, which is exactly the bias the recorded control exists to avoid. It
# is a valid matched control for the logitlens arm as it stands.
#
# DRY RUN BY DEFAULT. Run it first and read the projected file count -- the disk filled up
# once already, which is why any of this exists.
#
#   ./scripts/jlens_extend_logitlens.sh              # report what would be added
#   APPLY=1 ./scripts/jlens_extend_logitlens.sh      # do it
#   LIMIT=5 APPLY=1 ./scripts/jlens_extend_logitlens.sh   # ...on a handful first
#
# Do not run this while prepare_next_action_jlens_by_complexity.sh is reading the tree.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)   # so uv finds pyproject.toml

ACT=${ACT:-/workspace/activations/jlens_reasoning_tokens}
TRAJ=${TRAJ:-/workspace/trajectories/reveng/trajectories_train_single_step}
JLENS_DIR=${JLENS_DIR:-/workspace/jlens}
SIGNAL_JSON=${SIGNAL_JSON:-$REPO/data/jlens/direction_tokens_full.json}

LAYERS=${LAYERS:-7:23}
METHODS=${METHODS:-logitlens}     # arms to add; ones already recorded are left alone
NUM_TOKENS=${NUM_TOKENS:-20}
NUM_LAYERS=${NUM_LAYERS:-3}
ALWAYS_LAYERS=${ALWAYS_LAYERS:-15}
SELECT_SEED=${SELECT_SEED:-42}
DIRECTION_CLASSES=${DIRECTION_CLASSES:-all}

SIZES=${SIZES:-}
COMPLEXITIES=${COMPLEXITIES:-}
LIMIT=${LIMIT:-}
BATCH_SIZE=${BATCH_SIZE:-256}
IO_WORKERS=${IO_WORKERS:-16}
FORWARD_BATCH_SIZE=${FORWARD_BATCH_SIZE:-4}
DEVICE=${DEVICE:-cuda}
APPLY=${APPLY:-}                  # 1 to actually write; otherwise --dry-run

# --select-random-tokens 0: the control is inherited from the record, never redrawn.
# --lens logitlens: the jlens CSV already exists and is not recomputed.
CMD=(
    uv run --project "$REPO" python "$REPO/scripts/jlens_reasoning_tokens.py"
    --trajectory-paths "$TRAJ"
    --jlens_dir "$JLENS_DIR"
    --activations-dir "$ACT"
    --layers "$LAYERS"
    --steps all
    --lens logitlens
    --extend
    --signal-json "$SIGNAL_JSON"
    --select-methods "$METHODS"
    --select-num-tokens "$NUM_TOKENS"
    --select-num-layers "$NUM_LAYERS"
    --select-always-layers "$ALWAYS_LAYERS"
    --select-random-tokens 0
    --select-seed "$SELECT_SEED"
    --direction-classes "$DIRECTION_CLASSES"
    --batch-size "$BATCH_SIZE"
    --io-workers "$IO_WORKERS"
    --forward-batch-size "$FORWARD_BATCH_SIZE"
    --device "$DEVICE"
)
[ -n "$SIZES" ] && CMD+=(--sizes "$SIZES")
[ -n "$COMPLEXITIES" ] && CMD+=(--complexities "$COMPLEXITIES")
[ -n "$LIMIT" ] && CMD+=(--max-trajectories "$LIMIT")

if [ -n "$APPLY" ]; then
    echo "About to WRITE new activations under $ACT (arms: $METHODS)."
    echo "Existing arms are preserved; nothing is deleted. Press Enter (Ctrl-C to abort)."
    read -r _
else
    CMD+=(--dry-run)
    echo "DRY RUN -- nothing will be written. Set APPLY=1 to extend."
fi

"${CMD[@]}"
