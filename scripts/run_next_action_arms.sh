#!/usr/bin/env bash
# Prepare and then train the next_action ("direction") probe for every arm, end to end.
#
# Just runs the two stages in order:
#   1. scripts/prepare_next_action_arms.sh   activation tree -> ${DATASETS}_${arm}
#   2. scripts/train_next_action_arms.sh     ${DATASETS}_${arm} -> probes + summary table
#
# The one thing worth having a master script for is that prepare's OUT and train's PREPARED
# have to be the same prefix. DATASETS sets both, so they cannot drift.
#
#   ./scripts/run_next_action_arms.sh
#   ARMS="jlens logitlens" ./scripts/run_next_action_arms.sh
#   COMPLEXITIES="0.0 0.2 0.4" DATASETS=/workspace/prepared/na_comp0-2-4 ./scripts/run_next_action_arms.sh
#   SKIP_PREPARE=1 ./scripts/run_next_action_arms.sh    # datasets already built; just train
#
# Every other knob belongs to one of the two stages and is passed through the environment
# untouched -- see those scripts for the full list (LAYERS, TOKENS_PER_TRAJ, MODEL_TYPES,
# DEVICE, SEED, ...).
#
# This reads the activation tree. Do not run it while delete_non_jlens_selected.py or an
# --extend gather is writing to it.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# The shared prefix. Prepare writes ${DATASETS}_${arm}; train reads the same.
DATASETS=${DATASETS:-/workspace/prepared/next_action}
ARMS=${ARMS:-"jlens logitlens random"}
PROBES=${PROBES:-/workspace/probes/next_action}

SKIP_PREPARE=${SKIP_PREPARE:-}
SKIP_TRAIN=${SKIP_TRAIN:-}

export ARMS PROBES
export OUT="$DATASETS"        # prepare_next_action_arms.sh writes here
export PREPARED="$DATASETS"   # train_next_action_arms.sh reads here

started=$(date +%s)

if [ -z "$SKIP_PREPARE" ]; then
    echo "############################################################"
    echo "# STAGE 1/2  prepare  ->  ${DATASETS}_<arm>"
    echo "############################################################"
    "$REPO/scripts/prepare_next_action_arms.sh"
else
    echo "# STAGE 1/2  prepare  -- SKIPPED (SKIP_PREPARE set)"
fi

# Stop before training if prepare produced nothing at all: every arm missing means the
# activation tree or the selection records are not what this run assumed, and 18 "no manifest"
# messages are a worse way to find that out.
built=0
for arm in $ARMS; do
    [ -f "${DATASETS}_${arm}/manifest.json" ] && built=$((built + 1))
done
if [ "$built" -eq 0 ]; then
    echo "!! no arm produced a manifest under ${DATASETS}_* -- nothing to train." >&2
    echo "   Check ACT/TRAJ, and that the trajectories have selection records." >&2
    exit 1
fi

if [ -z "$SKIP_TRAIN" ]; then
    echo ""
    echo "############################################################"
    echo "# STAGE 2/2  train  ($built/$(echo $ARMS | wc -w | tr -d ' ') arm(s) prepared)"
    echo "############################################################"
    "$REPO/scripts/train_next_action_arms.sh"
else
    echo "# STAGE 2/2  train  -- SKIPPED (SKIP_TRAIN set)"
fi

echo ""
echo "############################################################"
echo "# done in $(( ($(date +%s) - started) / 60 ))m"
echo "#   datasets: ${DATASETS}_<arm>"
echo "#   probes:   $PROBES"
echo "############################################################"
