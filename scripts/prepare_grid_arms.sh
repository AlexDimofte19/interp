#!/usr/bin/env bash
# Prepare one grid_tile ("cognitive map") probe dataset per arm: jlens, logitlens, random.
#
# The sibling of scripts/prepare_next_action_arms.sh, and deliberately its twin: same tree,
# same selection records, same tokens, same layers. Only the label changes -- the identity of
# every grid cell instead of the agent's next action. That is what makes the two comparable:
# any difference between a grid arm and an action arm is the label, not the sample.
#
# Arms come from the {stem}_jlens_selection.json the gather or the pruner wrote, never from
# re-scoring a CSV: on a pruned tree a uniform draw over the survivors is not a uniform draw
# over the reasoning chain, so the control has to be read back. Hence recorded_*.
#
# Cheap in the same way: nothing is copied. Each manifest references the gathered token .pt
# files through `activations_root`, and the per-cell payload is stored once per
# (trajectory, step) under `cells` rather than on each of that trajectory's ~20 entries.
#
#   ./scripts/prepare_grid_arms.sh
#   ARMS="jlens logitlens" LAYERS=7:23 ./scripts/prepare_grid_arms.sh
#   LAYERS=15 OUT=/workspace/prepared/grid_l15 ./scripts/prepare_grid_arms.sh
#   MAX_CELLS=0 ./scripts/prepare_grid_arms.sh          # every cell, no per-trajectory cap
#
# MAX_CELLS is the knob that decides whether this is affordable. Padded to the widest grid
# every trajectory has 225 cells, and ~72k token entries x 225 is 16.2M rows per epoch. The
# default caps each (trajectory, step) at 25 class-balanced cells, ~1.8M rows. The draw is
# seeded per trajectory, so every arm is scored on the SAME cells -- arms must differ in
# their tokens and in nothing else.
#
# An arm the record does not hold prepares nothing and is reported as MISSING rather than
# aborting the others -- add it with scripts/jlens_extend_logitlens.sh.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)   # so uv finds pyproject.toml

ACT=${ACT:-/workspace/activations/jlens_reasoning_tokens}
TRAJ=${TRAJ:-/workspace/trajectories/reveng/trajectories_train_single_step}
OUT=${OUT:-/workspace/prepared/grid}                 # datasets land at ${OUT}_${arm}

ARMS=${ARMS:-"jlens logitlens random"}   # "eos" is also accepted, see EOS_ACT below
LAYERS=${LAYERS:-15}         # narrows the RECORDED layers; 15 gives a layer-15-only dataset
STEPS=${STEPS:-all}
SEED=${SEED:-42}
MAX_CELLS=${MAX_CELLS:-25}   # cells kept per (trajectory, step); 0 means all of them
PAD=${PAD:-}                 # pad_to_size; empty means auto (the widest grid present)

# The reasoning-EOS control is not a lens arm: that tree has no selection record, so its
# arm is "every gathered token" (--token-major) rather than a recorded pick. Point these at
# the view restricted to the SAME trajectories the lens arms cover -- the full tree is 10x
# larger, and an arm covering different trajectories cannot share their eval set.
EOS_ACT=${EOS_ACT:-/workspace/activations/activations_train_single_step_reasoning_eos_view}
EOS_TRAJ=${EOS_TRAJ:-$TRAJ}

[ -d "$ACT" ]  || { echo "!! activations dir not found: $ACT"  >&2; exit 1; }
[ -d "$TRAJ" ] || { echo "!! trajectories dir not found: $TRAJ" >&2; exit 1; }

cell_args=()
if [ "$MAX_CELLS" != "0" ]; then
    # Balanced per trajectory as well as capped: an uncapped 15x15 grid is mostly walls, and
    # a 25-cell uniform draw from it would be too.
    cell_args=(--max-positions-per-trajectory "$MAX_CELLS" --balance-classes-per-trajectory)
fi
[ -z "$PAD" ] || cell_args+=(--pad-to-size "$PAD")

echo ""
echo "activations: $ACT"
echo "layers:      $LAYERS"
echo "arms:        $ARMS"
echo "cells:       ${MAX_CELLS:-all} per (trajectory, step)"
echo ""

failed=""
for arm in $ARMS; do
    echo "============================================================"
    echo "Preparing grid arm: $arm  ->  ${OUT}_${arm}"
    echo "============================================================"

    arm_act=$ACT
    arm_traj=$TRAJ
    arm_args=(--token-selection "recorded_${arm}")
    if [ "$arm" = "eos" ]; then
        arm_act=$EOS_ACT
        arm_traj=$EOS_TRAJ
        arm_args=(--token-major)
        [ -d "$arm_act" ] || {
            echo "!! eos: activations dir not found: $arm_act (set EOS_ACT)" >&2
            failed="$failed $arm"; echo ""; continue
        }
    fi

    if uv run --project "$REPO" interp-cli prepare_activations_for_probing \
        --activations-dir "$arm_act" \
        --trajectories-dir "$arm_traj" \
        --probe-type grid_tile \
        --layers "$LAYERS" \
        --steps "$STEPS" \
        --output-indices all \
        "${arm_args[@]}" \
        --seed "$SEED" \
        "${cell_args[@]}" \
        --output-path "${OUT}_${arm}" \
        --verbose
    then
        echo "   ok"
    else
        echo "!! ${arm}: prepared nothing -- is that arm in the selection records?" >&2
        failed="$failed $arm"
    fi
    echo ""
done

echo "============================================================"
echo "SUMMARY"
echo "============================================================"
for arm in $ARMS; do
    d="${OUT}_${arm}"
    if [ -f "$d/manifest.json" ]; then
        n=$(uv run --project "$REPO" python -c \
            "import json,sys; m=json.load(open(sys.argv[1])); print(len(m['trajectories']), len({e['name'] for e in m['trajectories']}))" \
            "$d/manifest.json" 2>/dev/null || echo "? ?")
        printf '  %-12s %8s entries / %5s trajectories   %s\n' "$arm" ${n} "$d"
    else
        printf '  %-12s %8s                                %s\n' "$arm" "MISSING" "$d"
    fi
done

if [ -n "$failed" ]; then
    echo ""
    echo "Arms that prepared nothing:$failed"
    echo "  A lens arm missing from the records is added by scripts/jlens_extend_logitlens.sh;"
    echo "  pruning removed the tokens it needs, so prepare cannot re-derive it."
fi

echo ""
echo "Train them with:"
echo "  ARMS=\"$ARMS\" PREPARED=$OUT ./scripts/train_grid_arms.sh"
