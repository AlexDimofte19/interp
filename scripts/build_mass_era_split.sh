#!/usr/bin/env bash
# Materialise the mass-era 3,600 as TWO SEPARATE DATASETS: 2,880 train and 720 eval.
#
# WHY. The partition already exists, but only as a flag. `mass_l15_names.txt` lists all 3,600
# and `next_action_mass_l15_eval_names.txt` lists the 720; "the 2,880" was never written down
# anywhere -- `audit_trajectory_sets.py` had to recover it by reading the names out of ONE
# arm's prepared manifest (`local_belief_p2_split_train`). Every honest number therefore
# depended on a caller remembering to pass `--eval-names`, and every tool that walks a TREE
# rather than a manifest (the rollouts, `eval_probe_per_token.py`, the loudness builds, the
# convinced datasets) saw all 3,600 at once and had to be told which half to ignore.
#
# This makes each half a first-class dataset, exactly as `heldout360` is one: its own name
# list, and its own activation + trajectory directory pair. Point a tool at the train view and
# it CANNOT see an eval trajectory, because there is no path from the view to one.
#
# WHAT IT DOES NOT DO. It does not re-draw the split. The 720 are the same 720 every probe on
# disk was scored against; re-drawing them would invalidate every published number. The train
# half is defined as the complement, and is verified here to equal the 2,880 the project has
# been calling "train 2880" all along. Nothing existing is moved, rewritten or deleted -- the
# source tree stays where it is and the views are symlinks into it, so every prepared dataset,
# every manifest `activations_root` and every recorded invocation keeps working unchanged.
#
#   ./scripts/build_mass_era_split.sh              # build (idempotent)
#   DRY_RUN=1 ./scripts/build_mass_era_split.sh    # say what it would do
#   VERIFY_ONLY=1 ./scripts/build_mass_era_split.sh  # just re-check what is there
#
# Costs 7,200 symlinks and no data. Safe to re-run.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WS=${WS:-/workspace}
ACT=${ACT:-$WS/activations}
SPLITS=${SPLITS:-$WS/splits}
PREPARED=${PREPARED:-$WS/prepared}
RT=${RT:-$WS/reasoning_theatre}
TRAJ=${TRAJ:-$WS/trajectories/reveng/trajectories_train_single_step}

# The source of truth for the split, as it stands today.
MASS_NAMES=${MASS_NAMES:-$RT/rollout_strategies/mass_l15_names.txt}
MASS_EVAL_NAMES=${MASS_EVAL_NAMES:-$PREPARED/next_action_mass_l15_eval_names.txt}

# The tree the two views are cut from.
SRC_TREE=${SRC_TREE:-$ACT/jlens_mass_l15}

# What this script creates.
TRAIN_NAMES=${TRAIN_NAMES:-$SPLITS/mass_train_2880.txt}
EVAL_NAMES=${EVAL_NAMES:-$SPLITS/mass_eval_720.txt}
TRAIN_VIEW=${TRAIN_VIEW:-$ACT/mass_train2880_view}
EVAL_VIEW=${EVAL_VIEW:-$ACT/mass_eval720_view}

DRY_RUN=${DRY_RUN:-}
VERIFY_ONLY=${VERIFY_ONLY:-}

for f in "$MASS_NAMES" "$MASS_EVAL_NAMES"; do
    [ -s "$f" ] || { echo "missing or empty: $f" >&2; exit 1; }
done
[ -d "$SRC_TREE" ] || { echo "missing source tree: $SRC_TREE" >&2; exit 1; }

echo "mass-era split"
echo "  source tree : $SRC_TREE"
echo "  all names   : $MASS_NAMES ($(grep -c . "$MASS_NAMES"))"
echo "  eval names  : $MASS_EVAL_NAMES ($(grep -c . "$MASS_EVAL_NAMES"))"
echo

# ---- 1. the two name lists -------------------------------------------------------------
# Train is the complement, never a fresh draw. `comm` needs sorted input; the lists are
# written sorted so a diff between two runs is empty rather than a reordering.
if [ -z "$VERIFY_ONLY" ]; then
    echo "  [1/3] name lists"
    if [ -n "$DRY_RUN" ]; then
        echo "      + would write $TRAIN_NAMES (2880) and $EVAL_NAMES (720)"
    else
        mkdir -p "$SPLITS"
        sort -u "$MASS_EVAL_NAMES" > "$EVAL_NAMES"
        comm -23 <(sort -u "$MASS_NAMES") "$EVAL_NAMES" > "$TRAIN_NAMES"
        echo "      wrote $TRAIN_NAMES ($(grep -c . "$TRAIN_NAMES"))"
        echo "      wrote $EVAL_NAMES ($(grep -c . "$EVAL_NAMES"))"
    fi
fi

# ---- 2. the two views ------------------------------------------------------------------
# On a dry run the lists from step 1 do not exist yet, so the view builder cannot be asked to
# count them -- report the two builds from the counts we already know instead.
if [ -z "$VERIFY_ONLY" ]; then
    echo "  [2/3] activation + trajectory views"
    if [ -n "$DRY_RUN" ]; then
        echo "      + would build $TRAIN_VIEW (2880 activation + 2880 trajectory symlinks)"
        echo "      + would build $EVAL_VIEW (720 activation + 720 trajectory symlinks)"
    else
        SRC="$SRC_TREE" TRAJ="$TRAJ" \
            bash "$REPO/scripts/build_activation_view.sh" "$TRAIN_NAMES" "$TRAIN_VIEW"
        SRC="$SRC_TREE" TRAJ="$TRAJ" \
            bash "$REPO/scripts/build_activation_view.sh" "$EVAL_NAMES" "$EVAL_VIEW"
    fi
fi

[ -n "$DRY_RUN" ] && { echo; echo "  (dry run -- nothing written)"; exit 0; }

# ---- 3. verify -------------------------------------------------------------------------
echo "  [3/3] verify"
TRAIN_NAMES="$TRAIN_NAMES" EVAL_NAMES="$EVAL_NAMES" MASS_NAMES="$MASS_NAMES" \
TRAIN_VIEW="$TRAIN_VIEW" EVAL_VIEW="$EVAL_VIEW" SRC_TREE="$SRC_TREE" \
PREPARED="$PREPARED" WS="$WS" \
    python3 "$REPO/scripts/verify_mass_era_split.py"
