#!/usr/bin/env bash
# A symlink view of the sentence-end activation tree, restricted to one trajectory set.
#
# `activations_train_single_step_reasoning_eos` holds layer-15 activations at every sentence
# end of all 36,000 trajectories. Every consumer wants a SUBSET of that -- the count-era
# 3,600, the mass-era 3,600 -- and `prepare_activations_for_probing` takes a directory, not a
# name list, so the subset has to exist as a directory. Symlinks rather than copies: the tree
# is ~75k tensors per view and they are the same bytes.
#
# The view pairs activations with trajectories so one --activations-dir / --trajectories-dir
# pair addresses exactly the wanted names.
#
#   bash scripts/build_eos_view.sh <names-file> <view-dir>
#
# Idempotent: a view already holding 2x its name count is left alone. DRY_RUN=1 reports only.
set -euo pipefail

NAMES_FILE=${1:?usage: build_eos_view.sh <names-file> <view-dir>}
VIEW=${2:?usage: build_eos_view.sh <names-file> <view-dir>}
SRC=${SRC:-/workspace/activations/activations_train_single_step_reasoning_eos}
TRAJ=${TRAJ:-/workspace/trajectories/reveng/trajectories_train_single_step}
DRY_RUN=${DRY_RUN:-}

want=$(grep -c . "$NAMES_FILE")
need=$((want * 2))
# `|| true`, and the -d guard: under `set -euo pipefail` a find over a view that does not exist
# yet fails the pipeline and kills the script before it can build the thing it was asked for.
have=0
[ -d "$VIEW" ] && have=$( (find "$VIEW" -maxdepth 3 -type l 2>/dev/null || true) | wc -l)
if [ "$have" -ge "$need" ]; then
    echo "      ($VIEW already complete: $have symlinks)"
    exit 0
fi

echo "      + build $VIEW ($want activation + $want trajectory symlinks) from $(basename "$SRC")"
[ -n "$DRY_RUN" ] && exit 0

while read -r name; do
    [ -z "$name" ] && continue
    size=${name#*_size}; size="size${size%%_*}"
    mkdir -p "$VIEW/activations/$size" "$VIEW/trajectories/$size"
    ln -sfn "$SRC/$size/$name" "$VIEW/activations/$size/$name"
    ln -sfn "$TRAJ/$size/$name.json" "$VIEW/trajectories/$size/$name.json"
done < "$NAMES_FILE"

echo "      built $(find "$VIEW" -maxdepth 3 -type l | wc -l) symlinks"
