#!/usr/bin/env bash
# A symlink view of an activation tree, restricted to one trajectory set.
#
# A tree holds every trajectory it was gathered over; a consumer almost always wants a SUBSET
# -- the count-era 3,600, the mass-era 3,600, the 2,880 train half, the 720 eval half -- and
# `prepare_activations_for_probing` and the per-token eval scripts take a DIRECTORY, not a name
# list. So the subset has to exist as a directory. Symlinks rather than copies: the same bytes,
# and one symlink per trajectory instead of thousands per tensor.
#
# The link is at the TRAJECTORY level, so whatever else the gather wrote inside
# `{size}/{name}/` -- the analysis CSV, the direction-mass table and its `.meta.json`, the
# selection record -- comes along with the activations and stays consistent with them.
#
# The view pairs activations with trajectories so one --activations-dir / --trajectories-dir
# pair addresses exactly the wanted names, and nothing outside them can leak in.
#
#   SRC=/workspace/activations/jlens_mass_l15 \
#       bash scripts/build_activation_view.sh <names-file> <view-dir>
#
# SRC defaults to the 36,000-trajectory sentence-end tree, which is where this started and is
# still its most common use. Idempotent: a view already holding 2x its name count is left
# alone. DRY_RUN=1 reports only.
set -euo pipefail

NAMES_FILE=${1:?usage: build_activation_view.sh <names-file> <view-dir>}
VIEW=${2:?usage: build_activation_view.sh <names-file> <view-dir>}
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
