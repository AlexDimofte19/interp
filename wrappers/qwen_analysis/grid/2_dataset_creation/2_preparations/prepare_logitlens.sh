#!/usr/bin/env bash
# Qwen P2 GRID, dataset creation step 2b: the grid_tile PREPARE stage for the LOGITLENS arm -- train and val.
#
# The grid twin of ../../../2_dataset_creation/2_preparations/prepare_logitlens.sh. The label is
# the identity of each grid CELL, not the next action. The binary probes in
# ../../3_train_and_eval_probes read one class out of it as positive (empty/wall/agent/goal).
# THERE IS NO ROLLOUT STAGE: a cell's contents are read straight off the trajectory's
# grid_state, so there is no "local belief" to measure and nothing to relabel. A dataset is
# finished when this has run.
#
# THE ARM IS THE ONLY THING THAT DIFFERS between this file and its two siblings: same tree,
# same trajectories, same layer, same cells. ARM names which picks in
# {stem}_jlens_selection.json are read. Here: the tokens the LOGIT lens scored most
# GRID-loaded at layer 27.
#
# recorded_<arm>, NEVER <lens>_direction: the tree was gathered selectively, so only the
# selected tokens have a .pt, and the random arm can only be read back from the record.
#
# THE CELLS, as grid_cell_analysis/binary_grid_probes.sh sets them (ICLR entry 55):
#   * MAX_CELLS=25 cells per (trajectory, step), seeded per (trajectory, step), so every
#     arm is scored on the SAME cells.
#   * NO --balance-classes-per-trajectory: with a cell cap it yields 4 cells per step, not 25,
#     because every grid holds exactly one 'A' and one 'G'. Imbalance is handled at train
#     time with --class-weight balanced.
#
# !! THE DATASET IS PADDED, WHATEVER binary_grid_probes.sh SAYS. That script leaves out
# --pad-to-size to get native-size grids, but prepare_activations_for_probing's multi-size
# path (_process_multi_size) replaces pad_to_size=None with the widest size present. So
# every manifest comes out padded to 15. The existing grid_binary_l15_* manifests record
# "pad_to_size": 15. For a size-5 grid most of the 25 drawn cells are then '+' padding, and
# every binary probe gets them as easy negatives. This wrapper matches those datasets, so
# the numbers stay comparable. It does NOT fix the problem, which needs a code change in
# prepare (an explicit "native" option). Read every per-size number with this in mind.
#
# Cheap: nothing is copied. Each manifest references the tree in place through an absolute
# activations_root, and the per-cell payload is stored once per (trajectory, step). CPU only.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)   # repo root

ARM=logitlens

ACT_TRAIN=/workspace/activations/qwen_p2_grid_selection
TRAJ_TRAIN=/workspace/trajectories/qwen3.6-35b/replayed_single_step/mass_train_576

ACT_VAL=/workspace/activations/qwen_p2_grid_selection_eval
TRAJ_VAL=/workspace/trajectories/qwen3.6-35b/replayed_single_step/mass_eval_144

PREP=/workspace/prepared/qwen_p2_grid        # -> ${PREP}_${ARM}_{train,val}

LAYER=27
PROBE_TYPE=grid_tile
MAX_CELLS=25           # cells per (trajectory, step); matches grid_cell_analysis/binary_grid_probes.sh
SEED=42                # seeds the per-(trajectory, step) cell draw; keep equal across arms
PAD=15                 # PINNED, never auto: auto pads to the widest size PRESENT, and the Qwen eval
                       # set has no size-15 grid, so it came out padded to 13 against train's 15

cd "$REPO"

uv run interp-cli prepare_activations_for_probing \
    --activations-dir "$ACT_TRAIN" \
    --trajectories-dir "$TRAJ_TRAIN" \
    --probe-type "$PROBE_TYPE" \
    --layers "$LAYER" \
    --steps all \
    --output-indices all \
    --token-selection "recorded_${ARM}" \
    --seed "$SEED" \
    --max-positions-per-trajectory "$MAX_CELLS" \
    --pad-to-size "$PAD" \
    --output-path "${PREP}_${ARM}_train" \
    --verbose

uv run interp-cli prepare_activations_for_probing \
    --activations-dir "$ACT_VAL" \
    --trajectories-dir "$TRAJ_VAL" \
    --probe-type "$PROBE_TYPE" \
    --layers "$LAYER" \
    --steps all \
    --output-indices all \
    --token-selection "recorded_${ARM}" \
    --seed "$SEED" \
    --max-positions-per-trajectory "$MAX_CELLS" \
    --pad-to-size "$PAD" \
    --output-path "${PREP}_${ARM}_val" \
    --verbose
