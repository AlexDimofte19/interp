#!/usr/bin/env bash
# gpt-oss P2 GRID, dataset creation step 2b: the grid_tile PREPARE stage for the LOGITLENS arm -- train and val.
#
# The gpt-oss grid twin of wrappers/qwen_analysis/grid/2_dataset_creation/2_preparations/prepare_logitlens.sh.
# The label is the identity of each grid CELL. The binary probes in ../../3_train_and_eval_probes
# read one class of it as positive (empty/wall/agent/goal). THERE IS NO ROLLOUT STAGE: a cell's
# contents are read off grid_state, so there is no local belief to measure.
#
# THE ARM IS THE ONLY THING THAT DIFFERS between this file and its two siblings (plus which
# tree it lives in). Same trajectories, same layer, same cells. Here: the 20 tokens per
# trajectory the LOGIT lens scored most GRID-loaded at layer 14 (the grid profile's argmax), from the grid gathers in
# ../1_p2_selection.
#
# recorded_<arm>: only selected tokens have a .pt, and the control can only be read back.
#
# THE CELLS match grid_cell_analysis/binary_grid_probes.sh and the existing grid_binary_l15_*
# datasets: 25 per (trajectory, step), seeded per (trajectory, step) so every arm sees the SAME
# cells, and no --balance-classes-per-trajectory (4 cells per step instead of 25, ICLR entry 55).
#
# THE DATASET IS PADDED TO 15, pinned by PAD below. binary_grid_probes.sh's header says native
# size, but prepare's multi-size path replaces pad_to_size=None with the widest size present, and
# grid_binary_l15_random's manifest records "pad_to_size": 15. For a size-5 grid most of the
# 25 drawn cells are '+' padding, which every binary probe gets as an easy negative. This
# matches the existing datasets, so the numbers stay comparable. Fixing it needs a code
# change in prepare, not a flag.
#
# Cheap: nothing is copied. CPU only.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../grid_layer.sh"   # GRID_LAYER

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)   # repo root

ARM=logitlens

ACT_TRAIN=/workspace/activations/gptoss_grid_mass_l${GRID_LAYER}
ACT_VAL=/workspace/activations/gptoss_grid_mass_l${GRID_LAYER}_eval
TRAJ_TRAIN=/workspace/activations/mass_train2880_view/trajectories
TRAJ_VAL=/workspace/activations/mass_eval720_view/trajectories

PREP=/workspace/prepared/gptoss_p2_grid        # -> ${PREP}_${ARM}_{train,val}

LAYER=$GRID_LAYER
PROBE_TYPE=grid_tile
MAX_CELLS=25           # cells per (trajectory, step); matches grid_binary_l15_* and the Qwen grid line
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
