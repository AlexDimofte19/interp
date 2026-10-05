#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)
ARM=jlens
ACT_TRAIN=/workspace/activations/gptoss_grid_mass_l14
ACT_VAL=/workspace/activations/gptoss_grid_mass_l14_eval
TRAJ_TRAIN=/workspace/activations/mass_train2880_view/trajectories
TRAJ_VAL=/workspace/activations/mass_eval720_view/trajectories
PREP=/workspace/prepared/gptoss_p2_grid
LAYER=14
PROBE_TYPE=grid_tile
MAX_CELLS=25
SEED=42
PAD=15

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
