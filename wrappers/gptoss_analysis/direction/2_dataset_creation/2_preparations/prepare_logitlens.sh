#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)
ARM=logitlens
ACT_TRAIN=/workspace/activations/gptoss_direction_mass_l15
ACT_VAL=/workspace/activations/gptoss_direction_mass_l15_eval
TRAJ_TRAIN=/workspace/activations/mass_train2880_view/trajectories
TRAJ_VAL=/workspace/activations/mass_eval720_view/trajectories
PREP=/workspace/prepared/gptoss_p2
LAYER=15
PROBE_TYPE=next_action

cd "$REPO"
uv run interp-cli prepare_activations_for_probing \
    --activations-dir "$ACT_TRAIN" \
    --trajectories-dir "$TRAJ_TRAIN" \
    --probe-type "$PROBE_TYPE" \
    --layers "$LAYER" \
    --steps all \
    --output-indices all \
    --token-selection "recorded_${ARM}" \
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
    --output-path "${PREP}_${ARM}_val" \
    --verbose
