#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)
ARM=random
ACT_TRAIN=/workspace/activations/qwen_p2_selection
ACT_VAL=/workspace/activations/qwen_p2_selection_eval
TRAJ_TRAIN=/workspace/trajectories/qwen3.6-35b/replayed_single_step/mass_train_576
TRAJ_VAL=/workspace/trajectories/qwen3.6-35b/replayed_single_step/mass_eval_144
PREP=/workspace/prepared/qwen_p2
LAYER=27
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
