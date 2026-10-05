#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)
ARM=logitlens
TRAJ_TRAIN=/workspace/activations/mass_train2880_view/trajectories
TRAJ_VAL=/workspace/activations/mass_eval720_view/trajectories
NAMES_TRAIN=/workspace/splits/mass_train_2880.txt
NAMES_VAL=/workspace/splits/mass_eval_720.txt
LENS_TRAIN=/workspace/activations/gptoss_direction_mass_l15
LENS_VAL=/workspace/activations/gptoss_direction_mass_l15_eval
ROLLOUTS=/workspace/rollouts/gptoss_sentence
ACT=/workspace/activations/gptoss_sentence
PREP=/workspace/prepared/gptoss_sentence
MODEL=openai/gpt-oss-20b
LAYER=15
cd "$REPO"
uv run interp-cli prepare_activations_for_probing \
    --activations-dir "${ACT}_${ARM}_train" \
    --trajectories-dir "$TRAJ_TRAIN" \
    --probe-type next_action \
    --layers "$LAYER" \
    --steps all \
    --output-indices all \
    --output-path "${PREP}_${ARM}_train_final" \
    --verbose
uv run python telos_interp/loudness_analysis/rollouts/relabel_manifest_from_rollout.py \
    "${PREP}_${ARM}_train_final" "${ROLLOUTS}/${ARM}_train" "${PREP}_${ARM}_train_local" \
    --report-csv "${PREP}_${ARM}_train_relabel.csv"
uv run interp-cli prepare_activations_for_probing \
    --activations-dir "${ACT}_${ARM}_val" \
    --trajectories-dir "$TRAJ_VAL" \
    --probe-type next_action \
    --layers "$LAYER" \
    --steps all \
    --output-indices all \
    --output-path "${PREP}_${ARM}_val_final" \
    --verbose
uv run python telos_interp/loudness_analysis/rollouts/relabel_manifest_from_rollout.py \
    "${PREP}_${ARM}_val_final" "${ROLLOUTS}/${ARM}_val" "${PREP}_${ARM}_val_local" \
    --report-csv "${PREP}_${ARM}_val_relabel.csv"
