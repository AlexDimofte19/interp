#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)
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
for arm in jlens logitlens random; do
    uv run python telos_interp/loudness_analysis/rollouts/link_end_of_reasoning_activations.py \
        "${ROLLOUTS}/${arm}_train" "${ACT}_${arm}_train" \
        --source-tree "${ACT}_eos_train" \
        --names-file "$NAMES_TRAIN" \
        --apply
    uv run python telos_interp/loudness_analysis/rollouts/link_end_of_reasoning_activations.py \
        "${ROLLOUTS}/${arm}_val" "${ACT}_${arm}_val" \
        --source-tree "${ACT}_eos_val" \
        --names-file "$NAMES_VAL" \
        --apply
done
