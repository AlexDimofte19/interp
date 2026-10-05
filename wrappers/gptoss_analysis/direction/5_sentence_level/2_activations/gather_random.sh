#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)
ARM=random
KIND=random_in_sentence
ENDPOINTS=""
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
export CUDA_VISIBLE_DEVICES=0
cd "$REPO"
uv run --extra gpu python telos_interp/loudness_analysis/rollouts/gather_local_belief_activations.py \
    --rollout-dir "${ROLLOUTS}/${ARM}_train" \
    --trajectory-paths "$TRAJ_TRAIN" \
    --names-file "$NAMES_TRAIN" \
    --out "${ACT}_${ARM}_train" \
    --interior-kinds "$KIND" \
    $ENDPOINTS \
    --model-id "$MODEL" \
    --device-map cuda
uv run --extra gpu python telos_interp/loudness_analysis/rollouts/gather_local_belief_activations.py \
    --rollout-dir "${ROLLOUTS}/${ARM}_val" \
    --trajectory-paths "$TRAJ_VAL" \
    --names-file "$NAMES_VAL" \
    --out "${ACT}_${ARM}_val" \
    --interior-kinds "$KIND" \
    $ENDPOINTS \
    --model-id "$MODEL" \
    --device-map cuda
