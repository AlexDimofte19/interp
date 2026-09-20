#!/usr/bin/env bash
# Qwen P2, dataset creation step 2c: the PREPARE stage for the RANDOM control arm -- train and val.
#
# Stage 1 of three. This one builds token-major manifests labelled with the trajectory's
# final agent_action; ../3_rollouts/rollout_random.sh measures the local belief at those same
# tokens and swaps it in. A dataset is not finished until that has run.
#
# THE ARM IS THE ONLY THING THAT DIFFERS between this file and its two siblings: same tree,
# same trajectories, same layer. ARM below names which set of picks in
# {stem}_jlens_selection.json is read -- here, the matched CONTROL: 60 tokens drawn uniformly
# over the whole reasoning chain at gather time, with the same N and seed as the lens arms.
#
# ITS DRAW CANNOT BE RE-MADE, which is the whole reason recorded_random exists. Re-deriving
# it now would draw from the selection rather than from the chain, and the control would stop
# being a control.
#
# recorded_<arm>, NEVER <lens>_direction. These trees were gathered selectively, so only the
# selected tokens have a .pt. Re-scoring a CSV would still find the loud ones, but the random
# arm cannot be recomputed at all -- a uniform draw over the reasoning chain has to be read
# back from the record that reserved it.
#
# Cheap: next_action copies no activations. Each manifest references the tree in place
# through an absolute activations_root, so both outputs are a lone manifest.json. CPU only.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)   # repo root, as ../../p2_selection does

ARM=random

ACT_TRAIN=/workspace/activations/qwen_p2_selection
TRAJ_TRAIN=/workspace/trajectories/qwen3.6-35b/replayed_single_step/mass_train_576

ACT_VAL=/workspace/activations/qwen_p2_selection_eval
TRAJ_VAL=/workspace/trajectories/qwen3.6-35b/replayed_single_step/mass_eval_144

PREP=/workspace/prepared/qwen_p2        # -> ${PREP}_${ARM}_{train,val}

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
