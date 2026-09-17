#!/usr/bin/env bash
# Qwen P2, step 4: prepare the probe datasets -- three arms, train and val.
#
# Six prepares, no split script. The train/val partition already exists as two separate
# gathers (mass_train_576 -> qwen_p2_selection, mass_eval_144 -> qwen_p2_selection_eval),
# so split_next_action_manifest.py has nothing to do here: each tree is prepared on its
# own and the two manifests are handed to the trainer as --train-data-path and
# --eval-data-path. When --eval-data-path is given the trainer's own --eval-split is
# ignored, which is what keeps a trajectory's tokens out of both halves.
#
# recorded_<arm>, NEVER <lens>_direction. These trees were gathered selectively, so only
# the selected tokens have a .pt. Re-scoring a CSV would still find the loud ones, but the
# random arm cannot be recomputed at all -- a uniform draw over the reasoning chain has to
# be read back from the record that reserved it.
#
# THE SAME ARM ON BOTH SIDES. Each tree carries its own selection record, so the val
# prepare has to name the same recorded_<arm> as the train prepare. Crossing them trains
# on one selection and scores on another, which answers nothing.
#
# LAYER 27 EVERYWHERE, pinned here at prepare time rather than by --single-layer later.
# It is the only layer in either tree -- step 3 gathered with --select-candidate-layers 27
# -- so this is a statement of intent, not a filter that drops anything.
#
# next_action copies no activations: each manifest references the tree in place through an
# absolute activations_root, so all six outputs are a lone manifest.json and cost nothing.
# CPU only, no model, minutes.
set -euo pipefail

REPO=/workspace/repo/interp

ACT_TRAIN=/workspace/activations/qwen_p2_selection
TRAJ_TRAIN=/workspace/trajectories/qwen3.6-35b/replayed_single_step/mass_train_576

ACT_VAL=/workspace/activations/qwen_p2_selection_eval
TRAJ_VAL=/workspace/trajectories/qwen3.6-35b/replayed_single_step/mass_eval_144

OUT=/workspace/prepared/qwen_p2        # datasets land at ${OUT}_${arm}_{train,val}

LAYER=27
PROBE_TYPE=next_action

cd "$REPO"

# ---------------------------------------------------------------- jlens (train, val)
uv run interp-cli prepare_activations_for_probing \
    --activations-dir "$ACT_TRAIN" \
    --trajectories-dir "$TRAJ_TRAIN" \
    --probe-type "$PROBE_TYPE" \
    --layers "$LAYER" \
    --steps all \
    --output-indices all \
    --token-selection recorded_jlens \
    --output-path "${OUT}_jlens_train" \
    --verbose

uv run interp-cli prepare_activations_for_probing \
    --activations-dir "$ACT_VAL" \
    --trajectories-dir "$TRAJ_VAL" \
    --probe-type "$PROBE_TYPE" \
    --layers "$LAYER" \
    --steps all \
    --output-indices all \
    --token-selection recorded_jlens \
    --output-path "${OUT}_jlens_val" \
    --verbose

# ----------------------------------------------------------- logitlens (train, val)
uv run interp-cli prepare_activations_for_probing \
    --activations-dir "$ACT_TRAIN" \
    --trajectories-dir "$TRAJ_TRAIN" \
    --probe-type "$PROBE_TYPE" \
    --layers "$LAYER" \
    --steps all \
    --output-indices all \
    --token-selection recorded_logitlens \
    --output-path "${OUT}_logitlens_train" \
    --verbose

uv run interp-cli prepare_activations_for_probing \
    --activations-dir "$ACT_VAL" \
    --trajectories-dir "$TRAJ_VAL" \
    --probe-type "$PROBE_TYPE" \
    --layers "$LAYER" \
    --steps all \
    --output-indices all \
    --token-selection recorded_logitlens \
    --output-path "${OUT}_logitlens_val" \
    --verbose

# --------------------------------------------------------------- random (train, val)
# The matched control: a uniform draw over each chain's reasoning tokens, reserved by the
# step-3 gather with the same N and seed as the two lens arms.
uv run interp-cli prepare_activations_for_probing \
    --activations-dir "$ACT_TRAIN" \
    --trajectories-dir "$TRAJ_TRAIN" \
    --probe-type "$PROBE_TYPE" \
    --layers "$LAYER" \
    --steps all \
    --output-indices all \
    --token-selection recorded_random \
    --output-path "${OUT}_random_train" \
    --verbose

uv run interp-cli prepare_activations_for_probing \
    --activations-dir "$ACT_VAL" \
    --trajectories-dir "$TRAJ_VAL" \
    --probe-type "$PROBE_TYPE" \
    --layers "$LAYER" \
    --steps all \
    --output-indices all \
    --token-selection recorded_random \
    --output-path "${OUT}_random_val" \
    --verbose
