#!/usr/bin/env bash
# gpt-oss P2 GRID, step 3: the GRID-ranked selecting gather on the TRAIN 2,880. Three arms,
# one forward pass, activations saved. p2_eval_selection.sh is the identical run on the 720.
#
# The gpt-oss grid twin of wrappers/qwen_analysis/2_dataset_creation/1_p2_selection/p2_training_selection.sh.
# It matches the reused direction tree (jlens_mass_l15) in N=20, ranking by logprob_mass_full
# and the whole chain (no --data_sample_p). It differs in the vocabulary and in the LAYER.
#
# LAYER 14, NOT 15. The grid loudest-layer profile (../../1_loudest_layer/, train 2,880, whole
# chain) puts the J-lens grid argmax at L14 (-3.710 vs L13 -3.778, L15 -4.024), in 500 of 500
# trajectory resamples. On the direction line 15 is a convention; for the grid it would be
# the wrong layer.
#
# SO NOTHING AT LAYER 15 IS REUSED, INCLUDING THE CONTROL. jlens_mass_l15's recorded random
# arm was drawn at candidate layer 15 and only has layer-15 tensors, so this gather draws its
# own matched random arm at 14, before anything is pruned. The same goes for the held-out
# tensors (heldout_sample.sh gathers them at 14).
#
# WHY THIS ONE MUST GATHER. jlens_mass_l15 is pruned to the DIRECTION arms' tokens, and no tree
# holds full-chain train activations (activations_train_single_step_reasoning_all keeps 3
# output tokens per trajectory). --extend cannot add them either: an arm's name is its method,
# so a grid-ranked "jlens" would collide with the recorded one.
#
# THE TRAJECTORIES are the mass-era view's trajectory folder, so this tree covers the 2,880 and
# nothing else, and p2_eval_selection.sh's tree covers the 720.
#
# --signal-name grid IS REQUIRED: without it every loudness column would be named after the
# filename ("grid_tokens_pruned").
#
# IF YOU PRUNE THIS TREE LATER, give delete_non_jlens_selected.py --candidate-layers 14.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)   # so uv finds pyproject.toml

MODEL=openai/gpt-oss-20b
DATASET=/workspace/activations/mass_train2880_view/trajectories
JLENS_DIR=/workspace/jlens   # this host keeps gpt-oss-20b_jacobian_lens.pt + _unembed.pt here; its gridenv/ holds only ckpt.pt
SIGNAL_JSON=/workspace/repo/interp/data/jlens/grid_tokens_pruned.json
SIGNAL_NAME=grid
OUT=/workspace/activations/gptoss_grid_mass_l14

LAYER=14               # the grid profile's J-lens argmax; see the header
SAMPLE_PERCENT=1.0     # NO SAMPLING: select over the whole chain, as jlens_mass_l15 did
SAMPLE_SEED=42

METHODS=jlens,logitlens,random   # the control is drawn HERE, at 14; the layer-15 one cannot be reused
SCORE=logprob_mass_full   # rank on the mass table, not the top-20 count; NOT the default
NUM_TOKENS=20          # matches the gpt-oss direction arms
RANDOM_TOKENS=20       # the control, matched to NUM_TOKENS
NUM_LAYERS=1           # one candidate layer, so one layer per selected token
ALWAYS_LAYERS=""       # the default force-keeps 15, which this line does not use
SELECT_SEED=42         # the control draw, combined with each trajectory stem

LENS=both              # required: the logitlens arm ranks on its own CSV
LAYERS=$LAYER          # the CSV and mass table; the full 7:23 profile is 1_loudest_layer's
BATCH_SIZE=256         # reasoning tokens per lens matmul

# Single GPU: device_map="auto" across several produces NaNs for this MoE.
export CUDA_VISIBLE_DEVICES=0

[ -f "$SIGNAL_JSON" ] || { echo "!! grid vocabulary not found: $SIGNAL_JSON" >&2; exit 1; }

cd "$REPO"
uv run --extra gpu python telos_interp/loudness_analysis/build_loudness_tables.py \
    --trajectory-paths "$DATASET" \
    --activations-dir "$OUT" \
    --model-id "$MODEL" \
    --jlens_dir "$JLENS_DIR" \
    --signal-json "$SIGNAL_JSON" \
    --signal-name "$SIGNAL_NAME" \
    --direction-score "$SCORE" \
    --select-methods "$METHODS" \
    --select-num-tokens "$NUM_TOKENS" \
    --select-random-tokens "$RANDOM_TOKENS" \
    --select-num-layers "$NUM_LAYERS" \
    --select-always-layers "$ALWAYS_LAYERS" \
    --select-candidate-layers "$LAYER" \
    --select-seed "$SELECT_SEED" \
    --data_sample_p "$SAMPLE_PERCENT" \
    --data-sample-seed "$SAMPLE_SEED" \
    --lens "$LENS" \
    --layers "$LAYERS" \
    --batch-size "$BATCH_SIZE" \
    --device cuda
