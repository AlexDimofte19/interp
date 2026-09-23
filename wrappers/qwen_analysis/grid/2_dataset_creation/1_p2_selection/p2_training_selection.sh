#!/usr/bin/env bash
# Qwen P2 GRID, step 3: the selecting gather on the TRAIN set, ranked by GRID loudness --
# three arms, one forward pass, activations saved. p2_eval_selection.sh is the identical run
# on mass_eval_144.
#
# The grid twin of ../../../2_dataset_creation/1_p2_selection/p2_training_selection.sh. Same
# trajectories, same sample, same arms, same N, same seeds. Only the vocabulary and the output
# tree differ.
#
# WHY A NEW TREE AND NOT THE DIRECTION ONE. qwen_p2_selection holds a .pt only for the tokens
# the DIRECTION arms picked. A grid-ranked arm wants different tokens, and those were never
# saved. Nor can it be --extend'ed in: an arm's name is its method name ("jlens"), so a
# grid-ranked jlens arm would collide with the direction-ranked one already in the record.
#
# THREE ARMS IN ONE RUN, AND THE CONTROL CANNOT BE DEFERRED. The random arm is the matched
# control; once the tree holds only the loud tokens, a uniform draw over the chain can never
# be made again.
#
# --signal-name grid IS REQUIRED: the filename alone would name the signal
# "grid_tokens_pruned_qwen3-6-35b-a3b" and every loudness column after it.
#
# THE LAYER IS INHERITED, NOT YET RE-DERIVED. LAYER=27 is the DIRECTION profile's j-lens
# argmax. The grid profile (../../1_loudest_layer/) has not run. If its argmax is not 27,
# change LAYER here, in p2_eval_selection.sh and everywhere downstream. The held-out
# activations (qwen_p2_heldout) exist only at 27, so a different layer also means a full
# held-out re-gather, not just a CSV pass. Do not pick 39: that is the Jacobian's fit target,
# where the two lenses are identical and the jlens and logitlens arms collapse into one.
#
# IF YOU PRUNE THIS TREE LATER, delete_non_jlens_selected.py --candidate-layers must be given
# the same pool, or it keeps different files than this gather wrote.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)   # so uv finds pyproject.toml

MODEL=Qwen/Qwen3.6-35B-A3B
DATASET=/workspace/trajectories/qwen3.6-35b/replayed_single_step/mass_train_576
JLENS_DIR=/workspace/jlens/qwen3_6_35b
SIGNAL_JSON=/workspace/repo/interp/data/jlens/qwen/grid_tokens_pruned_qwen3-6-35b-a3b.json
SIGNAL_NAME=grid
# ^ the SAME vocabulary the grid step-1 profile was gathered against. A mass table is not
# self-describing; its fingerprint is recorded in every .meta.json written here.
OUT=/workspace/activations/qwen_p2_grid_selection

LAYER=27               # INHERITED from the direction line; see the header
SAMPLE_PERCENT=0.2     # the subsample all three arms select from
SAMPLE_SEED=42

METHODS=jlens,logitlens,random
SCORE=logprob_mass_full   # rank on the mass table, not the top-20 count; NOT the default
NUM_TOKENS=60          # P2: the loudest 60 tokens per trajectory, per scored arm
RANDOM_TOKENS=60       # the control, matched to NUM_TOKENS
NUM_LAYERS=1           # one candidate layer, so one layer per selected token
ALWAYS_LAYERS=""       # the default is 15, a gpt-oss convention; 15/40 is not mid-stack here
SELECT_SEED=42         # the control draw, combined with each trajectory stem

LENS=both              # required: the logitlens arm ranks on its own CSV
LAYERS=$LAYER          # what the CSV and mass table cover; 'all' re-emits the full profile
BATCH_SIZE=256         # reasoning tokens per lens matmul, caps the [B, vocab] logits
FORWARD_BATCH_SIZE=1   # one 18k-token chain at a time; the default of 4 pads four into one pass

# Single GPU: device_map="auto" across several produces NaNs for this MoE, as it does for gpt-oss.
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
    --forward-batch-size "$FORWARD_BATCH_SIZE" \
    --device cuda
