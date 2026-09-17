#!/usr/bin/env bash
# Qwen P2, step 3b: the selecting gather on the EVAL set.
#
# Byte-for-byte the same run as p2_training_selection.sh beside this file -- same arms, same
# layer, same sample fraction, same seeds -- pointed at mass_eval_144 and writing its own
# tree. Read that script's header for why each flag is what it is; only DATASET and OUT
# differ. Keep the two in step: an eval arm selected under different parameters than the
# train arm it scores is not the same measurement.
#
# THE SEEDS ARE DELIBERATELY IDENTICAL. --data-sample-seed and --select-seed are combined
# with each trajectory's stem inside the script, so the same value here does NOT reuse the
# train set's draw -- it just means the two runs are reproducible the same way. The
# trajectories are disjoint, so the draws are too.
#
# ###################################################################################
# # mass_eval_144 IS INCOMPLETE. The directory holds 55 trajectory JSONs, not 144    #
# # (the inventory recorded 52 at the time it was written, so the replay may still   #
# # be running, or it stalled). Gathering now produces an eval set that is 38% of     #
# # the one the name promises, and every accuracy computed on it will be quoted      #
# # against a number that does not exist. Settle whether the replay is finished       #
# # before spending GPU here.                                                         #
# ###################################################################################
#
# The fit and the vocabulary are both in place; step 1 ran against them.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)   # so uv finds pyproject.toml

MODEL=Qwen/Qwen3.6-35B-A3B
DATASET=/workspace/trajectories/qwen3.6-35b/replayed_single_step/mass_eval_144
JLENS_DIR=/workspace/jlens/qwen3_6_35b
SIGNAL_JSON=/workspace/repo/interp/data/jlens/qwen/direction_tokens_full_qwen3-6-35b-a3b.json
# ^ the SAME vocabulary the step-1 profile was gathered against. A mass table is not
# self-describing, so pointing this elsewhere silently produces numbers step 1's layer
# choice has no bearing on. Its fingerprint is recorded in every .meta.json written here.
OUT=/workspace/activations/qwen_p2_selection_eval

LAYER=27               # keep equal to the train run's; see its header for why not 39
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

# 66 GiB of weights on an 80 GiB card leaves ~13 GiB for a 33k-token chain. Step 1 died there
# with 7.58 GiB reserved-but-unallocated -- fragmentation, not a real shortage.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
uv run --extra gpu python telos_interp/loudness_analysis/build_loudness_tables.py \
    --trajectory-paths "$DATASET" \
    --activations-dir "$OUT" \
    --model-id "$MODEL" \
    --jlens_dir "$JLENS_DIR" \
    --signal-json "$SIGNAL_JSON" \
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
