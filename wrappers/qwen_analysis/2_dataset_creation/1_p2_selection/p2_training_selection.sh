#!/usr/bin/env bash
# Qwen P2, step 3: the selecting gather on the TRAIN set -- three arms, one forward pass,
# activations saved. p2_eval_selection.sh is the identical run on mass_eval_144.
#
# Same shape as ../loudest_layer/sample_loudness_profile.sh, with three differences: it
# passes --signal-json (which is what switches the script into selective mode), it names a
# layer to select at, and it does NOT pass --no-save-activations, so the selected .pt files
# land on disk in gather_activations' exact tree layout.
#
# THREE ARMS IN ONE RUN, AND THE CONTROL CANNOT BE DEFERRED. --select-methods
# jlens,logitlens,random records all three in the same pass. The random arm is the matched
# control: once the tree holds only the loud tokens, a uniform draw over the reasoning chain
# can never be made again, so it is reserved now or never. --lens both is required because
# the logitlens arm needs its own CSV to rank on; it costs one extra unembed per chunk, not
# a second forward pass.
#
# WHAT --data_sample_p MEANS HERE. It narrows the pool BEFORE the selection runs, which is
# the intent: all three arms choose from the same uniform 20% of each step's reasoning
# tokens. The top-60 is therefore the loudest 60 of the sample, not of the chain, and the
# control draws from that same sample -- so the arms stay matched, which is the only
# property the comparison needs. Set it to 1.0 to select over the full chain instead.
#
# WHY LAYERS AND CANDIDATE LAYERS ARE THE SAME NUMBER. --select-candidate-layers narrows
# what the selection ranks and saves; --layers narrows what the CSV and the mass table
# cover. Step 1 already wrote the full (token x layer) profile over this dataset, so there
# is nothing to gain by paying for every layer's unembed again -- both are pinned to the one
# layer. Set LAYERS=all to re-emit the full profile alongside the selection.
#
# IF YOU PRUNE THIS TREE LATER, delete_non_jlens_selected.py --candidate-layers must be given
# the same pool, or it keeps different files than this gather wrote.
#
# The fit and the vocabulary are both in place; step 1 ran against them.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)   # so uv finds pyproject.toml

MODEL=Qwen/Qwen3.6-35B-A3B
DATASET=/workspace/trajectories/qwen3.6-35b/replayed_single_step/mass_train_576
JLENS_DIR=/workspace/jlens/qwen3_6_35b
SIGNAL_JSON=/workspace/repo/interp/data/jlens/qwen/direction_tokens_full_qwen3-6-35b-a3b.json
# ^ the SAME vocabulary the step-1 profile was gathered against. A mass table is not
# self-describing, so pointing this elsewhere silently produces numbers step 1's layer
# choice has no bearing on. Its fingerprint is recorded in every .meta.json written here.
OUT=/workspace/activations/qwen_p2_selection

LAYER=27               # the step-2 notebook's j-lens argmax over all 549 trajectories
# -5.249 nats, clear of the runner-up L19 (-5.495) by 0.25. NOT 39: that is the Jacobian's
# fit target, where the lens is the identity -- both lenses read -10.353720 there, to six
# decimals over all 1,841,963 rows -- so selecting at 39 would collapse the jlens and
# logitlens arms into the same arm and this run would answer nothing.
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
