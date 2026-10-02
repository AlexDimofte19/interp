#!/usr/bin/env bash
# gpt-oss P2 GRID, step 3b: the GRID-ranked selecting gather on the EVAL 720.
#
# Byte-for-byte the same run as p2_training_selection.sh beside this file, pointed at the
# mass-era eval view and writing its own tree. Read that script's header for why each flag is
# what it is (layer 14, and a fresh random arm). Only DATASET and OUT differ. Keep the two
# in step.
#
# The 720 are the pinned next_action_mass_l15_eval_names.txt every gpt-oss probe on disk is
# scored against. It is a plain random draw, not stratified, but it covers every grid size,
# unlike Qwen's 52.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../grid_layer.sh"   # GRID_LAYER

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)   # so uv finds pyproject.toml

MODEL=openai/gpt-oss-20b
DATASET=/workspace/activations/mass_eval720_view/trajectories
JLENS_DIR=/workspace/jlens/gridenv   # the GRID-ENVIRONMENT lens; /workspace/jlens holds the wikitext fit (self-check 2026-09-24)
SIGNAL_JSON=/workspace/repo/interp/data/jlens/grid_tokens_pruned.json
SIGNAL_NAME=grid
OUT=/workspace/activations/gptoss_grid_mass_l${GRID_LAYER}_eval

LAYER=$GRID_LAYER               # the grid profile's J-lens argmax; see the header
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
