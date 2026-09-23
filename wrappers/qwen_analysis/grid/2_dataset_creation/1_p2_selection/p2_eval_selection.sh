#!/usr/bin/env bash
# Qwen P2 GRID, step 3b: the grid-ranked selecting gather on the EVAL set.
#
# Byte-for-byte the same run as p2_training_selection.sh beside this file (same vocabulary,
# arms, layer, sample fraction and seeds), pointed at mass_eval_144 and writing its own tree.
# Read that script's header for why each flag is what it is. Only DATASET and OUT differ.
# Keep the two in step: an eval arm selected under different parameters than the train arm
# it scores is not the same measurement.
#
# The seeds are combined with each trajectory's stem inside the script, so the same value
# here does NOT reuse the train set's draw. The trajectories are disjoint, so the draws are too.
#
# mass_eval_144 IS INCOMPLETE: it held 55 trajectory JSONs (52 gathered) when the direction
# line ran. Every grid number on it rests on the same 52.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)   # so uv finds pyproject.toml

MODEL=Qwen/Qwen3.6-35B-A3B
DATASET=/workspace/trajectories/qwen3.6-35b/replayed_single_step/mass_eval_144
JLENS_DIR=/workspace/jlens/qwen3_6_35b
SIGNAL_JSON=/workspace/repo/interp/data/jlens/qwen/grid_tokens_pruned_qwen3-6-35b-a3b.json
SIGNAL_NAME=grid
OUT=/workspace/activations/qwen_p2_grid_selection_eval

LAYER=27               # keep equal to the train run's; INHERITED from the direction line
SAMPLE_PERCENT=0.2     # the subsample all three arms select from
SAMPLE_SEED=42

METHODS=jlens,logitlens,random
SCORE=logprob_mass_full   # rank on the mass table, not the top-20 count; NOT the default
NUM_TOKENS=60
RANDOM_TOKENS=60
NUM_LAYERS=1
ALWAYS_LAYERS=""       # the default is 15, a gpt-oss convention
SELECT_SEED=42

LENS=both              # required: the logitlens arm ranks on its own CSV
LAYERS=$LAYER
BATCH_SIZE=256
FORWARD_BATCH_SIZE=1

# Single GPU: device_map="auto" across several produces NaNs for this MoE.
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
