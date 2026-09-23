#!/usr/bin/env bash
# Qwen P2 GRID, step 3c: GRID mass tables over the held-out 72. CSV-only, and it reuses the
# direction line's activations.
#
# The direction twin (../../../2_dataset_creation/1_p2_selection/heldout_sample.sh) wrote
# qwen_p2_heldout: a layer-27 .pt for EVERY reasoning token of the 72, plus DIRECTION mass
# tables. The tensors do not depend on the vocabulary, so they are reused as they are.
# Only the loudness covariate does. A mass table is baked against one vocabulary at gather
# time, so the grid one needs this CSV-only pass into its own lens directory.
#
# --no-save-activations, and a SEPARATE OUT. Pointing this at qwen_p2_heldout would mix
# grid and direction mass tables in one tree under the same filenames. The per-token scorer
# takes the two apart anyway: --activations-dir qwen_p2_heldout for the tensors and
# --lens-dir for this tree's tables.
#
# NO SAMPLING. SAMPLE_PERCENT=1.0, as in the direction twin: the held-out analysis reads
# every token, and a token without a mass cell drops out of every loudness bin.
#
# No --signal-json, so no selection: the held-out set is scored on tokens no lens chose.
#
# LAYER must equal the grid probes' layer. It is 27 because qwen_p2_heldout only holds 27.
# See p2_training_selection.sh.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)   # so uv finds pyproject.toml

MODEL=Qwen/Qwen3.6-35B-A3B
DATASET=/workspace/trajectories/qwen3.6-35b/replayed_single_step/heldout_72
JLENS_DIR=/workspace/jlens/qwen3_6_35b
SIGNAL_JSON=/workspace/repo/interp/data/jlens/qwen/grid_tokens_pruned_qwen3-6-35b-a3b.json
SIGNAL_NAME=grid
OUT=/workspace/activations/qwen_p2_heldout_grid_lens   # CSVs + grid mass tables only; the .pt stay in qwen_p2_heldout

LAYER=27               # the probes' layer; keep equal to the two selecting runs
SAMPLE_PERCENT=1.0     # NO THINNING: every reasoning token gets a grid mass cell
SAMPLE_SEED=42

LENS=both              # both lenses' CSVs and mass tables, from one forward pass
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
    --direction-mass-json "$SIGNAL_JSON" \
    --signal-name "$SIGNAL_NAME" \
    --data_sample_p "$SAMPLE_PERCENT" \
    --data-sample-seed "$SAMPLE_SEED" \
    --lens "$LENS" \
    --layers "$LAYERS" \
    --no-save-activations \
    --batch-size "$BATCH_SIZE" \
    --forward-batch-size "$FORWARD_BATCH_SIZE" \
    --device cuda
