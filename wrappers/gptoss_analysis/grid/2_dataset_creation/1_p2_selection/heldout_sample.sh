#!/usr/bin/env bash
# gpt-oss P2 GRID, step 3c: the held-out gather at LAYER 14 -- every reasoning token, no
# selection, no sampling. Activations AND grid mass tables, from one forward pass.
#
# The gpt-oss grid twin of wrappers/qwen_analysis/2_dataset_creation/1_p2_selection/heldout_sample.sh.
#
# WHY A GATHER AND NOT A REUSE. heldout360_l15 holds a .pt for every held-out token, but at
# layer 15. The grid probes are trained at 14 (the grid profile's J-lens argmax; see
# p2_training_selection.sh), and a layer-14 probe scored on layer-15 activations measures
# nothing. heldout360_lens_grid is grid, but the FULL 1258-token vocabulary. So both halves
# are gathered here, into one tree, the way Qwen's held-out gather did it.
#
# WHAT MAKES IT UNFILTERED: no --signal-json. Without it there is no selection record, and
# every reasoning token gets a .pt. --direction-mass-json is what writes the grid mass tables,
# the loudness covariate the held-out notebook bins on.
#
# NO SAMPLING: SAMPLE_PERCENT=1.0. 87,221 tokens over 360 trajectories, so every one is kept.
#
# A NEW TREE, never an existing one: pointing OUT at heldout360_l15 or heldout360_lens_grid
# would put a second layer or a second vocabulary's tables beside the first under the same
# filenames.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../grid_layer.sh"   # GRID_LAYER

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)   # so uv finds pyproject.toml

MODEL=openai/gpt-oss-20b
DATASET=/workspace/trajectories/heldout360
JLENS_DIR=/workspace/jlens/gridenv   # the GRID-ENVIRONMENT lens; /workspace/jlens holds the wikitext fit (self-check 2026-09-24)
SIGNAL_JSON=/workspace/repo/interp/data/jlens/grid_tokens_pruned.json
SIGNAL_NAME=grid
OUT=/workspace/activations/heldout360_l${GRID_LAYER}_grid   # .pt at L14 + both lenses' grid mass tables at L14

LAYER=$GRID_LAYER               # the probes' layer; keep equal to the two selecting runs
SAMPLE_PERCENT=1.0     # NO THINNING: every reasoning token gets a .pt and a mass cell
SAMPLE_SEED=42

LENS=both              # both lenses' CSVs and mass tables, from one forward pass
LAYERS=$LAYER          # CSV, mass table and .pt all at the one layer
BATCH_SIZE=256

# Single GPU: device_map="auto" across several produces NaNs for this MoE.
export CUDA_VISIBLE_DEVICES=0

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
    --batch-size "$BATCH_SIZE" \
    --device cuda
