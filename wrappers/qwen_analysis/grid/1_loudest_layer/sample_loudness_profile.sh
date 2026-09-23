#!/usr/bin/env bash
# Qwen P2 GRID, step 1: the all-layers GRID loudness profile, on a 5% sample of the reasoning tokens.
#
# The grid twin of ../../1_loudest_layer/sample_loudness_profile.sh. Same run and seeds, but a
# 5% sample instead of 20% (a uniform draw keeps a per-layer mean unbiased and only widens
# its error bars, and the profile only has to name one argmax layer). Two more things differ,
# and they are the whole point:
#
#   * the vocabulary is the PRUNED Qwen grid vocabulary (533 tokens, WALL/OPEN/GOAL/AGENT/
#     AXIS/STATUS), not the direction one. data/jlens/README.md records why the pruned file
#     and not the full 1316: the full one is dominated by camelCase identifiers and drift.
#   * the output tree is its own. A mass table is baked against one vocabulary at gather time,
#     so the direction profile's tables cannot be re-scored against grid words.
#
# --signal-name grid IS REQUIRED, not decoration. Without it the signal name is inferred from
# the filename, and grid_tokens_pruned_qwen3-6-35b-a3b.json would name every loudness column
# downstream after "grid_tokens_pruned_qwen3-6-35b-a3b" instead of "grid".
#
# CSV-ONLY. No .pt is written and no selection is made -- this run answers "which layer is the
# jlens most GRID-loaded at", via join_loudness_profile.sh and the notebook beside it. The
# answer is what p2_training_selection.sh pins with --select-candidate-layers. Read the
# AXIS caveat in data/jlens/README.md before reading that answer: most grid mass is
# row/column words, so "grid-loud" is largely "coordinate-loud".
#
# WHY --direction-mass-json AND NOT --signal-json. --signal-json switches the script into
# selective mode. Passing the vocabulary as --direction-mass-json gets the mass table and
# nothing else, which is all this run wants.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)   # so uv finds pyproject.toml

MODEL=Qwen/Qwen3.6-35B-A3B
DATASET=/workspace/trajectories/qwen3.6-35b/replayed_single_step/mass_train_576
JLENS_DIR=/workspace/jlens/qwen3_6_35b
SIGNAL_JSON=/workspace/repo/interp/data/jlens/qwen/grid_tokens_pruned_qwen3-6-35b-a3b.json
SIGNAL_NAME=grid
OUT=/workspace/activations/qwen/qwen_grid_loudness_profile_p05

SAMPLE_PERCENT=0.05     # 5%: enough rows for a per-layer mean; see the header
SAMPLE_SEED=42
LENS=both              # jlens + logitlens from ONE forward pass: one extra unembed, not a second pass
LAYERS=all             # what the CSV and the mass tables cover; jlens is restricted to its fitted layers anyway
BATCH_SIZE=256         # reasoning tokens per lens matmul, caps the [B, vocab] logits
FORWARD_BATCH_SIZE=1   # one 18k-token chain at a time; the default of 4 pads four of them into one pass

# Single GPU: device_map="auto" across several produces NaNs for this MoE, as it does for gpt-oss.
export CUDA_VISIBLE_DEVICES=0

# 66 GiB of weights on an 80 GiB card leaves ~13 GiB for a 33k-token chain; expandable
# segments stop the allocator stranding reserved-but-unallocated blocks.
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
