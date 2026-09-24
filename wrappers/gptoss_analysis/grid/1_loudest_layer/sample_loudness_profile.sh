#!/usr/bin/env bash
# gpt-oss P2 GRID, step 1: the all-layers GRID loudness profile, over the WHOLE chain.
#
# The gpt-oss grid twin of wrappers/qwen_analysis/1_loudest_layer/sample_loudness_profile.sh.
# Unlike the direction profile, this one cannot be reused. The existing trees' mass tables are
# baked against the direction vocabulary. heldout360_lens_grid is grid, but the FULL 1258-token
# vocabulary, on the held-out set. No tree has a mass table against the pruned 642. A mass
# table is fixed at gather time, so this is a CSV-only pass.
#
# NO SAMPLING: SAMPLE_PERCENT=1.0, as for every gpt-oss tree this line reads. gpt-oss chains
# are ~10x shorter than Qwen's, so the lens and unembed terms that forced Qwen down to 20%
# are affordable in full.
#
# THE TRAIN 2,880 ONLY, via the mass-era view's trajectory folder, so the layer is not chosen
# on the 720 it will be scored on.
#
# --signal-name grid IS REQUIRED: from the filename alone the signal would be named
# "grid_tokens_pruned", and so would every loudness column downstream.
#
# CSV-ONLY: --direction-mass-json, not --signal-json, so no selection and no .pt. On gpt-oss
# the probe layer is 15 by convention; read the notebook's argmax as a description of where 15
# sits, and see the AXIS caveat in data/jlens/README.md (most grid mass is row/column words).
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)   # so uv finds pyproject.toml

MODEL=openai/gpt-oss-20b
DATASET=/workspace/activations/mass_train2880_view/trajectories
JLENS_DIR=/workspace/jlens/gridenv   # the GRID-ENVIRONMENT lens; /workspace/jlens holds the wikitext fit (self-check 2026-09-24)
SIGNAL_JSON=/workspace/repo/interp/data/jlens/grid_tokens_pruned.json
SIGNAL_NAME=grid
OUT=/workspace/activations/gptoss_grid_loudness_profile

SAMPLE_PERCENT=1.0     # NO THINNING: the whole chain
SAMPLE_SEED=42
LENS=both              # jlens + logitlens from ONE forward pass
LAYERS=7:23            # every layer the gpt-oss jlens is fitted at

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
    --no-save-activations \
    --device cuda
