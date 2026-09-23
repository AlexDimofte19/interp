#!/usr/bin/env bash
# gpt-oss P2 GRID, loudness evaluation step 1: heldout360 as a grid_tile dataset at LAYER 14.
# Every reasoning token, no selection.
#
# The gpt-oss twin of wrappers/qwen_analysis/grid/4_loudness_evaluation/prepare_heldout_grid.sh.
# The source is heldout360_l14_grid (../2_dataset_creation/1_p2_selection/heldout_sample.sh):
# a layer-14 .pt for every reasoning token of the 360. The layer-15 dataset
# grid_binary_l15_heldout360 is NOT reused, because the probes are trained at 14.
#
# --token-selection all --token-major: the held-out gather has no selection record, so every
# gathered token is one entry.
#
# SAME CELLS AS THE ARMS: MAX_CELLS, SEED and PAD match ../2_dataset_creation/2_preparations/,
# so a held-out (trajectory, step) is scored on the same cells whichever arm's probe reads it.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)   # repo root

ACT=/workspace/activations/heldout360_l14_grid
TRAJ=/workspace/trajectories/heldout360
OUT=/workspace/prepared/gptoss_p2_grid_heldout

LAYER=14
MAX_CELLS=25
SEED=42
PAD=15                 # PINNED, never auto: auto pads to the widest size PRESENT

if [ -f "$OUT/manifest.json" ]; then
    echo "exists: $OUT/manifest.json -- delete it to rebuild"
    exit 0
fi

cd "$REPO"
uv run interp-cli prepare_activations_for_probing \
    --activations-dir "$ACT" \
    --trajectories-dir "$TRAJ" \
    --probe-type grid_tile \
    --layers "$LAYER" \
    --steps all \
    --output-indices all \
    --token-selection all \
    --token-major \
    --seed "$SEED" \
    --max-positions-per-trajectory "$MAX_CELLS" \
    --pad-to-size "$PAD" \
    --output-path "$OUT" \
    --verbose
