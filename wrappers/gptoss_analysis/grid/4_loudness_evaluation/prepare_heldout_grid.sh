#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
ACT=/workspace/activations/heldout360_l14_grid
TRAJ=/workspace/trajectories/heldout360
OUT=/workspace/prepared/gptoss_p2_grid_heldout
LAYER=14
MAX_CELLS=25
SEED=42
PAD=15
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
