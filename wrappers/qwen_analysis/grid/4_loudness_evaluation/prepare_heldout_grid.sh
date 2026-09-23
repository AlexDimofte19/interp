#!/usr/bin/env bash
# Qwen P2 GRID, loudness evaluation step 1: the held-out 72 as a grid_tile dataset. Every
# reasoning token, no selection.
#
# The binary probes are scored here on tokens no lens chose. That is the population that
# says whether the jlens arm's probe generalises, and the one where the jlens probe fell
# below the control on the direction line (ICLR entry 49(c)).
#
# REUSES THE DIRECTION LINE'S TENSORS. qwen_p2_heldout holds a layer-27 .pt for every
# reasoning token of the 72, and a tensor does not depend on the vocabulary. Only the
# loudness does, and that is ../2_dataset_creation/1_p2_selection/heldout_sample.sh's
# separate grid lens tree. Nothing is gathered for this step.
#
# --token-selection all --token-major: qwen_p2_heldout was gathered without --signal-json,
# so it has no selection record and nothing to replay. Every gathered token is one entry.
#
# SAME CELLS AS THE ARMS: MAX_CELLS=25 and SEED=42, drawn per (trajectory, step). This
# dataset is padded to the widest grid, exactly like the arm datasets. See the padding note
# in ../2_dataset_creation/2_preparations/prepare_jlens.sh.
#
# COST. 1,169,734 tokens x 25 cells is ~29M (token, cell) rows, and the first pass opens
# 1.17M single-token .pt files over MooseFS (~9 h at the 37 files/s the gather managed).
# The manifest copies nothing, but it is large.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)   # repo root

ACT=/workspace/activations/qwen_p2_heldout
TRAJ=/workspace/trajectories/qwen3.6-35b/replayed_single_step/heldout_72
OUT=/workspace/prepared/qwen_p2_grid_heldout

LAYER=27
MAX_CELLS=25
SEED=42
PAD=15                 # PINNED, never auto: auto pads to the widest size PRESENT, and the Qwen eval
                       # set has no size-15 grid, so it came out padded to 13 against train's 15

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
