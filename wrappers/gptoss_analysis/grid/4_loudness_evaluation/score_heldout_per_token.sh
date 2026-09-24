#!/usr/bin/env bash
# gpt-oss P2 GRID, loudness evaluation step 3: every grid probe on EVERY held-out token, with
# the grid loudness beside each verdict -- the per_token_scores.csv the two decile notebooks
# beside this file (and join_opposite_signal.sh) read.
#
# These two runs used to exist only as the SCORED_WITH text in those notebooks; this is that
# text as a script, at GRID_LAYER (../grid_layer.sh). Cells as prepare's: pad 15, 25 per step,
# seed 42, so the pooled numbers reproduce eval_{binary,multiclass}_probes_heldout.sh.
# Resumable: an existing table is kept.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../grid_layer.sh"   # GRID_LAYER

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
PROBES=/workspace/probes/gptoss_p2_grid
ACT=/workspace/activations/heldout360_l${GRID_LAYER}_grid
TRAJ=/workspace/trajectories/heldout360
SIGNAL_JSON=/workspace/repo/interp/data/jlens/grid_tokens_pruned.json
R=/workspace/results/gptoss_p2_grid

cd "$REPO"
score() {  # probe_type out_dir probe_glob
    local out="$R/$2/per_token_scores.csv"
    [ -s "$out" ] && { echo "exists: $out"; return; }
    mkdir -p "$R/$2"
    uv run --extra gpu python telos_interp/loudness_analysis/score_probes_per_token.py \
        --probe-type "$1" $(for p in $3; do printf -- '--probe %s ' "$p"; done) \
        --activations-dir "$ACT" --trajectories-dir "$TRAJ" \
        --signal-json "$SIGNAL_JSON" --signal-name grid \
        --layer "$GRID_LAYER" --pad-to-size 15 --max-cells 25 --seed 42 \
        --cache-activations --read-threads 16 --device cuda --out "$out"
}
score grid_binary heldout "$(ls $PROBES/gptoss_p2_grid_*_{empty,wall,agent,goal}_l${GRID_LAYER}_{lr,mlp}.pt)"
score grid_multiclass heldout_multiclass "$(ls $PROBES/gptoss_p2_grid_*_multiclass_l${GRID_LAYER}_{lr,mlp}.pt)"
