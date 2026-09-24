#!/usr/bin/env bash
# Qwen P2 GRID: every MULTICLASS grid probe on every reasoning token of the held-out 72.
#
# The per-token table ./probe_accuracy_by_loudness_decile_multiclass_fixnorm.ipynb reads: one row
# per (trajectory, step, token), with per-class correct counts for each of the 6 probes
# ({jlens, logitlens, random} x {lr, mlp}) and both lenses' grid loudness columns at LAYER.
# It is the command recorded in the notebook's SCORED_WITH, with one change: --lens-dir points at
# the CORRECTED Qwen lens tables (final norm `1 + w`, wrappers/qwen_analysis/norm_fix/), not at
# qwen_p2_heldout_grid_lens. The probe verdicts do not depend on the lens; only the loudness does.
# That is why the output folder is heldout_multiclass_fixnorm while the per-probe JSONs the
# notebook checks against stay in heldout_multiclass (./eval_multiclass_probes_heldout.sh).
#
# --pad-to-size / --max-cells / --seed must match prepare_heldout_grid.sh, so the cells drawn per
# step are the ones eval_grid_probe.py scored and the notebook's cross-check can reproduce its JSON.
# --cache-dir reuses the packed layer-27 activations the direction scoring already built (70
# files instead of 1.17M single-token reads on MooseFS).
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)   # repo root

PROBES=/workspace/probes/qwen_p2_grid
ACT=/workspace/activations/qwen_p2_heldout
LENS=/workspace/activations/qwen_fixnorm/heldout70_grid     # corrected-norm grid lens tables
TRAJ=/workspace/trajectories/qwen3.6-35b/replayed_single_step/heldout_72
OUT=/workspace/results/qwen_p2_grid/heldout_multiclass_fixnorm/per_token_scores.csv
CACHE=/workspace/results/qwen_p2_local_belief/heldout/_act_cache
SIGNAL_JSON=$REPO/data/jlens/qwen/grid_tokens_pruned_qwen3-6-35b-a3b.json
LAYER=27
DEVICE=cuda
READ_THREADS=16

probes=()
for arm in jlens logitlens random; do
    for mt in lr mlp; do
        p="${PROBES}/qwen_p2_grid_${arm}_multiclass_l${LAYER}_${mt}.pt"
        [ -f "$p" ] || { echo "!! missing probe: $p" >&2; exit 1; }
        probes+=(--probe "$p")
    done
done

mkdir -p "$(dirname "$OUT")"
cd "$REPO"

uv run --extra gpu python telos_interp/loudness_analysis/score_probes_per_token.py \
    --probe-type grid_multiclass \
    "${probes[@]}" \
    --activations-dir "$ACT" \
    --lens-dir "$LENS" \
    --trajectories-dir "$TRAJ" \
    --signal-json "$SIGNAL_JSON" \
    --signal-name grid \
    --layer "$LAYER" \
    --pad-to-size 15 \
    --max-cells 25 \
    --seed 42 \
    --cache-activations \
    --cache-dir "$CACHE" \
    --read-threads "$READ_THREADS" \
    --device "$DEVICE" \
    --out "$OUT"

echo "wrote $OUT"
echo "next: ./probe_accuracy_by_loudness_decile_multiclass_fixnorm.ipynb"
