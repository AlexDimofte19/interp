#!/usr/bin/env bash
# Add the OPPOSITE signal's loudness to each held-out per-token probe table, for
# probe_performance_by_loudness.ipynb beside this file (plots 5-6).
#
# CPU only: join_signal_loudness.py on (name, step, abs_pos), probe columns copied verbatim.
# Each opposite ruler is read at ITS OWN line's layer, not the probe's: gpt-oss direction
# probes (L15) get grid loudness at L14, gpt-oss grid probes (L14) get direction loudness at
# L15. Qwen is L27 on both sides. No tree holds the other layer, and re-gathering was judged
# not worth it.
#
# Outputs go under $OUT/tables/, never beside the source tables, so the join's
# run_config.json cannot overwrite the scorer's.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../gptoss_analysis/grid/grid_layer.sh"   # GPT-oss grid GRID_LAYER

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUT=/workspace/loudness_probe_performance_analysis/tables
R=/workspace/results
A=/workspace/activations
J=$REPO/data/jlens

cd "$REPO"
join() {  # name table lens_root signal_name signal_json layer
    local out="$OUT/$1/per_token_scores.csv"
    if [ -s "$out" ]; then echo "exists: $out"; return; fi
    mkdir -p "$(dirname "$out")"
    uv run python -m telos_interp.loudness_analysis.join_signal_loudness \
        --table "$2" --lens-root "$3" --signal-name "$4" --signal-json "$5" --layer "$6" --out "$out"
}

join gptoss_direction "$R/gptoss_p2_local_belief/heldout/per_token_scores.csv" \
    "$A/heldout360_l${GRID_LAYER}_grid" grid "$J/grid_tokens_pruned.json" "$GRID_LAYER"
join gptoss_grid "$R/gptoss_p2_grid/heldout_multiclass/per_token_scores.csv" \
    "$A/heldout360_lens" direction "$J/direction_tokens_full.json" 15
join qwen_direction "$R/qwen_p2_local_belief/heldout_fixnorm/per_token_scores.csv" \
    "$A/qwen_fixnorm/heldout70_grid" grid "$J/qwen/grid_tokens_pruned_qwen3-6-35b-a3b.json" 27
# The Qwen direction trees' sidecars name the vocabulary by its file stem, so the columns
# come out as {lens}_direction_tokens_full_qwen3-6-35b-a3b_logmass_L27; the notebook maps it.
join qwen_grid "$R/qwen_p2_grid/heldout_multiclass_fixnorm/per_token_scores.csv" \
    "$A/qwen_fixnorm/heldout70_direction" direction_tokens_full_qwen3-6-35b-a3b \
    "$J/qwen/direction_tokens_full_qwen3-6-35b-a3b.json" 27
