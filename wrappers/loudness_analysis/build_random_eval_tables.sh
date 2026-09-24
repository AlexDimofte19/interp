#!/usr/bin/env bash
# Per-token probe tables on each line's RANDOM p2 eval slice (the uniformly drawn eval tokens),
# for probe_performance_by_loudness.ipynb with EVAL = "random_eval".
#
# Per (model, signal), three stages:
#   1. each of the 6 probes on the slice, --per-token-out (count form; direction probes scored
#      against the LOCAL belief, the label they were trained on). These are the cells of the
#      random_p2 column of the cross-selection tables, row by row.
#   2. build_slice_per_token_table.py: probes side by side, abs_pos and token from the
#      trajectory, signal-word flags over the whole reasoning chain (radius 2).
#   3. join_signal_loudness.py: own and opposite signal, both lenses. Each ruler at its own
#      line's layer (gpt-oss direction L15, gpt-oss grid L14, Qwen L27), as the held-out run.
#      Qwen's lens trees are the NORM-FIXED ones (qwen_fixnorm/eval52_*): same seeded 20%
#      sample the selection was drawn from, and every slice token is in it.
#
# Every stage writes its own folder, so no join overwrites another's run_config.json. The
# notebook reads stages/4_*/per_token_scores.csv. Resumable: a finished file is skipped.
#
# Usage: bash wrappers/loudness_analysis/build_random_eval_tables.sh    (ONLY="gptoss_grid qwen_grid" to subset)
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
T=/workspace/loudness_probe_performance_analysis/tables/random_eval
A=/workspace/activations
P=/workspace/prepared
J=$REPO/data/jlens
GPTOSS_TRAJ=$A/mass_eval720_view/trajectories
QWEN_TRAJ=/workspace/trajectories/qwen3.6-35b/replayed_single_step/mass_eval_144
ARMS="jlens logitlens random"
MODEL_TYPES="lr mlp"
ONLY=${ONLY:-"gptoss_direction gptoss_grid qwen_direction qwen_grid"}

cd "$REPO"
PY="uv run python"

evaluate() {  # run kind slice probe_glob_prefix probe_suffix signal_json
    local run=$1 kind=$2 slice=$3 prefix=$4 suffix=$5 sig=$6 pids="" failed=0
    mkdir -p "$T/$run/stages/1_probe_counts"
    for arm in $ARMS; do
        for mt in $MODEL_TYPES; do
            local probe="${prefix}_${arm}_${suffix}_${mt}.pt" out
            out="$T/$run/stages/1_probe_counts/$(basename "$probe" .pt).csv"
            [ -s "$out" ] && { echo "exists: $out"; continue; }
            if [ "$kind" = grid ]; then
                $PY telos_interp/loudness_analysis/eval_grid_probe.py "$probe" "$slice" \
                    --signal-json "$sig" --signal-name grid --cache-activations --per-token-out "$out" \
                    > "${out%.csv}.txt" 2>&1 &
            else
                $PY telos_interp/loudness_analysis/rollouts/eval_local_belief.py "$probe" "$slice" \
                    --signal-json "$sig" --signal-name direction --per-token-out "$out" \
                    > "${out%.csv}.txt" 2>&1 &
            fi
            pids="$pids $!"
        done
    done
    for pid in $pids; do wait "$pid" || failed=1; done
    [ "$failed" -eq 0 ] || { echo "!! an evaluation failed under $T/$run/stages/1_probe_counts" >&2; exit 1; }
}

merge() {  # run traj_dir direction_json grid_json
    local out="$T/$1/stages/2_merged/per_token_scores.csv"
    [ -s "$out" ] && { echo "exists: $out"; return; }
    $PY -m telos_interp.loudness_analysis.build_slice_per_token_table \
        $(for f in "$T/$1"/stages/1_probe_counts/*.csv; do printf -- '--probe-counts %s ' "$f"; done) \
        --trajectories-dir "$2" --signal "direction=$3" --signal "grid=$4" --exclude-radius 2 --out "$out"
}

join() {  # run from_stage to_stage lens_root lenses signal_name signal_json layer
    local src="$T/$1/stages/$2/per_token_scores.csv" out="$T/$1/stages/$3/per_token_scores.csv"
    [ -s "$out" ] && { echo "exists: $out"; return; }
    mkdir -p "$(dirname "$out")"
    $PY -m telos_interp.loudness_analysis.join_signal_loudness --table "$src" --lens-root "$4" \
        --lenses "$5" --signal-name "$6" --signal-json "$7" --layer "$8" --out "$out"
}

QDIR=direction_tokens_full_qwen3-6-35b-a3b
for run in $ONLY; do
    echo "=== $run ==="
    case $run in
    gptoss_direction)
        evaluate $run direction "$P/entry49_random_belief_split_eval" \
            /workspace/probes/gptoss_p2_local_belief/gptoss_p2_local_belief l15 "$J/direction_tokens_full.json"
        merge $run "$GPTOSS_TRAJ" "$J/direction_tokens_full.json" "$J/grid_tokens_pruned.json"
        join $run 2_merged 3a_direction_jlens "$A/jlens_mass_l15" jlens direction "$J/direction_tokens_full.json" 15
        join $run 3a_direction_jlens 3b_direction_logitlens "$A/logitlens_mass_l15" logitlens direction "$J/direction_tokens_full.json" 15
        join $run 3b_direction_logitlens 4_grid "$A/gptoss_grid_mass_l14_eval" jlens,logitlens grid "$J/grid_tokens_pruned.json" 14
        ;;
    gptoss_grid)
        evaluate $run grid "$P/gptoss_p2_grid_random_val" \
            /workspace/probes/gptoss_p2_grid/gptoss_p2_grid multiclass_l14 "$J/grid_tokens_pruned.json"
        merge $run "$GPTOSS_TRAJ" "$J/direction_tokens_full.json" "$J/grid_tokens_pruned.json"
        join $run 2_merged 3a_grid "$A/gptoss_grid_mass_l14_eval" jlens,logitlens grid "$J/grid_tokens_pruned.json" 14
        join $run 3a_grid 3b_direction_jlens "$A/jlens_mass_l15" jlens direction "$J/direction_tokens_full.json" 15
        join $run 3b_direction_jlens 4_direction_logitlens "$A/logitlens_mass_l15" logitlens direction "$J/direction_tokens_full.json" 15
        ;;
    qwen_direction)
        evaluate $run direction "$P/qwen_p2_local_belief_random_val" \
            /workspace/probes/qwen_p2_local_belief/qwen_p2_local_belief l27 "$J/qwen/$QDIR.json"
        merge $run "$QWEN_TRAJ" "$J/qwen/$QDIR.json" "$J/qwen/grid_tokens_pruned_qwen3-6-35b-a3b.json"
        join $run 2_merged 3_direction "$A/qwen_fixnorm/eval52_direction" jlens,logitlens $QDIR "$J/qwen/$QDIR.json" 27
        join $run 3_direction 4_grid "$A/qwen_fixnorm/eval52_grid" jlens,logitlens grid "$J/qwen/grid_tokens_pruned_qwen3-6-35b-a3b.json" 27
        ;;
    qwen_grid)
        evaluate $run grid "$P/qwen_p2_grid_random_val" \
            /workspace/probes/qwen_p2_grid/qwen_p2_grid multiclass_l27 "$J/qwen/grid_tokens_pruned_qwen3-6-35b-a3b.json"
        merge $run "$QWEN_TRAJ" "$J/qwen/$QDIR.json" "$J/qwen/grid_tokens_pruned_qwen3-6-35b-a3b.json"
        join $run 2_merged 3_grid "$A/qwen_fixnorm/eval52_grid" jlens,logitlens grid "$J/qwen/grid_tokens_pruned_qwen3-6-35b-a3b.json" 27
        join $run 3_grid 4_direction "$A/qwen_fixnorm/eval52_direction" jlens,logitlens $QDIR "$J/qwen/$QDIR.json" 27
        ;;
    esac
done
echo "done -> $T/<run>/stages/4_*/per_token_scores.csv"
