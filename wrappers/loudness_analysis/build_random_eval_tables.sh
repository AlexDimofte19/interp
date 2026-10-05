#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
T=/workspace/loudness_probe_performance_analysis/tables/random_eval
A=/workspace/activations
P=/workspace/prepared
PR=/workspace/probes
J=$REPO/data/jlens
GPTOSS_TRAJ=$A/mass_eval720_view/trajectories
QWEN_TRAJ=/workspace/trajectories/qwen3.6-35b/replayed_single_step/mass_eval_144
ARMS="jlens logitlens random"
MODEL_TYPES="lr mlp"
cd "$REPO"

evaluate() {
    local run=$1 kind=$2 slice=$3 prefix=$4 suffix=$5 sig=$6 pids=""
    mkdir -p "$T/$run/stages/1_probe_counts"
    for arm in $ARMS; do
        for mt in $MODEL_TYPES; do
            local probe="${prefix}_${arm}_${suffix}_${mt}.pt"
            local out="$T/$run/stages/1_probe_counts/$(basename "$probe" .pt).csv"
            if [ "$kind" = grid ]; then
                uv run python telos_interp/loudness_analysis/eval_grid_probe.py "$probe" "$slice" \
                    --signal-json "$sig" --signal-name grid --cache-activations --per-token-out "$out" \
                    > "${out%.csv}.txt" 2>&1 &
            else
                uv run python telos_interp/loudness_analysis/rollouts/eval_local_belief.py "$probe" "$slice" \
                    --signal-json "$sig" --signal-name direction --per-token-out "$out" \
                    > "${out%.csv}.txt" 2>&1 &
            fi
            pids="$pids $!"
        done
    done
    for pid in $pids; do
        wait "$pid"
    done
}

merge() {
    local run=$1 traj=$2 direction=$3 grid=$4
    mkdir -p "$T/$run/stages/2_merged"
    uv run python -m telos_interp.loudness_analysis.build_slice_per_token_table \
        $(for f in "$T/$run"/stages/1_probe_counts/*.csv; do printf -- '--probe-counts %s ' "$f"; done) \
        --trajectories-dir "$traj" \
        --signal "direction=$direction" \
        --signal "grid=$grid" \
        --exclude-radius 2 \
        --out "$T/$run/stages/2_merged/per_token_scores.csv"
}

join() {
    local run=$1 from=$2 to=$3 tree=$4 name=$5 sig=$6 layer=$7
    mkdir -p "$T/$run/stages/$to"
    uv run python -m telos_interp.loudness_analysis.join_signal_loudness \
        --table "$T/$run/stages/$from/per_token_scores.csv" \
        --lens-root "$tree" \
        --lenses jlens,logitlens \
        --signal-name "$name" \
        --signal-json "$sig" \
        --layer "$layer" \
        --out "$T/$run/stages/$to/per_token_scores.csv"
}

evaluate gptoss_direction direction "$P/gptoss_p2_local_belief_random_val" \
    "$PR/gptoss_p2_local_belief/gptoss_p2_local_belief" l15 "$J/direction_tokens_full.json"
merge gptoss_direction "$GPTOSS_TRAJ" "$J/direction_tokens_full.json" "$J/grid_tokens_pruned.json"
join gptoss_direction 2_merged 3_direction "$A/gptoss_direction_mass_l15_eval" direction "$J/direction_tokens_full.json" 15
join gptoss_direction 3_direction 4_grid "$A/gptoss_grid_mass_l14_eval" grid "$J/grid_tokens_pruned.json" 14

evaluate gptoss_grid grid "$P/gptoss_p2_grid_random_val" \
    "$PR/gptoss_p2_grid/gptoss_p2_grid" multiclass_l14 "$J/grid_tokens_pruned.json"
merge gptoss_grid "$GPTOSS_TRAJ" "$J/direction_tokens_full.json" "$J/grid_tokens_pruned.json"
join gptoss_grid 2_merged 3_grid "$A/gptoss_grid_mass_l14_eval" grid "$J/grid_tokens_pruned.json" 14
join gptoss_grid 3_grid 4_direction "$A/gptoss_direction_mass_l15_eval" direction "$J/direction_tokens_full.json" 15

evaluate qwen_direction direction "$P/qwen_p2_local_belief_random_val" \
    "$PR/qwen_p2_local_belief/qwen_p2_local_belief" l27 "$J/qwen/direction_tokens_full_qwen3-6-35b-a3b.json"
merge qwen_direction "$QWEN_TRAJ" "$J/qwen/direction_tokens_full_qwen3-6-35b-a3b.json" "$J/qwen/grid_tokens_pruned_qwen3-6-35b-a3b.json"
join qwen_direction 2_merged 3_direction "$A/qwen_p2_selection_eval" direction "$J/qwen/direction_tokens_full_qwen3-6-35b-a3b.json" 27
join qwen_direction 3_direction 4_grid "$A/qwen_p2_grid_selection_eval" grid "$J/qwen/grid_tokens_pruned_qwen3-6-35b-a3b.json" 27

evaluate qwen_grid grid "$P/qwen_p2_grid_random_val" \
    "$PR/qwen_p2_grid/qwen_p2_grid" multiclass_l27 "$J/qwen/grid_tokens_pruned_qwen3-6-35b-a3b.json"
merge qwen_grid "$QWEN_TRAJ" "$J/qwen/direction_tokens_full_qwen3-6-35b-a3b.json" "$J/qwen/grid_tokens_pruned_qwen3-6-35b-a3b.json"
join qwen_grid 2_merged 3_grid "$A/qwen_p2_grid_selection_eval" grid "$J/qwen/grid_tokens_pruned_qwen3-6-35b-a3b.json" 27
join qwen_grid 3_grid 4_direction "$A/qwen_p2_selection_eval" direction "$J/qwen/direction_tokens_full_qwen3-6-35b-a3b.json" 27
