#!/usr/bin/env bash
# The belief-baseline probes: label and selection separated.
#
# The six probes, from the three rollout arms of scripts/rollout_belief_baseline_arms.sh
# (ICLR log entry 49). Run AFTER those finish and the GPU is free.
#
#   random_belief   random selection  -> belief   tokens: the recorded `random` control arm,
#                                                 already on disk in the jlens tree, so no
#                                                 gather and no new prepare -- the existing
#                                                 final-action manifest is RELABELLED.
#   logitlens_p1    logitlens P1      -> belief   tokens: each sentence's loudest by the
#                                                 logit lens. New positions, so a new L15
#                                                 gather (~45 min).
#   logitlens_p2    logitlens P2      -> belief   tokens: the logitlens global top-20, whose
#                                                 .pt were written by the step-1 gather.
#
# Everything else is the local-belief recipe verbatim (ICLR log entry 45) -- layer 15, next_action, lr+mlp, the SAME
# 2880/720 partition pinned by --eval-names -- so a difference between one of these arms and
# the local-belief P2 is the token selection and nothing else, and a difference from the
# next_action_mass_l15 baseline is the label and nothing else.
#
# NEVER an internal --eval-split here: train_cognitive_map_probe splits over ENTRIES, and one
# entry is one (token, layer) in a token-major manifest, so a row-level split leaks a
# trajectory across train and eval. split_next_action_manifest.py splits over names.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT_ROOT=${OUT_ROOT:-/workspace/reasoning_theatre/rollout_strategies_baselines}
# The belief-baseline data dirs keep their original on-disk names (see RENAMES.md): only the
# scripts were renamed, so the run already on disk stays readable and resumable.
BELIEF_BASELINE_ROOT=${BELIEF_BASELINE_ROOT:-/workspace/reasoning_theatre/entry49_baselines}
BELIEF_BASELINE_PREPARED_PREFIX=${BELIEF_BASELINE_PREPARED_PREFIX:-/workspace/prepared/entry49}
HERE=${HERE:-$BELIEF_BASELINE_ROOT}
LOGS=$HERE/logs
PROBES=${PROBES:-/workspace/probes/local_belief_baselines}
mkdir -p "$LOGS" "$PROBES"

TRAJ=${TRAJ:-/workspace/trajectories/reveng/trajectories_train_single_step}
NAMES_FILE=${NAMES_FILE:-/workspace/reasoning_theatre/rollout_strategies/mass_l15_names.txt}
EVAL_NAMES=${EVAL_NAMES:-/workspace/prepared/next_action_mass_l15_eval_names.txt}
LOGITLENS_ROOT=${LOGITLENS_ROOT:-/workspace/activations/logitlens_mass_l15}
ACT_P1=${ACT_P1:-/workspace/activations/logitlens_argmax_per_sentence_l15}
SEED=${SEED:-42}
EPOCHS=${EPOCHS:-50}

cd "$REPO"
ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
UV="uv run --project $REPO --extra gpu"

relabel() {  # relabel <prepared-in> <rollout-dir> <prepared-out> <tag>
    echo "[$(ts)] === $4: relabel from $(basename "$2") ==="
    $UV python "$REPO/scripts/inference_oss/relabel_manifest_from_rollout.py" \
        "$1" "$2" "$3" --report-csv "$HERE/$4_relabel_report.csv" \
        2>&1 | tee "$LOGS/$4_relabel.txt"
}

split_and_train() {  # split_and_train <tag> <prepared-local>
    local tag=$1 prepared=$2 split="${BELIEF_BASELINE_PREPARED_PREFIX}_${1}_split"
    echo "[$(ts)] === $tag: split (eval pinned to the shared 720) ==="
    $UV python "$REPO/scripts/split_next_action_manifest.py" \
        "$prepared" --eval-names "$EVAL_NAMES" --single-layer 15 --seed "$SEED" \
        --train-out "${split}_train" --eval-out "${split}_eval" \
        2>&1 | tee "$LOGS/${tag}_split.txt"
    for mt in lr mlp; do
        echo "[$(ts)] === $tag: train $mt ==="
        $UV interp-cli train_next_action_probe \
            --train-data-path "${split}_train" --eval-data-path "${split}_eval" \
            --output-path "$PROBES/next_action_probe_${tag}_${mt}.pt" \
            --model-type "$mt" --hidden-dims 1024 --learning-rate 3e-4 --weight-decay 0.001 \
            --dropout 0.0 --num-epochs "$EPOCHS" --batch-size 512 --class-weight balanced \
            --normalize --seed "$SEED" --device cuda --verbose \
            2>&1 | tee "$LOGS/${tag}_${mt}.txt"
    done
}

# ---- arm 1: random selection -> belief -------------------------------------------------
# The existing manifest already points at the control's .pt; only the label changes.
relabel /workspace/prepared/next_action_mass_l15_random \
        "$OUT_ROOT/recorded_selection" \
        "${BELIEF_BASELINE_PREPARED_PREFIX}_random_belief" random_belief
split_and_train random_belief "${BELIEF_BASELINE_PREPARED_PREFIX}_random_belief"

# ---- arm 2: logitlens P1 -> belief ------------------------------------------------------
echo "[$(ts)] === logitlens_p1: gather L15 at the per-sentence-loudest cutoffs ==="
$UV python "$REPO/scripts/inference_oss/gather_local_belief_activations.py" \
    --rollout-dir "$OUT_ROOT/jlens_argmax_per_sentence" \
    --trajectory-paths "$TRAJ" --names-file "$NAMES_FILE" --out "$ACT_P1" \
    2>&1 | tee "$LOGS/logitlens_p1_gather.txt"

echo "[$(ts)] === logitlens_p1: prepare (token-selection all, layer 15) ==="
$UV interp-cli prepare_activations_for_probing \
    --activations-dir "$ACT_P1" --trajectories-dir "$TRAJ" \
    --probe-type next_action --layers 15 --steps all --output-indices all \
    --output-path "${BELIEF_BASELINE_PREPARED_PREFIX}_logitlens_p1_final" --verbose \
    2>&1 | tee "$LOGS/logitlens_p1_prepare.txt"

relabel "${BELIEF_BASELINE_PREPARED_PREFIX}_logitlens_p1_final" \
        "$OUT_ROOT/jlens_argmax_per_sentence" \
        "${BELIEF_BASELINE_PREPARED_PREFIX}_logitlens_p1_local" logitlens_p1
split_and_train logitlens_p1 "${BELIEF_BASELINE_PREPARED_PREFIX}_logitlens_p1_local"

# ---- arm 3: logitlens P2 -> belief ------------------------------------------------------
# The step-1 gather already saved these tokens' layer-15 .pt, so this only reads the record.
echo "[$(ts)] === logitlens_p2: prepare (recorded_logitlens, layer 15) ==="
$UV interp-cli prepare_activations_for_probing \
    --activations-dir "$LOGITLENS_ROOT" --trajectories-dir "$TRAJ" \
    --probe-type next_action --layers 15 --steps all --output-indices all \
    --token-selection recorded_logitlens \
    --output-path "${BELIEF_BASELINE_PREPARED_PREFIX}_logitlens_p2_final" --verbose \
    2>&1 | tee "$LOGS/logitlens_p2_prepare.txt"

relabel "${BELIEF_BASELINE_PREPARED_PREFIX}_logitlens_p2_final" \
        "$OUT_ROOT/jlens_top_k_global" \
        "${BELIEF_BASELINE_PREPARED_PREFIX}_logitlens_p2_local" logitlens_p2
split_and_train logitlens_p2 "${BELIEF_BASELINE_PREPARED_PREFIX}_logitlens_p2_local"

echo "[$(ts)] === ALL DONE ==="
for t in random_belief logitlens_p1 logitlens_p2; do for m in lr mlp; do
    printf '%-16s %-4s ' "$t" "$m"
    grep -m1 'Best balanced accuracy' "$LOGS/${t}_${m}.txt" 2>/dev/null || echo '?'
done; done
