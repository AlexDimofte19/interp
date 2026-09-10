#!/usr/bin/env bash
# Four more belief arms: the per-sentence cadence, completed.
#
# The eight probes from the two rollout arms of wrappers/rollout_more_belief_arms.sh, plus one
# thinning that needed no rollout at all. Run AFTER those finish and the GPU is free.
#
#   eos_belief                    last token of each sentence   uncapped, ~71.4k
#   random_sentence_belief        random token of each sentence uncapped, ~75k
#   random_sentence_belief_top20  ^ thinned to 20/trajectory    ~41.6k
#   logitlens_p1_top20            logitlens loudest per sentence, thinned to 20/trajectory
#
# Read the round as cadence x which-token-inside-the-span. The uncapped row was
# jlens (local_belief_p1) and logitlens (entry 49's logitlens_p1) only; eos_belief and
# random_sentence_belief complete it. The thinned row was jlens (local_belief_p1_top20) only;
# the other two complete that. Every arm is layer 15, next_action, lr+mlp, seed 42, and the
# SAME 2880/720 partition pinned by --eval-names, so a difference between any two of them is
# the token selection and nothing else.
#
# TWO ARMS NEED NO GATHER, FOR DIFFERENT REASONS.
#
#   eos_belief          the activations already exist. activations_train_single_step_reasoning_eos
#                       holds layer-15 tensors at every sentence end of all 36,000 trajectories
#                       -- a superset of the mass-era 3,600 -- and only the eos_lens3600_view
#                       symlink view was ever restricted to the count-era names, which is what
#                       made the tree look unusable. Verified against a current-code rollout:
#                       of the sentence_end + end_of_reasoning positions it emits, 0 are
#                       missing from the tree. So this arm needs a VIEW, not a GPU pass.
#   logitlens_p1_top20  a thinning of an existing prepared dataset, so neither rollout nor
#                       gather nor prepare: one split, then train.
#
# WHY eos_belief PASSES --keep-kinds. Every other arm's tree was written by
# gather_local_belief_activations.py, which saves exactly the positions the arm wants. This
# one reuses a tree built for a different purpose, and that tree carries EVERY sentence end
# including the chain's last -- whose cutoff is end_of_reasoning, and whose local-belief label
# is the final action at p~1.0 by construction. Keeping those would add one trivially
# decodable row per trajectory (~75.0k instead of ~71.4k) and break comparability with the
# interior-only arms, so the relabel drops them.
#
# NEVER an internal --eval-split here: train_cognitive_map_probe splits over ENTRIES, and one
# entry is one (token, layer) in a token-major manifest, so a row-level split leaks a
# trajectory across train and eval. split_next_action_manifest.py splits over names.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT_ROOT=${OUT_ROOT:-/workspace/reasoning_theatre/rollout_strategies_baselines}
HERE=${HERE:-/workspace/reasoning_theatre/more_belief_arms}
PREPARED_PREFIX=${PREPARED_PREFIX:-/workspace/prepared/more_belief}
LOGS=$HERE/logs
PROBES=${PROBES:-/workspace/probes/local_belief_baselines}
mkdir -p "$LOGS" "$PROBES"

TRAJ=${TRAJ:-/workspace/trajectories/reveng/trajectories_train_single_step}
NAMES_FILE=${NAMES_FILE:-/workspace/reasoning_theatre/rollout_strategies/mass_l15_names.txt}
EVAL_NAMES=${EVAL_NAMES:-/workspace/prepared/next_action_mass_l15_eval_names.txt}
ACT_EOS_VIEW=${ACT_EOS_VIEW:-/workspace/activations/eos_mass3600_view}
ACT_RANDOM=${ACT_RANDOM:-/workspace/activations/random_per_sentence_l15}
LOGITLENS_P1_LOCAL=${LOGITLENS_P1_LOCAL:-/workspace/prepared/entry49_logitlens_p1_local}
SEED=${SEED:-42}
EPOCHS=${EPOCHS:-50}

cd "$REPO"
ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
UV="uv run --project $REPO --extra gpu"

WANT=${1:-all}
want() { [ "$WANT" = all ] || [ "$WANT" = "$1" ]; }

relabel() {  # relabel <prepared-in> <rollout-dir> <prepared-out> <tag> [extra flags...]
    local pin=$1 rollout=$2 pout=$3 tag=$4; shift 4
    echo "[$(ts)] === $tag: relabel from $(basename "$rollout") ==="
    $UV python "$REPO/scripts/inference_oss/relabel_manifest_from_rollout.py" \
        "$pin" "$rollout" "$pout" --report-csv "$HERE/${tag}_relabel_report.csv" "$@" \
        2>&1 | tee "$LOGS/${tag}_relabel.txt"
}

split_and_train() {  # split_and_train <tag> <prepared-local> [extra split flags...]
    local tag=$1 prepared=$2 split="${PREPARED_PREFIX}_${1}_split"; shift 2
    echo "[$(ts)] === $tag: split (eval pinned to the shared 720) ==="
    $UV python "$REPO/scripts/split_next_action_manifest.py" \
        "$prepared" --eval-names "$EVAL_NAMES" --single-layer 15 --seed "$SEED" "$@" \
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

# ---- arm 3: logitlens per-sentence, thinned to 20 ---------------------------------------
# No rollout, no gather, no prepare: entry 49's logitlens_p1 dataset is already relabelled to
# the belief, and --tokens-per-trajectory 20 is the identical operation that made
# local_belief_p1_top20 from local_belief_p1. It shares p1's train/eval strata, because
# split_next_action_manifest.py computes them from the UNTHINNED samples.
#
# Expect ~41.6k samples, not 20 x 3600: the per-sentence count is skewed (median 10/traj,
# mean 20.8, max 230) and only 29.6% of trajectories have more than 20 tokens, so the total
# is sum(min(n, 20)).
if want logitlens_p1_top20; then
    split_and_train logitlens_p1_top20 "$LOGITLENS_P1_LOCAL" --tokens-per-trajectory 20
fi

# ---- arm 1: sentence ends -> belief -------------------------------------------------------
if want eos_belief; then
    echo "[$(ts)] === eos_belief: view the existing sentence-end tree at the mass-era names ==="
    bash "$REPO/scripts/build_activation_view.sh" "$NAMES_FILE" "$ACT_EOS_VIEW" \
        2>&1 | tee "$LOGS/eos_belief_view.txt"

    echo "[$(ts)] === eos_belief: prepare (token-selection all, layer 15) ==="
    $UV interp-cli prepare_activations_for_probing \
        --activations-dir "$ACT_EOS_VIEW/activations" --trajectories-dir "$ACT_EOS_VIEW/trajectories" \
        --probe-type next_action --layers 15 --steps all --output-indices all \
        --output-path "${PREPARED_PREFIX}_eos_belief_final" --verbose \
        2>&1 | tee "$LOGS/eos_belief_prepare.txt"

    relabel "${PREPARED_PREFIX}_eos_belief_final" "$OUT_ROOT/eos" \
            "${PREPARED_PREFIX}_eos_belief_local" eos_belief --keep-kinds sentence_end
    split_and_train eos_belief "${PREPARED_PREFIX}_eos_belief_local"
fi

# ---- arms 2a / 2b: a random token per sentence -> belief -----------------------------------
# One rollout, one gather, one prepare, one relabel; the two arms diverge only at the split,
# so 2b costs one extra split and two extra trainings and no GPU beyond them. --out is set
# explicitly because already_done() checks only that a .pt exists, not which cutoff kind wrote
# it, so pointing a second kind at another arm's tree would skip every write silently.
if want random_sentence_belief; then
    echo "[$(ts)] === random_sentence_belief: gather L15 at the random-in-sentence cutoffs ==="
    $UV python "$REPO/scripts/inference_oss/gather_local_belief_activations.py" \
        --rollout-dir "$OUT_ROOT/random_per_sentence" --interior-kinds random_in_sentence \
        --trajectory-paths "$TRAJ" --names-file "$NAMES_FILE" --out "$ACT_RANDOM" \
        2>&1 | tee "$LOGS/random_sentence_belief_gather.txt"

    echo "[$(ts)] === random_sentence_belief: prepare (token-selection all, layer 15) ==="
    $UV interp-cli prepare_activations_for_probing \
        --activations-dir "$ACT_RANDOM" --trajectories-dir "$TRAJ" \
        --probe-type next_action --layers 15 --steps all --output-indices all \
        --output-path "${PREPARED_PREFIX}_random_sentence_belief_final" --verbose \
        2>&1 | tee "$LOGS/random_sentence_belief_prepare.txt"

    relabel "${PREPARED_PREFIX}_random_sentence_belief_final" "$OUT_ROOT/random_per_sentence" \
            "${PREPARED_PREFIX}_random_sentence_belief_local" random_sentence_belief

    # 2a uncapped, then 2b thinned. A control arm records no direction_count, so the thinning
    # DRAWS its 20 uniformly rather than ranking -- which is what a matched control requires.
    split_and_train random_sentence_belief "${PREPARED_PREFIX}_random_sentence_belief_local"
    split_and_train random_sentence_belief_top20 \
        "${PREPARED_PREFIX}_random_sentence_belief_local" --tokens-per-trajectory 20
fi

echo "[$(ts)] === ALL DONE ==="
for t in eos_belief random_sentence_belief random_sentence_belief_top20 logitlens_p1_top20; do
    for m in lr mlp; do
        printf '%-30s %-4s ' "$t" "$m"
        grep -m1 'Best balanced accuracy' "$LOGS/${t}_${m}.txt" 2>/dev/null || echo '?'
    done
done
