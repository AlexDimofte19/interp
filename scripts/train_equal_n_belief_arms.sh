#!/usr/bin/env bash
# The per-sentence arms rebuilt so they hold the SAME SENTENCES, and the control made a control.
#
# Fourteen probes: four uncapped p1 arms and three p1-top20 arms, lr+mlp each. This supersedes
# the p1 and p1-top20 probes of ICLR log entries 45, 49 and 51, which are left on disk so the
# published numbers stay reproducible and the size of the correction is measurable.
#
# WHY THIS ROUND EXISTS -- two bugs, both of which changed WHICH ROWS a p1 arm holds while
# looking like nothing was wrong.
#
# 1. THE FINAL SENTENCE LEFT THE DATASET. Every per-sentence strategy emits one cutoff per
#    sentence plus an end_of_reasoning bookend, then merges cutoffs sharing a position. The
#    bookend sits at the last token of the last span, so a pick landing there merged into it
#    and was retagged end_of_reasoning -- a kind no p1 dataset keeps. Only the LAST sentence
#    can collide (the no_reasoning bookend sits at eos[0], outside every span), so at most one
#    row per step is lost, but the RATE is a property of the rule: jlens 30, logitlens 67, a
#    uniform draw 776, and eos all 3600 -- for eos it is a definition, since its pick IS each
#    sentence's last token. Arms meant to differ only in which token inside a span they cut
#    ended up differing in how many spans they held, by up to 4.8%. Worse than a count
#    mismatch: those rows are where the model has already decided (99.97% agree with the final
#    action at mean p 0.9994, against 70.12% / 0.8706 for interior rows), so the three
#    non-eos arms each carried ~4.8% near-free rows that eos did not.
#
# 2. THE RANDOM TOP-20 CONTROL WAS RANKED BY LOUDNESS. split_next_action_manifest.py inferred
#    rank-vs-draw from whether any row carried a direction score. random_per_sentence records
#    its cutoff's loudness as an analysis COVARIATE, so every row carried one, so the thinning
#    ranked it: mean layer-15 mass -3.402 -> -2.900, a shift as large as the jlens arm's own
#    ranking. It was a loudness-selected arm wearing a control's name. --thin-mode uniform now
#    says outright what the arm is instead of leaving it to be inferred.
#
# NEITHER NEEDS THE GPU BEYOND TRAINING. The labels already exist -- every arm's rollout
# evaluated its end_of_reasoning cutoff, 3600 of them, with a measured model_action, so the
# restored rows carry a real local belief rather than one assumed to equal the final action.
# And the activation is the step's last reasoning token, which is the same position for every
# arm, so link_end_of_reasoning_activations.py symlinks it out of the eos tree instead of
# paying ~873 forward passes for tensors that already exist.
#
# THE LABEL IS THE LOCAL BELIEF THROUGHOUT, as in entries 45-51: relabel_manifest_from_rollout.py
# writes the action the model emitted when its chain was cut at that token, and keeps the
# trajectory's own agent_action only as final_label. It differs from the final action ~30% of
# the time, which is what makes it worth decoding.
#
# READ THE RESULT AS DIFFERENCES, NOT LEVELS. ~4.8% of every arm is now near-deterministic by
# construction, so every absolute balanced accuracy rises. Entry 51's levels are not comparable
# to this round's; the arm-to-arm gaps are.
#
# NEVER an internal --eval-split: a token-major manifest splits over ENTRIES there, which leaks
# a trajectory across both halves. split_next_action_manifest.py splits over names.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT_ROOT=${OUT_ROOT:-/workspace/reasoning_theatre/rollout_strategies_baselines}
JLENS_ROLLOUT=${JLENS_ROLLOUT:-/workspace/reasoning_theatre/rollout_strategies/jlens_argmax_per_sentence}
# Entry 49's logitlens arm reused the jlens strategy class with a logitlens mass table, so its
# rollout lives under the jlens name in the baselines tree. Not a typo.
LOGITLENS_ROLLOUT=${LOGITLENS_ROLLOUT:-$OUT_ROOT/jlens_argmax_per_sentence}
HERE=${HERE:-/workspace/reasoning_theatre/equal_n_belief_arms}
PREPARED=${PREPARED:-/workspace/prepared/equal_n}
LOGS=$HERE/logs
PROBES=${PROBES:-/workspace/probes/local_belief_equalN}
mkdir -p "$LOGS" "$PROBES/p1" "$PROBES/p1-top20"

TRAJ=${TRAJ:-/workspace/trajectories/reveng/trajectories_train_single_step}
NAMES_FILE=${NAMES_FILE:-/workspace/reasoning_theatre/rollout_strategies/mass_l15_names.txt}
EVAL_NAMES=${EVAL_NAMES:-/workspace/prepared/next_action_mass_l15_eval_names.txt}
ACT=${ACT:-/workspace/activations}
EOS_TREE=${EOS_TREE:-$ACT/eos_mass3600_view/activations}
SEED=${SEED:-42}
EPOCHS=${EPOCHS:-50}
SINGLE_LAYER=${SINGLE_LAYER:-15}

cd "$REPO"
ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
UV="uv run --project $REPO --extra gpu"

WANT=${1:-all}
want() { [ "$WANT" = all ] || [ "$WANT" = "$1" ]; }

# tag -> (rollout dir, activation tree). eos is absent from the link/prepare loop on purpose:
# its tree already holds every endpoint, so nothing is linked and its existing prepared dir is
# reused as-is.
ARM_ROLLOUT_jlens=$JLENS_ROLLOUT
ARM_ROLLOUT_logitlens=$LOGITLENS_ROLLOUT
ARM_ROLLOUT_random=$OUT_ROOT/random_per_sentence
ARM_TREE_jlens=$ACT/argmax_per_sentence_l15
ARM_TREE_logitlens=$ACT/logitlens_argmax_per_sentence_l15
ARM_TREE_random=$ACT/random_per_sentence_l15

# ---- step 1: restore the collided final-sentence tensors -----------------------------------
if want link; then
    for arm in jlens logitlens random; do
        rollout=$(eval echo "\$ARM_ROLLOUT_$arm"); tree=$(eval echo "\$ARM_TREE_$arm")
        echo "[$(ts)] === $arm: link the end_of_reasoning tensors the dedupe cost us ==="
        $UV python "$REPO/scripts/link_end_of_reasoning_activations.py" \
            "$rollout" "$tree" --source-tree "$EOS_TREE" --names-file "$NAMES_FILE" --apply \
            2>&1 | tee "$LOGS/${arm}_link.txt"
    done
fi

# ---- step 2: prepare, then relabel to the local belief -------------------------------------
# eos skips the prepare: its tree is untouched by step 1, so more_belief_eos_belief_final
# (75,039 entries, both kinds already in it) is still exactly right. It only needs relabelling
# WITHOUT --keep-kinds this time, so its final-sentence rows survive.
if want prepare; then
    for arm in jlens logitlens random; do
        tree=$(eval echo "\$ARM_TREE_$arm"); rollout=$(eval echo "\$ARM_ROLLOUT_$arm")
        echo "[$(ts)] === $arm: prepare (layer $SINGLE_LAYER, every output token in the tree) ==="
        $UV interp-cli prepare_activations_for_probing \
            --activations-dir "$tree" --trajectories-dir "$TRAJ" \
            --probe-type next_action --layers "$SINGLE_LAYER" --steps all --output-indices all \
            --output-path "${PREPARED}_${arm}_final" --verbose \
            2>&1 | tee "$LOGS/${arm}_prepare.txt"

        echo "[$(ts)] === $arm: relabel to the local belief ==="
        $UV python "$REPO/telos_interp/loudness_analysis/rollouts/relabel_manifest_from_rollout.py" \
            "${PREPARED}_${arm}_final" "$rollout" "${PREPARED}_${arm}_local" \
            --report-csv "$HERE/${arm}_relabel_report.csv" \
            2>&1 | tee "$LOGS/${arm}_relabel.txt"
    done

    echo "[$(ts)] === eos: relabel to the local belief, endpoints KEPT this time ==="
    $UV python "$REPO/telos_interp/loudness_analysis/rollouts/relabel_manifest_from_rollout.py" \
        "/workspace/prepared/more_belief_eos_belief_final" "$OUT_ROOT/eos" \
        "${PREPARED}_eos_local" --report-csv "$HERE/eos_relabel_report.csv" \
        2>&1 | tee "$LOGS/eos_relabel.txt"
fi

# ---- step 3: cut all four arms to the sentences they all hold -------------------------------
# Equal counts are not enough: this pairs them row for row on (name, step, sentence_idx), so an
# eval-720 half holds the same sentences in every arm.
if want intersect; then
    echo "[$(ts)] === intersect the four arms ==="
    $UV python "$REPO/scripts/intersect_belief_arms.py" _eq \
        "${PREPARED}_jlens_local" "${PREPARED}_logitlens_local" \
        "${PREPARED}_eos_local" "${PREPARED}_random_local" \
        2>&1 | tee "$LOGS/intersect.txt"
fi

# ---- step 4: split and train ----------------------------------------------------------------
split_and_train() {  # split_and_train <tag> <subdir> <prepared> [extra split flags...]
    local tag=$1 subdir=$2 prepared=$3 split="${PREPARED}_${1}_split"; shift 3
    echo "[$(ts)] === $tag: split (eval pinned to the shared 720) ==="
    $UV python "$REPO/scripts/split_next_action_manifest.py" \
        "$prepared" --eval-names "$EVAL_NAMES" --single-layer "$SINGLE_LAYER" --seed "$SEED" "$@" \
        --train-out "${split}_train" --eval-out "${split}_eval" \
        2>&1 | tee "$LOGS/${tag}_split.txt"
    for mt in lr mlp; do
        echo "[$(ts)] === $tag: train $mt ==="
        $UV interp-cli train_next_action_probe \
            --train-data-path "${split}_train" --eval-data-path "${split}_eval" \
            --output-path "$PROBES/$subdir/next_action_probe_${tag}_${mt}.pt" \
            --model-type "$mt" --hidden-dims 1024 --learning-rate 3e-4 --weight-decay 0.001 \
            --dropout 0.0 --num-epochs "$EPOCHS" --batch-size 512 --class-weight balanced \
            --normalize --seed "$SEED" --device cuda --verbose \
            2>&1 | tee "$LOGS/${tag}_${mt}.txt"
    done
}

# The uncapped row: one token per sentence, every sentence.
if want p1; then
    for arm in jlens logitlens eos random; do
        split_and_train "$arm" p1 "${PREPARED}_${arm}_local_eq"
    done
fi

# The thinned row. The two lens arms rank by loudness, which is the selection under test. The
# control must NOT -- it carries a score on every row (recorded as a covariate) and would be
# ranked by it under --thin-mode auto, which is bug 2. No eos_top20 arm: eos records no
# loudness, so its thinning could only ever be a uniform draw, a different rule from the two
# lens arms it would sit beside.
if want p1-top20; then
    split_and_train jlens_top20     p1-top20 "${PREPARED}_jlens_local_eq"     --tokens-per-trajectory 20 --thin-mode rank
    split_and_train logitlens_top20 p1-top20 "${PREPARED}_logitlens_local_eq" --tokens-per-trajectory 20 --thin-mode rank
    split_and_train random_top20    p1-top20 "${PREPARED}_random_local_eq"    --tokens-per-trajectory 20 --thin-mode uniform
fi

echo "[$(ts)] === ALL DONE ==="
for t in jlens logitlens eos random jlens_top20 logitlens_top20 random_top20; do
    for m in lr mlp; do
        printf '%-20s %-4s ' "$t" "$m"
        grep -m1 'Best balanced accuracy' "$LOGS/${t}_${m}.txt" 2>/dev/null || echo '?'
    done
done
