#!/usr/bin/env bash
# Every belief probe of a round, read on all 87,221 held-out tokens -- numbers, no figures.
#
#   bash scripts/eval_belief_arms_heldout.sh 24probes   # the entry-49/50 round: 24 probes
#   bash scripts/eval_belief_arms_heldout.sh equal_n    # the entry-52 round: those 24 + 14
#
# Pass 2 of a round. Pass 1 -- each probe on the tokens its OWN selection picked -- is already
# inside every checkpoint's `results` block, written by train_next_action_probe. This is the
# other population: every probe read on the SAME tokens, every reasoning token of the disjoint
# held-out 360, with no selection sitting between the axis and the label.
#
# THE TWO POPULATIONS ARE NOT COMPARABLE AND THE ORDERING INVERTS BETWEEN THEM (entry 49): on
# its own loud tokens the jlens arm wins and the random control is worst; on all 87,221 the
# random control wins and the jlens arm is worst, because a loud-selected probe is specialised
# to loud tokens while a uniform draw generalises across the chain. Both numbers are real.
# Name the population.
#
# EARLIER ROUNDS' PROBES ARE ALWAYS RE-SCORED IN THE SAME PASS, for two reasons. They are the
# regression check -- each must reproduce its published held-out number to 4 dp, or the join
# moved and the new numbers sit on a different measurement -- and they are the before-picture,
# which is the whole point of writing new probes to a separate directory: the size of a
# correction is then measurable rather than asserted.
#
# WHAT MOVES AND WHAT DOES NOT, for `equal_n`. The held-out population is unchanged -- all
# 87,221 tokens, no selection -- so a difference here comes only from the probe weights. The
# eval-720 numbers move for a second reason as well: ~4.8% of each p1 arm is now the
# near-deterministic final-sentence row, so absolute levels rise in every arm. Compare gaps
# across rounds, never levels.
#
# Deliberately stops before the figures. build_sixteen_probe_loudness_report.sh's steps
# 6b/6c/7/8 (analyze, plot, direction-word isolation, the report page) are untouched and run
# later against the same per-token CSV this produces, so nothing here has to be redone.
#
# ~2 h GPU for step 1 (87,221 individual .pt reads); steps 2-3 are CPU joins, minutes.
#
# This file replaced eval_more_belief_arms.sh and eval_equal_n_belief_arms.sh, which were 83%
# line-identical: the same three steps over a different probe list. A round is a `case` entry.
set -euo pipefail

ROUND=${1:?usage: eval_belief_arms_heldout.sh <24probes|equal_n>}

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ACT=${ACT:-/workspace/activations}
PROBES=${PROBES:-/workspace/probes}
RT=${RT:-/workspace/reasoning_theatre}
TRAJ=${TRAJ:-/workspace/trajectories/reveng/trajectories_train_single_step}
SIGNAL_JSON=${SIGNAL_JSON:-/workspace/jlens/direction_tokens_full.json}
LB=${LB:-$RT/local_belief_probes/probes}
MASS=${MASS:-$PROBES/next_action_mass_l15}
BASELINES=${BASELINES:-$PROBES/local_belief_baselines}
EQUAL_N=${EQUAL_N:-$PROBES/local_belief_equalN}

# eval_probe_per_token.py keys a column "<parent dir>.<stem MINUS the next_action_probe_ prefix>"
# -- e.g. local_belief_equalN/p1/next_action_probe_jlens_lr.pt becomes "p1.jlens_lr". The prefix
# IS stripped; assuming otherwise cost the equal-N round one failed join. The subdirectories are
# what keep the two cadences apart.
BASE_KEYS="randb_lr=local_belief_baselines.random_belief_lr,randb_mlp=local_belief_baselines.random_belief_mlp,\
ll1_lr=local_belief_baselines.logitlens_p1_lr,ll1_mlp=local_belief_baselines.logitlens_p1_mlp,\
ll2_lr=local_belief_baselines.logitlens_p2_lr,ll2_mlp=local_belief_baselines.logitlens_p2_mlp,\
eosb_lr=local_belief_baselines.eos_belief_lr,eosb_mlp=local_belief_baselines.eos_belief_mlp,\
rsb_lr=local_belief_baselines.random_sentence_belief_lr,rsb_mlp=local_belief_baselines.random_sentence_belief_mlp,\
rsb20_lr=local_belief_baselines.random_sentence_belief_top20_lr,\
rsb20_mlp=local_belief_baselines.random_sentence_belief_top20_mlp,\
ll1t20_lr=local_belief_baselines.logitlens_p1_top20_lr,ll1t20_mlp=local_belief_baselines.logitlens_p1_top20_mlp"

# The ten probes every round scores: the six local-belief arms and the four mass-era arms.
COMMON_PROBES=(
    "$LB"/local_belief_p1_lr.pt "$LB"/local_belief_p1_mlp.pt
    "$LB"/local_belief_p1_top20_lr.pt "$LB"/local_belief_p1_top20_mlp.pt
    "$LB"/local_belief_p2_lr.pt "$LB"/local_belief_p2_mlp.pt
    "$MASS"/next_action_probe_jlens_topall_lr.pt "$MASS"/next_action_probe_jlens_topall_mlp.pt
    "$MASS"/next_action_probe_random_topall_lr.pt "$MASS"/next_action_probe_random_topall_mlp.pt
)

case "$ROUND" in
24probes)
    OUT=${OUT:-$RT/probe_loudness_heldout360_24probes}
    CSV=${CSV:-$OUT/heldout360_24probes.csv}
    N_PROBES=24
    EXTRA_BUILD="$BASE_KEYS"
    GLOBS=("$BASELINES"/*.pt)
    ;;
equal_n)
    OUT=${OUT:-$RT/probe_loudness_heldout360_equal_n}
    CSV=${CSV:-$OUT/heldout360_equal_n.csv}
    N_PROBES=38
    # The `eq*` keys are this round; everything before them is the previous rounds', unchanged
    # so the report and the inventory keep resolving them.
    EXTRA_BUILD="$BASE_KEYS,\
eq_p1_jlens_lr=p1.jlens_lr,eq_p1_jlens_mlp=p1.jlens_mlp,\
eq_p1_ll_lr=p1.logitlens_lr,eq_p1_ll_mlp=p1.logitlens_mlp,\
eq_p1_eos_lr=p1.eos_lr,eq_p1_eos_mlp=p1.eos_mlp,\
eq_p1_rand_lr=p1.random_lr,eq_p1_rand_mlp=p1.random_mlp,\
eq_t20_jlens_lr=p1-top20.jlens_top20_lr,\
eq_t20_jlens_mlp=p1-top20.jlens_top20_mlp,\
eq_t20_ll_lr=p1-top20.logitlens_top20_lr,\
eq_t20_ll_mlp=p1-top20.logitlens_top20_mlp,\
eq_t20_rand_lr=p1-top20.random_top20_lr,\
eq_t20_rand_mlp=p1-top20.random_top20_mlp"
    GLOBS=("$BASELINES"/*.pt "$EQUAL_N"/p1/*.pt "$EQUAL_N"/p1-top20/*.pt)
    ;;
*)
    echo "unknown round: $ROUND (expected 24probes or equal_n)" >&2
    exit 1
    ;;
esac

mkdir -p "$OUT/logs"
cd "$REPO"
ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
UV="uv run --project $REPO --extra gpu"

# ---- step 1: score every probe on every held-out token ------------------------------------
# Skipped if the CSV is already there, because it is the only GPU-hours step in the file.
if [ -s "$CSV" ]; then
    echo "[$(ts)] === step 1: $CSV already exists, skipping the GPU pass ==="
else
    echo "[$(ts)] === step 1: $N_PROBES probes x 87,221 held-out tokens ($ROUND) ==="
    args=()
    for p in "${COMMON_PROBES[@]}" "${GLOBS[@]}"; do
        [ -e "$p" ] || { echo "MISSING probe: $p" >&2; exit 1; }
        args+=(--probe "$p")
    done
    echo "    ${#args[@]} flags = $(( ${#args[@]} / 2 )) probes"
    $UV python "$REPO/telos_interp/loudness_analysis/score_probes_per_token.py" "${args[@]}" \
        --activations-dir "$ACT/heldout360_l15" --lens-dir "$ACT/heldout360_lens" \
        --trajectories-dir "$TRAJ" --signal-json "$SIGNAL_JSON" \
        --layer 15 --full-probs --out "$CSV" \
        2>&1 | tee "$OUT/logs/step1_eval_probes.log"
fi

# ---- step 2: join the at-token belief label onto those predictions -------------------------
# The rollout supplies label_local; the commitment CSV supplies label_final and the loudness
# coordinates. Pure CPU. --mass-column stays at the jlens ruler: neither round is comparing
# rulers, and the logitlens cut can be produced later from the same inputs.
echo "[$(ts)] === step 2: join -> per_token.csv ==="
$UV python "$REPO/telos_interp/loudness_analysis/join_rollouts.py" \
    --probe-csv "$CSV" --out "$OUT/per_token.csv" --extra-probes "$EXTRA_BUILD" \
    --mass-column jlens_mass_L15 \
    2>&1 | tee "$OUT/logs/step2_build.log"

# ---- step 3: the numbers ------------------------------------------------------------------
echo "[$(ts)] === step 3: balanced accuracy per probe, both label definitions ==="
$UV python "$REPO/telos_interp/loudness_analysis/summarise_probe_accuracy.py" "$OUT/per_token.csv" \
    --out "$OUT/heldout_balanced_accuracy.csv" --json-out "$OUT/heldout_balanced_accuracy.json" \
    2>&1 | tee "$OUT/logs/step3_score.log"

echo "[$(ts)] === DONE -> $OUT/heldout_balanced_accuracy.csv ==="
