#!/usr/bin/env bash
# The fourteen equal-N belief probes, plus the 24 they supersede, on all 87,221 held-out tokens.
#
# Pass 2 of the equal-N round. Pass 1 (each probe on the tokens its OWN selection picked) is in
# every checkpoint's `results` block already, and is now genuinely like-for-like: the four p1
# arms hold the same trajectories AND the same sentences, so an eval-720 half is paired row for
# row across arms. This is the other population -- every probe read on the same tokens, every
# reasoning token of the disjoint held-out 360, no selection between the axis and the label.
#
# THE OLD PROBES ARE RE-SCORED IN THE SAME PASS FOR TWO REASONS. They are the regression check
# (each must reproduce its published held-out number to 4 dp, or the join moved and the new
# numbers sit on a different measurement), and they are the before-picture: the whole point of
# writing the new probes to a separate directory is that the size of the correction is
# measurable rather than asserted.
#
# WHAT MOVES AND WHAT DOES NOT. The held-out population is unchanged -- all 87,221 tokens, no
# selection -- so a difference here comes only from the probe weights. The eval-720 numbers move
# for a second reason as well: ~4.8% of each p1 arm is now the near-deterministic final-sentence
# row, so absolute levels rise in every arm. Compare gaps across rounds, never levels.
#
# Deliberately stops before the figures, as the previous round did. The per-token CSV this
# writes is what build_sixteen_probe_loudness_report.sh's steps 6b/6c/7/8 consume.
#
# ~2 h GPU for step 1 (87,221 individual .pt reads); steps 2-3 are CPU joins, minutes.
set -euo pipefail

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
OUT=${OUT:-$RT/probe_loudness_heldout360_equal_n}
CSV=$OUT/heldout360_equal_n.csv
mkdir -p "$OUT/logs"

cd "$REPO"
ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
UV="uv run --project $REPO --extra gpu"

# eval_probe_per_token.py keys a column "<parent dir>.<stem MINUS the next_action_probe_ prefix>"
# -- e.g. local_belief_equalN/p1/next_action_probe_jlens_lr.pt becomes "p1.jlens_lr". The prefix
# IS stripped; assuming otherwise cost this round one failed join. The subdirectories are what
# keep the two cadences apart. The `eq*` keys below are this round; the rest are the previous
# rounds' keys, unchanged so the report and the inventory keep resolving them.
EXTRA_BUILD="randb_lr=local_belief_baselines.random_belief_lr,randb_mlp=local_belief_baselines.random_belief_mlp,\
ll1_lr=local_belief_baselines.logitlens_p1_lr,ll1_mlp=local_belief_baselines.logitlens_p1_mlp,\
ll2_lr=local_belief_baselines.logitlens_p2_lr,ll2_mlp=local_belief_baselines.logitlens_p2_mlp,\
eosb_lr=local_belief_baselines.eos_belief_lr,eosb_mlp=local_belief_baselines.eos_belief_mlp,\
rsb_lr=local_belief_baselines.random_sentence_belief_lr,rsb_mlp=local_belief_baselines.random_sentence_belief_mlp,\
rsb20_lr=local_belief_baselines.random_sentence_belief_top20_lr,\
rsb20_mlp=local_belief_baselines.random_sentence_belief_top20_mlp,\
ll1t20_lr=local_belief_baselines.logitlens_p1_top20_lr,ll1t20_mlp=local_belief_baselines.logitlens_p1_top20_mlp,\
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

# ---- step 1: score every probe on every held-out token ------------------------------------
# Skipped if the CSV is already there, because it is the only GPU-hours step in the file.
if [ -s "$CSV" ]; then
    echo "[$(ts)] === step 1: $CSV already exists, skipping the GPU pass ==="
else
    echo "[$(ts)] === step 1: 38 probes x 87,221 held-out tokens ==="
    args=()
    for p in "$LB"/local_belief_p1_lr.pt "$LB"/local_belief_p1_mlp.pt \
             "$LB"/local_belief_p1_top20_lr.pt "$LB"/local_belief_p1_top20_mlp.pt \
             "$LB"/local_belief_p2_lr.pt "$LB"/local_belief_p2_mlp.pt \
             "$MASS"/next_action_probe_jlens_topall_lr.pt "$MASS"/next_action_probe_jlens_topall_mlp.pt \
             "$MASS"/next_action_probe_random_topall_lr.pt "$MASS"/next_action_probe_random_topall_mlp.pt \
             "$BASELINES"/*.pt "$EQUAL_N"/p1/*.pt "$EQUAL_N"/p1-top20/*.pt; do
        [ -e "$p" ] || { echo "MISSING probe: $p" >&2; exit 1; }
        args+=(--probe "$p")
    done
    echo "    ${#args[@]} flags = $(( ${#args[@]} / 2 )) probes"
    $UV python "$REPO/scripts/eval_probe_per_token.py" "${args[@]}" \
        --activations-dir "$ACT/heldout360_l15" --lens-dir "$ACT/heldout360_lens" \
        --trajectories-dir "$TRAJ" --signal-json "$SIGNAL_JSON" \
        --layer 15 --full-probs --out "$CSV" \
        2>&1 | tee "$OUT/logs/step1_eval_probes.log"
fi

# ---- step 2: join the at-token belief label onto those predictions -------------------------
echo "[$(ts)] === step 2: join -> per_token.csv ==="
$UV python "$REPO/scripts/build_probe_loudness_heldout.py" \
    --probe-csv "$CSV" --out "$OUT/per_token.csv" --extra-probes "$EXTRA_BUILD" \
    --mass-column jlens_mass_L15 \
    2>&1 | tee "$OUT/logs/step2_build.log"

# ---- step 3: the numbers ------------------------------------------------------------------
echo "[$(ts)] === step 3: balanced accuracy per probe, both label definitions ==="
$UV python "$REPO/scripts/score_probes_heldout.py" "$OUT/per_token.csv" \
    --out "$OUT/heldout_balanced_accuracy.csv" --json-out "$OUT/heldout_balanced_accuracy.json" \
    2>&1 | tee "$OUT/logs/step3_score.log"

echo "[$(ts)] === DONE -> $OUT/heldout_balanced_accuracy.csv ==="
