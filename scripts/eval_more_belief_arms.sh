#!/usr/bin/env bash
# The eight new belief probes on all 87,221 held-out tokens -- numbers, no figures.
#
# Pass 2 of the round. Pass 1 (each probe on the tokens its OWN selection picked) is already
# inside every checkpoint's `results` block, written by train_next_action_probe. This is the
# other population: every probe read on the SAME tokens, every reasoning token of the disjoint
# held-out 360, with no selection sitting between the axis and the label.
#
# THE TWO ARE NOT COMPARABLE AND THE ORDERING INVERTS BETWEEN THEM (entry 49): on its own loud
# tokens the jlens arm wins and the random control is worst; on all 87,221 the random control
# wins and the jlens arm is worst, because a loud-selected probe is specialised to loud tokens
# while a uniform draw generalises across the chain. Both numbers are real. Name the population.
#
# The 16 existing probes are scored in the SAME pass, not because this round needs them but
# because they are the regression check: every one must reproduce its published held-out
# number to 4 dp, or the join moved and the eight new numbers sit on a different measurement.
#
# Deliberately stops before the figures. build_sixteen_probe_loudness_report.sh's steps 6b/6c/7/8
# (analyze, plot, direction-word isolation, the report page) are untouched and can be run later
# against the same per-token CSV this produces, so nothing here has to be redone to build the
# 24-probe report.
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
OUT=${OUT:-$RT/probe_loudness_heldout360_24probes}
CSV=$OUT/heldout360_24probes.csv
mkdir -p "$OUT/logs"

cd "$REPO"
ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
UV="uv run --project $REPO --extra gpu"

# The eight added this round; the six entry-49 arms keep the keys the report already uses.
EXTRA_BUILD="randb_lr=local_belief_baselines.random_belief_lr,randb_mlp=local_belief_baselines.random_belief_mlp,\
ll1_lr=local_belief_baselines.logitlens_p1_lr,ll1_mlp=local_belief_baselines.logitlens_p1_mlp,\
ll2_lr=local_belief_baselines.logitlens_p2_lr,ll2_mlp=local_belief_baselines.logitlens_p2_mlp,\
eosb_lr=local_belief_baselines.eos_belief_lr,eosb_mlp=local_belief_baselines.eos_belief_mlp,\
rsb_lr=local_belief_baselines.random_sentence_belief_lr,rsb_mlp=local_belief_baselines.random_sentence_belief_mlp,\
rsb20_lr=local_belief_baselines.random_sentence_belief_top20_lr,\
rsb20_mlp=local_belief_baselines.random_sentence_belief_top20_mlp,\
ll1t20_lr=local_belief_baselines.logitlens_p1_top20_lr,ll1t20_mlp=local_belief_baselines.logitlens_p1_top20_mlp"

# ---- step 1: score every probe on every held-out token ------------------------------------
# Skipped if the CSV is already there, because it is the only GPU-hours step in the file.
if [ -s "$CSV" ]; then
    echo "[$(ts)] === step 1: $CSV already exists, skipping the GPU pass ==="
else
    echo "[$(ts)] === step 1: 24 probes x 87,221 held-out tokens ==="
    args=()
    for p in "$LB"/local_belief_p1_lr.pt "$LB"/local_belief_p1_mlp.pt \
             "$LB"/local_belief_p1_top20_lr.pt "$LB"/local_belief_p1_top20_mlp.pt \
             "$LB"/local_belief_p2_lr.pt "$LB"/local_belief_p2_mlp.pt \
             "$MASS"/next_action_probe_jlens_topall_lr.pt "$MASS"/next_action_probe_jlens_topall_mlp.pt \
             "$MASS"/next_action_probe_random_topall_lr.pt "$MASS"/next_action_probe_random_topall_mlp.pt \
             "$BASELINES"/*.pt; do
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
# The rollout supplies label_local; the commitment CSV supplies label_final and the loudness
# coordinates. Pure CPU. --mass-column stays at the jlens ruler: this round is not comparing
# rulers, and the logitlens cut can be produced later from the same inputs.
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
