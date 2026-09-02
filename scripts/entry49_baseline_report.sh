#!/usr/bin/env bash
# ICLR entry 49, part 3: read all SIXTEEN probes on the heldout 360 and rebuild the loudness
# report as a CLONE of entry 48's.
#
# Entry 48's page is the same figures over ten probes. Nothing here edits it: the output root
# is new and the three report scripts are extended through their --extra-probes flags, which
# default to empty, so re-running any of them against entry 48's inputs is byte-identical
# (entry 47 established the convention after a HEADLINE edit silently mixed two arms).
#
# The six new arms join the `p2` rowset. Since entry 48 every rowset holds IDENTICAL rows --
# every reasoning token of the 360 -- so the rowset name now selects which probes are read,
# not which tokens, and p2 is the one whose figures carry the label contrast.
#
# Step 5 is ~2 h of GPU: 87,221 .pt reads dominate, so six more probes cost almost nothing on
# top of entry 48's ten. Steps 6a-6c are CPU and cheap enough to re-run freely.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=${OUT:-/workspace/reasoning_theatre/probe_loudness_heldout360_16probes}
LB=${LB:-/workspace/reasoning_theatre/local_belief_probes/probes}
MASS=${MASS:-/workspace/probes/next_action_mass_l15}
NEW=${NEW:-/workspace/probes/local_belief_baselines}
mkdir -p "$OUT/logs"
cd "$REPO"
UV="uv run --project $REPO --extra gpu"
CSV=$OUT/heldout360_16probes.csv

# key=probe_key, where probe_key is "<parent dir>.<stem>" with next_action_probe_ stripped.
EXTRA_BUILD="randb_lr=local_belief_baselines.random_belief_lr,\
randb_mlp=local_belief_baselines.random_belief_mlp,\
ll1_lr=local_belief_baselines.logitlens_p1_lr,\
ll1_mlp=local_belief_baselines.logitlens_p1_mlp,\
ll2_lr=local_belief_baselines.logitlens_p2_lr,\
ll2_mlp=local_belief_baselines.logitlens_p2_mlp"
EXTRA_NAMES="randb_lr,randb_mlp,ll1_lr,ll1_mlp,ll2_lr,ll2_mlp"
# ';' separated, because the labels contain commas.
EXTRA_PLOT="randb_lr=random selection (belief), lr;randb_mlp=random selection (belief), mlp;\
ll1_lr=logitlens P1 (belief), lr;ll1_mlp=logitlens P1 (belief), mlp;\
ll2_lr=logitlens P2 (belief), lr;ll2_mlp=logitlens P2 (belief), mlp"

if [ ! -s "$CSV" ]; then
    echo "=== step 5: score 16 probes over every reasoning token of the heldout 360 (~2 h)"
    $UV python "$REPO/scripts/eval_probe_per_token.py" \
        --probe "$LB/local_belief_p1_lr.pt"        --probe "$LB/local_belief_p1_mlp.pt" \
        --probe "$LB/local_belief_p1_top20_lr.pt"  --probe "$LB/local_belief_p1_top20_mlp.pt" \
        --probe "$LB/local_belief_p2_lr.pt"        --probe "$LB/local_belief_p2_mlp.pt" \
        --probe "$MASS/next_action_probe_jlens_topall_lr.pt" \
        --probe "$MASS/next_action_probe_jlens_topall_mlp.pt" \
        --probe "$MASS/next_action_probe_random_topall_lr.pt" \
        --probe "$MASS/next_action_probe_random_topall_mlp.pt" \
        --probe "$NEW/next_action_probe_random_belief_lr.pt" \
        --probe "$NEW/next_action_probe_random_belief_mlp.pt" \
        --probe "$NEW/next_action_probe_logitlens_p1_lr.pt" \
        --probe "$NEW/next_action_probe_logitlens_p1_mlp.pt" \
        --probe "$NEW/next_action_probe_logitlens_p2_lr.pt" \
        --probe "$NEW/next_action_probe_logitlens_p2_mlp.pt" \
        --activations-dir /workspace/activations/heldout360_l15 \
        --lens-dir /workspace/activations/heldout360_lens \
        --trajectories-dir /workspace/trajectories/reveng/trajectories_train_single_step \
        --signal-json /workspace/jlens/direction_tokens_full.json \
        --layer 15 --full-probs --out "$CSV" \
        2>&1 | tee "$OUT/logs/step5_eval_probes.log"
else
    echo "=== step 5: $CSV exists, skipping"
fi

echo "=== step 6a: join -> per_token.csv"
$UV python "$REPO/scripts/build_probe_loudness_heldout.py" \
    --probe-csv "$CSV" --out "$OUT/per_token.csv" --extra-probes "$EXTRA_BUILD" \
    2>&1 | tee "$OUT/logs/step6a_build.log"

echo "=== step 6b: tables"
$UV python "$REPO/scripts/analyze_probe_loudness.py" \
    --per-token "$OUT/per_token.csv" --out "$OUT" --extra-probes "$EXTRA_NAMES" \
    2>&1 | tee "$OUT/logs/step6b_analyze.log"

echo "=== step 6c: figures"
$UV python "$REPO/scripts/plot_probe_loudness.py" \
    --per-token "$OUT/per_token.csv" --out "$OUT/plots" --extra-probes "$EXTRA_PLOT" \
    2>&1 | tee "$OUT/logs/step6c_plot.log"

echo "=== done -> $OUT"
