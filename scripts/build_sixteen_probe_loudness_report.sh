#!/usr/bin/env bash
# Sixteen probes on the held-out 360, under both loudness rulers.
#
# Reads all SIXTEEN probes on the heldout 360 and rebuilds the loudness report as a CLONE of
# the ten-probe page (ICLR log entry 49, building on entry 48).
#
# The ten-probe page is the same figures over ten probes. Nothing here edits it: the output
# root is new and the three report scripts are extended through their --extra-probes flags,
# which default to empty, so re-running any of them against the ten-probe inputs is
# byte-identical (the convention was established after a HEADLINE edit silently mixed two
# arms).
#
# BOTH RULERS, ALWAYS. `dir_logmass` -- the loudness axis every figure bins on -- is a COLUMN
# CHOICE, not a re-gather: each source CSV carries jlens_mass_L15 and logitlens_mass_L15 side
# by side. Binning everything by the jlens alone would order the two logitlens-selected arms
# by a ruler they were never selected by, and at layer 15 the two lenses' top-20 sets overlap
# only ~50%. So steps 6a-6c run TWICE, once per ruler, and the page draws every figure in a
# left/right pair. Loudness is never unqualified here: it is jlens loudness or logitlens
# loudness, and every title, axis and caption names its lens.
#
# The six new arms join the `p2` rowset. Since the ten-probe page every rowset holds IDENTICAL rows --
# every reasoning token of the 360 -- so the rowset name now selects which probes are read,
# not which tokens, and p2 is the one whose figures carry the label contrast.
#
# Step 5 is ~2 h of GPU: 87,221 .pt reads dominate, so six more probes cost almost nothing on
# top of the ten. Steps 6a-6c are CPU and cheap enough to re-run freely, and step 7 (the
# direction-word isolation) and step 8 (the page) are seconds.
#
#   bash scripts/build_sixteen_probe_loudness_report.sh
#   OUT=/tmp/clone bash scripts/build_sixteen_probe_loudness_report.sh   # a clone, not the page
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

# ---- steps 6a-6c, once per ruler -------------------------------------------------------
# The ONLY difference between the two passes is --mass-column, i.e. which lens's layer-15
# direction mass becomes `dir_logmass`. Both columns are already in $CSV, so neither pass
# re-reads an activation.
echo "=== step 6a: join -> per_token.csv (jlens loudness)"
$UV python "$REPO/scripts/build_probe_loudness_heldout.py" \
    --probe-csv "$CSV" --out "$OUT/per_token.csv" --extra-probes "$EXTRA_BUILD" \
    --mass-column jlens_mass_L15 \
    2>&1 | tee "$OUT/logs/step6a_build.log"

# The direction-word isolation reads the two rulers by name (per_token_<lens>_loudness.csv),
# and the jlens cut IS per_token.csv. Hardlink rather than copy -- they are byte-identical and
# 190 MB each; fall back to a copy on a filesystem that refuses the link.
ln -f "$OUT/per_token.csv" "$OUT/per_token_jlens_loudness.csv" 2>/dev/null \
    || cp "$OUT/per_token.csv" "$OUT/per_token_jlens_loudness.csv"

echo "=== step 6a': join -> per_token_logitlens_loudness.csv (logitlens loudness)"
$UV python "$REPO/scripts/build_probe_loudness_heldout.py" \
    --probe-csv "$CSV" --out "$OUT/per_token_logitlens_loudness.csv" --extra-probes "$EXTRA_BUILD" \
    --mass-column logitlens_mass_L15 \
    2>&1 | tee "$OUT/logs/step6a_build_logitlens_loudness.log"

echo "=== step 6b: tables (jlens loudness)"
$UV python "$REPO/scripts/analyze_probe_loudness.py" \
    --per-token "$OUT/per_token.csv" --out "$OUT" --extra-probes "$EXTRA_NAMES" \
    2>&1 | tee "$OUT/logs/step6b_analyze.log"

echo "=== step 6b': tables (logitlens loudness)"
$UV python "$REPO/scripts/analyze_probe_loudness.py" \
    --per-token "$OUT/per_token_logitlens_loudness.csv" --out "$OUT/logitlens_loudness" \
    --extra-probes "$EXTRA_NAMES" \
    2>&1 | tee "$OUT/logs/analyze_logitlens_loudness.log"

echo "=== step 6c: figures (jlens loudness)"
$UV python "$REPO/scripts/plot_probe_loudness.py" \
    --per-token "$OUT/per_token.csv" --out "$OUT/plots" --extra-probes "$EXTRA_PLOT" \
    2>&1 | tee "$OUT/logs/step6c_plot.log"

echo "=== step 6c': figures (logitlens loudness)"
$UV python "$REPO/scripts/plot_probe_loudness.py" \
    --per-token "$OUT/per_token_logitlens_loudness.csv" --out "$OUT/plots_logitlens_loudness" \
    --extra-probes "$EXTRA_PLOT" \
    2>&1 | tee "$OUT/logs/plot_logitlens_loudness.log"

# ---- step 7: is loudness just reading a word the model already typed? -------------------
echo "=== step 7: direction-word isolation, under both rulers"
$UV python "$REPO/scripts/analyze_direction_word_isolation.py" --src "$OUT" \
    2>&1 | tee "$OUT/logs/step7_direction_words.log"

# ---- step 8: the page ------------------------------------------------------------------
# Reads plots/ and plots_logitlens_loudness/ plus both summary.json, so it must come last.
echo "=== step 8: build the page"
$UV python "$REPO/scripts/build_sixteen_probe_report_page.py" --out "$OUT" \
    2>&1 | tee "$OUT/logs/step8_report.log"

echo "=== done -> $OUT"
echo "    page:    $OUT/report.html"
echo "    figures: $OUT/plots (jlens loudness) and $OUT/plots_logitlens_loudness (logitlens loudness)"
