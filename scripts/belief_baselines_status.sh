#!/usr/bin/env bash
# Status of the belief-baseline round: one snapshot of
# gather -> rollout arms -> probes -> report (ICLR log entry 49).
#
# Cheap enough for `watch -n 1 ./scripts/belief_baselines_status.sh` -- it only does readdirs and
# reads the head/tail of a few logs, never a recursive find. Everything it reads lives
# under /workspace, so any session can run it; nothing is scoped to the one that started
# the run.
#
#   watch -n 1 ./scripts/belief_baselines_status.sh
#   ./scripts/belief_baselines_status.sh            # once, to find out where the round stopped
set -uo pipefail

ACT_LL=${ACT_LL:-/workspace/activations/logitlens_mass_l15}
ACT_P1=${ACT_P1:-/workspace/activations/logitlens_argmax_per_sentence_l15}
ARMS_ROOT=${ARMS_ROOT:-/workspace/reasoning_theatre/rollout_strategies_baselines}
PROBES=${PROBES:-/workspace/probes/local_belief_baselines}
# The belief-baseline data dirs keep their original on-disk names (see RENAMES.md); only
# the scripts were renamed, so an existing run stays readable and resumable.
BELIEF_BASELINE_ROOT=${BELIEF_BASELINE_ROOT:-/workspace/reasoning_theatre/entry49_baselines}
PLOGS=${PLOGS:-$BELIEF_BASELINE_ROOT/logs}
REPORT=${REPORT:-/workspace/reasoning_theatre/probe_loudness_heldout360_16probes}
BELIEF_BASELINE_LOGS=${BELIEF_BASELINE_LOGS:-/workspace/logs/entry49}
LOGDIR=${LOGDIR:-$BELIEF_BASELINE_LOGS}
EXPECTED=${EXPECTED:-3600}
SIZES=${SIZES:-"5 7 9 11 13 15"}

now_utc=$(date -u +%H:%M:%S)
now_loc=$(TZ=${TZ_LOCAL:-Europe/Berlin} date +%H:%M:%S)
echo "=== entry 49 @ $now_utc UTC / $now_loc CEST"

# `pgrep -af` + an argv match, never a bare name, for two reasons: a SIGSTOPped leftover from
# an older run sits in the process table for days matching a bare pgrep for the script name,
# and this script's own name must never match itself (it mentions every process it looks for).
# -ww: unlimited width. Without it ps truncates argv to the terminal width, and the
# arm match (which looks for the output root inside --output-dir) silently fails.
PS_SNAPSHOT=$(ps -ewwo pid,args --no-headers 2>/dev/null | grep -v belief_baselines_status)
alive() { printf '%s\n' "$PS_SNAPSHOT" | grep -F -- "$1" | grep -qF -- "${2:-}"; }
state() { alive "$1" "${2:-}" && echo ALIVE || echo "-"; }

count_dirs() { ls -U "$1" 2>/dev/null | wc -l; }
count_json() { ls -U "$1" 2>/dev/null | grep -c '\.json$'; }
# grep -c exits 1 when it counts zero, so `|| echo 0` would print a SECOND zero.
errs() {
    [ -e "$1" ] || { echo 0; return; }
    local c
    c=$(grep -ciE 'traceback|CUDA out of memory|Killed|^Error' "$1" 2>/dev/null) || true
    echo "${c:-0}"
}

# ---- 1. the logitlens mass gather ------------------------------------------------------
glog=$(ls -1t "$LOGDIR"/step1_logitlens_mass_l15_*.txt 2>/dev/null | head -1)
tot=0
line=""
for s in $SIZES; do c=$(count_dirs "$ACT_LL/size$s"); tot=$((tot+c)); line="$line size$s=$c"; done
printf "1. logitlens gather   %-6s  %5s/%s  errors=%s\n" \
    "$(state jlens_mass_l15.sh)" "$tot" "$EXPECTED" "$(errs "$glog")"
echo "  ..$line"
if [ -n "$glog" ] && [ "$tot" -gt 0 ] && [ "$tot" -lt "$EXPECTED" ]; then
    st=$(sed -n 's/^=== START: \(.*\) UTC$/\1/p;s/^=== START: \([^ ]*\) .*/\1/p' "$glog" | head -1)
    if [ -n "$st" ]; then
        el=$(( $(date -u +%s) - $(date -u -d "$st" +%s 2>/dev/null || echo 0) ))
        if [ "$el" -gt 60 ]; then
            rate=$(awk -v t=$tot -v e=$el 'BEGIN{printf "%.1f", t/(e/60)}')
            eta=$(awk -v t=$tot -v x=$EXPECTED -v e=$el 'BEGIN{printf "%d", (x-t)*(e/t)/60}')
            echo "  ..$rate/min, ~$eta min left (~$(date -u -d "+$eta minutes" +%H:%M) UTC / $(TZ=${TZ_LOCAL:-Europe/Berlin} date -d "+$eta minutes" +%H:%M) CEST)"
        fi
    fi
fi

# ---- 2. the three rollout arms ---------------------------------------------------------
printf "2. rollout arms       %-6s\n" "$(state run_inference.py "$ARMS_ROOT")"
# The two logitlens arms REUSE the jlens strategy names -- the strategy is "cut at each
# sentence's loudest token", the lens that defines loudness is a separate flag -- so the
# arm's directory says jlens while holding logitlens results. Print the lens each arm was
# actually run with, read from its own log header, so the name cannot mislead.
for a in recorded_selection jlens_argmax_per_sentence jlens_top_k_global; do
    n=$(count_json "$ARMS_ROOT/$a")
    al=$(ls -1t "$ARMS_ROOT/$a"/logs/run_*.txt 2>/dev/null | head -1)
    lens=$(sed -n 's/.*[[:space:]]lens=\([a-z]*\).*/\1/p' "$al" 2>/dev/null | head -1)
    printf "  ..%-26s %5s/%s  lens=%-9s errors=%s\n" \
        "$a" "$n" "$EXPECTED" "${lens:-?}" "$(errs "$al")"
done

# ---- 3. the logitlens-P1 activation gather ---------------------------------------------
p1=0; for s in $SIZES; do p1=$((p1 + $(count_dirs "$ACT_P1/size$s"))); done
printf "3. P1 L15 gather      %-6s  %5s/%s\n" \
    "$(state gather_local_belief_activations.py)" "$p1" "$EXPECTED"

# ---- 4. the six probes -----------------------------------------------------------------
printf "4. probes             %-6s\n" "$(state train_next_action_probe)"
for t in random_belief logitlens_p1 logitlens_p2; do for m in lr mlp; do
    f="$PROBES/next_action_probe_${t}_${m}.pt"
    if [ -e "$f" ]; then
        acc=$(grep -m1 -oE 'Best balanced accuracy[^0-9]*[0-9.]+' "$PLOGS/${t}_${m}.txt" 2>/dev/null | grep -oE '[0-9.]+$')
        printf "  ..%-16s %-4s done  %s\n" "$t" "$m" "${acc:-?}"
    else
        printf "  ..%-16s %-4s -\n" "$t" "$m"
    fi
done; done

# ---- 5/6. scoring and the report -------------------------------------------------------
csv=$REPORT/heldout360_16probes.csv
printf "5. heldout scoring    %-6s  %s\n" "$(state eval_probe_per_token.py)" \
    "$([ -s "$csv" ] && echo "$(wc -l < "$csv") rows" || echo '-')"
printf "6. report             %-6s  per_token=%s tables=%s plots=%s\n" \
    "$(state plot_probe_loudness.py)" \
    "$([ -s "$REPORT/per_token.csv" ] && echo yes || echo '-')" \
    "$(count_dirs "$REPORT/tables")" "$(count_dirs "$REPORT/plots")"

# Only surface a tail when something actually went wrong.
for l in "$glog" "$ARMS_ROOT"/*/logs/run_*.txt; do
    [ -e "$l" ] || continue
    if [ "$(errs "$l")" -gt 0 ] 2>/dev/null; then
        echo "!! errors in $l:"; grep -iE 'traceback|CUDA out of memory|Killed|^Error' "$l" | tail -3
    fi
done
