#!/usr/bin/env bash
# Status for a running train_next_action_probe sweep.  watch -n 1 scripts/watch_probe_training.sh
#
# IT DOES NOT READ THE TRAINER'S STDOUT. `uv run python` buffers when it is not on a tty, so a
# redirected run writes nothing until it exits and a log tail shows an empty file for 36 minutes.
# Everything here comes from /proc and from four stat()s instead.
#
# NOTHING HERE WALKS AN ACTIVATION TREE. /workspace is MooseFS; a find over millions of .pt at
# 1 Hz would cost more than the job being watched. The per-arm progress bar is the .pt the
# process currently has OPEN, looked up in a ~550-line trajectory order cached on first tick.
set -uo pipefail

PREPARED=/workspace/prepared
PROBES=/workspace/probes/qwen_p2_local_belief
STATE=${STATE:-/tmp/watch_probe_training}
CACHE_NAME=_packed_activations.pt
ARMS=(jlens logitlens random)

mkdir -p "$STATE"

# The python process, not the `uv run` wrapper that spawned it -- and anchored on the
# interpreter, because a bare -f pattern also matches any shell whose command line merely
# mentions the trainer, including this script's own author.
PID=$(pgrep -f '^[^ ]*/python3 [^ ]*/interp-cli train_next_action_probe' | tail -1)

printf '=== qwen P2 probe training === %s\n\n' "$(date +%H:%M:%S)"

if [ -z "$PID" ]; then
    printf 'process:  NOT RUNNING\n\n'
else
    CMD=$(tr '\0' ' ' < "/proc/$PID/cmdline")
    TRAIN_PATH=$(sed -n 's/.*--train-data-path \([^ ]*\).*/\1/p' <<<"$CMD")
    MODEL_TYPE=$(sed -n 's/.*--model-type \([^ ]*\).*/\1/p' <<<"$CMD")
    ARM=$(basename "$TRAIN_PATH" | sed 's/qwen_p2_local_belief_//; s/_train$//')
    printf 'process:  pid %s   arm %-9s type %-3s   elapsed %s\n' \
        "$PID" "$ARM" "$MODEL_TYPE" "$(ps -o etime= -p "$PID" | tr -d ' ')"

    # The .pt currently open tells us both the phase and how far the load has got. Each open is
    # only ~20 ms wide, so a tick can miss it; the last hit is remembered rather than flickering.
    CUR=$(ls -l "/proc/$PID/fd" 2>/dev/null | grep -o '/workspace/activations/[^ ]*\.pt$' | tail -1)
    # The remembered hit is keyed on the pid, so a new process does not inherit the previous
    # one's last file and report a load that has already finished.
    if [ -n "$CUR" ]; then echo "$PID $CUR" > "$STATE/last_open"; fi
    if [ -z "$CUR" ]; then
        read -r lpid lpath < "$STATE/last_open" 2>/dev/null || lpid=
        [ "${lpid:-}" = "$PID" ] && CUR=${lpath:-}
    fi

    if [ -n "$CUR" ]; then
        case "$CUR" in
            */qwen_p2_selection_eval/*) HALF=val;   DS="${PREPARED}/qwen_p2_local_belief_${ARM}_val" ;;
            *)                          HALF=train; DS="${PREPARED}/qwen_p2_local_belief_${ARM}_train" ;;
        esac
        TRAJ=$(sed 's#.*/\(together_ai[^/]*\)/.*#\1#' <<<"$CUR")
        ORDER="$STATE/$(basename "$DS").order"
        if [ ! -s "$ORDER" ]; then
            printf 'loading:  %s -- building trajectory index (first tick only)...\n' "$HALF"
            python3 -c "
import json,sys
m=json.load(open(sys.argv[1]+'/manifest.json'))
seen=dict.fromkeys(s['name'] for s in m['samples'])
open(sys.argv[2],'w').write('\n'.join(seen)+'\n')" "$DS" "$ORDER" 2>/dev/null
        fi
        IDX=$(grep -n -m1 -F -x "$TRAJ" "$ORDER" 2>/dev/null | cut -d: -f1)
        TOT=$(wc -l < "$ORDER" 2>/dev/null)
        if [ -n "${IDX:-}" ] && [ "${TOT:-0}" -gt 0 ]; then
            PCT=$(( IDX * 100 / TOT ))
            BAR=$(printf '%*s' $(( PCT / 4 )) '' | tr ' ' '#')
            printf 'loading:  %-5s  traj %s/%s  [%-25s] %s%%\n' "$HALF" "$IDX" "$TOT" "$BAR" "$PCT"
        else
            printf 'loading:  %-5s  %s\n' "$HALF" "$TRAJ"
        fi
    else
        printf 'loading:  (no .pt open -- training, or between datasets)\n'
    fi
    echo
fi

printf 'GPU:      %s\n\n' "$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1)"

# A packed cache appearing means that dataset's ~36 min of small reads is done and will not be
# paid again by the mlp pass. Probes are the finished product.
printf '%-10s %-14s %-14s %s\n' arm cache-train cache-val probes
for a in "${ARMS[@]}"; do
    ct=NO; cv=NO
    [ -f "${PREPARED}/qwen_p2_local_belief_${a}_train/${CACHE_NAME}" ] && ct=packed
    [ -f "${PREPARED}/qwen_p2_local_belief_${a}_val/${CACHE_NAME}" ]   && cv=packed
    p=$(ls "${PROBES}/qwen_p2_local_belief_${a}_"*.pt 2>/dev/null | xargs -r -n1 basename | sed 's/.*_l27_//; s/\.pt//' | paste -sd, -)
    printf '%-10s %-14s %-14s %s\n' "$a" "$ct" "$cv" "${p:--}"
done
