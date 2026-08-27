#!/usr/bin/env bash
# One snapshot of the whole grid round: gather, then prepare, then train.
#
# Designed for `watch -n 30 ./scripts/grid_round_status.sh`, and for a fresh session to run
# once to find out where the round actually stopped. Everything it reads lives under
# /workspace, so it works from any session -- nothing here is scoped to the one that
# started the run.
#
#   ./scripts/grid_round_status.sh
#   ACT=/tmp/smoke PROBES=/tmp/probes ./scripts/grid_round_status.sh
set -uo pipefail

ACT=${ACT:-/workspace/activations/grid_reasoning_tokens}
PREPARED=${PREPARED:-/workspace/prepared/grid_l15}     # reads ${PREPARED}_<arm>
PROBES=${PROBES:-/workspace/probes/grid}
LOGS=${LOGS:-$PROBES/logs}
LOG=${LOG:-/workspace/logs/grid_gather.log}
STATUS=${STATUS:-/workspace/logs/grid_gather_status.txt}
ARMS=${ARMS:-"jlens logitlens random"}
SEEDS=${SEEDS:-"42 43 44"}
TAG=${TAG:-l15}
EXPECTED=${EXPECTED:-3600}

echo "=== 1. gather ==="
# The argv check matters: a stopped (state Tl) leftover from an older run sits in the
# process table for days and matches a bare pgrep for the script name.
if pgrep -af "jlens_reasoning_tokens.py" 2>/dev/null | grep -q -- "$ACT"; then
    echo "  gather: ALIVE"
else
    echo "  gather: not running"
fi
tail -2 "$STATUS" 2>/dev/null | sed 's/^/  /'
errs=$(grep -ciE "traceback|cuda error|out of memory" "$LOG" 2>/dev/null || echo 0)
echo "  errors: $errs"
[ "$errs" = "0" ] || grep -iE -m2 -A6 "traceback|cuda error|out of memory" "$LOG" | sed 's/^/    /'

echo
echo "=== 2. tree ==="
if [ -d "$ACT" ]; then
    total=0
    for d in "$ACT"/size*/; do
        [ -d "$d" ] || continue
        n=$(find "$d" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l)
        printf '  %-8s %5d\n' "$(basename "$d")" "$n"
        total=$((total + n))
    done
    printf '  %-8s %5d / %s trajectories\n' TOTAL "$total" "$EXPECTED"
    printf '  %-8s %5d selection record(s)\n' records \
        "$(find "$ACT" -name '*_jlens_selection.json' 2>/dev/null | wc -l)"
else
    echo "  (no tree at $ACT yet)"
fi

echo
echo "=== 3. prepared ==="
for arm in $ARMS; do
    m="${PREPARED}_${arm}/manifest.json"
    if [ -f "$m" ]; then
        # Never read a manifest into a model's context; count with one line of python.
        read -r entries names < <(python3 -c "
import json,sys
m=json.load(open(sys.argv[1]))
e=m.get('trajectories') or m.get('samples') or []
print(len(e), len({x['name'] for x in e}))" "$m" 2>/dev/null || echo "? ?")
        printf '  %-11s %8s entries / %5s trajectories\n' "$arm" "$entries" "$names"
    else
        printf '  %-11s %8s\n' "$arm" "not prepared"
    fi
done

echo
echo "=== 4. probes ==="
printf '  %-11s %-6s %-6s %s\n' arm seed model result
done_n=0; total_n=0
for arm in $ARMS; do
    for seed in $SEEDS; do
        for model in lr mlp; do
            total_n=$((total_n + 1))
            f="$LOGS/${arm}_${TAG}_seed${seed}_${model}.txt"
            r=$(grep -m1 -oE "Balanced Accuracy: [0-9.]+" "$f" 2>/dev/null | grep -oE "[0-9.]+$")
            if [ -n "$r" ]; then
                done_n=$((done_n + 1))
                printf '  %-11s %-6s %-6s %s\n' "$arm" "$seed" "$model" "$r"
            elif [ -f "$f" ]; then
                printf '  %-11s %-6s %-6s %s\n' "$arm" "$seed" "$model" "running/failed"
            fi
        done
    done
done
echo "  $done_n/$total_n runs complete"

echo
date -Is
