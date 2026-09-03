#!/usr/bin/env bash
# Live status of the truncation-strategy rollouts (ICLR log entry 43).
#
#   watch -n 1 bash scripts/inference_oss/rollout_status.sh
#
# Reads only the output tree and the per-arm logs -- it starts nothing and touches
# no run state, so it is safe to leave running in another terminal.
#
# Env: OUT_ROOT (default /workspace/reasoning_theatre/rollout_strategies)
#      ARMS     (default "jlens_argmax_per_sentence jlens_top_k_global")
#      TOTAL    (default: line count of $OUT_ROOT/mass_l15_names.txt)

OUT_ROOT=${OUT_ROOT:-/workspace/reasoning_theatre/rollout_strategies}
ARMS=${ARMS:-"jlens_argmax_per_sentence jlens_top_k_global"}
NAMES_FILE=${NAMES_FILE:-$OUT_ROOT/mass_l15_names.txt}
TOTAL=${TOTAL:-$( [ -s "$NAMES_FILE" ] && wc -l < "$NAMES_FILE" || echo 3600 )}
STATE=${STATE:-$OUT_ROOT/.status_samples}
LOCAL_TZ=${LOCAL_TZ:-Europe/Berlin}

bold=$'\033[1m'; dim=$'\033[2m'; red=$'\033[31m'; grn=$'\033[32m'; ylw=$'\033[33m'; off=$'\033[0m'

now=$(date -u +%s)
printf '%s%s%s   %s UTC  /  %s\n' "$bold" "TRUNCATION ROLLOUTS" "$off" \
    "$(date -u -d @"$now" +'%Y-%m-%d %H:%M:%S')" "$(TZ=$LOCAL_TZ date -d @"$now" +'%H:%M:%S %Z')"
echo "$dim$OUT_ROOT$off"
echo

# ---- processes -------------------------------------------------------------
launcher=$(pgrep -f 'run_inference_strategies.sh' | head -1)
worker=$(pgrep -f 'run_inference.py --strategy' | head -1)
if [ -n "$worker" ]; then
    wstrat=$(tr '\0' ' ' < "/proc/$worker/cmdline" 2>/dev/null | sed -n 's/.*--strategy \([^ ]*\).*/\1/p')
    wpy=$(pgrep -f 'run_inference.py --strategy' | tail -1)
    wrss=$(awk '/VmRSS/{printf "%.1f GiB", $2/1048576}' "/proc/${wpy:-$worker}/status" 2>/dev/null)
    wsec=$(( now - $(date -u -d "$(ps -o lstart= -p "$worker")" +%s) ))
    printf '  worker   %sALIVE%s  pid %-8s %-28s rss %-10s up %s\n' "$grn" "$off" "$worker" "$wstrat" "$wrss" \
        "$(printf '%dh%02dm' $((wsec/3600)) $(((wsec%3600)/60)))"
else
    printf '  worker   %sDEAD%s   (no run_inference.py --strategy process)\n' "$red" "$off"
fi
if [ -n "$launcher" ]; then
    printf '  launcher %sALIVE%s  pid %s\n' "$grn" "$off" "$launcher"
else
    printf '  launcher %sDEAD%s\n' "$red" "$off"
fi

# ---- gpu -------------------------------------------------------------------
gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
        --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -n "$gpu" ]; then
    IFS=', ' read -r gu gm gt gtemp gpw <<< "$gpu"
    printf '  gpu      util %3s%%   mem %5s / %5s MiB   %sC   %sW\n' "$gu" "$gm" "$gt" "$gtemp" "$gpw"
fi
echo

# ---- per-arm progress ------------------------------------------------------
# ETA is WORK-weighted, not count-weighted. The names file is sorted, so the run walks
# size11 comp0.0 (short reasoning, few cutoffs) first and the heavy trajectories last -- a
# count-based ETA read 23:26 CEST while only 1.7% of the actual work was done. WORK_TABLE
# holds cumulative prompt tokens in processing order, one line per position, per arm.
WORK_TABLE=${WORK_TABLE:-$OUT_ROOT/.work_by_position.txt}
declare -A WCOL=( [jlens_argmax_per_sentence]=2 [jlens_top_k_global]=3 )
declare -A WTOT WCUM
if [ -s "$WORK_TABLE" ]; then
    read -r _ WTOT[jlens_argmax_per_sentence] WTOT[jlens_top_k_global] <<< "$(tail -1 "$WORK_TABLE")"
fi

printf '%s  %-28s %-9s %8s %7s  %-22s %-10s %s%s\n' "$bold" ARM STATE DONE 'WORK%' PROGRESS RATE "ETA (UTC / $LOCAL_TZ)" "$off"
touch "$STATE" 2>/dev/null
tmp_state=$(mktemp 2>/dev/null) || tmp_state=""
work_rate=""       # prompt tokens/second, measured on whichever arm is moving
chain_secs=0       # seconds of queued work ahead of the arm being printed

for arm in $ARMS; do
    dir=$OUT_ROOT/$arm
    log=$(ls -t "$dir"/logs/run_*.txt 2>/dev/null | head -1)
    done_n=$(ls -U "$dir" 2>/dev/null | grep -c '\.json$')
    pct=$(( TOTAL > 0 ? 100 * done_n / TOTAL : 0 ))

    # state
    if [ "$done_n" -ge "$TOTAL" ] && [ "$TOTAL" -gt 0 ]; then
        state="${grn}DONE${off}    "
    elif [ -n "$worker" ] && [ "$wstrat" = "$arm" ]; then
        state="${grn}RUNNING${off} "
    elif [ "$done_n" -gt 0 ]; then
        state="${ylw}STALLED${off} "
    else
        state="${dim}pending${off} "
    fi

    # work done so far = the cumulative-work table at this position (processing order)
    col=${WCOL[$arm]:-0}
    cum=0
    if [ -s "$WORK_TABLE" ] && [ "$done_n" -gt 0 ] && [ "$col" -gt 0 ]; then
        cum=$(awk -v n="$done_n" -v c="$col" '$1==n {print $c; exit}' "$WORK_TABLE")
    fi
    WCUM[$arm]=${cum:-0}

    # Two work rates. The 10-minute one is instantaneous and swings hard, because the run
    # walks the name list in size order and a block of size5 trajectories is ~45 files/min of
    # tiny prompts while size9 is a few files/min of long ones -- same GPU, half the tokens/s.
    # The ETA therefore uses the LIFETIME rate (this arm's work / its own elapsed generation
    # time), which averages over blocks; the windowed one is displayed as the current speed.
    echo "$now $arm $done_n ${cum:-0}" >> "$STATE"
    ref=$(awk -v a="$arm" -v t=$((now - 600)) '$2==a && $1>=t {print $1, $3, $4; exit}' "$STATE")
    rate=""; eta_u="-"; eta_l=""; win_rate=""
    if [ -n "$ref" ]; then
        set -- $ref
        dt=$(( now - $1 )); dn=$(( done_n - $2 )); dw=$(( ${cum:-0} - ${3:-0} ))
        if [ "$dt" -ge 45 ] && [ "$dn" -gt 0 ]; then
            rate=$(awk -v dn="$dn" -v dt="$dt" 'BEGIN{printf "%.1f/min", dn*60/dt}')
            [ "$dw" -gt 0 ] && win_rate=$(awk -v dw="$dw" -v dt="$dt" 'BEGIN{printf "%.0f", dw/dt}')
        fi
    fi
    # lifetime: cumulative work over elapsed generation time (log START + ~150 s model load)
    if [ -n "$log" ] && [ "${cum:-0}" -gt 0 ]; then
        st=$(sed -n 's/^=== START: \([0-9T:-]*\)Z .*/\1/p' "$log" | head -1)
        if [ -n "$st" ]; then
            st_e=$(date -u -d "${st}Z" +%s 2>/dev/null)
            gen=$(( now - ${st_e:-now} - 150 ))
            if [ "$gen" -gt 120 ]; then
                life_rate=$(awk -v w="$cum" -v g="$gen" 'BEGIN{printf "%.0f", w/g}')
                [ "${life_rate:-0}" -gt 0 ] && work_rate=$life_rate
            fi
        fi
    fi
    # The arms run SEQUENTIALLY, so a queued arm finishes after every arm ahead of it:
    # carry the running total rather than pricing each arm as if it started now.
    leftw=$(( ${WTOT[$arm]:-0} - ${cum:-0} ))
    if [ "$leftw" -gt 0 ] && [ -n "$work_rate" ] && [ "$work_rate" -gt 0 ]; then
        secs=$(( leftw / work_rate ))
        chain_secs=$(( chain_secs + secs ))
        eta_u=$(date -u -d "@$((now + chain_secs))" +'%m-%d %H:%M')
        eta_l=$(TZ=$LOCAL_TZ date -d "@$((now + chain_secs))" +'%H:%M')
    fi
    [ -n "$win_rate" ] && rate="$rate"

    [ -z "$rate" ] && rate="-"
    [ -n "$eta_l" ] && eta="$eta_u / $eta_l" || eta="$eta_u"
    # % is of WORK, not of files, for the same reason the ETA is
    if [ -n "${WTOT[$arm]:-}" ] && [ "${WTOT[$arm]:-0}" -gt 0 ]; then
        pct=$(( 100 * ${cum:-0} / ${WTOT[$arm]} ))
    fi

    filled=$(( pct / 5 )); bar=$(printf '%*s' "$filled" '' | tr ' ' '#')$(printf '%*s' $((20-filled)) '')
    printf '  %-28s %s %5d/%-4d %6s%%  [%s] %-10s %s\n' "$arm" "$state" "$done_n" "$TOTAL" "$pct" "$bar" "$rate" "$eta"

    # errors: hard failures first, then the script's own WARNING lines (skipped
    # trajectories), and never the benign `kernels` UserWarning at model load.
    if [ -n "$log" ]; then
        # A recovered OOM (the batch was halved and retried) is not a failure -- count it
        # apart from the fatal signatures, or the display cries wolf for hours.
        hard=$(grep -cE 'Traceback|RuntimeError|CUDA error|Killed|OutOfMemoryError' "$log" 2>/dev/null)
        oom=$(grep -cE 'WARNING: OOM on' "$log" 2>/dev/null)
        warn=$(grep -cE '^\s*WARNING:' "$log" 2>/dev/null)
        if [ "${hard:-0}" -gt 0 ]; then
            printf '      %sErrors: %s%s  oom-retries: %s  warnings: %s   %s\n' "$red" "$hard" "$off" \
                "${oom:-0}" "${warn:-0}" \
                "$dim$(grep -E 'Traceback|RuntimeError|CUDA error|Killed|OutOfMemoryError' "$log" | tail -1 | cut -c1-90)$off"
        else
            printf '      %sErrors: 0   oom-retries: %s   warnings: %s%s\n' "$dim" "${oom:-0}" "${warn:-0}" "$off"
        fi
    fi
done

# keep the sample file bounded
if [ -n "$tmp_state" ]; then tail -n 400 "$STATE" > "$tmp_state" 2>/dev/null && mv "$tmp_state" "$STATE" 2>/dev/null; fi

# ---- when everything is done -----------------------------------------------
# One work rate (prompt tokens/s) drives every arm: the arms differ in how much work they
# are, not in what a token costs.
if [ -n "$work_rate" ] && [ "$work_rate" -gt 0 ]; then
    if [ "$chain_secs" -gt 0 ]; then
        secs=$chain_secs
        printf '\n  %sALL ARMS DONE ~ %s UTC / %s %s%s   (%s prompt tok/s lifetime, %.1f h left)%s\n' "$bold" \
            "$(date -u -d "@$((now + secs))" +'%a %m-%d %H:%M')" \
            "$(TZ=$LOCAL_TZ date -d "@$((now + secs))" +'%a %H:%M')" \
            "$(TZ=$LOCAL_TZ date -d "@$((now + secs))" +'%Z')" "$off" \
            "$work_rate" "$(awk -v s="$secs" 'BEGIN{printf "%.1f", s/3600}')" ""
    fi
fi

# ---- tail of the active log ------------------------------------------------
active=""
for arm in $ARMS; do
    [ -n "$worker" ] && [ "$wstrat" = "$arm" ] && active=$OUT_ROOT/$arm
done
[ -z "$active" ] && active=$OUT_ROOT/$(echo $ARMS | awk '{print $NF}')
alog=$(ls -t "$active"/logs/run_*.txt 2>/dev/null | head -1)
if [ -n "$alog" ]; then
    echo
    echo "$dim--- $(basename "$alog") ---$off"
    tail -c 4000 "$alog" | tr '\r' '\n' | grep -v '^$' | tail -6 | cut -c1-160
fi
