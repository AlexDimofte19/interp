#!/usr/bin/env bash
# gpt-oss P2, step 6: the cross-selection matrix. Every local-belief probe is scored on every
# arm's val slice: 3 slices x 3 probes x {lr, mlp} = 18 cells.
#
# The gpt-oss twin of wrappers/qwen_analysis/3_train_and_eval_probes/cross_eval_local_belief_all_selections.sh.
#
# READ DOWN A COLUMN: within one slice every probe sees identical rows, so the differences
# are the probe weights and nothing else. Across rows the populations differ.
#
# THE DIAGONAL IS A CHECK against the trainer's FINAL balanced accuracy (not the best: the
# trainer keeps the last epoch's weights, so the best-epoch number belongs to discarded
# weights). A diagonal off by more than ~1 point means the split moved.
#
# THE SIGNAL VOCABULARY is gpt-oss's direction_tokens_full.json, the one the reused trees'
# sidecars name. The verbalization split in each cell is only meaningful against that.
#
# PARALLEL WITHIN A SLICE, SERIAL ACROSS: nearly all the time is MooseFS latency on .pt reads.
set -euo pipefail

REPO=/workspace/repo/interp

PROBES=/workspace/probes/gptoss_p2_local_belief
PREPARED=/workspace/prepared/gptoss_p2_local_belief    # reads ${PREPARED}_${slice}_val
OUT=/workspace/results/gptoss_p2_local_belief/cross_selection_eval

SIGNAL_JSON=/workspace/jlens/direction_tokens_full.json
SIGNAL_NAME=direction

LAYER=15
ARMS="jlens logitlens random"
SLICES="jlens logitlens random"
MODEL_TYPES="lr mlp"

cd "$REPO"
EVAL="uv run python telos_interp/loudness_analysis/rollouts/eval_local_belief.py"

# The val slices must be the RELABELLED ones: eval_local_belief.py falls back to
# `s.get("final_label", s["label"])`, so an unrelabelled slice silently reports local == final.
for slice in $SLICES; do
    d="${PREPARED}_${slice}_val"
    [ -f "$d/manifest.json" ] || { echo "!! missing slice: $d/manifest.json" >&2; exit 1; }
    grep -q '"final_label"' "$d/manifest.json" || { echo "!! $d is NOT relabelled (no final_label)" >&2; exit 1; }
done

failed=0
for slice in $SLICES; do
    echo "=== slice: ${slice} (${PREPARED}_${slice}_val) ==="
    mkdir -p "$OUT/$slice"
    pids=""
    for arm in $ARMS; do
        for mt in $MODEL_TYPES; do
            probe="${PROBES}/gptoss_p2_local_belief_${arm}_l${LAYER}_${mt}.pt"
            [ -f "$probe" ] || { echo "!! missing probe: $probe" >&2; failed=1; continue; }
            $EVAL "$probe" "${PREPARED}_${slice}_val" \
                  --signal-json "$SIGNAL_JSON" \
                  --signal-name "$SIGNAL_NAME" \
                  > "$OUT/$slice/${arm}_${mt}.txt" 2>&1 &
            pids="$pids $!"
        done
    done
    for pid in $pids; do
        wait "$pid" || failed=1
    done
done

echo
echo "=== balanced accuracy vs LOCAL belief, by slice (rows) and probe (columns) ==="
for slice in $SLICES; do
    for arm in $ARMS; do
        for mt in $MODEL_TYPES; do
            f="$OUT/$slice/${arm}_${mt}.txt"
            # `|| bal=` is load-bearing under `set -e` with pipefail: a cell that did not run
            # has no such line, and without it the summary dies on the first gap.
            bal=$(grep -m1 'bal acc vs LOCAL belief' "$f" 2>/dev/null | awk '{print $NF}') || bal=
            diag=$([ "$slice" = "$arm" ] && echo " <- diagonal" || echo "")
            printf '  slice=%-10s probe=%-10s %-3s  %s%s\n' "$slice" "$arm" "$mt" "${bal:-MISSING}" "$diag"
        done
    done
done

echo
if [ "$failed" -ne 0 ]; then
    echo "!! at least one cell failed -- check the .txt files under $OUT" >&2
    exit 1
fi
echo "done -> $OUT/<slice>/<probe-arm>_<model-type>.txt"
