#!/usr/bin/env bash
# Qwen P2, step 6: the cross-selection matrix -- every local-belief probe on every arm's val
# slice. 3 slices x 3 probes x {lr, mlp} = 18 cells.
#
# WHY A MATRIX AND NOT A SCOREBOARD. Each probe has only ever been scored on the tokens its
# OWN selection picked -- the diagonal of this matrix -- which leaves "the jlens probe is
# better" confounded with "the jlens probe is specialised to jlens tokens". Fixing the test
# slice and varying the probe separates them. READ DOWN A COLUMN: within one slice every
# probe sees identical rows, so the differences are the probe weights and nothing else.
# Across rows nothing is comparable, because the populations differ.
#
# THE DIAGONAL IS A CHECK. Cell (jlens slice, jlens probe) re-computes the number
# train_next_action_probes_all_selections.sh already reported for that probe, the same way
# (bal_acc is the statistic the checkpoints publish). If it does not reproduce, the split
# moved under the probe and no other cell means anything either.
#
# READ THE BALANCED ACCURACY, printed first. The local-belief label is imbalanced and the
# arms differ in exactly that prior, so plain accuracy pays a probe for leaning on the
# majority class and can move a column's ORDERING, not just its level.
#
# THE SIGNAL VOCABULARY IS QWEN'S. eval_local_belief.py defaults to the gpt-oss direction
# vocabulary, which would score Qwen tokens against strings from another tokenizer and
# quietly report that almost nothing is a direction word. The verbalization split at the
# bottom of each cell -- is the cut token itself a direction word the model already typed? --
# is only meaningful against the vocabulary the selection actually used.
#
# PARALLEL WITHIN A SLICE, SERIAL ACROSS. The six probes of one slice run concurrently and
# the script waits before starting the next, so the .pt reads of one slice are issued
# together while only one slice's tensors are in flight. Nearly all the time here is MooseFS
# latency on individual .pt reads, not compute.
set -euo pipefail

REPO=/workspace/repo/interp

PROBES=/workspace/probes/qwen_p2_local_belief
PREPARED=/workspace/prepared/qwen_p2_local_belief    # reads ${PREPARED}_${slice}_val
OUT=/workspace/results/qwen_p2_local_belief/cross_selection_eval

SIGNAL_JSON=/workspace/repo/interp/data/jlens/qwen/direction_tokens_full_qwen3-6-35b-a3b.json
SIGNAL_NAME=direction

LAYER=27
ARMS="jlens logitlens random"
SLICES="jlens logitlens random"
MODEL_TYPES="lr mlp"

cd "$REPO"
EVAL="uv run python telos_interp/loudness_analysis/rollouts/eval_local_belief.py"

# The val slices must be the RELABELLED ones. eval_local_belief.py falls back to
# `s.get("final_label", s["label"])`, so a slice straight out of prepare silently reports
# local == final on every row and the whole local-vs-final half of the matrix reads as a
# finding rather than an artefact.
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
            probe="${PROBES}/qwen_p2_local_belief_${arm}_l${LAYER}_${mt}.pt"
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

# One line per cell, grouped by slice: this is the matrix the cells are there to make.
echo
echo "=== balanced accuracy vs LOCAL belief, by slice (rows) and probe (columns) ==="
for slice in $SLICES; do
    for arm in $ARMS; do
        for mt in $MODEL_TYPES; do
            f="$OUT/$slice/${arm}_${mt}.txt"
            bal=$(grep -m1 'bal acc vs LOCAL belief' "$f" 2>/dev/null | awk '{print $NF}')
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
