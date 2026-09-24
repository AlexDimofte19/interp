#!/usr/bin/env bash
# gpt-oss P2 GRID, step 6b: the cross-selection matrix for the MULTICLASS grid probes -- every
# probe from ./train_grid_probes_all_selections.sh on every arm's val slice.
# 3 slices x 3 probes x {lr, mlp} = 18 cells.
#
# The gpt-oss twin of wrappers/qwen_analysis/grid/3_train_and_eval_probes/cross_eval_grid_all_selections.sh,
# cell for cell (probes at layer 14, gpt-oss's pruned grid vocabulary). As there: telos_interp/loudness_analysis/eval_grid_probe.py replaces eval_local_belief.py
# because the label is a cell, not an action. (The binary probes have their own matrix,
# ./cross_eval_binary_grid_all_selections.sh.)
#
# WHY A MATRIX AND NOT A SCOREBOARD. Each probe has only ever been scored on the tokens its
# OWN selection picked -- the diagonal of this matrix -- which leaves "the jlens probe is
# better" confounded with "the jlens probe is specialised to jlens tokens". Fixing the test
# slice and varying the probe separates them. READ DOWN A COLUMN: within one slice every
# probe sees identical (token, cell) rows -- the cell draw is seeded per (trajectory, step),
# not per arm -- so the differences are the probe weights and nothing else. Across rows
# nothing is comparable, because the populations differ.
#
# THE DIAGONAL IS A CHECK. Cell (jlens slice, jlens probe) must reproduce the FINAL balanced
# accuracy train_cognitive_map_probe reported for that probe (not the best epoch's: the saved
# weights are the last ones). If it does not reproduce, the split moved under the probe and
# no other cell means anything either.
#
# READ THE BALANCED ACCURACY, and read it WITHOUT PADDING too. The first number is the
# trainer's (mean recall over all five classes); it counts '+' padding, which is an easy class
# and most of a small grid's cells. The no-padding number averages A # G _ only.
#
# THE SIGNAL VOCABULARY IS GPT-OSS'S PRUNED GRID ONE, the vocabulary the selection used. The split at
# the bottom of each cell -- is the token itself a grid word? -- means nothing against another.
#
# PARALLEL WITHIN A SLICE, SERIAL ACROSS, as in the template. Every cell reads its slice
# through --cache-activations, the pack the trainer wrote beside each val manifest.
set -euo pipefail

REPO=/workspace/repo/interp

PROBES=/workspace/probes/gptoss_p2_grid
PREPARED=/workspace/prepared/gptoss_p2_grid          # reads ${PREPARED}_${slice}_val
OUT=/workspace/results/gptoss_p2_grid/cross_selection_eval_multiclass

SIGNAL_JSON=/workspace/repo/interp/data/jlens/grid_tokens_pruned.json
SIGNAL_NAME=grid

LAYER=14
ARMS="jlens logitlens random"
SLICES="jlens logitlens random"
MODEL_TYPES="lr mlp"

cd "$REPO"
EVAL="uv run python telos_interp/loudness_analysis/eval_grid_probe.py"

for slice in $SLICES; do
    d="${PREPARED}_${slice}_val"
    [ -f "$d/manifest.json" ] || { echo "!! missing slice: $d/manifest.json" >&2; exit 1; }
    grep -q '"probe_type": "grid_tile"' "$d/manifest.json" || { echo "!! $d is not a grid_tile manifest" >&2; exit 1; }
done

failed=0
for slice in $SLICES; do
    echo "=== slice: ${slice} (${PREPARED}_${slice}_val) ==="
    mkdir -p "$OUT/$slice"
    pids=""
    for arm in $ARMS; do
        for mt in $MODEL_TYPES; do
            probe="${PROBES}/gptoss_p2_grid_${arm}_multiclass_l${LAYER}_${mt}.pt"
            [ -f "$probe" ] || { echo "!! missing probe: $probe" >&2; failed=1; continue; }
            $EVAL "$probe" "${PREPARED}_${slice}_val" \
                  --signal-json "$SIGNAL_JSON" \
                  --signal-name "$SIGNAL_NAME" \
                  --cache-activations \
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
echo "=== balanced accuracy (incl. padding / no padding), by slice (rows) and probe (columns) ==="
for slice in $SLICES; do
    for arm in $ARMS; do
        for mt in $MODEL_TYPES; do
            f="$OUT/$slice/${arm}_${mt}.txt"
            # `|| x=` is load-bearing under set -e + pipefail; see the template.
            bal=$(grep -m1 'bal acc (trainer' "$f" 2>/dev/null | awk '{print $NF}') || bal=
            nop=$(grep -m1 'bal acc (no padding)' "$f" 2>/dev/null | awk '{print $NF}') || nop=
            diag=$([ "$slice" = "$arm" ] && echo " <- diagonal" || echo "")
            printf '  slice=%-10s probe=%-10s %-3s  %s / %s%s\n' "$slice" "$arm" "$mt" "${bal:-MISSING}" "${nop:-MISSING}" "$diag"
        done
    done
done

echo
if [ "$failed" -ne 0 ]; then
    echo "!! at least one cell failed -- check the .txt files under $OUT" >&2
    exit 1
fi
echo "done -> $OUT/<slice>/<probe-arm>_<model-type>.txt"
