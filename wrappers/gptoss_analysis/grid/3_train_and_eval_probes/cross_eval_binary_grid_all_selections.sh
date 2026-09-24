#!/usr/bin/env bash
# gpt-oss P2 GRID, step 6: the cross-selection matrix. Every binary grid probe is scored on
# every arm's val slice: 3 slices x 3 probe arms x 4 classes x {lr, mlp} = 72 cells.
#
# The grid twin of wrappers/qwen_analysis/3_train_and_eval_probes/cross_eval_local_belief_all_selections.sh.
# eval_binary_cognitive_map_probe replaces eval_local_belief.py, because the label is a cell,
# not an action.
#
# WHY A MATRIX AND NOT A SCOREBOARD. Each probe has only been scored on the tokens its OWN
# selection picked (the diagonal). That leaves "the jlens probe is better" confounded with
# "the jlens probe is specialised to jlens tokens". READ DOWN A COLUMN: within one slice
# every probe sees identical (token, cell) rows, because the cell draw is seeded per
# (trajectory, step), not per arm. Across rows nothing is comparable.
#
# THE DIAGONAL IS A CHECK. Cell (jlens slice, jlens probe) must reproduce what the trainer
# reported for that probe on its own val half. Compare against the FINAL epoch's number, not
# the best one: the trainer keeps the last weights, not the best, so the high-water mark
# belongs to weights that were discarded. If the diagonal does not reproduce, the split moved
# under the probe and no other cell means anything.
#
# READ BALANCED ACCURACY AND AUROC. The positive class is ~26% of cells for empty and ~1% for
# agent and goal, so plain accuracy pays a probe for never firing.
#
# SERIAL. Each eval reads its slice through --cache-activations, the pack the trainer wrote
# beside the val manifest. That makes every cell seconds of GPU time, so there is nothing to
# gain by running them concurrently.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../grid_layer.sh"   # GRID_LAYER

REPO=/workspace/repo/interp

PROBES=/workspace/probes/gptoss_p2_grid
PREP=/workspace/prepared/gptoss_p2_grid       # reads ${PREP}_${slice}_val
OUT=/workspace/results/gptoss_p2_grid/cross_selection_eval

LAYER=$GRID_LAYER
ARMS="jlens logitlens random"
SLICES="jlens logitlens random"
CLASSES="empty wall agent goal"
MODEL_TYPES="lr mlp"
THRESHOLD=0.5
DEVICE=cuda

cd "$REPO"

for slice in $SLICES; do
    d="${PREP}_${slice}_val"
    [ -f "$d/manifest.json" ] || { echo "!! missing slice: $d/manifest.json" >&2; exit 1; }
done

failed=0
for slice in $SLICES; do
    echo "=== slice: ${slice} (${PREP}_${slice}_val) ==="
    mkdir -p "$OUT/$slice"
    for arm in $ARMS; do
        for cls in $CLASSES; do
            for mt in $MODEL_TYPES; do
                probe="${PROBES}/gptoss_p2_grid_${arm}_${cls}_l${LAYER}_${mt}.pt"
                [ -f "$probe" ] || { echo "!! missing probe: $probe" >&2; failed=1; continue; }
                uv run interp-cli eval_binary_cognitive_map_probe \
                    --probe-path "$probe" \
                    --data-path "${PREP}_${slice}_val" \
                    --output-path "$OUT/$slice/${arm}_${cls}_${mt}.json" \
                    --threshold "$THRESHOLD" \
                    --cache-activations \
                    --device "$DEVICE" \
                    > "$OUT/$slice/${arm}_${cls}_${mt}.txt" 2>&1 || failed=1
            done
        done
    done
done

# One line per cell, grouped by class then slice: this is the matrix the cells are there to make.
echo
echo "=== balanced accuracy / AUROC, by class, slice (rows) and probe (columns) ==="
uv run python - "$OUT" "$SLICES" "$ARMS" "$CLASSES" "$MODEL_TYPES" <<'EOF'
import json, sys
from pathlib import Path

out, slices, arms, classes, mts = sys.argv[1], *(s.split() for s in sys.argv[2:])
for cls in classes:
    print(f"\n  class={cls}")
    for mt in mts:
        for sl in slices:
            cells = []
            for arm in arms:
                f = Path(out) / sl / f"{arm}_{cls}_{mt}.json"
                if not f.exists():
                    cells.append(f"{arm}=MISSING")
                    continue
                g = json.loads(f.read_text())["global"]
                mark = "*" if arm == sl else " "
                cells.append(f"{arm}{mark}={g['balanced_accuracy']:.4f}/{g['auroc']:.4f}")
            print(f"    {mt:<3} slice={sl:<10} " + "  ".join(cells))
print("\n  (* = diagonal; each cell is balanced_accuracy/auroc)")
EOF

echo
if [ "$failed" -ne 0 ]; then
    echo "!! at least one cell failed -- check the .txt files under $OUT" >&2
    exit 1
fi
echo "done -> $OUT/<slice>/<probe-arm>_<class>_<model-type>.json"
