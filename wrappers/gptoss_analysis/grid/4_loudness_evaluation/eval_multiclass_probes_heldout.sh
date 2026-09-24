#!/usr/bin/env bash
# gpt-oss P2 GRID: score each MULTICLASS grid probe on every held-out token (heldout360), one JSON per probe.
#
# The multiclass twin of ./eval_binary_probes_heldout.sh: same prepared held-out manifest (so the
# same tokens and the same 25 cells per step), telos_interp/loudness_analysis/eval_grid_probe.py in
# place of eval_binary_cognitive_map_probe. Its JSON is what cell 2 of
# ./probe_accuracy_by_loudness_decile_multiclass.ipynb must reproduce from the per-token table.
#
# Balanced accuracy is reported twice: the trainer's (mean recall over all five classes, padding
# included) and without padding (A # G _ only). The notebook reads the second.
#
# Resumable: a probe whose JSON exists is skipped.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../grid_layer.sh"   # GRID_LAYER

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)   # repo root

PROBES=/workspace/probes/gptoss_p2_grid
DATA=/workspace/prepared/gptoss_p2_grid_heldout   # built by prepare_heldout_grid.sh
OUT=/workspace/results/gptoss_p2_grid/heldout_multiclass
SIGNAL_JSON=$REPO/data/jlens/grid_tokens_pruned.json
LAYER=$GRID_LAYER
ARMS="jlens logitlens random"
MODEL_TYPES=${MODEL_TYPES:-"lr mlp"}
DEVICE=cuda

[ -f "$DATA/manifest.json" ] || { echo "!! missing $DATA/manifest.json -- run prepare_heldout_grid.sh first" >&2; exit 1; }
mkdir -p "$OUT"
cd "$REPO"

failed=0
for arm in $ARMS; do
    for mt in $MODEL_TYPES; do
        probe="${PROBES}/gptoss_p2_grid_${arm}_multiclass_l${LAYER}_${mt}.pt"
        res="$OUT/${arm}_multiclass_${mt}.json"
        [ -f "$probe" ] || { echo "!! missing probe: $probe" >&2; failed=1; continue; }
        [ -f "$res" ] && { echo "- have $(basename "$res"), skipping"; continue; }
        uv run python telos_interp/loudness_analysis/eval_grid_probe.py "$probe" "$DATA" \
            --signal-json "$SIGNAL_JSON" --signal-name grid \
            --cache-activations --device "$DEVICE" \
            --out-json "$res" \
            > "$OUT/${arm}_multiclass_${mt}.txt" 2>&1 || failed=1
    done
done

echo
echo "=== heldout360: balanced accuracy, no padding (incl. padding) ==="
for mt in $MODEL_TYPES; do
    line="  multiclass/${mt}:"
    for arm in $ARMS; do
        f="$OUT/${arm}_multiclass_${mt}.json"
        v=$(uv run python -c "import json,sys; g=json.load(open(sys.argv[1]))['global']; print(f\"{g['balanced_accuracy_no_padding']:.4f} ({g['balanced_accuracy']:.4f})\")" "$f" 2>/dev/null) || v=MISSING
        line="$line  ${arm}=${v}"
    done
    echo "$line"
done
[ "$failed" -eq 0 ] || { echo "!! at least one probe failed -- see the .txt files under $OUT" >&2; exit 1; }
echo "done -> $OUT"
