#!/usr/bin/env bash
# Qwen P2 GRID, loudness evaluation step 2: every binary grid probe on the held-out 72.
#
# 3 arms x 4 classes x {lr, mlp} = 24 JSONs, each scored on the same (token, cell) rows of
# prepare_heldout_grid.sh's dataset. This is the one population no lens chose, so it is the
# only row of the comparison that speaks to tokens at large. Read down the arms within one
# class.
#
# WHAT THIS DOES NOT DO: bin by loudness. The direction line's stage 4
# (score_probes_per_token.py -> join -> decile notebook) needs a per-token scorer for these
# probes, and the pipeline does not have one. probes.py::GridTileProbeType loads
# CognitiveMapProbe, which cannot read a one-vs-rest probe's 2-wide head as 8 classes. The
# fix is a registry entry (a binary grid probe type in loudness_analysis/probes.py), not a
# new script. Until it exists, this file gives the overall held-out number per probe.
#
# --cache-activations: the first probe packs the 1.17M held-out tensors beside the manifest
# (~6.5 GB). The other 23 read the pack instead of MooseFS.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)   # repo root

PROBES=/workspace/probes/qwen_p2_grid
DATA=/workspace/prepared/qwen_p2_grid_heldout
OUT=/workspace/results/qwen_p2_grid/heldout

LAYER=27
ARMS="jlens logitlens random"
CLASSES="empty wall agent goal"
MODEL_TYPES="lr mlp"
THRESHOLD=0.5
DEVICE=cuda

[ -f "$DATA/manifest.json" ] || { echo "!! missing $DATA/manifest.json -- run prepare_heldout_grid.sh first" >&2; exit 1; }

mkdir -p "$OUT"
cd "$REPO"

failed=0
for cls in $CLASSES; do
    for arm in $ARMS; do
        for mt in $MODEL_TYPES; do
            probe="${PROBES}/qwen_p2_grid_${arm}_${cls}_l${LAYER}_${mt}.pt"
            res="$OUT/${arm}_${cls}_${mt}.json"
            [ -f "$probe" ] || { echo "!! missing probe: $probe" >&2; failed=1; continue; }
            [ -f "$res" ] && { echo "- have $(basename "$res"), skipping"; continue; }
            uv run interp-cli eval_binary_cognitive_map_probe \
                --probe-path "$probe" \
                --data-path "$DATA" \
                --output-path "$res" \
                --threshold "$THRESHOLD" \
                --cache-activations \
                --device "$DEVICE" \
                > "$OUT/${arm}_${cls}_${mt}.txt" 2>&1 || failed=1
        done
    done
done

echo
echo "=== held-out 72: balanced accuracy / AUROC ==="
for cls in $CLASSES; do
    for mt in $MODEL_TYPES; do
        line="  ${cls}/${mt}:"
        for arm in $ARMS; do
            f="$OUT/${arm}_${cls}_${mt}.json"
            v=$(uv run python -c "import json,sys; g=json.load(open(sys.argv[1]))['global']; print(f\"{g['balanced_accuracy']:.4f}/{g['auroc']:.4f}\")" "$f" 2>/dev/null) || v=MISSING
            line="$line  ${arm}=${v}"
        done
        echo "$line"
    done
done

[ "$failed" -eq 0 ] || { echo "!! at least one probe failed -- see the .txt files under $OUT" >&2; exit 1; }
echo "done -> $OUT"
