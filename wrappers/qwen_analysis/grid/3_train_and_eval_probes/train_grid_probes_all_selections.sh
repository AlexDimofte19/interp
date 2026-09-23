#!/usr/bin/env bash
# Qwen P2 GRID, step 5b: train one MULTICLASS grid-cell probe per arm x model type.
#
# The multiclass twin of ./train_binary_grid_probes_all_selections.sh: same manifests
# (${PREP}_${arm}_{train,val} from ../2_dataset_creation/build_grid_datasets.sh), same layer,
# same hyperparameters, same seed. The one change is train_cognitive_map_probe in place of
# train_binary_cognitive_map_probe, so ONE probe reads every cell symbol at once instead of
# four one-vs-rest probes. Any difference from the binary sweep is the head and nothing else.
#
# THE CLASSES ARE WHATEVER THE DATA HOLDS. The trainer remaps the symbols it finds
# (grid_utils.CELL_SYMBOL_TO_ID) to contiguous indices: A # G _ and, because every grid is
# padded to 15, '+' padding as well. Padding is an easy class and most of the drawn cells
# in a small grid, so read per-class recall and balanced accuracy, never raw accuracy.
# --class-weight balanced is what keeps A and G (~1% of rows each) from being ignored.
#
# WHAT EACH ARM REPORTS, AND WHAT IT DOES NOT. As in the binary sweep, each probe is scored
# on the VAL half of its OWN selection: a comparison of arms, not a statement about tokens
# at large.
#
# NO SPLIT: --eval-data-path is a separate gather (mass_eval_144), so no trajectory is in both
# halves. (The trainer refuses an internal --eval-split on these token-major manifests anyway.)
#
# CAVEAT, 2026-09-23: every Qwen lens output applied the final RMSNorm as `* w`, not
# `* (1 + w)` (claude_session_readme.md, last section). The jlens and logitlens arms were
# therefore SELECTED with the wrong-norm lens. Labels and the random arm are unaffected.
#
# --cache-activations reuses the pack the binary sweep already wrote beside each manifest
# (fingerprinted, identical tensors), so no arm pays the MooseFS file-open cost again.
#
# Resumable: a probe already on disk is skipped. ARMS / MODEL_TYPES are overridable.
set -euo pipefail

REPO=/workspace/repo/interp

PREP=/workspace/prepared/qwen_p2_grid   # reads ${PREP}_${arm}_{train,val}
PROBES=/workspace/probes/qwen_p2_grid
LOGS=$PROBES/logs

LAYER=27
ARMS=${ARMS:-"jlens logitlens random"}
MODEL_TYPES=${MODEL_TYPES:-"lr mlp"}

# Probe hyperparameters: identical to the binary sweep beside this file, which took them
# from grid_cell_analysis/binary_grid_probes.sh. Pinned, not overridable: this file is the
# record of what produced the probes.
HIDDEN_DIMS=1024
LEARNING_RATE=3e-4
WEIGHT_DECAY=0.001
DROPOUT=0.0
NUM_EPOCHS=50
BATCH_SIZE=2048        # rows are (token, cell) pairs: 25 per token, not 1
SEED=42
DEVICE=cuda

mkdir -p "$PROBES" "$LOGS"
cd "$REPO"

for arm in $ARMS; do
    for half in train val; do
        d="${PREP}_${arm}_${half}"
        [ -f "$d/manifest.json" ] || { echo "!! missing manifest: $d/manifest.json -- run ../2_dataset_creation/build_grid_datasets.sh first" >&2; exit 1; }
        grep -q '"probe_type": "grid_tile"' "$d/manifest.json" || { echo "!! $d is not a grid_tile manifest" >&2; exit 1; }
    done
done

echo "arms: $ARMS   model types: $MODEL_TYPES   layer: $LAYER   device: $DEVICE   (multiclass)"

for arm in $ARMS; do
    for mt in $MODEL_TYPES; do
        out="${PROBES}/qwen_p2_grid_${arm}_multiclass_l${LAYER}_${mt}.pt"
        if [ -f "$out" ]; then
            echo "- have $(basename "$out"), skipping"
            continue
        fi
        echo
        echo "=== ${arm} / multiclass / ${mt} ==============================================="
        uv run interp-cli train_cognitive_map_probe \
            --train-data-path "${PREP}_${arm}_train" \
            --eval-data-path "${PREP}_${arm}_val" \
            --model-type "$mt" \
            --output-path "$out" \
            --hidden-dims "$HIDDEN_DIMS" \
            --learning-rate "$LEARNING_RATE" \
            --weight-decay "$WEIGHT_DECAY" \
            --dropout "$DROPOUT" \
            --num-epochs "$NUM_EPOCHS" \
            --batch-size "$BATCH_SIZE" \
            --class-weight balanced \
            --normalize \
            --seed "$SEED" \
            --device "$DEVICE" \
            --cache-activations \
            --verbose \
            2>&1 | tee "$LOGS/${arm}_multiclass_l${LAYER}_${mt}.txt"
    done
done

echo
echo "done -> $PROBES/qwen_p2_grid_<arm>_multiclass_l${LAYER}_<type>.pt   logs -> $LOGS"
