#!/usr/bin/env bash
# Qwen P2 GRID, step 5: train one BINARY grid-cell probe per arm x cell class x model type.
#
# The grid twin of ../../3_train_and_eval_probes/train_next_action_probes_all_selections.sh.
# It reads the grid_tile manifests ../2_dataset_creation/build_grid_datasets.sh writes, and
# trains train_binary_cognitive_map_probe instead of train_next_action_probe. So a grid arm
# and a direction arm differ in the vocabulary that ranked the tokens and in the label.
#
# ONE-VS-REST, FOUR CLASSES. empty ('_'), wall ('#'), agent ('A'), goal ('G'), the four of
# grid_cell_analysis/binary_grid_probes.sh with its hyperparameters. The arms differ only in
# their tokens; within an arm, the probes differ only in which symbol counts as positive.
# A and G are one cell per grid, so at 25 drawn cells they are ~1% of the rows. Read those two
# through balanced accuracy and AUROC, never raw accuracy.
#
# WHAT EACH ARM REPORTS, AND WHAT IT DOES NOT. Each probe is scored on the VAL half of its OWN
# selection, so three arms report numbers on three different populations. That is a comparison
# of arms, not a statement about tokens at large. The matrix beside this file fixes the slice
# and varies the probe. ../4_loudness_evaluation scores every probe on every held-out token.
#
# NO SPLIT: --eval-data-path is a separate gather (mass_eval_144), so no trajectory is in both
# halves and the trainer's internal row-level split never runs.
#
# --cache-activations packs (activations, labels) beside each manifest on first use. A
# token-major manifest names one .pt per sample over MooseFS, so the first pass is file-open
# bound and every later pass (the other classes, the other head, the cross-eval) is not.
# Classes are therefore the INNER loop: an arm's dataset is loaded once and reused eight times.
#
# Resumable: a probe already on disk is skipped. ARMS / CLASSES / MODEL_TYPES are overridable
# to resume a partial sweep; the filenames carry all three.
set -euo pipefail

REPO=/workspace/repo/interp

PREP=/workspace/prepared/qwen_p2_grid   # reads ${PREP}_${arm}_{train,val}
PROBES=/workspace/probes/qwen_p2_grid
LOGS=$PROBES/logs

LAYER=27
ARMS=${ARMS:-"jlens logitlens random"}
CLASSES=${CLASSES:-"empty wall agent goal"}   # aliases for _  #  A  G
MODEL_TYPES=${MODEL_TYPES:-"lr mlp"}

# Probe hyperparameters: grid_cell_analysis/binary_grid_probes.sh's, so these sit beside the
# existing gpt-oss binary probes rather than beside the next_action ones. Pinned, not
# overridable: this file is the record of what produced the probes.
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

echo "arms: $ARMS   classes: $CLASSES   model types: $MODEL_TYPES   layer: $LAYER   device: $DEVICE"

for arm in $ARMS; do
    for cls in $CLASSES; do
        for mt in $MODEL_TYPES; do
            out="${PROBES}/qwen_p2_grid_${arm}_${cls}_l${LAYER}_${mt}.pt"
            if [ -f "$out" ]; then
                echo "- have $(basename "$out"), skipping"
                continue
            fi
            echo
            echo "=== ${arm} / ${cls} / ${mt} ==============================================="
            uv run interp-cli train_binary_cognitive_map_probe \
                --train-data-path "${PREP}_${arm}_train" \
                --eval-data-path "${PREP}_${arm}_val" \
                --positive-class "$cls" \
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
                2>&1 | tee "$LOGS/${arm}_${cls}_l${LAYER}_${mt}.txt"
        done
    done
done

echo
echo "done -> $PROBES/qwen_p2_grid_<arm>_<class>_l${LAYER}_<type>.pt   logs -> $LOGS"
