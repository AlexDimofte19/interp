#!/usr/bin/env bash
# gpt-oss P2, step 5: train one next_action probe per arm per model type, on the LOCAL BELIEF.
#
# The gpt-oss twin of wrappers/qwen_analysis/3_train_and_eval_probes/train_next_action_probes_all_selections.sh.
# It consumes the six RELABELLED manifests ../2_dataset_creation/build_local_belief_datasets.sh
# writes, so keep OUT equal to that master's rollout OUT.
#
# THE LABEL IS THE LOCAL BELIEF, NOT THE FINAL ACTION, and the only thing that decides which is
# the directory OUT names. /workspace/prepared/gptoss_p2_* are the manifests as prepared;
# /workspace/prepared/gptoss_p2_local_belief_* are the same manifests after the rollout swapped
# in what the model answered when cut at that token. The trainer cannot tell them apart, so
# the guard below checks for `final_label`.
#
# WHAT EACH ARM REPORTS, AND WHAT IT DOES NOT. Each probe is scored on the VAL half of its OWN
# selection, so three arms report three different populations. The cross-selection matrix
# beside this file and the held-out scoring in ../4_loudness_evaluation are what compare them
# on shared rows. On gpt-oss the ordering INVERTED between the two (ICLR entry 49).
#
# NO SPLIT, NO --eval-split: --eval-data-path is the mass-era 720, disjoint from the 2,880 by
# construction (scripts/verify_mass_era_split.py).
#
# BOTH MODEL TYPES, IN ONE RUN. MODEL_TYPES / ARMS are overridable to resume a half-built sweep.
#
# --cache-activations packs each manifest's tensors on first use. The first pass opens one .pt
# per sample over MooseFS; every later pass reads the pack.
set -euo pipefail

REPO=/workspace/repo/interp

OUT=/workspace/prepared/gptoss_p2_local_belief  # reads ${OUT}_${arm}_{train,val}; the RELABELLED ones
PROBES=/workspace/probes/gptoss_p2_local_belief
LOGS=$PROBES/logs

LAYER=15
ARMS=${ARMS:-"jlens logitlens random"}
MODEL_TYPES=${MODEL_TYPES:-"lr mlp"}

# Probe hyperparameters: identical to the Qwen P2 record, so the two models differ in the
# model and nothing else. Pinned, not overridable.
HIDDEN_DIMS=1024
LEARNING_RATE=3e-4
WEIGHT_DECAY=0.001
NUM_EPOCHS=50
BATCH_SIZE=512
SEED=42
DEVICE=cuda

mkdir -p "$PROBES" "$LOGS"
cd "$REPO"

# A relabelled manifest keeps the original label as `final_label` on every sample; a manifest
# straight out of prepare has no such key.
for arm in $ARMS; do
    for half in train val; do
        d="${OUT}_${arm}_${half}"
        [ -f "$d/manifest.json" ] || { echo "!! missing manifest: $d/manifest.json -- run ../2_dataset_creation/build_local_belief_datasets.sh first" >&2; exit 1; }
        grep -q '"final_label"' "$d/manifest.json" || { echo "!! $d is NOT relabelled (no final_label) -- it carries the final action, not the local belief" >&2; exit 1; }
    done
done

echo "arms: $ARMS   model types: $MODEL_TYPES   layer: $LAYER   device: $DEVICE"

for arm in $ARMS; do
    for mt in $MODEL_TYPES; do
        echo
        echo "=== ${arm} / ${mt} ==============================================="
        uv run interp-cli train_next_action_probe \
            --train-data-path "${OUT}_${arm}_train" \
            --eval-data-path "${OUT}_${arm}_val" \
            --output-path "${PROBES}/gptoss_p2_local_belief_${arm}_l${LAYER}_${mt}.pt" \
            --model-type "$mt" \
            --hidden-dims "$HIDDEN_DIMS" \
            --learning-rate "$LEARNING_RATE" \
            --weight-decay "$WEIGHT_DECAY" \
            --num-epochs "$NUM_EPOCHS" \
            --batch-size "$BATCH_SIZE" \
            --class-weight balanced \
            --normalize \
            --seed "$SEED" \
            --device "$DEVICE" \
            --cache-activations \
            --verbose \
            | tee "$LOGS/${arm}_l${LAYER}_${mt}.txt"
    done
done

echo
echo "done -> $PROBES/gptoss_p2_local_belief_<arm>_l${LAYER}_<type>.pt   logs -> $LOGS"
