#!/usr/bin/env bash
# Qwen P2, step 5: train one next_action probe per arm per model type, on the LOCAL BELIEF.
#
# The sibling of ../2_dataset_creation/build_local_belief_datasets.sh -- it consumes the six
# RELABELLED manifests that master's rollout stage wrote, so keep OUT below equal to its OUT.
#
# THE LABEL IS THE LOCAL BELIEF, NOT THE FINAL ACTION, and the only thing that decides which
# is the directory OUT names. /workspace/prepared/qwen_p2_* are the manifests as prepared,
# labelled with the trajectory's final agent_action; /workspace/prepared/qwen_p2_local_belief_* are
# those same manifests after relabel_manifest_from_rollout.py swapped in what the model
# answered when its reasoning was cut at that token. Both are valid v3 manifests of the same
# shape and the trainer cannot tell them apart, so pointing OUT at the wrong one trains the
# wrong probe and reports a perfectly healthy accuracy for it. The guard below is there
# because that is not a failure anything downstream would surface.
#
# WHAT EACH ARM REPORTS, AND WHAT IT DOES NOT. Every train command scores its probe on the
# VAL half of its OWN selection: the jlens probe on jlens-selected tokens, the control on
# uniformly drawn ones. Three arms therefore report three numbers measured on three
# different populations, which is a comparison of arms and NOT a statement about the
# model's tokens at large. Reading all three on one shared population -- every reasoning
# token of the held-out 72 -- is a separate script; on the gpt-oss line the ordering
# INVERTED between the two (ICLR entry 49), so neither substitutes for the other.
#
# WHY THE CONTROL IS HERE AT ALL. The loudest tokens are largely the direction words the
# model has already verbalized, so a lens arm on its own says nothing. `random` is what
# reports whether the lens found anything a uniform draw of the same size would not have.
#
# NO SPLIT, NO --eval-split. --eval-data-path is a separate gather (mass_eval_144), so the
# trainer's internal row-level split never runs and no trajectory can land in both halves.
#
# BOTH MODEL TYPES, IN ONE RUN. This was a single MODEL_TYPE with a comment saying to flip it
# to mlp and run again, which is how the sweep came to be half-built: the consumer next door,
# cross_eval_local_belief_all_selections.sh, hardcodes "lr mlp" and needs all 18 probes to
# fill its matrix. It now loops, as scripts/train_next_action_arms.sh (the gpt-oss original)
# always did. MODEL_TYPES and ARMS are overridable so a half-finished sweep can be resumed
# without retraining what is already on disk -- MODEL_TYPES=mlp ./train_...sh -- and the probe
# filenames carry the arm and the type, so a resume adds files rather than overwriting.
#
# WHICH IS WHY --cache-activations IS ON. A next_action manifest copies no tensors -- it names
# each .pt in the gather tree individually -- so the FIRST pass opens 107,610 small files over
# MooseFS at ~50/s: ~36 minutes of load with the GPU idle, against seconds of actual training.
# The cache packs (activations, labels) beside each manifest on first use, ~880 MB for all six,
# so every pass after the first pays nothing. It is keyed on a fingerprint of the manifest and
# rebuilds when that moves; same tensors, same order, same NaN filtering, so a cached run and
# an uncached one train on identical data.
set -euo pipefail

REPO=/workspace/repo/interp

OUT=/workspace/prepared/qwen_p2_local_belief  # reads ${OUT}_${arm}_{train,val}; the RELABELLED ones
PROBES=/workspace/probes/qwen_p2_local_belief
LOGS=$PROBES/logs

LAYER=27
ARMS=${ARMS:-"jlens logitlens random"}
MODEL_TYPES=${MODEL_TYPES:-"lr mlp"}

# Probe hyperparameters: the general_probe_train.sh values, with the batch size lowered
# because a selection is ~60 rows per trajectory, not ~C cells per trajectory. Pinned, not
# overridable: this file is the record of what produced the published probes.
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
# straight out of prepare has no such key. That is the only on-disk difference between a
# local-belief dataset and a final-action one, so it is what is checked.
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
            --output-path "${PROBES}/qwen_p2_local_belief_${arm}_l${LAYER}_${mt}.pt" \
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
echo "done -> $PROBES/qwen_p2_local_belief_<arm>_l${LAYER}_<type>.pt   logs -> $LOGS"
