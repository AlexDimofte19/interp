#!/usr/bin/env bash
# Qwen P2, step 5: train one next_action probe per arm, on the LOCAL BELIEF. Training only.
#
# The sibling of prepare_local_belief_probe_datasets.sh -- it consumes the six RELABELLED
# manifests that script's third stage wrote, so keep OUT below equal to its OUT.
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
# MODEL_TYPE is one variable on purpose: run it as lr, then flip to mlp and run again. The
# probe filenames carry the type, so the second pass adds files rather than overwriting.
set -euo pipefail

REPO=/workspace/repo/interp

OUT=/workspace/prepared/qwen_p2_local_belief  # reads ${OUT}_${arm}_{train,val}; the RELABELLED ones
PROBES=/workspace/probes/qwen_p2_local_belief

LAYER=27
MODEL_TYPE=lr          # flip to mlp and re-run for the second half of the sweep

# Probe hyperparameters: the general_probe_train.sh values, with the batch size lowered
# because a selection is ~60 rows per trajectory, not ~C cells per trajectory.
HIDDEN_DIMS=1024
LEARNING_RATE=3e-4
WEIGHT_DECAY=0.001
NUM_EPOCHS=50
BATCH_SIZE=512
SEED=42
DEVICE=cuda

mkdir -p "$PROBES"
cd "$REPO"

# A relabelled manifest keeps the original label as `final_label` on every sample; a manifest
# straight out of prepare has no such key. That is the only on-disk difference between a
# local-belief dataset and a final-action one, so it is what is checked.
for d in "${OUT}_jlens_train" "${OUT}_jlens_val" \
         "${OUT}_logitlens_train" "${OUT}_logitlens_val" \
         "${OUT}_random_train" "${OUT}_random_val"; do
    [ -f "$d/manifest.json" ] || { echo "!! missing manifest: $d/manifest.json -- run prepare_local_belief_probe_datasets.sh first" >&2; exit 1; }
    grep -q '"final_label"' "$d/manifest.json" || { echo "!! $d is NOT relabelled (no final_label) -- it carries the final action, not the local belief" >&2; exit 1; }
done

# ------------------------------------------------------------------- jlens
uv run interp-cli train_next_action_probe \
    --train-data-path "${OUT}_jlens_train" \
    --eval-data-path "${OUT}_jlens_val" \
    --output-path "${PROBES}/qwen_p2_local_belief_jlens_l${LAYER}_${MODEL_TYPE}.pt" \
    --model-type "$MODEL_TYPE" \
    --hidden-dims "$HIDDEN_DIMS" \
    --learning-rate "$LEARNING_RATE" \
    --weight-decay "$WEIGHT_DECAY" \
    --num-epochs "$NUM_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --class-weight balanced \
    --normalize \
    --seed "$SEED" \
    --device "$DEVICE" \
    --verbose

# --------------------------------------------------------------- logitlens
uv run interp-cli train_next_action_probe \
    --train-data-path "${OUT}_logitlens_train" \
    --eval-data-path "${OUT}_logitlens_val" \
    --output-path "${PROBES}/qwen_p2_local_belief_logitlens_l${LAYER}_${MODEL_TYPE}.pt" \
    --model-type "$MODEL_TYPE" \
    --hidden-dims "$HIDDEN_DIMS" \
    --learning-rate "$LEARNING_RATE" \
    --weight-decay "$WEIGHT_DECAY" \
    --num-epochs "$NUM_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --class-weight balanced \
    --normalize \
    --seed "$SEED" \
    --device "$DEVICE" \
    --verbose

# ------------------------------------------------------------------ random
uv run interp-cli train_next_action_probe \
    --train-data-path "${OUT}_random_train" \
    --eval-data-path "${OUT}_random_val" \
    --output-path "${PROBES}/qwen_p2_local_belief_random_l${LAYER}_${MODEL_TYPE}.pt" \
    --model-type "$MODEL_TYPE" \
    --hidden-dims "$HIDDEN_DIMS" \
    --learning-rate "$LEARNING_RATE" \
    --weight-decay "$WEIGHT_DECAY" \
    --num-epochs "$NUM_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --class-weight balanced \
    --normalize \
    --seed "$SEED" \
    --device "$DEVICE" \
    --verbose
