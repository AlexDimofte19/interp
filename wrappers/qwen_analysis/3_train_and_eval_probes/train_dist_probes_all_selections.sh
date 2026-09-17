#!/usr/bin/env bash
# Qwen P2, step 5: train one direction probe per arm, then read all three on the held-out 72.
#
# The sibling of prepare_probe_datasets.sh -- it consumes the six manifests that script
# wrote, so keep OUT below equal to its OUT.
#
# TWO POPULATIONS, AND THEY ARE NOT COMPARABLE. Each train command reports its arm on the
# VAL half of its OWN selection: the jlens probe scored on jlens-selected tokens, the
# control on uniformly drawn ones. The held-out pass at the bottom is the other population
# -- all three probes on the SAME tokens, every reasoning token of 72 trajectories no lens
# ever touched. On the gpt-oss line the ordering INVERTED between the two (ICLR entry 49):
# on its own loud tokens the jlens arm won and the control was worst; on every token of a
# disjoint set the control won. Both numbers are real. Name the population.
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
#
# THE HELD-OUT PASS NEEDS GPU AND THE TREE FINISHED. qwen_p2_heldout is written by
# heldout_sample.sh; scoring it before that run completes silently scores a subset. It is
# one pass over ~1.17M .pt files -- budget hours, not minutes -- and it wants the same GPU
# the gather is using, so do not start it while heldout_sample.sh is still running.
set -euo pipefail

REPO=/workspace/repo/interp

OUT=/workspace/prepared/qwen_p2        # reads ${OUT}_${arm}_{train,val}
PROBES=/workspace/probes/qwen_p2
RESULTS=/workspace/results/qwen_p2

ACT_HELDOUT=/workspace/activations/qwen_p2_heldout
TRAJ_HELDOUT=/workspace/trajectories/qwen3.6-35b/replayed_single_step/heldout_72
SIGNAL_JSON=/workspace/repo/interp/data/jlens/qwen/direction_tokens_full_qwen3-6-35b-a3b.json
# ^ the vocabulary every Qwen P2 artifact so far was scored against. A mass table is not
# self-describing; pointing this elsewhere produces loudness columns the selection has no
# bearing on.

LAYER=27
PROBE_TYPE=next_action
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

mkdir -p "$PROBES" "$RESULTS"
cd "$REPO"

# =====================================================================  TRAIN
# ------------------------------------------------------------------- jlens
uv run interp-cli train_next_action_probe \
    --train-data-path "${OUT}_jlens_train" \
    --eval-data-path "${OUT}_jlens_val" \
    --output-path "${PROBES}/qwen_p2_jlens_l${LAYER}_${MODEL_TYPE}.pt" \
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
    --output-path "${PROBES}/qwen_p2_logitlens_l${LAYER}_${MODEL_TYPE}.pt" \
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
    --output-path "${PROBES}/qwen_p2_random_l${LAYER}_${MODEL_TYPE}.pt" \
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

# ================================================================  EVAL (held-out 72)
# All three probes in ONE pass: --probe is repeatable, and scoring them together is what
# guarantees they are read on exactly the same rows. --lens-dir is left to default to
# --activations-dir, because heldout_sample.sh pinned the .pt files and the CSVs to the
# same layer in the same tree.
#
# The output CSV is already a loudness table: beside each probe's verdict it carries, per
# lens, the top-k count and the full-vocabulary mass at $LAYER. There is no join step.
uv run python telos_interp/loudness_analysis/score_probes_per_token.py \
    --probe-type "$PROBE_TYPE" \
    --probe "${PROBES}/qwen_p2_jlens_l${LAYER}_${MODEL_TYPE}.pt" \
    --probe "${PROBES}/qwen_p2_logitlens_l${LAYER}_${MODEL_TYPE}.pt" \
    --probe "${PROBES}/qwen_p2_random_l${LAYER}_${MODEL_TYPE}.pt" \
    --activations-dir "$ACT_HELDOUT" \
    --trajectories-dir "$TRAJ_HELDOUT" \
    --signal-json "$SIGNAL_JSON" \
    --layer "$LAYER" \
    --out "${RESULTS}/heldout72_per_token_${MODEL_TYPE}.csv" \
    --device "$DEVICE"

# One row per probe: balanced accuracy on the held-out population. Only the vs-final-action
# columns are populated here -- label_local comes from a rollout, and the rollout line is
# gpt-oss-specific (it appends the harmony final-channel prefix), so no Qwen rollout exists.
uv run python telos_interp/loudness_analysis/summarise_probe_accuracy.py \
    "${RESULTS}/heldout72_per_token_${MODEL_TYPE}.csv" \
    --out "${RESULTS}/heldout72_summary_${MODEL_TYPE}.csv" \
    --json-out "${RESULTS}/heldout72_summary_${MODEL_TYPE}.json"
