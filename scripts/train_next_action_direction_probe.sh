#!/usr/bin/env bash
# Train the next_action ("direction") probe on lens-selected reasoning tokens, against
# their matched random control, across a sweep of top-K token budgets.
#
# Consumes the prepared dirs written by scripts/prepare_next_action_jlens_by_complexity.sh.
# A lens run only means something next to a `random` run over the same trajectories -- the
# top-scoring tokens are the direction words the model already verbalized, so a high number
# in isolation says nothing. jlens vs logitlens is the second comparison: same tokens
# available, same tree, different lens deciding which of them to keep.
#
# split_next_action_manifest.py reshapes each arm three ways before training:
#   * TOKENS_PER_TRAJ -- each trajectory's K top-ranked tokens, identical to having
#     prepared with --num-tokens K. Sweeping it here rather than at prepare time is why
#     top-1/2/3 costs three splits instead of three multi-hour prepares.
#   * LAYERS_PER_TOKEN -- one layer per token: the trainer pools every row into one (N, D)
#     matrix fit by a single weight vector, and layers 7 and 22 are not in a shared basis.
#   * a trajectory-grouped, label-stratified holdout: train_next_action_probe's own
#     --eval-split is random over rows, and every trajectory owns all of its selected
#     tokens with one shared label.
# The split is computed from the unthinned strata, so all K share one train/eval partition
# and the numbers below are comparable across the sweep.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)   # so uv finds pyproject.toml

PREPARED=${PREPARED:-/workspace/prepared/next_action_comp0.0-0.2-0.4}   # ${PREPARED}_${arm}
ARMS=${ARMS:-"jlens logitlens random"}
PROBES=${PROBES:-/workspace/probes/next_action_comp0.0-0.2-0.4}
LOGS=${LOGS:-$PROBES/logs}

EVAL_SPLIT=${EVAL_SPLIT:-0.2}
TOKENS_PER_TRAJ=${TOKENS_PER_TRAJ:-"1 2 3"}   # top-K sweep; one prepared dataset serves all
LAYERS_PER_TOKEN=${LAYERS_PER_TOKEN:-1}
SEED=${SEED:-42}
MODEL_TYPES=${MODEL_TYPES:-"lr mlp"}

# Probe hyperparameters (general_probe_train.sh values; batch size lowered because a
# jlens selection is ~20 rows/trajectory, not ~C cells/trajectory).
HIDDEN_DIMS=1024
LEARNING_RATE=3e-4
WEIGHT_DECAY=0.001
NUM_EPOCHS=50
BATCH_SIZE=512
DROPOUT=0.0

mkdir -p "$PROBES" "$LOGS"

train_arm() {
    local arm_name=$1 prepared=$2

    if [ ! -f "$prepared/manifest.json" ]; then
        echo "!! ${arm_name}: no manifest at $prepared/manifest.json -- skipping" >&2
        return 1
    fi

    for k in $TOKENS_PER_TRAJ; do
        local tag="${arm_name}_top${k}"
        local split_dir="${prepared}_top${k}"

        echo ""
        echo "============================================================"
        echo "Arm: ${arm_name}   top-${k} token(s)/trajectory   ($prepared)"
        echo "============================================================"

        # Writes ${split_dir}_train and ${split_dir}_eval, each a lone manifest.json --
        # next_action copies no activations, so a split costs nothing but the JSON.
        uv run --project "$REPO" python "$REPO/scripts/split_next_action_manifest.py" \
            "$prepared" \
            --tokens-per-trajectory "$k" \
            --layers-per-token "$LAYERS_PER_TOKEN" \
            --eval-split "$EVAL_SPLIT" \
            --seed "$SEED" \
            --train-out "${split_dir}_train" \
            --eval-out "${split_dir}_eval" \
            | tee "$LOGS/${tag}_split.txt"

        for model_type in $MODEL_TYPES; do
            echo ""
            echo "- Training ${model_type} probe (${tag})"

            uv run --project "$REPO" interp-cli train_next_action_probe \
                --train-data-path "${split_dir}_train" \
                --eval-data-path "${split_dir}_eval" \
                --output-path "$PROBES/next_action_probe_${tag}_${model_type}.pt" \
                --model-type "$model_type" \
                --hidden-dims $HIDDEN_DIMS \
                --learning-rate $LEARNING_RATE \
                --weight-decay $WEIGHT_DECAY \
                --dropout $DROPOUT \
                --num-epochs $NUM_EPOCHS \
                --batch-size $BATCH_SIZE \
                --class-weight balanced \
                --normalize \
                --seed "$SEED" \
                --device cuda \
                --verbose \
                | tee "$LOGS/${tag}_${model_type}.txt"
        done
    done
}

for arm in $ARMS; do
    train_arm "$arm" "${PREPARED}_${arm}" || true
done

echo ""
echo "============================================================"
echo "SUMMARY (best balanced accuracy)"
echo "  read down a column for the top-K effect, across for lens vs lens vs control"
echo "============================================================"
printf '%-10s %-6s %-6s %s\n' "arm" "top-K" "model" "result"
for arm in $ARMS; do
    for k in $TOKENS_PER_TRAJ; do
        for model_type in $MODEL_TYPES; do
            f="$LOGS/${arm}_top${k}_${model_type}.txt"
            [ -f "$f" ] || continue
            printf '%-10s %-6s %-6s %s\n' "$arm" "$k" "$model_type" \
                "$(grep -m1 'Best balanced accuracy' "$f" || echo 'no result')"
        done
    done
done
echo ""
echo "Probes: $PROBES"
echo "Logs:   $LOGS"
