#!/usr/bin/env bash
# Train the next_action ("direction") probe once per arm, across a sweep of top-K budgets.
#
# The sibling of scripts/prepare_next_action_arms.sh: it consumes ${PREPARED}_${arm} for each
# arm that script wrote, so pass the same prefix you gave it as OUT.
#
# Why three arms. A lens result means nothing next to nothing: the top-scoring tokens are the
# direction words the model already verbalized, so `random` is what says whether the lens
# found anything a uniform draw would not have. jlens vs logitlens is the second comparison
# -- same trajectories, same tree, same control, different lens deciding which tokens to keep.
#
# split_next_action_manifest.py reshapes each arm before training:
#   * TOKENS_PER_TRAJ -- each trajectory's K top-ranked tokens, identical to having prepared
#     with --num-tokens K. Sweeping it here rather than at prepare time is why top-1/2/3
#     costs three splits instead of three prepares.
#   * LAYERS_PER_TOKEN -- one layer per token: the trainer pools every row into one (N, D)
#     matrix fit by a single weight vector, and layers 7 and 22 are not in a shared basis.
#   * a trajectory-grouped, label-stratified holdout: train_next_action_probe's own
#     --eval-split is random over rows, and every trajectory owns all of its selected tokens
#     with one shared label.
# The split is computed from the unthinned strata, so all K share one train/eval partition
# and the numbers are comparable across the sweep.
#
#   PREPARED=/workspace/prepared/next_action ./scripts/train_next_action_arms.sh
#   ARMS="jlens logitlens" TOKENS_PER_TRAJ="1 3" ./scripts/train_next_action_arms.sh
#   MODEL_TYPES=lr DEVICE=cpu ./scripts/train_next_action_arms.sh      # quick smoke run
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)   # so uv finds pyproject.toml

PREPARED=${PREPARED:-/workspace/prepared/next_action}   # reads ${PREPARED}_${arm}
ARMS=${ARMS:-"jlens logitlens random"}
PROBES=${PROBES:-/workspace/probes/next_action}
LOGS=${LOGS:-$PROBES/logs}

EVAL_SPLIT=${EVAL_SPLIT:-0.2}
TOKENS_PER_TRAJ=${TOKENS_PER_TRAJ:-"1 2 3"}   # top-K sweep; one prepared dataset serves all
LAYERS_PER_TOKEN=${LAYERS_PER_TOKEN:-1}
SEED=${SEED:-42}
MODEL_TYPES=${MODEL_TYPES:-"lr mlp"}
DEVICE=${DEVICE:-cuda}

# Probe hyperparameters (general_probe_train.sh values; batch size lowered because a lens
# selection is ~20 rows/trajectory, not ~C cells/trajectory).
HIDDEN_DIMS=${HIDDEN_DIMS:-1024}
LEARNING_RATE=${LEARNING_RATE:-3e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.001}
NUM_EPOCHS=${NUM_EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-512}
DROPOUT=${DROPOUT:-0.0}

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
                --device "$DEVICE" \
                --verbose \
                | tee "$LOGS/${tag}_${model_type}.txt"
        done
    done
}

echo "prepared prefix: ${PREPARED}_<arm>"
echo "arms:            $ARMS"
echo "top-K sweep:     $TOKENS_PER_TRAJ   layers/token: $LAYERS_PER_TOKEN"
echo "models:          $MODEL_TYPES   device: $DEVICE"

# An arm whose dataset is missing must not take the others down -- it usually means that arm
# is not in the selection records yet (see scripts/jlens_extend_logitlens.sh).
skipped=""
for arm in $ARMS; do
    train_arm "$arm" "${PREPARED}_${arm}" || skipped="$skipped $arm"
done

echo ""
echo "============================================================"
echo "SUMMARY (best balanced accuracy)"
echo "  down a column: the top-K effect. across: lens vs lens vs control."
echo "============================================================"
printf '%-11s %-6s %-6s %s\n' "arm" "top-K" "model" "result"
for arm in $ARMS; do
    for k in $TOKENS_PER_TRAJ; do
        for model_type in $MODEL_TYPES; do
            f="$LOGS/${arm}_top${k}_${model_type}.txt"
            if [ -f "$f" ]; then
                printf '%-11s %-6s %-6s %s\n' "$arm" "$k" "$model_type" \
                    "$(grep -m1 'Best balanced accuracy' "$f" || echo 'no result in log')"
            else
                printf '%-11s %-6s %-6s %s\n' "$arm" "$k" "$model_type" "not run"
            fi
        done
    done
done

if [ -n "$skipped" ]; then
    echo ""
    echo "Arms with no prepared dataset:$skipped"
    echo "  Run scripts/prepare_next_action_arms.sh first. If an arm prepares nothing, it is"
    echo "  not in the selection records -- add it with scripts/jlens_extend_logitlens.sh."
fi

echo ""
echo "Probes: $PROBES"
echo "Logs:   $LOGS"
echo ""
echo "NOTE: the arms will not have equal sample counts -- the lens arms select overlapping"
echo "      tokens with more layers each, and the control's layer draw is independent. The"
echo "      accuracies above are computed on different N; read them with that in view."
