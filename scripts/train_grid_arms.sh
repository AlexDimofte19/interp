#!/usr/bin/env bash
# Train the grid_tile ("cognitive map") probe once per arm, over a sweep of seeds.
#
# The sibling of scripts/train_next_action_arms.sh: it consumes ${PREPARED}_${arm} for each
# arm that scripts/prepare_grid_arms.sh wrote, so pass the same prefix you gave it as OUT.
#
# Two things differ from the action version, both forced by the data rather than chosen:
#
#   * The split is not optional. A grid arm's manifest is token-major -- a trajectory owns
#     ~20 entries that all carry its grid -- so train_cognitive_map_probe's own --eval-split
#     would put the same trajectory in both halves. It refuses to, and points here. Every
#     run therefore trains on a pre-split pair.
#   * EVAL_NAMES pins the eval set. The seeded split is a function of the names present, so
#     two datasets covering different trajectories get different eval sets at the same seed.
#     Pointing every arm at one name list is what makes "jlens vs logitlens vs random"
#     a comparison rather than three numbers. Use the same file the action arms used.
#
# Arms are the outer loop on purpose: --cache-activations packs each dataset on first use,
# so the first run of an arm pays the file-opening cost (one .pt per sample, tens of
# thousands of small reads) and every later run of that arm reuses it. Interleaving arms
# would thrash that. Runs already finished are skipped, so the script is resumable.
#
#   PREPARED=/workspace/prepared/grid EVAL_NAMES=/path/eval_720.txt ./scripts/train_grid_arms.sh
#   ARMS="jlens logitlens" SEEDS=42 ./scripts/train_grid_arms.sh          # the 7:23 pair
#   MODEL_TYPES=lr DEVICE=cpu NUM_EPOCHS=2 ./scripts/train_grid_arms.sh   # quick smoke run
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)   # so uv finds pyproject.toml

PREPARED=${PREPARED:-/workspace/prepared/grid}   # reads ${PREPARED}_${arm}
ARMS=${ARMS:-"jlens logitlens random eos"}
SEEDS=${SEEDS:-"42 43 44"}
PROBES=${PROBES:-/workspace/probes/grid}
LOGS=${LOGS:-$PROBES/logs}
TAG=${TAG:-l15}                                  # goes in the probe/log filename

EVAL_NAMES=${EVAL_NAMES:-}                       # file of eval trajectory names, one per line
EVAL_SPLIT=${EVAL_SPLIT:-0.2}                    # only used when EVAL_NAMES is empty
LAYERS_PER_TOKEN=${LAYERS_PER_TOKEN:-1}
TOKENS_PER_TRAJ=${TOKENS_PER_TRAJ:-all}
SPLIT_SEED=${SPLIT_SEED:-42}                     # NOT the training seed: see below
MODEL_TYPES=${MODEL_TYPES:-"lr mlp"}
DEVICE=${DEVICE:-cuda}

# Probe hyperparameters from configs/cell_identity_probes/*.conf, so these numbers sit next
# to the published cognitive-map results rather than next to the next_action ones.
HIDDEN_DIMS=${HIDDEN_DIMS:-1024}
LEARNING_RATE=${LEARNING_RATE:-3e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.001}
NUM_EPOCHS=${NUM_EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-2048}
DROPOUT=${DROPOUT:-0.0}

mkdir -p "$PROBES" "$LOGS"

split_arm() {
    local arm=$1 prepared=$2 split_dir=$3

    # The split seed is deliberately NOT the training seed. Bumping it would give each seed a
    # different eval set and destroy the shared trajectories every arm is scored on; the
    # sweep is meant to vary initialisation and batch order, nothing else.
    local args=(--layers-per-token "$LAYERS_PER_TOKEN" --seed "$SPLIT_SEED")
    [ "$TOKENS_PER_TRAJ" = "all" ] || args+=(--tokens-per-trajectory "$TOKENS_PER_TRAJ")
    if [ -n "$EVAL_NAMES" ]; then
        args+=(--eval-names "$EVAL_NAMES")
    else
        args+=(--eval-split "$EVAL_SPLIT")
    fi

    uv run --project "$REPO" python "$REPO/scripts/split_next_action_manifest.py" \
        "$prepared" "${args[@]}" \
        --train-out "${split_dir}_train" \
        --eval-out "${split_dir}_eval" \
        | tee "$LOGS/${arm}_${TAG}_split.txt"
}

echo "prepared prefix: ${PREPARED}_<arm>"
echo "arms:            $ARMS      seeds: $SEEDS      models: $MODEL_TYPES"
echo "eval set:        ${EVAL_NAMES:-seeded split at $EVAL_SPLIT (NOT shared across arms)}"
[ -n "$EVAL_NAMES" ] || echo "  !! without EVAL_NAMES the arms are scored on different trajectories"

skipped=""
failed=""
for arm in $ARMS; do
    prepared="${PREPARED}_${arm}"
    split_dir="${prepared}_${TAG}"

    if [ ! -f "$prepared/manifest.json" ]; then
        echo "!! ${arm}: no manifest at $prepared/manifest.json -- skipping" >&2
        skipped="$skipped $arm"
        continue
    fi

    echo ""
    echo "============================================================"
    echo "Arm: ${arm}   ($prepared)"
    echo "============================================================"
    if [ -f "${split_dir}_train/manifest.json" ] && [ -f "${split_dir}_eval/manifest.json" ]; then
        echo "  split already at ${split_dir}_{train,eval}, reusing"
    else
        split_arm "$arm" "$prepared" "$split_dir" || { failed="$failed ${arm}:split"; continue; }
    fi

    for seed in $SEEDS; do
        for model_type in $MODEL_TYPES; do
            tag="${arm}_${TAG}_seed${seed}_${model_type}"
            if grep -q "Balanced Accuracy" "$LOGS/${tag}.txt" 2>/dev/null; then
                echo "- have ${tag}, skipping"
                continue
            fi
            echo ""
            echo "- Training ${model_type} probe (${tag})"

            uv run --project "$REPO" interp-cli train_cognitive_map_probe \
                --train-data-path "${split_dir}_train" \
                --eval-data-path "${split_dir}_eval" \
                --output-path "$PROBES/grid_probe_${tag}.pt" \
                --model-type "$model_type" \
                --hidden-dims $HIDDEN_DIMS \
                --learning-rate $LEARNING_RATE \
                --weight-decay $WEIGHT_DECAY \
                --dropout $DROPOUT \
                --num-epochs $NUM_EPOCHS \
                --batch-size $BATCH_SIZE \
                --class-weight balanced \
                --normalize \
                --subset 1.0 \
                --seed "$seed" \
                --device "$DEVICE" \
                --cache-activations \
                --verbose \
                | tee "$LOGS/${tag}.txt"
            rc=${PIPESTATUS[0]}
            [ "$rc" -eq 0 ] || failed="$failed ${tag}(rc=$rc)"
        done
    done
done

echo ""
echo "============================================================"
echo "SUMMARY (balanced accuracy)"
echo "  down a column: seed noise. across: lens vs lens vs control."
echo "============================================================"
printf '%-11s %-6s %-6s %s\n' "arm" "seed" "model" "result"
for arm in $ARMS; do
    for seed in $SEEDS; do
        for model_type in $MODEL_TYPES; do
            f="$LOGS/${arm}_${TAG}_seed${seed}_${model_type}.txt"
            if [ -f "$f" ]; then
                printf '%-11s %-6s %-6s %s\n' "$arm" "$seed" "$model_type" \
                    "$(grep -m1 'Balanced Accuracy' "$f" || echo 'no result in log')"
            else
                printf '%-11s %-6s %-6s %s\n' "$arm" "$seed" "$model_type" "not run"
            fi
        done
    done
done

[ -z "$skipped" ] || { echo ""; echo "Arms with no prepared dataset:$skipped"; }
[ -z "$failed" ]  || { echo ""; echo "FAILED:$failed"; }

echo ""
echo "Probes: $PROBES"
echo "Logs:   $LOGS"
[ -z "$failed" ]
