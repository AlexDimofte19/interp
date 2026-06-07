TRAIN_LAYERS=(7 15 23)
TEST_GRID_SIZES=(7 9 11 13 15)
PROBING_LOCATION_NAMES=("suffix" "output") # Pre- and post-reasoning

ACTIVATIONS_TRAIN_PATH="/workspace/activations/activations_train_single_step"
TRAJECTORIES_TRAIN_PATH="/workspace/trajectories/reveng/trajectories_train_single_step"
ACTIVATION_TEST_PATH="/workspace/activations/activations_test_full"
TRAJECTORIES_TEST_PATH="/workspace/trajectories/trajectories_test_full"
TRAJECTORIES_OUTPUT_PATH="/workspace/trajectories/trajectories_test_full_with_probes"
PROBES_PATH="/workspace/probes/probes_train_single_step"

PROBE_TYPES=("mlp" "lr")
MLP_PROBE_HIDDEN_DIMS=1024
PROBE_LEARNING_RATE=3e-4
PROBE_WEIGHT_DECAY=0.001
PROBE_NUM_EPOCHS=50
PROBE_BATCH_SIZE=2048
PROBE_DROPOUT=0.0
PROBE_EVAL_SPLIT=0.005
PROBE_PER_CLASS_MAX_COUNT=357000 # ~10x minority classes for general probes (36k train grids)

for layer in "${TRAIN_LAYERS[@]}"; do

    echo "=== Training and evaluating probes on layer ${layer} activations ==="

    for location_name in "${PROBING_LOCATION_NAMES[@]}"; do

        if [ "$location_name" == "suffix" ]; then
            location="pre_reasoning"

            echo "- Preparing padded activations for general probes (${location})"

            interp-cli prepare_activations_for_probing \
                --trajectories-dir $TRAJECTORIES_TRAIN_PATH \
                --activations-dir $ACTIVATIONS_TRAIN_PATH \
                --probe-type grid_tile \
                --layers $layer \
                --prompt_suffix_indices all

            for size in "${TEST_GRID_SIZES[@]}"; do
                echo "- Preparing activations for size-specific probes (${location}, grid size ${size}x${size})"

                interp-cli prepare_activations_for_probing \
                    --trajectories-dir $TRAJECTORIES_TRAIN_PATH/size${size} \
                    --activations-dir $ACTIVATIONS_TRAIN_PATH/size${size} \
                    --probe-type grid_tile \
                    --layers ${layer} \
                    --prompt_suffix_indices all
            done
        elif [ "$location_name" == "output" ]; then
            location="post_reasoning"

            echo "- Preparing padded activations for general probes (${location})"

            interp-cli prepare_activations_for_probing \
                --trajectories-dir $TRAJECTORIES_TRAIN_PATH \
                --activations-dir $ACTIVATIONS_TRAIN_PATH \
                --probe-type grid_tile \
                --layers $layer \
                --output_indices all

            for size in "${TEST_GRID_SIZES[@]}"; do
                echo "- Preparing activations for size-specific probes (${location}, grid size ${size}x${size})"

                interp-cli prepare_activations_for_probing \
                    --trajectories-dir $TRAJECTORIES_TRAIN_PATH/size${size} \
                    --activations-dir $ACTIVATIONS_TRAIN_PATH/size${size} \
                    --probe-type grid_tile \
                    --layers ${layer} \
                    --output_indices all
            done
        fi

        echo ""
        echo "Activation preparation completed!"
        echo ""

        for probe_type in "${PROBE_TYPES[@]}"; do

            echo "- Training general ${probe_type} probe on all ${location} activations"

            CURRENT_PROBE_PATH="${PROBES_PATH}/cognitive_map_probe_layer${layer}_${probe_type}_${location}_all_general.pt"

            interp-cli train_cognitive_map_probe \
                --train-data-path ${ACTIVATIONS_TRAIN_PATH}/cognitive_map_activations_l${layer}_s0_${location_name}_all_grid_tile_pad15_merged.pt \
                --output-path $CURRENT_PROBE_PATH \
                --model-type ${probe_type} \
                --hidden-dims $MLP_PROBE_HIDDEN_DIMS \
                --learning-rate $PROBE_LEARNING_RATE \
                --weight-decay $PROBE_WEIGHT_DECAY \
                --dropout $PROBE_DROPOUT \
                --num-epochs $PROBE_NUM_EPOCHS \
                --batch-size $PROBE_BATCH_SIZE  \
                --eval-split $PROBE_EVAL_SPLIT \
                --per-class-max-count $PROBE_PER_CLASS_MAX_COUNT \
                --device cuda \
                --verbose \
                --normalize \
                --subset 1.0 \
                --balance-classes
            
            echo "- Evaluating general ${probe_type} probe on all ${location} activations"

            EVAL_OUTPUT_DIR="${TRAJECTORIES_OUTPUT_PATH}/layer${layer}/${probe_type}_general/${location}"
            EVAL_OUTPUT_BASE="eval_cognitive_map_probe_layer${layer}_${probe_type}_${location}_all_general"
            mkdir -p "${EVAL_OUTPUT_DIR}"

            interp-cli eval_cognitive_map_probe \
                --trajectories-dir $TRAJECTORIES_TEST_PATH \
                --activations-dir $ACTIVATION_TEST_PATH \
                --probe-path $CURRENT_PROBE_PATH \
                --layers $layer \
                --steps all \
                --output-indices all \
                --pad-to-size 15 \
                --output_path "${EVAL_OUTPUT_DIR}/${EVAL_OUTPUT_BASE}.json" \
                --verbose \
                | tee "${EVAL_OUTPUT_DIR}/${EVAL_OUTPUT_BASE}.txt"

            echo "- Generating predictions for test trajectories for general ${probe_type} probe on all ${location} activations"

            for size in "${TEST_GRID_SIZES[@]}"; do
                interp-cli apply_cognitive_map_probe \
                    --trajectories-dir $TRAJECTORIES_TEST_PATH/size${size} \
                    --activations-dir $ACTIVATION_TEST_PATH/size${size} \
                    --probe-path $CURRENT_PROBE_PATH \
                    --output-dir ${TRAJECTORIES_OUTPUT_PATH}/layer${layer}/${probe_type}_general/${location}/size${size} \
                    --layers $layer \
                    --steps all \
                    --output-indices all
            done

            for size in "${TEST_GRID_SIZES[@]}"; do

                CURRENT_PROBE_PATH="${PROBES_PATH}/cognitive_map_probe_layer${layer}_${probe_type}_${location}_all_size${size}.pt"

                echo "- Training ${size}x${size} ${probe_type} probe on all ${location} activations"

                interp-cli train_cognitive_map_probe \
                    --train-data-path ${ACTIVATIONS_TRAIN_PATH}/cognitive_map_activations_l${layer}_s0_${location_name}_all_grid_tile_grid${size}.pt \
                    --output-path $CURRENT_PROBE_PATH \
                    --model-type ${probe_type} \
                    --hidden-dims $MLP_PROBE_HIDDEN_DIMS \
                    --learning-rate $PROBE_LEARNING_RATE \
                    --weight-decay $PROBE_WEIGHT_DECAY \
                    --dropout $PROBE_DROPOUT \
                    --num-epochs $PROBE_NUM_EPOCHS \
                    --batch-size $PROBE_BATCH_SIZE  \
                    --eval-split $PROBE_EVAL_SPLIT \
                    --per-class-max-count $PROBE_PER_CLASS_MAX_COUNT \
                    --device cuda \
                    --verbose \
                    --normalize \
                    --subset 1.0 \
                    --balance-classes

                echo "- Evaluating ${size}x${size} ${probe_type} probe on all ${location} activations"

                EVAL_OUTPUT_DIR="${TRAJECTORIES_OUTPUT_PATH}/layer${layer}/${probe_type}/${location}/size${size}"
                EVAL_OUTPUT_BASE="eval_cognitive_map_probe_layer${layer}_${probe_type}_${location}_all_size${size}"
                mkdir -p "${EVAL_OUTPUT_DIR}"

                interp-cli eval_cognitive_map_probe \
                    --trajectories-dir $TRAJECTORIES_TEST_PATH/size${size} \
                    --activations-dir $ACTIVATION_TEST_PATH/size${size} \
                    --probe-path $CURRENT_PROBE_PATH \
                    --layers $layer \
                    --steps all \
                    --output-indices all \
                    --pad-to-size $size \
                    --output_path "${EVAL_OUTPUT_DIR}/${EVAL_OUTPUT_BASE}.json" \
                    --verbose \
                    | tee "${EVAL_OUTPUT_DIR}/${EVAL_OUTPUT_BASE}.txt"

                echo "- Generating predictions for test trajectories for ${size}x${size} ${probe_type} probe on all ${location} activations"

                interp-cli apply_cognitive_map_probe \
                    --trajectories-dir $TRAJECTORIES_TEST_PATH/size${size} \
                    --activations-dir $ACTIVATION_TEST_PATH/size${size} \
                    --probe-path $CURRENT_PROBE_PATH \
                    --output-dir ${TRAJECTORIES_OUTPUT_PATH}/layer${layer}/${probe_type}/${location}/size${size} \
                    --layers $layer \
                    --steps all \
                    --output-indices all
            
            done
        done
    done
done
