TEST_GRID_SIZES=(7 9 11 13 15)

TRAJECTORIES_TRAIN_PATH="/workspace/trajectories/reveng/trajectories_train_single_step"
ACTIVATIONS_TRAIN_PATH="/workspace/activations/activations_train_single_step"
ACTIVATION_TEST_PATH="/workspace/activations/activations_test_full/activations_test_full"
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


layer=15
location="pre_reasoning"
echo "=== Training and evaluating probes on layer ${layer} activations ==="
echo "- Preparing padded activations for general probes (${location})"

# interp-cli prepare_activations_for_probing \
#     --trajectories-dir $TRAJECTORIES_TRAIN_PATH \
#     --activations-dir $ACTIVATIONS_TRAIN_PATH \
#     --probe-type grid_tile \
#     --layers $layer \
#     --prompt_suffix_indices all


# echo ""
# echo "Activation preparation completed!"
# echo ""

for probe_type in "${PROBE_TYPES[@]}"; do

    echo "- Training general ${probe_type} probe on all ${location} activations"

    CURRENT_PROBE_PATH="${PROBES_PATH}/cognitive_map_probe_layer${layer}_${probe_type}_${location}_all_general.pt"
    
    interp-cli train_cognitive_map_probe \
        --train-data-path ${ACTIVATIONS_TRAIN_PATH}/cognitive_map_activations_l${layer}_s0_suffix_all_grid_tile_pad15_merged \
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
done
