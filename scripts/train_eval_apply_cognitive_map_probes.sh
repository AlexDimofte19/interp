#!/bin/bash
# Script to run cognitivie map probe training, evaluation, and application steps
# Supports parallel training across multiple GPUs

# ==============================================================================
# GPU CONFIGURATION - Set this to control parallelism
# ==============================================================================
# Examples:
#   GPUS=(0)           # Single GPU, sequential training
#   GPUS=(0 1 2 3)     # 4 GPUs, 4 jobs in parallel
#   GPUS=(0 1 2 3 4 5) # 6 GPUs, 6 jobs in parallel
GPUS=(0 1 2 3 4 5)
NUM_GPUS=${#GPUS[@]}

# ==============================================================================
# Path configuration
# ==============================================================================
TRAIN_LAYERS=(7 15 23)
TEST_GRID_SIZES=(7 9 11 13 15)

ACTIVATIONS_TRAIN_PATH="interp/activations_train_single_step"
TRAJECTORIES_TRAIN_PATH="reveng/trajectories_train_single_step"
ACTIVATION_TEST_PATH="interp/activations_test_full"
TRAJECTORIES_TEST_PATH="reveng/trajectories_test_full"
TRAJECTORIES_OUTPUT_PATH="reveng/trajectories_test_full_with_cognitive_map_probes"
EVAL_RESULTS_PATH="reveng/cognitive_map_probes_results"
PROBES_PATH="interp/cognitive_map_probes"

PROBE_TYPES=("mlp" "lr")
MLP_PROBE_HIDDEN_DIMS=1024
PROBE_LEARNING_RATE=3e-4
PROBE_WEIGHT_DECAY=0.001
PROBE_NUM_EPOCHS=50
PROBE_BATCH_SIZE=2048
PROBE_DROPOUT=0.0
PROBE_EVAL_SPLIT=0.005
PROBE_PER_CLASS_MAX_COUNT=357000

# Create logs directory
mkdir -p logs

# ==============================================================================
# Helper function to eval a probe if eval JSON doesn't exist
# ==============================================================================
eval_probe_if_missing() {
    local probe_path=$1
    local eval_output_dir=$2
    local eval_output_base=$3
    local trajectories_dir=$4
    local activations_dir=$5
    local layer=$6
    local pad_size=$7

    local eval_json="${eval_output_dir}/${eval_output_base}.json"
    local eval_txt="${eval_output_dir}/${eval_output_base}.txt"

    if [ -f "$eval_json" ]; then
        echo "  [SKIP] Eval exists: $eval_json"
        return 0
    fi

    if [ ! -f "$probe_path" ]; then
        echo "  [ERROR] Probe missing, cannot eval: $probe_path"
        return 1
    fi

    mkdir -p "$eval_output_dir"

    echo "  [EVAL] Evaluating probe: $probe_path"
    interp-cli eval_cognitive_map_probe \
        --trajectories-dir "$trajectories_dir" \
        --activations-dir "$activations_dir" \
        --probe-path "$probe_path" \
        --layers $layer \
        --steps all \
        --output-indices all \
        --pad-to-size $pad_size \
        --output_path "$eval_json" \
        --verbose \
        | tee "$eval_txt"
}

# ==============================================================================
# Helper function to apply a probe if output doesn't exist
# ==============================================================================
apply_probe_if_missing() {
    local probe_path=$1
    local output_dir=$2
    local trajectories_dir=$3
    local activations_dir=$4
    local layer=$5

    # Check if output dir has trajectory files (not just eval files)
    if [ -d "$output_dir" ]; then
        local traj_count=$(ls "$output_dir"/*.json 2>/dev/null | grep -v "^eval_" | wc -l)
        if [ "$traj_count" -gt 0 ]; then
            echo "  [SKIP] Apply output exists: $output_dir"
            return 0
        fi
    fi

    if [ ! -f "$probe_path" ]; then
        echo "  [ERROR] Probe missing, cannot apply: $probe_path"
        return 1
    fi

    echo "  [APPLY] Applying probe to: $output_dir"
    interp-cli apply_cognitive_map_probe \
        --trajectories-dir "$trajectories_dir" \
        --activations-dir "$activations_dir" \
        --probe-path "$probe_path" \
        --output-dir "$output_dir" \
        --layers $layer \
        --steps all \
        --output-indices all
}

# ==============================================================================
# Helper function to run training jobs in parallel batches
# ==============================================================================
run_training_jobs_parallel() {
    local -n probe_paths=$1
    local -n train_data_paths=$2
    local -n probe_types_arr=$3
    local -n descriptions=$4

    local num_jobs=${#probe_paths[@]}

    if [ $num_jobs -eq 0 ]; then
        echo "No training jobs to run."
        return 0
    fi

    echo "Running $num_jobs training jobs across $NUM_GPUS GPU(s)..."

    local job_idx=0
    while [ $job_idx -lt $num_jobs ]; do
        # Launch up to NUM_GPUS jobs in parallel
        local pids=()
        local gpu_assignments=()

        for ((gpu_slot=0; gpu_slot<NUM_GPUS && job_idx<num_jobs; gpu_slot++, job_idx++)); do
            local gpu_id=${GPUS[$gpu_slot]}
            local probe_path="${probe_paths[$job_idx]}"
            local train_data="${train_data_paths[$job_idx]}"
            local probe_type="${probe_types_arr[$job_idx]}"
            local description="${descriptions[$job_idx]}"

            echo "  [GPU $gpu_id] Starting: $description"

            CUDA_VISIBLE_DEVICES=$gpu_id interp-cli train_cognitive_map_probe \
                --train-data-path "$train_data" \
                --output-path "$probe_path" \
                --model-type "$probe_type" \
                --hidden-dims $MLP_PROBE_HIDDEN_DIMS \
                --learning-rate $PROBE_LEARNING_RATE \
                --weight-decay $PROBE_WEIGHT_DECAY \
                --dropout $PROBE_DROPOUT \
                --num-epochs $PROBE_NUM_EPOCHS \
                --batch-size $PROBE_BATCH_SIZE \
                --eval-split $PROBE_EVAL_SPLIT \
                --per-class-max-count $PROBE_PER_CLASS_MAX_COUNT \
                --device cuda \
                --normalize \
                --subset 1.0 \
                --balance-classes \
                > "logs/train_${description}.log" 2>&1 &

            pids+=($!)
            gpu_assignments+=("GPU $gpu_id: $description")
        done

        echo ""
        echo "  Waiting for batch to complete..."
        for assignment in "${gpu_assignments[@]}"; do
            echo "    $assignment"
        done

        # Wait for all jobs in this batch to complete
        for pid in "${pids[@]}"; do
            wait $pid
            local status=$?
            if [ $status -ne 0 ]; then
                echo "  [WARNING] Job with PID $pid exited with status $status"
            fi
        done

        echo "  Batch complete. Progress: $job_idx / $num_jobs"
        echo ""
    done
}

# ==============================================================================
# PART 1: Prepare missing activations
# ==============================================================================
echo "=========================================="
echo "PART 1: Preparing missing activations"
echo "=========================================="

# Check and prepare general activations (padded to 15)
for layer in "${TRAIN_LAYERS[@]}"; do
    # Pre-reasoning (suffix)
    act_file="${ACTIVATIONS_TRAIN_PATH}/cognitive_map_activations_l${layer}_s0_suffix_all_grid_tile_pad15_merged.pt"
    if [ ! -f "$act_file" ]; then
        echo "Preparing pre_reasoning general activations for layer ${layer}"
        interp-cli prepare_activations_for_probing \
            --trajectories-dir $TRAJECTORIES_TRAIN_PATH \
            --activations-dir $ACTIVATIONS_TRAIN_PATH \
            --probe-type grid_tile \
            --layers $layer \
            --prompt_suffix_indices all
    else
        echo "[SKIP] Pre-reasoning general activations exist for layer ${layer}"
    fi

    # Post-reasoning (output)
    act_file="${ACTIVATIONS_TRAIN_PATH}/cognitive_map_activations_l${layer}_s0_output_all_grid_tile_pad15_merged.pt"
    if [ ! -f "$act_file" ]; then
        echo "Preparing post_reasoning general activations for layer ${layer}"
        interp-cli prepare_activations_for_probing \
            --trajectories-dir $TRAJECTORIES_TRAIN_PATH \
            --activations-dir $ACTIVATIONS_TRAIN_PATH \
            --probe-type grid_tile \
            --layers $layer \
            --output_indices all
    else
        echo "[SKIP] Post-reasoning general activations exist for layer ${layer}"
    fi
done

# Check and prepare size-specific activations
for layer in "${TRAIN_LAYERS[@]}"; do
    for size in "${TEST_GRID_SIZES[@]}"; do
        # Pre-reasoning (suffix)
        act_file="${ACTIVATIONS_TRAIN_PATH}/size${size}/cognitive_map_activations_l${layer}_s0_suffix_all_grid_tile_grid${size}.pt"
        if [ ! -f "$act_file" ]; then
            echo "Preparing pre_reasoning size${size} activations for layer ${layer}"
            interp-cli prepare_activations_for_probing \
                --trajectories-dir $TRAJECTORIES_TRAIN_PATH/size${size} \
                --activations-dir $ACTIVATIONS_TRAIN_PATH/size${size} \
                --probe-type grid_tile \
                --layers ${layer} \
                --prompt_suffix_indices all
        fi

        # Post-reasoning (output)
        act_file="${ACTIVATIONS_TRAIN_PATH}/size${size}/cognitive_map_activations_l${layer}_s0_output_all_grid_tile_grid${size}.pt"
        if [ ! -f "$act_file" ]; then
            echo "Preparing post_reasoning size${size} activations for layer ${layer}"
            interp-cli prepare_activations_for_probing \
                --trajectories-dir $TRAJECTORIES_TRAIN_PATH/size${size} \
                --activations-dir $ACTIVATIONS_TRAIN_PATH/size${size} \
                --probe-type grid_tile \
                --layers ${layer} \
                --output_indices all
        fi
    done
done

# ==============================================================================
# PART 2: Train GENERAL probes (parallel)
# ==============================================================================
echo ""
echo "=========================================="
echo "PART 2: Training general probes (parallel)"
echo "=========================================="

# Build job queue for general probes
declare -a GENERAL_PROBE_PATHS
declare -a GENERAL_TRAIN_DATA
declare -a GENERAL_PROBE_TYPES
declare -a GENERAL_DESCRIPTIONS

for layer in "${TRAIN_LAYERS[@]}"; do
    for probe_type in "${PROBE_TYPES[@]}"; do
        for location in pre_reasoning post_reasoning; do
            if [ "$location" == "pre_reasoning" ]; then
                location_name="suffix"
            else
                location_name="output"
            fi

            probe_path="${PROBES_PATH}/cognitive_map_probe_layer${layer}_${probe_type}_${location}_all_general.pt"
            train_data="${ACTIVATIONS_TRAIN_PATH}/cognitive_map_activations_l${layer}_s0_${location_name}_all_grid_tile_pad15_merged.pt"

            if [ -f "$probe_path" ]; then
                echo "[SKIP] Probe exists: $probe_path"
            elif [ ! -f "$train_data" ]; then
                echo "[ERROR] Training data missing: $train_data"
            else
                GENERAL_PROBE_PATHS+=("$probe_path")
                GENERAL_TRAIN_DATA+=("$train_data")
                GENERAL_PROBE_TYPES+=("$probe_type")
                GENERAL_DESCRIPTIONS+=("general_layer${layer}_${probe_type}_${location}")
                echo "[QUEUED] $probe_path"
            fi
        done
    done
done

echo ""
run_training_jobs_parallel GENERAL_PROBE_PATHS GENERAL_TRAIN_DATA GENERAL_PROBE_TYPES GENERAL_DESCRIPTIONS

# ==============================================================================
# PART 3: Eval and apply GENERAL probes (sequential)
# ==============================================================================
echo ""
echo "=========================================="
echo "PART 3: Eval/apply general probes"
echo "=========================================="

for layer in "${TRAIN_LAYERS[@]}"; do
    for probe_type in "${PROBE_TYPES[@]}"; do
        for location in pre_reasoning post_reasoning; do
            echo ""
            echo "=== Layer ${layer}, ${probe_type}, ${location} (general) ==="

            probe_path="${PROBES_PATH}/cognitive_map_probe_layer${layer}_${probe_type}_${location}_all_general.pt"

            # Eval
            eval_output_dir="${EVAL_RESULTS_PATH}/layer${layer}/${probe_type}_general/${location}"
            eval_output_base="eval_cognitive_map_probe_layer${layer}_${probe_type}_${location}_all_general"
            eval_probe_if_missing "$probe_path" "$eval_output_dir" "$eval_output_base" \
                "$TRAJECTORIES_TEST_PATH" "$ACTIVATION_TEST_PATH" "$layer" 15

            # Apply for each size
            for size in "${TEST_GRID_SIZES[@]}"; do
                apply_output_dir="${TRAJECTORIES_OUTPUT_PATH}/layer${layer}/${probe_type}_general/${location}/size${size}"
                apply_probe_if_missing "$probe_path" "$apply_output_dir" \
                    "$TRAJECTORIES_TEST_PATH/size${size}" "$ACTIVATION_TEST_PATH/size${size}" "$layer"
            done
        done
    done
done

# ==============================================================================
# PART 4: Train SIZE-SPECIFIC probes (parallel)
# ==============================================================================
echo ""
echo "=========================================="
echo "PART 4: Training size-specific probes (parallel)"
echo "=========================================="

# Build job queue for size-specific probes
declare -a SIZE_PROBE_PATHS
declare -a SIZE_TRAIN_DATA
declare -a SIZE_PROBE_TYPES
declare -a SIZE_DESCRIPTIONS

for layer in "${TRAIN_LAYERS[@]}"; do
    for probe_type in "${PROBE_TYPES[@]}"; do
        for location in pre_reasoning post_reasoning; do
            if [ "$location" == "pre_reasoning" ]; then
                location_name="suffix"
            else
                location_name="output"
            fi

            for size in "${TEST_GRID_SIZES[@]}"; do
                probe_path="${PROBES_PATH}/cognitive_map_probe_layer${layer}_${probe_type}_${location}_all_size${size}.pt"
                train_data="${ACTIVATIONS_TRAIN_PATH}/size${size}/cognitive_map_activations_l${layer}_s0_${location_name}_all_grid_tile_grid${size}.pt"

                if [ -f "$probe_path" ]; then
                    echo "[SKIP] Probe exists: $probe_path"
                elif [ ! -f "$train_data" ]; then
                    echo "[ERROR] Training data missing: $train_data"
                else
                    SIZE_PROBE_PATHS+=("$probe_path")
                    SIZE_TRAIN_DATA+=("$train_data")
                    SIZE_PROBE_TYPES+=("$probe_type")
                    SIZE_DESCRIPTIONS+=("size${size}_layer${layer}_${probe_type}_${location}")
                    echo "[QUEUED] $probe_path"
                fi
            done
        done
    done
done

echo ""
run_training_jobs_parallel SIZE_PROBE_PATHS SIZE_TRAIN_DATA SIZE_PROBE_TYPES SIZE_DESCRIPTIONS

# ==============================================================================
# PART 5: Eval and apply SIZE-SPECIFIC probes (sequential)
# ==============================================================================
echo ""
echo "=========================================="
echo "PART 5: Eval/apply size-specific probes"
echo "=========================================="

for layer in "${TRAIN_LAYERS[@]}"; do
    for probe_type in "${PROBE_TYPES[@]}"; do
        for location in pre_reasoning post_reasoning; do
            echo ""
            echo "=== Layer ${layer}, ${probe_type}, ${location} (size-specific) ==="

            for size in "${TEST_GRID_SIZES[@]}"; do
                echo "--- Size ${size} ---"

                probe_path="${PROBES_PATH}/cognitive_map_probe_layer${layer}_${probe_type}_${location}_all_size${size}.pt"

                # Eval - output to results folder (size-specific probes use single-size mode)
                eval_output_dir="${EVAL_RESULTS_PATH}/layer${layer}/${probe_type}/${location}"
                eval_output_base="eval_cognitive_map_probe_layer${layer}_${probe_type}_${location}_all_size${size}"
                eval_probe_if_missing "$probe_path" "$eval_output_dir" "$eval_output_base" \
                    "$TRAJECTORIES_TEST_PATH/size${size}" "$ACTIVATION_TEST_PATH/size${size}" "$layer" "$size"

                # Apply - trajectories go to size subfolder
                apply_output_dir="${TRAJECTORIES_OUTPUT_PATH}/layer${layer}/${probe_type}/${location}/size${size}"
                apply_probe_if_missing "$probe_path" "$apply_output_dir" \
                    "$TRAJECTORIES_TEST_PATH/size${size}" "$ACTIVATION_TEST_PATH/size${size}" "$layer"
            done
        done
    done
done

echo ""
echo "=========================================="
echo "Script completed!"
echo "Your probes are in ${PROBES_PATH}"
echo "Your trajectories with probe predictions are in ${TRAJECTORIES_OUTPUT_PATH}"
echo "Your evaluation results are in ${EVAL_RESULTS_PATH}"
echo "Check logs/ directory for probe training logs"
echo "=========================================="
