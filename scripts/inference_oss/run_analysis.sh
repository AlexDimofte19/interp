#!/usr/bin/env bash
# Aggregate + plot the reasoning-theatre stats that run_inference.py wrote.
# INPUT_DIR is the run_inference --output-dir (it contains the size*/ subfolders);
# analysis.py walks size*/*.json itself, so there is no per-size loop here.

INPUT_DIR="/workspace/reasoning_theatre/trajectories_train_single_step_probs/"
# Source trajectories: needed for grid_complexity (the heatmaps' y axis), which the
# results JSONs do not carry. analysis.py joins them to results by filename stem.
TRAJECTORY_DIR="/workspace/reasoning_theatre/trajectories_train_single_step/"
OUTPUT_DIR="/workspace/reasoning_theatre/trajectories_train_single_step_plots/"

LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

LOG_FILE="$LOG_DIR/analysis_$(date +%Y%m%d_%H%M%S).txt"
echo "=== START: $(date) ===" > "$LOG_FILE"

python interp/scripts/inference_oss/analysis.py \
        --input-folder "$INPUT_DIR" \
        --trajectory-folder "$TRAJECTORY_DIR" \
        --output-dir "$OUTPUT_DIR" \
        >> "$LOG_FILE" 2>&1

echo "=== END: $(date) (exit code: $?) ===" >> "$LOG_FILE"

runpodctl stop pod "$RUNPOD_POD_ID"
