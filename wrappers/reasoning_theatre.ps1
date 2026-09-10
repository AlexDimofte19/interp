$TRAJECTORY_DIR = "C:\Uni\Thesis\data\reveng\trajectories_train_single_step"
$OUTPUT_DIR = "C:\Uni\Thesis\data\reasoning_theatre"
$STEPS = "0"

$LOG_DIR = Join-Path $OUTPUT_DIR "logs"
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null

foreach ($SIZE_DIR in Get-ChildItem -Path $TRAJECTORY_DIR -Directory -Filter "size*") {

    $SIZE_NAME = $SIZE_DIR.Name

    $env:CUDA_VISIBLE_DEVICES = "0"
    python scripts/inference_oss/run_inference.py `
        --trajectory-paths "$($SIZE_DIR.FullName)/*.json" `
        --output-dir $OUTPUT_DIR `
        --torch-dtype bfloat16 `
        --dry-run
}
