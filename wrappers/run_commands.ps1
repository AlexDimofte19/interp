# python download_trajectories.py

# arguments: train/test
# training single step
$activations_train_single_step = "C:\Uni\Thesis\data\activations_train_single_step"
# $trajectories_train_single_step = "C:\Uni\Thesis\repos\interp\data\trajectories_train_single_step"
$trajectories_train_single_step = "C:\Uni\Thesis\data\reveng\trajectories_train_single_step"
# testing
$activations_test_full = "C:\Uni\Thesis\repos\interp\data\activations\activations_test_full"
$trajectories_test_full = "C:\Uni\Thesis\repos\interp\data\trajectories_test_full"

$cognitive_map_activations="C:\Uni\Thesis\data\activations_train_single_step\cognitive_map_activations_l15_s0_output_all_grid_tile_pad15_merged"

# Write-Host "--------------------------------------------------------------------------"
# Write-Host "Running interp-cli prepare_activations_for_probing..."
# interp-cli prepare_activations_for_probing `
#     --activations-dir $activations_train_single_step `
#     --trajectories-dir $trajectories_train_single_step `
#     --probe-type grid_tile `
#     --layers 15 `
#     --output-indices all `
#     --verbose

# Write-Host "--------------------------------------------------------------------------"
# Write-Host "Running interp-cli train_cognitive_map_probe for linear layer..."
# interp-cli train_cognitive_map_probe `
#     --train-data-path $cognitive_map_activations `
#     --model-type lr `
#     --hidden-dims "1024" `
#     --num-epochs 50 `
#     --learning-rate 0.0003 `
#     --batch-size 2048 `
#     --weight-decay 0.001 `
#     --dropout 0.0 `
#     --eval-split 0.005 `
#     --balance_classes true


interp-cli train_cognitive_map_probe `
    --train-data-path $cognitive_map_activations `
    --model-type mlp `
    --dropout 0.0 `
    --num-epochs 50 `
    --learning-rate 0.001 `
    --batch-size 2048 `
    --weight-decay 1e-3 `
    --eval-split 0.005 `
    --seed 42 `
    --device cuda `
    --class-weight balanced `
    --output-path "C:\Uni\Thesis\data\cognitive_map_probes\trained\cognitive_map_probe_layer15_mlp_pre_reasoning_all_general.pt"
    # --hidden-dims 512,256 `

# Write-Host "--------------------------------------------------------------------------"
# Write-Host "Running interp-cli eval_cognitive_map_probe..."
interp-cli eval_cognitive_map_probe --probe-path "C:\Uni\Thesis\repos\interp\data\cognitive_map_probes\cognitive_map_probe_layer15_lr_pre_reasoning_all_general.pt" --trajectories-dir "C:\Uni\Thesis\repos\interp\data\trajectories_test_full" --activations-dir "C:\Uni\Thesis\repos\interp\data\activations\activations_test_full" --prompt-suffix-indices "-3:-1"
 

interp-cli eval_cognitive_map_probe --probe-path "C:\Uni\Thesis\data\cognitive_map_probes\trained\cognitive_map_probe_layer15_mlp_pre_reasoning_all_general.pt" --trajectories-dir "C:\Uni\Thesis\repos\interp\data\trajectories_test_full" --activations-dir "C:\Uni\Thesis\repos\interp\data\activations\activations_test_full" --prompt-suffix-indices "-3:-1"
