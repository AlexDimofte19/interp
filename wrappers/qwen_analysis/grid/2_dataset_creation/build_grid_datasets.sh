#!/usr/bin/env bash
# Qwen P2 GRID dataset creation: the grid_tile probe datasets, all three arms, train and val.
#
# The grid twin of ../../2_dataset_creation/build_local_belief_datasets.sh, with one stage
# instead of two:
#
#   2_preparations/prepare_<arm>.sh    grid-ranked tree + selection record -> token-major
#                                      grid_tile manifest, 25 cells per (trajectory, step)
#                                      (CPU, minutes)
#
# NO ROLLOUT STAGE, ON PURPOSE. The direction line rolls out at every selected token because
# its label -- where the model WAS, not where it ended up -- only exists once measured. A grid
# cell's contents are read straight off grid_state and do not change along the chain, so
# there is no local-belief label to measure and no relabel to do. The manifests prepare
# writes are what the trainer consumes.
#
# The arms are jlens, logitlens and random: the Jacobian lens' 60 most GRID-loaded tokens per
# trajectory, the logit lens' 60, and 60 drawn uniformly as the matched control.
#
# OUTPUT. /workspace/prepared/qwen_p2_grid_<arm>_{train,val}. Point
# ../3_train_and_eval_probes at that prefix.
#
# NOT COVERED HERE: the gathers that build the trees and records (1_p2_selection/), and the
# held-out grid dataset (../4_loudness_evaluation/prepare_heldout_grid.sh).
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

bash "$HERE/2_preparations/prepare_jlens.sh"
bash "$HERE/2_preparations/prepare_logitlens.sh"
bash "$HERE/2_preparations/prepare_random.sh"
