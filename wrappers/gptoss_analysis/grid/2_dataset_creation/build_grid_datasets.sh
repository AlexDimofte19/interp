#!/usr/bin/env bash
# gpt-oss P2 GRID dataset creation: the grid_tile probe datasets, all three arms, train and val.
#
# The gpt-oss twin of wrappers/qwen_analysis/grid/2_dataset_creation/build_grid_datasets.sh.
# One stage:
#
#   2_preparations/prepare_<arm>.sh    tree + selection record -> token-major grid_tile manifest,
#                                      25 cells per (trajectory, step)  (CPU, minutes)
#
# NO ROLLOUT STAGE: a cell's contents are read off grid_state, so there is no local-belief
# label to measure. The prepared manifests are what the trainer consumes.
#
# The arms: jlens and logitlens are the 20 most GRID-loaded tokens per trajectory at layer 14
# (the grid profile's argmax), and random is 20 drawn uniformly as the matched control -- all
# three from the same grid gathers in 1_p2_selection/, all over the whole chain.
#
# OUTPUT. /workspace/prepared/gptoss_p2_grid_<arm>_{train,val}, 2,880 / 720 trajectories.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

bash "$HERE/2_preparations/prepare_jlens.sh"
bash "$HERE/2_preparations/prepare_logitlens.sh"
bash "$HERE/2_preparations/prepare_random.sh"
