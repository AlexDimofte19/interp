#!/usr/bin/env bash
# gpt-oss P2 dataset creation: the LOCAL-BELIEF probe datasets, all three arms, train and val.
#
# The gpt-oss twin of wrappers/qwen_analysis/2_dataset_creation/build_local_belief_datasets.sh.
# Prepare first for every arm, then roll out and relabel for every arm:
#
#   2_preparations/prepare_<arm>.sh    reused tree + selection record -> token-major manifest
#                                      labelled with the trajectory's FINAL agent_action
#                                      (CPU, minutes)
#   3_rollouts/rollout_<arm>.sh        truncate at each of those tokens, force one action
#                                      token, then swap the answer in as `label`
#                                      (GPU, then a CPU join)
#
# The arms are jlens, logitlens and random: each lens' loudest 20 tokens per trajectory at
# layer 15, and 20 drawn uniformly over the whole chain as the matched control. All three
# were recorded long ago in jlens_mass_l15 / logitlens_mass_l15 and are only READ here
# (see 1_p2_selection/).
#
# WHY THE ROLLOUT IS NOT OPTIONAL. Prepare can only label a token with where the trajectory
# ENDED UP. A dataset that has been prepared but not relabelled is a final-action dataset
# wearing a local-belief name, which is why the train wrapper checks for `final_label`.
#
# OUTPUT. /workspace/prepared/gptoss_p2_local_belief_<arm>_{train,val}.
#
# SERIAL ON PURPOSE: the rollout holds the model on one GPU. Every stage is resumable.
#
# NOT COVERED HERE: the held-out rollout (3_rollouts/rollout_heldout_every_token.sh), which
# belongs to the held-out loudness analysis and is already on disk.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

bash "$HERE/2_preparations/prepare_jlens.sh"
bash "$HERE/2_preparations/prepare_logitlens.sh"
bash "$HERE/2_preparations/prepare_random.sh"

bash "$HERE/3_rollouts/rollout_jlens.sh"
bash "$HERE/3_rollouts/rollout_logitlens.sh"
bash "$HERE/3_rollouts/rollout_random.sh"
