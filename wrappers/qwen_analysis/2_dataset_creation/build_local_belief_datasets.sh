#!/usr/bin/env bash
# Qwen P2 dataset creation: the LOCAL-BELIEF probe datasets, all three arms, train and val.
#
# A master over the two stage folders beside it. Prepare first for every arm, then roll out
# and relabel for every arm:
#
#   2_preparations/prepare_<arm>.sh    tree + selection record -> token-major manifest
#                                      labelled with the trajectory's FINAL agent_action
#                                      (CPU, minutes)
#   3_rollouts/rollout_<arm>.sh        truncate at each of those tokens, force one action
#                                      token, then swap the answer in as `label`
#                                      (GPU hours, then a CPU join)
#
# The arms are jlens, logitlens and random -- the Jacobian lens' loudest 60 tokens per
# trajectory, the logit lens' loudest 60 (~half the same tokens at layer 27), and 60 drawn
# uniformly as the matched control. Each script is self-contained: run one alone to rebuild
# one arm's half of one stage.
#
# WHY THE ROLLOUT IS NOT OPTIONAL. Prepare can only label a token with where the trajectory
# ENDED UP. The probe is meant to read where the model WAS at that token, and only a rollout
# measures that. A dataset that has been prepared but not relabelled is a final-action
# dataset wearing a local-belief name -- which is why the train wrapper checks for
# `final_label` before it trains on anything.
#
# WHY THE CONTROL EXISTS. The loudest tokens are largely the direction words the model has
# already verbalized, so a lens arm on its own says nothing. `random` is what reports whether
# the lens found anything a uniform draw of the same size would not have.
#
# OUTPUT. /workspace/prepared/qwen_p2_local_belief_<arm>_{train,val}. The final-action
# manifests stay at /workspace/prepared/qwen_p2_<arm>_{train,val} -- both are valid v3
# manifests of identical shape, and nothing but the presence of `final_label` distinguishes
# them on disk. Point ../3_train_and_eval_probes at the local_belief prefix.
#
# SERIAL ON PURPOSE. The rollout loads 66 GiB of weights onto one GPU (device_map=auto would
# spread this MoE over several and produce NaNs), so two arms cannot share a card. Every
# stage is resumable -- --skip-existing on the rollout, and prepare/relabel rewrite their
# manifest -- so a killed run is restarted by running this again.
#
# NOT COVERED HERE: the gathers that built the activation trees and the selection records
# (../p2_selection/), and the held-out rollout (3_rollouts/rollout_heldout_every_token.sh),
# which belongs to the held-out loudness analysis rather than to probe training.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# bash "$HERE/2_preparations/prepare_jlens.sh"
# bash "$HERE/2_preparations/prepare_logitlens.sh"
# bash "$HERE/2_preparations/prepare_random.sh"

bash "$HERE/3_rollouts/rollout_jlens.sh"
bash "$HERE/3_rollouts/rollout_logitlens.sh"
bash "$HERE/3_rollouts/rollout_random.sh"
