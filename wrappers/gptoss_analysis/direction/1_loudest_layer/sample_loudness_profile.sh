#!/usr/bin/env bash
# gpt-oss P2, step 1: the all-layers direction loudness profile. REUSED, NOT GATHERED.
#
# The gpt-oss twin of wrappers/qwen_analysis/1_loudest_layer/sample_loudness_profile.sh.
# On Qwen this is a CSV-only gather on a 20% sample. On gpt-oss the profile already exists,
# at 100%: the layer-15 selecting gathers (wrappers/jlens_mass_l15.sh and its logitlens twin)
# wrote a direction-mass table over --layers 7:23 for every reasoning token of all 3,600
# trajectories. They did not pass --data_sample_p, so it is the whole chain, which is what
# this step wants. SAMPLE_PERCENT below is 1.0 to say so. It is not passed to anything.
#
# So this checks those tables instead of re-gathering them. A gather pointed at either tree
# would write into a tree that is already pruned to its selection. join_loudness_profile.sh
# beside this file reads the tables in place.
#
# THE PROFILE COVERS ALL 3,600, not only the train 2,880. The logitlens tree has no train-only
# view (mass_train2880_view links the jlens tree only), and both lenses should average over the
# same rows. On gpt-oss the probe layer is 15 by convention, not by this argmax. The profile is
# a description here, not a choice made on the eval set.
#
# The recorded invocations, if the trees ever have to be rebuilt from nothing:
#   NAMES_FILE=/workspace/reasoning_theatre/rollout_strategies/mass_l15_names.txt bash wrappers/jlens_mass_l15.sh
#   LENS=logitlens SELECT_METHODS=logitlens ACTIVATIONS_DIR=/workspace/activations/logitlens_mass_l15 \
#     NAMES_FILE=/workspace/reasoning_theatre/rollout_strategies/mass_l15_names.txt bash wrappers/jlens_mass_l15.sh
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)   # so uv finds pyproject.toml
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

JLENS_TREE=/workspace/activations/jlens_mass_l15
LOGITLENS_TREE=/workspace/activations/logitlens_mass_l15
TRAJ_TRAIN=/workspace/activations/mass_train2880_view/trajectories
TRAJ_VAL=/workspace/activations/mass_eval720_view/trajectories
SIGNAL=direction_tokens_full.json    # /workspace/jlens/direction_tokens_full.json, the deployed data/jlens copy

SAMPLE_PERCENT=1.0     # the whole chain; the trees were gathered without --data_sample_p
LAYERS=7:23            # what the profile must cover: every layer the gpt-oss jlens is fitted at

cd "$REPO"
for TRAJ in "$TRAJ_TRAIN" "$TRAJ_VAL"; do
    uv run python "$HERE/check_reused_tree.py" --tree "$JLENS_TREE" --trajectories "$TRAJ" \
        --lens jlens --layers "$LAYERS" --signal "$SIGNAL"
    uv run python "$HERE/check_reused_tree.py" --tree "$LOGITLENS_TREE" --trajectories "$TRAJ" \
        --lens logitlens --layers "$LAYERS" --signal "$SIGNAL"
done

echo
echo "next: wrappers/gptoss_analysis/direction/1_loudest_layer/join_loudness_profile.sh"
