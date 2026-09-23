#!/usr/bin/env bash
# gpt-oss P2, step 3: the selecting gather on the TRAIN set. REUSED, NOT GATHERED.
#
# The gpt-oss twin of wrappers/qwen_analysis/2_dataset_creation/1_p2_selection/p2_training_selection.sh.
# On Qwen this is a three-arm selecting gather. On gpt-oss the selections already exist, and
# the trees are PRUNED to them. The tokens a fresh gather would want may have been deleted,
# and a gather pointed at these trees would add to a record every probe on disk reads. So
# this checks the trees and writes nothing.
#
# WHERE EACH ARM LIVES. Two trees, over the same 3,600 trajectories, pinned by name:
#
#   jlens_mass_l15       arms jlens + random   (wrappers/jlens_mass_l15.sh)
#   logitlens_mass_l15   arm  logitlens        (its logitlens twin, NAMES_FILE-pinned)
#
# The random arm is in the jlens tree only. The control was drawn once, before pruning. The
# logitlens run was deliberately given no random arm, because a second draw could only
# sample survivors.
#
# HOW IT DIFFERS FROM QWEN P2, and it cannot be changed without a re-gather:
#   * N = 20 tokens per trajectory per arm, not 60.
#   * layer 15, the gpt-oss convention, force-kept for every selected token.
#   * NO SAMPLING. The trees were gathered without --data_sample_p, so each arm picked from the
#     whole chain (Qwen P2 picks from a 20% sample). SAMPLE_PERCENT=1.0 below records that.
#   * one gather over all 3,600, not a train gather and an eval gather. The train/eval split
#     is the mass-era partition (scripts/build_mass_era_split.sh): 2,880 train here,
#     p2_eval_selection.sh checks the 720.
#
# Both arms were ranked by logprob_mass_full, the direction mass over the whole vocabulary.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)   # so uv finds pyproject.toml
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

JLENS_TREE=/workspace/activations/jlens_mass_l15         # arms jlens, random
LOGITLENS_TREE=/workspace/activations/logitlens_mass_l15 # arm logitlens
TRAJ=/workspace/activations/mass_train2880_view/trajectories
SIGNAL=direction_tokens_full.json

LAYER=15               # the gpt-oss probe layer; every arm's candidate_layers
SAMPLE_PERCENT=1.0     # the whole chain; no --data_sample_p was passed

cd "$REPO"
uv run python "$HERE/check_reused_tree.py" --tree "$JLENS_TREE" --trajectories "$TRAJ" \
    --lens jlens --layers "$LAYER" --signal "$SIGNAL" \
    --arms jlens,random --candidate-layer "$LAYER" --pt-layer "$LAYER"
uv run python "$HERE/check_reused_tree.py" --tree "$LOGITLENS_TREE" --trajectories "$TRAJ" \
    --lens logitlens --layers "$LAYER" --signal "$SIGNAL" \
    --arms logitlens --candidate-layer "$LAYER" --pt-layer "$LAYER"
