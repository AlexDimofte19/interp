#!/usr/bin/env bash
# gpt-oss P2, dataset creation step 2b: the PREPARE stage for the LOGITLENS arm -- train and val.
#
# The gpt-oss twin of wrappers/qwen_analysis/2_dataset_creation/2_preparations/prepare_logitlens.sh.
# Stage 1 of three: token-major manifests labelled with the trajectory's final agent_action.
# ../3_rollouts/rollout_logitlens.sh measures the local belief at those same tokens and swaps it
# in. A dataset is not finished until that has run.
#
# THE REUSED TREE, SPLIT BY ITS TRAJECTORIES. Qwen has a train tree and an eval tree. gpt-oss
# has one tree over all 3,600, and the split is the mass-era partition: each call names the
# mass_train2880_view / mass_eval720_view trajectory folder. prepare skips any gathered
# trajectory whose JSON is not in --trajectories-dir, so each manifest covers its half and
# nothing else. ACT is the real tree, not the view's activations/ link farm, so every arm's
# activations_root names the tree the gather wrote.
#
# THE ARM IS THE ONLY THING THAT DIFFERS between this file and its two siblings (plus which
# tree it lives in). Here: the 20 tokens per trajectory the LOGIT lens scored loudest at
# layer 15, ranked by logprob_mass_full over the whole chain. It lives in its own tree,
# logitlens_mass_l15, gathered over the same 3,600 names as the jlens one.
#
# recorded_<arm>, NEVER <lens>_direction. The tree is pruned, so only selected tokens have a
# .pt, and the random arm can only be read back from the record that reserved it.
#
# Cheap: next_action copies no activations. CPU only.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)   # repo root

ARM=logitlens
ACT=/workspace/activations/logitlens_mass_l15    # the logitlens twin tree; jlens + random are in jlens_mass_l15

TRAJ_TRAIN=/workspace/activations/mass_train2880_view/trajectories
TRAJ_VAL=/workspace/activations/mass_eval720_view/trajectories

PREP=/workspace/prepared/gptoss_p2        # -> ${PREP}_${ARM}_{train,val}

LAYER=15
PROBE_TYPE=next_action

cd "$REPO"

uv run interp-cli prepare_activations_for_probing \
    --activations-dir "$ACT" \
    --trajectories-dir "$TRAJ_TRAIN" \
    --probe-type "$PROBE_TYPE" \
    --layers "$LAYER" \
    --steps all \
    --output-indices all \
    --token-selection "recorded_${ARM}" \
    --output-path "${PREP}_${ARM}_train" \
    --verbose

uv run interp-cli prepare_activations_for_probing \
    --activations-dir "$ACT" \
    --trajectories-dir "$TRAJ_VAL" \
    --probe-type "$PROBE_TYPE" \
    --layers "$LAYER" \
    --steps all \
    --output-indices all \
    --token-selection "recorded_${ARM}" \
    --output-path "${PREP}_${ARM}_val" \
    --verbose
