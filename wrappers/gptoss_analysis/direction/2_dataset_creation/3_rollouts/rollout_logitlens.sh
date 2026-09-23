#!/usr/bin/env bash
# gpt-oss P2, dataset creation step 3b: ROLLOUT + RELABEL for the LOGITLENS arm -- train and val.
#
# The gpt-oss twin of wrappers/qwen_analysis/2_dataset_creation/3_rollouts/rollout_logitlens.sh.
# Stages 2 and 3 of three. ../2_preparations/prepare_logitlens.sh must have run: this reads the
# manifests it wrote, measures what the model would have answered at each of their tokens,
# and writes the relabelled dataset the trainer consumes.
#
# WHY THE ROLLOUT EXISTS. Prepare labels every sample with the trajectory's final
# agent_action, i.e. where the model ENDED UP. The probe is supposed to read where it WAS
# at that token. --strategy recorded_selection replays this arm's own token_idx picks from
# the reused tree's selection record, so the cut points are exactly the tokens its probe
# trains on.
#
# WHAT THE ROLLOUT DOES. output_tokens[:pos+1] is kept, the trajectory's own final-channel
# prefix is appended verbatim, and exactly one token is generated. Because the prefix primes
# `{ "action": "`, that token IS the action.
#
# --keep-kinds recorded DROPS THE BOOKENDS (no_reasoning, end_of_reasoning): the same two
# prompts in every arm, which would inflate every arm equally. They stay in the rollout JSONs.
#
# SAMPLES ARE DROPPED where a cutoff produced no parseable action, so the arms end up with
# slightly different row counts. Read the printed drop counts.
#
# LENS FOLLOWS THE ARM: dir_logmass is a covariate here, recorded on the ruler that chose the arm.
#
# TRAJECTORIES are the mass-era views' size*/*.json. The selection record and the mass
# tables are read from the reused tree by stem, so the view decides which half is rolled out.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)   # repo root

ARM=logitlens
LENS=logitlens      # the ruler that chose this arm's tokens
TREE=/workspace/activations/logitlens_mass_l15   # the logitlens twin tree: its record and its mass tables

TRAJ_TRAIN="/workspace/activations/mass_train2880_view/trajectories/size*/*.json"
TRAJ_VAL="/workspace/activations/mass_eval720_view/trajectories/size*/*.json"

PREP=/workspace/prepared/gptoss_p2              # read: ../2_preparations wrote these
ROLLOUTS=/workspace/rollouts/gptoss_p2          # -> ${ROLLOUTS}/${ARM}_{train,val}/
OUT=/workspace/prepared/gptoss_p2_local_belief  # -> ${OUT}_${ARM}_{train,val}

MODEL=openai/gpt-oss-20b
LAYER=15
DEVICE_MAP=cuda:0   # NOT "auto": spreading this MoE over several GPUs produces NaNs
BATCH_SIZE=16       # run_inference_strategies.sh's gpt-oss defaults; chains are ~10x shorter than Qwen's
MAX_BATCH_TOKENS=49152

mkdir -p "$ROLLOUTS"
cd "$REPO"

# ------------------------------------------------------------------ 2. ROLLOUT
for half in train val; do
    if [ "$half" = train ]; then TRAJ=$TRAJ_TRAIN; else TRAJ=$TRAJ_VAL; fi
    uv run --extra gpu python telos_interp/loudness_analysis/rollouts/run_inference.py \
        --trajectory-paths "$TRAJ" \
        --output-dir "${ROLLOUTS}/${ARM}_${half}" \
        --model-id "$MODEL" \
        --strategy recorded_selection \
        --selection-arm "$ARM" \
        --selection-root "$TREE" \
        --lens-root "$TREE" \
        --lens "$LENS" \
        --loudness-layer "$LAYER" \
        --batch-size "$BATCH_SIZE" \
        --max-batch-tokens "$MAX_BATCH_TOKENS" \
        --device-map "$DEVICE_MAP" \
        --branch-cache \
        --skip-existing
done

# ------------------------------------------------------------------ 3. RELABEL
# Nothing about the manifest changes but the label. CPU, minutes.
for half in train val; do
    uv run python telos_interp/loudness_analysis/rollouts/relabel_manifest_from_rollout.py \
        "${PREP}_${ARM}_${half}" "${ROLLOUTS}/${ARM}_${half}" "${OUT}_${ARM}_${half}" \
        --keep-kinds recorded \
        --report-csv "${OUT}_${ARM}_${half}_relabel.csv"
done
