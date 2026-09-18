#!/usr/bin/env bash
# Qwen P2, dataset creation step 3a: ROLLOUT + RELABEL for the JLENS arm -- train and val.
#
# Stages 2 and 3 of three. ../2_preparations/prepare_jlens.sh must have run: this reads the
# manifests it wrote, measures what the model would have answered at each of their tokens,
# and writes the relabelled dataset the trainer consumes.
#
# WHY THE ROLLOUT EXISTS. Prepare labels every sample with the trajectory's final
# agent_action -- where the model ENDED UP. The probe is supposed to read where it WAS at
# that token. --strategy recorded_selection replays this arm's own token_idx picks, so the
# cut points are exactly the tokens its probe trains on and nothing else is rolled out; the
# other four strategies choose their own cut points and would label the wrong ones.
#
# WHAT THE ROLLOUT DOES. output_tokens[:pos+1] is kept, the trajectory's own final-channel
# prefix is appended verbatim (lifted from the data, not re-tokenized), and exactly one token
# is generated -- which, because the prefix primes `{ "action": "`, IS the action. The model
# gets no opportunity to say anything else.
#
# LOCAL BELIEF vs FINAL ACTION. The new `label` is where the model was at that token; relabel
# keeps the old one as `final_label`, alongside rollout_answer_prob (its confidence in the
# answer it gave), rollout_correct (label == final_label, i.e. already committed), cutoff_kind
# and dir_logmass. The two coming apart before the model commits is the entire reason this
# pipeline exists.
#
# --keep-kinds recorded DROPS THE BOOKENDS. Every strategy appends a no_reasoning and an
# end_of_reasoning cutoff so its first and last eval matches every other arm's. They are the
# same two prompts in all three arms and end_of_reasoning is near-deterministic, so keeping
# them would inflate every arm equally and dilute the contrast. They remain in the rollout
# JSONs if an endpoint analysis ever wants them.
#
# SAMPLES ARE DROPPED. A cutoff that produced no parseable action disappears from the
# manifest, so the arms end up with slightly different row counts and none has the full 60
# per trajectory. Read the printed drop counts rather than assuming.
#
# LENS FOLLOWS THE ARM. dir_logmass is a covariate, not a cut point -- the picks come from the
# record -- but an arm selected by one lens and annotated with the other's mass would be
# described by a ruler that did not choose it.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)   # repo root, as ../../p2_selection does

ARM=jlens
LENS=jlens          # the ruler that chose this arm's tokens; see the header

ACT_TRAIN=/workspace/activations/qwen_p2_selection
# size*/*.json, NOT the bare directory. run_inference.py's expand_paths rglobs a directory
# for *.json, and these dataset folders also hold replay_batch_summary{,1}.json plus stale
# .ipynb_checkpoints/*-checkpoint.json copies. Handed the directory, the train set resolves
# to 556 files rather than 549: the run dies on the first summary with
# KeyError: 'model_params', and the checkpoint copies would be rolled out as trajectories
# whose stem matches no selection record. This glob is the layout expand_paths documents,
# and it reproduces the gather's 549 train / 52 val exactly.
TRAJ_TRAIN="/workspace/trajectories/qwen3.6-35b/replayed_single_step/mass_train_576/size*/*.json"

ACT_VAL=/workspace/activations/qwen_p2_selection_eval
TRAJ_VAL="/workspace/trajectories/qwen3.6-35b/replayed_single_step/mass_eval_144/size*/*.json"

PREP=/workspace/prepared/qwen_p2              # read: ../2_preparations wrote these
ROLLOUTS=/workspace/rollouts/qwen_p2          # -> ${ROLLOUTS}/${ARM}_{train,val}/
OUT=/workspace/prepared/qwen_p2_local_belief  # -> ${OUT}_${ARM}_{train,val}

MODEL=Qwen/Qwen3.6-35B-A3B
LAYER=27
DEVICE_MAP=cuda:0   # NOT "auto": spreading this MoE over several GPUs produces NaNs
BATCH_SIZE=4
MAX_BATCH_TOKENS=49152

mkdir -p "$ROLLOUTS"
cd "$REPO"

# ------------------------------------------------------------------ 2. ROLLOUT
# --selection-root holds the records, --lens-root the mass tables. One gather wrote both.
uv run --extra gpu python telos_interp/loudness_analysis/rollouts/run_inference.py \
    --trajectory-paths "$TRAJ_TRAIN" \
    --output-dir "${ROLLOUTS}/${ARM}_train" \
    --model-id "$MODEL" \
    --strategy recorded_selection \
    --selection-arm "$ARM" \
    --selection-root "$ACT_TRAIN" \
    --lens-root "$ACT_TRAIN" \
    --lens "$LENS" \
    --loudness-layer "$LAYER" \
    --batch-size "$BATCH_SIZE" \
    --max-batch-tokens "$MAX_BATCH_TOKENS" \
    --device-map "$DEVICE_MAP" \
    --branch-cache \
    --skip-existing

uv run --extra gpu python telos_interp/loudness_analysis/rollouts/run_inference.py \
    --trajectory-paths "$TRAJ_VAL" \
    --output-dir "${ROLLOUTS}/${ARM}_val" \
    --model-id "$MODEL" \
    --strategy recorded_selection \
    --selection-arm "$ARM" \
    --selection-root "$ACT_VAL" \
    --lens-root "$ACT_VAL" \
    --lens "$LENS" \
    --loudness-layer "$LAYER" \
    --batch-size "$BATCH_SIZE" \
    --max-batch-tokens "$MAX_BATCH_TOKENS" \
    --device-map "$DEVICE_MAP" \
    --branch-cache \
    --skip-existing

# ------------------------------------------------------------------ 3. RELABEL
# Nothing about the manifest changes but the label: still token-major, activations_root
# untouched, no tensor moves. CPU, minutes.
uv run python telos_interp/loudness_analysis/rollouts/relabel_manifest_from_rollout.py \
    "${PREP}_${ARM}_train" "${ROLLOUTS}/${ARM}_train" "${OUT}_${ARM}_train" \
    --keep-kinds recorded \
    --report-csv "${OUT}_${ARM}_train_relabel.csv"

uv run python telos_interp/loudness_analysis/rollouts/relabel_manifest_from_rollout.py \
    "${PREP}_${ARM}_val" "${ROLLOUTS}/${ARM}_val" "${OUT}_${ARM}_val" \
    --keep-kinds recorded \
    --report-csv "${OUT}_${ARM}_val_relabel.csv"
