#!/usr/bin/env bash
# Qwen P2, step 4: build the LOCAL-BELIEF probe datasets -- three arms, train and val.
#
# Three stages, six datasets. Prepare labels every sample with the trajectory's final
# agent_action; the rollout measures what the model would have answered had its reasoning
# stopped at that token; relabel swaps the second in for the first. The output of stage 3
# is what the trainer reads.
#
#   1. prepare   (CPU, minutes)  tree + selection record -> token-major manifest
#   2. rollout   (GPU, hours)    truncate at each selected token, force one action token
#   3. relabel   (CPU, minutes)  join model_action onto the manifest as `label`
#
# WHY THE ROLLOUT IS PER ARM. --strategy recorded_selection replays the token_idx picks of
# one arm of {stem}_jlens_selection.json, so the cut points are exactly the tokens that
# arm's probe trains on and nothing else is rolled out. The other four strategies choose
# their own cut points from the trajectory, which would label the wrong tokens here.
#
# WHAT THE ROLLOUT DOES. output_tokens[:pos+1] is kept, the trajectory's own final-channel
# prefix is appended verbatim (it is lifted from the data, not re-tokenized), and exactly
# one token is generated -- which, because the prefix primes `{ "action": "`, IS the
# action. The model gets no opportunity to say anything else.
#
# LOCAL BELIEF vs FINAL ACTION. The new `label` is where the model was at that token;
# relabel keeps the old one as `final_label`, alongside rollout_answer_prob (its confidence
# in the answer it gave), rollout_correct (label == final_label, i.e. already committed),
# cutoff_kind and dir_logmass. The two labels coming apart before the model commits is the
# entire reason this pipeline exists.
#
# --keep-kinds recorded DROPS THE BOOKENDS. Every strategy appends a no_reasoning and an
# end_of_reasoning cutoff so its first and last eval matches every other arm's. They are
# the same two prompts in all three arms and end_of_reasoning is near-deterministic, so
# keeping them would inflate every arm equally and dilute the contrast. Drop them here;
# they are still in the rollout JSONs if an endpoint analysis ever wants them.
#
# SAMPLES ARE DROPPED. A cutoff that produced no parseable action disappears from the
# manifest, so the arms end up with slightly different row counts and none of them has the
# full 60 per trajectory. Read stage 3's printed drop counts rather than assuming.
#
# ---------------------------------------------------------------------------------------
# BLOCKER: run_inference.py HAS NO --model-id AND WILL FAIL ON THESE TRAJECTORIES.
# It reads the id from the trajectory JSON (run_inference.py:710), which here is
# "gsarti/qwen3.6-35b" -- the Together AI serving name, not a HuggingFace repo -- so
# from_pretrained raises before the first rollout. build_loudness_tables.py already took
# the --model-id override for exactly this reason; run_inference.py has not. Stage 2 cannot
# run until it does. Stages 1 and 3 are unaffected.
# ---------------------------------------------------------------------------------------
set -euo pipefail

REPO=/workspace/repo/interp

ACT_TRAIN=/workspace/activations/qwen_p2_selection
TRAJ_TRAIN=/workspace/trajectories/qwen3.6-35b/replayed_single_step/mass_train_576

ACT_VAL=/workspace/activations/qwen_p2_selection_eval
TRAJ_VAL=/workspace/trajectories/qwen3.6-35b/replayed_single_step/mass_eval_144

PREP=/workspace/prepared/qwen_p2              # stage 1 -> ${PREP}_${arm}_{train,val}
ROLLOUTS=/workspace/rollouts/qwen_p2          # stage 2 -> ${ROLLOUTS}/${arm}_{train,val}/
OUT=/workspace/prepared/qwen_p2_local         # stage 3 -> ${OUT}_${arm}_{train,val}
# ^ the trainer's OUT. Point train_next_action_probes_all_selections.sh at this, not at
# PREP -- PREP still carries the final-action label.

LAYER=27
PROBE_TYPE=next_action
LENS=jlens          # which mass table supplies dir_logmass; the cut points come from the
                    # record, so this is a covariate only and does not move a single cutoff
DEVICE_MAP=cuda:0   # NOT "auto": spreading this MoE over several GPUs produces NaNs
BATCH_SIZE=4
MAX_BATCH_TOKENS=49152

mkdir -p "$ROLLOUTS"
cd "$REPO"

# =====================================================================  1. PREPARE
# ---------------------------------------------------------------- jlens (train, val)
uv run interp-cli prepare_activations_for_probing \
    --activations-dir "$ACT_TRAIN" \
    --trajectories-dir "$TRAJ_TRAIN" \
    --probe-type "$PROBE_TYPE" \
    --layers "$LAYER" \
    --steps all \
    --output-indices all \
    --token-selection recorded_jlens \
    --output-path "${PREP}_jlens_train" \
    --verbose

uv run interp-cli prepare_activations_for_probing \
    --activations-dir "$ACT_VAL" \
    --trajectories-dir "$TRAJ_VAL" \
    --probe-type "$PROBE_TYPE" \
    --layers "$LAYER" \
    --steps all \
    --output-indices all \
    --token-selection recorded_jlens \
    --output-path "${PREP}_jlens_val" \
    --verbose

# ----------------------------------------------------------- logitlens (train, val)
uv run interp-cli prepare_activations_for_probing \
    --activations-dir "$ACT_TRAIN" \
    --trajectories-dir "$TRAJ_TRAIN" \
    --probe-type "$PROBE_TYPE" \
    --layers "$LAYER" \
    --steps all \
    --output-indices all \
    --token-selection recorded_logitlens \
    --output-path "${PREP}_logitlens_train" \
    --verbose

uv run interp-cli prepare_activations_for_probing \
    --activations-dir "$ACT_VAL" \
    --trajectories-dir "$TRAJ_VAL" \
    --probe-type "$PROBE_TYPE" \
    --layers "$LAYER" \
    --steps all \
    --output-indices all \
    --token-selection recorded_logitlens \
    --output-path "${PREP}_logitlens_val" \
    --verbose

# --------------------------------------------------------------- random (train, val)
# The matched control: a uniform draw over each chain's reasoning tokens, reserved by the
# step-3 gather with the same N and seed as the two lens arms.
uv run interp-cli prepare_activations_for_probing \
    --activations-dir "$ACT_TRAIN" \
    --trajectories-dir "$TRAJ_TRAIN" \
    --probe-type "$PROBE_TYPE" \
    --layers "$LAYER" \
    --steps all \
    --output-indices all \
    --token-selection recorded_random \
    --output-path "${PREP}_random_train" \
    --verbose

uv run interp-cli prepare_activations_for_probing \
    --activations-dir "$ACT_VAL" \
    --trajectories-dir "$TRAJ_VAL" \
    --probe-type "$PROBE_TYPE" \
    --layers "$LAYER" \
    --steps all \
    --output-indices all \
    --token-selection recorded_random \
    --output-path "${PREP}_random_val" \
    --verbose

# =====================================================  2. TRUNCATED-REASONING ROLLOUTS
# --selection-root is the tree that holds the records; --lens-root the one that holds the
# mass tables. One gather wrote both, so they are the same directory here.
# ---------------------------------------------------------------- jlens (train, val)
uv run --extra gpu python telos_interp/loudness_analysis/rollouts/run_inference.py \
    --trajectory-paths "$TRAJ_TRAIN" \
    --output-dir "${ROLLOUTS}/jlens_train" \
    --strategy recorded_selection \
    --selection-arm jlens \
    --selection-root "$ACT_TRAIN" \
    --lens-root "$ACT_TRAIN" \
    --lens "$LENS" \
    --loudness-layer "$LAYER" \
    --batch-size "$BATCH_SIZE" \
    --max-batch-tokens "$MAX_BATCH_TOKENS" \
    --device-map "$DEVICE_MAP" \
    --skip-existing

uv run --extra gpu python telos_interp/loudness_analysis/rollouts/run_inference.py \
    --trajectory-paths "$TRAJ_VAL" \
    --output-dir "${ROLLOUTS}/jlens_val" \
    --strategy recorded_selection \
    --selection-arm jlens \
    --selection-root "$ACT_VAL" \
    --lens-root "$ACT_VAL" \
    --lens "$LENS" \
    --loudness-layer "$LAYER" \
    --batch-size "$BATCH_SIZE" \
    --max-batch-tokens "$MAX_BATCH_TOKENS" \
    --device-map "$DEVICE_MAP" \
    --skip-existing

# ----------------------------------------------------------- logitlens (train, val)
uv run --extra gpu python telos_interp/loudness_analysis/rollouts/run_inference.py \
    --trajectory-paths "$TRAJ_TRAIN" \
    --output-dir "${ROLLOUTS}/logitlens_train" \
    --strategy recorded_selection \
    --selection-arm logitlens \
    --selection-root "$ACT_TRAIN" \
    --lens-root "$ACT_TRAIN" \
    --lens "$LENS" \
    --loudness-layer "$LAYER" \
    --batch-size "$BATCH_SIZE" \
    --max-batch-tokens "$MAX_BATCH_TOKENS" \
    --device-map "$DEVICE_MAP" \
    --skip-existing

uv run --extra gpu python telos_interp/loudness_analysis/rollouts/run_inference.py \
    --trajectory-paths "$TRAJ_VAL" \
    --output-dir "${ROLLOUTS}/logitlens_val" \
    --strategy recorded_selection \
    --selection-arm logitlens \
    --selection-root "$ACT_VAL" \
    --lens-root "$ACT_VAL" \
    --lens "$LENS" \
    --loudness-layer "$LAYER" \
    --batch-size "$BATCH_SIZE" \
    --max-batch-tokens "$MAX_BATCH_TOKENS" \
    --device-map "$DEVICE_MAP" \
    --skip-existing

# --------------------------------------------------------------- random (train, val)
uv run --extra gpu python telos_interp/loudness_analysis/rollouts/run_inference.py \
    --trajectory-paths "$TRAJ_TRAIN" \
    --output-dir "${ROLLOUTS}/random_train" \
    --strategy recorded_selection \
    --selection-arm random \
    --selection-root "$ACT_TRAIN" \
    --lens-root "$ACT_TRAIN" \
    --lens "$LENS" \
    --loudness-layer "$LAYER" \
    --batch-size "$BATCH_SIZE" \
    --max-batch-tokens "$MAX_BATCH_TOKENS" \
    --device-map "$DEVICE_MAP" \
    --skip-existing

uv run --extra gpu python telos_interp/loudness_analysis/rollouts/run_inference.py \
    --trajectory-paths "$TRAJ_VAL" \
    --output-dir "${ROLLOUTS}/random_val" \
    --strategy recorded_selection \
    --selection-arm random \
    --selection-root "$ACT_VAL" \
    --lens-root "$ACT_VAL" \
    --lens "$LENS" \
    --loudness-layer "$LAYER" \
    --batch-size "$BATCH_SIZE" \
    --max-batch-tokens "$MAX_BATCH_TOKENS" \
    --device-map "$DEVICE_MAP" \
    --skip-existing

# ==========================================================  3. RELABEL (local belief)
# Nothing about the manifest changes but the label: still token-major, activations_root
# untouched, no tensor moves. Each run prints how many samples it dropped -- read it.
# ---------------------------------------------------------------- jlens (train, val)
uv run python telos_interp/loudness_analysis/rollouts/relabel_manifest_from_rollout.py \
    "${PREP}_jlens_train" "${ROLLOUTS}/jlens_train" "${OUT}_jlens_train" \
    --keep-kinds recorded \
    --report-csv "${OUT}_jlens_train_relabel.csv"

uv run python telos_interp/loudness_analysis/rollouts/relabel_manifest_from_rollout.py \
    "${PREP}_jlens_val" "${ROLLOUTS}/jlens_val" "${OUT}_jlens_val" \
    --keep-kinds recorded \
    --report-csv "${OUT}_jlens_val_relabel.csv"

# ----------------------------------------------------------- logitlens (train, val)
uv run python telos_interp/loudness_analysis/rollouts/relabel_manifest_from_rollout.py \
    "${PREP}_logitlens_train" "${ROLLOUTS}/logitlens_train" "${OUT}_logitlens_train" \
    --keep-kinds recorded \
    --report-csv "${OUT}_logitlens_train_relabel.csv"

uv run python telos_interp/loudness_analysis/rollouts/relabel_manifest_from_rollout.py \
    "${PREP}_logitlens_val" "${ROLLOUTS}/logitlens_val" "${OUT}_logitlens_val" \
    --keep-kinds recorded \
    --report-csv "${OUT}_logitlens_val_relabel.csv"

# --------------------------------------------------------------- random (train, val)
uv run python telos_interp/loudness_analysis/rollouts/relabel_manifest_from_rollout.py \
    "${PREP}_random_train" "${ROLLOUTS}/random_train" "${OUT}_random_train" \
    --keep-kinds recorded \
    --report-csv "${OUT}_random_train_relabel.csv"

uv run python telos_interp/loudness_analysis/rollouts/relabel_manifest_from_rollout.py \
    "${PREP}_random_val" "${ROLLOUTS}/random_val" "${OUT}_random_val" \
    --keep-kinds recorded \
    --report-csv "${OUT}_random_val_relabel.csv"
