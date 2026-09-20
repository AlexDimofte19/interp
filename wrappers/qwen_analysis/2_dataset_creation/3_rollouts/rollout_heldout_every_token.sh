#!/usr/bin/env bash
# Qwen P2, step 7a: the held-out rollout -- every 64th reasoning token, no selection.
#
# This is what turns the held-out set into a LOCAL-BELIEF measurement. score_probes_per_token.py
# (step 7b) labels each token with its step's `agent_action`, i.e. where the trajectory ENDED
# UP, and the probes being scored decode where the model WAS. `label_local` does not exist
# until a rollout measures it, and join_rollouts.py is what puts it on the per-token table
# that probe_accuracy_by_loudness.py then bins into loudness deciles.
#
# NO SELECTION, AND THAT IS THE POINT. --strategy every_token chooses nothing: the cutoffs are
# the tokens, not a lens' opinion of them. On the held-out set a lens-chosen grid would put the
# same selection between the loudness axis and the label that the whole held-out population
# exists to remove.
#
# --------------------------------------------------------------------------------------
# WHY STRIDE 64. Measured over 6 held-out trajectories: median 18,800 reasoning tokens per
# step, one step per trajectory, median sentence 10 tokens, ~1,230 sentences per chain. A
# dense grid is therefore ~1.17M truncated prompts of up to ~19k tokens each -- on the order
# of a WEEK of GPU, two orders of magnitude more than the gather it depends on. Stride 64
# cuts that to ~294 cutoffs per trajectory, ~21k prompts, a few hours.
#
# What the thinning does NOT cost:
#
#   * PRECISION. probe_accuracy_by_loudness.py bootstraps over trajectory NAMES, not rows --
#     tokens inside a trajectory share a sentence structure and a label. The interval is
#     bounded by 72 clusters, and no stride changes that number. The dense grid buys rows the
#     bootstrap cannot convert into confidence.
#   * BIAS. A uniform stride is uncorrelated with loudness, so decile edges and per-decile
#     means are unaffected. It thins, it does not skew.
#   * THE WITHIN-SENTENCE CONTROL. At a 10-token median sentence any stride >= 10 gives at
#     most one cut per sentence, which looks fatal for "is it loudness or position in the
#     sentence?" -- but that control needs frac_in_sentence to vary ACROSS rows, not several
#     cuts inside one sentence. With ~1,230 sentences per chain and a stride uncorrelated
#     with sentence phase, the within-sentence position is sampled uniformly anyway.
#
# What it does cost: ~2.1k rows per decile instead of ~117k, which is still far past where a
# balanced accuracy over four classes gets noisy. Lower STRIDE if a later analysis wants
# per-position resolution; 32 doubles the rows and the cost, 1 is the dense grid and is not
# affordable here.
#
# THE ENDPOINTS ARE KEPT WHATEVER THE STRIDE. no_reasoning and end_of_reasoning are added to
# every arm so its first and last eval is the same prompt as every other arm's; dropping them
# makes final accuracy and the commitment indices incomparable.
# --------------------------------------------------------------------------------------
#
# LOUDNESS IS A COVARIATE HERE, NOT A CHOOSER. every_token sets needs_loudness = False: it
# records dir_logmass per cutoff but never ranks on it, so a trajectory whose mass table is
# missing loses the covariate rather than being skipped.
#
# ORDER. The gather that writes qwen_p2_heldout must have FINISHED -- it is the same GPU, and
# the mass tables this reads are written by it. Rolling out against a half-written tree gets
# no error, just cutoffs with no loudness.
#
# Resumable: --skip-existing skips trajectory files whose output JSON already exists.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)   # repo root, as ../../p2_selection does

# size*/*.json, NOT the bare directory. run_inference.py's expand_paths rglobs a directory
# for *.json, and these dataset folders also hold replay_batch_summary{,1}.json plus stale
# .ipynb_checkpoints/*-checkpoint.json copies. Handed the directory, the train set resolves
# to 556 files rather than 549: the run dies on the first summary with
# KeyError: 'model_params', and the checkpoint copies would be rolled out as trajectories
# whose stem matches no selection record. This glob is the layout expand_paths documents,
# and it reproduces the gather's 549 train / 52 val exactly.
TRAJ="/workspace/trajectories/qwen3.6-35b/replayed_single_step/heldout_72/size*/*.json"
LENS_ROOT=/workspace/activations/qwen_p2_heldout   # the mass tables, written by heldout_sample.sh
OUT=/workspace/rollouts/qwen_p2_heldout_every_token

MODEL=Qwen/Qwen3.6-35B-A3B
STRIDE=64              # see the block above before changing this
LAYER=27               # the mass-table layer whose direction mass is recorded per cutoff
LENS=jlens

DEVICE_MAP=cuda:0      # NOT "auto": spreading this MoE over several GPUs produces NaNs
BATCH_SIZE=4
MAX_BATCH_TOKENS=49152

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$OUT"
cd "$REPO"

# Print the cutoff count and the prompt tails for the first step, then exit. Worth one run
# before committing hours: it is where a stride that is not what you meant shows up.
#   DRY_RUN=1 bash wrappers/qwen_analysis/2_dataset_creation/3_rollouts/rollout_heldout_every_token.sh
DRY_RUN=${DRY_RUN:-0}
EXTRA=""
[ "$DRY_RUN" = "1" ] && EXTRA="--dry-run"

uv run --extra gpu python telos_interp/loudness_analysis/rollouts/run_inference.py \
    --trajectory-paths "$TRAJ" \
    --output-dir "$OUT" \
    --model-id "$MODEL" \
    --strategy every_token \
    --stride "$STRIDE" \
    --lens-root "$LENS_ROOT" \
    --lens "$LENS" \
    --loudness-layer "$LAYER" \
    --batch-size "$BATCH_SIZE" \
    --max-batch-tokens "$MAX_BATCH_TOKENS" \
    --device-map "$DEVICE_MAP" \
    --skip-existing \
    $EXTRA
