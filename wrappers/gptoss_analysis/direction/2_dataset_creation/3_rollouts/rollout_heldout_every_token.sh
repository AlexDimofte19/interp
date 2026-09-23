#!/usr/bin/env bash
# gpt-oss P2, step 7a: the held-out rollout. EVERY reasoning token, NO STRIDE, no selection.
#
# The gpt-oss twin of wrappers/qwen_analysis/2_dataset_creation/3_rollouts/rollout_heldout_every_token.sh.
# This is what turns the held-out set into a LOCAL-BELIEF measurement: label_local does not
# exist until a rollout measures it, and join_rollouts.py is what puts it on the per-token table.
#
# STRIDE=1, NOT 64. Qwen strides because its chains are ~18.8k tokens (a dense grid there is
# ~1.17M prompts, about a week of GPU). heldout360 is 87,221 reasoning tokens in all, so the
# dense grid is affordable. It also buys what the strided Qwen arm cannot have: the commitment
# boundary resolved to the token, with no relapse hidden between two sampled cutoffs.
#
# ALREADY ON DISK, AND REUSED. OUT is the existing dense arm,
# reasoning_theatre/rollout_strategies_heldout360/every_token: 360 result files, rolled out with
# strategy every_token, stride 1, lens jlens on heldout360_lens, layer 15, each recorded in its
# own `strategy` block. --skip-existing makes this a no-op check on that host, and a resume
# anywhere a trajectory is missing. Do not point it at a new directory: it would re-roll
# ~88k prompts to produce what is already there.
#
# NO SELECTION: --strategy every_token chooses nothing, so no lens sits between the loudness
# axis and the label. Loudness is a covariate (needs_loudness = False).
#
# THE ENDPOINTS are kept, so the first and last eval is the same prompt as every other arm's.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)   # repo root

# size*/*.json: the layout run_inference.py's expand_paths documents. heldout360 holds exactly
# the 360 trajectory JSONs, but the glob keeps any stray summary or checkpoint JSON out.
TRAJ="/workspace/trajectories/heldout360/size*/*.json"
LENS_ROOT=/workspace/activations/heldout360_lens   # both lenses' mass tables, 7:23
OUT=/workspace/reasoning_theatre/rollout_strategies_heldout360/every_token   # REUSED; see the header

MODEL=openai/gpt-oss-20b
STRIDE=1               # dense: every reasoning token. See the header.
LAYER=15               # the mass-table layer whose direction mass is recorded per cutoff
LENS=jlens

DEVICE_MAP=cuda:0      # NOT "auto": spreading this MoE over several GPUs produces NaNs
BATCH_SIZE=16
MAX_BATCH_TOKENS=49152

mkdir -p "$OUT"
cd "$REPO"

# Print the cutoff count and the prompt tails for the first step, then exit.
#   DRY_RUN=1 bash wrappers/gptoss_analysis/direction/2_dataset_creation/3_rollouts/rollout_heldout_every_token.sh
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
