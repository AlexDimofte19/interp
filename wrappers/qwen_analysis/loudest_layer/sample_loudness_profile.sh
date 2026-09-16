#!/usr/bin/env bash
# Qwen P2, step 1: the all-layers loudness profile, on a 20% sample of the reasoning tokens.
#
# CSV-ONLY. No .pt is written and no selection is made -- this run exists to answer one
# question, "which layer is the jlens loudest at, over the whole dataset", and the answer
# comes from scripts/jlens_layer_profile.py reading the direction-mass tables this leaves
# behind. The layer it names is what the later selecting gather pins with
# --select-candidate-layers. Do not take it from a manifest's argmax: that mean is
# conditional on selection, this one is not.
#
# WHY --direction-mass-json AND NOT --signal-json. --signal-json switches the script into
# selective mode (a filter, a second forward pass, a selection record). Passing the
# vocabulary as --direction-mass-json gets the mass table with none of that, which is all
# this run wants. The control arm is therefore NOT drawn here -- it must be drawn in the
# selecting gather that follows, on the full chain, before anything is pruned.
#
# WHY 0.2. The forward pass is linear in chain length and cannot be shortened, but the lens
# transport, the unembed and the mass reduction scale with (tokens x layers), and on Qwen's
# ~18k-token chains over a ~250k vocabulary that is the dominant term. A uniform draw is
# unbiased for a per-layer mean, so the profile keeps its shape and only its error bars
# widen. The draw is seeded per (trajectory, step) and recorded in each table's .meta.json.
#
# STILL BLOCKED ON THE FIT. The §7 code blockers are fixed (models.py carries the lens
# filename, the unembed cache and its weight keys, the config path and the target layer),
# but JLENS_DIR holds ckpt.pt, not the finished
# Qwen3.6-35B-A3B_gridenv_jacobian_lens.pt, and SIGNAL_JSON has not been uploaded.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)   # so uv finds pyproject.toml

# The HF repo the weights come from. The trajectory JSONs record "gsarti/qwen3.6-35b",
# the Together endpoint alias, which is not a HF repo -- models.py knows it as an alias so
# this would resolve without the flag, but it is passed explicitly because a run this long
# should say out loud which checkpoint it read.
MODEL=Qwen/Qwen3.6-35B-A3B
DATASET=/workspace/trajectories/qwen3.6-35b/replayed_single_step/mass_train_576
JLENS_DIR=/workspace/jlens/qwen3_6_35b
SIGNAL_JSON=/workspace/jlens/qwen_direction_tokens.json   # NOT YET UPLOADED -- point this at your local vocabulary
OUT=/workspace/activations/qwen_loudness_profile_p20

SAMPLE_PERCENT=0.2
SAMPLE_SEED=42
LENS=both              # jlens + logitlens from ONE forward pass: one extra unembed, not a second pass
LAYERS=all             # what the CSV and the mass tables cover; jlens is restricted to its fitted layers anyway
BATCH_SIZE=256         # reasoning tokens per lens matmul, caps the [B, vocab] logits
FORWARD_BATCH_SIZE=1   # one 18k-token chain at a time; the default of 4 pads four of them into one pass

# Single GPU: device_map="auto" across several produces NaNs for this MoE, as it does for gpt-oss.
export CUDA_VISIBLE_DEVICES=0

cd "$REPO"
uv run --extra gpu python telos_interp/loudness_analysis/build_loudness_tables.py \
    --trajectory-paths "$DATASET" \
    --activations-dir "$OUT" \
    --model-id "$MODEL" \
    --jlens_dir "$JLENS_DIR" \
    --direction-mass-json "$SIGNAL_JSON" \
    --data_sample_p "$SAMPLE_PERCENT" \
    --data-sample-seed "$SAMPLE_SEED" \
    --lens "$LENS" \
    --layers "$LAYERS" \
    --no-save-activations \
    --batch-size "$BATCH_SIZE" \
    --forward-batch-size "$FORWARD_BATCH_SIZE" \
    --device cuda
