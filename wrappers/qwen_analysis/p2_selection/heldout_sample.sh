#!/usr/bin/env bash
# Qwen P2, step 3c: the held-out gather -- no selection, sampling only.
#
# The two selecting runs beside this one keep 20 tokens per trajectory per arm. This keeps
# EVERY token of the sample, at one layer, with no arms and no selection record. That is the
# point: the held-out set is where a probe is scored on tokens it did not choose, so letting
# a lens pick them would be the same circularity the control arm exists to rule out.
#
# WHAT MAKES IT UNFILTERED: no --signal-json. That flag is what switches the script into
# selective mode; without it there is no filter, no second pass and no
# {stem}_jlens_selection.json, and every reasoning token that survives --data_sample_p gets
# a .pt. --direction-mass-json is passed in its place so the mass tables are still written
# -- they are the loudness covariate the held-out analysis crosses probe accuracy against,
# and without a vocabulary named somewhere no table is written at all.
#
# NO --select-* FLAGS APPEAR HERE. They are inert without --signal-json, and listing them
# would suggest a selection is happening.
#
# ONE PASS, NOT TWO, BECAUSE THE LAYERS ARE PINNED. score_probes_per_token.py takes an
# --activations-dir (a .pt at the probe's layer for every token) and a --lens-dir (the CSVs),
# and the inventory calls for two passes because those two normally want different layer
# coverage -- one layer of tensors, every layer of CSV. Here both are pinned to $LAYER, so a
# single tree is both, and --lens-dir can be left to default to --activations-dir. If you
# later want the full (token x layer) profile over the held-out set, that is a second
# CSV-only pass with --layers all --no-save-activations, not a change to this one.
#
# IT DOES NOT SAMPLE, AND THAT IS DELIBERATE. SAMPLE_PERCENT=1.0, so --data_sample_p is a
# no-op and every reasoning token lands on disk. The flag is still passed, so the sidecar
# records that this was a full pass rather than leaving it unstated.
#
# The reasoning (inventory §2): the held-out eval reads every token. A uniform draw would
# keep the means honest -- accuracy-by-loudness-decile would stay unbiased -- but it thins
# each chain, so per-trajectory and per-sentence curves get noisier, and a token that is not
# in the tree cannot be scored later without re-running the gather. 72 trajectories at one
# layer is the cheap end of this pipeline, so there is nothing to buy by thinning it.
#
# Set SAMPLE_PERCENT below 1.0 only if this turns out to be unaffordable after all, and
# expect to re-run it whole if the analysis later wants the tokens back.
#
# STILL BLOCKED ON THE FIT and on the vocabulary upload, exactly as the other two are.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)   # so uv finds pyproject.toml

MODEL=Qwen/Qwen3.6-35B-A3B
DATASET=/workspace/trajectories/qwen3.6-35b/replayed_single_step/heldout_72
JLENS_DIR=/workspace/jlens/qwen3_6_35b
SIGNAL_JSON=/workspace/jlens/qwen_direction_tokens.json   # NOT YET UPLOADED -- point this at your local vocabulary
OUT=/workspace/activations/qwen_p2_heldout

LAYER=39               # the probe's layer; keep equal to the two selecting runs
SAMPLE_PERCENT=1.0     # NO THINNING: every reasoning token gets a .pt. See the header.
SAMPLE_SEED=42

LENS=both              # both lenses' CSVs and mass tables, from one forward pass
LAYERS=$LAYER          # CSV, mass table and .pt all at the one layer
BATCH_SIZE=256         # reasoning tokens per lens matmul, caps the [B, vocab] logits
FORWARD_BATCH_SIZE=1   # one 18k-token chain at a time; the default of 4 pads four into one pass

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
    --batch-size "$BATCH_SIZE" \
    --forward-batch-size "$FORWARD_BATCH_SIZE" \
    --device cuda
