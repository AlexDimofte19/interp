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
# MEASURED, since "the cheap end of this pipeline" is about compute, not I/O: this writes
# 1,169,734 .pt files, and /workspace is MooseFS, which tops out near 100 files/s and does
# NOT scale with --io-workers (1 thread 52/s, 8 -> 107, 16 -> 95, 32 -> 88, 64 -> 91). So
# raising io-workers past its default of 16 makes it worse. The 2026-09-17 run measured
# ~40 files/s sustained -- the BOTTOM of that range, not the top -- so budget ~8 h for the
# write, not the ~3.5 h the 100 files/s figure suggests. Kept at 1.0 anyway: the argument
# above is about what the analysis needs, and that has not changed.
#
# Set SAMPLE_PERCENT below 1.0 only if this turns out to be unaffordable after all, and
# expect to re-run it whole if the analysis later wants the tokens back.
#
# The fit and the vocabulary are both in place; step 1 ran against them.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)   # so uv finds pyproject.toml

MODEL=Qwen/Qwen3.6-35B-A3B
DATASET=/workspace/trajectories/qwen3.6-35b/replayed_single_step/heldout_72
JLENS_DIR=/workspace/jlens/qwen3_6_35b
SIGNAL_JSON=/workspace/repo/interp/data/jlens/qwen/direction_tokens_full_qwen3-6-35b-a3b.json
# ^ the SAME vocabulary the step-1 profile was gathered against. A mass table is not
# self-describing, so pointing this elsewhere silently produces numbers step 1's layer
# choice has no bearing on. Its fingerprint is recorded in every .meta.json written here.
OUT=/workspace/activations/qwen_p2_heldout

LAYER=27               # the probe's layer; keep equal to the two selecting runs
SAMPLE_PERCENT=1.0     # NO THINNING: every reasoning token gets a .pt. See the header.
SAMPLE_SEED=42

LENS=both              # both lenses' CSVs and mass tables, from one forward pass
LAYERS=$LAYER          # CSV, mass table and .pt all at the one layer
BATCH_SIZE=256         # reasoning tokens per lens matmul, caps the [B, vocab] logits
FORWARD_BATCH_SIZE=1   # one 18k-token chain at a time; the default of 4 pads four into one pass

# Single GPU: device_map="auto" across several produces NaNs for this MoE, as it does for gpt-oss.
export CUDA_VISIBLE_DEVICES=0

# 66 GiB of weights on an 80 GiB card leaves ~13 GiB for a 33k-token chain. Step 1 died there
# with 7.58 GiB reserved-but-unallocated -- fragmentation, not a real shortage.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
