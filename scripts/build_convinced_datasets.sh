#!/usr/bin/env bash
# The four loudness -> convinced datasets, as produced. A record, like configs/**/*.conf.
#
#   train        2880 train trajectories, uniform random 20 tokens each
#   validation1  the 720 eval trajectories, SAME uniform-random selection as training
#   validation2  the 720 eval trajectories, jlens top-20 by logprob_mass_full ("P2")
#   evaluation   360 held-out trajectories, EVERY reasoning token
#
# validation1 and validation2 cover the same trajectories and differ only in HOW their tokens
# were chosen; train and validation1 share a selection mechanism and differ only in the split.
#
# THE SPLIT LIST IS NOT INTERCHANGEABLE. /workspace/splits/eval_trajectories_720.txt and
# lens_trajectories_3600.txt belong to the older count-era tree and overlap this one by only
# 63 and 348 names -- using them would put ~657 eval trajectories into training, silently.
# The mass tree's split is next_action_mass_l15_eval_names.txt.
set -euo pipefail

PY=${PY:-.venv/bin/python}
LENS_ROOT=${LENS_ROOT:-/workspace/activations/jlens_mass_l15}
HELDOUT_ROOT=${HELDOUT_ROOT:-/workspace/activations/heldout360_lens}
EVAL_NAMES=${EVAL_NAMES:-/workspace/prepared/next_action_mass_l15_eval_names.txt}
OUT=${OUT:-/workspace/reasoning_theatre/convinced_classifier}

mkdir -p "$OUT"

uv run python build_convinced_dataset.py \
    --lens-root "$LENS_ROOT" --exclude-names "$EVAL_NAMES" \
    --selection-arm random --out "$OUT/train_random20.csv"

uv run python build_convinced_dataset.py \
    --lens-root "$LENS_ROOT" --names "$EVAL_NAMES" \
    --selection-arm random --out "$OUT/val1_random20.csv"

uv run python build_convinced_dataset.py \
    --lens-root "$LENS_ROOT" --names "$EVAL_NAMES" \
    --selection-arm jlens --out "$OUT/val2_jlens20.csv"

uv run python build_convinced_dataset.py \
    --lens-root "$HELDOUT_ROOT" \
    --selection-arm none --out "$OUT/eval_heldout360_all.csv"
