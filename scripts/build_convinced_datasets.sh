#!/usr/bin/env bash
# The three loudness -> convinced datasets, as produced. A record, like configs/**/*.conf.
#
#   train   2880 train trajectories, EVERY reasoning token
#   val      720 eval trajectories,  EVERY reasoning token
#   eval     360 held-out trajectories, EVERY reasoning token
#
# Every token everywhere, so the three differ only by which trajectories they cover. An earlier
# version of this script thinned train/val to 20 tokens per trajectory (a uniform-random arm and
# a jlens top-20 arm); that thinning is gone. With no selection the two 720-trajectory arms would
# have been the same rows, so there is one val set, not two.
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

$PY scripts/build_convinced_dataset.py \
    --lens-root "$LENS_ROOT" --exclude-names "$EVAL_NAMES" \
    --selection-arm none --out "$OUT/train_all.csv"

$PY scripts/build_convinced_dataset.py \
    --lens-root "$LENS_ROOT" --names "$EVAL_NAMES" \
    --selection-arm none --out "$OUT/val_all.csv"

$PY scripts/build_convinced_dataset.py \
    --lens-root "$HELDOUT_ROOT" \
    --selection-arm none --out "$OUT/eval_heldout360_all.csv"
