#!/usr/bin/env bash
# Qwen P2, step 2: join the per-trajectory mass tables into one token-level CSV per lens.
#
# sample_loudness_profile.sh leaves one direction-mass table per trajectory -- one row per
# reasoning token, one L{layer} column per layer. This concatenates them, per lens, into a
# single CSV with a `trajectory` column saying where each row came from. That is all it
# does: no aggregation, no scoring. The mean per layer and the plots are the notebook's job
# (qwen_p2_loudest_layer.ipynb, beside this file).
#
# CPU ONLY, and it reads nothing but CSVs. Safe to run while a fit is on the GPUs.
#
# TWO LENSES, TWO FILES. The gather ran --lens both. They are joined separately and never
# concatenated together: at layer 15 on gpt-oss the two lenses' top-20 sets overlap only
# about half, so a row pooled across lenses is not a quantity. The jlens file is what P2
# uses; the logitlens one is the baseline that says whether the Jacobian bought anything.
#
# The joiner refuses to concatenate tables gathered against different vocabularies -- a mass
# table is not self-describing, and mixing two produces numbers with no referent.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)   # so uv finds pyproject.toml

# The tree sample_loudness_profile.sh wrote. Its $OUT, unchanged.
TREE=/workspace/activations/qwen_loudness_profile_p20
OUT_DIR=/workspace/loudness_evaluation/qwen_p2_layer_profile

mkdir -p "$OUT_DIR"
cd "$REPO"

for LENS in jlens logitlens; do
    uv run python -m telos_interp.loudness_analysis.join_mass_tables "$TREE" \
        --lens "$LENS" \
        --out "$OUT_DIR/${LENS}_tokens.csv"
done

echo
echo "wrote $OUT_DIR/{jlens,logitlens}_tokens.csv"
echo "next: wrappers/qwen_analysis/loudest_layer/qwen_p2_loudest_layer.ipynb"
