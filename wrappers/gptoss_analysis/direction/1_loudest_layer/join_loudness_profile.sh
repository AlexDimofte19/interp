#!/usr/bin/env bash
# gpt-oss P2, step 2: join the per-trajectory direction-mass tables into one token-level CSV per lens.
#
# The gpt-oss twin of wrappers/qwen_analysis/1_loudest_layer/join_loudness_profile.sh. Two
# differences, both because the trees are reused (see sample_loudness_profile.sh):
#
#   * each lens lives in its OWN tree. jlens_mass_l15 holds the jlens tables and
#     logitlens_mass_l15 the logitlens ones, so the loop pairs a lens with its tree
#     rather than reading one tree twice.
#   * the rows are every reasoning token of the 3,600 at 7:23, not a 20% sample.
#
# CPU ONLY, and it reads nothing but CSVs. Two lenses, two files, never pooled.
#
# The joiner refuses to concatenate tables gathered against different vocabularies.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)   # so uv finds pyproject.toml

OUT_DIR=/workspace/loudness_evaluation/gptoss_p2_layer_profile

mkdir -p "$OUT_DIR"
cd "$REPO"

for pair in jlens:/workspace/activations/jlens_mass_l15 logitlens:/workspace/activations/logitlens_mass_l15; do
    LENS=${pair%%:*}
    TREE=${pair#*:}
    uv run python -m telos_interp.loudness_analysis.join_mass_tables "$TREE" \
        --lens "$LENS" \
        --out "$OUT_DIR/${LENS}_tokens.csv"
done

echo
echo "wrote $OUT_DIR/{jlens,logitlens}_tokens.csv"
echo "next: wrappers/gptoss_analysis/direction/1_loudest_layer/gptoss_p2_loudest_layer.ipynb"
