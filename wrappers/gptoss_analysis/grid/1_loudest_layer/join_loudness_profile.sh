#!/usr/bin/env bash
# gpt-oss P2 GRID, step 2: join the per-trajectory GRID mass tables into one token-level CSV per lens.
#
# The gpt-oss grid twin of wrappers/qwen_analysis/1_loudest_layer/join_loudness_profile.sh,
# pointed at the tree sample_loudness_profile.sh beside this file writes. Each cell is
# log P(any grid word) at that (token, layer), over the pruned 642-token grid vocabulary.
#
# CPU ONLY. Two lenses, two files, never pooled. The joiner refuses to mix vocabularies.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)   # so uv finds pyproject.toml

TREE=/workspace/activations/gptoss_grid_loudness_profile
OUT_DIR=/workspace/loudness_evaluation/gptoss_p2_grid_layer_profile

mkdir -p "$OUT_DIR"
cd "$REPO"

for LENS in jlens logitlens; do
    uv run python -m telos_interp.loudness_analysis.join_mass_tables "$TREE" \
        --lens "$LENS" \
        --out "$OUT_DIR/${LENS}_tokens.csv"
done

echo
echo "wrote $OUT_DIR/{jlens,logitlens}_tokens.csv"
echo "next: wrappers/gptoss_analysis/grid/1_loudest_layer/gptoss_p2_grid_loudest_layer.ipynb"
