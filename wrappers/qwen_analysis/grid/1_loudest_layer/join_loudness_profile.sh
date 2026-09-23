#!/usr/bin/env bash
# Qwen P2 GRID, step 2: join the per-trajectory GRID mass tables into one token-level CSV per lens.
#
# The grid twin of ../../1_loudest_layer/join_loudness_profile.sh: identical, pointed at the
# tree sample_loudness_profile.sh beside this file writes. Each cell is
# log P(any grid word) at that (token, layer), over the pruned Qwen grid vocabulary.
#
# CPU ONLY, and it reads nothing but CSVs.
#
# TWO LENSES, TWO FILES, never pooled: at a single layer the two lenses' top-20 sets overlap
# only about half, so a row pooled across lenses is not a quantity.
#
# The joiner refuses to concatenate tables gathered against different vocabularies, so this
# cannot silently mix the grid tree with the direction one.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)   # so uv finds pyproject.toml

# The tree sample_loudness_profile.sh wrote. Its $OUT, unchanged -- note the qwen/ level.
TREE=/workspace/activations/qwen/qwen_grid_loudness_profile_p05
OUT_DIR=/workspace/loudness_evaluation/qwen_p2_grid_layer_profile

mkdir -p "$OUT_DIR"
cd "$REPO"

for LENS in jlens logitlens; do
    uv run python -m telos_interp.loudness_analysis.join_mass_tables "$TREE" \
        --lens "$LENS" \
        --out "$OUT_DIR/${LENS}_tokens.csv"
done

echo
echo "wrote $OUT_DIR/{jlens,logitlens}_tokens.csv"
echo "next: wrappers/qwen_analysis/grid/1_loudest_layer/qwen_p2_grid_loudest_layer.ipynb"
