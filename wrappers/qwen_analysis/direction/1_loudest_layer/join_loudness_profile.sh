#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
TREE=/workspace/activations/qwen_direction_loudness_profile
OUT_DIR=/workspace/loudness_evaluation/qwen_p2_layer_profile
mkdir -p "$OUT_DIR"
cd "$REPO"
for LENS in jlens logitlens; do
    uv run python -m telos_interp.loudness_analysis.join_mass_tables "$TREE" \
        --lens "$LENS" \
        --out "$OUT_DIR/${LENS}_tokens.csv"
done
