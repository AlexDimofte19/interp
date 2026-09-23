#!/usr/bin/env bash
# gpt-oss P2, step 3c: the held-out gather -- no selection, no sampling. REUSED, NOT GATHERED.
#
# The gpt-oss twin of wrappers/qwen_analysis/2_dataset_creation/1_p2_selection/heldout_sample.sh.
# Qwen's held-out set is 72 trajectories gathered for P2. gpt-oss's is heldout360: 360
# trajectories, 10 per (size x complexity) cell, drawn from the pool the 3,600 never touched.
# Both trees it needs already exist:
#
#   heldout360_l15    a layer-15 .pt for EVERY reasoning token (87,221), no selection record,
#                     plus a jlens direction-mass table at layer 15. The tensors.
#   heldout360_lens   CSV-only, both lenses, direction-mass tables at 7:23. The loudness
#                     covariate, for both rulers.
#
# Qwen fits both into one tree by pinning --layers to the probe layer. Here they are two
# trees, so the per-token scorer is given --activations-dir heldout360_l15 and --lens-dir
# heldout360_lens (../../4_loudness_evaluation/score_probes_heldout_per_token.sh).
#
# NO SAMPLING, as on Qwen: every reasoning token has a .pt and a mass cell. SAMPLE_PERCENT=1.0
# records that, and the check fails on any sidecar that says otherwise.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)   # so uv finds pyproject.toml
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

ACT=/workspace/activations/heldout360_l15
LENS_DIR=/workspace/activations/heldout360_lens
TRAJ=/workspace/trajectories/heldout360
SIGNAL=direction_tokens_full.json

LAYER=15
SAMPLE_PERCENT=1.0     # NO THINNING

cd "$REPO"
uv run python "$HERE/check_reused_tree.py" --tree "$ACT" --trajectories "$TRAJ" \
    --lens jlens --layers "$LAYER" --signal "$SIGNAL" --pt-layer "$LAYER"
uv run python "$HERE/check_reused_tree.py" --tree "$LENS_DIR" --trajectories "$TRAJ" \
    --lens jlens --lens logitlens --layers 7:23 --signal "$SIGNAL"
