#!/usr/bin/env bash
# gpt-oss P2, step 3b: the selecting gather on the EVAL set. REUSED, NOT GATHERED.
#
# The same check as p2_training_selection.sh beside this file, over the other half of the
# mass-era partition: the 720 of mass_eval720_view, byte-for-byte the pinned
# next_action_mass_l15_eval_names.txt every gpt-oss probe on disk was scored against. Read
# that script's header for where each arm lives and how it differs from Qwen P2.
#
# Unlike Qwen's mass_eval_144 (52 of 144 gathered), this eval set is complete, but it is a
# plain random draw, not stratified: its (size x complexity) cells run 14-29.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)   # so uv finds pyproject.toml
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

JLENS_TREE=/workspace/activations/jlens_mass_l15         # arms jlens, random
LOGITLENS_TREE=/workspace/activations/logitlens_mass_l15 # arm logitlens
TRAJ=/workspace/activations/mass_eval720_view/trajectories
SIGNAL=direction_tokens_full.json

LAYER=15
SAMPLE_PERCENT=1.0     # the whole chain; no --data_sample_p was passed

cd "$REPO"
uv run python "$HERE/check_reused_tree.py" --tree "$JLENS_TREE" --trajectories "$TRAJ" \
    --lens jlens --layers "$LAYER" --signal "$SIGNAL" \
    --arms jlens,random --candidate-layer "$LAYER" --pt-layer "$LAYER"
uv run python "$HERE/check_reused_tree.py" --tree "$LOGITLENS_TREE" --trajectories "$TRAJ" \
    --lens logitlens --layers "$LAYER" --signal "$SIGNAL" \
    --arms logitlens --candidate-layer "$LAYER" --pt-layer "$LAYER"
