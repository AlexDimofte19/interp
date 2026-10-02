#!/usr/bin/env bash
# gpt-oss AGENT loudness on the eval 720: the mass tables the agent-probe figures read.
#
# NO MODEL, NO GATHER. mass_from_activations.py puts the layer-15 .pt files jlens_mass_l15
# already holds through the gather's own lens arithmetic (apply_lens_transport ->
# unembed -> logsumexp over the vocabulary's ids). The gather lenses exactly the tensor it
# saves, so this is the number a gather against agent_tokens.json would have written: its
# --self-check reproduces jlens_mass_l15's direction mass at L15 from its own .pt: exact where
# the .pt came from the scoring pass, within ~0.3 (bf16 batching noise) where the selecting
# gather re-ran the forward to save them -- but ONLY with --jlens_dir /workspace/jlens/gridenv;
# the jacobian_lens.pt one level up is the wikitext J and misses by 2-4 nats.
#
# Tokens: every eval-720 token with a layer-15 .pt in the pruned tree (jlens, logitlens and
# random picks), which covers every token of grid_binary_l15_random_split_eval.
# data/jlens/agent_tokens.json is the AGENT class of grid_tokens_full.json, copied by hand,
# not built by a vocabulary notebook. Resumable: a trajectory with both tables is skipped.
# Consumed by build_random_eval_tables.sh (ONLY=gptoss_agent), then the notebook's section 8.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

SOURCE=/workspace/activations/jlens_mass_l15
TRAJ=/workspace/activations/mass_eval720_view/trajectories   # restricts the walk to the 720
JLENS_DIR=/workspace/jlens/gridenv
SIGNAL_JSON=$REPO/data/jlens/agent_tokens.json
SIGNAL_NAME=agent
OUT=/workspace/activations/gptoss_agent_mass_l15_eval
LAYER=15

cd "$REPO"
uv run python -m telos_interp.loudness_analysis.mass_from_activations --self-check 3 \
    --source-root "$SOURCE" --trajectories-dir "$TRAJ" --jlens_dir "$JLENS_DIR" \
    --signal-json "$SIGNAL_JSON" --signal-name "$SIGNAL_NAME" --out "$OUT" --layer "$LAYER"
uv run python -m telos_interp.loudness_analysis.mass_from_activations \
    --source-root "$SOURCE" --trajectories-dir "$TRAJ" --jlens_dir "$JLENS_DIR" \
    --signal-json "$SIGNAL_JSON" --signal-name "$SIGNAL_NAME" --out "$OUT" --layer "$LAYER"
