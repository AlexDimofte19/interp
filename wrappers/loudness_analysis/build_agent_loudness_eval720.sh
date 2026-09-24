#!/usr/bin/env bash
# gpt-oss AGENT loudness on the eval 720: the mass tables the agent-probe figures read.
#
# GPU. One CSV-only forward pass per trajectory (--no-save-activations: no .pt is written,
# no selection is made) over the mass-era eval view, scored against data/jlens/agent_tokens.json
# -- the AGENT class of grid_tokens_full.json, copied by hand, NOT built by a vocabulary
# notebook. The trees this line already has were baked against the direction and grid
# vocabularies, and a mass table is fixed at gather time, so agent loudness needs its own pass.
#
# Layer 15 only: the binary agent probes (/workspace/probes/grid_binary_l15) read layer 15.
# Both lenses from the one pass. The sidecars name the vocabulary `agent`, which is what
# join_signal_loudness.py --signal-name agent checks before reading a cell.
#
# The 720 are the pinned next_action_mass_l15_eval_names.txt, the trajectories of
# grid_binary_l15_random_split_eval that the probes were scored on (eval_agent_*_eval720.json).
# Resumable: a trajectory whose tables exist is skipped. Consumed by build_random_eval_tables.sh
# (ONLY=gptoss_agent), then probe_performance_by_loudness.ipynb cell 9.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)   # so uv finds pyproject.toml

MODEL=openai/gpt-oss-20b
DATASET=/workspace/activations/mass_eval720_view/trajectories
JLENS_DIR=/workspace/jlens   # as p2_eval_selection.sh: this host keeps the lens + unembed here
SIGNAL_JSON=$REPO/data/jlens/agent_tokens.json
SIGNAL_NAME=agent
OUT=/workspace/activations/gptoss_agent_mass_l15_eval

LAYERS=15              # the agent probes' layer
LENS=both
BATCH_SIZE=256         # reasoning tokens per lens matmul

# Single GPU: device_map="auto" across several produces NaNs for this MoE.
export CUDA_VISIBLE_DEVICES=0

[ -f "$SIGNAL_JSON" ] || { echo "!! agent vocabulary not found: $SIGNAL_JSON" >&2; exit 1; }

cd "$REPO"
uv run --extra gpu python telos_interp/loudness_analysis/build_loudness_tables.py \
    --trajectory-paths "$DATASET" \
    --activations-dir "$OUT" \
    --model-id "$MODEL" \
    --jlens_dir "$JLENS_DIR" \
    --signal-json "$SIGNAL_JSON" \
    --signal-name "$SIGNAL_NAME" \
    --no-save-activations \
    --lens "$LENS" \
    --layers "$LAYERS" \
    --batch-size "$BATCH_SIZE" \
    --device cuda
