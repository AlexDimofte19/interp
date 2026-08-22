#!/usr/bin/env bash
# next_action prepare with jlens token/layer selection, restricted to comp 0.0/0.2/0.4.
#
# prepare_activations_for_probing has no complexity filter -- it processes every folder
# under --activations-dir -- so link the wanted trajectories into a view and prepare that.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)   # so uv finds pyproject.toml
ACT=${ACT:-/workspace/activations/jlens_reasoning_tokens}
TRAJ=${TRAJ:-/workspace/trajectories/reveng/trajectories_train_single_step}
VIEW=${VIEW:-${ACT}_comp0.0-0.2-0.4}
OUT=${OUT:-/workspace/prepared/next_action_comp0.0-0.2-0.4_jlens}

rm -rf "$VIEW"; mkdir -p "$VIEW/activations" "$VIEW/trajectories"
for c in 0.0 0.2 0.4; do
    for d in "$ACT"/size*/*_comp${c}_*/; do
        s=$(basename "$(dirname "$d")"); n=$(basename "$d")   # sizeN, trajectory stem
        mkdir -p "$VIEW/activations/$s" "$VIEW/trajectories/$s"
        ln -sfn "${d%/}" "$VIEW/activations/$s/$n"
        ln -sfn "$TRAJ/$s/$n.json" "$VIEW/trajectories/$s/$n.json"
    done
done

uv run --project "$REPO" interp-cli prepare_activations_for_probing \
    --activations-dir "$VIEW/activations" \
    --trajectories-dir "$VIEW/trajectories" \
    --probe-type next_action \
    --layers 7:23 --steps all --output-indices all \
    --token-selection jlens_direction --layer-selection jlens_direction \
    --num-tokens 20 --num-layers 3 \
    --direction-tokens-path /workspace/jlens/direction_tokens_full.json \
    --output-path "$OUT" --verbose

# Matched control (same N/M/seed) -- required for the jlens run to mean anything:
#   uv run interp-cli ... --token-selection random --layer-selection random \
#       --output-path "${OUT%_jlens}_random"
