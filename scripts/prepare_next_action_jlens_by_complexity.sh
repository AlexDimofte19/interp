#!/usr/bin/env bash
# next_action prepare from a pruned activation tree, restricted to comp 0.0/0.2/0.4.
#
# Every arm comes from the {stem}_jlens_selection.json that jlens_reasoning_tokens.py (with
# --signal-json, or --extend) or delete_non_jlens_selected.py wrote. Re-scoring a CSV would
# still find the right lens tokens, but on a pruned tree "draw N tokens uniformly" can only
# draw from what survived -- which is not a uniform draw over the reasoning chain. The
# control has to be read back, not recomputed.
#
# ARMS names which arms to prepare. An arm the record does not hold prepares nothing; add
# it first with jlens_reasoning_tokens.py --extend.
#
# prepare_activations_for_probing has no complexity filter -- it processes every folder
# under --activations-dir -- so link the wanted trajectories into a view and prepare that.
#
# Narrow the selection at training time, not here: split_next_action_manifest.py takes
# --tokens-per-trajectory and --layers-per-token off a single prepared dataset, so top-1 /
# top-2 / top-3 need one prepare between them, not three.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)   # so uv finds pyproject.toml
ACT=${ACT:-/workspace/activations/jlens_reasoning_tokens}
TRAJ=${TRAJ:-/workspace/trajectories/reveng/trajectories_train_single_step}
VIEW=${VIEW:-${ACT}_comp0.0-0.2-0.4}
OUT=${OUT:-/workspace/prepared/next_action_comp0.0-0.2-0.4_jlens}
LAYERS=${LAYERS:-7:23}   # narrows the recorded layers; e.g. 15 for a layer-15-only dataset
ARMS=${ARMS:-"jlens logitlens random"}

rm -rf "$VIEW"; mkdir -p "$VIEW/activations" "$VIEW/trajectories"
for c in 0.0 0.2 0.4; do
    for d in "$ACT"/size*/*_comp${c}_*/; do
        s=$(basename "$(dirname "$d")"); n=$(basename "$d")   # sizeN, trajectory stem
        mkdir -p "$VIEW/activations/$s" "$VIEW/trajectories/$s"
        ln -sfn "${d%/}" "$VIEW/activations/$s/$n"
        ln -sfn "$TRAJ/$s/$n.json" "$VIEW/trajectories/$s/$n.json"
    done
done

# All arms from the same record. A lens number without its matched control means nothing,
# so the control is run alongside rather than left as a TODO. `|| true` because an arm the
# record lacks is a legitimate state, not a reason to abandon the arms that are there.
for arm in $ARMS; do
    uv run --project "$REPO" interp-cli prepare_activations_for_probing \
        --activations-dir "$VIEW/activations" \
        --trajectories-dir "$VIEW/trajectories" \
        --probe-type next_action \
        --layers "$LAYERS" --steps all --output-indices all \
        --token-selection "recorded_${arm}" \
        --output-path "${OUT%_jlens}_${arm}" --verbose \
        || echo "!! ${arm}: prepared nothing (is that arm in the record?)" >&2
done

echo ""
echo "Prepared:"
for arm in $ARMS; do
    d="${OUT%_jlens}_${arm}"
    [ -f "$d/manifest.json" ] && echo "  $d" || echo "  $d  (MISSING)"
done
echo "Train them with scripts/train_next_action_direction_probe.sh"
