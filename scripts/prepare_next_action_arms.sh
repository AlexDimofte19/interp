#!/usr/bin/env bash
# Prepare one next_action probe dataset per arm: jlens, logitlens, random.
#
# Every arm is read from the {stem}_jlens_selection.json that jlens_reasoning_tokens.py
# (--signal-json, or --extend) or delete_non_jlens_selected.py wrote. Re-scoring a lens CSV
# would still find the right tokens, but on a pruned tree "draw N tokens uniformly" can only
# draw from what survived -- which is not a uniform draw over the reasoning chain. The
# control has to be read back, not recomputed. Hence recorded_* rather than <lens>_direction.
#
# Cheap: next_action copies no activations. Each manifest just references .pt files in the
# tree through `activations_root`, so three arms cost three passes over the JSON, not three
# copies of the data.
#
#   ./scripts/prepare_next_action_arms.sh
#   ARMS="jlens logitlens" ./scripts/prepare_next_action_arms.sh    # skip the control
#   LAYERS=15 OUT=/workspace/prepared/next_action_l15 ./scripts/prepare_next_action_arms.sh
#   COMPLEXITIES="0.0 0.2 0.4" ./scripts/prepare_next_action_arms.sh
#
# An arm the record does not hold prepares nothing and is reported as MISSING rather than
# aborting the others -- add it with scripts/jlens_extend_logitlens.sh.
#
# Do not run this while delete_non_jlens_selected.py or an --extend gather is writing.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)   # so uv finds pyproject.toml

ACT=${ACT:-/workspace/activations/jlens_reasoning_tokens}
TRAJ=${TRAJ:-/workspace/trajectories/reveng/trajectories_train_single_step}
OUT=${OUT:-/workspace/prepared/next_action}          # datasets land at ${OUT}_${arm}

ARMS=${ARMS:-"jlens logitlens random"}
LAYERS=${LAYERS:-7:23}      # narrows the RECORDED layers; 15 gives a layer-15-only dataset
STEPS=${STEPS:-all}
COMPLEXITIES=${COMPLEXITIES:-}   # e.g. "0.0 0.2 0.4"; empty means the whole tree
VIEW=${VIEW:-${ACT}_view}        # only used when COMPLEXITIES is set

[ -d "$ACT" ]  || { echo "!! activations dir not found: $ACT"  >&2; exit 1; }
[ -d "$TRAJ" ] || { echo "!! trajectories dir not found: $TRAJ" >&2; exit 1; }

SRC_ACT=$ACT
SRC_TRAJ=$TRAJ

# prepare_activations_for_probing has no complexity filter -- it processes every folder under
# --activations-dir -- so when one is asked for, link the wanted trajectories into a view and
# prepare that instead. size{N}/{stem} nesting is preserved; prepare auto-detects it.
if [ -n "$COMPLEXITIES" ]; then
    [ "$VIEW" != "$ACT" ] || { echo "!! VIEW must not be ACT ($VIEW)" >&2; exit 1; }
    case "$VIEW" in /|/*[!/]*) ;; *) echo "!! VIEW looks unsafe: $VIEW" >&2; exit 1 ;; esac
    echo "Building view for complexities: $COMPLEXITIES"
    rm -rf "$VIEW"; mkdir -p "$VIEW/activations" "$VIEW/trajectories"
    linked=0
    for c in $COMPLEXITIES; do
        for d in "$ACT"/size*/*_comp${c}_*/; do
            [ -d "$d" ] || continue
            s=$(basename "$(dirname "$d")"); n=$(basename "$d")   # sizeN, trajectory stem
            mkdir -p "$VIEW/activations/$s" "$VIEW/trajectories/$s"
            ln -sfn "${d%/}" "$VIEW/activations/$s/$n"
            ln -sfn "$TRAJ/$s/$n.json" "$VIEW/trajectories/$s/$n.json"
            linked=$((linked + 1))
        done
    done
    [ "$linked" -gt 0 ] || { echo "!! no trajectories matched $COMPLEXITIES" >&2; exit 1; }
    echo "  linked $linked trajectory folder(s) into $VIEW"
    SRC_ACT="$VIEW/activations"
    SRC_TRAJ="$VIEW/trajectories"
fi

echo ""
echo "activations: $SRC_ACT"
echo "layers:      $LAYERS"
echo "arms:        $ARMS"
echo ""

failed=""
for arm in $ARMS; do
    echo "============================================================"
    echo "Preparing arm: $arm  ->  ${OUT}_${arm}"
    echo "============================================================"
    # An arm absent from the record selects nothing, which surfaces as prepare's
    # "No activations were extracted". That is the correct outcome, and it must not take the
    # other arms down with it.
    if uv run --project "$REPO" interp-cli prepare_activations_for_probing \
        --activations-dir "$SRC_ACT" \
        --trajectories-dir "$SRC_TRAJ" \
        --probe-type next_action \
        --layers "$LAYERS" \
        --steps "$STEPS" \
        --output-indices all \
        --token-selection "recorded_${arm}" \
        --output-path "${OUT}_${arm}" \
        --verbose
    then
        echo "   ok"
    else
        echo "!! ${arm}: prepared nothing -- is that arm in the selection records?" >&2
        failed="$failed $arm"
    fi
    echo ""
done

echo "============================================================"
echo "SUMMARY"
echo "============================================================"
for arm in $ARMS; do
    d="${OUT}_${arm}"
    if [ -f "$d/manifest.json" ]; then
        n=$(uv run --project "$REPO" python -c \
            "import json,sys; print(len(json.load(open(sys.argv[1]))['samples']))" \
            "$d/manifest.json" 2>/dev/null || echo "?")
        printf '  %-12s %8s samples   %s\n' "$arm" "$n" "$d"
    else
        printf '  %-12s %8s              %s\n' "$arm" "MISSING" "$d"
    fi
done

if [ -n "$failed" ]; then
    echo ""
    echo "Arms that prepared nothing:$failed"
    echo "  A lens arm missing from the records is added by scripts/jlens_extend_logitlens.sh;"
    echo "  pruning removed the tokens it needs, so prepare cannot re-derive it."
fi

echo ""
echo "Train them with:"
echo "  ARMS=\"$ARMS\" PREPARED=$OUT ./scripts/train_next_action_direction_probe.sh"
