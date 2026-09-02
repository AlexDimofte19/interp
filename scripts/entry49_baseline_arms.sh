#!/usr/bin/env bash
# ICLR entry 49: the three belief-label baselines the heldout-360 loudness report was missing.
#
# Entry 48 reads ten probes on the heldout 360. In those ten, the LABEL axis and the
# SELECTION axis are confounded: every local-belief probe was selected by the jlens, and the
# only non-jlens selection (`random`) exists only with the final-action label. So "+11.7 pp
# for the belief label" cannot be separated from "the jlens picked the tokens", and no second
# lens has ever been trained against the belief label. These three arms cross the two:
#
#   recorded_selection        random selection  -> belief   the matched control for P2, on the
#                                                           SAME tokens the existing random /
#                                                           final-action probe trained on
#   jlens_argmax_per_sentence logitlens P1      -> belief   loudest token of each sentence
#   jlens_top_k_global        logitlens P2      -> belief   the 20 globally loudest tokens
#
# The two logitlens arms reuse the jlens strategies unchanged: loudness comes from
# MassTableLoudness, which is parameterised by --lens, so LENS=logitlens is the whole change.
# They need the logitlens direction-mass tables, which the training tree did NOT have --
# scripts/jlens_mass_l15.sh with LENS=logitlens builds them (step 1, ~1.6 h; see the header
# of that script for the invocation).
#
# WHY THE RANDOM ARM REPLAYS RATHER THAN RE-DRAWS. Its tokens must be the ones the existing
# random probe trained on, or the pair stops isolating the label. That seeded uniform draw
# was taken before the activation tree was pruned to the selection and can never be made
# again, so the arm reads arms.random.picks out of {stem}_jlens_selection.json. Its loudness
# root is therefore the JLENS tree, which is where those records live.
#
# ALL THREE SHARE ONE TRAJECTORY SET -- the 3600 of mass_l15_names.txt, which is also the set
# the logitlens tree was pinned to by name. Anything else and the 2880/720 partition of
# next_action_mass_l15_eval_names.txt stops applying and no arm can be compared to another.
#
# ~14-18 h on one GPU (measured siblings: jlens_argmax_per_sentence 6h04m,
# jlens_top_k_global 3h55m over the same 3600). Every arm resumes from what is on disk.
#
# Usage:
#   bash scripts/entry49_baseline_arms.sh            # all three, in order
#   bash scripts/entry49_baseline_arms.sh random     # just the control arm
#   DRY_RUN=1 bash scripts/entry49_baseline_arms.sh  # cutoffs only, no model
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

NAMES_FILE=${NAMES_FILE:-/workspace/reasoning_theatre/rollout_strategies/mass_l15_names.txt}
OUT_ROOT=${OUT_ROOT:-/workspace/reasoning_theatre/rollout_strategies_baselines}
JLENS_ROOT=${JLENS_ROOT:-/workspace/activations/jlens_mass_l15}
LOGITLENS_ROOT=${LOGITLENS_ROOT:-/workspace/activations/logitlens_mass_l15}
export DRY_RUN=${DRY_RUN:-}

WANT=${1:-all}

if [ "$WANT" = all ] || [ "$WANT" = random ]; then
    echo "### arm 1/3: random selection -> belief (replaying the recorded control)"
    NAMES_FILE="$NAMES_FILE" OUT_ROOT="$OUT_ROOT" \
    LENS=jlens LENS_ROOT="$JLENS_ROOT" SELECTION_ARM=random \
        bash "$HERE/inference_oss/run_inference_strategies.sh" recorded_selection
fi

if [ "$WANT" = all ] || [ "$WANT" = logitlens ]; then
    echo "### arms 2-3/3: logitlens P1 and P2 -> belief"
    NAMES_FILE="$NAMES_FILE" OUT_ROOT="$OUT_ROOT" \
    LENS=logitlens LENS_ROOT="$LOGITLENS_ROOT" \
        bash "$HERE/inference_oss/run_inference_strategies.sh" \
            jlens_argmax_per_sentence jlens_top_k_global
fi

echo
echo "Arms under $OUT_ROOT:"
for d in "$OUT_ROOT"/*/; do
    [ -d "$d" ] && echo "  $(basename "$d"): $(find "$d" -maxdepth 1 -name '*.json' | wc -l) result file(s)"
done
