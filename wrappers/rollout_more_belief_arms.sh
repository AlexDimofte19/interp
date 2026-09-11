#!/usr/bin/env bash
# Two more belief rollout arms: sentence ends, and a random token per sentence.
#
# Entry 49 filled the label x selection grid at the GLOBAL top-20 cadence. What it left empty
# is the per-sentence cadence: `jlens_argmax_per_sentence` (loudest in the span) and its
# logitlens twin exist, but neither of the two obvious controls does. These are those --
# the same sentence grid, a different rule for which token inside each span gets the cut:
#
#   eos                  the LAST token of each sentence   -- the positional control
#   random_per_sentence  a UNIFORMLY RANDOM token of each  -- the chance control
#
# Both are matched to `jlens_argmax_per_sentence` span for span, because all three walk
# `sentence_spans(reasoning_eos_positions(...))`. That is the point: a difference between the
# arms is the position inside the sentence and nothing else.
#
# random_per_sentence is NOT the same control as `recorded_selection --selection-arm random`,
# which draws a fixed count uniformly over the WHOLE chain with no sentence awareness, so a
# long sentence can take several picks and a short one none. Here every sentence contributes
# exactly one. SEED picks the draw and is recorded in each arm's strategy config, because a
# control's draw is the one thing that cannot be reconstructed after the fact.
#
# WHY THE EOS ARM IS RE-RUN RATHER THAN READ OFF DISK. A finished sentence-end rollout already
# covers all 36,000 trajectories at $RT/trajectories_train_single_step_probs -- but it predates
# truncation_strategies.py, has no `cutoff_kind` field, and was produced by different batching.
# On the 10 trajectories where both it and a current-code eos rollout exist
# (rollout_strategies_smoke/eos) the interior sentence_end cutoffs DISAGREE ON THE ACTION 17.9%
# of the time (7/39, mean |dp| .076), while end_of_reasoning agrees to 0.0000 -- the confident
# case, exactly as RUN_STATE.md predicted. ~18% label noise is the same order as the effects
# being measured, so the old labels cannot be the label source for an arm compared against the
# others. Re-running on the 3,600 also finally settles RUN_STATE.md's open question, on 78,642
# cutoffs instead of 39.
#
# Note this writes to a DIFFERENT directory than the historical arm. reproduce_all.sh's
# run_sentence_end_rollout targets trajectories_train_single_step_probs WITH --skip-existing,
# so re-running that stage would skip all 36,000 files and refresh nothing.
#
# ~12 h on one GPU (measured sibling: jlens_argmax_per_sentence 6h04m over the same 3,600, and
# both arms here have the same one-cutoff-per-sentence cadence). Every arm resumes from disk.
#
# Usage:
#   bash wrappers/rollout_more_belief_arms.sh            # both, in order
#   bash wrappers/rollout_more_belief_arms.sh eos        # just the sentence-end arm
#   DRY_RUN=1 bash wrappers/rollout_more_belief_arms.sh  # cutoffs only, no model
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

NAMES_FILE=${NAMES_FILE:-/workspace/reasoning_theatre/rollout_strategies/mass_l15_names.txt}
OUT_ROOT=${OUT_ROOT:-/workspace/reasoning_theatre/rollout_strategies_baselines}
JLENS_ROOT=${JLENS_ROOT:-/workspace/activations/jlens_mass_l15}
SEED=${SEED:-42}
export DRY_RUN=${DRY_RUN:-}

WANT=${1:-all}

# LENS_ROOT is passed to both arms although neither selects on loudness: eos ignores it
# outright (EosStrategy is not a LoudnessStrategy), and random_per_sentence records
# dir_logmass as a COVARIATE so the analysis can ask how loud the random pick happened to be
# against the loud one on the same span.
if [ "$WANT" = all ] || [ "$WANT" = eos ]; then
    echo "### arm 1/2: sentence ends -> belief"
    NAMES_FILE="$NAMES_FILE" OUT_ROOT="$OUT_ROOT" \
    LENS=jlens LENS_ROOT="$JLENS_ROOT" \
        bash "$REPO/telos_interp/loudness_analysis/rollouts/run_inference_strategies.sh" eos
fi

if [ "$WANT" = all ] || [ "$WANT" = random ]; then
    echo "### arm 2/2: a random token per sentence -> belief (seed $SEED)"
    NAMES_FILE="$NAMES_FILE" OUT_ROOT="$OUT_ROOT" SEED="$SEED" \
    LENS=jlens LENS_ROOT="$JLENS_ROOT" \
        bash "$REPO/telos_interp/loudness_analysis/rollouts/run_inference_strategies.sh" random_per_sentence
fi

echo
echo "Arms under $OUT_ROOT:"
for d in "$OUT_ROOT"/*/; do
    [ -d "$d" ] && echo "  $(basename "$d"): $(find "$d" -maxdepth 1 -name '*.json' | wc -l) result file(s)"
done
