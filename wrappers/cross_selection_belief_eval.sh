#!/usr/bin/env bash
# The cross-selection matrix: every local-belief probe read on every other arm's tokens.
#
# ICLR log entry 58. Until this run, every belief probe had been scored on exactly two
# populations, and they are the two extremes: the tokens its OWN selection picked (the
# eval-720 slice it was trained against) and ALL 87,221 reasoning tokens of the heldout 360
# with no selection at all. Entry 49(c) found the ordering INVERTS between them -- on its own
# loud tokens the jlens arm wins and the random control is worst, on the full heldout the
# random control wins and jlens is worst. With only those two points, "this probe is better"
# cannot be separated from "this probe is specialised to the tokens it was trained on".
#
# This fills the middle. Fix the test population (a column), vary the TRAINING selection
# (the rows), and the confound becomes the measurement.
#
# WHAT RUNS: 7 eval slices x the probes whose cadence matches, 50 evaluations.
#
#   slice           eval split                          rows    probes read on it
#   jlens_p1        equal_n_jlens_split_eval            14,472  p1 jlens/logitlens/random/eos
#   logitlens_p1    equal_n_logitlens_split_eval        14,472  p1 jlens/logitlens/random/eos
#   random_p1       equal_n_random_split_eval           14,472  p1 jlens/logitlens/random/eos
#   eos_p1          equal_n_eos_split_eval              14,472  p1 jlens/logitlens/random/eos
#   jlens_p2        local_belief_p2_split_eval          14,391  p2 jlens/logitlens/random
#   logitlens_p2    entry49_logitlens_p2_split_eval     14,391  p2 jlens/logitlens/random
#   random_p2       entry49_random_belief_split_eval    14,391  p2 jlens/logitlens/random
#
# Each arm is read as both an lr and an mlp probe, so 4 p1 slices x 8 + 3 p2 slices x 6 = 50.
#
# A PROBE IS NEVER READ ACROSS CADENCES. p1 is one cutoff per sentence at that sentence's
# loudest token; p2 is the 20 globally loudest tokens of the chain. They are different token
# populations with different row counts (14,472 vs 14,391), so a p1 probe on a p2 slice would
# compare nothing. The eos arm exists only at p1, which is why the p2 columns hold six cells
# and not eight.
#
# THE TWO CADENCES ARE DIFFERENT VINTAGES, ON PURPOSE. p1 is the entry-52 equal-N rebuild;
# p2 is the entry-45/49 original and never needed rebuilding, because a fixed min(n, 20)
# budget RELABELS a row that collides with the end_of_reasoning bookend where the uncapped
# per-sentence rule DELETES it. Read down a column, never across a row.
#
# BALANCED ACCURACY IS THE NUMBER TO READ, and eval_local_belief.py prints it first. The
# local-belief label is imbalanced on these splits (UP ~.35 of rows against DOWN's ~.17), so
# plain accuracy pays a probe for leaning on the majority class -- and the arms differ in
# exactly that prior, which can move a column's ORDERING and not just its level. Balanced is
# also what the probe checkpoints publish as best_balanced_accuracy, so the 14 diagonal cells
# reproduce their numbers exactly; that agreement is the check that the probes, the splits and
# the label join are the ones they claim to be. Plain accuracy is printed too, one line down.
#
# THE DIRECTION-WORD SPLIT IS UNAVAILABLE ON FIVE OF THE SEVEN SLICES. It keys on each
# sample's `token` field, and only local_belief_p2_split_eval and entry49_logitlens_p2_split_eval
# carry one. That is a property of those manifests, not of the evaluator.
#
# COST: ~10 min per cell and it is all MooseFS latency -- ~14,400 individual .pt reads against
# ~30 s of CPU. Re-reading the same slice does not get faster, so the only lever is width;
# PAR=10 puts the whole matrix in about 50 minutes.
#
# Usage:
#   bash wrappers/cross_selection_belief_eval.sh              # all 50, 10 at a time
#   PAR=4 bash wrappers/cross_selection_belief_eval.sh        # narrower
#   DRY_RUN=1 bash wrappers/cross_selection_belief_eval.sh    # print the 50 commands, run none
#   ONLY=jlens_p2 bash wrappers/cross_selection_belief_eval.sh # one slice
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PREPARED=${PREPARED:-/workspace/prepared}
# The canonical belief-probe root. NOT probes/local_belief_equalN -- that is a compatibility
# symlink left over from the rename and is meant to go away.
PROBES=${PROBES:-/workspace/probes/local_belief_action_l15}
OUT=${OUT:-/workspace/reasoning_theatre/cross_selection_eval}
PAR=${PAR:-10}
ONLY=${ONLY:-}
DRY_RUN=${DRY_RUN:-}
EVAL=telos_interp/loudness_analysis/rollouts/eval_local_belief.py

# slice:eval-split:cadence
SLICES=(
    "jlens_p1:equal_n_jlens_split_eval:p1"
    "logitlens_p1:equal_n_logitlens_split_eval:p1"
    "random_p1:equal_n_random_split_eval:p1"
    "eos_p1:equal_n_eos_split_eval:p1"
    "jlens_p2:local_belief_p2_split_eval:p2"
    "logitlens_p2:entry49_logitlens_p2_split_eval:p2"
    "random_p2:entry49_random_belief_split_eval:p2"
)
P1_ARMS=(jlens logitlens random eos)
P2_ARMS=(jlens logitlens random)

jobs_file=$(mktemp)
trap 'rm -f "$jobs_file"' EXIT

for entry in "${SLICES[@]}"; do
    IFS=: read -r slice split cadence <<<"$entry"
    [ -n "$ONLY" ] && [ "$ONLY" != "$slice" ] && continue
    if [ "$cadence" = p1 ]; then arms=("${P1_ARMS[@]}"); else arms=("${P2_ARMS[@]}"); fi
    for arm in "${arms[@]}"; do
        for mt in lr mlp; do
            probe="$PROBES/$cadence/next_action_probe_${arm}_${mt}.pt"
            eval_dir="$PREPARED/$split"
            # Fail here rather than 10 minutes in: eval_local_belief.py opens the probe only
            # AFTER it has read the whole split, so a bad probe path costs a full cell.
            [ -f "$probe" ] || { echo "missing probe: $probe" >&2; exit 1; }
            [ -f "$eval_dir/manifest.json" ] || { echo "missing split: $eval_dir" >&2; exit 1; }
            printf '%s\t%s\t%s\n' "$slice" "$eval_dir" "$probe" >>"$jobs_file"
        done
    done
done

n=$(wc -l <"$jobs_file")
echo "$n evaluations, $PAR at a time -> $OUT/<slice>/<probe>.txt"

if [ -n "$DRY_RUN" ]; then
    while IFS=$'\t' read -r slice eval_dir probe; do
        echo "uv run --project $REPO python $EVAL $probe $eval_dir" \
             "> $OUT/$slice/$(basename "$probe" .pt).txt"
    done <"$jobs_file"
    exit 0
fi

export OUT REPO EVAL
cd "$REPO"
# -n 3 because each line is (slice, eval_dir, probe); "$0" is the slice.
tr '\t' '\n' <"$jobs_file" | xargs -P "$PAR" -n 3 bash -c '
    mkdir -p "$OUT/$0"
    cd "$REPO"
    uv run --project "$REPO" python "$EVAL" "$2" "$1" \
        > "$OUT/$0/$(basename "$2" .pt).txt" 2>&1 || echo "FAILED $0 $2"
'
echo "done: $(find "$OUT" -name '*.txt' -not -path '*.ipynb_checkpoints*' | wc -l) cell files under $OUT"
