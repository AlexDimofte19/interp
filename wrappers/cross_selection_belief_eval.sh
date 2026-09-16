#!/usr/bin/env bash
# The cross-selection matrix: every local-belief probe read on every other arm's tokens.
# ICLR log entry 58.
#
# Every belief probe had only ever been scored on two populations, and they are the extremes:
# the tokens its OWN selection picked, and all 87,221 heldout tokens with no selection at all.
# Entry 49(c) found the ordering INVERTS between them, which leaves "this probe is better"
# confounded with "this probe is specialised to its own tokens". This fills the middle: fix the
# test slice, vary the training selection, and READ DOWN A COLUMN.
#
# READ THE BALANCED ACCURACY. eval_local_belief.py prints it first and plain accuracy below it.
# The local-belief label is imbalanced (UP ~.35 of rows against DOWN's ~.17), so plain accuracy
# pays a probe for leaning on the majority class -- and the arms differ in exactly that prior,
# which can move a column's ORDERING and not just its level. Balanced is also the statistic the
# probe checkpoints publish, so the 14 diagonal cells reproduce their numbers.
#
# A p1 probe is never read on a p2 slice. p1 is one cutoff per sentence at its loudest token,
# p2 is the 20 globally loudest tokens of the chain: different populations, 14,472 rows against
# 14,391, so the comparison would mean nothing. The eos arm exists only at p1, which is why the
# p2 slices get 6 evaluations and the p1 slices 8. 4x8 + 3x6 = 50.
#
# ~10 minutes per evaluation, nearly all of it MooseFS latency: ~14,400 individual .pt reads
# against ~30 s of CPU, and re-reading a slice does not get faster. Each slice's probes run
# concurrently and the script waits between slices, so the whole matrix takes about 70 minutes.
#
# Usage: bash wrappers/cross_selection_belief_eval.sh
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PROBES=/workspace/probes/local_belief_action_l15   # NOT local_belief_equalN, that is a symlink
PREPARED=/workspace/prepared
OUT=/workspace/reasoning_theatre/cross_selection_eval

cd "$REPO"
EVAL="uv run --project $REPO python telos_interp/loudness_analysis/rollouts/eval_local_belief.py"

# ---- 1. j-lens p1 -- the loudest token of each sentence, jlens ruler -----------------------
mkdir -p "$OUT/jlens_p1"
for arm in jlens logitlens random eos; do
    for mt in lr mlp; do
        $EVAL "$PROBES/p1/next_action_probe_${arm}_${mt}.pt" \
              "$PREPARED/equal_n_jlens_split_eval" \
              > "$OUT/jlens_p1/next_action_probe_${arm}_${mt}.txt" &
    done
done
wait

# ---- 2. logit-lens p1 -- the same rule, scored by the logit lens ---------------------------
mkdir -p "$OUT/logitlens_p1"
for arm in jlens logitlens random eos; do
    for mt in lr mlp; do
        $EVAL "$PROBES/p1/next_action_probe_${arm}_${mt}.pt" \
              "$PREPARED/equal_n_logitlens_split_eval" \
              > "$OUT/logitlens_p1/next_action_probe_${arm}_${mt}.txt" &
    done
done
wait

# ---- 3. random p1 -- one uniformly drawn token per sentence, the control -------------------
mkdir -p "$OUT/random_p1"
for arm in jlens logitlens random eos; do
    for mt in lr mlp; do
        $EVAL "$PROBES/p1/next_action_probe_${arm}_${mt}.pt" \
              "$PREPARED/equal_n_random_split_eval" \
              > "$OUT/random_p1/next_action_probe_${arm}_${mt}.txt" &
    done
done
wait

# ---- 4. eos p1 -- the last token of each sentence, no loudness in the rule -----------------
mkdir -p "$OUT/eos_p1"
for arm in jlens logitlens random eos; do
    for mt in lr mlp; do
        $EVAL "$PROBES/p1/next_action_probe_${arm}_${mt}.pt" \
              "$PREPARED/equal_n_eos_split_eval" \
              > "$OUT/eos_p1/next_action_probe_${arm}_${mt}.txt" &
    done
done
wait

# ---- 5. j-lens p2 -- the 20 globally loudest tokens, jlens ruler ---------------------------
mkdir -p "$OUT/jlens_p2"
for arm in jlens logitlens random; do
    for mt in lr mlp; do
        $EVAL "$PROBES/p2/next_action_probe_${arm}_${mt}.pt" \
              "$PREPARED/local_belief_p2_split_eval" \
              > "$OUT/jlens_p2/next_action_probe_${arm}_${mt}.txt" &
    done
done
wait

# ---- 6. logit-lens p2 -- the same rule, logit-lens ruler -----------------------------------
mkdir -p "$OUT/logitlens_p2"
for arm in jlens logitlens random; do
    for mt in lr mlp; do
        $EVAL "$PROBES/p2/next_action_probe_${arm}_${mt}.pt" \
              "$PREPARED/entry49_logitlens_p2_split_eval" \
              > "$OUT/logitlens_p2/next_action_probe_${arm}_${mt}.txt" &
    done
done
wait

# ---- 7. random p2 -- 20 tokens per trajectory drawn uniformly, the control -----------------
mkdir -p "$OUT/random_p2"
for arm in jlens logitlens random; do
    for mt in lr mlp; do
        $EVAL "$PROBES/p2/next_action_probe_${arm}_${mt}.pt" \
              "$PREPARED/entry49_random_belief_split_eval" \
              > "$OUT/random_p2/next_action_probe_${arm}_${mt}.txt" &
    done
done
wait

echo "done -> $OUT/<slice>/<probe>.txt"
