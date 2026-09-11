#!/usr/bin/env bash
# The loudness analysis and its figures, for ONE ruler, over any signal.
#
#   bash wrappers/loudness_report.sh                              # jlens x direction
#   LENS=logitlens bash wrappers/loudness_report.sh               # the other ruler
#   SIGNAL=grid PROBE_TYPE=grid_tile bash wrappers/loudness_report.sh
#   EXCLUDE_SIGNAL_WORDS=1 EXCLUDE_RADIUS=2 bash wrappers/loudness_report.sh
#
# This is the recorded invocation for the two analysers and the figure CLI that replaced
# analyze_probe_loudness / analyze_sentence_loudness / analyze_direction_word_isolation /
# analyze_grid_loudness_correlation and the three plot_*_loudness scripts.
#
# ONE RULER PER RUN, AND THE OUTPUT FOLDER CARRIES ITS NAME. At layer 15 the two lenses'
# top-20 sets overlap only about half, so a folder holding both is a folder of figures that
# cannot be compared with each other. provenance.RunConfig.guard enforces it -- a second run
# with a different --lens into the same --out fails rather than overwriting half the figures.
# Run it twice, once per LENS, and put the two side by side.
#
# EXCLUDE_SIGNAL_WORDS is the verbalisation control, and it is the whole reason the confound
# script went away: the tokens a lens calls loud are disproportionately the words the model
# has ALREADY typed. Setting it reproduces every table and figure from the rows that remain.
# EXCLUDE_RADIUS widens the exclusion to a +-N window, which is what separates "the residual
# is signal-loaded here" from "a signal word is about to be written" -- the lens predicts the
# NEXT tokens, so the token just before " up" is loud without being a signal word itself.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
M="$REPO/telos_interp/loudness_analysis"

RT=${RT:-/workspace/reasoning_theatre}
LENS=${LENS:-jlens}
SIGNAL=${SIGNAL:-direction}
LAYER=${LAYER:-15}
PROBE_TYPE=${PROBE_TYPE:-next_action}
SIGNAL_JSON=${SIGNAL_JSON:-/workspace/jlens/${SIGNAL}_tokens_full.json}

# The joined probe table, and the whole-chain table the distribution analysis reads.
PROBE_TABLE=${PROBE_TABLE:-$RT/probe_loudness_heldout360_16probes/per_token_${LENS}_loudness.csv}
DIST_TABLE=${DIST_TABLE:-$RT/loudness/per_token.csv}
OUT=${OUT:-$RT/loudness_report/${LENS}_${SIGNAL}}

EXCLUDE_SIGNAL_WORDS=${EXCLUDE_SIGNAL_WORDS:-}
EXCLUDE_RADIUS=${EXCLUDE_RADIUS:-0}
DRY_RUN=${DRY_RUN:-}

UV="uv run --project $REPO"
x() { echo "+ $*"; [ -n "$DRY_RUN" ] || "$@"; }

excl=()
if [ -n "$EXCLUDE_SIGNAL_WORDS" ]; then
    excl=(--exclude-signal-words --exclude-radius "$EXCLUDE_RADIUS" --signal-words "$SIGNAL_JSON")
    OUT="${OUT}_no_signal_words${EXCLUDE_RADIUS}"
fi

echo "=== ruler: $LENS x $SIGNAL @ L$LAYER  ->  $OUT ==="
mkdir -p "$OUT"

# ---- probe accuracy by loudness ------------------------------------------------------------
if [ -s "$PROBE_TABLE" ]; then
    x $UV python "$M/analysis/probe_accuracy_by_loudness.py" \
        --per-token "$PROBE_TABLE" --out "$OUT/probe_accuracy" \
        --probe-type "$PROBE_TYPE" --lens "$LENS" --signal-name "$SIGNAL" --layer "$LAYER" \
        "${excl[@]}"
    x $UV python "$M/plotting/figures.py" \
        --probe-table "$PROBE_TABLE" --out "$OUT/probe_accuracy/plots" \
        --lens "$LENS" --signal-name "$SIGNAL" --layer "$LAYER"
else
    echo "    --    no probe table at $PROBE_TABLE, skipping the probe-accuracy half"
fi

# ---- where loudness lives ------------------------------------------------------------------
if [ -s "$DIST_TABLE" ]; then
    x $UV python "$M/analysis/loudness_distribution.py" \
        --per-token "$DIST_TABLE" --out "$OUT/distribution" \
        --lens "$LENS" --signal-name "$SIGNAL" --layer "$LAYER" \
        "${excl[@]}"
    x $UV python "$M/plotting/figures.py" \
        --distribution-table "$DIST_TABLE" --out "$OUT/distribution/plots" \
        --lens "$LENS" --signal-name "$SIGNAL" --layer "$LAYER"
else
    echo "    --    no distribution table at $DIST_TABLE, skipping that half"
fi

echo "=== DONE -> $OUT  (run_config.json in each folder records the ruler and the aggregation) ==="
