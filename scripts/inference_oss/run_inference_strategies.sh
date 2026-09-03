#!/usr/bin/env bash
# Reasoning-theatre rollouts, one arm per TRUNCATION STRATEGY.
#
# Same model, same trajectories, same prompts-per-cutoff construction; the only thing
# that differs between the three arms is WHERE the reasoning is cut before the model is
# asked for its action (scripts/inference_oss/truncation_strategies.py):
#
#   eos                        every reasoning sentence end. The grid every rollout on
#                              disk was measured on, re-run here so the baseline is
#                              written by this code version over these trajectories.
#   jlens_argmax_per_sentence  one cutoff per sentence, at its LOUDEST token -- loudness
#                              being the layer-15 full-vocabulary direction mass of ICLR
#                              log entry 42, sum(exp(logprob(t))) over
#                              direction_tokens_full.json. Entry 42(a) says a sentence
#                              end is systematically its quietest point; entry 41(c)
#                              says the commitment is already there a median ~7 tokens
#                              earlier. This arm asks the model at the loud point instead.
#   jlens_top_k_global         the TOP_K loudest tokens of the whole chain, wherever they
#                              fall, so the sampling grid follows the lens rather than the
#                              punctuation.
#
# THE TRAJECTORY SET IS THE SAME FOR ALL THREE. Loudness comes from the direction-mass
# tables, which exist only for the 3600 trajectories of $LENS_ROOT, so NAMES_FILE pins
# every arm (the eos one included) to those names. It is built from $LENS_ROOT on first
# run and reused afterwards.
#
# Usage:
#   bash scripts/inference_oss/run_inference_strategies.sh              # all three, in order
#   bash scripts/inference_oss/run_inference_strategies.sh jlens_top_k_global
#   DRY_RUN=1 bash scripts/inference_oss/run_inference_strategies.sh    # cutoffs only, no model
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)

TRAJECTORIES=${TRAJECTORIES:-/workspace/trajectories/reveng/trajectories_train_single_step/}
LENS_ROOT=${LENS_ROOT:-/workspace/activations/jlens_mass_l15}
OUT_ROOT=${OUT_ROOT:-/workspace/reasoning_theatre/rollout_strategies}
NAMES_FILE=${NAMES_FILE:-$OUT_ROOT/mass_l15_names.txt}

LENS=${LENS:-jlens}
LOUDNESS_LAYER=${LOUDNESS_LAYER:-15}
TOP_K=${TOP_K:-20}

# Throughput. gpt-oss has no SDPA kernel and flash-attn is not installed here, so without
# an explicit backend transformers falls back to eager, whose attention memory grows as
# rows*seq_len^2 -- hence the padded-area cap rather than a bigger row count.
BATCH_SIZE=${BATCH_SIZE:-16}
MAX_BATCH_TOKENS=${MAX_BATCH_TOKENS:-49152}
# The cap that actually bounds eager attention: rows * padded_len^2. The padded-AREA cap above
# is linear in the sequence length and does not -- a first run OOMed at 16 rows x 1587 tokens
# (area 25,392, inside its 49,152 budget) asking for a 4.80 GiB [rows, heads, L, L] tensor.
# 16e6 keeps one such tensor near 1.9 GiB and leaves 16 rows untouched below ~1000 tokens.
MAX_ATTN_ELEMS=${MAX_ATTN_ELEMS:-16000000}
MAX_WINDOW_PROMPTS=${MAX_WINDOW_PROMPTS:-512}
ATTN=${ATTN:-}                       # e.g. flex_attention
SKIP_EXISTING=${SKIP_EXISTING:-1}    # set to "" to recompute finished trajectories
DRY_RUN=${DRY_RUN:-}
EXTRA=${EXTRA:-}

FAILED=""
STRATEGIES=("${@:-}")
if [ -z "${1:-}" ]; then
    STRATEGIES=(eos jlens_argmax_per_sentence jlens_top_k_global)
fi

# The failing run held 4.89 GiB reserved-but-unallocated. Batches here vary in shape by design
# (length-sorted rows, and the OOM retry halves them), which is exactly what fragments a
# fixed-block pool.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mkdir -p "$OUT_ROOT"
if [ ! -s "$NAMES_FILE" ]; then
    echo "Building $NAMES_FILE from $LENS_ROOT ..."
    find "$LENS_ROOT" -mindepth 2 -maxdepth 2 -type d -printf '%f\n' | sort > "$NAMES_FILE"
fi
echo "$(wc -l < "$NAMES_FILE") trajectory name(s) in $NAMES_FILE"

for STRATEGY in "${STRATEGIES[@]}"; do
    OUTPUT_DIR="$OUT_ROOT/$STRATEGY"
    LOG_DIR="$OUTPUT_DIR/logs"
    mkdir -p "$LOG_DIR"
    LOG_FILE="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).txt"

    echo "=== $STRATEGY -> $OUTPUT_DIR (log: $LOG_FILE)"
    {
        echo "=== START: $(date -u +'%Y-%m-%dT%H:%M:%SZ') (UTC) / $(date +'%H:%M %Z')"
        echo "=== strategy=$STRATEGY top_k=$TOP_K layer=$LOUDNESS_LAYER lens=$LENS"
    } > "$LOG_FILE"

    set +e
    uv run --project "$REPO" python "$HERE/run_inference.py" \
        --strategy "$STRATEGY" \
        --trajectory-paths "$TRAJECTORIES" \
        --names-file "$NAMES_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --lens-root "$LENS_ROOT" \
        --lens "$LENS" \
        --loudness-layer "$LOUDNESS_LAYER" \
        --top-k "$TOP_K" \
        --batch-size "$BATCH_SIZE" \
        --max-batch-tokens "$MAX_BATCH_TOKENS" \
        --max-attn-elems "$MAX_ATTN_ELEMS" \
        --max-window-prompts "$MAX_WINDOW_PROMPTS" \
        ${ATTN:+--attn-implementation "$ATTN"} \
        ${SKIP_EXISTING:+--skip-existing} \
        ${DRY_RUN:+--dry-run} \
        $EXTRA \
        >> "$LOG_FILE" 2>&1
    RC=$?
    set -e

    # Record the ARM's exit code, not the preceding date's, and keep going: the arms are
    # independent, and every one of them resumes from what is already on disk.
    echo "=== END: $(date -u +'%Y-%m-%dT%H:%M:%SZ') (UTC) (exit code: $RC)" >> "$LOG_FILE"
    N_DONE=$(find "$OUTPUT_DIR" -maxdepth 1 -name '*.json' | wc -l)
    if [ "$RC" -ne 0 ]; then
        echo "    *** $STRATEGY FAILED (exit $RC) after $N_DONE result file(s); see $LOG_FILE"
        FAILED="$FAILED $STRATEGY"
    else
        echo "    done -> $OUTPUT_DIR ($N_DONE result file(s))"
    fi
done

if [ -n "${FAILED// /}" ]; then
    echo "FAILED arm(s):$FAILED"
fi

echo
echo "Analyse with:  python $HERE/analysis.py --input-folder $OUT_ROOT/<strategy> \\"
echo "                   --trajectory-folder $TRAJECTORIES --output-dir $OUT_ROOT/<strategy>_plots"
