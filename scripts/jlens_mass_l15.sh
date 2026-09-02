#!/usr/bin/env bash
# Layer-15 gather, ranked by direction PROBABILITY MASS instead of by a count.
#
# The count-era tree (jlens_reasoning_tokens) answers "how many of the top-20 lens
# predictions are direction words". This one answers "how much probability does the lens
# put on direction words", over the WHOLE vocabulary rather than a top-20 window, and it
# asks that question at a single layer. Three deliberate differences from
# jlens_reasoning_tokens_filtered.sh:
#
#   1. --direction-score logprob_mass_full     -- rank on the direction-mass table.
#   2. --select-candidate-layers 15            -- score AND save layer 15 only, so one probe
#                                                 weight vector reads one representation
#                                                 space. The CSV and the mass table still
#                                                 cover $LAYERS, so the cross-layer profile
#                                                 (scripts/jlens_layer_profile.py) is still
#                                                 computable from this run.
#   3. a fresh --activations-dir               -- the existing tree is pruned to the count
#                                                 arms' picks, so a mass arm's tokens are
#                                                 not recoverable there by re-filtering.
#
# The control arm is drawn here for the same reason it always is: once this tree holds only
# the mass arm's tokens, a uniform draw over the reasoning chain can never be made again.
# THE LOGITLENS TWIN (ICLR entry 49). The same run with the other lens, pinned by name to
# the trajectories this one drew, so the two trees differ in the lens and nothing else:
#
#   LENS=logitlens SELECT_METHODS=logitlens \
#   ACTIVATIONS_DIR=/workspace/activations/logitlens_mass_l15 \
#   NAMES_FILE=/workspace/reasoning_theatre/rollout_strategies/mass_l15_names.txt \
#     bash scripts/jlens_mass_l15.sh
#
# No random arm there: the control's draw already exists in this tree's records and must be
# read, never re-drawn (scripts/inference_oss/truncation_strategies.py::recorded_selection).
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)   # so uv finds pyproject.toml

TRAJECTORIES=${TRAJECTORIES:-/workspace/trajectories/reveng/trajectories_train_single_step/}
JLENS_DIR=${JLENS_DIR:-/workspace/jlens/gridenv}
ACTIVATIONS_DIR=${ACTIVATIONS_DIR:-/workspace/activations/jlens_mass_l15}
SIGNAL_JSON=${SIGNAL_JSON:-/workspace/jlens/direction_tokens_full.json}
LENS=${LENS:-jlens}
LAYERS=${LAYERS:-"7:23"}   # what the CSV and the mass table cover -- NOT what is selected
# Same sampling as the count-era tree (PER_COMBO=100, SEED=0), so the two draw the same
# 3600 trajectories and an arm-vs-arm comparison is not confounded by the trajectory set.
# NAMES_FILE pins the trajectory set by NAME, and when it is given it is the SOLE
# authority: PER_COMBO defaults off, because a balanced re-draw over an already-pinned
# list can only narrow it, and entry 36's correction is that --per-combo/--seed does not
# reproduce a previous draw anyway (two runs with identical flags overlapped by 348/3600).
# This is how a second tree is made to cover the same trajectories as an existing one.
NAMES_FILE=${NAMES_FILE:-}
if [ -n "$NAMES_FILE" ]; then
    PER_COMBO=${PER_COMBO-}
else
    PER_COMBO=${PER_COMBO-100}   # set PER_COMBO="" to disable and use MAX_TRAJ instead
fi
MAX_TRAJ=${MAX_TRAJ:-}     # global cap instead of PER_COMBO (mutually exclusive)
SEED=${SEED:-0}
SIZES=${SIZES:-}
COMPLEXITIES=${COMPLEXITIES:-}
OVERWRITE=${OVERWRITE:-}

# Selection.
DIRECTION_SCORE=${DIRECTION_SCORE:-logprob_mass_full}
CANDIDATE_LAYERS=${CANDIDATE_LAYERS:-15}
SELECT_METHODS=${SELECT_METHODS:-jlens,random}
NUM_TOKENS=${NUM_TOKENS:-20}
NUM_LAYERS=${NUM_LAYERS:-1}
ALWAYS_LAYERS=${ALWAYS_LAYERS:-15}
RANDOM_TOKENS=${RANDOM_TOKENS:-20}
SELECT_SEED=${SELECT_SEED:-42}
DIRECTION_CLASSES=${DIRECTION_CLASSES:-all}
# Vocabulary the mass table is baked against; recorded in its .meta.json sidecar.
DIRECTION_MASS_JSON=${DIRECTION_MASS_JSON:-$SIGNAL_JSON}

# Throughput knobs; empty = script defaults.
BATCH_SIZE=${BATCH_SIZE:-}
IO_WORKERS=${IO_WORKERS:-}
FWD_BATCH_SIZE=${FWD_BATCH_SIZE:-}
FWD_BATCH_TOKENS=${FWD_BATCH_TOKENS:-}
PROFILE=${PROFILE:-}
EXTRA=${EXTRA:-}           # e.g. --dry-run, --no-top-logprobs
# This script loads gpt-oss-20b, so it needs the `gpu` extra (accelerate, and the pinned
# kernels==0.12.0), which is not in the default dependencies. A `uv run` WITHOUT it also
# syncs the extra back out of a venv that had it, so leaving this off does not merely risk
# this run -- it can break the next one that assumed accelerate was there.
UV_EXTRAS=${UV_EXTRAS:---extra gpu}

[ -f "$SIGNAL_JSON" ] || { echo "!! signal JSON not found: $SIGNAL_JSON" >&2; exit 1; }

uv run --project "$REPO" $UV_EXTRAS python "$REPO/scripts/jlens_reasoning_tokens.py" \
    --trajectory-paths "$TRAJECTORIES" \
    ${NAMES_FILE:+--names-file "$NAMES_FILE"} \
    --jlens_dir "$JLENS_DIR" \
    --activations-dir "$ACTIVATIONS_DIR" \
    --lens "$LENS" \
    --layers "$LAYERS" \
    --steps "all" \
    --signal-json "$SIGNAL_JSON" \
    --direction-mass-json "$DIRECTION_MASS_JSON" \
    --direction-score "$DIRECTION_SCORE" \
    --select-methods "$SELECT_METHODS" \
    --select-candidate-layers "$CANDIDATE_LAYERS" \
    --select-num-tokens "$NUM_TOKENS" \
    --select-num-layers "$NUM_LAYERS" \
    --select-always-layers "$ALWAYS_LAYERS" \
    --select-random-tokens "$RANDOM_TOKENS" \
    --select-seed "$SELECT_SEED" \
    --direction-classes "$DIRECTION_CLASSES" \
    ${BATCH_SIZE:+--batch-size "$BATCH_SIZE"} \
    ${IO_WORKERS:+--io-workers "$IO_WORKERS"} \
    ${FWD_BATCH_SIZE:+--forward-batch-size "$FWD_BATCH_SIZE"} \
    ${FWD_BATCH_TOKENS:+--forward-batch-tokens "$FWD_BATCH_TOKENS"} \
    ${PROFILE:+--profile} \
    ${SIZES:+--sizes "$SIZES"} \
    ${COMPLEXITIES:+--complexities "$COMPLEXITIES"} \
    ${PER_COMBO:+--per-combo "$PER_COMBO"} \
    ${MAX_TRAJ:+--max-trajectories "$MAX_TRAJ"} \
    ${SEED:+--seed "$SEED"} \
    ${OVERWRITE:+--overwrite} \
    $EXTRA

echo "Done -> $ACTIVATIONS_DIR"
echo "Prepare probe data with --token-selection recorded_jlens / recorded_random --layers 15."
