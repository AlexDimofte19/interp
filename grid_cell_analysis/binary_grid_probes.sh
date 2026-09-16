#!/usr/bin/env bash
# Four binary grid-cell probes -- empty / wall / agent / goal -- end to end.
#
# Trains on the mass-era 2,880, scores on the pinned 720, scores again on the disjoint
# heldout360 tree. Every arm sees the same tokens and the same cells; the only thing that
# differs between them is which symbol counts as positive.
#
#   stage 1  prepare the 3,600 at NATIVE grid size (no padding class)
#   stage 2  split 2,880 / 720 on the pinned eval names
#   stage 3  prepare heldout360 -- every reasoning token, not a selection
#   stage 4  train 4 classes x {lr, mlp}
#   stage 5  score each probe on the 720 and on the 360
#   stage 6  fold the 16 result JSONs into one CSV
#
# Every stage tests for its own output first, so a re-run is a no-op check. DRY_RUN=1
# prints the commands without running them; ONLY="1 4" runs just those stages.
#
# Two things this script deliberately does NOT do, both for reasons recorded in
# grid_cell_analysis/README.md and ICLR log entry 55:
#   * it never passes --pad-to-size. Padding to 15 makes '+' ~90% of a size-5 grid's
#     sampled cells, and every binary arm would then be measuring padding.
#   * it never passes --balance-classes-per-trajectory. Combined with a cell cap that
#     yields 4 cells per step instead of MAX_CELLS, because every grid holds exactly one
#     'A' and one 'G' so the per-class minimum is 1. Class imbalance is handled at train
#     time with --class-weight balanced instead.

set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

# --- inputs -------------------------------------------------------------------------------
ACT=${ACT:-/workspace/activations/jlens_mass_l15}
TRAJ=${TRAJ:-/workspace/trajectories/reveng/trajectories_train_single_step}
HELD_ACT=${HELD_ACT:-/workspace/activations/heldout360_l15}
HELD_TRAJ=${HELD_TRAJ:-/workspace/trajectories/heldout360}
EVAL_NAMES=${EVAL_NAMES:-/workspace/splits/mass_eval_720.txt}

# --- outputs ------------------------------------------------------------------------------
PREPARED=${PREPARED:-/workspace/prepared/grid_binary_l15_random}
HELD_PREPARED=${HELD_PREPARED:-/workspace/prepared/grid_binary_l15_heldout360}
PROBES=${PROBES:-/workspace/probes/grid_binary_l15}
LOGS=${LOGS:-$PROBES/logs}
SUMMARY=${SUMMARY:-$PROBES/binary_grid_summary.csv}

# --- knobs --------------------------------------------------------------------------------
CLASSES=${CLASSES:-"empty wall agent goal"}   # aliases for _  #  A  G
MODEL_TYPES=${MODEL_TYPES:-"lr mlp"}
TOKEN_SELECTION=${TOKEN_SELECTION:-recorded_random}
LAYER=${LAYER:-15}
STEPS=${STEPS:-all}
# Cells kept per (trajectory, step). MUST be <= 25: the smallest grid is 5x5 = 25 cells at
# native size, and prepare requires a uniform cell count -- ask for more and it drops every
# size folder that cannot supply it.
MAX_CELLS=${MAX_CELLS:-25}
SEED=${SEED:-42}
SPLIT_SEED=${SPLIT_SEED:-42}
DEVICE=${DEVICE:-cuda}
THRESHOLD=${THRESHOLD:-0.5}

NUM_EPOCHS=${NUM_EPOCHS:-50}
HIDDEN_DIMS=${HIDDEN_DIMS:-1024}
LEARNING_RATE=${LEARNING_RATE:-3e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.001}
BATCH_SIZE=${BATCH_SIZE:-2048}
DROPOUT=${DROPOUT:-0.0}
CLASS_WEIGHT=${CLASS_WEIGHT:-balanced}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8192}

DRY_RUN=${DRY_RUN:-0}
FORCE=${FORCE:-0}
ONLY=${ONLY:-}
SKIP=${SKIP:-}

UV=${UV:-"uv run --project $REPO"}

mkdir -p "$PROBES" "$LOGS"

# --- plumbing -----------------------------------------------------------------------------
run() {
    if [ "$DRY_RUN" = "1" ]; then
        printf '  +'; printf ' %q' "$@"; printf '\n'
        return 0
    fi
    "$@"
}

stage_enabled() {
    local n=$1
    if [ -n "$ONLY" ]; then
        case " $ONLY " in *" $n "*) ;; *) return 1 ;; esac
    fi
    if [ -n "$SKIP" ]; then
        case " $SKIP " in *" $n "*) return 1 ;; esac
    fi
    return 0
}

banner() { echo; echo "=== stage $1: $2 ==="; }

# `done` is a shell keyword, hence the name.
have() { [ "$FORCE" != "1" ] && [ -e "$1" ]; }

require() {
    local missing=0
    for path in "$@"; do
        [ -e "$path" ] || { echo "MISSING: $path" >&2; missing=1; }
    done
    [ "$missing" = "0" ] || { echo "Refusing to start with missing inputs." >&2; exit 1; }
}

echo "repo            $REPO"
echo "source tree     $ACT"
echo "trajectories    $TRAJ"
echo "held-out tree   $HELD_ACT"
echo "eval names      $EVAL_NAMES"
echo "prepared        $PREPARED{,_split_train,_split_eval}"
echo "held prepared   $HELD_PREPARED"
echo "probes          $PROBES"
echo "classes         $CLASSES x $MODEL_TYPES"
echo "layer           $LAYER    max cells $MAX_CELLS    token selection $TOKEN_SELECTION"
[ "$DRY_RUN" = "1" ] && echo "(DRY RUN -- nothing will be written)"

[ "$DRY_RUN" = "1" ] || require "$ACT" "$TRAJ" "$HELD_ACT" "$HELD_TRAJ" "$EVAL_NAMES"

if [ "$MAX_CELLS" -gt 25 ]; then
    echo "MAX_CELLS=$MAX_CELLS > 25: a 5x5 grid has only 25 cells at native size, and" >&2
    echo "prepare drops every size folder whose cell count disagrees. Lower it." >&2
    exit 1
fi

# --- stage 1: prepare the 3,600 -----------------------------------------------------------
if stage_enabled 1; then
    banner 1 "prepare $PREPARED (native grid size, $TOKEN_SELECTION tokens, layer $LAYER)"
    if have "$PREPARED/manifest.json"; then
        echo "  exists, skipping"
    else
        run $UV interp-cli prepare_activations_for_probing \
            --activations-dir "$ACT" \
            --trajectories-dir "$TRAJ" \
            --probe-type grid_tile \
            --layers "$LAYER" \
            --steps "$STEPS" \
            --output-indices all \
            --token-selection "$TOKEN_SELECTION" \
            --seed "$SEED" \
            --max-positions-per-trajectory "$MAX_CELLS" \
            --output-path "$PREPARED" \
            --verbose 2>&1 | tee "$LOGS/prepare_train.txt"
    fi
fi

# --- stage 2: split 2880 / 720 ------------------------------------------------------------
if stage_enabled 2; then
    banner 2 "split on $EVAL_NAMES"
    if have "${PREPARED}_split_train/manifest.json" && have "${PREPARED}_split_eval/manifest.json"; then
        echo "  exists, skipping"
    else
        run $UV python "$REPO/scripts/split_next_action_manifest.py" \
            "$PREPARED" \
            --eval-names "$EVAL_NAMES" \
            --single-layer "$LAYER" \
            --seed "$SPLIT_SEED" \
            --train-out "${PREPARED}_split_train" \
            --eval-out "${PREPARED}_split_eval" 2>&1 | tee "$LOGS/split.txt"
    fi
fi

# --- stage 3: prepare the held-out 360, every token ---------------------------------------
if stage_enabled 3; then
    banner 3 "prepare $HELD_PREPARED (every reasoning token of heldout360)"
    if have "$HELD_PREPARED/manifest.json"; then
        echo "  exists, skipping"
    else
        # --token-selection all: heldout360_l15 carries no *_jlens_selection.json, so there
        # is no recorded arm to replay. It holds layer-15 .pt for EVERY reasoning token,
        # which is the wider grid we want for the generalisation number anyway.
        run $UV interp-cli prepare_activations_for_probing \
            --activations-dir "$HELD_ACT" \
            --trajectories-dir "$HELD_TRAJ" \
            --probe-type grid_tile \
            --layers "$LAYER" \
            --steps "$STEPS" \
            --output-indices all \
            --token-selection all \
            --token-major \
            --seed "$SEED" \
            --max-positions-per-trajectory "$MAX_CELLS" \
            --output-path "$HELD_PREPARED" \
            --verbose 2>&1 | tee "$LOGS/prepare_heldout.txt"
    fi
fi

# --- stage 4: train ------------------------------------------------------------------------
probe_path() { echo "$PROBES/grid_binary_probe_${1}_${2}.pt"; }

if stage_enabled 4; then
    banner 4 "train $(echo $CLASSES | wc -w) classes x $(echo $MODEL_TYPES | wc -w) heads"
    for cls in $CLASSES; do
        for mt in $MODEL_TYPES; do
            out=$(probe_path "$cls" "$mt")
            if have "$out"; then
                echo "  $cls/$mt exists, skipping"
                continue
            fi
            echo "  --- $cls / $mt ---"
            run $UV interp-cli train_binary_cognitive_map_probe \
                --train-data-path "${PREPARED}_split_train" \
                --eval-data-path "${PREPARED}_split_eval" \
                --positive-class "$cls" \
                --model-type "$mt" \
                --output-path "$out" \
                --hidden-dims "$HIDDEN_DIMS" \
                --learning-rate "$LEARNING_RATE" \
                --weight-decay "$WEIGHT_DECAY" \
                --dropout "$DROPOUT" \
                --num-epochs "$NUM_EPOCHS" \
                --batch-size "$BATCH_SIZE" \
                --class-weight "$CLASS_WEIGHT" \
                --normalize \
                --cache-activations \
                --seed "$SEED" \
                --device "$DEVICE" \
                --verbose 2>&1 | tee "$LOGS/${cls}_${mt}_train.txt"
        done
    done
fi

# --- stage 5: evaluate ---------------------------------------------------------------------
result_path() { echo "$PROBES/eval_${1}_${2}_${3}.json"; }

if stage_enabled 5; then
    banner 5 "score each probe on the 720 and on heldout360"
    for cls in $CLASSES; do
        for mt in $MODEL_TYPES; do
            probe=$(probe_path "$cls" "$mt")
            if [ "$DRY_RUN" != "1" ] && [ ! -e "$probe" ]; then
                echo "  $cls/$mt: no probe at $probe, skipping"
                continue
            fi
            for split in eval720 heldout360; do
                case "$split" in
                    eval720)    data="${PREPARED}_split_eval" ;;
                    heldout360) data="$HELD_PREPARED" ;;
                esac
                out=$(result_path "$cls" "$mt" "$split")
                if have "$out"; then
                    echo "  $cls/$mt on $split exists, skipping"
                    continue
                fi
                echo "  --- $cls / $mt on $split ---"
                run $UV interp-cli eval_binary_cognitive_map_probe \
                    --probe-path "$probe" \
                    --data-path "$data" \
                    --output-path "$out" \
                    --threshold "$THRESHOLD" \
                    --batch-size "$EVAL_BATCH_SIZE" \
                    --cache-activations \
                    --device "$DEVICE" \
                    --verbose 2>&1 | tee "$LOGS/${cls}_${mt}_${split}_eval.txt"
            done
        done
    done
fi

# --- stage 6: one table --------------------------------------------------------------------
if stage_enabled 6; then
    banner 6 "fold the result JSONs into $SUMMARY"
    run $UV python "$REPO/grid_cell_analysis/collect_binary_grid_results.py" \
        --results-dir "$PROBES" \
        --out "$SUMMARY"
fi

echo
echo "done."
