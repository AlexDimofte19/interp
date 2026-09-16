#!/usr/bin/env bash
# The boundary grid-probe run: four probes, two evaluation suites, one comparison.
#
# The recorded invocation of scripts/boundary_cognitive_maps.py and, for the training
# stage, of `interp-cli train_cognitive_map_probe`. Everything here is a PARAMETER VALUE
# pinned onto those two -- changing a default in this file rewrites what the published
# numbers mean, so add a variable rather than editing one.
#
# What it builds, in order. Every stage tests for its own output first, so a re-run on the
# host that produced the work is a no-op check and an interrupted run resumes:
#
#   extract    layer-15 residuals at the six boundary positions, all three partitions
#   prepare    six token-major grid_tile manifests (3 partitions x 2 boundaries)
#   train      four probes: {lr, mlp} x {pre_reasoning, post_reasoning}, on the 2,880
#   loudness   J-lens grid log-mass at the same six positions
#   evaluate   one counts-mode table per (dataset, boundary), carrying both architectures
#   compare    post minus pre, paired per architecture, with trajectory-clustered intervals
#   plots      the SHARED grid figures (plotting/figures.py) over each boundary table
#   manifest   identities, hashes, parameters and coverage
#
# The two evaluation datasets stay in separate trees the whole way down --
# eval_720_data/ and heldout_360_data/ -- so no figure and no table ever mixes the set the
# checkpoints were selected on with the set reserved for the final numbers.
#
#   ./wrappers/boundary_cognitive_maps.sh                     # everything, resumable
#   DRY_RUN=1 ./wrappers/boundary_cognitive_maps.sh           # print the commands
#   STAGES="evaluate compare" ./wrappers/boundary_cognitive_maps.sh
#   LIMIT=4 NUM_EPOCHS=2 ./wrappers/boundary_cognitive_maps.sh              # smoke run
#
# THE THREE MEMBERSHIP LISTS ARE USED EXACTLY AS THEY ARE. `mass_eval_720.txt` is
# byte-for-byte the eval set every probe already on disk was scored against and the 2,880
# is its complement; the 360 is reserved for the final numbers and is never used to choose
# anything. Nothing here re-splits, and the manifest stage re-verifies all of it.
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)   # so uv finds pyproject.toml
ADAPTER="$REPO/scripts/boundary_cognitive_maps.py"

WORKSPACE=${WORKSPACE:-/workspace}
EXPERIMENT=${EXPERIMENT:-$WORKSPACE/probes/2880_trajectory_trained_cogn_maps}
EVALUATION=${EVALUATION:-$WORKSPACE/loudness_evaluation/2880_trajectory-trained_cogn_maps}
PREPARED=${PREPARED:-$EXPERIMENT/prepared}
TREE=${TREE:-$PREPARED/activations}
LOGS=${LOGS:-$EXPERIMENT/logs}

# The lens ruler. The vocabulary is the COMMITTED grid one, used as-is: data/jlens is the
# repo copy and /workspace/jlens the deployed one, and its content hash -- not its path --
# is what the manifest records.
JLENS_DIR=${JLENS_DIR:-/workspace/jlens/gridenv}
SIGNAL_JSON=${SIGNAL_JSON:-$REPO/data/jlens/grid_tokens_full.json}
LENS=${LENS:-jlens}
# The shared plotting CLI defaults --signal-name to "direction"; this experiment measures
# the GRID vocabulary, and the name goes into every axis label and column it resolves.
SIGNAL_NAME=${SIGNAL_NAME:-grid}
LAYER=${LAYER:-15}

# Probe hyperparameters. From the plan, and matching configs/cell_identity_probes/*.conf
# where the two agree, so these four probes sit beside the published cognitive-map ones.
SEED=${SEED:-42}
NUM_EPOCHS=${NUM_EPOCHS:-50}
LEARNING_RATE=${LEARNING_RATE:-3e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.001}
BATCH_SIZE=${BATCH_SIZE:-2048}
HIDDEN_DIMS=${HIDDEN_DIMS:-1024}
DROPOUT=${DROPOUT:-0.0}
MAX_CELLS=${MAX_CELLS:-25}
PAD_TO_SIZE=${PAD_TO_SIZE:-15}
DEVICE=${DEVICE:-cuda:0}

# Extraction throughput. A SINGLE device: device_map="auto" across several GPUs produces
# NaNs for this MoE model.
# gpt-oss runs EAGER attention (it carries sinks), so memory is quadratic in the PADDED
# batch width, not linear in its token count. build_loudness_tables.py's 16384 is sized for
# an 80 GB card; this run is on 32 GB, where four ~1900-token chains in one batch ask for
# 6.6 GiB in a single allocation and die. Raise it only on a bigger card.
FWD_BATCH_SIZE=${FWD_BATCH_SIZE:-4}
FWD_BATCH_TOKENS=${FWD_BATCH_TOKENS:-4096}
IO_WORKERS=${IO_WORKERS:-8}

N_BOOT=${N_BOOT:-500}
N_BINS=${N_BINS:-10}
LIMIT=${LIMIT:-}                 # trajectories per partition; smoke runs only
STAGES=${STAGES:-"extract prepare train loudness evaluate compare plots manifest"}
DRY_RUN=${DRY_RUN:-}
FORCE=${FORCE:-}                 # redo a stage whose output already exists
# Loading gpt-oss-20b needs the `gpu` extra (accelerate, and the pinned kernels==0.12.0),
# which is not in the default dependencies -- and a `uv run` WITHOUT it syncs the extra
# back out of a venv that had it, breaking the next run that assumed accelerate was there.
UV_EXTRAS=${UV_EXTRAS:---extra gpu}

ARCHS=${ARCHS:-"lr mlp"}
BOUNDARIES=${BOUNDARIES:-"pre_reasoning post_reasoning"}
EVAL_PARTITIONS=${EVAL_PARTITIONS:-"eval_720 heldout_360"}
declare -A EVAL_DIR=([eval_720]=eval_720_data [heldout_360]=heldout_360_data)

COMMANDS_JSON="$LOGS/commands.json"
mkdir -p "$LOGS"
failed=""
run() {
    echo "+ $*"
    printf '%s\n' "$*" >> "$LOGS/commands.log"
    [ -n "$DRY_RUN" ] && return 0
    "$@"
}
has_stage() { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }

echo "experiment : $EXPERIMENT"
echo "evaluation : $EVALUATION"
echo "tree       : $TREE"
echo "vocabulary : $SIGNAL_JSON"
echo "stages     : $STAGES${LIMIT:+   (LIMIT=$LIMIT)}"
echo ""

# ---------------------------------------------------------------------- 1. extract
if has_stage extract; then
    echo "== extract =================================================="
    run uv run --project "$REPO" $UV_EXTRAS python "$ADAPTER" extract \
        --workspace "$WORKSPACE" --experiment-root "$EXPERIMENT" --tree "$TREE" \
        --layer "$LAYER" --device "$DEVICE" \
        --forward-batch-size "$FWD_BATCH_SIZE" --forward-batch-tokens "$FWD_BATCH_TOKENS" \
        --io-workers "$IO_WORKERS" \
        ${LIMIT:+--limit "$LIMIT"} ${FORCE:+--overwrite} \
        2>&1 | tee "$LOGS/extract.txt"
    [ "${PIPESTATUS[0]}" -eq 0 ] || failed="$failed extract"
fi

# ---------------------------------------------------------------------- 2. prepare
if has_stage prepare; then
    echo ""
    echo "== prepare =================================================="
    run uv run --project "$REPO" $UV_EXTRAS python "$ADAPTER" prepare \
        --workspace "$WORKSPACE" --experiment-root "$EXPERIMENT" --tree "$TREE" \
        --layer "$LAYER" --max-cells "$MAX_CELLS" --pad-to-size "$PAD_TO_SIZE" \
        --cell-seed "$SEED" \
        ${LIMIT:+--limit "$LIMIT"} ${FORCE:+--overwrite} \
        2>&1 | tee "$LOGS/prepare.txt"
    [ "${PIPESTATUS[0]}" -eq 0 ] || failed="$failed prepare"
fi

# ---------------------------------------------------------------------- 3. train
# The training set is the 2,880 and the eval set is the 720 -- as two SEPARATE manifests,
# never an internal --eval-split. A boundary manifest is token-major (three entries share
# one trajectory's grid), so a row-level split would put the same grid in both halves;
# train_cognitive_map_probe refuses one on such a manifest for exactly that reason.
if has_stage train; then
    echo ""
    echo "== train ===================================================="
    for arch in $ARCHS; do
        for boundary in $BOUNDARIES; do
            out="$EXPERIMENT/general_l15_${arch}/${boundary}"
            probe="$out/cognitive_map_probe_boundary_${boundary}_l15_${arch}.pt"
            log="$out/train.log"
            mkdir -p "$out"
            # The configuration lives beside the checkpoint and the log, so a folder is
            # self-describing without reading back up the wrapper.
            [ -n "$DRY_RUN" ] || cat > "$out/config.json" <<JSON
{
  "boundary": "$boundary",
  "model_type": "$arch",
  "layer": $LAYER,
  "train_manifest": "$PREPARED/${boundary}_train_2880",
  "eval_manifest": "$PREPARED/${boundary}_eval_720",
  "checkpoint": "$probe",
  "seed": $SEED,
  "num_epochs": $NUM_EPOCHS,
  "learning_rate": "$LEARNING_RATE",
  "weight_decay": "$WEIGHT_DECAY",
  "batch_size": $BATCH_SIZE,
  "hidden_dims": "$HIDDEN_DIMS",
  "dropout": $DROPOUT,
  "class_weight": "balanced",
  "normalize": true,
  "balance_classes_per_trajectory": false,
  "train_max_cells": $MAX_CELLS,
  "pad_to_size": $PAD_TO_SIZE,
  "held_out": "the 360 is not read by this stage"
}
JSON
            if [ -f "$probe" ] && [ -z "$FORCE" ]; then
                echo "- have $probe, skipping"
                continue
            fi
            echo ""
            echo "- ${arch} / ${boundary}"
            run uv run --project "$REPO" $UV_EXTRAS interp-cli train_cognitive_map_probe \
                --train-data-path "$PREPARED/${boundary}_train_2880" \
                --eval-data-path "$PREPARED/${boundary}_eval_720" \
                --output-path "$probe" \
                --model-type "$arch" \
                --hidden-dims "$HIDDEN_DIMS" \
                --learning-rate "$LEARNING_RATE" \
                --weight-decay "$WEIGHT_DECAY" \
                --dropout "$DROPOUT" \
                --num-epochs "$NUM_EPOCHS" \
                --batch-size "$BATCH_SIZE" \
                --class-weight balanced \
                --normalize \
                --subset 1.0 \
                --seed "$SEED" \
                --device "$DEVICE" \
                --cache-activations \
                --verbose \
                2>&1 | tee "$log"
            [ "${PIPESTATUS[0]}" -eq 0 ] || failed="$failed train:${arch}/${boundary}"
        done
    done
fi

# ---------------------------------------------------------------------- 4. loudness
if has_stage loudness; then
    echo ""
    echo "== loudness ================================================="
    run uv run --project "$REPO" $UV_EXTRAS python "$ADAPTER" loudness \
        --workspace "$WORKSPACE" --experiment-root "$EXPERIMENT" --tree "$TREE" \
        --layer "$LAYER" --lens "$LENS" --jlens-dir "$JLENS_DIR" --signal-json "$SIGNAL_JSON" \
        --device "$DEVICE" \
        ${LIMIT:+--limit "$LIMIT"} ${FORCE:+--overwrite} \
        2>&1 | tee "$LOGS/loudness.txt"
    [ "${PIPESTATUS[0]}" -eq 0 ] || failed="$failed loudness"
fi

# ---------------------------------------------------------------------- 5. evaluate
# ONE table per (dataset, boundary), carrying EVERY architecture. The probes of a boundary
# are scored on identical rows -- same tokens, same seeded cell draw, so the same
# `n_true_{class}` denominators -- which is the shape a counts-mode grid table and the
# shared grid figures are built for. Splitting them across tables would cost a second pass
# over the activations and would let a later concatenation pool one probe's hits against
# another's cells.
#
# A probe is still scored ONLY on the boundary it was trained on, and the 360 is scored
# only after the 720 -- the held-out numbers are the final ones and nothing upstream is
# allowed to consult them.
if has_stage evaluate; then
    echo ""
    echo "== evaluate ================================================="
    for partition in $EVAL_PARTITIONS; do
        for boundary in $BOUNDARIES; do
            out="$EVALUATION/${EVAL_DIR[$partition]}/${boundary}"
            probe_args=()
            missing=""
            for arch in $ARCHS; do
                probe="$EXPERIMENT/general_l15_${arch}/${boundary}/cognitive_map_probe_boundary_${boundary}_l15_${arch}.pt"
                if [ -f "$probe" ]; then
                    probe_args+=(--probe "$probe")
                else
                    missing="$missing $arch"
                fi
            done
            [ -z "$missing" ] || echo "!! no ${boundary} probe for:$missing" >&2
            if [ ${#probe_args[@]} -eq 0 ]; then
                failed="$failed evaluate:${partition}/${boundary}(no probes)"
                continue
            fi
            if [ -f "$out/summary.json" ] && [ -z "$FORCE" ]; then
                echo "- have $out/summary.json, skipping"
                continue
            fi
            echo ""
            echo "- ${partition} / ${boundary}"
            run uv run --project "$REPO" $UV_EXTRAS python "$ADAPTER" evaluate \
                --workspace "$WORKSPACE" --experiment-root "$EXPERIMENT" --tree "$TREE" \
                --layer "$LAYER" --lens "$LENS" --signal-json "$SIGNAL_JSON" \
                "${probe_args[@]}" --partition "$partition" --boundary "$boundary" \
                --out "$out" --device "$DEVICE" --n-boot "$N_BOOT" --seed "$SEED" \
                ${LIMIT:+--limit "$LIMIT"} ${FORCE:+--overwrite} \
                2>&1 | tee "$LOGS/evaluate_${partition}_${boundary}.txt"
            [ "${PIPESTATUS[0]}" -eq 0 ] || failed="$failed evaluate:${partition}/${boundary}"
        done
    done
fi

# ---------------------------------------------------------------------- 6. compare
# The pre/post contrast, per architecture, inside one dataset. This is the part the shared
# plotting registry cannot draw: its `grid` figures describe ONE table, and a contrast needs
# two.
if has_stage compare; then
    echo ""
    echo "== compare =================================================="
    for partition in $EVAL_PARTITIONS; do
        dataset_dir="$EVALUATION/${EVAL_DIR[$partition]}"
        if [ -f "$dataset_dir/comparison/mlp/aggregates.json" ] && [ -z "$FORCE" ]; then
            echo "- have $dataset_dir/comparison, skipping"
            continue
        fi
        echo ""
        echo "- ${partition}"
        run uv run --project "$REPO" $UV_EXTRAS python "$ADAPTER" compare \
            --workspace "$WORKSPACE" --experiment-root "$EXPERIMENT" --tree "$TREE" \
            --layer "$LAYER" --lens "$LENS" --signal-json "$SIGNAL_JSON" \
            --dataset-dir "$dataset_dir" --n-boot "$N_BOOT" --n-bins "$N_BINS" --seed "$SEED" \
            2>&1 | tee "$LOGS/compare_${partition}.txt"
        [ "${PIPESTATUS[0]}" -eq 0 ] || failed="$failed compare:${partition}"
    done
fi

# ---------------------------------------------------------------------- 7. plots
# The SHARED registry, not this experiment's own drawer. A boundary table is a counts-mode
# `grid` table -- `n_true_{class}`, `{probe}_n_correct`, `{probe}_correct_{class}` and a
# canonical loudness column -- which is exactly what `plotting/figures.py` registers its
# three grid figures against, so it is pointed at each table instead of the figures being
# written twice. It writes the decile tables beside the PNGs, so a reader who wants the
# number rather than the shape does not have to re-derive it.
#
# One invocation per (dataset, boundary): the datasets stay separate, and both
# architectures ride in one table, which is what `--grid-probe`'s default expects.
if has_stage plots; then
    echo ""
    echo "== plots ===================================================="
    for partition in $EVAL_PARTITIONS; do
        for boundary in $BOUNDARIES; do
            table="$EVALUATION/${EVAL_DIR[$partition]}/${boundary}/per_token.csv"
            out="$EVALUATION/${EVAL_DIR[$partition]}/${boundary}/figures"
            [ -f "$table" ] || { echo "- no table at $table, skipping"; continue; }
            if [ -f "$out/grid_accuracy_by_loudness.png" ] && [ -z "$FORCE" ]; then
                echo "- have $out, skipping"
                continue
            fi
            echo ""
            echo "- ${partition} / ${boundary}"
            run uv run --project "$REPO" $UV_EXTRAS python \
                "$REPO/telos_interp/loudness_analysis/plotting/figures.py" \
                --grid-table "$table" --out "$out" \
                --lens "$LENS" --signal-name "$SIGNAL_NAME" --layer "$LAYER" \
                --deciles "$N_BINS" --boot "$N_BOOT" \
                2>&1 | tee "$LOGS/plots_${partition}_${boundary}.txt"
            [ "${PIPESTATUS[0]}" -eq 0 ] || failed="$failed plots:${partition}/${boundary}"
        done
    done
fi

# ---------------------------------------------------------------------- 8. manifest
if has_stage manifest; then
    echo ""
    echo "== manifest ================================================="
    training_json="$LOGS/training_params.json"
    if [ -z "$DRY_RUN" ]; then
        cat > "$training_json" <<JSON
{
  "trainer": "interp-cli train_cognitive_map_probe",
  "seed": $SEED,
  "num_epochs": $NUM_EPOCHS,
  "learning_rate": "$LEARNING_RATE",
  "weight_decay": "$WEIGHT_DECAY",
  "batch_size": $BATCH_SIZE,
  "hidden_dims": "$HIDDEN_DIMS",
  "dropout": $DROPOUT,
  "class_weight": "balanced",
  "normalize": true,
  "balance_classes_per_trajectory": false,
  "subset": 1.0,
  "checkpoint_selection": "the 720; the trainer writes the final epoch's weights",
  "held_out": "the 360, used for final numbers only"
}
JSON
        python3 - "$LOGS/commands.log" "$COMMANDS_JSON" <<'PY'
import json, sys
lines = [l.rstrip("\n") for l in open(sys.argv[1])] if len(sys.argv) > 1 else []
json.dump(lines, open(sys.argv[2], "w"), indent=2)
PY
    fi
    run uv run --project "$REPO" $UV_EXTRAS python "$ADAPTER" manifest \
        --workspace "$WORKSPACE" --experiment-root "$EXPERIMENT" --tree "$TREE" \
        --evaluation-root "$EVALUATION" --layer "$LAYER" --lens "$LENS" \
        --jlens-dir "$JLENS_DIR" --signal-json "$SIGNAL_JSON" \
        --max-cells "$MAX_CELLS" --pad-to-size "$PAD_TO_SIZE" --cell-seed "$SEED" \
        --commands "$COMMANDS_JSON" --training-params "$training_json" \
        2>&1 | tee "$LOGS/manifest.txt"
    [ "${PIPESTATUS[0]}" -eq 0 ] || failed="$failed manifest"
fi

echo ""
echo "============================================================"
echo "SUMMARY (native-cell balanced accuracy, padding excluded)"
echo "============================================================"
printf '%-14s %-6s %-16s %s\n' "dataset" "arch" "boundary" "balanced accuracy"
for partition in $EVAL_PARTITIONS; do
    for arch in $ARCHS; do
        for boundary in $BOUNDARIES; do
            f="$EVALUATION/${EVAL_DIR[$partition]}/${boundary}/summary.json"
            if [ -f "$f" ]; then
                value=$(python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))['by_probe']
hit = [v for k, v in d.items() if k.rsplit('_', 1)[-1] == sys.argv[2]]
print(f\"{hit[0]['balanced_accuracy']:.4f}  [{hit[0]['balanced_accuracy_ci95'][0]:.4f}, {hit[0]['balanced_accuracy_ci95'][1]:.4f}]\" if hit else 'no such arm')
" "$f" "$arch")
            else
                value="not run"
            fi
            printf '%-14s %-6s %-16s %s\n' "$partition" "$arch" "$boundary" "$value"
        done
        f="$EVALUATION/${EVAL_DIR[$partition]}/comparison/${arch}/aggregates.json"
        if [ -f "$f" ]; then
            python3 -c "import json,sys; d=json.load(open(sys.argv[1]))['overall']; print(f\"  post - pre: {d['delta_balanced_accuracy']:+.4f} [{d['delta_ci95'][0]:+.4f}, {d['delta_ci95'][1]:+.4f}] over {d['n_trajectories']} trajectories\")" "$f"
        fi
    done
done

echo ""
[ -z "$failed" ] || { echo "FAILED:$failed"; exit 1; }
echo "Checkpoints : $EXPERIMENT/general_l15_{lr,mlp}/{pre,post}_reasoning"
echo "Evaluations : $EVALUATION"
echo "Manifest    : $EVALUATION/provenance/run_manifest.json"
