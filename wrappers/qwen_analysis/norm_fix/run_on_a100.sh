#!/usr/bin/env bash
# Qwen final-norm fix: every GPU command, in order, for one 80 GB card. See README.md beside this.
#
# Every Qwen lens output on disk applied the final RMSNorm as `* w`; Qwen3.5/3.6 is `* (1 + w)`.
# This re-emits the Qwen lens tables with the corrected norm. It is CSV-ONLY: no .pt is written,
# and every output goes to a NEW tree under $OUT_ROOT, so nothing already on disk is overwritten
# and the old tables stay for a before/after comparison.
#
# Each gather is the recorded wrapper's own invocation, with the same trajectories, sample,
# seed, lens and layers, so the corrected tables cover exactly the tokens the old ones did.
# Only the output tree changes, plus --no-save-activations:
#
#   stage      replaces the tables of                                         tokens      ~time
#   check      (3-token pass/fail: does the gather's norm reproduce the logits?)          ~5 min
#   eval_dir   2_dataset_creation/1_p2_selection/p2_eval_selection.sh          eval 52, 20%, L27   ~11 min
#   eval_grid  grid/2_dataset_creation/1_p2_selection/p2_eval_selection.sh     eval 52, 20%, L27   ~11 min
#   ho_dir     2_dataset_creation/1_p2_selection/heldout_sample.sh             held-out 70, all, L27  ~20 min
#   ho_grid    grid/2_dataset_creation/1_p2_selection/heldout_sample.sh        held-out 70, all, L27  ~20 min
#   prof_dir   1_loudest_layer/sample_loudness_profile.sh                      train 549, 5%, all layers  ~61 min
#   prof_grid  grid/1_loudest_layer/sample_loudness_profile.sh                 train 549, 5%, all layers  ~61 min
#
# ~3.2 h in all; the times are from the original runs' status files. One exception to "same
# sample": the old DIRECTION profile was 20%; it is redone at 5%, as the grid profile was, which
# is ~460k tokens and ~1 h instead of ~4 h. That is the only population that changes.
#
# The eval runs do NOT re-select. They re-score the same 20% sample; the recorded picks in the
# old trees are what the probes trained on, and README.md step 4 compares them to the corrected
# ranking. Re-selecting is a separate decision.
#
# Usage (on the A100 host):
#   bash wrappers/qwen_analysis/norm_fix/run_on_a100.sh                 # everything, in order
#   STAGES="check eval_dir eval_grid" bash .../run_on_a100.sh           # a subset
#   DRY_RUN=1 bash .../run_on_a100.sh                                   # print, run nothing
#
# Resumable: the gather skips any trajectory whose CSV already exists in its tree, so re-running
# after a crash picks up where it stopped. `check` must PASS before any gather runs.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)   # so uv finds pyproject.toml

MODEL=Qwen/Qwen3.6-35B-A3B
TRAJ=/workspace/trajectories/qwen3.6-35b/replayed_single_step
JLENS_DIR=/workspace/jlens/qwen3_6_35b
DIRECTION_JSON=$REPO/data/jlens/qwen/direction_tokens_full_qwen3-6-35b-a3b.json
GRID_JSON=$REPO/data/jlens/qwen/grid_tokens_pruned_qwen3-6-35b-a3b.json

OUT_ROOT=/workspace/activations/qwen_fixnorm
LOGS=/workspace/logs/qwen_fixnorm
STAGES=${STAGES:-"check eval_dir eval_grid ho_dir ho_grid prof_dir prof_grid"}
DRY_RUN=${DRY_RUN:-0}

# Shared by every gather: the recorded wrappers' values.
SEED=42
BATCH_SIZE=256          # reasoning tokens per lens matmul, caps the [B, vocab] logits
FORWARD_BATCH_SIZE=1    # one ~12k-token chain at a time

export CUDA_VISIBLE_DEVICES=0                       # ONE card: device_map across GPUs NaNs this MoE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# The 67 GiB of weights are already on the network volume. A fresh pod without HF_HOME would
# otherwise download them into its own home directory.
export HF_HOME=${HF_HOME:-/workspace/shared/hf_cache}
[ -d "$HF_HOME/hub/models--Qwen--Qwen3.6-35B-A3B" ] \
    || echo "!! no Qwen weights under $HF_HOME/hub: the first gather will download ~72 GB" >&2

mkdir -p "$OUT_ROOT" "$LOGS"
cd "$REPO"

for f in "$DIRECTION_JSON" "$GRID_JSON"; do
    [ -f "$f" ] || { echo "!! vocabulary not found: $f (data/jlens/qwen/*.json is not in git; copy it over)" >&2; exit 1; }
done
[ -f "$JLENS_DIR/Qwen3.6-35B-A3B_gridenv_jacobian_lens.pt" ] || { echo "!! no Qwen J-lens under $JLENS_DIR" >&2; exit 1; }
# The fix must be in this checkout, or every gather below re-writes the same wrong tables.
grep -q "norm_offset" telos_interp/jlens_utils/models.py \
    || { echo "!! this checkout has no norm_offset in telos_interp/jlens_utils/models.py: pull the fix first" >&2; exit 1; }

run() {  # run <stage> <command...>: logged, timed, recorded in status.txt
    local stage=$1; shift
    echo "=== $stage: $*"
    [ "$DRY_RUN" = 1 ] && return 0
    echo "START $stage $(date -u -Iseconds)" >> "$LOGS/status.txt"
    if "$@" > "$LOGS/$stage.log" 2>&1; then
        echo "END $stage rc=0 $(date -u -Iseconds)" >> "$LOGS/status.txt"
    else
        local rc=$?
        echo "END $stage rc=$rc $(date -u -Iseconds)" >> "$LOGS/status.txt"
        echo "!! $stage failed (rc=$rc); see $LOGS/$stage.log" >&2
        exit $rc
    fi
}

gather() {  # gather <out> <trajectories> <vocabulary> <signal-name|-> <sample_p> <layers>
    local out=$1 traj=$2 vocab=$3 name=$4 sample=$5 layers=$6
    local args=(
        --trajectory-paths "$traj" --activations-dir "$out" --model-id "$MODEL" --jlens_dir "$JLENS_DIR"
        --direction-mass-json "$vocab" --data_sample_p "$sample" --data-sample-seed "$SEED"
        --lens both --layers "$layers" --no-save-activations
        --batch-size "$BATCH_SIZE" --forward-batch-size "$FORWARD_BATCH_SIZE" --device cuda
    )
    [ "$name" = - ] || args+=(--signal-name "$name")
    uv run --extra gpu python telos_interp/loudness_analysis/build_loudness_tables.py "${args[@]}"
}

for stage in $STAGES; do
    case $stage in
        check)
            # Three consecutive reasoning tokens of one held-out trajectory. PASS or stop.
            traj_file=$(find "$TRAJ/heldout_72" -name "*.json" -path "*/size*/*" | sort | head -1)
            run check uv run --extra gpu python scripts/check_lens_final_norm.py \
                --trajectory "$traj_file" --jlens_dir "$JLENS_DIR"
            [ "$DRY_RUN" = 1 ] || tail -n 8 "$LOGS/check.log"
            ;;
        eval_dir)  run eval_dir  gather "$OUT_ROOT/eval52_direction"  "$TRAJ/mass_eval_144"  "$DIRECTION_JSON" -    0.2  27 ;;
        eval_grid) run eval_grid gather "$OUT_ROOT/eval52_grid"       "$TRAJ/mass_eval_144"  "$GRID_JSON"      grid 0.2  27 ;;
        ho_dir)    run ho_dir    gather "$OUT_ROOT/heldout70_direction" "$TRAJ/heldout_72"   "$DIRECTION_JSON" -    1.0  27 ;;
        ho_grid)   run ho_grid   gather "$OUT_ROOT/heldout70_grid"    "$TRAJ/heldout_72"     "$GRID_JSON"      grid 1.0  27 ;;
        prof_dir)  run prof_dir  gather "$OUT_ROOT/profile_p05_direction" "$TRAJ/mass_train_576" "$DIRECTION_JSON" - 0.05 all ;;
        prof_grid) run prof_grid gather "$OUT_ROOT/profile_p05_grid"  "$TRAJ/mass_train_576" "$GRID_JSON"      grid 0.05 all ;;
        *) echo "!! unknown stage: $stage" >&2; exit 1 ;;
    esac
done

echo
echo "done -> $OUT_ROOT/{eval52,heldout70,profile_p05}_{direction,grid}   logs + status -> $LOGS"
echo "next (CPU, any host): README.md steps 2-4"
