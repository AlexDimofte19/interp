#!/usr/bin/env bash
# Rerun the whole gpt-oss P2 GRID line under the GRID-ENVIRONMENT jlens, overwriting in place.
#
# WHY. The first round's gathers passed --jlens_dir /workspace/jlens, which holds the lens fit
# on WIKITEXT; the grid-environment fit is /workspace/jlens/gridenv. mass_from_activations.py
# --self-check on gptoss_grid_mass_l14_eval (2026-09-24): its jlens mass matches the wikitext J
# to 6e-5 and misses the gridenv J by 2.6-3.9 nats. So the grid jlens loudness, the jlens
# selection arm, and the L14 layer pick were all made with the wrong lens. The logit lens needs
# no J and was unaffected, but its arm is re-gathered too: one forward pass writes all three.
#
# TWO STAGES, because the layer is an output of the first:
#
#   STAGE=profile   GPU. Deletes the old profile tree, re-runs 1_loudest_layer/ with gridenv,
#                   and prints the per-layer J-lens grid mass. Then STOP: set the argmax as the
#                   default in grid_layer.sh (the one place the layer lives).
#   STAGE=rest      GPU then CPU. Deletes every downstream artifact of the old round (at the OLD
#                   layer 14 and at the new one), then: the train/eval selecting gathers and
#                   the held-out gather -> prepare -> train (binary + multiclass) -> cross-eval
#                   -> held-out prepare/eval/per-token -> the loudness tables the cross-model
#                   notebook reads (probe_performance_by_loudness.ipynb, gptoss rows).
#
# Needs a GPU that holds gpt-oss-20b in bf16 (~42 GB; kernels==0.12 forces the dequant). A
# 24 GB card OOMs. Deletion only with CONFIRM_DELETE=1; STAGE=rest also wants
# CONFIRM_LAYER=<the layer in grid_layer.sh>, so a stale default cannot slip through.
#
# Usage:
#   CONFIRM_DELETE=1 STAGE=profile bash wrappers/gptoss_analysis/grid/rerun_gridenv.sh
#   # edit grid_layer.sh to the printed argmax, then:
#   CONFIRM_DELETE=1 CONFIRM_LAYER=<L> STAGE=rest bash wrappers/gptoss_analysis/grid/rerun_gridenv.sh
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
source "$HERE/grid_layer.sh"   # GRID_LAYER

STAGE=${STAGE:?set STAGE=profile or STAGE=rest}
OLD_LAYER=14   # the wikitext round's layer; its paths are cleared too
A=/workspace/activations
LOUD=/workspace/loudness_probe_performance_analysis/tables

wipe() {  # path...
    for p in "$@"; do
        [ -e "$p" ] || continue
        [ "${CONFIRM_DELETE:-0}" = 1 ] || { echo "!! would delete $p -- rerun with CONFIRM_DELETE=1" >&2; exit 1; }
        echo "deleting $p"
        rm -rf "$p"
    done
}

case $STAGE in
profile)
    PROFILE_TREE=$A/gptoss_grid_loudness_profile
    PROFILE_OUT=/workspace/loudness_evaluation/gptoss_p2_grid_layer_profile
    wipe "$PROFILE_TREE" "$PROFILE_OUT"
    bash "$HERE/1_loudest_layer/sample_loudness_profile.sh"
    bash "$HERE/1_loudest_layer/join_loudness_profile.sh"
    cd "$REPO"
    for lens in jlens logitlens; do
        uv run python scripts/jlens_layer_profile.py "$PROFILE_TREE" --lens "$lens" \
            --direction-score logprob_mass_full \
            --out "$PROFILE_OUT/layer_profile_${lens}.json" --out-csv "$PROFILE_OUT/layer_profile_${lens}.csv"
    done
    echo
    echo "NEXT: set GRID_LAYER in $HERE/grid_layer.sh to the J-lens argmax above (currently $GRID_LAYER),"
    echo "      then CONFIRM_DELETE=1 CONFIRM_LAYER=<L> STAGE=rest bash $0"
    ;;
rest)
    [ "${CONFIRM_LAYER:-}" = "$GRID_LAYER" ] || {
        echo "!! grid_layer.sh says GRID_LAYER=$GRID_LAYER; pass CONFIRM_LAYER=$GRID_LAYER once it is the gridenv argmax" >&2
        exit 1
    }
    for L in $OLD_LAYER $GRID_LAYER; do
        wipe "$A/gptoss_grid_mass_l$L" "$A/gptoss_grid_mass_l${L}_eval" "$A/heldout360_l${L}_grid"
    done
    wipe /workspace/prepared/gptoss_p2_grid_{jlens,logitlens,random}_{train,val} /workspace/prepared/gptoss_p2_grid_heldout \
        /workspace/probes/gptoss_p2_grid \
        /workspace/results/gptoss_p2_grid/{cross_selection_eval,cross_selection_eval_multiclass,heldout,heldout_multiclass} \
        "$LOUD/gptoss_grid" "$LOUD/gptoss_direction" "$LOUD/random_eval/gptoss_grid" \
        "$LOUD/random_eval/gptoss_direction/stages/4_grid"

    # GPU: the three gathers (selection on train and eval, every held-out token)
    bash "$HERE/2_dataset_creation/1_p2_selection/p2_training_selection.sh"
    bash "$HERE/2_dataset_creation/1_p2_selection/p2_eval_selection.sh"
    bash "$HERE/2_dataset_creation/1_p2_selection/heldout_sample.sh"
    # prepare, train, cross-evaluate
    bash "$HERE/2_dataset_creation/build_grid_datasets.sh"
    bash "$HERE/3_train_and_eval_probes/train_binary_grid_probes_all_selections.sh"
    bash "$HERE/3_train_and_eval_probes/train_grid_probes_all_selections.sh"
    bash "$HERE/3_train_and_eval_probes/cross_eval_binary_grid_all_selections.sh"
    bash "$HERE/3_train_and_eval_probes/cross_eval_grid_all_selections.sh"
    # held-out: prepare, evaluate, per-token table with loudness
    bash "$HERE/4_loudness_evaluation/prepare_heldout_grid.sh"
    bash "$HERE/4_loudness_evaluation/eval_binary_probes_heldout.sh"
    bash "$HERE/4_loudness_evaluation/eval_multiclass_probes_heldout.sh"
    bash "$HERE/4_loudness_evaluation/score_heldout_per_token.sh"
    # the cross-model notebook's gpt-oss tables (the direction ones carry the grid ruler)
    bash "$REPO/wrappers/loudness_analysis/join_opposite_signal.sh"
    ONLY="gptoss_direction gptoss_grid" bash "$REPO/wrappers/loudness_analysis/build_random_eval_tables.sh"
    echo
    echo "done. Re-run the notebooks: 4_loudness_evaluation/*.ipynb and wrappers/loudness_analysis/probe_performance_by_loudness.ipynb"
    ;;
*) echo "!! STAGE must be profile or rest" >&2; exit 1 ;;
esac
