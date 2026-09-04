#!/usr/bin/env bash
# Recreate every result in the jlens -> probe -> rollout line, from scratch, in order.
#
# One stage per experiment, each named for what it produced rather than for the log entry that
# produced it. Every stage is RESUMABLE: it tests for its own output on disk first and skips
# itself when that output is already there, so running this on the host that produced the work
# is a no-op check, and running it on an empty machine rebuilds the line end to end.
#
#   LIST=1 ./scripts/reproduce_all.sh          what exists, what does not, and what each costs
#   DRY_RUN=1 ./scripts/reproduce_all.sh       print every command, run nothing
#   ./scripts/reproduce_all.sh                 build whatever is missing, in dependency order
#   ONLY="sentence_loudness" ./scripts/reproduce_all.sh
#   FROM=mass_era_gather ./scripts/reproduce_all.sh
#   SKIP="cognitive_map_probes" FORCE=1 ./scripts/reproduce_all.sh
#
# WHAT CANNOT BE REPRODUCED BY RE-RUNNING, and is therefore pinned rather than re-drawn:
#
#   * A TRAJECTORY SAMPLE. --per-combo/--seed does NOT reproduce a previous draw: two runs with
#     identical flags overlapped by 348 of 3600. Every stage here pins its set with a names
#     file, and those files are inputs, not outputs -- if they are gone, the sets they name are
#     gone with them. They are the first thing to back up.
#   * A CONTROL ARM'S DRAW. A uniform draw over the reasoning chain can only be made before the
#     tree is pruned to the selection. Afterwards the record is the only place it survives, so
#     controls are replayed (`recorded_*`, `recorded_selection`), never re-drawn.
#
# WHAT THIS SCRIPT DOES NOT COVER, because it was never run (do not let the stage list imply
# otherwise): the sentence-end token cap; the random-position truncation arm; the arm-vs-arm
# truncation comparison; regenerating the 165 rollout files written by the pre-fix batching.
# The first three are behind UNRUN=1 as explicitly-labelled NEW work, not reproduction.
#
# GPU stages need one visible card -- device_map="auto" across several GPUs gives silent NaNs
# on this MoE -- and the `gpu` extra. A `uv run` without --extra gpu syncs accelerate back OUT
# of the venv, so every model-loading call here passes it.
#
# Do not run two stages against one activation tree at once, and never while a prune or an
# --extend gather is writing to it.
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)   # so uv finds pyproject.toml

# Roots.
WS=${WS:-/workspace}
ACT=${ACT:-$WS/activations}
PREPARED=${PREPARED:-$WS/prepared}
PROBES=${PROBES:-$WS/probes}
RT=${RT:-$WS/reasoning_theatre}
TRAJ_ROOT=${TRAJ_ROOT:-$WS/trajectories}
JLENS_DIR=${JLENS_DIR:-$WS/jlens/gridenv}
SIGNAL_JSON=${SIGNAL_JSON:-$WS/jlens/direction_tokens_full.json}
GRID_SIGNAL_JSON=${GRID_SIGNAL_JSON:-$WS/jlens/grid_tokens_full.json}
LOGROOT=${LOGROOT:-$WS/logs/reproduce}

# The trajectory sets. Pinned BY NAME, never by seed -- see the header.
TRAJ=${TRAJ:-$TRAJ_ROOT/reveng/trajectories_train_single_step}
COUNT_NAMES=${COUNT_NAMES:-$WS/splits/lens_trajectories_3600.txt}
COUNT_EVAL_NAMES=${COUNT_EVAL_NAMES:-$WS/splits/eval_trajectories_720.txt}
MASS_NAMES=${MASS_NAMES:-$RT/rollout_strategies/mass_l15_names.txt}
MASS_EVAL_NAMES=${MASS_EVAL_NAMES:-$PREPARED/next_action_mass_l15_eval_names.txt}
HELDOUT_NAMES=${HELDOUT_NAMES:-$TRAJ_ROOT/heldout360_names.txt}
HELDOUT_TRAJ=${HELDOUT_TRAJ:-$TRAJ_ROOT/heldout360}

# Selection knobs, frozen at the values the trees on disk were built with.
LAYERS=${LAYERS:-7:23}
SELECT_SEED=${SELECT_SEED:-42}
SEED=${SEED:-42}

# Driver.
LIST=${LIST:-}          # print the stage table and exit
DRY_RUN=${DRY_RUN:-}    # print commands, run nothing
FORCE=${FORCE:-}        # run a stage even when its output is already there
ONLY=${ONLY:-}          # whitespace-separated stage names; everything else is skipped
SKIP=${SKIP:-}          # whitespace-separated stage names to skip
FROM=${FROM:-}          # start at this stage, skip everything before it
GRID_ROUND=${GRID_ROUND:-}      # include the grid-label round (it never produced a result)
UNRUN=${UNRUN:-}                # include the analyses that were written but never run
REBUILD_VOCAB=${REBUILD_VOCAB:-} # re-run the vocabulary notebooks instead of deploying the repo copies
UV_EXTRAS=${UV_EXTRAS:---extra gpu}

UV="uv run --project $REPO $UV_EXTRAS"
UVC="uv run --project $REPO"     # CPU-only stages; no accelerate needed
ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }

# Echo a command, and run it unless DRY_RUN. Env-prefixed calls go through `env` so they stay
# one argv and print faithfully.
x() {
    printf '      + '; printf '%q ' "$@"; echo
    [ -n "$DRY_RUN" ] && return 0
    "$@"
}

# Marker helpers. `nfiles <dir> <pattern> <n>` is true when at least n matches exist. Counting
# is always shallow or name-directed: a recursive find over these trees takes minutes.
nfiles() { [ "$(find "$1" -maxdepth "${4:-1}" -name "$2" 2>/dev/null | head -n "$3" | wc -l)" -ge "$3" ]; }
ndirs()  { [ "$(find "$1" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -n "$2" | wc -l)" -ge "$2" ]; }

# ---------------------------------------------------------------------------------------
# The stages, in dependency order.
# ---------------------------------------------------------------------------------------
STAGES="host_setup deploy_vocabularies fit_jacobian_lens cognitive_map_probes
        sentence_end_rollout sentence_end_activation_tree
        count_era_gather count_era_next_action_arms
        mass_era_gather mass_era_next_action_probes
        heldout_trees score_probes_on_heldout probe_vs_rollout
        sentence_loudness loud_rollout_arms local_belief_probes
        probe_loudness_eval720 belief_probes_commitment
        heldout_every_token_rollout probe_loudness_heldout
        logitlens_mass_gather belief_baseline_rollout_arms belief_baseline_probes
        sixteen_probe_loudness_report"
[ -n "$GRID_ROUND" ] && STAGES="$STAGES grid_probing_round"
[ -n "$UNRUN" ] && STAGES="$STAGES unrun_analyses"

# meta <stage> -> "<gpu>|<cost>|<title>"
meta() {
    case "$1" in
    host_setup)              echo "no|minutes|Host bootstrap: uv, the gpu extra, the HF cache";;
    deploy_vocabularies)     echo "no|seconds|Direction and grid vocabularies, built from the model vocabulary alone";;
    fit_jacobian_lens)       echo "GPU|not recorded|Fit the Jacobian lens on grid-environment prompts";;
    cognitive_map_probes)    echo "GPU|not recorded|The published cognitive-map (grid-cell) probes";;
    sentence_end_rollout)    echo "GPU|not recorded|Sentence-end rollout over all 36,000 trajectories";;
    sentence_end_activation_tree) echo "GPU|not recorded|Layer-15 activations at every sentence end";;
    count_era_gather)        echo "GPU|not recorded|Count-scored selective gather, then the logit-lens arm";;
    count_era_next_action_arms) echo "GPU|9h39m + 2h02m + 3h12m + 4h27m|Next-action arms: pooled 7:23, layer 15, sentence-end, three seeds";;
    mass_era_gather)         echo "GPU|2h34m|Probability-mass ranking at layer 15";;
    mass_era_next_action_probes) echo "GPU|~1h|Mass-era probes: jlens top-1/2/3/all and the random control";;
    heldout_trees)           echo "GPU|45m + 6m|The held-out 360: layer-15 activations, and both lenses' tables";;
    score_probes_on_heldout) echo "GPU|~2h|Every probe on every held-out token, under both rankings";;
    probe_vs_rollout)        echo "GPU|~35m + CPU|The probe reads the current belief: join, tables, commitment";;
    sentence_loudness)       echo "no|~5m|Loudness through a sentence, over the training 2,880";;
    loud_rollout_arms)       echo "GPU|6h03m + 3h55m|Cut where the lens is loud: per-sentence argmax, global top-k";;
    local_belief_probes)     echo "GPU|~45m + train|Relabel to the model's belief at the cut: six probes";;
    probe_loudness_eval720)  echo "no|~15m|Probe accuracy read by loudness, on the eval 720";;
    belief_probes_commitment) echo "GPU|~2h|The commitment boundary re-read by the belief-trained probes";;
    heldout_every_token_rollout) echo "GPU|6h13m|Dense rollout at every held-out reasoning token";;
    probe_loudness_heldout)  echo "GPU|1h40m + CPU|Loudness with the selection removed: ten probes, all 87,221 tokens";;
    logitlens_mass_gather)   echo "GPU|1h55m|The logit-lens twin of the mass tree, pinned by name";;
    belief_baseline_rollout_arms) echo "GPU|14h13m|Belief-baseline rollout arms: random replay, two logit-lens arms";;
    belief_baseline_probes)  echo "GPU|~2h|Belief-baseline probes: label and selection separated";;
    sixteen_probe_loudness_report) echo "GPU|~2h + CPU|Sixteen probes on the held-out 360, under both loudness rulers";;
    grid_probing_round)      echo "GPU|not recorded|Grid-label twin of the arms -- NEVER RAN, no result exists";;
    unrun_analyses)          echo "mixed|~1h|Written but never run: NEW work, not reproduction";;
    esac
}

# ---- markers -----------------------------------------------------------------------------
done_host_setup()              { [ -x "$REPO/.venv/bin/python" ]; }
done_deploy_vocabularies()     { [ -s "$SIGNAL_JSON" ] && [ -s "$GRID_SIGNAL_JSON" ]; }
done_fit_jacobian_lens()       { [ -s "$JLENS_DIR/gpt-oss-20b_jacobian_lens.pt" ]; }
done_cognitive_map_probes()    { nfiles "$PROBES/start_of_reasoning/downloaded" 'cognitive_map_probe_*.pt' 1; }
done_sentence_end_rollout()    { ndirs "$RT/trajectories_train_single_step_probs" 6; }
done_sentence_end_activation_tree() { ndirs "$ACT/activations_train_single_step_reasoning_eos" 6; }
done_count_era_gather()        { nfiles "$ACT/jlens_reasoning_tokens" '*_logitlens_analysis.csv' 1 4; }
done_count_era_next_action_arms() { nfiles "$PROBES/next_action_seeds" '*.pt' 18; }
done_mass_era_gather()         { nfiles "$ACT/jlens_mass_l15" '*_direction_mass.csv' 1 4; }
done_mass_era_next_action_probes() { nfiles "$PROBES/next_action_mass_l15" '*.pt' 8; }
done_heldout_trees()           { ndirs "$ACT/heldout360_l15" 6 && nfiles "$ACT/heldout360_lens" '*_direction_mass.csv' 1 4; }
done_score_probes_on_heldout() { [ -s "$PROBES/heldout360_all_probes.csv" ]; }
done_probe_vs_rollout()        { [ -s "$RT/probe_vs_rollout/summary.json" ] && [ -s "$RT/probe_vs_rollout/per_token_probs.csv" ]; }
done_sentence_loudness()       { [ -s "$RT/loudness/summary.json" ]; }
done_loud_rollout_arms()       { nfiles "$RT/rollout_strategies/jlens_top_k_global" '*.json' 1 2; }
done_local_belief_probes()     { nfiles "$RT/local_belief_probes/probes" '*.pt' 6; }
done_probe_loudness_eval720()  { [ -s "$RT/probe_loudness/summary.json" ]; }
done_belief_probes_commitment(){ [ -s "$RT/probe_vs_rollout_lb/summary.json" ]; }
done_heldout_every_token_rollout() { nfiles "$RT/rollout_strategies_heldout360/every_token" '*.json' 1 2; }
done_probe_loudness_heldout()  { [ -s "$RT/probe_loudness_heldout360/summary.json" ]; }
done_logitlens_mass_gather()   { nfiles "$ACT/logitlens_mass_l15" '*_direction_mass.csv' 1 4; }
done_belief_baseline_rollout_arms() { nfiles "$RT/rollout_strategies_baselines/recorded_selection" '*.json' 1 2; }
done_belief_baseline_probes()  { nfiles "$PROBES/local_belief_baselines" '*.pt' 6; }
done_sixteen_probe_loudness_report() { [ -s "$RT/probe_loudness_heldout360_16probes/report.html" ]; }
done_grid_probing_round()      { nfiles "$PROBES/grid" '*.pt' 1; }
done_unrun_analyses()          { [ -s "$RT/rollout_strategies/truncation_comparison/summary.json" ]; }

# ---- stage 0: host bootstrap --------------------------------------------------------------
run_host_setup() {
    x bash "$REPO/runpod_setup.sh"
}

# ---- stage 1: the signal vocabularies ------------------------------------------------------
# Built from the model vocabulary alone, never from what is frequent in the j-space -- the
# j-space is what they measure. The repo copies are the deployed ones; see data/jlens/README.md
# for the grid file's known pre-rule token count.
run_deploy_vocabularies() {
    if [ -n "$REBUILD_VOCAB" ]; then
        for nb in direction_tokens grid_tokens; do
            x $UVC --extra notebook jupyter run "$REPO/notebooks/$nb.ipynb"
        done
    fi
    x mkdir -p "$(dirname "$SIGNAL_JSON")"
    x cp "$REPO/data/jlens/direction_tokens_full.json" "$SIGNAL_JSON"
    x cp "$REPO/data/jlens/grid_tokens_full.json" "$GRID_SIGNAL_JSON"
}

# ---- stage 2: the Jacobian lens ------------------------------------------------------------
# Fitted on 1000 grid-environment prompts drawn from the TRAINING set, so the test set is
# untouched. Also writes the unembed cache every later lens run reads -- which is why
# --jlens_dir is needed even for --lens logitlens.
run_fit_jacobian_lens() {
    x $UV python "$REPO/jlens/jlens_fit_gpt_oss.py" \
        --trajectories-dir "$TRAJ" --out-dir "$JLENS_DIR" \
        --n-prompts 1000 --eval-every 10 --stop-at-delta 0.002
}

# ---- stage 3: the cognitive-map probes -----------------------------------------------------
# The published grid-cell line, and the only stage whose recorded invocation is a top-level
# script rather than one under scripts/. script.sh is a RECORDED INVOCATION -- it is the record
# of how the published probes were produced, so it is run verbatim rather than refactored.
# NOTE: the probes on this host under $PROBES/{start,end}_of_reasoning/downloaded/ were pulled
# from the Hub, not trained here; this stage retrains them from the activations.
run_cognitive_map_probes() {
    x bash "$REPO/script.sh"
}

# ---- stage 4: the sentence-end rollout -----------------------------------------------------
# The arm every later "sentence-end answer" is read from, over the whole 36,000. --skip-existing
# makes it resumable; the rollout root is a FUSE mount, so it is addressed by name and never
# globbed.
run_sentence_end_rollout() {
    x $UV python "$REPO/scripts/inference_oss/run_inference.py" \
        --strategy eos --trajectory-paths "$TRAJ" \
        --output-dir "$RT/trajectories_train_single_step_probs" \
        --skip-existing
    x bash "$REPO/scripts/inference_oss/run_analysis.sh"
}

# ---- stage 5: the sentence-end activation tree ---------------------------------------------
# One layer, one step, output tokens at sentence ends only: the tree the sentence-end probe arm
# trains on. No lens selection and no selection record, so prepare's default token_selection
# "all" is the right mode against it.
run_sentence_end_activation_tree() {
    x $UV interp-cli gather_activations \
        --trajectory-paths "$TRAJ/*/*.json" \
        --output-dir "$ACT/activations_train_single_step_reasoning_eos" \
        --layers 15 --steps 0 --output-indices eos
}

# ---- stage 6: the count-era gather ---------------------------------------------------------
# Two passes over one tree. The first selects and saves the jlens arm plus the matched random
# control -- the control has to be drawn HERE, before anything is pruned, because a uniform draw
# over the chain can never be made again afterwards. The second adds the logit-lens arm to the
# already-pruned tree, which is the only way to add an arm: re-filtering cannot recover tokens
# that were deleted. It merges into the record and inherits the control rather than redrawing.
run_count_era_gather() {
    x env NAMES_FILE="$COUNT_NAMES" TRAJECTORIES="$TRAJ" \
        JLENS_DIR="$JLENS_DIR" SIGNAL_JSON="$SIGNAL_JSON" \
        ACTIVATIONS_DIR="$ACT/jlens_reasoning_tokens" LAYERS="$LAYERS" \
        NUM_TOKENS=20 NUM_LAYERS=3 ALWAYS_LAYERS=15 RANDOM_TOKENS=20 SELECT_SEED="$SELECT_SEED" \
        bash "$REPO/scripts/jlens_reasoning_tokens_filtered.sh"
    x env ACT="$ACT/jlens_reasoning_tokens" TRAJ="$TRAJ" JLENS_DIR="$JLENS_DIR" \
        SIGNAL_JSON="$SIGNAL_JSON" METHODS=logitlens NUM_TOKENS=20 NUM_LAYERS=3 \
        ALWAYS_LAYERS=15 SELECT_SEED="$SELECT_SEED" APPLY=1 ASSUME_YES=1 \
        bash "$REPO/scripts/jlens_extend_logitlens.sh"
}

# ---- stage 7: the count-era next-action arms -----------------------------------------------
# Four runs off one tree. OUT and PROBES are overridden per run and both are load-bearing: share
# either and the previous run's datasets, probes and logs are clobbered.
#
#   pooled 7:23     jlens and logitlens across the layer range
#   layer 15        the same jlens arm pinned to one layer -- which BEAT the pooled one, so
#                   layer selection was negative and jlens > logitlens at 7:23 is not a lens
#                   result but a layer-spread confound
#   random          the control, pinned to layer 15 at PREPARE time: with no scores to rank by,
#                   thinning samples layers uniformly and the control would drift off L15
#   sentence-end    the fourth arm, over a symlink view of the eos tree restricted to the same
#                   3600 trajectories, so the comparison is not confounded by data volume
#   three seeds     42/43/44, training seed only -- the split stays at 42
run_count_era_next_action_arms() {
    x env ARMS="jlens logitlens" LAYERS="$LAYERS" ACT="$ACT/jlens_reasoning_tokens" \
        TRAJ="$TRAJ" OUT="$PREPARED/next_action" \
        bash "$REPO/scripts/prepare_next_action_arms.sh"
    x env ARMS="random" LAYERS=15 ACT="$ACT/jlens_reasoning_tokens" \
        TRAJ="$TRAJ" OUT="$PREPARED/next_action" \
        bash "$REPO/scripts/prepare_next_action_arms.sh"
    x env ARMS="jlens logitlens random" TOKENS_PER_TRAJ=all LAYERS_PER_TOKEN=1 \
        PREPARED="$PREPARED/next_action" PROBES="$PROBES/next_action" \
        bash "$REPO/scripts/train_next_action_arms.sh"

    # The layer-15 follow-up. Both overrides matter -- see above.
    x env ARMS="jlens" LAYERS=15 ACT="$ACT/jlens_reasoning_tokens" TRAJ="$TRAJ" \
        OUT="$PREPARED/next_action_l15" \
        bash "$REPO/scripts/prepare_next_action_arms.sh"
    x env ARMS="jlens" TOKENS_PER_TRAJ=all LAYERS_PER_TOKEN=1 \
        PREPARED="$PREPARED/next_action_l15" PROBES="$PROBES/next_action_l15" \
        bash "$REPO/scripts/train_next_action_arms.sh"

    # The sentence-end arm needs a view of the eos tree holding exactly the lens 3600, so the
    # two arms differ in their tokens and not in how much data they saw.
    build_eos_view
    x env ARMS="eos" LAYERS=15 ACT="$ACT/eos_lens3600_view/activations" \
        TRAJ="$ACT/eos_lens3600_view/trajectories" OUT="$PREPARED/next_action_eos" \
        bash "$REPO/scripts/prepare_next_action_arms.sh"
    x env ARMS="eos" TOKENS_PER_TRAJ=all LAYERS_PER_TOKEN=1 EVAL_NAMES="$COUNT_EVAL_NAMES" \
        PREPARED="$PREPARED/next_action_eos" PROBES="$PROBES/next_action_eos" \
        bash "$REPO/scripts/train_next_action_arms.sh"

    # The seed sweep. Seed 42 is already trained above and is reused, not repeated.
    for s in 43 44; do
        x env ARMS="jlens logitlens random eos" TOKENS_PER_TRAJ=all LAYERS_PER_TOKEN=1 \
            SEED="$s" PREPARED="$PREPARED/next_action" PROBES="$PROBES/next_action_seeds" \
            bash "$REPO/scripts/train_next_action_arms.sh"
    done
}

# A symlink view: 3600 activation dirs and 3600 trajectory JSONs, no bytes copied. Rebuilt only
# when the link count is short, so a resumed run costs one readdir.
build_eos_view() {
    local view="$ACT/eos_lens3600_view"
    local src="$ACT/activations_train_single_step_reasoning_eos"
    if [ "$(find "$view" -maxdepth 3 -type l 2>/dev/null | head -7200 | wc -l)" -ge 7200 ]; then
        echo "      (eos view already complete)"
        return 0
    fi
    echo "      + build $view (3600 activation + 3600 trajectory symlinks)"
    [ -n "$DRY_RUN" ] && return 0
    while read -r name; do
        [ -z "$name" ] && continue
        local size=${name#*_size}; size="size${size%%_*}"
        mkdir -p "$view/activations/$size" "$view/trajectories/$size"
        ln -sfn "$src/$size/$name" "$view/activations/$size/$name"
        ln -sfn "$TRAJ/$size/$name.json" "$view/trajectories/$size/$name.json"
    done < "$COUNT_NAMES"
}

# ---- stage 8: the mass-era gather ----------------------------------------------------------
# Ranks tokens by log P(any direction word) over the WHOLE vocabulary, computed while the logits
# are still on the device, and ranks at layer 15 rather than by a cross-layer total. The CSV and
# the mass table still cover 7:23, so the full layer profile survives for jlens_layer_profile.py
# even though only layer 15 lands on disk. Every mass table is written with a .meta.json naming
# the vocabulary that produced it -- never read one without it.
run_mass_era_gather() {
    x env NAMES_FILE="$MASS_NAMES" TRAJECTORIES="$TRAJ" JLENS_DIR="$JLENS_DIR" \
        SIGNAL_JSON="$SIGNAL_JSON" ACTIVATIONS_DIR="$ACT/jlens_mass_l15" \
        bash "$REPO/scripts/jlens_mass_l15.sh"
}

# ---- stage 9: the mass-era next-action probes ----------------------------------------------
# The top-K sweep is three SPLITS off one prepared dataset, not three prepares: split_next_action
# _manifest.py computes the strata from the unthinned samples, so every K shares one split and
# the results compare.
run_mass_era_next_action_probes() {
    for arm in jlens random; do
        x env ARMS="$arm" LAYERS=15 ACT="$ACT/jlens_mass_l15" TRAJ="$TRAJ" \
            OUT="$PREPARED/next_action_mass_l15" \
            bash "$REPO/scripts/prepare_next_action_arms.sh"
    done
    x env ARMS="jlens random" TOKENS_PER_TRAJ="1 2 3 all" LAYERS_PER_TOKEN=1 SINGLE_LAYER=15 \
        EVAL_NAMES="$MASS_EVAL_NAMES" \
        PREPARED="$PREPARED/next_action_mass_l15" PROBES="$PROBES/next_action_mass_l15" \
        bash "$REPO/scripts/train_next_action_arms.sh"
}

# ---- stage 10: the held-out 360 trees ------------------------------------------------------
# TWO passes, because --layers drives what is SAVED as well as what is analysed. The first saves
# layer-15 activations for every reasoning token (87,221 of them, no selection at all); the
# second writes both lenses' analysis CSVs and both mass tables across 7:23 and saves no .pt.
# These 360 trajectories are disjoint from the 3600 -- they are the only set nothing was
# selected or trained on.
run_heldout_trees() {
    x $UV python "$REPO/scripts/jlens_reasoning_tokens.py" \
        --trajectory-paths "$HELDOUT_TRAJ" --names-file "$HELDOUT_NAMES" \
        --jlens_dir "$JLENS_DIR" --activations-dir "$ACT/heldout360_l15" \
        --signal-json "$SIGNAL_JSON" --lens jlens --layers 15 --steps all
    x $UV python "$REPO/scripts/jlens_reasoning_tokens.py" \
        --trajectory-paths "$HELDOUT_TRAJ" --names-file "$HELDOUT_NAMES" \
        --jlens_dir "$JLENS_DIR" --activations-dir "$ACT/heldout360_lens" \
        --signal-json "$SIGNAL_JSON" --lens both --layers "$LAYERS" --steps all \
        --direction-score logprob_mass_full --no-save-activations
}

# ---- stage 11: every probe on every held-out token -----------------------------------------
# There is no eval_next_action_probe in the CLI; this is that tool. Probes are keyed
# "<parent dir>.<stem>" because the same filename exists in three probe directories.
run_score_probes_on_heldout() {
    local args=()
    for p in "$PROBES"/next_action/*.pt "$PROBES"/next_action_l15/*.pt \
             "$PROBES"/next_action_eos/*.pt "$PROBES"/next_action_seeds/*.pt \
             "$PROBES"/next_action_mass_l15/*.pt; do
        [ -e "$p" ] && args+=(--probe "$p")
    done
    x $UV python "$REPO/scripts/eval_probe_per_token.py" "${args[@]}" \
        --activations-dir "$ACT/heldout360_l15" --lens-dir "$ACT/heldout360_lens" \
        --trajectories-dir "$TRAJ" --signal-json "$SIGNAL_JSON" \
        --layer 15 --out "$PROBES/heldout360_all_probes.csv"
}

# ---- stage 12: the probe reads the current belief ------------------------------------------
# Joins the sentence-end rollout to the per-token probe verdicts. The join is on
# (name, step, token_idx) -- token_idx indexes output_tokens, while abs_pos is prompt-inclusive;
# joining on the wrong one yields an empty table and no error.
#
# The direction-class pass supersedes the four-literal-word readout: quote its AUC, and never
# the argmax accuracy (the argmax is degenerate, ~63% one class).
run_probe_vs_rollout() {
    x $UVC python "$REPO/scripts/build_probe_rollout_join.py" \
        --probe-csv "$PROBES/heldout360_all_probes.csv" \
        --probs-root "$RT/trajectories_train_single_step_probs" \
        --lens-root "$ACT/heldout360_l15" --direction-json "$SIGNAL_JSON" \
        --out "$RT/probe_vs_rollout/per_token.csv"
    x $UVC python "$REPO/scripts/analyze_probe_rollout.py" \
        --per-token "$RT/probe_vs_rollout/per_token.csv" --out-dir "$RT/probe_vs_rollout"
    x $UVC python "$REPO/scripts/analyze_jlens_direction_classes.py" \
        --per-token "$RT/probe_vs_rollout/per_token.csv" \
        --direction-json "$SIGNAL_JSON" --out-dir "$RT/probe_vs_rollout"
    x env LENS_ROOT="$ACT/heldout360_l15" $UVC python "$REPO/scripts/jlens_direction_vocab_diagnostic.py"
    x $UVC python "$REPO/scripts/plot_probe_rollout.py" --dir "$RT/probe_vs_rollout"

    # Commitment at token resolution. The probabilities are the readout that shows a sentence
    # opening on the PREVIOUS belief; the mlp arm is near one-hot, so read the lr arm.
    x $UV python "$REPO/scripts/eval_probe_per_token.py" \
        --probe "$PROBES/next_action_mass_l15/next_action_probe_jlens_topall_lr.pt" \
        --probe "$PROBES/next_action_mass_l15/next_action_probe_jlens_topall_mlp.pt" \
        --probe "$PROBES/next_action_mass_l15/next_action_probe_random_topall_lr.pt" \
        --probe "$PROBES/next_action_mass_l15/next_action_probe_random_topall_mlp.pt" \
        --activations-dir "$ACT/heldout360_l15" --lens-dir "$ACT/heldout360_lens" \
        --trajectories-dir "$TRAJ" --signal-json "$SIGNAL_JSON" \
        --layer 15 --full-probs --out "$RT/probe_vs_rollout/per_token_probs.csv"
    x $UVC python "$REPO/scripts/plot_commitment_all_tokens.py"
    x $UVC python "$REPO/scripts/plot_commitment_probs.py"
}

# ---- stage 13: loudness through a sentence -------------------------------------------------
# Over the TRAINING 2,880 (the mass tree minus the pinned eval names), so these are distribution
# statements and never an accuracy number. The header offset when placing a token in its
# sentence is eos[0] + 1 per trajectory, not a constant.
run_sentence_loudness() {
    x $UVC python "$REPO/scripts/build_sentence_loudness.py" \
        --lens-root "$ACT/jlens_mass_l15" --probs-root "$RT/trajectories_train_single_step_probs" \
        --eval-names "$MASS_EVAL_NAMES" --direction-tokens-path "$SIGNAL_JSON" \
        --out "$RT/loudness/per_token.csv"
    x $UVC python "$REPO/scripts/plot_sentence_loudness.py" \
        --per-token "$RT/loudness/per_token.csv" --out "$RT/loudness"
    x $UVC python "$REPO/scripts/analyze_sentence_loudness.py" \
        --per-token "$RT/loudness/per_token.csv" --out "$RT/loudness"
}

# ---- stage 14: the loud rollout arms -------------------------------------------------------
# Same trajectory set and same endpoints as the sentence-end grid; only WHERE the cut goes
# changes. Raw accuracies between arms are NOT comparable -- the arms cut different numbers of
# tokens, and there is a ~3.3% intrinsic disagreement floor at the shared endpoints from batching
# alone. Do not raise BATCH_SIZE or MAX_ATTN_ELEMS: attention is eager and allocates rows x
# heads x L^2.
run_loud_rollout_arms() {
    x env TRAJECTORIES="$TRAJ" LENS_ROOT="$ACT/jlens_mass_l15" \
        OUT_ROOT="$RT/rollout_strategies" NAMES_FILE="$MASS_NAMES" \
        bash "$REPO/scripts/inference_oss/run_inference_strategies.sh" \
        jlens_argmax_per_sentence jlens_top_k_global
    for arm in jlens_argmax_per_sentence jlens_top_k_global; do
        x $UVC python "$REPO/scripts/plot_loud_vs_sentence_end.py" --arm "$arm"
    done
}

# ---- stage 15: the local-belief probes -----------------------------------------------------
# The label changes and nothing else: same tokens, same layer, same split, same hyperparameters
# as the mass-era baseline, relabelled to what the model ANSWERS if its reasoning stops there.
# P1 needs its own gather because the per-sentence-loudest cuts are new positions; P2 reuses the
# tokens the mass gather already saved.
run_local_belief_probes() {
    local lbp="$RT/local_belief_probes"
    x $UV python "$REPO/scripts/inference_oss/gather_local_belief_activations.py" \
        --rollout-dir "$RT/rollout_strategies/jlens_argmax_per_sentence" \
        --trajectory-paths "$TRAJ" --names-file "$MASS_NAMES" \
        --out "$ACT/argmax_per_sentence_l15"
    x $UV interp-cli prepare_activations_for_probing \
        --activations-dir "$ACT/argmax_per_sentence_l15" --trajectories-dir "$TRAJ" \
        --probe-type next_action --layers 15 --steps all --output-indices all \
        --output-path "$PREPARED/local_belief_p1_final"
    x $UVC python "$REPO/scripts/inference_oss/relabel_manifest_from_rollout.py" \
        "$PREPARED/local_belief_p1_final" "$RT/rollout_strategies/jlens_argmax_per_sentence" \
        "$PREPARED/local_belief_p1_local" --report-csv "$lbp/p1_relabel_report.csv"
    x $UVC python "$REPO/scripts/inference_oss/relabel_manifest_from_rollout.py" \
        "$PREPARED/next_action_mass_l15_jlens" "$RT/rollout_strategies/jlens_top_k_global" \
        "$PREPARED/local_belief_p2_local" --report-csv "$lbp/p2_relabel_report.csv"

    # Three datasets x two model types. p1_top20 is the same p1 rows thinned to the loudest 20.
    lb_train local_belief_p1       "$PREPARED/local_belief_p1_local" ""
    lb_train local_belief_p1_top20 "$PREPARED/local_belief_p1_local" "--tokens-per-trajectory 20"
    lb_train local_belief_p2       "$PREPARED/local_belief_p2_local" ""

    for p in "$lbp"/probes/*.pt; do
        [ -e "$p" ] && x $UVC python "$REPO/scripts/inference_oss/eval_local_belief.py" \
            "$p" "$PREPARED/${p##*/}_split_eval"
    done
}

# The six trainings the original driver (scripts/train_all.sh) never made it into the repo for.
# Reconstructed from the belief-baseline recipe, which is the same one: lr+mlp, hidden 1024,
# lr 3e-4, wd 1e-3, dropout 0, 50 epochs, batch 512, balanced, normalized, seed 42.
lb_train() {  # lb_train <tag> <prepared-local> <extra-split-flags>
    local tag=$1 prepared=$2 extra=$3 split="$PREPARED/${1}_split"
    x $UVC python "$REPO/scripts/split_next_action_manifest.py" "$prepared" \
        --eval-names "$MASS_EVAL_NAMES" --single-layer 15 --seed "$SEED" $extra \
        --train-out "${split}_train" --eval-out "${split}_eval"
    for mt in lr mlp; do
        x $UV interp-cli train_next_action_probe \
            --train-data-path "${split}_train" --eval-data-path "${split}_eval" \
            --output-path "$RT/local_belief_probes/probes/${tag}_${mt}.pt" \
            --model-type "$mt" --hidden-dims 1024 --learning-rate 3e-4 --weight-decay 0.001 \
            --dropout 0.0 --num-epochs 50 --batch-size 512 --class-weight balanced \
            --normalize --seed "$SEED" --device cuda --verbose
    done
}

# ---- stage 16: probe accuracy read by loudness ---------------------------------------------
# On the eval 720, where the selection still truncates the loudness axis. Two of this stage's
# claims -- the belief-vs-ending differential under the chain-length control, and the baseline's
# crossover -- do NOT survive the selection-free version in stage 19. Inside a fixed top-K arm
# loudness correlates +0.42 with chain length, so read the chain-length-quartile control.
run_probe_loudness_eval720() {
    x $UV python "$REPO/scripts/build_probe_loudness.py" --out "$RT/probe_loudness/per_token.csv"
    x $UVC python "$REPO/scripts/analyze_probe_loudness.py" \
        --per-token "$RT/probe_loudness/per_token.csv" --out "$RT/probe_loudness"
    x $UVC python "$REPO/scripts/plot_probe_loudness.py" \
        --per-token "$RT/probe_loudness/per_token.csv" --out "$RT/probe_loudness/plots"
}

# ---- stage 17: the commitment boundary, re-read ---------------------------------------------
# A CLONE: merge_probe_rollout_arms.py writes a new table and never mutates the original, and
# HEADLINE in analyze_probe_rollout.py must not be edited -- --headline-extra is the seam.
run_belief_probes_commitment() {
    local lbp="$RT/local_belief_probes/probes"
    x $UV python "$REPO/scripts/eval_probe_per_token.py" \
        --probe "$lbp/local_belief_p1_mlp.pt" --probe "$lbp/local_belief_p2_mlp.pt" \
        --activations-dir "$ACT/heldout360_l15" --lens-dir "$ACT/heldout360_lens" \
        --trajectories-dir "$TRAJ" --signal-json "$SIGNAL_JSON" \
        --layer 15 --full-probs --out "$RT/probe_vs_rollout_lb/local_belief_per_token.csv"
    x $UVC python "$REPO/scripts/merge_probe_rollout_arms.py" \
        --extra "$RT/probe_vs_rollout_lb/local_belief_per_token.csv" \
        --out "$RT/probe_vs_rollout_lb/per_token.csv"
    x $UVC python "$REPO/scripts/analyze_probe_rollout.py" \
        --per-token "$RT/probe_vs_rollout_lb/per_token.csv" --out-dir "$RT/probe_vs_rollout_lb" \
        --headline-extra local_belief.p1_mlp,local_belief.p2_mlp
}

# ---- stage 18: the dense held-out rollout ---------------------------------------------------
# No selection at all: a cutoff at every reasoning token, so a downstream join has a measured
# belief for any token it asks about and nothing sits between the loudness axis and the label.
run_heldout_every_token_rollout() {
    x env NAMES_FILE="$HELDOUT_NAMES" TRAJECTORIES="$HELDOUT_TRAJ" \
        LENS_ROOT="$ACT/heldout360_lens" OUT_ROOT="$RT/rollout_strategies_heldout360" \
        bash "$REPO/scripts/inference_oss/run_inference_strategies.sh" every_token
}

# ---- stage 19: loudness with the selection removed -----------------------------------------
# Ten probes over all 87,221 held-out tokens. Levels here are NOT comparable to stage 16's --
# different rows, different population -- only shapes are.
run_probe_loudness_heldout() {
    local out="$RT/probe_loudness_heldout360" lbp="$RT/local_belief_probes/probes"
    local args=()
    for p in "$lbp"/*.pt "$PROBES"/next_action_mass_l15/next_action_probe_{jlens,random}_topall_{lr,mlp}.pt; do
        [ -e "$p" ] && args+=(--probe "$p")
    done
    x $UV python "$REPO/scripts/eval_probe_per_token.py" "${args[@]}" \
        --activations-dir "$ACT/heldout360_l15" --lens-dir "$ACT/heldout360_lens" \
        --trajectories-dir "$TRAJ" --signal-json "$SIGNAL_JSON" \
        --layer 15 --full-probs --out "$out/heldout360_10probes.csv"
    x $UVC python "$REPO/scripts/build_probe_loudness_heldout.py" \
        --probe-csv "$out/heldout360_10probes.csv" --out "$out/per_token.csv"
    x $UVC python "$REPO/scripts/analyze_probe_loudness.py" --per-token "$out/per_token.csv" --out "$out"
    x $UVC python "$REPO/scripts/plot_probe_loudness.py" --per-token "$out/per_token.csv" --out "$out/plots"
}

# ---- stage 20: the logit-lens mass tree ----------------------------------------------------
# Pinned BY NAME to the trajectories the jlens tree drew, so the two trees differ in the lens and
# nothing else. No random arm: the control's draw already exists in the jlens tree's records and
# must be read, never re-drawn.
run_logitlens_mass_gather() {
    x env LENS=logitlens SELECT_METHODS=logitlens NAMES_FILE="$MASS_NAMES" \
        TRAJECTORIES="$TRAJ" JLENS_DIR="$JLENS_DIR" SIGNAL_JSON="$SIGNAL_JSON" \
        ACTIVATIONS_DIR="$ACT/logitlens_mass_l15" \
        bash "$REPO/scripts/jlens_mass_l15.sh"
}

# ---- stages 21-23: the belief baselines and the report --------------------------------------
run_belief_baseline_rollout_arms() { x bash "$REPO/scripts/rollout_belief_baseline_arms.sh"; }
run_belief_baseline_probes()       { x bash "$REPO/scripts/train_belief_baseline_probes.sh"; }
run_sixteen_probe_loudness_report(){ x bash "$REPO/scripts/build_sixteen_probe_loudness_report.sh"; }

# ---- stage 24 (opt-in): the grid-label round ------------------------------------------------
# NEVER RAN, and produced no result. Two unresolved problems before spending GPU here: the grid
# vocabulary on disk is the PRE-rule version (data/jlens/README.md), and
# --balance-classes-per-trajectory collapses to the rarest class, whose count depends on padding
# -- prepare then SKIPS trajectories whose cell count disagrees with the first, so mixing sizes
# can silently drop whole sizes. Settle both first.
run_grid_probing_round() {
    x env TRAJECTORIES="$TRAJ" JLENS_DIR="$JLENS_DIR" SIGNAL_JSON="$GRID_SIGNAL_JSON" \
        ACT="$ACT/grid_reasoning_tokens" \
        bash "$REPO/scripts/gather_grid_arms.sh"
    x env ARMS="jlens logitlens random" LAYERS=15 ACT="$ACT/grid_reasoning_tokens" \
        TRAJ="$TRAJ" OUT="$PREPARED/grid_l15" \
        bash "$REPO/scripts/prepare_grid_arms.sh"
    x env ARMS="jlens logitlens random" SEEDS="42 43 44" TAG=l15 \
        EVAL_NAMES="$COUNT_EVAL_NAMES" PREPARED="$PREPARED/grid_l15" PROBES="$PROBES/grid" \
        bash "$REPO/scripts/train_grid_arms.sh"
}

# ---- stage 25 (opt-in): written but never run -----------------------------------------------
# NEW WORK, not reproduction. Nothing downstream depends on these and no claim rests on them.
run_unrun_analyses() {
    x $UVC python "$REPO/scripts/analyze_truncation_strategies.py" \
        --out-root "$RT/rollout_strategies" --lens-root "$ACT/jlens_mass_l15" \
        --signal-json "$SIGNAL_JSON"
    # The sentence-end cap: does that arm lose because of per-trajectory skew, or because the
    # tokens are wrong? Capping it at 20 separates the two.
    x $UVC python "$REPO/scripts/split_next_action_manifest.py" "$PREPARED/next_action_eos_eos" \
        --eval-names "$COUNT_EVAL_NAMES" --tokens-per-trajectory 20 --single-layer 15 \
        --seed "$SEED" --train-out "$PREPARED/next_action_eos_cap20_train" \
        --eval-out "$PREPARED/next_action_eos_cap20_eval"
    for mt in lr mlp; do
        x $UV interp-cli train_next_action_probe \
            --train-data-path "$PREPARED/next_action_eos_cap20_train" \
            --eval-data-path "$PREPARED/next_action_eos_cap20_eval" \
            --output-path "$PROBES/next_action_eos/next_action_probe_eos_cap20_${mt}.pt" \
            --model-type "$mt" --hidden-dims 1024 --learning-rate 3e-4 --weight-decay 0.001 \
            --dropout 0.0 --num-epochs 50 --batch-size 512 --class-weight balanced \
            --normalize --seed "$SEED" --device cuda --verbose
    done
}

# ---------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------
selected() {
    local st=$1
    [ -n "$ONLY" ] && { case " $ONLY " in *" $st "*) ;; *) return 1;; esac; }
    [ -n "$SKIP" ] && { case " $SKIP " in *" $st "*) return 1;; esac; }
    return 0
}

if [ -n "$LIST" ]; then
    printf '%-32s %-6s %-26s %s\n' STAGE STATE COST TITLE
    printf '%-32s %-6s %-26s %s\n' "$(printf '%.32s' '--------------------------------')" ------ \
        "$(printf '%.26s' '--------------------------')" '-----'
    ndone=0; nmissing=0
    for st in $STAGES; do
        IFS='|' read -r gpu cost title <<<"$(meta "$st")"
        if "done_$st" 2>/dev/null; then state=done; ndone=$((ndone + 1))
        else state=MISSING; nmissing=$((nmissing + 1)); fi
        [ "$gpu" = "GPU" ] && cost="$cost [GPU]"
        printf '%-32s %-6s %-26s %s\n' "$st" "$state" "$cost" "$title"
    done
    echo
    echo "$ndone stage(s) already on disk, $nmissing missing."
    [ -z "$GRID_ROUND" ] && echo "GRID_ROUND=1 adds the grid-label round (it never produced a result)."
    [ -z "$UNRUN" ] && echo "UNRUN=1 adds the analyses that were written but never run (new work)."
    exit 0
fi

mkdir -p "$LOGROOT"
started_all=$(date +%s)
ran=""; skipped=""; failed=""
reached_from=${FROM:+}

echo "############################################################"
echo "# reproduce_all  --  $(ts) (UTC) / $(date +'%H:%M %Z')"
echo "#   repo   $REPO"
echo "#   root   $WS"
[ -n "$DRY_RUN" ] && echo "#   DRY_RUN -- printing commands, running nothing"
echo "############################################################"

for st in $STAGES; do
    IFS='|' read -r gpu cost title <<<"$(meta "$st")"

    if [ -n "$FROM" ] && [ -z "$reached_from" ]; then
        if [ "$st" = "$FROM" ]; then reached_from=1; else skipped="$skipped $st"; continue; fi
    fi
    selected "$st" || { skipped="$skipped $st"; continue; }
    if [ -z "$FORCE" ] && "done_$st" 2>/dev/null; then
        echo "[$(ts)] -- $st: already on disk, skipping"
        skipped="$skipped $st"
        continue
    fi

    echo ""
    echo "############################################################"
    echo "# $st  ($gpu, $cost)"
    echo "# $title"
    echo "############################################################"
    log="$LOGROOT/${st}_$(date -u +%Y%m%d_%H%M%S).txt"
    started=$(date +%s)
    ( "run_$st" ) 2>&1 | tee "$log"
    RC=${PIPESTATUS[0]}
    mins=$(( ($(date +%s) - started) / 60 ))
    if [ "$RC" -ne 0 ]; then
        echo "    *** $st FAILED (exit $RC) after ${mins}m -- see $log"
        failed="$failed $st"
    else
        echo "    done in ${mins}m -- log: $log"
        ran="$ran $st"
    fi
done

echo ""
echo "############################################################"
echo "# SUMMARY  --  $(ts) (UTC), $(( ($(date +%s) - started_all) / 60 ))m total"
echo "#   ran:     ${ran:-(none)}"
echo "#   skipped: ${skipped:-(none)}"
echo "#   failed:  ${failed:-(none)}"
echo "#"
echo "#   LIST=1 $0   for the stage table"
echo "############################################################"
[ -z "$failed" ]
