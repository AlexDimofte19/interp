#!/usr/bin/env bash
# Push the whole dataset -- every split, every activation tree, every resulting CSV -- to one
# Hugging Face dataset repo.
#
# Set HF_ORG (and, if you want a different name, HF_REPO_NAME) and run it. Nothing else is
# required; the token already on the host under HF_HOME is used and never printed.
#
#   HF_ORG=your-org DRY_RUN=1 ./scripts/push_to_huggingface.sh    # layout + real sizes, no network
#   HF_ORG=your-org ./scripts/push_to_huggingface.sh              # everything
#   HF_ORG=your-org ONLY="splits probes" ./scripts/push_to_huggingface.sh
#   HF_ORG=your-org SKIP=activations ./scripts/push_to_huggingface.sh
#
# WHY THE ACTIVATION TREES ARE TARRED. They hold millions of ~7 KB .pt files -- one per
# (token, layer) -- and pushing them as individual files would be both far past what a Hub repo
# should hold and hopeless to resume. Each tree is packed into one uncompressed tar per size
# directory, and the lens tables (analysis CSVs, direction-mass tables, selection records and
# their .meta.json sidecars) into a second tar per size, so the tables can be fetched without
# the tensors. A tar is written once and marked with a .done sidecar, so a re-run repacks
# nothing.
#
# SIZES ARE MEASURED, NOT du'd. /workspace is MooseFS and `du` reports allocated blocks, which
# for a tree of tiny files overstates the real bytes by 5-40x (one tree: 238 GiB by du, 2.3 GB
# of actual content). Everything here sizes with `find -printf '%s'`, which is what transfers.
#
# WHAT IS DELIBERATELY NOT UPLOADED:
#   * the symlink views -- they hold no bytes; the README records what each one points at
#   * $WS/shared/hf_cache and $WS/.cache -- the model weights and package caches
#   * the cached unembed ($WS/jlens/**/gpt-oss-20b_unembed.pt, 1.2 GB per copy), which any run
#     rebuilds from the model; WITH_UNEMBED_CACHE=1 includes it
#
# This writes to a public-by-default-off repo (PRIVATE=1) but it still PUBLISHES: everything
# here becomes visible to anyone the repo is shared with. Check ONLY/SKIP before the first run.
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)   # so uv finds pyproject.toml

# The destination. HF_ORG has no default on purpose -- there is no safe guess.
HF_ORG=${HF_ORG:-}
HF_REPO_NAME=${HF_REPO_NAME:-jlens_decodability_property}
HF_REPO_ID=${HF_REPO_ID:-$HF_ORG/$HF_REPO_NAME}
PRIVATE=${PRIVATE:-1}          # empty to create a public repo

# Roots.
WS=${WS:-/workspace}
ACT=${ACT:-$WS/activations}
PREPARED=${PREPARED:-$WS/prepared}
PROBES=${PROBES:-$WS/probes}
RT=${RT:-$WS/reasoning_theatre}
TRAJ_ROOT=${TRAJ_ROOT:-$WS/trajectories}

# Staging. The tarballs are built here before upload; they are as large as the trees.
STAGING=${STAGING:-$WS/hf_staging}
KEEP_STAGING=${KEEP_STAGING:-}   # 1 to keep the tarballs after a successful upload

# Sections, in upload order: small and cheap first, so an early failure is cheap to diagnose.
SECTIONS=${SECTIONS:-"readme splits probes prepared results jlens logs trajectories activations"}
ONLY=${ONLY:-}
SKIP=${SKIP:-}
DRY_RUN=${DRY_RUN:-}
WITH_UNEMBED_CACHE=${WITH_UNEMBED_CACHE:-}

# Which activation trees to pack. The two *_view entries are symlink-only and excluded.
TREES=${TREES:-"jlens_mass_l15 logitlens_mass_l15 jlens_reasoning_tokens heldout360_l15
                heldout360_lens argmax_per_sentence_l15 logitlens_argmax_per_sentence_l15
                activations_train_single_step activations_train_single_step_reasoning_eos
                activations_train_single_step_reasoning_all activations_test_full
                activations_one_spaced_trajectories_train_single_step_eos"}
SIZES=${SIZES:-"5 7 9 11 13 15"}

HF=${HF:-hf}
ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }

x() {
    printf '      + '; printf '%q ' "$@"; echo
    [ -n "$DRY_RUN" ] && return 0
    "$@"
}

# Real bytes under a path, optionally restricted to a name pattern. Never `du`.
bytes() {
    local root=$1 pat=${2:-}
    [ -d "$root" ] || [ -f "$root" ] || { echo 0; return; }
    if [ -n "$pat" ]; then
        find "$root" -type f -name "$pat" -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {print s+0}'
    else
        find "$root" -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {print s+0}'
    fi
}
human() { awk -v b="$1" 'BEGIN {u="B KB MB GB TB"; split(u,a," "); i=1;
    while (b>=1024 && i<5) {b/=1024; i++} printf "%.1f %s", b, a[i]}'; }

selected() {
    local s=$1
    [ -n "$ONLY" ] && { case " $ONLY " in *" $s "*) ;; *) return 1;; esac; }
    [ -n "$SKIP" ] && { case " $SKIP " in *" $s "*) return 1;; esac; }
    return 0
}

# ---- preflight ---------------------------------------------------------------------------
if [ -z "$HF_ORG" ] && [ -z "${HF_REPO_ID##/*}" ]; then
    echo "!! set HF_ORG to the account or organisation to push to, e.g." >&2
    echo "!!   HF_ORG=project-telos $0" >&2
    echo "!! (or set HF_REPO_ID=<org>/<name> directly)" >&2
    exit 1
fi
command -v "$HF" >/dev/null || { echo "!! the 'hf' CLI is not on PATH -- see runpod_setup.sh" >&2; exit 1; }

echo "############################################################"
echo "# push_to_huggingface  --  $(ts) (UTC) / $(date +'%H:%M %Z')"
echo "#   repo     $HF_REPO_ID  ($([ -n "$PRIVATE" ] && echo private || echo PUBLIC))"
echo "#   staging  $STAGING"
[ -n "$DRY_RUN" ] && echo "#   DRY_RUN -- sizing and printing only, no upload"
echo "############################################################"

if [ -z "$DRY_RUN" ]; then
    "$HF" auth whoami >/dev/null 2>&1 || { echo "!! not logged in: run 'hf auth login'" >&2; exit 1; }
    x "$HF" repo create "$HF_REPO_ID" --repo-type dataset ${PRIVATE:+--private} || true
fi

[ -n "$DRY_RUN" ] || mkdir -p "$STAGING"
uploaded=""; failed=""; skipped=""

# stage_dir <section> -> path. Created only for a real run: a DRY_RUN must leave no trace, and
# the copies that would fill it are themselves skipped by x().
sdir() { local d="$STAGING/$1"; [ -n "$DRY_RUN" ] || mkdir -p "$d"; echo "$d"; }

# upload <section>. Small sections go through `hf upload`; the big ones through
# upload-large-folder, which keeps its own resume state inside the folder.
upload() {
    local section=$1 src=$2 large=${3:-}
    [ -n "$DRY_RUN" ] && { echo "      (would upload $src -> $HF_REPO_ID:$section/)"; return 0; }
    if [ -n "$large" ]; then
        x "$HF" upload-large-folder "$HF_REPO_ID" "$src" --repo-type dataset
    else
        x "$HF" upload "$HF_REPO_ID" "$src" "$section" --repo-type dataset
    fi
}

# ---- readme ------------------------------------------------------------------------------
# The card is generated, not hand-written, so it cannot drift from what was actually pushed.
do_readme() {
    local d; d=$(sdir readme)
    {
        echo "---"
        echo "license: other"
        echo "tags: [interpretability, mechanistic-interpretability, gpt-oss, probing, jacobian-lens]"
        echo "---"
        echo
        echo "# jlens decodability property"
        echo
        echo "Artifacts from an interpretability study of gpt-oss-20b acting as an agent in 2D grid"
        echo "worlds: residual-stream activations, the lens tables used to choose which tokens to"
        echo "probe, the prepared probe datasets and their splits, the trained probes, the rollouts,"
        echo "and every resulting table and figure."
        echo
        echo "Generated by \`scripts/push_to_huggingface.sh\` on $(ts)."
        echo
        echo "## Layout"
        echo
        echo '| path | what |'
        echo '|---|---|'
        echo '| `splits/` | the name lists that define every trajectory set -- **the most important files here** |'
        echo '| `trajectories/` | the trajectory JSONs, the canonical data format |'
        echo '| `prepared/` | probe datasets: `manifest.json` (format_version 3) plus, for some, one packed tensor |'
        echo '| `activations/<tree>/size{N}_activations.tar` | the per-token `.pt` residual streams |'
        echo '| `activations/<tree>/size{N}_lens_tables.tar` | analysis CSVs, direction-mass tables, selection records, `.meta.json` |'
        echo '| `probes/` | trained probe weights and their training logs |'
        echo '| `results/` | rollouts, per-token CSVs, tables, figures and the HTML reports |'
        echo '| `jlens/` | the fitted Jacobian lens and the signal vocabularies |'
        echo '| `logs/` | run logs |'
        echo
        echo "## Reading this data"
        echo
        echo "**Trajectory sets are pinned by name, never by seed.** Re-running a sampler with the"
        echo "same flags does *not* reproduce a draw -- two such runs overlapped by 348 of 3600. The"
        echo "files in \`splits/\` are therefore inputs, not outputs: they are the only record of which"
        echo "trajectories a given result was measured on."
        echo
        echo "**Prepared manifests carry an absolute \`activations_root\`** pointing at \`/workspace/...\`"
        echo "on the machine that built them. After unpacking, rewrite it:"
        echo
        echo '```bash'
        echo "python - <<'PY'"
        echo "import json, pathlib"
        echo "NEW = '/your/path/activations'"
        echo "for m in pathlib.Path('prepared').glob('*/manifest.json'):"
        echo "    d = json.loads(m.read_text())"
        echo "    if 'activations_root' in d:"
        echo "        d['activations_root'] = d['activations_root'].replace('/workspace/activations', NEW)"
        echo "        m.write_text(json.dumps(d))"
        echo "PY"
        echo '```'
        echo
        echo "**Never read a direction-mass table without its \`.meta.json\` sidecar.** Two different"
        echo "vocabularies point at identically-shaped trees; the sidecar names the one that produced"
        echo "the table."
        echo
        echo "**Read the lens CSVs with \`csv.DictReader\`, not \`pandas.read_csv\`.** Decoded tokens"
        echo "include the literal string \`NA\`, empty strings, embedded commas and newlines, all of"
        echo "which pandas' NA handling silently corrupts."
        echo
        echo "## Symlink views, not uploaded"
        echo
        echo "Three directories on the source host are symlink-only -- they hold no bytes and are"
        echo "reconstructable from what is here:"
        echo
        echo '| view | points at |'
        echo '|---|---|'
        echo '| `eos_lens3600_view` | `activations_train_single_step_reasoning_eos`, restricted to the 3600 names in `splits/lens_trajectories_3600.txt` |'
        echo '| `jlens_reasoning_tokens_comp0.0-0.2-0.4` | `jlens_reasoning_tokens`, restricted to complexities 0.0/0.2/0.4 |'
        echo '| `trajectories/heldout360` | the 360 files named by `splits/heldout360_names.txt`, inside the training set |'
        echo
        echo "## Contents as pushed"
        echo
        echo '| section | real bytes |'
        echo '|---|---|'
        for s in splits trajectories prepared probes results jlens logs; do
            case $s in
            splits)       b=$(( $(bytes "$WS/splits") + $(bytes "$TRAJ_ROOT/heldout360_names.txt") ));;
            trajectories) b=$(bytes "$TRAJ_ROOT");;
            prepared)     b=$(bytes "$PREPARED");;
            probes)       b=$(bytes "$PROBES");;
            results)      b=$(bytes "$RT");;
            jlens)        b=$(bytes "$WS/jlens");;
            logs)         b=$(bytes "$WS/logs");;
            esac
            echo "| \`$s/\` | $(human "$b") |"
        done
        local tot_pt=0 tot_csv=0
        for t in $TREES; do
            [ -d "$ACT/$t" ] || continue
            tot_pt=$(( tot_pt + $(bytes "$ACT/$t" '*.pt') ))
            tot_csv=$(( tot_csv + $(bytes "$ACT/$t" '*.csv') ))
        done
        echo "| \`activations/\` tensors | $(human "$tot_pt") |"
        echo "| \`activations/\` lens tables | $(human "$tot_csv") |"
    } > "$d/README.md"
    echo "      wrote $d/README.md"
    upload readme "$d/README.md"
}

# ---- splits ------------------------------------------------------------------------------
do_splits() {
    local d; d=$(sdir splits) n=0 b=0
    for f in "$WS/splits/lens_trajectories_3600.txt" "$WS/splits/eval_trajectories_720.txt" \
             "$TRAJ_ROOT/heldout360_names.txt" \
             "$PREPARED/next_action_mass_l15_eval_names.txt" \
             "$RT/rollout_strategies/mass_l15_names.txt"; do
        [ -f "$f" ] || continue
        n=$((n + 1)); b=$((b + $(stat -c %s "$f")))
        x cp -f "$f" "$d/"
    done
    echo "      $n name list(s), $(human "$b")"
    upload splits "$d"
}

do_probes()       { upload probes "$PROBES" large; }
do_results()      { upload results "$RT" large; }
do_trajectories() { upload trajectories "$TRAJ_ROOT" large; }
do_logs()         { upload logs "$WS/logs"; }

# ---- prepared ----------------------------------------------------------------------------
# Every prepared dataset, plus the one that lives inside an activation tree rather than under
# prepared/: the merged grid-tile set, whose manifest is keyed `trajectories` and whose
# activations are local rather than referenced. A rule of "skip everything under activations/"
# would silently drop it.
do_prepared() {
    local merged="$ACT/activations_train_single_step/cognitive_map_activations_l15_s0_suffix_all_grid_tile_pad15_merged"
    upload prepared "$PREPARED" large
    if [ -d "$merged" ]; then
        echo "      + the merged grid-tile dataset (manifest keyed 'trajectories', local activations)"
        [ -n "$DRY_RUN" ] || x "$HF" upload "$HF_REPO_ID" "$merged" \
            "prepared/cognitive_map_grid_tile_l15_merged" --repo-type dataset
    fi
}

# ---- jlens -------------------------------------------------------------------------------
do_jlens() {
    local d; d=$(sdir jlens) b=0
    local files="$WS/jlens/direction_tokens_full.json $WS/jlens/grid_tokens_full.json
                 $WS/jlens/gridenv/gpt-oss-20b_jacobian_lens.pt $WS/jlens/gridenv/config.yaml
                 $WS/jlens/gridenv/convergence.csv $WS/jlens/gpt-oss-20b_convergence.csv"
    [ -n "$WITH_UNEMBED_CACHE" ] && files="$files $WS/jlens/gridenv/gpt-oss-20b_unembed.pt"
    for f in $files; do
        [ -f "$f" ] || continue
        b=$((b + $(stat -c %s "$f")))
        x cp -f "$f" "$d/"
    done
    echo "      $(human "$b") (unembed cache $([ -n "$WITH_UNEMBED_CACHE" ] && echo included || echo "skipped -- rebuilt from the model"))"
    upload jlens "$d" large
}

# ---- activations -------------------------------------------------------------------------
# Two tars per (tree, size): the tensors, and the lens tables. Each is skipped when its .done
# sidecar is there, so a resumed run repacks nothing.
pack() {   # pack <tree> <size> <suffix> <find-args...>
    local tree=$1 size=$2 suffix=$3; shift 3
    local src="$ACT/$tree/size$size"
    [ -d "$src" ] || return 0
    local out="$STAGING/activations/$tree/size${size}_${suffix}.tar"
    if [ -f "$out.done" ]; then echo "      (size$size $suffix already packed)"; return 0; fi
    local n; n=$(find "$src" -type f "$@" -printf '.' 2>/dev/null | wc -c)
    [ "$n" -eq 0 ] && return 0
    local b; b=$(find "$src" -type f "$@" -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {print s+0}')
    echo "      size$size $suffix: $n file(s), $(human "$b")"
    [ -n "$DRY_RUN" ] && return 0
    mkdir -p "$(dirname "$out")"
    ( cd "$ACT/$tree" && find "size$size" -type f "$@" -print0 \
        | tar --null -cf "$out.part" --files-from=- ) || return 1
    mv "$out.part" "$out" && : > "$out.done"
}

do_activations() {
    for tree in $TREES; do
        [ -d "$ACT/$tree" ] || { echo "   -- $tree: not present, skipping"; continue; }
        echo "   == $tree"
        for size in $SIZES; do
            pack "$tree" "$size" activations -name '*.pt'
            pack "$tree" "$size" lens_tables \( -name '*.csv' -o -name '*.json' \)
        done
        # A tree-level manifest, where one exists (the older flat layouts have one).
        [ -f "$ACT/$tree/manifest.json" ] && [ -z "$DRY_RUN" ] && \
            x cp -n "$ACT/$tree/manifest.json" "$STAGING/activations/$tree/" 2>/dev/null
        if [ -d "$STAGING/activations/$tree" ]; then
            upload "activations/$tree" "$STAGING/activations/$tree" large || return 1
            [ -z "$KEEP_STAGING" ] && [ -z "$DRY_RUN" ] && x rm -rf "$STAGING/activations/$tree"
        fi
    done
}

# ---- driver ------------------------------------------------------------------------------
started_all=$(date +%s)
for s in $SECTIONS; do
    selected "$s" || { skipped="$skipped $s"; continue; }
    echo ""
    echo "=== $s"
    started=$(date +%s)
    "do_$s"
    RC=$?
    mins=$(( ($(date +%s) - started) / 60 ))
    if [ "$RC" -ne 0 ]; then
        echo "    *** $s FAILED (exit $RC) after ${mins}m"
        failed="$failed $s"
    else
        uploaded="$uploaded $s"
    fi
done

echo ""
echo "############################################################"
echo "# SUMMARY  --  $(ts) (UTC), $(( ($(date +%s) - started_all) / 60 ))m"
echo "#   repo:     https://huggingface.co/datasets/$HF_REPO_ID"
echo "#   uploaded: ${uploaded:-(none)}"
echo "#   skipped:  ${skipped:-(none)}"
echo "#   failed:   ${failed:-(none)}"
[ -n "$KEEP_STAGING" ] && echo "#   staging kept at $STAGING"
echo "############################################################"
[ -z "$failed" ]
