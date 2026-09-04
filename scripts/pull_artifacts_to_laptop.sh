#!/usr/bin/env bash
# Pull everything worth having off the GPU host onto a laptop -- EXCEPT the per-token .pt
# activation trees, which are millions of tiny files and would swallow the disk.
#
# Run this ON THE LAPTOP, not on the GPU host.
#
#   HOST=mypod DRY_RUN=1 ./scripts/pull_artifacts_to_laptop.sh    # what it would fetch, and how big
#   HOST=mypod ./scripts/pull_artifacts_to_laptop.sh              # ~15 GB with the defaults
#   HOST=mypod WITH_TRAJECTORIES= ./scripts/pull_artifacts_to_laptop.sh          # ~9 GB
#   HOST=mypod WITH_ANALYSIS_CSV=1 ./scripts/pull_artifacts_to_laptop.sh         # +20 GB
#   HOST=mypod DEST=/Volumes/ext/telos ./scripts/pull_artifacts_to_laptop.sh
#
# HOST is whatever `ssh <HOST>` reaches -- an ~/.ssh/config alias, or user@address. Set
# HOST=local to copy from a local /workspace instead, which is how this is tested on the host
# itself.
#
# WHAT IS EXCLUDED, AND WHY
#   * `*.pt` under activations/    the per-token residual streams: ~39 GB in millions of ~7 KB
#                                  files. Everything downstream of them -- probes, per-token
#                                  CSVs, tables, figures -- is here instead, so nothing you can
#                                  read on a laptop needs them.
#   * `*_analysis.csv`             the top-20 lens predictions per (token, layer): ~20 GB. The
#                                  direction-mass tables and selection records, which are what
#                                  the selection and the loudness axis actually read, are
#                                  always pulled (~1 GB). WITH_ANALYSIS_CSV=1 adds the rest.
#   * the model and package caches, the venv, the worktrees, and .ssh
#   * .ipynb_checkpoints           exact duplicates of several 100+ MB CSVs and reports
#
# SIZES COME FROM RSYNC, NOT du. The host is MooseFS, where `du` reports allocated blocks and
# overstates a small-file tree by 5-40x. DRY_RUN=1 runs `rsync --dry-run --stats`, which reports
# what would actually transfer.
#
# ONE THING TO FIX AFTER: every prepared manifest hardcodes an absolute `activations_root` on
# the host. $DEST/README_PATHS.md is written with the one-liner that rewrites them.
set -uo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# Source and destination.
HOST=${HOST:-}
REMOTE_ROOT=${REMOTE_ROOT:-/workspace}
DEST=${DEST:-$HOME/telos-interp-artifacts}

# The repo itself is cloned rather than rsynced -- it is version control's job, and the clone
# brings the plotting and analysis scripts that read everything below.
REPO_URL=${REPO_URL:-git@github.com:AlexDimofte19/interp.git}
BRANCH=${BRANCH:-worktree-probe-loudness}
WITH_REPO=${WITH_REPO:-1}

# Tiers. The defaults land at roughly 15 GB.
WITH_PACKED=${WITH_PACKED:-1}              # packed prepared tensors, ~3.2 GB
WITH_TRAJECTORIES=${WITH_TRAJECTORIES:-1}  # the training trajectory JSONs, ~6.0 GB
WITH_TEST_TRAJECTORIES=${WITH_TEST_TRAJECTORIES:-1}   # ~0.6 GB
WITH_PROBED_TRAJECTORIES=${WITH_PROBED_TRAJECTORIES:-} # trajectories with probe outputs, ~3.5 GB
WITH_LENS=${WITH_LENS:-1}                  # the fitted Jacobian lens, ~382 MB
WITH_ANALYSIS_CSV=${WITH_ANALYSIS_CSV:-}   # per-token lens predictions, ~20 GB
WITH_EVALS=${WITH_EVALS:-}                 # the older probe eval dumps, ~1.8 GB
WITH_LOGS=${WITH_LOGS:-1}                  # run logs, ~18 MB

SECTIONS=${SECTIONS:-"splits probes prepared results activations jlens logs trajectories repo"}
ONLY=${ONLY:-}
SKIP=${SKIP:-}
DRY_RUN=${DRY_RUN:-}

RSYNC=${RSYNC:-rsync}
# -l keeps symlinks AS LINKS. Never -L here: the view directories are symlink farms pointing
# into the activation trees, and dereferencing them would drag in the very .pt files this
# script exists to avoid.
RSYNC_OPTS=${RSYNC_OPTS:--a -l --partial --human-readable}
ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }

command -v "$RSYNC" >/dev/null || { echo "!! rsync not found" >&2; exit 1; }
if [ -z "$HOST" ]; then
    echo "!! set HOST to the ssh alias for the GPU box, e.g." >&2
    echo "!!   HOST=mypod $0            (HOST=local copies from a local $REMOTE_ROOT)" >&2
    exit 1
fi

# src <path> -> "host:/path" or a local path
src() { if [ "$HOST" = "local" ]; then echo "$REMOTE_ROOT/$1"; else echo "$HOST:$REMOTE_ROOT/$1"; fi; }

selected() {
    local s=$1
    [ -n "$ONLY" ] && { case " $ONLY " in *" $s "*) ;; *) return 1;; esac; }
    [ -n "$SKIP" ] && { case " $SKIP " in *" $s "*) return 1;; esac; }
    return 0
}

# pull <label> <remote-subpath> <local-subdir> [extra rsync args...]
pull() {
    local label=$1 remote=$2 local_sub=$3; shift 3
    echo "   -- $label"
    mkdir -p "$DEST/$local_sub"
    local args=($RSYNC_OPTS "$@")
    [ -n "$DRY_RUN" ] && args+=(--dry-run --stats)
    [ -z "$DRY_RUN" ] && args+=(--info=progress2)
    printf '      + %s ' "$RSYNC"; printf '%q ' "${args[@]}" "$(src "$remote")" "$DEST/$local_sub/"; echo
    "$RSYNC" "${args[@]}" "$(src "$remote")" "$DEST/$local_sub/"
}

echo "############################################################"
echo "# pull_artifacts_to_laptop  --  $(ts) (UTC) / $(date +'%H:%M %Z')"
echo "#   from  ${HOST}:${REMOTE_ROOT}"
echo "#   into  $DEST"
[ -n "$DRY_RUN" ] && echo "#   DRY_RUN -- rsync --dry-run --stats, nothing written"
echo "############################################################"
mkdir -p "$DEST"
started_all=$(date +%s)
pulled=""; failed=""; skipped=""

# ---- splits: the name lists. Small, and nothing else can be interpreted without them. ------
do_splits() {
    pull "name lists" "splits/" splits
    pull "held-out names" "trajectories/heldout360_names.txt" splits
    pull "mass-era eval names" "prepared/next_action_mass_l15_eval_names.txt" splits
    pull "mass-era trajectory names" "reasoning_theatre/rollout_strategies/mass_l15_names.txt" splits
}

# ---- probes: the weights and their training logs -------------------------------------------
do_probes() {
    pull "probe weights and logs" "probes/" probes
    pull "local-belief probes" "reasoning_theatre/local_belief_probes/probes/" probes/local_belief_probes
}

# ---- prepared: manifests always; the packed tensors by tier --------------------------------
# Nothing here copies a per-token .pt: token-major manifests reference the tree in place, and
# the only tensors are the eight single packed files.
do_prepared() {
    local ex=()
    [ -z "$WITH_PACKED" ] && ex+=(--exclude='*_packed_activations.pt')
    pull "prepared datasets" "prepared/" prepared "${ex[@]}" --exclude='.ipynb_checkpoints/'
}

# ---- results: every table, figure, report and rollout --------------------------------------
do_results() {
    pull "results" "reasoning_theatre/" results \
        --exclude='.ipynb_checkpoints/' --exclude='local_belief_probes/probes/'
}

# ---- activations: the small high-value files only ------------------------------------------
# The exclude has to come BEFORE the includes for the .pt rule to win, and the '*/' include is
# what lets rsync descend into the size and trajectory directories at all.
do_activations() {
    local rules=(
        --exclude='*.pt'
        --exclude='.ipynb_checkpoints/'
        --include='*/'
        --include='*_direction_mass.csv'
        --include='*_selection.json'
        --include='*.meta.json'
        --include='manifest.json'
    )
    [ -n "$WITH_ANALYSIS_CSV" ] && rules+=(--include='*_analysis.csv')
    rules+=(--exclude='*')
    pull "lens tables and selection records" "activations/" activations "${rules[@]}"

    # The one prepared dataset that lives inside an activation tree: its manifest is keyed
    # `trajectories`, its activations are local rather than referenced, and it is one .pt per
    # trajectory (36,000 x ~20 KB), not one per token. A blanket "no .pt under activations/"
    # would silently drop the whole cognitive-map training set.
    pull "merged grid-tile dataset" \
        "activations/activations_train_single_step/cognitive_map_activations_l15_s0_suffix_all_grid_tile_pad15_merged/" \
        prepared/cognitive_map_grid_tile_l15_merged
}

# ---- jlens: the vocabularies always, the fitted lens by tier -------------------------------
# The unembed caches (1.2 GB each) and the 3.2 GB combined analysis CSV are never pulled: the
# first is rebuilt from the model on first use, the second is a laptop-hostile intermediate.
do_jlens() {
    local rules=(--include='*/' --include='*.json' --include='config.yaml' --include='convergence.csv'
                 --include='gpt-oss-20b_convergence.csv')
    [ -n "$WITH_LENS" ] && rules+=(--include='gpt-oss-20b_jacobian_lens.pt'
                                   --include='gpt-oss-20b_gridenv_jacobian_lens.pt')
    rules+=(--exclude='gpt-oss-20b_unembed.pt' --exclude='ckpt.pt'
            --exclude='.ipynb_checkpoints/' --exclude='*')
    pull "lens and vocabularies" "jlens/" jlens "${rules[@]}"
}

do_logs() { [ -n "$WITH_LOGS" ] && pull "run logs" "logs/" logs; return 0; }

# ---- trajectories: the canonical data format ------------------------------------------------
# heldout360 is a symlink farm into the training set, so -l keeps it as links and the 360 real
# files arrive with the training set itself.
do_trajectories() {
    [ -n "$WITH_TRAJECTORIES" ] && pull "training trajectories" \
        "trajectories/reveng/trajectories_train_single_step/" trajectories/trajectories_train_single_step
    [ -n "$WITH_TEST_TRAJECTORIES" ] && pull "test trajectories" \
        "trajectories/trajectories_test_full/" trajectories/trajectories_test_full
    [ -n "$WITH_PROBED_TRAJECTORIES" ] && pull "trajectories with probe outputs" \
        "trajectories/trajectories_test_full_with_probes/" trajectories/trajectories_test_full_with_probes
    [ -n "$WITH_EVALS" ] && pull "older probe evals" "evals/" evals
    return 0
}

# ---- the repo: cloned, not copied -----------------------------------------------------------
do_repo() {
    [ -z "$WITH_REPO" ] && return 0
    echo "   -- repo ($BRANCH)"
    if [ -d "$DEST/repo/.git" ]; then
        echo "      + git -C $DEST/repo pull --ff-only"
        [ -n "$DRY_RUN" ] || git -C "$DEST/repo" pull --ff-only
    else
        echo "      + git clone --branch $BRANCH $REPO_URL $DEST/repo"
        [ -n "$DRY_RUN" ] || git clone --branch "$BRANCH" "$REPO_URL" "$DEST/repo"
    fi
}

for s in $SECTIONS; do
    selected "$s" || { skipped="$skipped $s"; continue; }
    echo ""
    echo "=== $s"
    "do_$s"
    RC=$?
    if [ "$RC" -ne 0 ]; then
        echo "    *** $s FAILED (exit $RC)"
        failed="$failed $s"
    else
        pulled="$pulled $s"
    fi
done

# ---- the note that makes the copy usable ----------------------------------------------------
write_readme() {
    cat > "$DEST/README_PATHS.md" <<EOF
# Telos interp artifacts -- local copy

Pulled from \`${HOST}:${REMOTE_ROOT}\` on $(ts) by \`scripts/pull_artifacts_to_laptop.sh\`.

## Fix this first: the manifests point at the GPU host

Every prepared manifest carries an absolute \`activations_root\` under \`/workspace/activations\`.
Nothing that loads one will work until they are rewritten:

\`\`\`bash
python - <<'PY'
import json, pathlib
NEW = "$DEST/activations"
for m in pathlib.Path("$DEST/prepared").glob("*/manifest.json"):
    d = json.loads(m.read_text())
    if "activations_root" in d:
        d["activations_root"] = d["activations_root"].replace("/workspace/activations", NEW)
        m.write_text(json.dumps(d))
        print("rewrote", m)
PY
\`\`\`

Note that the token-major manifests reference **per-token .pt files that were deliberately not
pulled**. They are still useful to read -- sample counts, selections, labels, splits -- but a
probe cannot be retrained from them here. Train on the GPU host.

## What is here

| directory | contents |
|---|---|
| \`splits/\` | the name lists defining every trajectory set -- the most important files in this copy |
| \`probes/\` | trained probe weights and training logs |
| \`prepared/\` | probe dataset manifests$([ -n "$WITH_PACKED" ] && echo ", plus the packed tensors") |
| \`results/\` | rollouts, per-token CSVs, tables, figures, HTML reports |
| \`activations/\` | direction-mass tables, selection records, \`.meta.json\` sidecars$([ -n "$WITH_ANALYSIS_CSV" ] && echo ", analysis CSVs") |
| \`jlens/\` | the signal vocabularies$([ -n "$WITH_LENS" ] && echo " and the fitted Jacobian lens") |
| \`trajectories/\` | trajectory JSONs |
| \`repo/\` | the code, on branch \`$BRANCH\` -- the plotting and analysis scripts live here |

## What was NOT pulled

* **the per-token \`.pt\` activation trees** -- ~39 GB across millions of ~7 KB files. Every
  artifact derived from them is here instead.
$([ -z "$WITH_ANALYSIS_CSV" ] && echo "* \`*_analysis.csv\` (~20 GB) -- re-run with WITH_ANALYSIS_CSV=1 if you need the raw top-20 lens predictions.")
$([ -z "$WITH_TRAJECTORIES" ] && echo "* the training trajectory JSONs -- WITH_TRAJECTORIES=1.")
$([ -z "$WITH_PROBED_TRAJECTORIES" ] && echo "* trajectories with probe outputs -- WITH_PROBED_TRAJECTORIES=1.")
$([ -z "$WITH_EVALS" ] && echo "* the older probe eval dumps -- WITH_EVALS=1.")
* the cached unembed (rebuilt from the model), the model and package caches, and the venv.

## Reading the data

* **Never read a direction-mass table without its \`.meta.json\` sidecar** -- two vocabularies
  point at identically-shaped trees, and the sidecar names the one that produced the table.
* **Read lens CSVs with \`csv.DictReader\`, never \`pandas.read_csv\`** -- decoded tokens include
  the literal string \`NA\`, empty strings, embedded commas and newlines.
* **\`abs_pos\` is prompt-inclusive; \`token_idx\` indexes \`output_tokens\`.** Joining on the wrong
  one gives an empty result, never an error.
* Symlink directories (\`heldout360\`, the \`*_view\` trees) were copied as links, so they point at
  paths that do not exist here. The names files in \`splits/\` are the portable form.

Re-run: \`HOST=$HOST DEST=$DEST ./scripts/pull_artifacts_to_laptop.sh\`
EOF
    echo "   wrote $DEST/README_PATHS.md"
}
[ -n "$DRY_RUN" ] || write_readme

echo ""
echo "############################################################"
echo "# SUMMARY  --  $(ts) (UTC), $(( ($(date +%s) - started_all) / 60 ))m"
echo "#   into:    $DEST"
[ -z "$DRY_RUN" ] && echo "#   size:    $(du -sh "$DEST" 2>/dev/null | cut -f1) on this disk"
echo "#   pulled:  ${pulled:-(none)}"
echo "#   skipped: ${skipped:-(none)}"
echo "#   failed:  ${failed:-(none)}"
echo "#"
echo "#   read $DEST/README_PATHS.md first -- the manifests need one rewrite."
echo "############################################################"
[ -z "$failed" ]
