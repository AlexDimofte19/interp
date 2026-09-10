# Script renames, 2026-09-04

Six scripts were named after the ICLR log entry that produced them (`entry49_*`). A log number
says when something was written, not what it does, and it stops meaning anything once the log
moves on. They now carry descriptive names and titles. The log entry survives as a
parenthetical cross-reference inside each header, never as the name.

## The renames

| old | new | what it does |
|---|---|---|
| `scripts/entry49_baseline_arms.sh` | `wrappers/rollout_belief_baseline_arms.sh` | The belief-baseline rollout arms: random replay, and the two logit-lens loud arms |
| `scripts/entry49_baseline_probes.sh` | `scripts/train_belief_baseline_probes.sh` | The belief-baseline probes: label and selection separated |
| `scripts/entry49_baseline_report.sh` | `scripts/build_sixteen_probe_loudness_report.sh` | Sixteen probes on the held-out 360, under both loudness rulers |
| `scripts/entry49_status.sh` | `scripts/belief_baselines_status.sh` | Status of the belief-baseline round |
| `scripts/entry49_build_report.py` | `scripts/build_sixteen_probe_report_page.py` | Builds that report's HTML page from the probe/figure registries |
| `scripts/entry49_direction_word_analysis.py` | `scripts/analyze_direction_word_isolation.py` | Isolating the direction words under both loudness rulers |

Done with `git mv`, so `git log --follow` still reaches the whole history of each file.

## What did NOT change: the data paths

Three on-disk locations keep their original `entry49` spelling, because renaming them would
strand the run already on disk -- every resumability marker, every relabel report and every
provenance caption in the sixteen-probe page points at them:

```
/workspace/prepared/entry49_*                     the six belief-baseline datasets and splits
/workspace/reasoning_theatre/entry49_baselines    relabel reports and per-step logs
/workspace/logs/entry49                           the round's driver logs
```

They are no longer written as literals. Each is reached through a named variable, so a fresh
host can point them anywhere and the old spelling is a default, not a fact:

| variable | default |
|---|---|
| `BELIEF_BASELINE_PREPARED_PREFIX` | `/workspace/prepared/entry49` |
| `BELIEF_BASELINE_ROOT` | `/workspace/reasoning_theatre/entry49_baselines` |
| `BELIEF_BASELINE_LOGS` | `/workspace/logs/entry49` |

Rename the directories later by setting these; nothing else needs to change.

## Also folded in

`build_sixteen_probe_loudness_report.sh` previously stopped after the jlens-loudness figures,
and the second ruler was run by hand (the stray `analyze_logitlens_loudness.log` and
`plot_logitlens_loudness.log` in that output directory are the evidence). The two-ruler pass,
the direction-word isolation and the page build are now steps 6a'-8 of the script, so the
report rebuilds in one command.

## Not renamed

`ICLR log.txt` is append-only history and keeps every entry number, including in text that
names the old scripts. That is correct: it records what was true when it was written.

---

# Script rename, 2026-09-10

| old | new | why |
|---|---|---|
| `scripts/build_eos_view.sh` | `scripts/build_activation_view.sh` | It was never specific to the sentence-end tree -- `SRC` was always an env var. The mass-era train/eval split cuts two views out of `jlens_mass_l15` with the same script, and a name saying `eos` would have been wrong for both. |

Done with `git mv`; `SRC` still defaults to the sentence-end tree, so every existing call site
keeps its behaviour and only the path changed. The two callers
(`reproduce_all.sh`, `train_more_belief_arms.sh`) were updated. The `build_eos_view` *function*
inside `reproduce_all.sh` keeps its name -- it builds the eos view specifically, which is what
it says.

---

# Reorganisation, 2026-09-10

`scripts/` had grown to 86 files with no separation between *code that does work* and *a record
of what was once run*. Three directories now, and the rule is what a file **is**, not what it
touches:

| Directory | What belongs there |
|---|---|
| `scripts/` | machinery — anything with logic worth testing or changing |
| `wrappers/` | recorded invocations: an env block pinning one run's parameters onto one script. Never refactor one; changing a default rewrites the description of a run that already happened. See [wrappers/README.md](wrappers/README.md). |
| `grid_cell_analysis/` | the grid-cell ("cognitive map") line, both rounds. See [grid_cell_analysis/README.md](grid_cell_analysis/README.md). |

## Moved to `wrappers/` (16)

From `scripts/`: `jlens_reasoning_tokens.sh`, `jlens_reasoning_tokens_filtered.sh`,
`jlens_mass_l15.sh`, `jlens_extend_logitlens.sh`, `delete_non_jlens_selected.sh`,
`rollout_belief_baseline_arms.sh`, `rollout_more_belief_arms.sh`, `run_next_action_arms.sh`,
`train_next_action_direction_probe.sh`, `prepare_next_action_jlens_by_complexity.sh`,
`build_convinced_datasets.sh`. From `scripts/inference_oss/`: `run_analysis.sh`. From the repo
root: `script.sh`, `general_probe_train.sh`, `reasoning_theatre.ps1`, `run_commands.ps1` — the
four CLAUDE.md already called recorded invocations. Only `runpod_setup.sh` is left loose.

Each still resolves the repo from its own location, so `REPO=.../..` keeps working from
`wrappers/` exactly as it did from `scripts/`. Three needed a real fix, because they had been
finding siblings by directory rather than by repo:

* `rollout_belief_baseline_arms.sh`, `rollout_more_belief_arms.sh` — `$HERE/inference_oss/...`
  became `$REPO/scripts/inference_oss/...`.
* `run_analysis.sh` ran `python analysis.py` bare, which only worked from the one directory it
  used to live in. It now resolves the script from `$REPO`.
* `build_convinced_datasets.sh` used a relative `.venv/bin/python` and relative script paths.

## Moved to `grid_cell_analysis/` (20)

Round 2 (never produced a result): `gather_grid_arms.sh`, `prepare_grid_arms.sh`,
`train_grid_arms.sh`, `grid_round_status.sh`, `eval_grid_probe_per_token.py`,
`analyze_grid_loudness_correlation.py`.

Round 1 (published): `evaluation_scripts/compute_probe_accuracy.py` and all of
`plotting_scripts/` land under `grid_cell_analysis/plots/`, together with the four
`scripts/plot_*_by_distance*.py` — every one of them consumes
`eval_cognitive_map_probe_per_distance`, which is the cognitive-map evaluator. The
`evaluation_scripts/` and `plotting_scripts/` directories are gone.

## Merged

| Was | Now |
|---|---|
| `eval_more_belief_arms.sh` + `eval_equal_n_belief_arms.sh` (83% identical) | `scripts/eval_belief_arms_heldout.sh <24probes\|equal_n>` |
| `jlens_action_ranks.py` + `jlens_action_ranks_sampled.py` (73% identical) | `scripts/jlens_action_ranks.py`, with `--runs_per_combo` / `--trajectories_root` optional |

Both merges were checked against the originals before the originals were deleted: the belief
script's two `--extra-probes` key maps are byte-identical to what each old file built, and the
action-ranks CSV keeps its narrow schema unless `--trajectories_root` is passed. The importers of
`jlens_action_ranks_sampled` (`jlens_reasoning_tokens.py`, `tests/conftest.py`,
`tests/test_jlens_reasoning_tokens.py`) were repointed; the merged `action_token_ids` keeps the
fork's `(ids, tok)` return, which is what they expect.

## Verified

`DRY_RUN=1 FORCE=1 GRID_ROUND=1 UNRUN=1 ./scripts/reproduce_all.sh` names 44 distinct scripts;
all 44 exist. 408 tests pass. Nothing was deleted except the four files the two merges replaced.


---

# 2026-09-10 — `telos_interp/loudness_analysis`

The loudness line grew by forking: each new question copied the previous script and changed one
join. Four builders were the same operation, `bal_acc` existed five times in two non-equivalent
forms, the trajectory-clustered bootstrap was rolled by hand four times, and the same number was
written under three different column names. It is now one module, decomposed by **artifact
produced** rather than by which outcome happens to be joined.

`telos_interp/` was chosen over a top-level `loudness_analysis/` because `pyproject.toml`
declares `packages = ["telos_interp"]` — it is the only installed package, so anywhere else the
module is path-invoked rather than importable.

## Moved and renamed

| Was | Now | Why the name changed |
|---|---|---|
| `scripts/jlens_reasoning_tokens.py` | `loudness_analysis/build_loudness_tables.py` | said neither what it produces nor, since `--lens` grew a logitlens arm, which lens it uses |
| `scripts/eval_probe_per_token.py` | `loudness_analysis/score_probes_per_token.py` | — |
| `scripts/analyze_probe_loudness.py` | `loudness_analysis/analysis/probe_accuracy_by_loudness.py` | names the question |
| `scripts/analyze_sentence_loudness.py` | `loudness_analysis/analysis/loudness_distribution.py` | names the question |
| `scripts/build_probe_loudness_heldout.py` | `loudness_analysis/join_rollouts.py` | reads as "build the loudness of the probe", which is not what it does |
| `scripts/build_token_loudness_x_infered_action_probability.py` | `loudness_analysis/join_rollout_answers.py` | — |
| `scripts/build_loudness_x_reasoning_pos_heatmap.py` | `loudness_analysis/plotting/loudness_x_chain_position.py` | had a `build_` prefix but is a figure script |
| `scripts/score_probes_heldout.py` | `loudness_analysis/summarise_probe_accuracy.py` | it is a scoreboard, not a loudness analysis |
| `scripts/build_sentence_loudness.py` | `loudness_analysis/build_sentence_loudness.py` | moved only |
| `scripts/inference_oss/**` | `loudness_analysis/rollouts/**` | the whole rollout line, moved unchanged |

## Merged

| Was | Now | Verified by |
|---|---|---|
| `eval_probe_per_token.py` + `grid_cell_analysis/eval_grid_probe_per_token.py` | `score_probes_per_token.py --probe-type {next_action,grid_tile}` | golden CSVs written by **both originals** before deletion, asserted column for column (`tests/data/score_probes_per_token/`) |
| `analyze_probe_loudness.py` + `analyze_grid_loudness_correlation.py` | `probe_accuracy_by_loudness.py --probe-type grid_tile` (counts mode) | `tests/test_loudness_analysis.py` |
| `analyze_direction_word_isolation.py` | `--exclude-signal-words` / `--exclude-radius` on **both** analysers | ditto |
| `plot_probe_loudness.py` + `plot_sentence_loudness.py` + `plot_token_loudness_x_…py` | `plotting/figures.py` (16 figures) + `plotting/_style.py` | `tests/test_loudness_plotting.py` draws end to end |

## Deleted outright

`scripts/build_probe_loudness.py`. It loaded probes and ran torch inline over the eval
manifests; `score_probes_per_token.py` produces the same left table and the join that follows is
then CPU-only, which is what lets a second lens ruler be produced without the GPU.

## Deliberately NOT renamed

`jlens_reasoning_tokens` is also the name of an activation **tree** on disk
(`/workspace/activations/jlens_reasoning_tokens`), plus a CSV and several probe-inventory
entries — 24+ references. The tree was named after the script that made it; renaming those
strings would break every path to real data. The tree keeps its name and is now orphaned from
the script, as this file already records for three other directories.

`wrappers/jlens_reasoning_tokens{,_filtered}.sh` keep their filenames: a wrapper is the record
of a run that happened under that name. Only the script path inside was updated.

## Column names

`{lens}_{signal}_logmass_L{layer}`, e.g. `jlens_direction_logmass_L15`. Every legacy spelling —
`dir_logmass_L15`, `dir_logmass`, `{lens}_mass_L{layer}`, `{lens}_logmass_L{layer}` — is
accepted on read, so **every CSV already on disk still loads**. `dir_*` is refused for a
non-direction signal, because it predates any other vocabulary and so identifies the signal.

## Silent breakages the moves would otherwise have caused

* `run_inference.py` resolves its default trajectory with `Path(__file__).with_name()`, so the
  example JSON had to travel with it.
* `run_inference_strategies.sh` computed `REPO` as `$HERE/../..`, correct two levels deep;
  `rollouts/` is three, so `REPO` would have resolved to `telos_interp/`.
* Three files carried `sys.path.insert(..., parents[1])`, the repo root from `scripts/` and
  `telos_interp/` from two levels deeper.
* The grid call sites needed `--probe-type grid_tile`, not just a new path: the merged evaluator
  defaults to `next_action`, so a path-only update would have silently scored the wrong label.

## Verified

442 tests pass. Every script path named across 34 shell drivers exists (49 distinct paths, 0
missing). Loose script files: 99 → 73 (`wrappers/loudness_report.sh` is new).
