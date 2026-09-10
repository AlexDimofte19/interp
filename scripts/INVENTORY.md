# `scripts/` inventory

Every script under `scripts/`, what it does, and **which data it reads and writes**. 86 files:
54 Python, 32 shell. Paths shown are the *defaults*; almost everything is overridable by flag or
env var, and `reproduce_all.sh` is the canonical record of the values actually used.

Read [CLAUDE.md](../CLAUDE.md) first for the pipeline shape and the on-disk contracts. This file
is the map; §"Duplication" at the end is the refactoring backlog.

## Data roots

| Symbol | Path | What lives there |
|---|---|---|
| `TRAJ` | `/workspace/trajectories/reveng/trajectories_train_single_step` | the 36,000 trajectory JSONs |
| `HELDOUT` | `/workspace/trajectories/heldout360` + `heldout360_names.txt` | the 360-trajectory held-out set |
| `ACT` | `/workspace/activations` | per-token `.pt` trees + lens CSVs + mass tables |
| `PREPARED` | `/workspace/prepared` | `manifest.json` datasets |
| `PROBES` | `/workspace/probes` | trained probe `.pt` checkpoints |
| `RT` | `/workspace/reasoning_theatre` | rollouts, joins, tables, figures, reports |
| `JLENS` | `/workspace/jlens` | `direction_tokens_full.json`, `grid_tokens_full.json`, `gridenv/` |
| `SPLITS` | `/workspace/splits` | count-era name lists |

Three trajectory sets, mutually disjoint by design (`audit_trajectory_sets.py` checks it):

| Set | Names file | n |
|---|---|---|
| count-era lens set | `SPLITS/lens_trajectories_3600.txt` (eval: `eval_trajectories_720.txt`) | 3600 |
| mass-era lens set | `RT/rollout_strategies/mass_l15_names.txt` | 3600 |
| ├ mass-era **train** | `SPLITS/mass_train_2880.txt` → view `ACT/mass_train2880_view` | 2880 |
| └ mass-era **eval** | `SPLITS/mass_eval_720.txt` → view `ACT/mass_eval720_view` | 720 |
| held-out | `/workspace/trajectories/heldout360_names.txt` → tree `ACT/heldout360_lens` | 360 |

The count-era and mass-era sets overlap by 348 names; heldout360 is disjoint from the mass-era
set but shares 33 with the count-era one (a known landmine — see `claude_session_readme.md`).

---

## A. Gather — writes activation trees (GPU only)

| Script | Does | Reads | Writes |
|---|---|---|---|
| `jlens_reasoning_tokens.py` (1932L) | **The engine.** One forward pass per trajectory; emits per-token activations, the `{stem}_{lens}_analysis.csv` of top-20 lens predictions, the `{stem}_direction_mass.csv` wide table + `.meta.json`, and the `{stem}_jlens_selection.json` record. `--extend` merges a new lens arm into an existing record. | `TRAJ`, `--jlens_dir` (`JLENS/gridenv`), `--signal-json` | `--activations-dir` tree: `.pt` + 3 CSV/JSON artifacts per trajectory |
| `jlens_reasoning_tokens.sh` (48L) | Thin wrapper: full sweep, saves **every** (token, layer). Superseded. | `TRAJ`, `JLENS/gridenv` | `ACT/jlens_reasoning_tokens` |
| `jlens_reasoning_tokens_filtered.sh` (85L) | Same sweep but saves only the selection (~75× less disk). The count-era tree's recorded invocation. | `TRAJ`, `JLENS/direction_tokens_full.json` | `ACT/jlens_reasoning_tokens` |
| `jlens_mass_l15.sh` (123L) | The **mass-era** recorded invocation: `logprob_mass_full` ranking, `--select-candidate-layers 15`, layers 7:23 in the CSV. | `TRAJ`, `JLENS/direction_tokens_full.json` | `ACT/jlens_mass_l15` (or `logitlens_mass_l15` with `LENS=logitlens`) |
| `gather_grid_arms.sh` (149L) | Round-2 gather selecting on **grid** words instead of direction words. Never produced a result. | `TRAJ`, `JLENS/grid_tokens_full.json` | `ACT/grid_reasoning_tokens`, log at `/workspace/logs/grid_gather.log` |
| `jlens_extend_logitlens.sh` (98L) | Adds a logitlens arm to an already-pruned tree: CSV-only pass, gather only missing `.pt`, merge the arm. Dry-run by default. | existing tree + `TRAJ` | same tree, in place |
| `delete_non_jlens_selected.py` (406L) | Prunes a fully-gathered tree down to the selection, negatively applying the same `top_filter`. Dry-run by default. | `--activations-dir`, `--trajectories-dir`, `--signal-json` | deletes `.pt` in place |
| `delete_non_jlens_selected.sh` (65L) | Env-var wrapper for the above. `APPLY=1` to actually unlink. | `ACT/jlens_reasoning_tokens` | same |
| `build_activation_view.sh` (55L) | Symlink view of **any** activation tree restricted to one name list. Links at the trajectory level, so the lens CSV, mass table, `.meta.json` and selection record come along. `SRC` defaults to the sentence-end tree. *(was `build_eos_view.sh`)* | `SRC` tree + a names file | `<view>/activations` + `<view>/trajectories` symlink trees |
| `build_mass_era_split.sh` (105L) | Materialises the mass-era 3,600 as **two separate datasets** — train 2,880 and eval 720 — as name lists plus a view each. Does not re-draw the split. Idempotent, `DRY_RUN=1`, `VERIFY_ONLY=1`. | `mass_l15_names.txt`, `next_action_mass_l15_eval_names.txt`, `ACT/jlens_mass_l15` | `SPLITS/mass_{train_2880,eval_720}.txt`, `ACT/mass_{train2880,eval720}_view` |
| `verify_mass_era_split.py` (135L) | That split's contract: disjointness, closure, no dangling links, and that neither half drifted from the pinned definitions. Exits non-zero on failure. | the lists + the views | stdout |
| `inference_oss/gather_local_belief_activations.py` (247L) | Layer-15 residuals at the per-sentence cutoffs a rollout arm chose. | `RT/rollout_strategies/jlens_argmax_per_sentence`, `TRAJ`, `mass_l15_names.txt` | `ACT/argmax_per_sentence_l15` |
| `link_end_of_reasoning_activations.py` (171L) | Restores the final-sentence row `_dedupe` dropped, by symlinking from a source tree. | a rollout dir + arm tree + `--source-tree` | symlinks into the arm tree |

## B. Prepare — activation tree → `manifest.json` dataset

| Script | Does | Reads | Writes |
|---|---|---|---|
| `prepare_next_action_arms.sh` (125L) | One `next_action` dataset per arm (jlens/logitlens/random) via `interp-cli prepare_activations_for_probing --token-selection recorded_<arm>`. | `ACT/jlens_reasoning_tokens`, `TRAJ` | `PREPARED/next_action_<arm>` |
| `prepare_grid_arms.sh` (134L) | The `grid_tile` twin of the above — same tree, same tokens, different label. `MAX_CELLS` caps cells per (trajectory, step). | same | `PREPARED/grid_<arm>` |
| `prepare_next_action_jlens_by_complexity.sh` (59L) | The same prepare restricted to complexity 0.0/0.2/0.4 via a view. | `ACT/jlens_reasoning_tokens_comp0.0-0.2-0.4` | `PREPARED/next_action_comp0.0-0.2-0.4_jlens` |
| `split_next_action_manifest.py` (622L) | **Thin + split.** `--tokens-per-trajectory K`, `--layers-per-token M`, `--single-layer L`, `--thin-mode`, and a by-trajectory train/eval split (`--eval-names` pins it). Copies nothing. | a token-major `PREPARED/<ds>/manifest.json` | `<ds>_train/manifest.json`, `<ds>_eval/manifest.json` |
| `inference_oss/relabel_manifest_from_rollout.py` (237L) | Replaces each entry's label with the model's **local belief** at that cutoff. | a token-major manifest + a rollout dir | a new prepared dir + optional report CSV |
| `intersect_belief_arms.py` (100L) | Cuts several per-sentence arms down to the `(name, step, cut_sentence_idx)` sentences all of them hold. | several prepared arm dirs | `<arm><out_suffix>/manifest.json` each |

## C. Train

| Script | Does | Reads | Writes |
|---|---|---|---|
| `train_next_action_arms.sh` (179L) | `train_next_action_probe` per arm across a top-K sweep, lr + mlp. | `PREPARED/next_action_<arm>` | `PROBES/next_action/*.pt` + logs |
| `train_grid_arms.sh` (174L) | Sibling for `train_cognitive_map_probe`, over a seed sweep. | `PREPARED/grid_<arm>` | `PROBES/grid/*.pt` + logs |
| `run_next_action_arms.sh` (77L) | Master: prepare then train, in order. **Unreferenced.** | — | — |
| `train_next_action_direction_probe.sh` (22L) | Records the parameters of the published comp-0.0/0.2/0.4 run; delegates to `train_next_action_arms.sh`. | `PREPARED/next_action_comp…_jlens` | `PROBES/…` |
| `train_belief_baseline_probes.sh` (121L) | Six belief-baseline probes: relabel → prepare → split → train, per arm. | `RT/rollout_strategies_baselines`, `ACT/logitlens_*` | `PREPARED/entry49_*`, `PROBES/local_belief_baselines` |
| `train_more_belief_arms.sh` (161L) | Eight more belief probes at the per-sentence cadence. Same relabel→prepare→split→train shape. | `RT/rollout_strategies_baselines`, `ACT/eos_mass3600_view`, `ACT/random_per_sentence_l15` | `PREPARED/more_belief_*`, `PROBES/local_belief_baselines` |
| `train_equal_n_belief_arms.sh` (183L) | Fourteen equal-N probes: adds `link_end_of_reasoning_activations` + `intersect_belief_arms` before the same shape. | rollout arms + `ACT/eos_mass3600_view` | `PREPARED/equal_n_*`, `PROBES/local_belief_equalN` |

## D. Rollouts — the model's local belief at a cutoff

| Script | Does | Reads | Writes |
|---|---|---|---|
| `inference_oss/run_inference.py` (872L) | Re-runs gpt-oss at cutoffs inside its own reasoning; one results JSON per trajectory. | `TRAJ`, `--lens-root` (mass table), a names file | `--output-dir/size*/NAME.json` |
| `inference_oss/truncation_strategies.py` (807L) | The `STRATEGIES` registry deciding **where** to cut: `eos`, `jlens_argmax_per_sentence`, `jlens_top_k_global`, `every_token`, `recorded_selection`. Library, not a CLI. | mass tables + `.meta.json`, selection records | — |
| `inference_oss/run_inference_strategies.sh` (165L) | One arm per strategy; the recorded invocation. | `TRAJ`, `ACT/jlens_mass_l15`, `mass_l15_names.txt` | `RT/rollout_strategies/<arm>` |
| `rollout_belief_baseline_arms.sh` (73L) | Three baseline arms (random replay + two logitlens loud arms) — a wrapper around the above. | same | `RT/rollout_strategies_baselines/<arm>` |
| `rollout_more_belief_arms.sh` (78L) | Two more arms (`eos`, `random_per_sentence`) — the same wrapper shape. | same | same |
| `inference_oss/analysis.py` (536L) | Aggregates the results JSONs into 13 figures + summary stats. | a `run_inference` output dir | `--output-dir/*.png` |
| `inference_oss/run_analysis.sh` (25L) | Runs the above on `RT/trajectories_train_single_step_probs`. | that dir | `RT/trajectories_train_single_step_plots` |
| `inference_oss/rollout_status.sh` (194L) | Live progress of the rollout arms (readdirs + log tails only). | `RT/rollout_strategies` | stdout |

## E. Eval / scoring

| Script | Does | Reads | Writes |
|---|---|---|---|
| `eval_probe_per_token.py` (310L) | Scores a `next_action` probe on **every** reasoning token of a set, carrying each token's loudness. | `--probe`, `--activations-dir`, `--trajectories-dir`, `--signal-json` | `--out` per-token CSV |
| `eval_grid_probe_per_token.py` (437L) | The `grid_tile` twin — the specificity control. | same + `--max-cells`, `--pad-to-size` | `--out` per-token CSV |
| `eval_more_belief_arms.sh` (93L) | Reads 24 probes on all 87,221 held-out tokens. | `PROBES/{local_belief*,next_action_mass_l15}`, `ACT/heldout360_lens` | `RT/probe_loudness_heldout360_24probes` |
| `eval_equal_n_belief_arms.sh` (106L) | Same for the 38 equal-N-era probes. **83% line-identical to the above.** | same | `RT/probe_loudness_heldout360_equal_n` |
| `score_probes_heldout.py` (111L) | Balanced accuracy per probe against both label definitions. | a per-token CSV | table + optional JSON |
| `compare_equal_n_arms.py` (116L) | Old arms vs. their equal-N rebuilds, from `results.best_balanced_accuracy` in each checkpoint. **Unreferenced.** | `PROBES/**/*.pt` | stdout + optional JSON |
| `inference_oss/eval_local_belief.py` (106L) | A belief probe against both the local belief and the final action. | a probe + an eval prepared dir | stdout |

## F. Joins — the per-token tables everything downstream reads

| Script | Does | Reads | Writes |
|---|---|---|---|
| `build_probe_rollout_join.py` (318L) | Joins probe readouts × rollouts × lens into the headline table. | `PROBES/heldout360_all_probes.csv`, `RT/trajectories_train_single_step_probs`, `ACT/heldout360_l15`, direction JSON | `RT/probe_vs_rollout/per_token.csv` |
| `merge_probe_rollout_arms.py` (88L) | Clones that CSV with extra probe arms merged in. | the above + extra CSVs | a new per-token CSV |
| `build_sentence_loudness.py` (215L) | Per-token direction **mass** placed inside its reasoning sentence, on the eval-720 split. | `ACT/jlens_mass_l15`, `RT/trajectories_train_single_step_probs`, `PREPARED/next_action_mass_l15_eval_names.txt` | `RT/loudness/per_token.csv` |
| `build_probe_loudness.py` (356L) | One row per held-out token of the local-belief probes: loudness + sentence position + every probe's prediction. | `ACT/jlens_mass_l15`, `PROBES/*`, `PREPARED/*` | `RT/probe_loudness/per_token.csv` |
| `build_probe_loudness_heldout.py` (570L) | The same over the **held-out 360** and over every reasoning token. | `ACT/heldout360_lens`, `RT/rollout_strategies_heldout360/every_token`, `RT/probe_vs_rollout/per_token.csv` | `RT/probe_loudness_heldout360/per_token.csv` |
| `build_token_loudness_x_infered_action_probability.py` (513L) | Per-token loudness under **both** lenses against the action the model gives if cut there. | `RT/rollout_strategies_heldout360/every_token`, `ACT/heldout360_lens`, `HELDOUT` | `RT/loudness_vs_answer_prob/heldout360_per_token.csv` |
| `build_convinced_dataset.py` (333L) | Per-token loudness against a convinced / not-convinced label. | `ACT/jlens_mass_l15`, `RT/trajectories_train_single_step_probs`, direction JSON | `--out` CSV + `.meta.json` |
| `build_convinced_datasets.sh` (37L) | The three convinced datasets as produced: **train 2880 / val 720 / eval 360**. | as above + `heldout360_lens` | `RT/convinced_classifier/{train,val,eval_heldout360}_all.csv` |
| `verify_convinced_datasets.py` (115L) | Checks those three against independently-built artifacts. | `RT/convinced_classifier`, `RT/loudness`, `RT/probe_vs_rollout` | stdout |

## G. Analysis — table producers (read a per-token CSV, write `tables/` + `summary.json`)

| Script | Does | Reads | Writes |
|---|---|---|---|
| `analyze_probe_rollout.py` (442L) | The four reasoning-theatre questions Q2–Q4. | `RT/probe_vs_rollout/per_token.csv` | `RT/probe_vs_rollout/tables/*.csv` + `summary.json` |
| `analyze_probe_loudness.py` (391L) | Does a louder token decode better, or is it sentence position? | `RT/probe_loudness/per_token.csv`, `RT/loudness/per_token.csv` | `RT/probe_loudness/tables` + `summary.json` |
| `analyze_sentence_loudness.py` (227L) | Within-sentence decay and the commitment boundary. | `RT/loudness/per_token.csv` | `RT/loudness/loudness_summary.csv` + `summary.json` |
| `analyze_truncation_strategies.py` (356L) | Compares the rollout arms on one table of cutoffs. | `RT/rollout_strategies`, `ACT/jlens_mass_l15`, `TRAJ` | `--output-dir/*.csv` + `summary.json` |
| `analyze_direction_word_isolation.py` (408L) | The direction-word confound: does loudness just read a word the model typed? | `RT/probe_loudness_heldout360_16probes/per_token_{jlens,logitlens}_loudness.csv` | `tables/` + `plots/` + `summary.json` |
| `analyze_jlens_direction_classes.py` (279L) | The lens read through the full direction vocabulary, split by class. | `RT/probe_vs_rollout/per_token.csv`, direction JSON | `RT/probe_vs_rollout/q4v_*.csv` |
| `analyze_grid_loudness_correlation.py` (226L) | The specificity control: direction loudness vs. **grid** decodability. | an `eval_grid_probe_per_token.py` CSV | `--out` tables |

## H. Figures

| Script | Does | Reads | Writes |
|---|---|---|---|
| `plot_probe_rollout.py` (273L) | Re-styles the probe-vs-rollout figures from `tables/` only. | `RT/probe_vs_rollout/tables` | `…/plots/*.png` |
| `plot_probe_loudness.py` (386L) | The probe-loudness figures. | `RT/probe_loudness/per_token.csv` | `RT/probe_loudness/plots` |
| `plot_sentence_loudness.py` (526L) | Where the lens is loud inside a sentence and along the chain. | `RT/loudness/per_token.csv` | `RT/loudness/*.png` |
| `plot_commitment_all_tokens.py` (251L) | The commitment boundary at **token** resolution. | `RT/probe_vs_rollout/per_token.csv` | `…/plots/commit_all_tokens_*.png` |
| `plot_commitment_probs.py` (239L) | The same boundary in the probe's **probabilities**. | `…/per_token_probs.csv` | `…/plots` + `…/tables` |
| `plot_loud_vs_sentence_end.py` (278L) | Does a loud token's action match its sentence's conclusion or the previous one? | `RT/rollout_strategies`, `ACT/jlens_mass_l15` | `RT/rollout_strategies/comparison` |
| `plot_token_loudness_x_infered_action_probability.py` (379L) | Figures for the loudness × answer-probability table. **Unreferenced.** | `RT/loudness_vs_answer_prob/heldout360_per_token.csv` | `…/plots/*.png` |
| `build_loudness_x_reasoning_pos_heatmap.py` (509L) | Loudness × chain-position 2-D heatmap — separates the two axes. **Unreferenced.** | `RT/probe_loudness_heldout360_16probes` | `RT/360_held_out/loudness_x_reasoning_pos` |
| `plot_object_prediction_rate_by_distance.py` (195L) | Goal/agent prediction rate vs. distance, from a results JSON. | a results JSON | `--out-dir` |
| `plot_object_prediction_rate_by_distance_from_csv.py` (231L) | Same figure from the per-prediction CSV. **Unreferenced.** | a predictions CSV | `--out-dir` |
| `plot_per_class_accuracy_by_distance.py` (241L) | Per-class accuracy vs. distance, from a results JSON. | a results JSON | `--out-dir` |
| `plot_per_class_accuracy_by_distance_from_csv.py` (297L) | Same from the CSV. **Unreferenced.** | a predictions CSV | `--out-dir` |
| `jlens_rank_analysis.py` (135L) | Four supervisor-facing figures from an action-rank CSV. | a `jlens_action_ranks.csv` | `--out_dir/fig*.png` |
| `jlens_slice_page.py` (93L) | A jacobian-lens slice page for one trajectory step. **Unreferenced.** | a trajectory + a lens | `slice.html` |
| `jlens_viewer_export.py` (712L) | Streams the (3.2 GB) reasoning-token CSV into per-step viewer JSONs. | a lens CSV + `TRAJ` | `--out-dir/**/step_*.json` |

## I. Reports

| Script | Does | Reads | Writes |
|---|---|---|---|
| `build_sixteen_probe_loudness_report.sh` (139L) | End-to-end: 16 probes on the held-out 360 under both rulers → the report page. | `PROBES/*`, `ACT/heldout360_lens` | `RT/probe_loudness_heldout360_16probes` |
| `build_sixteen_probe_report_page.py` (746L) | Renders that page with full provenance under every figure. | that output root | `…/index.html` |
| `build_probe_inventory.py` (974L) | The probe inventory spreadsheet from what is on disk. | `PROBES`, `PREPARED`, `RT` | `probe_inventory.{xlsx,csv}` |

## J. Diagnostics, status, audit

| Script | Does | Reads | Writes |
|---|---|---|---|
| `audit_trajectory_sets.py` (107L) | Which of the three trajectory sets produced each artifact on disk. | `PREPARED`, `PROBES`, `RT`, `ACT/heldout360_lens` | stdout |
| `jlens_layer_profile.py` (188L) | The **unbiased** mean direction score per layer over a whole tree — where `--single-layer L` should come from. | an activations dir + `--signal-json` | stdout / `--out` |
| `jlens_direction_vocab_diagnostic.py` (55L) | Which direction-vocabulary tokens actually reach the lens top-20, per class. | `ACT/heldout360_l15`, direction JSON | stdout |
| `jlens_action_ranks.py` (153L) | Ranks the four action tokens under the lens for every saved activation. | `--activations_root`, `--jlens_dir` | `jlens_action_ranks.csv` |
| `jlens_action_ranks_sampled.py` (227L) | A sampled fork of the above (N runs per size×complexity×layer). **73% identical.** | same + `--trajectories_root` | same |
| `gather_reasoning_steps_statistics.py` (246L) | Counts `.pt` per (size, complexity) and plots reasoning-step counts. **Unreferenced.** | an activations root | `reasoning_step_figures/` |
| `belief_baselines_status.sh` (117L) | One snapshot of the belief-baseline round. | `ACT`, `PROBES`, `RT`, `/workspace/logs` | stdout |
| `grid_round_status.sh` (93L) | The same for the grid round. | `ACT/grid_reasoning_tokens`, `PREPARED/grid_l15`, `PROBES/grid` | stdout |

## K. Orchestration and distribution

| Script | Does | Reads | Writes |
|---|---|---|---|
| `reproduce_all.sh` (746L) | Rebuilds every result in dependency order, one stage per experiment. `LIST=1`, `DRY_RUN=1`, `ONLY`/`SKIP`/`FROM`/`FORCE`, `GRID_ROUND=1`, `UNRUN=1`. **The canonical record of every path and flag.** | all of `/workspace` | all of `/workspace` |
| `push_to_huggingface.sh` (358L) | Packs and pushes splits, trajectories, prepared datasets, trees, CSVs, probes, results to one HF dataset repo. | `/workspace` | `HF_ORG/jlens_decodability_property` |
| `pull_artifacts_to_laptop.sh` (303L) | Runs **on the laptop**; rsyncs back everything but the `.pt` trees. Tiers move the total between ~4 GB and ~57 GB. | `HOST:/workspace` | `$DEST` + `README_PATHS.md` |

---

## Duplication

Measured, not eyeballed: `difflib` ratio over comment-stripped lines, and AST-level comparison of
top-level function bodies.

### Byte-identical function bodies copied between scripts

| Function | Copies in |
|---|---|
| `find_act_folder`, `read_lens_tables`, `read_mass_columns`, `load_trajectory`, `trajectory_dirs` | `eval_probe_per_token.py` ↔ `eval_grid_probe_per_token.py` |
| `ensure_unembed_assets` (946 chars) | `jlens_action_ranks.py` ↔ `jlens_action_ranks_sampled.py` |
| `load_probe` | `build_probe_loudness.py` ↔ `inference_oss/eval_local_belief.py` |
| `bal_acc` | `analyze_direction_word_isolation.py` ↔ `plot_probe_loudness.py` |
| `_facets` (774 chars) | `plot_object_prediction_rate_by_distance_from_csv.py` ↔ `plot_per_class_accuracy_by_distance_from_csv.py` |
| `_complexity_levels` | `plot_object_prediction_rate_by_distance.py` ↔ `plot_per_class_accuracy_by_distance.py` |

Re-implemented rather than copied verbatim, but the same idea in several places:
`load` (×4), `make_figure` (×4), `bal_acc`/`balanced_accuracy` (×5), `qbin`, `wilson`, `zscore`,
`read_rows`, `expand_paths`, `entries_key`, `clustered_band`, `draw`, `score`, `collect`.

### Whole-file near-duplicates

| Ratio | Pair |
|---|---|
| 0.83 | `eval_equal_n_belief_arms.sh` ↔ `eval_more_belief_arms.sh` |
| 0.73 | `jlens_action_ranks.py` ↔ `jlens_action_ranks_sampled.py` |
| 0.73 | `jlens_mass_l15.sh` ↔ `jlens_reasoning_tokens_filtered.sh` |
| 0.57 | `prepare_grid_arms.sh` ↔ `prepare_next_action_arms.sh` |
| 0.55 | `rollout_belief_baseline_arms.sh` ↔ `rollout_more_belief_arms.sh` |
| 0.53 | `plot_object_prediction_rate_by_distance_from_csv.py` ↔ `plot_per_class_accuracy_by_distance_from_csv.py` |
| 0.47 | `plot_object_prediction_rate_by_distance.py` ↔ its `_from_csv` twin |
| 0.46 | `jlens_reasoning_tokens.sh` ↔ `jlens_reasoning_tokens_filtered.sh` |
| 0.45 | `plot_object_prediction_rate_by_distance.py` ↔ `plot_per_class_accuracy_by_distance.py` |

### Repeated shapes below the line-similarity threshold

Four gather wrappers (`jlens_reasoning_tokens{,_filtered}.sh`, `jlens_mass_l15.sh`,
`gather_grid_arms.sh`) each rebuild the same ~25-variable env block and `CMD=(...)` array over one
Python entry point; they differ in the *values*, which is exactly what a recorded invocation should
be — but the assembly is copied four times.

Three belief-round drivers (`train_belief_baseline_probes.sh`, `train_more_belief_arms.sh`,
`train_equal_n_belief_arms.sh`) each run relabel → prepare → split → train per arm, with the same
`EVAL_NAMES` / `SINGLE_LAYER=15` / `SEED=42` / `EPOCHS=50` block. Pairwise ratios 0.32–0.43 —
low, because the arm *lists* differ, not the machinery.

### Unreferenced by any script, doc, config or test (8)

`build_loudness_x_reasoning_pos_heatmap.py`, `compare_equal_n_arms.py`,
`gather_reasoning_steps_statistics.py`, `jlens_slice_page.py`,
`plot_object_prediction_rate_by_distance_from_csv.py`,
`plot_per_class_accuracy_by_distance_from_csv.py`,
`plot_token_loudness_x_infered_action_probability.py`, `run_next_action_arms.sh`.

"Unreferenced" is not "dead" — `build_loudness_x_reasoning_pos_heatmap.py` was committed
deliberately two commits ago. It means nothing else will run them, so they are the ones whose
provenance has to be read out of `ICLR log.txt` rather than off a call graph.
