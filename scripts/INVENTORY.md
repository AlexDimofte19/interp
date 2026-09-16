# `scripts/` inventory

Every script in the repo, what it does, and **which data it reads and writes**. Paths shown are
the *defaults*; almost everything is overridable by flag or env var, and `reproduce_all.sh` is the
canonical record of the values actually used.

Read [CLAUDE.md](../CLAUDE.md) first for the pipeline shape and the on-disk contracts.

## Where things live

| Directory | Holds | n |
|---|---|---|
| [`telos_interp/loudness_analysis/`](../telos_interp/loudness_analysis/README.md) | **the lens → probe → loudness pipeline**, and the rollout line. An importable package, not a script folder. | 29 |
| `scripts/` | **the machinery** — everything that does work and is not part of that pipeline | 41 |
| [`wrappers/`](../wrappers/README.md) | **the record** — recorded invocations that pin one run's parameters onto a script here. Do not refactor them; changing a default rewrites history. | 15 |
| [`grid_cell_analysis/`](../grid_cell_analysis/README.md) | the grid-cell ("cognitive map") line: the published round-1 probes, and the round-2 grid-label arms that **never produced a result** | 18 |
| `configs/**/*.conf` | recorded `interp-cli` invocations, same role as `wrappers/` | — |

Only `runpod_setup.sh` is left loose at the repo root.

**The loudness line no longer lives here.** Everything that produces, joins, analyses or draws
loudness moved to `telos_interp/loudness_analysis/`, which has [its own
README](../telos_interp/loudness_analysis/README.md) and is the place to start for that work.
Rows below that point into it are kept because the inventory is also a map of where things went.
See [RENAMES.md](../RENAMES.md) for the full table.

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
| `build_loudness_tables.py` (1932L) | **The engine.** One forward pass per trajectory; emits per-token activations, the `{stem}_{lens}_analysis.csv` of top-20 lens predictions, the `{stem}_direction_mass.csv` wide table + `.meta.json`, and the `{stem}_jlens_selection.json` record. `--extend` merges a new lens arm into an existing record. | `TRAJ`, `--jlens_dir` (`JLENS/gridenv`), `--signal-json` | `--activations-dir` tree: `.pt` + 3 CSV/JSON artifacts per trajectory |
| `wrappers/jlens_reasoning_tokens.sh` (48L) | Thin wrapper: full sweep, saves **every** (token, layer). Superseded. | `TRAJ`, `JLENS/gridenv` | `ACT/jlens_reasoning_tokens` |
| `wrappers/jlens_reasoning_tokens_filtered.sh` (85L) | Same sweep but saves only the selection (~75× less disk). The count-era tree's recorded invocation. | `TRAJ`, `JLENS/direction_tokens_full.json` | `ACT/jlens_reasoning_tokens` |
| `wrappers/jlens_mass_l15.sh` (123L) | The **mass-era** recorded invocation: `logprob_mass_full` ranking, `--select-candidate-layers 15`, layers 7:23 in the CSV. | `TRAJ`, `JLENS/direction_tokens_full.json` | `ACT/jlens_mass_l15` (or `logitlens_mass_l15` with `LENS=logitlens`) |
| `grid_cell_analysis/gather_grid_arms.sh` (149L) | Round-2 gather selecting on **grid** words instead of direction words. Never produced a result. | `TRAJ`, `JLENS/grid_tokens_full.json` | `ACT/grid_reasoning_tokens`, log at `/workspace/logs/grid_gather.log` |
| `wrappers/jlens_extend_logitlens.sh` (98L) | Adds a logitlens arm to an already-pruned tree: CSV-only pass, gather only missing `.pt`, merge the arm. Dry-run by default. | existing tree + `TRAJ` | same tree, in place |
| `delete_non_jlens_selected.py` (406L) | Prunes a fully-gathered tree down to the selection, negatively applying the same `top_filter`. Dry-run by default. | `--activations-dir`, `--trajectories-dir`, `--signal-json` | deletes `.pt` in place |
| `wrappers/delete_non_jlens_selected.sh` (65L) | Env-var wrapper for the above. `APPLY=1` to actually unlink. | `ACT/jlens_reasoning_tokens` | same |
| `build_activation_view.sh` (55L) | Symlink view of **any** activation tree restricted to one name list. Links at the trajectory level, so the lens CSV, mass table, `.meta.json` and selection record come along. `SRC` defaults to the sentence-end tree. *(was `build_eos_view.sh`)* | `SRC` tree + a names file | `<view>/activations` + `<view>/trajectories` symlink trees |
| `build_mass_era_split.sh` (105L) | Materialises the mass-era 3,600 as **two separate datasets** — train 2,880 and eval 720 — as name lists plus a view each. Does not re-draw the split. Idempotent, `DRY_RUN=1`, `VERIFY_ONLY=1`. | `mass_l15_names.txt`, `next_action_mass_l15_eval_names.txt`, `ACT/jlens_mass_l15` | `SPLITS/mass_{train_2880,eval_720}.txt`, `ACT/mass_{train2880,eval720}_view` |
| `verify_mass_era_split.py` (135L) | That split's contract: disjointness, closure, no dangling links, and that neither half drifted from the pinned definitions. Exits non-zero on failure. | the lists + the views | stdout |
| `telos_interp/loudness_analysis/rollouts/gather_local_belief_activations.py` (247L) | Layer-15 residuals at the per-sentence cutoffs a rollout arm chose. | `RT/rollout_strategies/jlens_argmax_per_sentence`, `TRAJ`, `mass_l15_names.txt` | `ACT/argmax_per_sentence_l15` |
| `link_end_of_reasoning_activations.py` (171L) | Restores the final-sentence row `_dedupe` dropped, by symlinking from a source tree. | a rollout dir + arm tree + `--source-tree` | symlinks into the arm tree |

## B. Prepare — activation tree → `manifest.json` dataset

| Script | Does | Reads | Writes |
|---|---|---|---|
| `prepare_next_action_arms.sh` (125L) | One `next_action` dataset per arm (jlens/logitlens/random) via `interp-cli prepare_activations_for_probing --token-selection recorded_<arm>`. | `ACT/jlens_reasoning_tokens`, `TRAJ` | `PREPARED/next_action_<arm>` |
| `grid_cell_analysis/prepare_grid_arms.sh` (134L) | The `grid_tile` twin of the above — same tree, same tokens, different label. `MAX_CELLS` caps cells per (trajectory, step). | same | `PREPARED/grid_<arm>` |
| `wrappers/prepare_next_action_jlens_by_complexity.sh` (59L) | The same prepare restricted to complexity 0.0/0.2/0.4 via a view. | `ACT/jlens_reasoning_tokens_comp0.0-0.2-0.4` | `PREPARED/next_action_comp0.0-0.2-0.4_jlens` |
| `split_next_action_manifest.py` (622L) | **Thin + split.** `--tokens-per-trajectory K`, `--layers-per-token M`, `--single-layer L`, `--thin-mode`, and a by-trajectory train/eval split (`--eval-names` pins it). Copies nothing. | a token-major `PREPARED/<ds>/manifest.json` | `<ds>_train/manifest.json`, `<ds>_eval/manifest.json` |
| `telos_interp/loudness_analysis/rollouts/relabel_manifest_from_rollout.py` (237L) | Replaces each entry's label with the model's **local belief** at that cutoff. | a token-major manifest + a rollout dir | a new prepared dir + optional report CSV |
| `intersect_belief_arms.py` (100L) | Cuts several per-sentence arms down to the `(name, step, cut_sentence_idx)` sentences all of them hold. | several prepared arm dirs | `<arm><out_suffix>/manifest.json` each |

## C. Train

| Script | Does | Reads | Writes |
|---|---|---|---|
| `train_next_action_arms.sh` (179L) | `train_next_action_probe` per arm across a top-K sweep, lr + mlp. | `PREPARED/next_action_<arm>` | `PROBES/next_action/*.pt` + logs |
| `grid_cell_analysis/train_grid_arms.sh` (174L) | Sibling for `train_cognitive_map_probe`, over a seed sweep. | `PREPARED/grid_<arm>` | `PROBES/grid/*.pt` + logs |
| `wrappers/run_next_action_arms.sh` (77L) | Master: prepare then train, in order. **Unreferenced.** | — | — |
| `wrappers/train_next_action_direction_probe.sh` (22L) | Records the parameters of the published comp-0.0/0.2/0.4 run; delegates to `train_next_action_arms.sh`. | `PREPARED/next_action_comp…_jlens` | `PROBES/…` |
| `train_belief_baseline_probes.sh` (121L) | Six belief-baseline probes: relabel → prepare → split → train, per arm. | `RT/rollout_strategies_baselines`, `ACT/logitlens_*` | `PREPARED/entry49_*`, `PROBES/local_belief_baselines` |
| `train_more_belief_arms.sh` (161L) | Eight more belief probes at the per-sentence cadence. Same relabel→prepare→split→train shape. | `RT/rollout_strategies_baselines`, `ACT/eos_mass3600_view`, `ACT/random_per_sentence_l15` | `PREPARED/more_belief_*`, `PROBES/local_belief_baselines` |
| `train_equal_n_belief_arms.sh` (183L) | Fourteen equal-N probes: adds `link_end_of_reasoning_activations` + `intersect_belief_arms` before the same shape. | rollout arms + `ACT/eos_mass3600_view` | `PREPARED/equal_n_*`, `PROBES/local_belief_equalN` |

## D. Rollouts — the model's local belief at a cutoff

| Script | Does | Reads | Writes |
|---|---|---|---|
| `telos_interp/loudness_analysis/rollouts/run_inference.py` (872L) | Re-runs gpt-oss at cutoffs inside its own reasoning; one results JSON per trajectory. | `TRAJ`, `--lens-root` (mass table), a names file | `--output-dir/size*/NAME.json` |
| `telos_interp/loudness_analysis/rollouts/truncation_strategies.py` (807L) | The `STRATEGIES` registry deciding **where** to cut: `eos`, `jlens_argmax_per_sentence`, `jlens_top_k_global`, `every_token`, `recorded_selection`. Library, not a CLI. | mass tables + `.meta.json`, selection records | — |
| `telos_interp/loudness_analysis/rollouts/run_inference_strategies.sh` (165L) | One arm per strategy; the recorded invocation. | `TRAJ`, `ACT/jlens_mass_l15`, `mass_l15_names.txt` | `RT/rollout_strategies/<arm>` |
| `wrappers/rollout_belief_baseline_arms.sh` (73L) | Three baseline arms (random replay + two logitlens loud arms) — a wrapper around the above. | same | `RT/rollout_strategies_baselines/<arm>` |
| `wrappers/rollout_more_belief_arms.sh` (78L) | Two more arms (`eos`, `random_per_sentence`) — the same wrapper shape. | same | same |
| `telos_interp/loudness_analysis/rollouts/analysis.py` (536L) | Aggregates the results JSONs into 13 figures + summary stats. | a `run_inference` output dir | `--output-dir/*.png` |
| `wrappers/run_analysis.sh` (31L) | Runs the above on `RT/trajectories_train_single_step_probs`. | that dir | `RT/trajectories_train_single_step_plots` |
| `telos_interp/loudness_analysis/rollouts/rollout_status.sh` (194L) | Live progress of the rollout arms (readdirs + log tails only). | `RT/rollout_strategies` | stdout |

## E. Eval / scoring

| Script | Does | Reads | Writes |
|---|---|---|---|
| `eval_probe_per_token.py` (310L) | Scores a `next_action` probe on **every** reasoning token of a set, carrying each token's loudness. | `--probe`, `--activations-dir`, `--trajectories-dir`, `--signal-json` | `--out` per-token CSV |
| `telos_interp/loudness_analysis/score_probes_per_token.py` (437L) | The `grid_tile` twin — the specificity control. | same + `--max-cells`, `--pad-to-size` | `--out` per-token CSV |
| `eval_belief_arms_heldout.sh` (146L) | Reads every belief probe of a round on all 87,221 held-out tokens. `24probes` = the entry-49/50 round; `equal_n` = those plus the 14 that supersede them. *(merged from `eval_more_belief_arms.sh` + `eval_equal_n_belief_arms.sh`, which were 83% identical)* | `PROBES/{local_belief*,local_belief_equalN,next_action_mass_l15}`, `ACT/heldout360_lens` | `RT/probe_loudness_heldout360_{24probes,equal_n}` |
| `score_probes_heldout.py` (111L) | Balanced accuracy per probe against both label definitions. | a per-token CSV | table + optional JSON |
| `telos_interp/loudness_analysis/rollouts/eval_local_belief.py` (106L) | A belief probe against both the local belief and the final action. | a probe + an eval prepared dir | stdout |

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
| `wrappers/build_convinced_datasets.sh` (37L) | The three convinced datasets as produced: **train 2880 / val 720 / eval 360**. | as above + `heldout360_lens` | `RT/convinced_classifier/{train,val,eval_heldout360}_all.csv` |
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
| `telos_interp/loudness_analysis/analysis/probe_accuracy_by_loudness.py --probe-type grid_tile` (226L) | The specificity control: direction loudness vs. **grid** decodability. | an `eval_grid_probe_per_token.py` CSV | `--out` tables |

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
| `grid_cell_analysis/plots/plot_object_prediction_rate_by_distance.py` (195L) | Goal/agent prediction rate vs. distance, from a results JSON. | a results JSON | `--out-dir` |
| `grid_cell_analysis/plots/plot_object_prediction_rate_by_distance_from_csv.py` (231L) | Same figure from the per-prediction CSV. **Unreferenced.** | a predictions CSV | `--out-dir` |
| `grid_cell_analysis/plots/plot_per_class_accuracy_by_distance.py` (241L) | Per-class accuracy vs. distance, from a results JSON. | a results JSON | `--out-dir` |
| `grid_cell_analysis/plots/plot_per_class_accuracy_by_distance_from_csv.py` (297L) | Same from the CSV. **Unreferenced.** | a predictions CSV | `--out-dir` |
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
| `jlens_action_ranks.py` (229L) | Ranks the four action tokens under the lens for every saved activation. `--runs_per_combo N` samples; `--trajectories_root` widens the CSV with `agent_action`, logprobs and top/bottom tokens. Without it the narrow schema is unchanged. *(absorbed `jlens_action_ranks_sampled.py`, 73% identical)* | `--activations_root`, `--jlens_dir` | `jlens_action_ranks.csv` |
| `gather_reasoning_steps_statistics.py` (246L) | Counts `.pt` per (size, complexity) and plots reasoning-step counts. **Unreferenced.** | an activations root | `reasoning_step_figures/` |
| `belief_baselines_status.sh` (117L) | One snapshot of the belief-baseline round. | `ACT`, `PROBES`, `RT`, `/workspace/logs` | stdout |
| `grid_cell_analysis/grid_round_status.sh` (93L) | The same for the grid round. | `ACT/grid_reasoning_tokens`, `PREPARED/grid_l15`, `PROBES/grid` | stdout |

## K. Orchestration and distribution

| Script | Does | Reads | Writes |
|---|---|---|---|
| `reproduce_all.sh` (746L) | Rebuilds every result in dependency order, one stage per experiment. `LIST=1`, `DRY_RUN=1`, `ONLY`/`SKIP`/`FROM`/`FORCE`, `GRID_ROUND=1`, `UNRUN=1`. **The canonical record of every path and flag.** | all of `/workspace` | all of `/workspace` |
| `push_to_huggingface.sh` (358L) | Packs and pushes splits, trajectories, prepared datasets, trees, CSVs, probes, results to one HF dataset repo. | `/workspace` | `HF_ORG/jlens_decodability_property` |
| `pull_artifacts_to_laptop.sh` (303L) | Runs **on the laptop**; rsyncs back everything but the `.pt` trees. Tiers move the total between ~4 GB and ~57 GB. | `HOST:/workspace` | `$DEST` + `README_PATHS.md` |
| `boundary_cognitive_maps.py` (2107L) | **The boundary round, end to end**: locate the six channel-boundary tokens (`<|end|><|start|>assistant`, once before the reasoning chain and once after it), gather their layer-15 residuals, write the token-major `grid_tile` manifests, score the J-lens grid mass at the same positions, evaluate each probe on its own boundary, and pair post against pre. Eight subcommands: `extract prepare loudness evaluate compare plots manifest`. Recorded invocation: `wrappers/boundary_cognitive_maps.sh`. | `SPLITS`, `TRAJ` via the mass views, `HELDOUT`, `JLENS/gridenv`, `data/jlens/grid_tokens_full.json` | `PROBES/2880_trajectory_trained_cogn_maps`, `/workspace/loudness_evaluation/2880_trajectory-trained_cogn_maps` |

---

## Duplication — what was resolved, and what is left

Measured, not eyeballed: `difflib` ratio over comment-stripped lines, and AST-level comparison of
top-level function bodies. Re-run it after any merge.

### Resolved

| Was | Now |
|---|---|
| `eval_equal_n_belief_arms.sh` ↔ `eval_more_belief_arms.sh` (0.83) | `eval_belief_arms_heldout.sh <round>` — a round is a `case` entry. Both key maps verified byte-identical before deletion. |
| `jlens_action_ranks.py` ↔ `jlens_action_ranks_sampled.py` (0.73) | one `jlens_action_ranks.py`; the extra flags are optional and the narrow CSV schema is unchanged without them |
| 11 parameter-record wrappers scattered through `scripts/` and the repo root | [`wrappers/`](../wrappers/README.md), where the copied `CMD=(...)` assembly is the *point* rather than a defect |
| the grid-cell line spread over `scripts/`, `plotting_scripts/`, `evaluation_scripts/` | [`grid_cell_analysis/`](../grid_cell_analysis/README.md); the two now-empty root folders are gone |
| **the five duplicated lens-IO helpers** (`find_act_folder`, `read_lens_tables`, `read_mass_columns`, `load_trajectory`, `trajectory_dirs`) | `loudness_analysis/lens_io.py`. `eval_grid_probe_per_token.py` is gone: `score_probes_per_token.py --probe-type grid_tile`. |
| **`bal_acc` × 5, in two non-equivalent forms**, and the trajectory-clustered bootstrap × 4 | `loudness_analysis/stats.py`. Both balanced accuracies are kept and named apart — a grid row summarises many predictions, so pooling counts and averaging rows are different numbers. |
| **the four builders** (`build_sentence_loudness`, `build_probe_loudness`, `build_probe_loudness_heldout`, `build_token_loudness_x_…`) | one join (`join_rollouts.py`) plus two tree-side joins; `build_probe_loudness.py` deleted outright as superseded. |
| **the four analysis scripts** | two: `analysis/probe_accuracy_by_loudness.py` and `analysis/loudness_distribution.py`. The confound script became `--exclude-signal-words`; the grid twin became `--probe-type grid_tile`. |
| **the three loudness plotters** (16 figures) | `plotting/figures.py` + `plotting/_style.py`, behind one registry and one CLI. |
| three spellings of the same loudness column | `{lens}_{signal}_logmass_L{layer}`, with every legacy name accepted on read — see `loudness_analysis/columns.py`. |

### Left, and why

**The two tree-side joins.** `join_rollout_answers.py` and `build_sentence_loudness.py` are both
"mass tree + rollout → per-token table"; they differ in how many lenses they read and which
rollout fields they carry. Folding them into `join_rollouts.py` as a `--lens-root` mode is the
remaining merge in this line. It is listed rather than done because both produce published
tables and every merge in this round was gated on golden-file equivalence against the originals
— that gate has not been built for these two, and doing the merge without it is exactly where a
silent change to a published number would come from.

**The four `*_by_distance` plots** (0.45–0.53 pairwise, `_facets` and `_complexity_levels`
verbatim) collapse cleanly into one `--metric {object,per-class} --source {json,csv}`. Dormant
round, and all four are unreferenced.

**Three belief-round drivers.** `train_belief_baseline_probes.sh`, `train_more_belief_arms.sh`
and `train_equal_n_belief_arms.sh` each run relabel → prepare → split → train per arm, with the
same `EVAL_NAMES` / `SINGLE_LAYER=15` / `SEED=42` / `EPOCHS=50` block. Pairwise ratios are only
0.32–0.43 — low, because the arm *lists* differ, not the machinery. The biggest remaining merge
in the active pipeline.

**`prepare_grid_arms.sh` ↔ `prepare_next_action_arms.sh` (0.57)** and `train_grid_arms.sh` ↔
`train_next_action_arms.sh` (0.42) are label-twins a `--label` flag would collapse. They sit
either side of the `grid_cell_analysis/` boundary, so merging them means deciding that boundary
matters less than the duplication. It currently does not.

### Unreferenced by any script, doc, config or test

`gather_reasoning_steps_statistics.py`, `jlens_slice_page.py`, the two
`grid_cell_analysis/plots/*_from_csv.py`, and `wrappers/run_next_action_arms.sh`.

"Unreferenced" is not "dead" — it means nothing else will run them, so their provenance has to be
read out of `ICLR log.txt` rather than off a call graph.
