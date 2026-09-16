# Qwen3.6-35B-A3B P2 round — what code already exists, per task

Written 2026-09-16. **Nothing in here has been run**, and nothing below is a recommendation to
run it yet: this is an inventory of the machinery each task needs, what transfers from the
gpt-oss line unchanged, and the short list of places that are gpt-oss-specific and have to be
touched first.

Read [CLAUDE.md](CLAUDE.md) for the pipeline shape and the on-disk contracts, and
[scripts/INVENTORY.md](scripts/INVENTORY.md) for the full script map. **INVENTORY.md's
sections E–H are stale**: `eval_probe_per_token.py`, `analyze_probe_loudness.py`,
`plot_probe_loudness.py`, `build_probe_loudness*.py`, `analyze_sentence_loudness.py`,
`score_probes_heldout.py` and `analyze_direction_word_isolation.py` no longer exist as files —
they were consolidated into `telos_interp/loudness_analysis/`. The current map is
[that package's README](telos_interp/loudness_analysis/README.md).

---

## 0. State of the Qwen line as of this writing

| Thing | Where | State |
|---|---|---|
| Jacobian-lens **fit** | `jlens/jlens_fit_qwen.py`, `wrappers/jlens_fit_qwen.sh` | committed (75e88b7); the fit itself was **running** on this host, ~112/144 prompts, ETA ~18:13 UTC 2026-09-16 |
| Fit run state | `/workspace/jlens/qwen3_6_35b/HANDOFF.md` | **read this before using the lens** — window 512 (not 1024), heavy right tail in per-prompt ‖J‖, SE of the mean ~5.8 % at n=144, `--stop-at-delta` inert |
| Lens checkpoint | `/workspace/jlens/qwen3_6_35b/ckpt.pt` (151 MB) | resumable checkpoint; the final artifact will be `Qwen3.6-35B-A3B_gridenv_jacobian_lens.pt` (`jlens_fit_qwen.py:773-774`) |
| Trajectories | `/workspace/trajectories/qwen3.6-35b/replayed_single_step/` | `mass_train_576` (553 JSONs), `mass_eval_144` (52 — **incomplete**), `heldout_72` (72) |

**Everything downstream of the fit is gpt-oss-only today.** The commit added the fit and its
test and nothing else; no gather, prepare, train, eval or analysis script has been pointed at
Qwen yet. The good news is that the blockers are narrow — see §7.

### What the Qwen trajectories already give us for free

Checked against `heldout_72/size5/*.json`: the replay pipeline already normalised them to the
canonical trajectory JSON (`telos_interp/trace_viewer/README.md`). In particular
`output_tokens[i]["token_groups"]` carries `["output", "analysis"]` / `["output", "final"]`
exactly as gpt-oss does (798 analysis tokens of 812 in the step inspected), so
`build_loudness_tables.reasoning_token_positions` (`build_loudness_tables.py:142`) works
**unchanged** — no harmony/`<think>` special-casing needed. `prompt.prompt_prefix_tokens`,
`prompt.prompt_suffix_tokens`, `step.grid_state_tokens`, `agent_action`, `astar_*` are all
present, so `abs_pos` arithmetic is unchanged too.

---

## 1. P2 per-token loudness + loudness-based selection (jlens / logitlens / random)

**Out: 3 selections + 3 activation gathers, for train and eval.**

### One script does all of it

`telos_interp/loudness_analysis/build_loudness_tables.py` (1956 L) — one forward pass per
trajectory emits, per trajectory:

1. `{stem}_{lens}_analysis.csv` — top-20 lens predictions per (reasoning token, layer) with
   `top_{i}_logprob`;
2. `{stem}_{lens}_direction_mass.csv` + `.meta.json` — the wide `(token × layer)` table of
   `log P(any signal word)` over the **whole** vocabulary;
3. `{stem}_jlens_selection.json` — the selection record, format v2, `arms: {name: {config, picks}}`;
4. the selected `.pt` activations, in `gather_activations`' exact tree layout.

**There is no separate "select" step** — `--select-methods` runs the selection inside the same
forward pass. Three arms in one run:

```
--lens both --select-methods jlens,logitlens,random
```

`--lens both` costs one extra unembed per chunk, not a second pass. The `random` arm is the
matched control and **must be drawn in the same run** — once the tree holds only the loud
tokens, a uniform draw over the reasoning chain can never be made again.

### The selection logic

`telos_interp/jlens_utils/` — stdlib only, no torch. Three consumers share one `top_filter`,
which is what makes them agree.

| File | Role |
|---|---|
| `methods.py` | `METHODS` registry: `jlens`, `logitlens`, `random`. `score_artifact_path()` maps a score to the file it reads. `DEFAULT_METHODS = ("jlens", "random")` |
| `scoring.py` | `SCORES` registry: `count`, `logprob_mass`, `logprob_sum`, `logprob_mass_full`. `DEFAULT_SCORE = "count"` — **pass `logprob_mass_full` explicitly** |
| `top_filter.py` | the filter itself (452 L) |
| `record.py` | read/write/merge the selection record; v2 with v1 read-back |
| `jlens_csv.py` | CSV reading, `output_start`, `step_folder_index`, the `abs_pos` ↔ `token_idx` coordinate map |
| `layer_profile.py` | the unbiased per-layer mean, behind `scripts/jlens_layer_profile.py` |

### Recorded invocations to copy from

| Wrapper | What it pins |
|---|---|
| `wrappers/jlens_mass_l15.sh` (123 L) | **the closest template** — `--direction-score logprob_mass_full`, `--select-candidate-layers 15`, layers `7:23` in the CSV, `PER_COMBO`/`SEED` sampling, and `NAMES_FILE` as sole authority when pinning a trajectory set. Its header also records the logitlens twin invocation |
| `wrappers/jlens_reasoning_tokens_filtered.sh` (85 L) | the count-era filtered gather |
| `wrappers/jlens_extend_logitlens.sh` (98 L) | adding a lens arm to an **already pruned** tree: CSV-only pass, gather only missing `.pt`, merge the arm, inherit the control. Dry-run by default |
| `wrappers/jlens_reasoning_tokens.sh` (48 L) | the unfiltered full sweep (superseded, but it is what a no-selection gather looks like — see §2) |

### Decisions this task needs

- **Candidate layers.** The Qwen lens only has `J` at 3, 7, 11, 15, 19, 23, 27, 31, 35 (+39,
  the identity target). `--select-candidate-layers` narrows what the selection ranks *and*
  saves; `--layers` still controls what the CSV and mass table cover, so one run can select a
  single layer and still leave the full profile behind for `scripts/jlens_layer_profile.py`.
  Layer 15 happens to be in Qwen's fitted set, but 15/40 is not the same place in the stack as
  15/24 was on gpt-oss — mid-stack here is ~19. Pick `L` from the layer profile, not by
  analogy. Whatever pool is chosen, `delete_non_jlens_selected.py --candidate-layers` must get
  the same one.
- **Train vs eval as two trees or one.** The mass-era precedent is two *views* over one tree
  (`scripts/build_mass_era_split.sh`, `scripts/verify_mass_era_split.py`), links at the
  trajectory level so the CSV, mass table, `.meta.json` and selection record come along. For
  Qwen the split already exists as three directories on disk, so a single gather per directory
  is simpler than a view — but `scripts/build_activation_view.sh` is there if a name-list view
  is wanted instead.
- `mass_eval_144` holds **52** JSONs, not 144. Either the replay is unfinished or it was
  abandoned; settle that before gathering on it.

---

## 2. Held-out activation gathering — "random sampling or no sampling?"

The answer is **forced by what the held-out eval reads**, and it is *no sampling*.

`telos_interp/loudness_analysis/score_probes_per_token.py` (226 L) is the held-out scorer
(§5), and its docstring states the contract outright — it needs two trees, because the two
artifacts have different layer coverage by design:

```
--activations-dir   a tree gathered with NO --signal-json, so every token has a .pt at
                    --layer (the probe's layer).
--lens-dir          a CSV-only tree (--no-save-activations) over the SAME trajectories,
                    covering every layer, which is where the scores come from.
```

So the held-out gather is **two passes over `heldout_72`**:

1. `build_loudness_tables.py` with **no `--signal-json`** and `--layers <the probe layer>` —
   saves every reasoning token's `.pt` at that one layer, no selection, no record;
2. `build_loudness_tables.py --no-save-activations` over the same 72 with the full
   `--layers` range and `--lens both` — the CSVs and mass tables, no `.pt`.

Both can be the same directory (`--lens-dir` defaults to `--activations-dir`). This is exactly
what `/workspace/activations/heldout360_lens` is on the gpt-oss side, and what
`scripts/eval_belief_arms_heldout.sh` consumes.

Cost note: 72 trajectories × one layer is cheap; 72 × Qwen's ~18 k-token median chain × the
full layer set is **not** — the gpt-oss held-out pass was ~2 h GPU for 87 k individual `.pt`
reads over 360 trajectories with a few-hundred-token median. Qwen chains are ~30× longer.
Sizing this is the one number worth estimating before committing GPU.

A `random` selection arm on held-out data would only make sense if the eval were to be
restricted to a subsample for cost. If it comes to that, draw it with the same
`--select-methods random` machinery so the draw is recorded rather than ad hoc.

---

## 3. P2 probe training (jlens / logitlens / random datasets)

### Prepare

`interp-cli prepare_activations_for_probing --probe-type next_action --token-selection recorded_<arm>`
(`telos_interp/commands/prepare_activations_for_probing/`, with the selection reader at
`jlens_token_selection.py` and the manifest reader at `manifest_loader.py`).

- **Use `recorded_*`, never `<lens>_direction`, on a tree that was gathered with a selection.**
  Re-scoring the CSV would find the right *loud* tokens but the control arm cannot be
  recomputed — on a pruned tree "draw N uniformly" can only draw from the survivors.
- `next_action` manifests are **token-major**: they copy nothing and point into the original
  tree via an absolute `activations_root`. Three arms cost three passes over JSON, not three
  copies of the data.

Recorded invocation: `scripts/prepare_next_action_arms.sh` (125 L) — one dataset per arm,
`ARMS="jlens logitlens random"`, `ACT` / `TRAJ` / `OUT` / `LAYERS` / `COMPLEXITIES` env vars,
and an arm the record does not hold is reported MISSING rather than aborting the others.

### Thin + split

`scripts/split_next_action_manifest.py` (622 L) — this is where the train/eval partition and
the top-K sweep happen, **not** at prepare time:

- `--tokens-per-trajectory K` — identical to having prepared with `--num-tokens K`, which is
  why a top-1/2/3 sweep costs three splits rather than three multi-hour prepares;
- `--layers-per-token M` and `--single-layer L` — pin the whole dataset to one layer;
  `--single-layer best` lets the manifest pick, but that mean is conditional on selection, so
  take `L` from `scripts/jlens_layer_profile.py` instead. **The control arm carries no scores
  and cannot pick — give it the same explicit `L`.**
- `--thin-mode auto|rank|uniform` — **every control arm passes `uniform` explicitly.** The
  old "has a score ⇒ wants ranking" inference has already failed once (layer-15 mass
  −3.402 → −2.900, a shift as large as the jlens arm's own ranking, and nothing raised);
- `--eval-names FILE` — pins the eval set to a name list, which is how arms prepared from
  different trees are scored on the same test trajectories;
- the strata are computed from the **unthinned** samples, so every K shares one split.

### Train

`interp-cli train_next_action_probe` (`telos_interp/commands/train_next_action_probe/`,
[README](telos_interp/commands/train_next_action_probe/README.md)) — `lr` or `mlp`,
`train_data_path` / `eval_data_path`, `num_epochs`, `learning_rate`, `balance_classes`,
`normalize`, `class_weight`, `seed`. Each checkpoint's `results` block already holds pass 1 —
the probe on the tokens its **own** selection picked.

Recorded invocation: `scripts/train_next_action_arms.sh` (179 L) — the arm × top-K × {lr, mlp}
sweep, consuming `${PREPARED}_${arm}`. Its header is the argument for why three arms exist and
what each knob does; read it before changing anything.

**Landmine, guarded:** `train_cognitive_map_probe_fn.py::_prepare_train_eval_v3` splits with
`torch.randperm` over *entries*, which leaks on a token-major manifest. It now refuses an
internal `--eval-split` and `--subset < 1.0` there and points at `split_next_action_manifest.py`.
Do not work around it — split over unique trajectory names.

---

## 4. P2 probe eval, within-distribution

Two populations, and **they are not comparable** (`scripts/eval_belief_arms_heldout.sh` header,
ICLR entry 49): on its own loud tokens the jlens arm wins and the random control is worst; on
every token of a disjoint set the ordering **inverts**. Name the population in every number.

- **Pass 1 — each probe on the tokens its own selection picked.** Already written into every
  checkpoint's `results` block by `train_next_action_probe`; nothing extra to run.
- **Pass 2 on the in-distribution eval split** — point `score_probes_per_token.py` (§5) at the
  eval half, or use the `eval_data_path` the split produced.
- `telos_interp/loudness_analysis/summarise_probe_accuracy.py` — one row per probe, balanced
  accuracy vs belief and vs final. A scoreboard, not a loudness analysis.
- `telos_interp/loudness_analysis/stats.py` — `bal_acc` (row-averaged) and
  `bal_acc_from_counts` (count-pooled) are **not the same number**; `run_config.json` records
  which ran.

There is no `eval_next_action_probe` subcommand — the CLI's eval subcommands are all
cognitive-map / distance. For `next_action`, evaluation is the checkpoint's own `results`
block plus `score_probes_per_token.py`.

---

## 5. P2 probe eval, held-out

`telos_interp/loudness_analysis/score_probes_per_token.py` — every probe on **every** reasoning
token of a trajectory set, one row per (trajectory, step, token), `--probe` repeatable, and
`--probe-type {next_action, grid_tile}` off the `probes.PROBE_TYPES` registry.

**Its output is already a loudness table** — beside each probe's verdict it writes, per lens,
the top-k count and the full-vocabulary mass at `--layer` and at the token's own best layer.
That is why there is no "join loudness onto probe predictions" script.

Recorded invocation: `scripts/eval_belief_arms_heldout.sh` (148 L), a three-step round
(GPU score → two CPU joins). Its shape is what to copy: every earlier round's probes are
re-scored in the same pass as a regression check and as the before-picture.

Related, if a rollout arm is ever wanted for Qwen (not in this task list):
`telos_interp/loudness_analysis/join_rollouts.py` adds what the rollout knows to the same
table. `rollouts/run_inference.py` and `truncation_strategies.py` are **gpt-oss-specific** —
`run_inference.py` appends the fixed harmony final-channel prefix
(`<|end|>...{\n  "action": "`), which Qwen does not use.

---

## 6. Decodability by loudness — the two graphs

### The analyser

`telos_interp/loudness_analysis/analysis/probe_accuracy_by_loudness.py` (≈650 L). It bins a
loudness column into `--deciles` (default 10) and reports balanced accuracy per bin with a
**trajectory-clustered** bootstrap (`--boot`, default 300 — tokens inside a trajectory share a
sentence structure, a commitment boundary and a label).

Relevant flags: `--per-token`, `--out`, `--probe-type`, `--lens`, `--signal-name`, `--layer`,
`--loudness-column`, `--deciles`, `--boot`, `--probes-filter`, `--extra-probes`,
`--extra-probes-rowset`, `--reference-gap`.

### Graph 2 — direction words removed — is a flag, not a script

```
--exclude-signal-words --exclude-radius N --signal-words <vocab.json>
```

It drops signal-word rows and produces the **same** tables from what is left.
`--exclude-radius` matters and should not be left at 0: the lens predicts the *next* tokens,
so the token just before ` up` is loud without being a signal word itself. Radius is what
separates "the residual is signal-loaded here" from "a signal word is about to be written".
The same pair of flags exists on `analysis/loudness_distribution.py`.

### The figures

`telos_interp/loudness_analysis/plotting/figures.py` — figures behind one `FIGURES` registry,
in four families selected by which table is passed (`--probe-table`, `--distribution-table`,
`--rollout-table`, `--grid-table`). `--list` prints them.

**Mind which copy you are reading.** The committed file (1,165 L) holds 16 figures; the main
checkout's **uncommitted** working-tree copy (1,544 L) holds 19 — the three extra are the whole
`grid` family, and so are `grid_decile_curve` and `draw_grid`. Line numbers below are the
working-tree copy's. See §10.

For "all probes in one graph vs loudness deciles" there are two existing candidates, and
**neither is exactly it**:

| Figure | Family | Shape | Fit |
|---|---|---|---|
| `accuracy_by_bin` (`fig_by_bin`, committed at line 105 / working tree 112) | `probe` | one axes, **one line per probe**, two panels (vs local belief / vs final action) | right shape, wrong plumbing — needs `label_local` **and** `label_final` columns, i.e. a `join_rollouts.py` table, and its `COLOR` / `LABEL` / `PROBES` dicts are keyed to the entry-48 probe names (`p1_lr`, `p2_mlp`, …). `--extra-probes` extends it without editing the registry |
| `grid_accuracy_by_loudness` (working tree line 866, **uncommitted**) | `grid` | **one panel per probe**, every ruler overlaid on each | right plumbing (reads a `score_probes_per_token.py` table directly, no rollout needed) but the wrong split — and `draw_grid` (line 1034) hard-codes `get_probe_type("grid_tile")` and requires `n_true_{class}` columns, so it will not accept a `next_action` table |

**This is the one genuinely missing piece in the whole list.** A `next_action` counts-mode
figure that puts every probe on one axes against loudness decile does not exist. The
ingredients do: `grid_decile_curve` (working tree line 840) is already generic over probe key,
score column and classes, and `stats.qbin` / `stats.bal_acc_from_counts` are shared. It is a
small figure function plus a registry entry, not new analysis — but it builds on uncommitted
code, so commit that first or the new figure has no base.

### The recorded end-to-end invocation

`wrappers/loudness_report.sh` — runs both analysers and the figure CLI for **one ruler**, with
`LENS`, `SIGNAL`, `LAYER`, `PROBE_TYPE`, `PROBE_TABLE`, `DIST_TABLE`, `OUT`,
`EXCLUDE_SIGNAL_WORDS`, `EXCLUDE_RADIUS`. Both graphs are two runs of it:

```bash
LENS=jlens bash wrappers/loudness_report.sh
EXCLUDE_SIGNAL_WORDS=1 EXCLUDE_RADIUS=2 LENS=jlens bash wrappers/loudness_report.sh
```

**One ruler per run, and the output folder carries its name.** `provenance.RunConfig.guard`
fails a second run with a different `--lens` into the same `--out` rather than overwriting half
the figures. Run it once per lens and put the two side by side.

Supporting pieces: `columns.py` (`{lens}_{signal}_logmass_L{layer}`, every legacy read-alias,
and the one axis label every table and figure uses), `provenance.py` (`run_config.json` with
the ruler, the vocabulary and a hash of its *contents*, bin edges, bootstrap seed, and row
counts before and after each filter), `lens_io.check_vocabulary` (the `.meta.json` sidecar
rule — never read a mass table without it).

---

## 7. The gpt-oss-specific blockers, in full

Every one of these is in `build_loudness_tables.py` or the helper it imports. None is deep;
all of them are silent-wrong or hard-fail if missed.

| # | Where | Problem | Note |
|---|---|---|---|
| 1 | `build_loudness_tables.py:1562` | `model_id` is read from the trajectory JSON's `model_params.model_id`, which for these files is **`"gsarti/qwen3.6-35b"`** — the Together AI serving name. `AutoModelForCausalLM.from_pretrained` on it will fail. There is **no `--model-id` flag** (confirmed against the full flag list) | the fit script uses `Qwen/Qwen3.6-35B-A3B`; the gather needs the same override |
| 2 | `build_loudness_tables.py:770` | lens filename hard-coded `gpt-oss-20b_jacobian_lens.pt` | the Qwen fit writes `Qwen3.6-35B-A3B_gridenv_jacobian_lens.pt` (`jlens_fit_qwen.py:773-774`) |
| 3 | `build_loudness_tables.py:135` | `TARGET_LAYER = 23` — the layer where the lens is the identity and `J` is skipped | Qwen's target is **39** |
| 4 | `scripts/jlens_action_ranks.py:53-76` (`ensure_unembed_assets`, imported at `build_loudness_tables.py:1229`) | caches `gpt-oss-20b_unembed.pt` under a hard-coded name, from a module-level `MODEL_ID`, reading the tensor keys `lm_head.weight` and `model.norm.weight` | Qwen3.6 is `Qwen3_5MoeForConditionalGeneration`, a VLM wrapper around the text stack — the weight keys are very likely **not** at those paths, and `tie_word_embeddings` needs checking. Get this wrong and the first run re-downloads a shard to rebuild a cache under the wrong name |
| 5 | `build_loudness_tables.py:1566` | `device_map` — `device_map="auto"` across multiple GPUs produces NaNs for the gpt-oss MoE. Qwen3.6 is also MoE, 256 experts | assume single-GPU until proven otherwise |
| 6 | `pyproject.toml` | the Qwen commit moved the transformers pin 4.42 → **5.17** and holds torch < 2.10, and dropped the unused `vllm` extra | the `gpu` extra (`accelerate`, `kernels==0.12.0`) is still required and is **not** in the default deps — a fresh worktree venv dies at `from_pretrained` |
| 7 | `NAME_RE` (`build_loudness_tables.py:139`) | `size(\d+)_comp([\d.]+)_(\d+)` | matches `together_ai_gsarti_qwen3_6-35b_size11_comp0.4_960` fine — **no change needed**, noted because it is the kind of thing that silently yields `{"size": "", …}` |

### The direction vocabulary — smaller problem than it looks

`data/jlens/direction_tokens_full.json` was built from the **gpt-oss-20b** vocabulary
(`admissible()`, cell 2 of `notebooks/direction_tokens.ipynb`, filters to real gpt-oss tokens).
Qwen's tokenizer is entirely different (`<|im_end|>` is id 248046).

It still mostly transfers, because the vocabulary is stored as **decoded strings**, not ids:

- the top-k scores (`count`, `logprob_mass`, `logprob_sum`) string-match the CSV's `top_i`
  columns and are tokenizer-agnostic;
- the mass table re-resolves the strings through the *run's* tokenizer at gather time —
  `build_direction_mass_columns` (`build_loudness_tables.py:325`) → `resolve_direction_ids`,
  which drops any entry that is not a single round-tripping token and **prints what it
  dropped**.

So the honest options are (a) reuse the gpt-oss vocabulary and read the dropped count as a
cost, or (b) rebuild it for Qwen with `notebooks/direction_tokens.ipynb`. If (b): build it
**from the model vocabulary alone**, never from what is frequent in the j-space — the j-space
is what it is used to measure. Either way the `.meta.json` sidecar records a hash of the
vocabulary's *contents*, so the two can never be silently compared.

---

## 8. Coordinate and naming traps that apply unchanged

Carried over verbatim from the gpt-oss line; all of them are silent.

- **`abs_pos` is prompt-inclusive; `token_idx` is `output_tokens`-relative.** Joining a rollout
  or a probe table on `abs_pos` yields an empty or wrong join, never an error.
- **`sentence_idx` is not the sentence** — it is the cutoff's ordinal in the eval list.
  `cut_sentence_idx` is the sentence the cut lands in, and it is what any per-sentence
  grouping, pairing or dedupe must key on. They coincide for `eos` alone, so a mix-up passes
  an eos spot-check and is wrong everywhere else.
- **`frac_in_sentence` vs `frac_of_chain`** — `sentence_frac` means different things in two
  tables; `columns.py` is the canonical naming.
- **Read lens CSVs with `csv.DictReader`, never `pandas.read_csv`** — decoded tokens include
  the literal `"NA"`, empty strings, embedded commas and newlines.
- **Loudness is never unqualified.** At layer 15 the two lenses' top-20 sets overlap only about
  half. The Qwen `HANDOFF.md` predicts jlens/logitlens divergence **deep** in the stack rather
  than near the top (cos(J, I) 0.14 at layer 3 → 0.86 at layer 35), which is a testable
  prediction and a reason to profile layers before pinning one.
- **Sizes on `/workspace` must be measured, never `du`'d** — MooseFS over-reports by 5–40× for
  a tree of millions of ~7 KB `.pt` files.

---

## 9. Not covered by existing code

1. The `next_action` "all probes on one axes vs loudness decile" figure (§6) — a new figure
   function plus a registry entry.
2. A Qwen entry point for `build_loudness_tables.py` (§7 items 1–4) — either flags
   (`--model-id`, `--jlens-file`, `--target-layer`) or a thin sibling, matching how
   `jlens_fit_qwen.py` sat beside `jlens_fit_gpt_oss.py` rather than editing it.
3. A Qwen `wrappers/*.sh` record for the gather, the prepare/train arms and the report. The
   wrappers are the record of a published run's parameters, so these are new files, never
   edits to the gpt-oss ones.
4. `mass_eval_144` is incomplete (52/144).

---

## 10. Files referenced here that are uncommitted on `reasoning_theatre`

This document sits on a worktree branch off commit 75e88b7. Two things it mentions exist only
as **uncommitted work in the main checkout** and are therefore not on this branch:

- `telos_interp/loudness_analysis/join_signal_loudness.py` — adds a *second* vocabulary's
  loudness to a table that already has one, with no GPU. This is what makes a double
  dissociation sayable (same probe columns, two rulers), and it is documented in the
  `loudness_analysis` README as part of the pipeline.
- `wrappers/all_probes_train.sh`.

And the modification that matters most for §6: **the entire `grid` figure family in
`plotting/figures.py` is uncommitted.** Committed = 1,165 L / 16 figures; working tree =
1,544 L / 19 figures. The three extra (`grid_accuracy_by_loudness`,
`grid_per_class_by_loudness`, `grid_ruler_gaps`) and the helpers `grid_decile_curve`,
`grid_probes`, `grid_rulers` and `draw_grid` exist only in the main checkout's working tree.
`grid_decile_curve` is the reusable ingredient §6 recommends building the missing figure on,
so it has to be committed before anything depends on it.

Also uncommitted in the main checkout: the binary-cognitive-map commands, the boundary round
(`scripts/boundary_cognitive_maps.py`), and edits to `cli.py`,
`analyze_truncation_strategies.py`, `plot_loud_vs_sentence_end.py`,
`delete_non_jlens_selected.py`, the two loudness test files and several READMEs. Check
`git status` on the main checkout before assuming a file's committed state.
