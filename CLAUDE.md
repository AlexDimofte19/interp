# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Interpretability of GPT-OSS-20B acting as an agent in 2D grid worlds (SPAR-Telos / project-telos). The
model's trajectories are stored as JSON, residual-stream activations are extracted from them, and linear
/ MLP **probes** are trained to decode grid cell identity (`grid_tile` / "cognitive map"), A* distance,
action sequences, and the next action. A second, newer line of work applies a **Jacobian lens (jlens)** to
reasoning tokens to decide *which* tokens and layers a probe should be trained on.

## Environment and commands

The project uses **uv**; the interpreter is `.venv/bin/python`. Test/lint tooling lives in the `lint`
extra, which is *not* installed into `.venv`, so `make test` (and `.venv/bin/python -m pytest`) fails with
"No module named pytest". Run tests through uv instead:

```bash
uv run --extra lint pytest -c pyproject.toml -q --ignore=tests/test_trajectory_activations.py   # 301 pass
uv run --extra lint pytest -c pyproject.toml tests/test_grid_utils.py::test_name -vv             # single test
make fix-style     # ruff format + ruff check --fix (line length 119, notebooks included)
make check-style
```

`tests/test_trajectory_activations.py` fails at import — it imports
`gather_activations.trajectory_activations`, a module that has never existed here. Pre-existing and
unrelated; always ignore it. Note `--doctest-modules` is on, so docstring examples are executed.

`claude_session_readme.md` says to run everything through `conda run -n interp`. That is true only on the
Linux GPU hosts; on this Mac there is no conda.

**GPU-only work.** Anything that loads gpt-oss-20b — `gather_activations`, `scripts/jlens_*.py`,
`jlens/jlens_fit_gpt_oss.py`, `scripts/inference_oss/run_inference.py` — cannot run on the laptop. Locally
you can only work on the code paths that consume already-extracted artifacts (CSVs, `.pt` files, manifests,
notebooks). Extract activations on a **single** GPU: `device_map="auto"` across multiple GPUs produces NaNs
for this MoE model.

Loading the model also needs the **`gpu` extra** (`accelerate`, and the pinned `kernels==0.12.0`), which
is *not* in the default dependencies: a long-lived `.venv` usually has `accelerate` already and never
notices, but a fresh worktree venv dies at `from_pretrained` with "requires `accelerate`".
`run_inference_strategies.sh` passes `--extra gpu` by default (`UV_EXTRAS` overrides); anything else
needs it spelled out.

**GPU-host paths.** `--jlens_dir` is **`/workspace/jlens/gridenv`**, not `/workspace/jlens`. It holds
`gpt-oss-20b_jacobian_lens.pt` *and* the cached `gpt-oss-20b_unembed.pt` (lm_head + final norm), so it is
required even for `--lens logitlens`, which never reads the Jacobian — get it wrong and the first run
re-downloads a 4.2 GB shard to rebuild the unembed cache. The direction vocabulary is one level **up**, at
`/workspace/jlens/direction_tokens_full.json` — the deployed copy of the repo's
`data/jlens/direction_tokens_full.json`, which is where the notebooks write it. The `scripts/*.sh`
defaults encode this layout; pass `JLENS_DIR` / `SIGNAL_JSON` to override.

```
/workspace/jlens/
    direction_tokens_full.json        <- --signal-json / --direction-tokens-path
    gridenv/                          <- --jlens_dir
        gpt-oss-20b_jacobian_lens.pt
        gpt-oss-20b_unembed.pt        (built on first run, reused after)
```

## Architecture

### CLI

`interp-cli` (`telos_interp/commands/cli.py`) is a **tyro** `subcommand_cli_from_dict`. The function
signature *is* the CLI — adding a parameter to a `*_fn.py` function adds a flag, no wiring needed. Each
subcommand is a directory `telos_interp/commands/<name>/` holding `<name>_fn.py`, an `__init__.py` that
re-exports the entry function, and a `README.md` with the full option table. Start from
[telos_interp/commands/README.md](telos_interp/commands/README.md) for the pipeline overview.

```
trajectory JSONs → gather_activations → per-token .pt tree
                 → prepare_activations_for_probing → manifest.json (v3) dataset
                 → train_{cognitive_map,distance,next_action}_probe → probe .pt
                 → eval_* / apply_cognitive_map_probe (writes predictions back into trajectory JSONs)
```

### Shared library (`telos_interp/`)

- `activation_loading.py` — discovery of the activation folder tree and `parse_index_specification`,
  the shared grammar for `--layers` / `--steps` / `--*-indices`: `"all"`, `"7"`, `"7,15"`, `"0:10"`
  (inclusive), `"-1"`, `"-3:-1"`.
- `grid_utils.py` — `CELL_SYMBOL_TO_ID` (`A #  G _ D K ? +`) and `parse_grid_state`, which turns the
  grid row strings into `[row, col, cell_id]` triples with optional padding.
- `probe_models.py` / `training.py` — the LR/MLP classification and regression probes, plus
  train-epoch, seeding, and normalization helpers used by all trainers.

### On-disk contracts (do not break these)

**Trajectory JSON** is the canonical data format; it is fully specified in
[telos_interp/trace_viewer/README.md](telos_interp/trace_viewer/README.md) (`grid_params`, `model_params`,
`prompt`, `steps[]` with `grid_state`, `*_tokens`, `agent_action`, `output_tokens`, and the `probes`
objects that `apply_cognitive_map_probe` writes back). That README also documents the jlens fork viewer.

**Activation tree** (written by `gather_activations`, and byte-compatibly by
`scripts/jlens_reasoning_tokens.py`):

```
{out}/{trajectory_name}/{model}/layer_{N}/step_{M}/{prompt_prefix|prompt_suffix|grid_state|output}/{token_idx}.pt
```

One `(D,)` tensor per token; existing folders are skipped so runs resume. Token index is *category
relative*, which matters when mapping a jlens CSV's absolute `abs_pos` back to a filename (see
`claude_session_readme.md` for the exact formula).

**Prepared dataset** is a directory with `manifest.json` (`format_version: 3`) plus `activations/*.pt`,
with `act_path` relative to the manifest so the directory is movable. **Token-major** manifests are the
exception: they copy nothing and their entries point into the original activations tree via an absolute
`activations_root`. That is always `next_action` (key `samples`), and `grid_tile` (key `trajectories`)
whenever it is given a `--token-selection` or `--token-major` — there one entry is one (token, layer),
a trajectory name repeats across entries, and the per-cell payload lives once per (trajectory, step)
under `cells`, keyed by each entry's `cells_key`. Loaders live in
`prepare_activations_for_probing/manifest_loader.py`.

### The lens line (jlens and logitlens)

`scripts/jlens_reasoning_tokens.py` does one forward pass per trajectory and emits both the activations
and a per-trajectory `{stem}_{lens}_analysis.csv` of top-20 lens predictions per (reasoning token, layer),
each with its `top_{i}_logprob`. Tokens are scored by how direction-loaded those predictions are, against a
vocabulary JSON (`data/jlens/direction_tokens_full.json`, `data/jlens/grid_tokens_full.json` in the repo;
deployed to `/workspace/jlens/` on the GPU host — see **GPU-host paths** above).

**How a token scores is a second registry**, `jlens_utils/scoring.py`, dispatched by name exactly as the
methods are: `--direction-score count|logprob_mass|logprob_sum|logprob_mass_full`. `count` is the original
(how many top-k predictions are direction words) and is the default so nothing already on disk changes
meaning. A mass score is the count weighted by belief — logsumexp of the matched logprobs, i.e. log of the
total probability the lens puts on direction words — and is the one to reach for. `logprob_sum` adds the
logprobs literally, which *multiplies* the probabilities and so scores a token **worse** for emitting
several direction words; it exists because it is the literal reading, not because it is right. Every mode is
"higher is better", so every ranker sorts by `-score` without knowing which ran. A row with no hit floors at
`NO_MATCH_LOGPROB` (-40), never 0 — 0 is `log(p=1)`, the best score there is. A per-layer cell is a real
log-probability (≤ 0); a token's cross-layer *total* adds probabilities across layers and can exceed 0 — it
orders tokens, it is not `log P(anything)`.

**Two artifacts, and `source` picks between them.** `logprob_mass` scores the analysis CSV's `top_i_logprob`
columns, so it only sees direction words that reached the top 20. `logprob_mass_full` reads the
**direction-mass table**, a wide `(token x layer)` CSV the gather writes beside the analysis CSV whose cells
are `log P(any direction word)` over the *whole* vocabulary — computed while the `[b, vocab]` logits are
still on the device, so it costs one gather plus one reduction against logits the unembed already paid for.
The trade is late binding: a top-k score can be recomputed against a different vocabulary from the CSV
alone, a mass table is baked at gather time. Since this repo points two vocabularies at the same trees,
every table is written with a `.meta.json` sidecar naming the one that produced it — **never read a mass
table without it**. `methods.score_artifact_path` is the single place that maps a score to its file.
`--direction-mass-json` (defaults to `--signal-json`) chooses the vocabulary, `--no-direction-mass` skips
the table, and `--no-top-logprobs` drops the 20 logprob columns for runs that will only ever score the mass
table.

The top-k logprob modes need the `top_{i}_logprob` columns, which only runs from this version emit;
`read_direction_scores` raises on an older CSV rather than scoring every row as a tie. **Every CSV and every
selection record currently on disk is pre-logprob**, i.e. a count, and no tree has a mass table yet.
Re-emitting is a CSV-only pass (`--overwrite --no-save-activations`) that writes no `.pt` and so leaves the
pruned tree alone; it produces the mass table too. The mode a record's numbers are in lives in that arm's
`config.direction_score`, and its absence means `count`.

`--lens jlens|logitlens|both`. The two lenses are **one code path**: `apply_lens_transport` with an empty
`J_rows` *is* the logit lens (unembed each layer where it sits), and layer 23 is that case already, which
is why the jlens is the identity there. `--lens both` emits both CSVs from one forward pass — the extra
cost is a second unembed per chunk, not a second pass. Only the jlens needs `gpt-oss-20b_jacobian_lens.pt`,
and only it is restricted to layers that have a fitted `J`; the logit lens covers every requested layer.

**Selection logic lives in `telos_interp/jlens_utils/`** — stdlib only, no torch, so the scripts that
decide what lands on disk can import it without the model stack. It is **method-dispatched**: `jlens`,
`logitlens` and `random` are entries in `methods.METHODS`, and adding one is a registry entry rather than
a branch in four files. See its README. Three consumers share one `top_filter`, which is what makes them
agree:

- `jlens_reasoning_tokens.py --signal-json --select-methods ...` saves *only* the selection (~75x less
  disk than saving every (token, layer), which is what filled the volume up).
- `scripts/delete_non_jlens_selected.py` applies the same filter negatively to trees already gathered in
  full. Dry-run by default; `tests/test_delete_non_jlens_selected.py` asserts prune(full) == filtered
  gather byte-for-byte.
- `prepare_activations_for_probing --token-selection recorded_<method>` reads the
  `{stem}_jlens_selection.json` those two write. `<lens>_direction`/`random` still score the CSV directly,
  but only make sense on an **unpruned** tree. Both mode families are generated from the registry.

The filter keeps one arm per method. The unscored one is not optional: after pruning, "draw N tokens
uniformly" can only draw from the survivors, so the matched control has to be reserved before the rest is
deleted, and the record is the only place it survives. Its entries deliberately carry **no** direction
counts — `scripts/split_next_action_manifest.py` keys off that absence to sample rather than rank, and a
control that recorded counts would silently collapse onto its lowest layer. `arm_seed()` freezes its draw
formula for the same reason: the tree was pruned to controls drawn with it. Layer 15 is force-kept for
every selected token of every arm so a fixed-layer baseline stays available (`--layers 15`) without
another gather — it is a guarantee, not a method.

**The tree is already pruned.** Everything outside the jlens ∪ random selection is gone, so a new lens arm
cannot be recovered by re-filtering — the tokens it would pick were deleted. `delete_non_jlens_selected.py`
refuses to widen an existing selection for exactly that reason. The path that works is
`jlens_reasoning_tokens.py --extend` (wrapped by `scripts/jlens_extend_logitlens.sh`, dry-run by default):
one CSV-only forward pass for the new lens, gather only the `.pt` files not already present, and **merge**
the arm into the record. Arms already recorded keep their picks and config verbatim; the control is
inherited, never redrawn, because a fresh draw could only sample the survivors. The selection record is
format v2 (`arms: {name: {config, picks}}`) and still reads v1 — it must, since every pruned trajectory
has one.

**One layer, not one per token.** `--layers-per-token 1` keeps each token's *own* best layer, so the
dataset still spans many layers and one probe weight vector is asked to read several representation spaces.
`split_next_action_manifest.py --single-layer L` pins the whole dataset to one layer instead (`SINGLE_LAYER`
in the `train_*_arms.sh` pair); `--single-layer best` lets the manifest pick its argmax, but that mean is
conditional on selection — a layer appears in a manifest only where it won, except layer 15 which is
force-kept for every token. `scripts/jlens_layer_profile.py` computes the unbiased mean per layer from the
CSVs (or, with `--direction-score logprob_mass_full`, from the mass tables — unbiased over the vocabulary
as well as over layers), where every token is scored at every layer, and that is where `L` should come
from. A control arm
carries no scores and cannot pick: give it the same explicit `L`.

Or pin the layer at *gather* time, which is stronger: `jlens_reasoning_tokens.py
--select-candidate-layers 15` narrows the pool the selection ranks and saves from, so tokens are
ranked by their layer-15 score rather than by a cross-layer total, and only layer 15 lands on disk.
It does **not** narrow the CSV or the mass table — those still cover `--layers` — so one run can
select a single layer and still leave the full `(token x layer)` profile behind for
`jlens_layer_profile.py`. It mirrors `--candidate-layers` on `delete_non_jlens_selected.py`, and the
two must be given the same pool or a prune keeps different files than the filtered gather wrote; the
value is recorded in each arm's `config.candidate_layers` (absent/`null` = every layer the artifact
covers). `scripts/jlens_mass_l15.sh` is the recorded invocation: `logprob_mass_full` ranking at layer
15 over the same 3600 trajectories the count-era tree drew.

Narrow at training time, not prepare time: `split_next_action_manifest.py --tokens-per-trajectory K
--layers-per-token M` takes top-K off one prepared dataset (identical to a `--num-tokens K` prepare),
which is why the top-1/2/3 sweep costs three splits rather than three multi-hour prepares. It computes
the train/eval strata from the unthinned samples so every K shares one split and the results compare.
It takes either token-major manifest despite the name; a `grid_tile` one has no scalar label, so it
strata by grid size instead. `--eval-names FILE` pins the eval set to a name list, which is how arms
prepared from different trees end up scored on the same test trajectories.

**The same arms with the grid label.** `scripts/prepare_grid_arms.sh` and `scripts/train_grid_arms.sh`
are the `grid_tile` twins of the `*_next_action_arms.sh` pair: same tree, same records, same tokens,
same layers, `train_cognitive_map_probe` instead of `train_next_action_probe`. Any difference between a
grid arm and an action arm is therefore the label and nothing else. Two knobs matter there. `MAX_CELLS`
decides affordability — padded to the widest grid every trajectory has 225 cells, and ~72k entries x 225
is 16.2M rows per epoch, so the default caps each (trajectory, step) at 25 class-balanced cells. And the
cell draw is seeded per (trajectory, step), never from the global RNG: arms consume different numbers of
draws, so a shared stream would hand them different cells and the arms would stop differing only in
their tokens.

Those vocabularies are built by `notebooks/direction_tokens.ipynb` and `notebooks/grid_tokens.ipynb`
**from the model vocabulary alone** — never from what is frequent in the j-space, since the j-space is
what they are used to measure. Keep that separation. That covers the scoring inputs too: every seed and
anchor must itself be a gpt-oss-20b token (`admissible()`, cell 2 of both notebooks, which filters the
literal lists at run time and prints what it dropped), and there is no system word list any more — if the
lexical stage admits junk, drop the seed that produced it rather than adding an outside filter. MiniLM is
still the similarity engine, but it only scores strings, it never supplies one. Any `<lens>_direction` run
needs a matched `random` run with the same N/M/seed to mean anything: the top-scoring tokens are literally
the direction words the model has already verbalized.

Read lens CSVs with `csv.DictReader`, never `pandas.read_csv` — decoded tokens include `"NA"`, empty
strings, embedded commas and newlines, which pandas' NA handling silently corrupts.

### The rollout line (truncation strategies)

The lens line asks what a *probe* reads off a token. The rollout line asks what the **model** would answer
if its reasoning stopped there. `scripts/inference_oss/run_inference.py` keeps `output_tokens[:pos + 1]`,
appends the fixed final-channel prefix (`<|end|>...{\n  "action": "`), and reads the single action token the
model then emits. That label is the **local belief** at `pos`, as against the trajectory's `agent_action`,
which is where it *ends up*; the two coming apart before the model commits is the whole point of entries
39-41 and 46-48.

**Where to cut is a third registry**, `scripts/inference_oss/truncation_strategies.py`
(`STRATEGIES` / `build_strategy`), dispatched by name exactly as the methods and scores are. Nothing else
about the rollout changes between arms, so any difference between two arms is the cut points and nothing
else. `--strategy`:

- `eos` — every reasoning sentence end. **The grid every rollout on disk was measured on**; its positions
  are pinned by a test against `reasoning_eos_positions` itself, because entry 42's whole loudness join is
  indexed by them.
- `jlens_argmax_per_sentence` — one cutoff per sentence, at its *loudest* token. Same count and same
  sentence grid as `eos`; only the position inside each sentence moves.
- `jlens_top_k_global` — the `--top-k` loudest tokens of the whole chain, so the grid follows the lens
  rather than the punctuation. A quiet sentence contributes nothing, a loud one several.
- `every_token` — no selection at all: every reasoning token, or every `--stride`-th. The dense grid, and
  the only one whose cuts do not depend on the lens.
- `recorded_selection` — replay an existing gather's picks: the `token_idx` values in
  `arms[--selection-arm]` of `{stem}_jlens_selection.json`. The other four choose cut points from the
  trajectory in front of them; this one reads a decision already made. It exists because a control arm's
  seeded uniform draw is taken **before** the tree is pruned and can never be re-made (same reason
  `--extend` inherits the control rather than redrawing it), so an arm that must label the *same* tokens an
  existing probe trained on has to read them. `--selection-root` defaults to `--lens-root`, since one
  gather writes the record beside the mass table it ranked with. Like `every_token` it records loudness per
  cutoff without ranking on it.

**Two class attributes, not one.** `uses_loudness` decides whether `build_strategy` wires a
`MassTableLoudness`; `needs_loudness` decides whether a trajectory without a usable mass table is fatal.
They differ for `every_token` and `recorded_selection`, which **record** loudness per cutoff but never rank
on it, so a missing table costs them the covariate rather than the trajectory. Do not collapse them back
into one flag. `recorded_selection` raises `SelectionUnavailable` (a `LoudnessUnavailable` subclass, so
`run_inference`'s single `except` still skips the trajectory) when the record or the arm is missing, and
also when a pick is not an analysis token — that last one catches an `abs_pos` handed in where a
`token_idx` belongs, which would otherwise join to nothing in silence.

**Every strategy writes the `eos` schema**, so `analysis.py` and the join scripts read all five unchanged:
`sentence_evals` is the ordered cutoff list, `eos_token_pos` the cut position **in `output_tokens`
coordinates**, and `n_reasoning_sentences` the number of cutoffs (a misnomer under the loud strategies;
`n_cutoffs` is the same number under an honest name). Each `Cutoff` is also placed on the `eos` grid
(`cut_sentence_idx`, `pos_in_sentence`, `sentence_len`) whether or not the strategy used it, which is what
lets two arms be compared in the same sentence coordinates. The endpoint cutoffs (`no_reasoning`,
`end_of_reasoning`) are added to every arm on purpose so its first and last eval is the same prompt as
every other arm's — `--no-endpoint-cutoffs` drops them and makes final accuracy and the commitment indices
incomparable across arms.

**Coordinate traps, both silent.** `abs_pos` in the lens CSVs and the per-token joins is
**prompt-inclusive**; the `output_tokens` index — what `eos_token_pos` and the probe-loudness `token_id`
mean — is **`token_idx`**. Joining a rollout on `abs_pos` yields an empty or wrong join, never an error.
And in `probe_vs_rollout/per_token.csv`, position *within* a sentence is `frac_in_sentence`; that file's
own `sentence_frac` is `sentence_idx / n_sentences`, a different quantity, and reading the wrong one
quietly destroys any loudness-vs-position control.

**A rollout label has a noise floor.** The same prompt re-run in a different batch shape agrees 100% at
`end_of_reasoning` but only ~80% at `no_reasoning` (mean |Δp| .105) — batching and padding move
low-confidence logits. See `rollout_strategies/RUN_STATE.md`. Do not read a few points of difference
between arms as signal without checking it against that floor.

## Conventions and gotchas

- `configs/**/*.conf`, `script.sh`, `general_probe_train.sh`, `*.ps1` are **recorded `interp-cli`
  invocations**, not parsed config files. They are the record of how published results were produced.
- `ICLR log.txt` is the running research log (findings, planned phases, known landmines) and
  `claude_session_readme.md` is the handoff note for the current branch — `worktree-probe-loudness` as of
  entry 48, forked from `reasoning_theatre`. Read them before touching the jlens → probe or rollout path;
  **append** to the log rather than rewriting it, and keep the readme's header pointer current, since it is
  the first thing a fresh session reads.
- Landmine, now guarded: `train_cognitive_map_probe_fn.py::_prepare_train_eval_v3` splits train/eval
  with `torch.randperm` over **entries**. That is a split by trajectory only while one entry is one
  trajectory; a token-major manifest leaks. It now refuses both an internal `--eval-split` and
  `--subset < 1.0` on such a manifest and points at `split_next_action_manifest.py`. Do not replace that
  guard with a row-level split — split over unique trajectory names.
- `parse_grid_state` has no symbol for fog (`*`), so fogged cells are silently dropped and padded. Fine
  while every trajectory is `fully_observable: true`.
- `spaces/trace-viewer` is a git submodule (a HuggingFace Space).
- Pre-commit (ruff format + ruff, trailing whitespace, 2500 KB file size cap) runs on commit and push.
