# Session context: jlens → probe pipeline

Notes for a fresh Claude session picking up this work. Branch: `reasoning_theatre`.

**Newest first, if you only read one thing:** the file is append-only and chronological, so
the last section is the current state. As of 2026-08-31 that is *Truncation strategies —
cutting where the lens is loud* (log entry 43): three rollout arms built and verified, none
of them run yet. Before it: *Loudness through a sentence* (entry 42) and *Probe vs. rollout*
(entries 39-41). The "Resuming — start here" section below is the round-2 grid-probing thread
and is older than all three.

Covers two connected changes: `scripts/jlens_reasoning_tokens.py` now persists activations
alongside its lens analysis (committed as `2296c92`), and
`prepare_activations_for_probing` can now select which of those tokens/layers a
`next_action` probe trains on (uncommitted at time of writing).

## The pipeline

```
jlens_reasoning_tokens.py                 prepare_activations_for_probing        train_next_action_probe
  forward pass + Jacobian lens              pick tokens & layers                   probe on (D,) activations
  -> .pt activations + per-traj CSV         -> manifest.json of samples
```

## 1. `scripts/jlens_reasoning_tokens.py` — dual artefacts

One forward pass now produces both the lens statistics and the residual streams, so
downstream work never re-runs `gather_activations` over the same trajectories.

```
{activations_dir}/size{S}/{stem}/{stem}_jlens_analysis.csv
{activations_dir}/size{S}/{stem}/openai__gpt-oss-20b/layer_{N}/step_{M}/output/{out_idx}.pt
```

- The `.pt` layout is byte-compatible with `gather_activations`, so every existing loader
  works unchanged. Reasoning tokens are output tokens, hence the `output` category.
- `--out` (one giant combined CSV — a previous run produced 3.2 GB / ~16M rows) was
  replaced by `--activations-dir`; there is now one CSV per trajectory.
- `--overwrite` re-does trajectories whose CSV exists; without it they are skipped. The CSV
  is written to `.csv.tmp` and `os.replace`d, so a crashed run leaves no CSV and is redone
  rather than falsely skipped.
- Activations are saved **before** the lens loop, so layers with no jlens `J` matrix (which
  the loop `continue`s past) still get their `.pt` files.
- `scripts/jlens_reasoning_tokens.sh` defaults to `LAYERS="7:23"` (inclusive → 17 layers).
- Needs a GPU that fits gpt-oss-20b — this cannot run on the laptop.

Deliberately unchanged: `reasoning_token_positions()` keeps its 3-tuple return, because
`scripts/jlens_viewer_export.py` imports it and its self-test asserts that exact shape.
`out_idx` is derived in the caller instead.

**Known follow-up (not done):** `jlens_viewer_export.py --csv` still expects one combined
CSV. It now needs a concatenation of the per-trajectory CSVs (`head -1` of the first +
`tail -n +2` of the rest) or a directory mode.

## 2. `prepare_activations_for_probing` — jlens-driven token/layer selection

Two orthogonal knobs, `next_action` only. Defaults reproduce the previous behaviour.

| `--token-selection` | which reasoning tokens become samples |
|---|---|
| `all` (default) | every token matching `--output-indices` |
| `jlens_direction` | top `--num-tokens` of the trajectory by direction count |
| `random` | `--num-tokens` drawn uniformly — the matched control |

| `--layer-selection` | which layers of each selected token |
|---|---|
| `spec` (default) | every layer in `--layers` (`--layers 15` = the manual middle-layer case) |
| `jlens_direction` | that token's top `--num-layers` by direction count |
| `random` | `--num-layers` drawn uniformly |

`--layers` is always the **candidate pool**; the non-default modes pick within it. Layers are
chosen per token, so two tokens may contribute different layers.

**Scoring.** For each `(reasoning token, layer)` CSV row, the score is how many of
`top_1..top_k` appear in `--direction-tokens-path`
(`/media/alex/D/Uni/northeastern/data/jlens/direction_tokens_full.json`, 539 tokens across
UP/DOWN/LEFT/RIGHT). A token's trajectory-level score sums over the candidate layers.
Ranking is **per trajectory, all steps pooled**. `--direction-classes` defaults to the union
of all four lists so selection does not depend on the label. This is the per-trajectory
version of the global analysis in `notebooks/direction_token_location_analysis.ipynb`.

New file: `telos_interp/commands/prepare_activations_for_probing/jlens_token_selection.py`
(stdlib only, no torch — unit-testable on its own). The seven knobs travel as one
`SelectionConfig` rather than as separate parameters through four nested call sites.

### Two behaviour changes to know about

- **Per-step labels.** Each sample is now labeled with the `agent_action` of *its own* step.
  Identical to before for the default `steps="0"`; the old code silently applied step 0's
  action to every token when `steps="all"`.
- **Manifest additions.** A `selection` block records how samples were chosen; samples picked
  via a `jlens_direction` mode also carry `token`, `direction_count`, `layer_direction_count`.
  Existing keys are untouched, so `load_next_action_compact` and `train_next_action_probe`
  needed no change.

### Invariants worth not breaking

- **`abs_pos` → filename.** The CSV stores `abs_pos`; the `.pt` filename is the
  output-relative index. The mapping is
  `out_idx = abs_pos - (len(prompt_prefix_tokens) + len(steps[si].grid_state_tokens) + len(prompt_suffix_tokens))`
  — exactly what the writer did. It needs the trajectory JSON, which prepare already loads.
- **`step_id` vs list index.** The CSV's `step` column is the list index into
  `trajectory["steps"]`; the folder is `step_{step_id}`. They agree in the data seen so far,
  but the code reads `step_id` rather than assuming it.
- **CSV must be read with `csv.DictReader`, not pandas.** Decoded tokens include strings like
  `"NA"`, `""`, embedded newlines and commas; pandas' default NA handling would corrupt them.
- Auto-named output directories encode the selection (`..._next_action_tokjl20_layjl3`), so
  each variant is a separate dataset. Producing several variants means several invocations.

### Running it

`interp-cli` is tyro-driven — the function signature *is* the CLI, so new parameters need no
extra wiring. Full examples live in
`telos_interp/commands/prepare_activations_for_probing/README.md`.

```bash
interp-cli prepare_activations_for_probing \
    --activations-dir /workspace/activations/jlens_reasoning_tokens \
    --trajectories-dir /workspace/trajectories/trajectories_test_full \
    --probe-type next_action --layers 7:23 --steps all --output-indices all \
    --token-selection jlens_direction --layer-selection jlens_direction \
    --num-tokens 20 --num-layers 3 \
    --direction-tokens-path /workspace/jlens/direction_tokens_full.json
```

## Environment

All Python must run through the conda env `interp`:
`conda run -n interp python ...`. Note that `conda run` sometimes swallows stdout — if a
command appears to produce nothing, write the script to a file and re-run, or check the exit
code.

## Verification status

- `conda run -n interp python -m pytest tests/ -q --ignore=tests/test_trajectory_activations.py`
  → 82 passed. `tests/test_trajectory_activations.py` fails at import on a module that has
  never existed in this repo — pre-existing, unrelated.
- End-to-end against the **real** global jlens CSV (one trajectory's rows, real quoting, real
  539-token direction vocab, multi-size layout, layers 7:23, N=5/M=3): selected
  `(step, token)` pairs matched an independently computed ranking exactly, every `act_path`
  loaded the tensor for the right `(layer, step, token)`, `direction_count` monotonically
  non-increasing. Picks were `Ġleft`, `Ġright`, `ĠMove`, `Ġdown`, `Ġdown` at layers 9–18.
- `scripts/jlens_reasoning_tokens.py --self-test` passes; its GPU-host run and the
  `torch.allclose` cross-check against an existing `gather_activations` tree have **not**
  been done yet.

## 3. `jlens_reasoning_tokens.py` — throughput rework

The sweep was unacceptably slow on the high-complexity cells. The cause is not grid size:
every per-step cost scales linearly with the length of the reasoning chain, and comp
0.6–1.0 trajectories reason far longer. A ~700-token chain over 17 layers is ~12k `.pt`
files *and* ~12k CSV rows per step, against roughly 0.5 s of actual GPU work.

The changes are all semantics-preserving — same tree, same rows, every token kept:

- **`--io-workers` (default 16).** `ActivationWriter` in `gather_activations_utils.py`
  writes the per-token tree from a thread pool, overlapping it with the next forward pass.
  Same paths, same tensors, same NaN accounting; `save_activations_to_files` is now a thin
  serial wrapper around it, so `gather_activations` is untouched. Drained before the CSV is
  `os.replace`d, and worker exceptions are re-raised there — a failed write can never be
  mistaken for a finished trajectory.
- **Decode memo.** `tok.decode([i])` ran TOP_K times per row, millions of times per
  trajectory. Now a dict lookup after the first sighting.
- **Lens matrices hoisted.** They were being `.float().to(dev)`-ed *per step* — 566 MB of
  H2D for weights that never change. Now once, and the per-layer `h @ J.T` is one `bmm`
  (`apply_lens_transport`).
- **`extract_activations_batched`.** Returns one stacked `(N, d)` tensor per layer via a
  single `index_select`, replacing ~12k individual `.cpu()` copies per step and the
  re-upload that followed. The residual streams stay on the device for the lens.
- **`--forward-batch-size` (default 4) / `--forward-batch-tokens`.** Packs consecutive
  steps into one right-padded forward pass. Consecutive and never reordered, so the CSV
  stays in step order.
- **`--batch-size` now defaults to 256** instead of "the whole step", and the per-step
  `torch.cuda.empty_cache()` is gone.

**Measure before tuning:** `--profile` prints the per-phase split (`build` / `forward` /
`save_pt` / `drain_pt` / `lens` / `rows` / `csv`) per trajectory and for the run.
`--no-save-activations` and `--benchmark-pt-write N` isolate what the `.pt` tree costs on
that host — a local NVMe does ~10k files/s, a network volume can be two orders of magnitude
slower, and which one you have decides whether the threading is enough.

### What is verified, and what is not

`tests/test_jlens_reasoning_tokens.py` runs the whole script against a stub 24-layer model
whose activations are a closed-form function of (layer, token id, position). It asserts the
rewritten loop reproduces **the pre-change implementation byte-for-byte** — the golden
reference `_reference_run` is the old inner loop transcribed verbatim — for the CSV and
tensor-for-tensor for the `.pt` tree, with batching and lens chunking on.

The one thing that stub cannot test: on the real model, batched bf16 GEMMs may reassociate
and shift a logit in its last bits, which could flip a rank between two near-tied tokens.
Use `--forward-batch-size 1` for bit-exact agreement with an unbatched run; that setting
disables only the packing, and every other optimisation above still applies.

## Current state on the GPU host (2026-08-24)

**Gather is done, for all three arms.** `/workspace/activations/jlens_reasoning_tokens/`
holds 3604 trajectory folders (size5/7/9/11/13/15, ~600 each). 3600 carry a **v2 selection
record with all three arms** — `jlens`, `logitlens`, `random` — and a
`*_logitlens_analysis.csv`, 600 per size with no gaps. The remaining 4 have no record at all
and are simply not in any arm.

The logitlens arm came from `scripts/jlens_extend_logitlens.sh` (`--select-num-layers 3`),
which finished cleanly: log `/workspace/logs/jlens_extend_logitlens_run_16_00.txt`, last
write 17:56 UTC, ending `done: 2994 trajectory folder(s)`. The only repeated message in it is
the harmless `--extend but no selection record ... skipping`. Note the "2994" is what that
run walked, not the tree size — arm coverage is the number that matters, and it is complete.

Per trajectory each arm records **at most 20** tokens (`rank_tokens(...)[:num_tokens]` and
`min(num_tokens, len(universe))` both truncate on a short reasoning chain): the lenses' top-20
by `direction_count`, the control's 20 uniform draws carrying no counts at all.

**Prepare has not been run for these arms.** The only thing in `/workspace/prepared/` is
`next_action_comp0.0-0.2-0.4_jlens`, an older run on the 3-complexity view using
`jlens_direction` (CSV re-scoring), not `recorded_*`.

### Running it: all selected tokens, one layer each

```bash
# lens arms: keep every recorded layer so the split can pick the top-scoring one
ARMS="jlens logitlens" LAYERS=7:23 OUT=/workspace/prepared/next_action \
    ./scripts/prepare_next_action_arms.sh

# control: pin to layer 15 HERE, not at split time (see below)
ARMS="random" LAYERS=15 OUT=/workspace/prepared/next_action \
    ./scripts/prepare_next_action_arms.sh

# train: every selected token, one layer per token
ARMS="jlens logitlens random" TOKENS_PER_TRAJ=all LAYERS_PER_TOKEN=1 \
    PREPARED=/workspace/prepared/next_action ./scripts/train_next_action_arms.sh
```

`TOKENS_PER_TRAJ=all` omits `--tokens-per-trajectory` entirely rather than passing a large K,
so `thin_tokens` is skipped and every token in the arm's record is trained on. It means *that
arm's* tokens, not the union across arms — the three overlap only partially, which is the
point: equal N, different chooser.

**Why the control is pinned at prepare time.** `thin_layers` ranks by `layer_direction_count`
when counts are present and *samples uniformly* when they are not, so `--layers-per-token 1`
gives the top-scoring layer for jlens/logitlens but a **random** one of the control's ~6
recorded layers. Layer 15 is only guaranteed to be among the candidates, never to be the one
drawn. Preparing that arm with `--layers 15` leaves one row per token, after which
`--layers-per-token 1` is a no-op that picks it.

The two prepares share one `OUT` prefix and write different arms of it — deliberate; train
reads `${PREPARED}_${arm}`. Expect ~3600 × ≤20 × 1 = ≤72k rows per arm, matched across arms.

## Open threads

- On the GPU host, run one comp-0.8 trajectory with `--profile` and record the split, then
  tune `--io-workers` against it. If file creation still dominates after threading, the next
  lever is a sharded `.pt` container (one file per (layer, step) instead of per token,
  ~150x fewer inodes) — deliberately not done, because it changes the on-disk contract.
- Confirm on the real model whether `--forward-batch-size 4` changes any CSV byte versus
  `1`. If it does, decide whether the speedup is worth the bf16 noise.
- Cross-check one saved tensor against an existing `gather_activations` run for the same
  trajectory/layer/step/token (`torch.allclose`) — both capture decoder-block outputs from
  the same truncated prefix, so they must match.
- Train and compare the three arms, using the commands above. The tree is pruned, so read
  every arm back with `recorded_<arm>` — re-scoring a CSV would still find the right tokens
  for a lens, but a uniform draw over the survivors is no longer a uniform draw over the
  reasoning chain, and the control would stop being one.
- Watch out for a stopped (state `Tl`) leftover `jlens_reasoning_tokens.py` from the original
  gather, still parked in the process table days later alongside `watch`/`tail` watchdogs.
  Not writing, but a `pgrep` for a live run will match it — check the argv (`--extend`,
  `--select-num-layers`) and the state flag, not just the name.

## Methodological caveat

The highest-scoring tokens are literally the direction words (`Ġleft`, `Ġright`, `Ġdown`).
A probe trained on them is partly reading a decision the model has already verbalized, so
accuracy on that dataset alone says little. The random control is what makes the comparison
meaningful — build both.

## Resuming — start here (2026-08-26)

Two rounds exist. **Round 1, "direction probing", is finished** (log entries 1-31): tokens
selected by how many of their top-k lens predictions are *direction* words, probing the next
action. Result: jlens > logitlens > random > eos, stable across three seeds (max sd 0.26 pp).
Nothing about round 1 is being re-run.

**Round 2, "grid probing", is in progress** (log entry 34): the same machinery pointed at
`/workspace/jlens/grid_tokens_full.json`, so tokens are selected by *grid* words, and the
probe trained on them is the cognitive map (`--probe-type grid_tile`). It writes to its own
tree, `/workspace/activations/grid_reasoning_tokens`, and does not touch round 1.

### Where round 2 actually stands

| stage | state |
|---|---|
| gather | **NOT started.** Smoke-tested on 2 trajectories and passed. |
| prepare / split / train | not started |

Run `./scripts/grid_round_status.sh` for the live picture — it reads only `/workspace`, so it
works from any session.

**One thing must be settled before the prepare stage** (entry 34, "OPEN LANDMINE"):
`--balance-classes-per-trajectory` collapses to the rarest class — a grid has exactly one `A`
and one `G`, so every trajectory keeps *one cell per class*, 4 out of 225. And the class count
depends on padding, while prepare **skips** trajectories whose cell count disagrees with the
first, so mixing sizes under balancing can drop whole sizes silently. Proposed fix: drop
per-trajectory balancing, keep a plain `--max-positions-per-trajectory` cap, and let the
trainer's `--class-weight balanced` handle imbalance. The A/B that would confirm it was
interrupted before producing numbers; it needs no GPU.

### The three round-2 scripts

```bash
./scripts/gather_grid_arms.sh          # build the tree (~3-4h on one GPU; resumable)
ARMS="jlens logitlens random" LAYERS=15 OUT=/workspace/prepared/grid_l15 \
    ACT=/workspace/activations/grid_reasoning_tokens ./scripts/prepare_grid_arms.sh
PREPARED=/workspace/prepared/grid_l15 TAG=l15 \
    EVAL_NAMES=/workspace/splits/eval_trajectories_720.txt \
    ARMS="jlens logitlens random" SEEDS="42 43 44" ./scripts/train_grid_arms.sh
```

`DRY_RUN=1` on the gather prints its invocation and runs nothing. The planned sweep is 22
runs: 18 at layer 15 (3 arms x 3 seeds x lr/mlp) + 4 at layers 7:23 (jlens/logitlens, seed 42).

`/workspace/splits/` holds `eval_trajectories_720.txt` (md5 `d7af16f3`) and
`lens_trajectories_3600.txt` — round 1's exact eval trajectories. Pin round 2 to them via
`EVAL_NAMES` so the two rounds are comparable.

### Round 3, queued: logprob scoring + one layer (2026-08-27, code only)

Log entry 35. Two changes landed in the code and **nothing on disk has been touched**:

- The lens CSVs now carry a `top_{i}_logprob` beside every `top_{i}`, and how a token's
  direction evidence becomes a score is a registry (`jlens_utils/scoring.py`):
  `--direction-score count|logprob_mass|logprob_sum|logprob_mass_full`. Default stays
  `count`, so nothing recorded changes meaning. Avoid `logprob_sum` — it adds logs, which
  multiplies probabilities and so scores a token *worse* the more direction words it emits.
- The gather also writes a **direction-mass table** per (trajectory, lens):
  `{stem}_{lens}_direction_mass.csv`, wide — one row per reasoning token, one `L{n}` column
  per layer, each cell `log P(any direction word)` over the *whole* vocabulary rather than
  a top-20 window. `--direction-score logprob_mass_full` scores it; that is the mode to
  use. Every table has a `.meta.json` sidecar naming the vocabulary that made it — read it,
  because round 1 and round 2 use different vocabularies on identically-shaped trees.
- `split_next_action_manifest.py --single-layer L|best` (`SINGLE_LAYER=` in the
  `train_*_arms.sh` pair) pins the whole dataset to one layer, instead of
  `--layers-per-token 1` which keeps each token's own best layer and leaves the dataset
  spanning the whole range.

**Every CSV and every selection record on disk is pre-logprob**, so a logprob mode raises
until the CSVs are re-emitted. That re-emit is CSV-only and writes no `.pt`, so the pruned
tree is safe:

```bash
# 1. re-emit the CSVs + the direction-mass tables (no .pt written, tree untouched).
#    --direction-mass-json is REQUIRED here: it defaults to --signal-json, which this
#    (non-selective) invocation does not set, and without it no table is written.
uv run python scripts/jlens_reasoning_tokens.py --overwrite --no-save-activations \
    --trajectory-paths ... --jlens_dir /workspace/jlens/gridenv \
    --activations-dir /workspace/activations/grid_reasoning_tokens --lens both \
    --direction-mass-json /workspace/jlens/grid_tokens_full.json

# 2. the unbiased per-layer profile -> pick L
uv run python scripts/jlens_layer_profile.py /workspace/activations/grid_reasoning_tokens \
    --signal-json /workspace/jlens/grid_tokens_full.json \
    --direction-score logprob_mass_full

# 3. train the arms pinned to it (give the CONTROL the same explicit L; it has no scores)
SINGLE_LAYER=<L> PREPARED=... ./scripts/train_grid_arms.sh
```

Caveat before spending GPU time: on a **pruned** tree a new score can only re-rank the ~20
tokens that survived. That is a usable top-K comparison, but a genuinely logprob-selected
arm needs a fresh selective gather (or `--extend`) on a tree that still holds the tokens.

### Round 1 leftovers worth knowing

- The token-major implementation entries 32-33 describe (prepare, loader, both trainers, the
  split script, `tests/test_train_cognitive_map_probe.py`) is what round 2 runs on.
- Parked, not forgotten: the eos cap (`--tokens-per-trajectory 20`, entry 30), and an eos arm
  for round 2 — that one needs no gather, only prepare+train, since it uses no lens selection.
- PID 1992 is a stopped (`Tl`) leftover from the original round-1 gather, days old and idle. A
  bare `pgrep jlens_reasoning_tokens.py` matches it; check the argv, as `grid_round_status.sh`
  does.

Everything above is **uncommitted working-tree state** on `reasoning_theatre`.

Do not run `make fix-style` or a bare `ruff check` on the whole repo: the first reformats
everything and the second is configured with `fix=true`, so it rewrites files in place. Both
pulled ~40 untouched files into the diff here.

## Probe vs. rollout — what the probe is decoding (2026-08-28)

New, and orthogonal to selection: `/workspace/reasoning_theatre/trajectories_train_single_step_probs/`
(written by `scripts/inference_oss/run_inference.py`) re-runs the model at every reasoning
sentence end with reasoning truncated there, and records what it *would* answer. That joins to
entry 38's `heldout360_all_probes.csv` for free — `eos_token_pos`, `token_idx` and the `.pt`
filename all index the same `step["output_tokens"]` list — so every reasoning token can be placed
inside the sentence whose truncation eval covers it.

```
scripts/build_probe_rollout_join.py          -> probe_vs_rollout/per_token.csv
scripts/analyze_probe_rollout.py             -> probe_vs_rollout/tables/*.csv, summary.json
scripts/plot_probe_rollout.py                -> probe_vs_rollout/plots/*.png
scripts/analyze_jlens_direction_classes.py   -> tables/q4v_*.csv   (entry 40)
scripts/jlens_direction_vocab_diagnostic.py  -> which vocabulary tokens actually fire
scripts/plot_commitment_all_tokens.py        -> plots/commit_all_tokens_*.png   (entry 41)
scripts/plot_commitment_probs.py             -> plots/commit_probs_*.png        (entry 41)
```

Entry 41 moves the boundary comparison from sentence ENDS to every token. Two things it
depends on that are easy to get wrong: past the convinced boundary the three comparators
(this belief / previous belief / final action) are **one series by definition**, so only
x <= 0 carries information; and `convinced_sentence_idx == 0` trajectories (95 of 360) must
be dropped, or the two sides of the boundary are computed over different trajectories.

`plot_commitment_probs.py` needs `eval_probe_per_token.py --full-probs`, which writes the
whole 4-way softmax as `{probe}_p_{ACTION}` in a fixed LEFT,UP,RIGHT,DOWN order rather than
the probe's own `label_to_idx` order, so the columns mean the same thing across arms. It is
~35 min for 4 probes over the 87k tokens -- the cost is reading 87k `.pt` files off the FUSE
mount, not the forward passes. Read the MLP arms' probabilities knowing they are near
one-hot (mean max p 0.91 at 0.45 accuracy); the LR arm (0.686) is the calibrated readout.

Result, in one line: **the probe decodes the model's in-flight belief, not the action it ends up
taking** — on the 2,074 sentence ends where those differ the headline arm
(`next_action_l15.jlens_topall_mlp`) matches the belief .431 and the final action .250
(chance .25; the best arm, `next_action_mass_l15.jlens_topall_mlp`, gets .444 / .227 — quote
the headline, not the maximum on the same rows), and all 26 probes lean the same way. The layer-15 lens does *not* see
the boundary: read through the full 446-token direction vocabulary split by class it reaches
prior-free AUC 0.54 at a sentence end (0.51 on an arbitrary token), against the probe's .517
accuracy on the same rows. Entries 39 and 40 of the log have the full account.

Four traps worth carrying:

- The rollout root holds all 36k training trajectories on a **FUSE mount**. Glob it and the join
  takes >5 min; address the 360 held-out files by name and it takes seconds.
- Argmax over the raw `{ACTION}_logprob` columns of a lens CSV is **degenerate** — 63% "DOWN",
  3% "LEFT", because it reads how common the uppercase string is. Center each column on its
  corpus mean/sd before comparing them, or use the AUC over (token, action) pairs.
- **Do not quote an argmax accuracy over the four direction classes either.** No per-column
  correction fixes it when the classes fire at very different rates (RIGHT is at the floor for
  74% of tokens), and every readout's argmax lands on one class for 48–79% of tokens. Quote the
  AUC, which a per-class constant cannot move. `argmax_max_share` in `q4v_class_agreement.csv`
  is the guard column.
- The **RIGHT class of `direction_tokens_full.json` is anti-predictive** (one-vs-rest AUC .486).
  `' right'` is 44% of its hits and sits *below* base rate — English "right" is a discourse word.
  This belongs back in `notebooks/direction_tokens.ipynb`: test a candidate class by
  P(answer = c | hit) against base rate before shipping it. Also, only 84 of the 446 tokens ever
  reach a top-20 here, so the multilingual entries are dead weight on this task.

---

## Loudness through a sentence, and at the commitment boundary (log entry 42)

Note the correction this makes to the paragraph above. "The layer-15 lens does *not* see the
boundary" is true of the readout entries 39/40 used — a **top-20 count** on the 360 held-out
trajectories. It is not true of the **full-vocabulary mass**, and that is the whole finding of
entry 42.

Three stdlib+pandas scripts, no GPU, all reading artifacts that already exist:

```bash
.venv/bin/python scripts/build_sentence_loudness.py    # -> loudness/per_token.csv  (~1m45s)
.venv/bin/python scripts/plot_sentence_loudness.py     # -> loudness/plots|tables   (22 figures)
.venv/bin/python scripts/analyze_sentence_loudness.py  # -> loudness/summary.json
```

Output root `/workspace/reasoning_theatre/loudness/`. Report:
https://claude.ai/code/artifact/a26ec417-feae-42eb-a09d-af529f603c06

What "loudness" now means: `exp(L15)` of the **direction-mass table**
(`{stem}_jlens_direction_mass.csv`), i.e. `sum(exp(logprob(t)) for t in direction_tokens_full)`
over all 446 tokens, at every reasoning token — not the top-20 count and not the top-20 mass.
The tree is `/workspace/activations/jlens_mass_l15` and the trajectories are the **2,880 the
mass arm trained on** (its 3,600 minus `next_action_mass_l15_eval_names.txt`), which is a
different draw from `lens_trajectories_3600.txt` — the two trees overlap by only 348 names.

Three traps to carry forward:

- **The header offset is not 3.** `build_probe_rollout_join.py` hardcodes
  `reasoning_pos = token_idx - 3`. The real gap between the mass table's `reasoning_pos`
  (analysis-tagged tokens) and the rollout's `eos_token_pos` (`step["output_tokens"]`) is
  `eos[0] + 1`. It happens to be 3 on every trajectory checked; read it anyway.
- **Sentence position dominates everything else.** Loudness falls 2.2x from a sentence's first
  decile to its last and resets at the break, so any effect measured at token resolution has to
  be read against that sawtooth. It is also why the convinced-index effect only shows up in a
  *paired within-trajectory* contrast at matched sentence positions.
- **The direction-token control needs a window, not a token.** `is_direction_token` flags the
  token the model emitted; dropping those rows is *not* enough, because the jlens predicts the
  next tokens and so is loud on the token just before `" up"`. Excluding a ±1 window cuts the
  boundary step from +.0266 to +.0076, and at ±3 (+.0041) it no longer beats its own placebo
  (+.0051). `verbalization_proximity` in `summary.json`. Any future "the lens sees X" claim on
  this data needs the same windowed control.
- **"Convinced" means correct-from-here-on, not settled.** Defined against ground truth in
  `run_inference.py::commitment_metrics`. `per_token.csv` carries `n_switches` and
  `sent_model_action` if you want to rebuild the boundary from actual belief switches instead —
  that is the obvious next cut.

## Truncation strategies — cutting where the lens is loud (log entry 43)

**Status: built and verified, NOT RUN.** `/workspace/reasoning_theatre/rollout_strategies/`
does not exist yet — the only output on disk is the 10-trajectory smoke at
`rollout_strategies_smoke/`, which shows all three arms run end to end and whose accuracies
(n = 10) are not a result. Launching the three arms is the next action.

`run_inference.py` no longer only cuts at sentence ends. Where it cuts is `--strategy`, and
the registry is `scripts/inference_oss/truncation_strategies.py`:

| `--strategy` | cutoffs per step | what it cuts at |
| --- | --- | --- |
| `eos` | ~24.5 | every reasoning sentence end (the original grid) |
| `jlens_argmax_per_sentence` | ~25.5 | the loudest token of each sentence |
| `jlens_top_k_global` | ~21.9 | the `--top-k` (20) loudest tokens of the whole chain |

Loudness is entry 42's: `exp(L15)` of `{stem}_jlens_direction_mass.csv`, i.e.
`sum(exp(logprob(t)) for t in direction_tokens_full)`. Ranking happens on the log (exp is
monotone) and the mass table is never read without its `.meta.json` — the vocabulary it was
baked against is copied into every results JSON.

```bash
bash scripts/inference_oss/run_inference_strategies.sh                     # all three arms
bash scripts/inference_oss/run_inference_strategies.sh jlens_top_k_global  # one arm
DRY_RUN=1 bash scripts/inference_oss/run_inference_strategies.sh           # cutoffs only, no model
```

Output root `/workspace/reasoning_theatre/rollout_strategies/<strategy>/`, logs in
`<strategy>/logs/`. `analysis.py` reads any of them unchanged.

Four things to carry:

- **All three arms run the same 3,600 trajectories**, pinned by `--names-file` (built from
  `/workspace/activations/jlens_mass_l15`, the only trees with mass tables). The eos arm is
  re-run over them rather than reused from the 36,000-trajectory
  `trajectories_train_single_step_probs/`, so the three differ only in cut placement. This is
  the mass tree's full 3,600 — nothing here trains a probe, so entry 42's 720 held-out names
  are *not* excluded.
- **The endpoints are kept in every arm** (no-reasoning first, full-reasoning last), so every
  arm's first and last eval is the same prompt and `final_sentence_accuracy` /
  `convinced_sentence_idx` stay comparable. `--no-endpoint-cutoffs` drops them and breaks that.
- **The schema is the eos schema.** `sentence_evals` is the cutoff list, `eos_token_pos` the
  cut position; `cutoff_kind`, `cut_sentence_idx`, `pos_in_sentence`, `sentence_len`,
  `dir_logmass` and `dir_prob` are added per eval. `n_reasoning_sentences` counts cutoffs in
  every arm (`n_cutoffs` is the honest alias).
- **The grids really are different.** The per-sentence argmax sits at mean fraction 0.417
  through its sentence and coincides with the sentence end only 10.4% of the time; only 3.4%
  of `jlens_top_k_global`'s cutoffs are sentence ends.
- **The loud arms are confounded by verbalization, structurally.** The loudest token in a
  sentence is very often the direction word the model is writing. On the dry-run trajectory the
  per-sentence argmaxes are ` up` (.870), `(7,` (.130) and ` UP` (.686) against a sentence-final
  `.` at .028 — so cutting at the argmax often means cutting just after the model typed the
  answer. A raw accuracy gain for a loud arm is not evidence of earlier commitment. Read it
  either with direction-word cutoffs dropped (entry 42(e)'s control; `dir_logmass` is on every
  eval and the token is in the mass table) or against a seeded random-position arm of the same
  K — that fourth strategy is one registry entry in `truncation_strategies.py` and is not built
  yet.
- **Do not raise `BATCH_SIZE` to buy throughput.** 16 rows / 49,152 padded-token area is the
  measured setting: 64 rows (area cap raised to match) OOMs a 32 GB card at ~500-token prompts,
  because eager attention allocates `rows x heads x seq^2`. Measured ~4.7 prompts/s on a mixed
  size-5/11/15 set, ~3.2/s on size-11 alone, i.e. roughly 5-8 h per arm (~85k prompts) and
  about a day for all three.

## Both loud arms are run (log entry 44)

**Status: RUN, on 2026-08-31/09-01.** The section above ("Truncation strategies", entry 43)
says they were not; it is superseded. `/workspace/reasoning_theatre/rollout_strategies/` now
holds 3,600 result JSONs per arm.

| arm | wall clock (UTC) | cutoff-evals | cut/step | raw cutoff acc |
| --- | --- | --- | --- | --- |
| `jlens_argmax_per_sentence` | 08-31 18:42 → 09-01 00:45 | 82,212 | 22.84 | 69.89% |
| `jlens_top_k_global` | 09-01 00:45 → 04:40 | 78,707 | 21.86 | 79.55% |

**The `eos` arm was NOT re-run.** The finished 36k rollout covers all 3,600 names, so it
supplies the sentence-end answers. See the noise floor below before relying on that.

```bash
bash scripts/inference_oss/run_inference_strategies.sh jlens_argmax_per_sentence jlens_top_k_global
watch -n 1 bash scripts/inference_oss/rollout_status.sh     # live, work-weighted ETA, read-only
python scripts/plot_loud_vs_sentence_end.py                 # -> comparison/loud_vs_sentence_end.{png,csv}
python scripts/analyze_truncation_strategies.py             # the full arm comparison -- NOT YET RUN
```

Five things to carry:

- **Do not compare the two raw arm accuracies.** Different grids, different sizes, and entry
  43's verbalization confound is unaddressed in both. 37.5% of the argmax arm's cutoff tokens
  *are* direction words. `analyze_truncation_strategies.py` does it properly and is unrun.
- **`--max-batch-tokens` was the wrong invariant and it OOMed the first launch.** Eager
  attention allocates `rows*heads*L^2`; the area cap is linear in `L`. 16 rows x 1587 tokens is
  an area of 25,392 inside a 49,152 budget *and* a 4.80 GiB tensor. Fixed by
  `--max-attn-elems` (default 16e6, caps `rows*L^2`) plus an OOM-halving retry;
  `tests/test_run_inference_batching.py` pins it. Do not raise it back.
- **There is an endpoint noise floor of ~3.3%, and it is intrinsic.** Two arms of the *same
  run* disagree on 3.3% of no-reasoning endpoints — identical prompt, identical position, only
  the batch neighbours differ, so it is left-padding (attention sinks are the suspect). The old
  eos rollout adds 4.6 points on top (7.97% total). At the end of reasoning, where p ~ 1, both
  comparisons agree 3600/3600 with mean |dp| 4e-5: it is argmax flipping between near-tied
  logits, not drift. Any arm-vs-arm difference of a few points is noise.
- **~5% of the argmax arm (165 files) was written by the pre-fix batching**, kept on resume.
  Re-running them is ~10 min.
- **Entry 44(d) is the one analysis actually run, and it now covers both arms.** A loud token
  leans toward the conclusion of the sentence it is IN rather than the previous one's, by the
  same ~15 points in both arms (.560/.397 argmax, .556/.404 top_k, on the ~11% of cutoffs whose
  sentence ends differently than the one before it). The two controls part them: excluding
  cutoffs that ARE the sentence end costs argmax most (.503/.450), while **excluding
  direction-word cutoffs kills top_k** (.485/.470 — inside the noise floor) and leaves argmax
  intact (.549/.399). Neither arm carries the claim alone. Figures:
  `comparison/loud_vs_sentence_end_{arm}.png`.
- **Do not subsample that analysis.** Both arms were first read on ~100 trajectories and both
  samples overstated the effect by ~2x (top_k read .792/.154 against a true .556/.404 and
  looked control-proof; it is not). The differ subset is ~11% of cutoffs.

### TODO (next actions, newest first)

- [ ] **Build and run the `random` truncation arm.** This is the missing control for entry
  44(d), not a nice-to-have: the loudest token sits ~42% through its sentence, so it is simply
  *closer in time* to that sentence's end than to the previous one's, and proximity alone could
  produce the whole .560/.397 lean. Without a matched random-position arm we cannot tell "the
  loud token knows" from "a token that far into the sentence knows", and the coincident-cutoff
  row (.503/.450) says the margin is small enough for that to matter. Spec: one registry entry
  in `scripts/inference_oss/truncation_strategies.py`, same shape as `JlensTopKGlobalStrategy`
  with a seeded uniform draw of the same K, endpoints kept, `arm_seed()`-style frozen draw;
  ~4 h of GPU for 3,600 trajectories. Then re-run `plot_loud_vs_sentence_end.py --arm random`
  and read entry 44(d)'s table against it.
- [ ] Run `scripts/analyze_truncation_strategies.py` (written, smoke-tested, unrun) for the
  arm-vs-arm comparison with the verbalization control.
- [ ] Decide whether to re-run the `eos` arm under this code. Entry 44(c): it removes 4.6 of
  the 7.97 points of endpoint disagreement and not the intrinsic 3.3%.
- [ ] Optionally re-run the 165 argmax files written by the pre-fix batching (~10 min).

---

## Local-belief probes — relabelling the probe target (2026-09-01, log entry 44)

**One line:** relabel the `next_action` probe from the trajectory's final `agent_action`
to the model's *local belief* at the probed token (what it answers when its reasoning is
truncated there), and layer-15 balanced accuracy jumps `.745 → .862` (mlp) on the same
tokens and split.

The label comes from the two truncation rollouts of entry 43:

| probe | tokens | label rollout | samples |
| --- | --- | --- | --- |
| **P2** | the existing `recorded_jlens` top-20 (== `next_action_mass_l15`'s tokens) | `jlens_top_k_global` | 71,913 |
| **P1** | every sentence's loudest token (`loudest_in_sentence`), a new L15 gather | `jlens_argmax_per_sentence` | 75,008 |

Join key: `(name, step, token_id == eos_token_pos)` — all three index `step["output_tokens"]`.

### Pipeline (all under `/workspace/reasoning_theatre/local_belief_probes/`)

```bash
# 1. probe-1 gather: layer-15 residual at every per-sentence-loudest token (GPU, ~45 min)
uv run --project /workspace/repo/interp python scripts/gather_local_belief_activations.py \
    --rollout-dir /workspace/reasoning_theatre/rollout_strategies/jlens_argmax_per_sentence \
    --out /workspace/activations/argmax_per_sentence_l15
# tree is size{N}/{stem}/openai__gpt-oss-20b/layer_15/step_0/output/{eos_token_pos}.pt

# 2. probe-1 prepare (probe-2 reuses /workspace/prepared/next_action_mass_l15_jlens)
interp-cli prepare_activations_for_probing --activations-dir /workspace/activations/argmax_per_sentence_l15 \
    --trajectories-dir /workspace/trajectories/reveng/trajectories_train_single_step \
    --probe-type next_action --layers 15 --steps all --output-indices all \
    --output-path /workspace/prepared/local_belief_p1_final

# 3. relabel: label <- rollout model_action, keep final_label / rollout_answer_prob /
#    rollout_correct / cutoff_kind / dir_logmass; direction_count = dir_logmass so the
#    split can rank by loudness. Drops rows with no matching cutoff / null model_action.
uv run ... python scripts/relabel_manifest_from_rollout.py \
    /workspace/prepared/local_belief_p1_final \
    /workspace/reasoning_theatre/rollout_strategies/jlens_argmax_per_sentence \
    /workspace/prepared/local_belief_p1_local --report-csv p1_relabel_report.csv

# 4. split (SAME 2880/720 partition as the mass baseline) + train lr/mlp
python scripts/split_next_action_manifest.py /workspace/prepared/local_belief_p1_local \
    --eval-names /workspace/prepared/next_action_mass_l15_eval_names.txt --single-layer 15 --seed 42 ...
# scripts/train_all.sh does all 6 (p1 full, p1 top20, p2) x (lr, mlp)

# 5. analysis: pred vs local belief, vs final action, verbalization-confound split
uv run ... python scripts/eval_local_belief.py <probe.pt> <split_eval_dir>
```

### Results (balanced accuracy, shared 720-trajectory held-out eval)

| arm | label | lr | mlp |
| --- | --- | --- | --- |
| **P2** jlens global top-20 | local belief | .802 | **.862** |
| **P1** per-sentence, top-20/traj | local belief | .710 | .774 |
| **P1** per-sentence, all | local belief | .606 | .678 |
| baseline `next_action_mass_l15` (same tokens as P2) | **final action** | .699 | .745 |
| random control | final action | .574 | .625 |

On the rows where local belief ≠ final action (P2: 19%): **P2-mlp predicts the local
belief .77, the final action .14** (entry 41 got .43/.25 at sentence ends). P2-mlp still
scores .747 against the final action — same as the baseline — so it is a strict
improvement.

**Survives the token-itself verbalization control.** 34 % of P2's eval tokens are
direction words (matches 44(d)'s 35 %). On the other 66 %, P2-mlp still scores .850 vs
local belief (baseline .745) and follows local over final .74/.16 on disagreement rows.
Verbalized tokens score higher but aren't the mechanism. Unlike 44(d) — where dropping
direction words sends the *rollout's* sentence-lean into the noise — the *probe* keeps
tracking belief without them. The ±k proximity window of 42(e) is still uncontrolled
(the loud token sits ~42 % into its sentence; "closer in time" isn't separated from
"the lens found it").

### Where things live / what needs merging

- New scripts live under `.../local_belief_probes/scripts/` (not in the repo — the
  `reasoning_theatre` branch had unrelated uncommitted state and the bg-isolation guard).
  They belong at `scripts/inference_oss/` (gather, relabel, eval) when someone commits.
- `SESSION_LOG.txt` → append to `ICLR log.txt` via `ICLR_LOG_ENTRY_DRAFT.txt` (entry 44).
- This file → append to `claude_session_readme.md`.
- Probes: `.../local_belief_probes/probes/`. Datasets: `/workspace/prepared/local_belief_p*`.

---

## Probe loudness — what loudness buys the probe (2026-09-01, log entry 46)

**One line:** the entry-45 local-belief probes, read *by loudness*, on their 720 held-out
trajectories — louder decodes better (P1 top-20 mlp `.600 → .947` across deciles, zero
reversals), it is the loudness and **not** the sentence position (2.5–3×), and a
final-action-trained probe **flips to reporting the local belief** at loud tokens.

Report: <https://claude.ai/code/artifact/2e873b12-7c49-4ac6-b861-9c2a7aa707f0>
(source `probe_loudness/report.html`).

```bash
python scripts/build_probe_loudness.py     # -> probe_loudness/per_token.csv  (~15 min)
python scripts/analyze_probe_loudness.py   # -> summary.json, tables/ (18 CSVs)
python scripts/plot_probe_loudness.py      # -> plots/ (20 figures)
```

The probe-side twin of the `build_/plot_/analyze_sentence_loudness.py` trio, reusing entry
42's sentence coordinates verbatim so the two `per_token.csv` files read side by side.
Output root `/workspace/reasoning_theatre/probe_loudness/`.

### The three rowsets, and why P2's matter most

| rowset | tokens | n | probes scored |
| --- | --- | --- | --- |
| `p1_full` | every sentence's loudest | 14,470 | `p1_{lr,mlp}` |
| `p1_top20` | thinned to 20/trajectory | 8,357 | `p1t20_{lr,mlp}` |
| `p2` | the 20 globally loudest | 14,391 | `p2_*` + **baseline** `next_action_mass_l15` + **random control** |

**P2's rows ARE the entry-38 baseline's eval split, exactly**, so the old final-action probe
and its random control are scored on the identical 14,391 tokens. Every difference there is
the label or the training selection, never the tokens — that is what makes the crossover
below readable. Join is `(name, step, token_id)`; 37,218/37,218 rows placed, 0 dropped, and
the manifest `dir_logmass` matches the mass table's L15 cell on every row. All ten probes
reproduce entry 45's own eval to 3 dp.

### What it found

- **Loudness orders decodability, ~3× the gradient entry 37 saw** with the final-action
  label. P1 top-20 mlp is perfectly monotonic over ten deciles (Spearman 1.00).
- **It is loudness, not position — this is 44(e)'s missing control.** Loudness tercile ×
  sentence-position tercile: accuracy moves `.233–.265` across loudness at fixed position
  against `.078–.110` across position at fixed loudness. The tokens *do* sit ~42% into
  their sentence as 44(e) said (mean `sentence_frac` .422/.425); it just isn't what's
  carrying the accuracy.
- **The crossover (the result).** On the 19% of rows where local belief ≠ final action, both
  *final-action-trained* probes reverse with loudness: the random control goes from
  `.278 local / .633 final` at the quietest decile to `.616 / .261` at the loudest. Nothing
  about the probes changed, only the loudness of the token. Loudness marks a residual that
  is about the model's **current** state rather than its eventual answer.
- **Verbalization is not the mechanism**, second time. Drop direction words entirely and
  loudness still orders decodability (P1 `.478 → .685`). Note this holds where the
  *rollout*-side version of the control failed in 44(d).

### Landmine: chain length is a confound inside any fixed top-K arm

A top-20 takes the 20 loudest of a 37-token chain and the 20 loudest of a 738-token one, so
**`corr(dir_logmass, n_reasoning_tokens) = +0.42` in P2** — and long chains are harder
(`local == final` falls .893 → .665 across length quartiles). The two effects run opposite
ways and roughly cancel, which is why P2's raw decile curve looks flat and its Spearman is
≈0. It is *not* that loudness stopped mattering. `analyze_probe_loudness.py` re-cuts the
loudness terciles inside chain-length quartiles (`chain_length` in `summary.json`,
`*_chain_length_control.csv`, and the four-panel figure) and the gradient reappears —
**for the local-belief probe only** (`+.151 +.022 +.076 +.111`), not for the final-action
baseline (`+.121 −.014 −.003 −.015`). Read any raw decile curve on a fixed-top-K arm
through this control first. P1 is immune (`r = −0.07`).

### Still open

- **The ±k proximity window of 42(e) is now the ONLY one of the three confounds still
  open.** Position-in-sentence is controlled here; "a token two positions before `" up"`" is
  not. That needs the exclusion-window sweep 42(e) ran, applied to the probe rather than the
  rollout.
- The random control is read off its own training distribution (entry 37(d)) — a floor, not
  a matched arm. Its `.695` vs final here against `.625` on its own tokens *is* that shift.
- Layer 15 only; `convinced` still defined by correctness against ground truth.

---

## Which dataset produced which artifact (audit, 2026-09-01)

Three canonical trajectory sets, verified mutually disjoint (`train ∩ eval = 0`,
`(train ∪ eval) ∩ heldout360 = 0`). Re-run the audit with
`python scripts/audit_trajectory_sets.py`.

| artifact | dataset |
| --- | --- |
| `loudness/` — *Where the Lens Is Loud* (entry 42), 22 figures | **training 2,880** |
| `probe_vs_rollout/` — *The Commitment Boundary* (39/40/41), 15 figures | **heldout 360** |
| `probe_loudness/` — *What Loudness Buys the Probe* (entry 46), 20 figures | **eval 720** |
| ↳ its grey "every reasoning token" reference series only | training 2,880 |
| `probes/next_action_mass_l15/heldout360_per_token.csv` (entry 37) | **heldout 360** |
| `probes/heldout360_all_probes.csv` (entry 38) | **heldout 360** |
| entry 45 local-belief probe results (the `.862` headline) | **eval 720** |
| `rollout_strategies/comparison/` (entry 44d), 2 figures | **all 3,600** ⚠️ |

**The convention**, so this stops drifting:

- **eval 720** (`next_action_mass_l15_eval_names.txt`) — anything scored against the
  mass-tree probes. Every arm shares this exact partition, which is what makes
  same-token comparisons against the baseline legitimate.
- **heldout 360** — the stronger claim: a disjoint *tree*, no shared draw at all.
  Use it when the point is generalisation rather than a matched comparison.
- **training 2,880** — distribution references only, never an accuracy number.

**⚠️ Entry 44(d) is the exception and should be read with it in mind.** Its two
`loud_vs_sentence_end_*` figures cover all 3,600 trajectories, train and eval mixed
(both arms have exactly 3,600 result JSONs). That is *not* leakage as it stands — the
analysis is rollout-vs-rollout and fits no model — but it is a different population from
everything else, so its numbers do not sit beside the others, and using those figures to
select or tune a probe *would* make it leakage. Re-cut to eval 720 before comparing it to
any probe result.

---

## The commitment boundary, re-read by the belief-trained probes (log entry 47)

**One line:** a **clone** of *The Commitment Boundary* with the two local-belief MLP probes of
entry 45 added to every probe figure, on the same 87,221 **heldout-360** tokens. It confirms
the original and revises one part of it.

Report: <https://claude.ai/code/artifact/95a74d99-bb0a-440f-839c-e6e4dbac8c65>
(source `probe_vs_rollout_lb/report.html`). The original is **untouched** — md5 verified.

```bash
# 1. score the 2 new probes on the same held-out tokens (GPU, ~2 h; the cost is 87k .pt reads)
python scripts/eval_probe_per_token.py \
    --probe /workspace/probes/local_belief/next_action_probe_p1_mlp.pt \
    --probe /workspace/probes/local_belief/next_action_probe_p2_mlp.pt \
    --activations-dir /workspace/activations/heldout360_l15 \
    --lens-dir /workspace/activations/heldout360_lens \
    --trajectories-dir /workspace/trajectories/reveng/trajectories_train_single_step \
    --signal-json /workspace/jlens/direction_tokens_full.json --layer 15 --full-probs --out <csv>
# 2. clone the join with the new arms merged on (name, step, abs_pos)
python scripts/merge_probe_rollout_arms.py --extra <csv> \
    --out /workspace/reasoning_theatre/probe_vs_rollout_lb/per_token.csv
# 3. the tables, with the new arms in the headline
python scripts/analyze_probe_rollout.py --per-token .../probe_vs_rollout_lb/per_token.csv \
    --out-dir .../probe_vs_rollout_lb --headline-extra local_belief.p1_mlp,local_belief.p2_mlp
```

### What it found

- **The new arms lead the dumbbell and pass the chance floor.** P1 `.550 / .174` (lift
  **+.377**) and P2 `.513 / .190` (+.322), against +.217 for the best of the original 26.
  28/28 arms lean the same way. Their agreement with the *final* action is **below** the .250
  chance floor — pointed away from the ending, not merely indifferent to it.
- **The revision: a sentence does not open holding its own conclusion.** Entry 39 read the
  within-sentence panel as "the change is already there at the first token" (baseline enters a
  switch sentence at .365 this / .304 prev). Both belief-trained probes instead **open holding
  the previous answer** (.481 / .484) and cross over around the fifth decile. This is entry
  41(i) — which found the same thing in the *probabilities* and said the argmax could not show
  it — now visible in the argmax directly.
- **The baseline inverts at rel −1** (.353 belief vs .375 final), the one point where entry
  39's own claim runs backwards. The new probes hold the split open across the whole approach.
- **They don't pay for it after the boundary**: at rel +2/+3/+4, P2 scores .665/.688/.632
  against the baseline's .618/.650/.591.

### How to read it, and two landmines

- **Ceiling, not competitor.** A probe trained on a belief label will agree with belief more.
  Not circular though: they were trained on the rollout answer at their own *loud cutoff*
  token, while every figure scores agreement at the *sentence end* — related labels, different
  cut points — and they're scored on all 87k tokens having trained only on loud ones.
- **The 26 published arms reproduce their report values exactly** under the re-run (4 dp).
  That is the check that the added arms sit on the same measurement; do it again if this is
  ever rebuilt.
- **⚠️ `analyze_probe_rollout.py` had no `if __name__ == "__main__"` block** — running it
  defined `main()` and exited 0 silently, writing nothing. Uncommitted WIP; anyone re-running
  the entry-39 tables before this fix got an empty success. Fixed.
- **⚠️ `HEADLINE` is not to be edited.** Its own comment records that it drifted from
  `plot_probe_rollout.py` once and the report mixed two arms. Use `--headline-extra`, which
  defaults to empty so existing outputs stay byte-identical, and clone rather than edit.
