# Session context: jlens → probe pipeline

Notes for a fresh Claude session picking up this work. Branch: `reasoning_theatre`.

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
