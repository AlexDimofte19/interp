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

## Open threads

- Run `scripts/jlens_reasoning_tokens.sh` on the GPU host to produce a real
  `{activations_dir}` tree; nothing local has jlens CSVs yet.
- On the GPU host, run one comp-0.8 trajectory with `--profile` and record the split, then
  tune `--io-workers` against it. If file creation still dominates after threading, the next
  lever is a sharded `.pt` container (one file per (layer, step) instead of per token,
  ~150x fewer inodes) — deliberately not done, because it changes the on-disk contract.
- Confirm on the real model whether `--forward-batch-size 4` changes any CSV byte versus
  `1`. If it does, decide whether the speedup is worth the bf16 noise.
- Cross-check one saved tensor against an existing `gather_activations` run for the same
  trajectory/layer/step/token (`torch.allclose`) — both capture decoder-block outputs from
  the same truncated prefix, so they must match.
- Train and compare: `jlens_direction`/`jlens_direction` vs the `random`/`random` control
  built with the same N, M and seed.

## Methodological caveat

The highest-scoring tokens are literally the direction words (`Ġleft`, `Ġright`, `Ġdown`).
A probe trained on them is partly reading a decision the model has already verbalized, so
accuracy on that dataset alone says little. The random control is what makes the comparison
meaningful — build both.
