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
uv run --extra lint pytest -c pyproject.toml -q --ignore=tests/test_trajectory_activations.py   # 101 pass
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
`jlens/jlens_fit_gpt_oss.py` — cannot run on the laptop. Locally you can only work on the code paths that
consume already-extracted artifacts (CSVs, `.pt` files, manifests, notebooks). Extract activations on a
**single** GPU: `device_map="auto"` across multiple GPUs produces NaNs for this MoE model.

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
with `act_path` relative to the manifest so the directory is movable. `next_action` is the exception: it
copies nothing and its manifest's `samples` point into the original activations tree via
`activations_root`. Loaders live in `prepare_activations_for_probing/manifest_loader.py`.

### The lens line (jlens and logitlens)

`scripts/jlens_reasoning_tokens.py` does one forward pass per trajectory and emits both the activations
and a per-trajectory `{stem}_{lens}_analysis.csv` of top-20 lens predictions per (reasoning token, layer).
Tokens are scored by how many of their top-k lens predictions are direction words, using a vocabulary
JSON (`data/jlens/direction_tokens_full.json`, `data/jlens/grid_tokens_full.json`).

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

Narrow at training time, not prepare time: `split_next_action_manifest.py --tokens-per-trajectory K
--layers-per-token M` takes top-K off one prepared dataset (identical to a `--num-tokens K` prepare),
which is why the top-1/2/3 sweep costs three splits rather than three multi-hour prepares. It computes
the train/eval strata from the unthinned samples so every K shares one split and the results compare.

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

## Conventions and gotchas

- `configs/**/*.conf`, `script.sh`, `general_probe_train.sh`, `*.ps1` are **recorded `interp-cli`
  invocations**, not parsed config files. They are the record of how published results were produced.
- `ICLR log.txt` is the running research log (findings, planned phases, known landmines) and
  `claude_session_readme.md` is the handoff note for the current `reasoning_theatre` branch. Read them
  before touching the jlens → probe path; append to the log rather than rewriting it.
- Known landmine recorded there: `train_cognitive_map_probe_fn.py::_prepare_train_eval_v3` splits
  train/eval with `torch.randperm` over **rows**. Once a trajectory contributes many rows (multi-token
  datasets), that leaks trajectories across the split. Split over unique trajectory names.
- `parse_grid_state` has no symbol for fog (`*`), so fogged cells are silently dropped and padded. Fine
  while every trajectory is `fully_observable: true`.
- `spaces/trace-viewer` is a git submodule (a HuggingFace Space).
- Pre-commit (ruff format + ruff, trailing whitespace, 2500 KB file size cap) runs on commit and push.
