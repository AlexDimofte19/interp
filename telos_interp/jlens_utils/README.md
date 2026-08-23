# `jlens_utils`

Shared, stdlib-only logic for deciding **which reasoning tokens and layers are worth
keeping**. Three consumers, one answer:

| consumer | what it does with the answer |
| --- | --- |
| `scripts/jlens_reasoning_tokens.py` | writes only the selected activations |
| `scripts/delete_non_jlens_selected.py` | deletes everything *but* the selected activations |
| `prepare_activations_for_probing` (`recorded_*` modes) | builds a probe dataset from the record |

Because all three go through `jlens_top_filter`, pruning an existing tree lands on exactly
the files a filtered gather would have written — which is what
`tests/test_delete_non_jlens_selected.py` asserts byte-for-byte.

**No torch.** Importing anything from `telos_interp/commands/prepare_activations_for_probing/`
runs that package's `__init__`, which imports torch; keeping this package independent is what
lets the pruning script and the tests stay light.

## The problem it solves

One forward pass over a 700-token reasoning chain at 17 layers writes ~12k `.pt` files
(~68 MB) **per step**. A probe trains on ~20 tokens × ≤4 layers of that. The rest is written,
stored, and never read — which is how the disk filled up.

## Modules

- **`jlens_csv.py`** — read `{stem}_jlens_analysis.csv` and count, per (token, layer), how
  many of that row's top-k lens predictions are direction words. Also the coordinate math:
  a CSV row's `(step, abs_pos)` maps to `layer_{N}/step_{step_id}/output/{abs_pos -
  output_start}.pt`. Read with `csv.DictReader`, never pandas — decoded tokens include
  `"NA"`, empty strings, commas and newlines, which pandas' NA handling corrupts.
- **`top_filter.py`** — `jlens_top_filter()` returns a `KeptTokens` with **two arms**.
- **`record.py`** — read/write `{stem}_jlens_selection.json`, the provenance file written
  beside the CSV.

## Why two arms

A `jlens_direction` result means nothing without a matched control: the top-scoring tokens
are literally the direction words the model already verbalized. The control is a uniform draw
over the reasoning chain — and **once the tree is pruned to the jlens arm, that draw is no
longer possible**. So the control is reserved *before* the deletion and recorded, rather than
re-derived afterwards.

Two consequences worth knowing:

- The arms are drawn from the same universe, so a token can land in both. That is a
  legitimate outcome of a uniform draw, not a bug; excluding the top scorers would make the
  control systematically low-scoring.
- The random arm's entries **carry no direction counts**, deliberately.
  `scripts/split_next_action_manifest.py` decides whether to *rank* a token's layers or
  *sample* them by whether a count is present, so a control that recorded counts would
  silently collapse onto its lowest-scoring layer.

## Layer 15

Kept for every selected token of both arms, on top of the top-N (so `num_layers=3` gives 4
layers, or 3 when 15 already scored into the top 3). Layer 15 is the project's standing
comparison point, and without forcing it a layer-15 baseline would need its own gather run.
After pruning, `--layers 15` carves one out of the same record.
