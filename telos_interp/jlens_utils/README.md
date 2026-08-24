# `jlens_utils`

Shared, stdlib-only logic for deciding **which reasoning tokens and layers are worth
keeping**. Three consumers, one answer:

| consumer | what it does with the answer |
| --- | --- |
| `scripts/jlens_reasoning_tokens.py` | writes only the selected activations |
| `scripts/delete_non_jlens_selected.py` | deletes everything *but* the selected activations |
| `prepare_activations_for_probing` | builds a probe dataset from the CSV or the record |

Because all three go through `top_filter` (and the ranking helpers it is built from),
pruning an existing tree lands on exactly the files a filtered gather would have written —
which is what `tests/test_delete_non_jlens_selected.py` asserts byte-for-byte.

**No torch.** Importing anything from `telos_interp/commands/prepare_activations_for_probing/`
runs that package's `__init__`, which imports torch; keeping this package independent is what
lets the pruning script and the tests stay light.

## The problem it solves

One forward pass over a 700-token reasoning chain at 17 layers writes ~12k `.pt` files
(~68 MB) **per step**. A probe trains on ~20 tokens × ≤4 layers of that. The rest is written,
stored, and never read — which is how the disk filled up.

## Methods

A **method** is one named recipe for choosing (token, layers). Everything is keyed by method
name, so adding one is a registry entry in `methods.py` rather than a branch in four files.

| method | CSV it scores | ranks or samples |
| --- | --- | --- |
| `jlens` | `{stem}_jlens_analysis.csv` | ranks by direction count |
| `logitlens` | `{stem}_logitlens_analysis.csv` | ranks by direction count |
| `random` | — | seeded uniform draw |

`jlens` and `logitlens` differ **only** in which CSV they read. The two CSVs share a schema,
so every line of scoring, ranking and coordinate code serves both; what differs is upstream,
in how the CSV was produced — the Jacobian lens transports a layer's residual stream into the
layer-23 space before the unembed, the logit lens unembeds it where it sits. At layer 23 the
jlens transport is the identity, so the two CSVs' rows there are literally the same numbers,
which `test_the_lenses_agree_at_the_target_layer` uses as a free end-to-end check.

The second axis is `scored`. It is a property of the method, not a knob, because an unscored
arm must record **no** direction counts (see below).

`random` has no CSV of its own, but it is not CSV-free: it samples over the reasoning chain,
and a lens CSV's rows are what enumerate that chain. So it needs a lens alongside it —
`top_filter` raises rather than silently returning an empty control. With two lenses read, it
draws over the union, so it can reach anything either lens could have picked.

## Modules

- **`methods.py`** — the registry: `METHODS`, `get_method`, `parse_methods`,
  `analysis_csv_path`. Also `abbrev`, which is what `prepare_activations_for_probing` puts in
  an auto-generated dataset directory name; it lives on the method so adding one cannot leave
  a hand-maintained lookup stale.
- **`jlens_csv.py`** — read an analysis CSV and count, per (token, layer), how many of that
  row's top-k lens predictions are direction words. Also the coordinate math: a CSV row's
  `(step, abs_pos)` maps to `layer_{N}/step_{step_id}/output/{abs_pos - output_start}.pt`.
  Read with `csv.DictReader`, never pandas — decoded tokens include `"NA"`, empty strings,
  commas and newlines, which pandas' NA handling corrupts.
- **`top_filter.py`** — `top_filter()` returns a `KeptTokens`, one arm per method.
  `rank_tokens` and `rank_layers_by_direction` are the two orderings, shared with
  `jlens_token_selection.py` so a prepared dataset and a pruned tree cannot drift.
- **`record.py`** — read/write `{stem}_jlens_selection.json`, the provenance file written
  beside the CSVs.

## Why a control arm

A lens result means nothing without a matched control: the top-scoring tokens are literally
the direction words the model already verbalized. The control is a uniform draw over the
reasoning chain — and **once the tree is pruned to a lens arm, that draw is no longer
possible**. So the control is reserved *before* the deletion and recorded, rather than
re-derived afterwards.

Three consequences worth knowing:

- The arms are drawn from the same universe, so a token can land in several. That is a
  legitimate outcome of a uniform draw, not a bug; excluding the top scorers would make the
  control systematically low-scoring.
- An unscored arm's entries **carry no direction counts**, deliberately.
  `scripts/split_next_action_manifest.py` decides whether to *rank* a token's layers or
  *sample* them by whether a count is present, so a control that recorded counts would
  silently collapse onto its lowest-scoring layer.
- `arm_seed()` freezes `random`'s draw formula at the bare `f"{seed}-{seed_key}"`. The tree
  on disk was pruned to controls drawn with it; changing the formula would make them
  irreproducible. A method added later gets its own stream.

## The record, and its two formats

`{stem}_jlens_selection.json` — the filename is fixed, because the pruned tree is already
full of them.

**v2** keys the arms by method name and gives each its own config block:

```json
{"format_version": 2, "stem": "...", "model": "...",
 "arms": {"jlens":     {"config": {...}, "picks": [...]},
          "logitlens": {"config": {...}, "picks": [...]},
          "random":    {"config": {...}, "picks": [...]}}}
```

Per-arm config is load-bearing: an arm added later by an incremental gather carries that
run's parameters, while the arms already on disk carry the original pruning run's.

**v1** — a flat `jlens`/`random` pair of lists and one shared `config` — is still read and
must stay readable. Every trajectory pruned before methods existed has one, and it is the
only surviving trace of that trajectory's control arm.

`merge_records()` is what lets a new arm be added without disturbing the others: an arm the
new record does not mention keeps its picks and its config verbatim.

## Layer 15

Kept for every selected token of every arm, on top of the top-N (so `num_layers=3` gives 4
layers, or 3 when 15 already scored into the top 3). Layer 15 is the project's standing
comparison point, and without forcing it a layer-15 baseline would need its own gather run.
After pruning, `--layers 15` carves one out of the same record. It is deliberately *not* a
method — it is a guarantee about every arm.
