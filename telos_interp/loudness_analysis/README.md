# `loudness_analysis` — the lens → probe → loudness pipeline

**Loudness** is the probability mass a lens puts on a *signal* vocabulary at a given token and
layer: `log P(any signal word)`. This module produces it, joins it to what the model and its
probes actually did, and draws the result.

Three axes vary independently, and every stage takes all three:

| axis | what it is | where it comes from |
|---|---|---|
| **lens** | how a layer's residual is read — `jlens`, `logitlens` | `jlens_utils.methods.METHODS` |
| **signal** | which vocabulary the mass is taken over — `direction`, `grid`, anything | [`signals.py`](signals.py) |
| **probe** | what a probe decodes — `next_action`, `grid_tile` | [`probes.py`](probes.py) |

None implies another. The grid analysis is deliberately pointed at the *direction* vocabulary:
the question is whether direction loudness predicts where the **grid** is decodable, and the
answer only means something if the ruler is unchanged.

## The pipeline

```
trajectories ──► build_loudness_tables.py ──► analysis CSV + mass table + selected .pt
                   --lens --signal-json --select-methods

              ──► rollouts/run_inference.py ──► one results JSON per trajectory
                   --strategy {eos, every_token, …}

probes ─────► score_probes_per_token.py ──► per-token table, ALREADY carrying loudness
                   --probe-type {next_action, grid_tile}

              ──► join_rollouts.py ──► + the model's belief, sentence and commitment coords
                   --table … --rollout-dir …

              ──► analysis/ ──► tables + summary.json + run_config.json
              ──► plotting/ ──► figures
```

**Stage 1 and stage 4 are one file.** `build_loudness_tables.py --select-methods` runs the
token selection in the same forward pass that writes the tables, so there is no separate
"select" step.

**`score_probes_per_token.py` output is already a loudness table.** It writes each lens's count
and mass beside every probe's prediction, both lenses in one pass. That is why there is no
"join loudness to probe predictions" script — the loudness is already there, and
`join_rollouts.py` only adds what the *rollout* knows.

**Rollout and join are deliberately separate.** The join is CPU-only and gets re-run once per
lens ruler; a rollout arm is 6–14 GPU-hours. Merging them would put the GPU in the path of
every re-join.

## Files

| File | Does |
|---|---|
| [`signals.py`](signals.py) | `SIGNALS` registry. Seeded with `direction` and `grid`, but any `{class: [tokens]}` JSON registers via `--signal-json` + `--signal-name`. |
| [`columns.py`](columns.py) | `{lens}_{signal}_logmass_L{layer}`, read-aliases for every legacy spelling, and the one axis label every table and figure uses. |
| [`lens_io.py`](lens_io.py) | Reading a gathered tree; and `check_vocabulary`, the `.meta.json` sidecar rule. |
| [`stats.py`](stats.py) | Both balanced accuracies, `qbin`, and the trajectory-clustered bootstrap. |
| [`probes.py`](probes.py) | `PROBE_TYPES` registry: features, labels, row shape, naming. |
| [`provenance.py`](provenance.py) | `run_config.json` per result folder — and the guard against mixing rulers. |
| `build_loudness_tables.py` | Stage 1: lens + signal → analysis CSV, mass table, selection, activations. **GPU.** |
| `rollouts/` | Stage 2: `run_inference.py` and the `STRATEGIES` registry. **GPU.** |
| `score_probes_per_token.py` | Stage 3: every probe on every reasoning token. **GPU.** |
| `join_rollouts.py` | The join, from a per-token **table**. CPU. |
| `join_rollout_answers.py` | The join, from a **mass tree**, against the rollout's answer probability, under both lenses. CPU. |
| `build_sentence_loudness.py` | The join, from a **mass tree**, placing each token in its sentence. CPU. |
| `summarise_probe_accuracy.py` | One row per probe: balanced accuracy vs belief and vs final. Not a loudness analysis — a scoreboard. |
| [`analysis/`](analysis) | Two analysers; neither touches a lens, both bin a column. |
| [`plotting/`](plotting) | 16 figures behind one registry, plus the loudness × chain-position heatmap. |

## Conventions that are not negotiable

**Read lens CSVs with `csv.DictReader`, never `pandas.read_csv`.** Decoded tokens include the
literal string `"NA"`, empty strings, embedded commas and newlines, all of which pandas' NA
handling silently corrupts. Where pandas is used it is always with
`keep_default_na=False, na_values=[""]`.

**Never read a mass table without its `.meta.json` sidecar.** The cells are `log P(any signal
word)` over *some* vocabulary, and this repo points several vocabularies at the same trees.
`lens_io.check_vocabulary` is the difference between comparing two rulers and comparing two
different questions.

**`abs_pos` is prompt-inclusive; `token_idx` is `output_tokens`-relative.** Joining a rollout
on `abs_pos` yields an empty or wrong join, never an error.

**Loudness is never unqualified.** At layer 15 the two lenses' top-20 sets overlap only about
half, so "loudness" without a lens and a vocabulary is not a quantity. `columns.axis_label`
exists so no figure can name its ruler differently from the table it was drawn from.

**The two balanced accuracies are not the same number.** `stats.bal_acc` averages over rows;
`stats.bal_acc_from_counts` pools per-class counts. A grid row summarises a whole step's
cells, so averaging per-token accuracies would weight a token with 2 cells the same as one
with 25. `run_config.json` records which one ran.

## Adding a signal

Write the vocabulary JSON, then pass it. Nothing else changes:

```bash
python telos_interp/loudness_analysis/build_loudness_tables.py \
    --signal-json data/jlens/shape_tokens_full.json --signal-name shape \
    --lens both --select-methods jlens,random ...
```

Columns become `jlens_shape_logmass_L15`, labels become "J-lens shape logmass", and the mass
table's sidecar records the name and a hash of the vocabulary's contents — because the path
alone cannot decide comparability: the same vocabulary is `data/jlens/…` in the repo and
`/workspace/jlens/…` when deployed, and a file edited in place keeps its path.

Build vocabularies **from the model vocabulary alone**, never from what is frequent in the
j-space — the j-space is what they are used to measure. See `notebooks/direction_tokens.ipynb`.
