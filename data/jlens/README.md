# Lens signal vocabularies

The two token vocabularies the lens scoring is measured against. Both are copies of what is
deployed on the GPU host at `/workspace/jlens/`, taken 2026-09-04, and both are consumed as
`--signal-json` / `--direction-mass-json` / `--direction-tokens-path`.

| file | classes | tokens | used by |
|---|---|---|---|
| `direction_tokens_full.json` | UP, DOWN, LEFT, RIGHT | **446** | every completed experiment in the direction / next-action line |
| `grid_tokens_full.json` | WALL, OPEN, GOAL, AGENT, AXIS, STATUS | **1258** | the grid-probing round only -- which never ran |

Deploy them with `scripts/reproduce_all.sh` stage `deploy_vocabularies`, or by hand:

```bash
cp data/jlens/*.json /workspace/jlens/
```

Both are built **from the model vocabulary alone** by `notebooks/direction_tokens.ipynb` and
`notebooks/grid_tokens.ipynb` -- never from what is frequent in the j-space, since the j-space
is what they are used to measure. Every seed and anchor must itself be a gpt-oss-20b token
(`admissible()`, cell 2 of both notebooks). Keep that separation.

## Known discrepancy: the grid vocabulary is the pre-rule version

ICLR log entry 19 records the model-token-only rule reducing **direction 540 -> 446** and
**grid 1258 -> 927**. The direction file here is the post-rule 446. The grid file here is
**1258**, i.e. the version from *before* the rule was applied, even though the host copy is
dated after it.

It is committed as-is on purpose: it is what is deployed, so it is what a re-run would use,
and rewriting it would change the vocabulary silently. Nothing published depends on it -- the
only consumer is the grid-probing round, which never started (log entries 32-35), so no result
was ever measured against it.

Before spending GPU on the grid round, regenerate it by running `notebooks/grid_tokens.ipynb`
start to finish and confirm the count lands at 927. Note that log entry 20 flags ~95 junk
tokens that the dropped dictionary now admits, concentrated in ten seeds
(`objectif`, `pared`, `livre`, `agent`, `ligne`, `position`, `barrier`, `blocked`, ...) -- the
recommended fix is to drop those seeds and re-run, and it was never applied. So a naive
regeneration produces a *third* version, not the 927 of entry 19. Decide which is wanted
before gathering against it.

The direction vocabulary needs none of this: 446 is the post-rule count, and it is what every
number in the research summary was measured with.

## Audit of the grid vocabulary, and what to prune

The regeneration above was never done, so the 1258-token file is still what a grid run
would use. Auditing it against real lens output says it should not be: `scripts/prune_grid_vocabulary.py`
drops **616 of the 1258** by named rule, and the number that matters is not the 616.

Measured on the 487,050 `(token x layer)` jlens rows of 120 trajectories in
`/workspace/activations/heldout360_lens_grid` -- a grid-signal gather that already exists,
alongside its direction twin in `heldout360_lens`:

| | full 1258 | pruned 642 |
|---|---|---|
| tokens that ever reach a top-20 | 428 (34%) | 293 (46%) |
| grid probability mass captured | 100% | 92.2% |
| mass from the top 5 tokens | -- | 50.4% |
| mass from the top 34 tokens | -- | 90% |
| AXIS share of all grid mass | 75.9% | 77.7% |

**Three findings, in order of how much they matter.**

1. **The vocabulary is effectively 34 tokens wide, not 1258.** Half the captured mass is
   ` row`, ` col`, ` column`, ` position`, ` coordinates`; 90% is 34 tokens; the remaining
   608 kept tokens split 10% between them and 830 of the original 1258 never fire at all.
   The count in the table above is decoration. Anything reported as "the grid signal" is
   dominated by five words.

2. **AXIS is 76% of the mass, and pruning cannot fix that** -- it nudges the share *up*, to
   78%. "Grid loudness" as this vocabulary defines it is really row/column-word loudness,
   which is a claim about coordinate bookkeeping, not about a cognitive map. The three
   classes a map would live in are marginal: WALL 4.8%, OPEN 3.6%, AGENT 2.5%. `Signal.load`
   flattens every class into one set, so this imbalance is invisible at the call site; the
   `classes=` argument is the existing lever, and a grid run should either score the classes
   **separately** or drop AXIS.

3. **OPEN cannot be measured lexically at all.** Over 268,097 sampled reasoning tokens the
   model emits an OPEN-class word 0.05% of the time -- it never verbalises empty cells, it
   writes `_`. That is the symbol gap the notebook's closing cell already flags, and no
   vocabulary fixes it.

**What the prune removes.** 7.8% of measured mass, concentrated in two families:

- `generic_modality` / `code_control_flow` -- ` cannot` alone is 2.8%, plus ` through`,
  ` fails`, ` pass`, ` unable`. STATUS loses 91 of its 120 tokens and 57% of its own mass;
  as built it is a negation class, not a grid class. The grid sense (` blocked`,
  ` unreachable`, ` reachable`, ` blockage`) is kept and is what remains.
- `code_index` / `english_line` -- ` index` + ` indexes` + variants are 3.3%, ` line` 0.4%.
  Programming words that entered AXIS beside `row`/`col`.

The rest is inert but indefensible, and it is what makes the file misleading to read:
`difflib_wall_neighbour` (`pared` -> ' spared', ' parsed', ' pred', ' parked', ' Parte';
`barrier` -> ' carrera', ' Karriere'; `mur` -> ' Mahl', ' Maus', ' mambo'),
`wall_drift_challenge` (`obstacle` -> ' challenge', ' hurdle', ' défi', ' Herausforderung'),
`anchor_drift_case` and `anchor_drift_field` (the French `case vide` and German `freies Feld`
anchors reaching the programming `case` and `field`), `anchor_drift_free` (`free square`
reaching the gambling tokens `เงินฟรี` and ` бесплат`), `geo_drift` (latitude/longitude/GPS),
`goal_drift_travel` (`destination` -> ' viajar', ' reisen', ' goalie'), and `foreign_noise`
(` полиция`, `总代理联系`, `不能提现`).

Two entries deserve singling out:

- **The bare lowercase letters.** `a`, ` a`, `,a`, `#a`, `:a` ... are in AGENT and `g`, ` g`,
  `,g`, `#g` in GOAL, 54 tokens between them. The grid legend's glyphs are uppercase `A` and
  `G`; the lowercase arms are the English article and a stray letter, and they fire on
  ordinary text. They contribute little mass but a broad noise floor, and the prune is
  case-sensitive precisely so `A` and `G` survive it.
- **` directional` and `Directional` are in AXIS.** They belong to the *direction*
  vocabulary's subject matter. Leaving them in makes grid loudness and direction loudness
  non-independent, which defeats the point of having two signals to compare.

**Not dropped, but wrong where they are:** ` square`/`-square`/`(square` sit in GOAL (from
the `target square` anchor) when they describe a cell, and the whole `position`/`positioned`/
`posição` family sits in AGENT when AXIS already owns `position`. Both are class-assignment
errors rather than junk -- reclassify, do not delete.

```bash
python scripts/prune_grid_vocabulary.py            # dry run: counts per class and per rule
python scripts/prune_grid_vocabulary.py --write    # -> grid_tokens_pruned.json + the report CSV
```

The prune is applied to the committed 1258 file and written to a **new** path; nothing
deployed is touched, because a mass table is baked against a vocabulary at gather time and
every table on disk names its own by content hash. `grid_tokens_prune_report.csv` carries one
row per token with the rule that dropped it, so the whole list is auditable without rerunning
anything.

**This does not replace regenerating the notebook.** The prune treats symptoms in the output;
log entry 20's recommendation -- drop the ten bad seeds and re-run -- treats the cause, and
findings 1-3 above are properties of the *method* that a regeneration will reproduce. Prune
if a grid run has to happen against what exists; regenerate if there is GPU budget to do it
properly. Either way, decide the AXIS question first: it is worth more than the token list.
