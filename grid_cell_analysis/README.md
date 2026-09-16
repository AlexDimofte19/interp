# `grid_cell_analysis/` — the grid-cell ("cognitive map") line

Everything that decodes **what is in a grid cell** rather than **what the agent does next**. Two
rounds live here, and only the first produced published results.

**Round 1 — the cognitive-map probes.** `train_cognitive_map_probe` over `gather_activations`
trees, scored by `compute_probe_accuracy.py` and drawn by `plots/`. These are the published
grid-cell probes; `reproduce_all.sh`'s `cognitive_map_probes` stage rebuilds them.

**Round 2 — the grid-label twin of the direction arms. This never produced a result.** It asks
whether a token selected for being *grid*-loaded decodes the grid better, exactly as the
direction arms ask it for the action. The scripts are complete and the gather was never run to
completion. Before spending GPU here, read two things:

* `data/jlens/README.md` — the grid vocabulary on disk is the **pre-rule** version (1258 tokens,
  not the 927 of ICLR log entry 19). Nothing published depends on it, because this round is the
  only consumer and it never ran. Settle it first.
* ICLR log entry 55 — `prepare_grid_arms.sh`'s default still yields **4 cells per step instead
  of 25**. `MAX_CELLS` decides affordability: padded to the widest grid every trajectory has 225
  cells, and ~72k entries × 225 is 16.2M rows per epoch.

Run it with `GRID_ROUND=1 ./scripts/reproduce_all.sh`, which is opt-in for exactly this reason.

## Round 2: the arms

Deliberate twins of the `*_next_action_arms.sh` pair in `scripts/` — same tree, same selection
records, same tokens, same layers. **Only the label changes**, so any difference between a grid
arm and an action arm is the label and nothing else.

| Script | Does | Reads | Writes |
|---|---|---|---|
| `gather_grid_arms.sh` | Selects reasoning tokens by **grid** words instead of direction words | `TRAJ`, `jlens/grid_tokens_full.json` | `activations/grid_reasoning_tokens` |
| `prepare_grid_arms.sh` | One `grid_tile` dataset per arm (jlens / logitlens / random) | that tree | `prepared/grid_<arm>` |
| `train_grid_arms.sh` | `train_cognitive_map_probe` per arm, over a seed sweep | `prepared/grid_<arm>` | `probes/grid/*.pt` |
| `grid_round_status.sh` | One snapshot of gather → prepare → train | all three | stdout |

The cell draw is seeded **per (trajectory, step)**, never from the global RNG: arms consume
different numbers of draws, so a shared stream would hand them different cells and the arms
would stop differing only in their tokens.

## Round 3 — the four binary probes

`binary_grid_probes.sh` trains **one-vs-rest** probes -- empty `_`, wall `#`, agent `A`, goal
`G` -- on the mass-era 2,880, scores them on the pinned 720, and scores them again on the
disjoint `heldout360` tree. One script, six stages, all of them resumable; `DRY_RUN=1` prints
the command table and `ONLY="4 5"` runs a subset.

```bash
DRY_RUN=1 ./grid_cell_analysis/binary_grid_probes.sh      # what it would run
./grid_cell_analysis/binary_grid_probes.sh                # the whole sweep (~6 h on one GPU)
```

| Stage | Does | Writes |
|---|---|---|
| 1 | prepare the 3,600 at **native grid size**, `recorded_random` tokens, layer 15 | `prepared/grid_binary_l15_random` |
| 2 | split on the pinned eval names | `..._split_train` (2,880) / `..._split_eval` (720) |
| 3 | prepare `heldout360` -- **every** reasoning token, not a selection | `prepared/grid_binary_l15_heldout360` |
| 4 | `train_binary_cognitive_map_probe`, 4 classes x {lr, mlp} | `probes/grid_binary_l15/grid_binary_probe_<class>_<head>.pt` |
| 5 | `eval_binary_cognitive_map_probe` on the 720 and on the 360 | `probes/grid_binary_l15/eval_<class>_<head>_<split>.json` |
| 6 | `collect_binary_grid_results.py` folds the 16 JSONs into one table | `probes/grid_binary_l15/binary_grid_summary.csv` |

Three things it does differently from round 2, each deliberate:

* **No `--pad-to-size`.** Round 1 and 2 pad to 15, which makes `+` roughly 90% of a size-5
  grid's sampled cells. A binary arm on a padded dataset measures padding. At native size the
  smallest grid is 5x5 = 25 cells, which is exactly why `MAX_CELLS` must stay <= 25: prepare
  requires a uniform cell count and drops whole size folders that cannot supply it.
* **No `--balance-classes-per-trajectory`** -- entry 55's confirmed-live bug. Imbalance is
  handled at train time with `--class-weight balanced`, and read through balanced accuracy and
  AUROC rather than raw accuracy.
* **Evaluation reads prepared manifests, not the tree.** `eval_cognitive_map_probe` concatenates
  the token indices it is given into one activation per (trajectory, step), so it can only ever
  hand a token-major probe a single token. `eval_binary_cognitive_map_probe` scores every
  (token, cell) the manifest names -- the conditions training saw.

`A` and `G` are one cell per grid, so at 25 uniformly drawn cells they are ~0.9% of the rows
and a size-15 grid contains the agent in only ~11% of draws. Those two arms are the ones to
read sceptically; if recall sits at the floor, the fix is a prepare-time option to force-keep
the `A`/`G` cells in the draw, which does not exist yet.

Nothing here is comparable to the round-1 numbers: this dataset drops padding and those did not.

## Scoring and specificity

| Script | Does |
|---|---|
| _(merged away)_ | Scoring a `grid_tile` probe on **every** reasoning token is now `telos_interp/loudness_analysis/score_probes_per_token.py --probe-type grid_tile`; this file and its five duplicated lens-IO helpers are gone. |
| _(merged away)_ | The specificity control behind ICLR log entry 55 — does *direction* loudness predict where the **grid** is decodable? It should not, and does not (−2.4 points of grid decodability against +15.6 of action). Now `telos_interp/loudness_analysis/analysis/probe_accuracy_by_loudness.py --probe-type grid_tile`, whose counts mode pools the per-class counts a grid row carries instead of averaging per-token accuracies. |
| `compute_probe_accuracy.py` | Overall and per-class accuracy of probe predictions written back into trajectory JSONs. |

## `plots/`

Round-1 figures, all consuming either a trajectory JSON with probe outputs or the
`eval_cognitive_map_probe_per_distance` output.

| Script | Draws |
|---|---|
| `plot_object_prediction_rate_by_distance.py` / `..._from_csv.py` | Goal/agent prediction rate vs. distance — the spatial smearing profile. JSON and CSV sources. |
| `plot_per_class_accuracy_by_distance.py` / `..._from_csv.py` | Per-class probe accuracy vs. distance. JSON and CSV sources. |
| `compare_probe_results.py` | Two probe result JSONs, by size / complexity / overall |
| `overall_accuracy.py` | Accuracy vs. grid size for the size-agnostic probes |
| `per_class_metrics.py` | Per-class precision and recall, stacked bars |
| `visualize_decoded_grid.py` | True grid state beside the grid reconstructed from probe predictions |
| `report_plots/` | The five figures used in the write-up |

The two `*_by_distance*.py` pairs share `_facets` and `_complexity_levels` verbatim — they are
the obvious next merge, and are left alone here because this round is dormant.
