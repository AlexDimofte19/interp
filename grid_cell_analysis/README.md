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

## Scoring and specificity

| Script | Does |
|---|---|
| _(merged away)_ | Scoring a `grid_tile` probe on **every** reasoning token is now `telos_interp/loudness_analysis/score_probes_per_token.py --probe-type grid_tile`; this file and its five duplicated lens-IO helpers are gone. |
| `analyze_grid_loudness_correlation.py` | The specificity control behind ICLR log entry 55: does *direction* loudness predict where the **grid** is decodable? It should not — and does not (−2.4 points of grid decodability against +15.6 of action). |
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
