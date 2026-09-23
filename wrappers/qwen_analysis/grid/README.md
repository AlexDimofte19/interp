# Qwen P2, grid: the grid-signal, grid-label twin of `qwen_analysis/`

The same four stages as `wrappers/qwen_analysis/`, with two things changed and everything else
held fixed. Trajectories, sample fraction, arms, N=60, seeds and layer are all the same.

1. **The signal** is the pruned Qwen grid vocabulary,
   `data/jlens/qwen/grid_tokens_pruned_qwen3-6-35b-a3b.json` (533 tokens). Tokens are ranked
   by how grid-loaded the lens reads them, not how direction-loaded.
2. **The label** is grid-cell identity, read by **binary** one-vs-rest probes
   (`train_binary_cognitive_map_probe`): empty `_`, wall `#`, agent `A`, goal `G`. This is
   `grid_cell_analysis/binary_grid_probes.sh`'s setup and hyperparameters.

**Not yet run.**

| stage | direction line | grid line |
|---|---|---|
| 1 loudest layer | direction mass, 20% sample | grid mass, **5%** sample, own tree (`qwen_grid_loudness_profile_p05`) |
| 2 selection | `qwen_p2_selection{,_eval}` | `qwen_p2_grid_selection{,_eval}`: a new gather, since the direction tree never saved grid-picked tokens |
| 2 held-out | `qwen_p2_heldout`: tensors + direction mass | tensors **reused**; grid mass tables from a CSV-only pass into `qwen_p2_heldout_grid_lens` |
| 2 prepare | `next_action`, final label | `grid_tile`, 25 cells per (trajectory, step) |
| 2 rollouts | local-belief relabel | **none**: a cell's contents are on `grid_state`, so there is no local belief to measure |
| 3 train | 3 arms x {lr, mlp} | 3 arms x 4 classes x {lr, mlp} = 24 binary probes; plus 3 arms x {lr, mlp} = 6 **multiclass** probes (`train_grid_probes_all_selections.sh`, same data and hyperparameters) |
| 3 cross-eval | `eval_local_belief.py`, 18 cells | `eval_binary_cognitive_map_probe`, 72 cells |
| 4 held-out | per-token scorer -> rollout join -> decile notebook | `prepare_heldout_grid.sh` + `eval_binary_probes_heldout.sh`: overall held-out numbers only |

Run order: `1_loudest_layer/` → `2_dataset_creation/1_p2_selection/*` →
`2_dataset_creation/build_grid_datasets.sh` → `3_train_and_eval_probes/` → `4_loudness_evaluation/`.

## Open decisions and known gaps

- **The layer is inherited, not derived.** `LAYER=27` is the *direction* profile's argmax.
  Run stage 1 and read the notebook first. If the grid argmax differs, change `LAYER`
  everywhere. That also means a full held-out re-gather, because `qwen_p2_heldout` only
  holds layer 27. Never 39: both lenses are identical there.
- **Stage 4 does not bin by loudness yet.** `score_probes_per_token.py --probe-type grid_tile`
  loads a multiclass `CognitiveMapProbe` and cannot score a binary probe. The fix is one
  registry entry in `loudness_analysis/probes.py::PROBE_TYPES` (CLAUDE.md says to add a
  registry entry, not a script). After that, the held-out grid mass tables from stage 2 are
  its `--lens-dir`.
- **The grid datasets are padded to 15.** `prepare_activations_for_probing` auto-pads
  grid_tile in multi-size mode, whatever `binary_grid_probes.sh`'s header claims. So most
  cells drawn from a small grid are `+` padding, i.e. easy negatives. Matches the existing
  gpt-oss binary datasets; fixing it is a code change in prepare.
- **AXIS dominates grid mass** (`data/jlens/README.md`): "grid-loud" is largely
  "row/column-word-loud".
