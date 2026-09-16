# eval_binary_cognitive_map_probe

Score a saved binary cell-identity probe against a prepared dataset, **one row per
(token, cell)**.

```bash
interp-cli eval_binary_cognitive_map_probe \
    --probe-path /workspace/probes/grid_binary_l15/grid_binary_probe_wall_mlp.pt \
    --data-path  /workspace/prepared/grid_binary_l15_heldout360 \
    --output-path /workspace/probes/grid_binary_l15/eval_wall_mlp_heldout360.json
```

## Why not `eval_cognitive_map_probe`

[`eval_cognitive_map_probe`](../eval_cognitive_map_probe/README.md) walks the gathered tree
and **concatenates** the token indices it is given into one activation per (trajectory,
step). A probe trained on a token-major manifest has `input_dim = D + 2` — one reasoning
token per row — so that command can only ever hand it a single token index. Training reads
every selected token on its own; an evaluation that does anything else is not measuring
what was trained.

This command reads the same artifact the trainer reads: a v3 `grid_tile` manifest
directory. Point it at the split-eval half for the matched comparison and at a held-out
dataset's manifest for the generalisation number — nothing but the manifest changes.

## Options

| Option | Default | Description |
|---|---|---|
| `--probe-path` | *required* | `.pt` written by `train_binary_cognitive_map_probe` |
| `--data-path` | *required* | v3 prepared dataset directory (`probe_type=grid_tile`) |
| `--output-path` | `eval_{probe}_{dataset}.json` beside the probe | Results JSON |
| `--threshold` | `0.5` | Positive-class probability above which a cell is called positive |
| `--batch-size` | `8192` | Rows per forward pass |
| `--cache-activations` | `False` | Shares `_packed_activations.pt` with the trainer |
| `--device` / `--verbose` | | |

## Output

```jsonc
{
  "probe": { "path", "positive_class", "positive_cell_id", "model_type", "input_dim",
             "normalized", "train_config" },
  "data":  { "path", "activations_root", "selection", "split",
             "n_entries", "n_trajectories", "n_cells_per_entry", "n_rows" },
  "config": { "threshold", "batch_size", "device" },
  "global":             { /* metric block */ },
  "by_size":            { "5": {...}, "7": {...}, ... },
  "by_complexity":      { "0.0": {...}, "0.2": {...}, ... },
  "by_size_complexity": { "size5|comp0.0": {...}, ... }
}
```

Each metric block: `n_rows`, `positive_support`, `positive_rate`, `predicted_positive`,
`accuracy`, `balanced_accuracy`, `precision`, `recall`, `specificity`, `f1`, `auroc`,
`tp`, `fp`, `tn`, `fn`.

`size` comes off the manifest entry; complexity is parsed out of the trajectory name
(`..._comp0.4_123`). A grouping key that cannot be resolved is left out of the breakdown
rather than bucketed as `unknown`.

## Notes

- Rows are built a chunk of entries at a time with tensor ops, not one at a time through a
  `Dataset` — at 25 cells per entry that is ~25x fewer Python-level fetches, which is what
  makes the held-out tree's 87k tokens minutes rather than an hour.
- The evaluator reproduces the trainer's own final numbers exactly when pointed at the same
  eval dataset. If it does not, the probe and the dataset disagree about something.
- It refuses a multiclass probe (no `positive_class`) and a dataset whose activation
  dimension does not match the probe's `input_dim - 2`.
