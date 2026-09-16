# train_binary_cognitive_map_probe

Train a **one-vs-rest** cell-identity probe: one probe per grid symbol, deciding "is this
cell a wall" rather than "which of the eight symbols is this cell".

```bash
interp-cli train_binary_cognitive_map_probe \
    --train-data-path /workspace/prepared/grid_binary_l15_random_split_train \
    --eval-data-path  /workspace/prepared/grid_binary_l15_random_split_eval \
    --positive-class wall \
    --model-type mlp \
    --output-path /workspace/probes/grid_binary_l15/grid_binary_probe_wall_mlp.pt \
    --class-weight balanced --normalize --cache-activations
```

## Why a separate command

[`train_cognitive_map_probe`](../train_cognitive_map_probe/README.md) fits a single N-way
softmax over whatever cell ids the manifest holds. That is the right shape for "decode the
map"; it is the wrong shape for "how legible is each symbol", because a class with 0.9% of
the rows contributes one row of a confusion matrix and no threshold of its own. The two
trainers share every label-agnostic helper — loading, the packed-activation cache, class
weighting, balancing, normalisation, the token-major guard — and differ in the label and
the metrics.

## Options

| Option | Default | Description |
|---|---|---|
| `--train-data-path` | *required* | v3 prepared dataset directory (`probe_type=grid_tile`) |
| `--positive-class` | *required* | Grid symbol or alias: `_`/`empty`, `#`/`wall`, `A`/`agent`, `G`/`goal`, `D`/`door`, `K`/`key`, `?`/`unknown`, `+`/`padding` |
| `--model-type` | `lr` | `lr` (linear) or `mlp` |
| `--eval-data-path` | `None` | Separate v3 dataset to evaluate on. **Required** for a token-major manifest |
| `--output-path` | next to the dataset | Where to write the probe `.pt` |
| `--num-epochs` | `50` | |
| `--learning-rate` | `3e-4` | |
| `--batch-size` | `2048` | |
| `--weight-decay` | `1e-3` | |
| `--hidden-dims` | `1024` | Comma-separated MLP hidden sizes; ignored for `lr` |
| `--dropout` | `0.0` | MLP only |
| `--eval-split` | `0.2` | Only used when `--eval-data-path` is absent |
| `--subset` | `1.0` | Refused below 1.0 on a token-major manifest |
| `--class-weight` | `balanced` | `balanced` or `None`. Defaults **on** here, unlike the multiclass trainer |
| `--balance-classes` | `False` | Upsample the minority class to the majority count |
| `--normalize` | `True` | Standardize the activation dims; the two position columns pass through |
| `--device` / `--seed` / `--verbose` | | |
| `--per-class-max-count` | `None` | Cap per class when `--balance-classes` upsamples |
| `--cache-activations` | `False` | `_packed_activations.pt` beside the manifest; shared by every arm over one dataset |

## What the probe carries

`BinaryCognitiveMapProbe.save` writes the same keys as `CognitiveMapProbe` plus
`positive_class` / `positive_cell_id`, so a probe on disk names what it decides without a
filename convention. Two details matter to anything reading it back:

- **`label_to_idx` covers every cell id** in `CELL_SYMBOL_TO_ID`, mapping each to 0 or 1.
  A consumer that binarises ground truth with `label_to_idx[cell_id]` therefore agrees
  with training and never meets a missing key, whatever symbols the training split held.
- **`idx_to_label` is `{0: -1, 1: <positive cell id>}`**, and `num_classes` reads off
  *that*, not off `label_to_idx`. `-1` is deliberately outside the cell-id range: the
  negative class is "every symbol but one", which no single id names, so a lookup in
  `CELL_ID_TO_SYMBOL` fails loudly instead of printing `A`.

## Metrics

Per epoch and at the end: accuracy, **balanced accuracy** (mean of recall and specificity),
positive-class precision / recall / F1, and **AUROC** (rank-based with tie correction, so
it costs one sort and no curve; NaN when a class is absent, never 0.5).

Read balanced accuracy and AUROC. Raw accuracy is meaningless for `A` and `G`, which are
one cell per grid — at 25 uniformly drawn cells they are ~0.9% of the rows, and a probe
that never fires scores 99.1%.

## Landmines

- **Only v3 manifest directories.** The legacy v1 flat-`.pt` path is not carried over.
- **The token-major guard is kept.** A `grid_tile` manifest prepared with a token selection
  has ~20 entries per trajectory sharing one grid, so an internal `--eval-split` (or
  `--subset < 1.0`) leaks. Both are refused; pre-split with
  `scripts/split_next_action_manifest.py` and pass `--eval-data-path`.
- **Prepare without `--pad-to-size`.** Padding to 15 makes `+` ~90% of a size-5 grid's
  sampled cells, and every binary arm then measures padding.
