# prepare_activations_for_probing

Prepare activations from `gather_activations` output for training probing classifiers.

## Overview

This command loads activations from the nested folder structure produced by `gather_activations`, combines them with metadata from trajectory JSON files, and prepares them for different probing tasks.

## Probe Types

The trainer's logical view of each probe type is given below. The on-disk layout (v3 manifest dir) is described under **Output Format**.

### `grid_tile` (default)

Predicts grid cell identity from activations.

- **Per-trajectory activation**: `(D,)` — one shared activation per trajectory, written once.
- **Per-cell metadata** (in manifest): `positions` `(C, 2)` and `labels` `(C,)` for each trajectory.
- **Trainer input rows**: `(activation_dim + 2,)` — activation concatenated with `[row_id, col_id]`, materialized lazily by `GridTileCompactDataset`.

Cell type mapping:
| Symbol | ID | Meaning |
|--------|-----|---------|
| A | 0 | Agent |
| # | 1 | Wall |
| G | 2 | Goal |
| _ | 3 | Empty |
| D | 4 | Door |
| K | 5 | Key |
| ? | 6 | Unknown |
| + | 7 | Padding |

### `distance`

Predicts A* distance to goal from activations.

- **Per-trajectory activation**: `(D,)`.
- **Label**: int A* distance from `grid_params.astar_distance` (in manifest as `astar_distance`).
- **Trainer input rows**: `(D,)` activation, `(T,)` labels.

### `action_sequence`

Predicts the sequence of actions taken in a trajectory.

- **Per-trajectory activation**: `(D,)`.
- **Label**: variable-length action list (in manifest as `actions`); loader pads to `max_seq_len` with -1.
- **Trainer input rows**: `(D,)` activation, `(T, max_seq_len)` labels.

Action mapping: `{LEFT: 0, TOP: 1, RIGHT: 2, DOWN: 3}`

### `next_action`

Predicts the agent's next action from a **single end-of-sentence (EOS) token activation**
taken from the reasoning chain. Each gathered EOS token becomes one i.i.d. training sample,
and every EOS token from a trajectory shares that trajectory's `agent_action` label.

- **Sample = one token**: each entry references an **existing** gathered token `.pt` file —
  **no activations are copied** and no new `.pt` files are written. The output directory
  contains only `manifest.json`.
- **Source tokens**: the `output` category only (the reasoning chain). EOS filtering happens
  at **gather time**, so you must run `gather_activations` with `--output-indices eos`; this
  mode then consumes whatever `output` tokens were gathered.
- **Single layer assumed**: pass a single layer via `--layers`. (If more than one layer is
  selected, one sample is emitted per `(token, layer)`.)
- **Label**: the `agent_action` of the step the token came from, mapped via
  `{LEFT: 0, UP: 1, RIGHT: 2, DOWN: 3}` (`TOP` is accepted as an alias for `UP`).
- **Trainer input rows**: `(D,)` activation (`D = hidden_dim`), `(N,)` labels.

Trained with the `train_next_action_probe` command.

#### Choosing which tokens and layers to train on

By default every gathered token of every selected layer becomes a sample. Two orthogonal
knobs narrow that down, so a probe can be trained on the reasoning positions where the
Jacobian lens says direction information lives — and against matched controls.

The mode names are **generated** from `telos_interp.jlens_utils.METHODS`, so every
registered lens has both of its modes and a new one needs no edit here.

| `--token-selection` | which reasoning tokens become samples |
|---|---|
| `all` (default) | every token matching `--output-indices` |
| `jlens_direction` | the `--num-tokens` tokens whose **Jacobian** lens top-k contains the most direction tokens |
| `logitlens_direction` | the same, scored against the **logit** lens CSV |
| `random` | `--num-tokens` tokens drawn uniformly (the control for either lens) |
| `recorded_jlens` | the jlens arm of `{name}_jlens_selection.json`, chosen when the tree was pruned |
| `recorded_logitlens` | the logitlens arm of that same record |
| `recorded_random` | the control arm of that same record |

**On a pruned tree, use the `recorded_*` modes.** Once
`scripts/delete_non_jlens_selected.py` (or a `--signal-json` gather) has removed everything
outside the selection, a `<lens>_direction` mode would still find the right tokens — but
`random` would draw from the survivors, which is no longer a uniform draw over the reasoning
chain. The control only exists in the record, so it has to be read back rather than
recomputed. For `recorded_*`, `--num-tokens` is an optional cap on each arm's top-K,
`--num-layers` and `--direction-tokens-path` are unused (the record fixed them), and
`--layer-selection` must stay `spec` — narrow with `--layers` instead.

A `recorded_*` mode naming an arm the record does not hold selects **nothing** (reported
under `--verbose`), which on an empty result surfaces as prepare's usual "no activations were
extracted". That is correct rather than a bug: the arm you name is the arm you get. Add a
missing arm with `scripts/jlens_reasoning_tokens.py --extend` — pruning removed the tokens it
would need, so it cannot be re-derived here.

| `--layer-selection` | which layers of each selected token become samples |
|---|---|
| `spec` (default) | every layer in `--layers` — use `--layers 15` for a fixed middle layer |
| `jlens_direction` / `logitlens_direction` | that token's top `--num-layers` layers by direction count |
| `random` | `--num-layers` layers drawn uniformly from the candidate pool |

When both axes name a lens they must name the **same** one — a selection reads one CSV.

`--layers` always defines the **candidate pool**; the non-`spec` modes pick within it. Layers
are chosen per token, so two tokens may contribute different layers.

The direction counts come from the `{trajectory_name}_{lens}_analysis.csv` that
`scripts/jlens_reasoning_tokens.py` writes next to the activations: for each
`(reasoning token, layer)` row, the score is how many of `top_1..top_k` appear in
`--direction-tokens-path` (a JSON mapping `UP`/`DOWN`/`LEFT`/`RIGHT` to token strings). A
token's trajectory-level score is the sum over the candidate layers; ranking is per
trajectory, pooling all steps. `--direction-classes` restricts which lists count (default:
the union of all four, so the selection does not depend on the label). Trajectories with no
CSV for the named lens are skipped. The two lenses' CSVs share a schema and differ only in
filename — and in which layers they cover, since the Jacobian lens can only score layers it
has a fitted matrix for.

Samples chosen through a `<lens>_direction` mode carry `token`, `direction_count` and
`layer_direction_count` in the manifest for later analysis; the manifest also records the
full selection under a `selection` key. Those two numbers hold the score under the
manifest's `direction_score` mode — a count, or a (negative) logprob — and higher is better
in either, which is what lets `split_next_action_manifest.py` rank by them without knowing
which mode ran. Random draws are seeded per trajectory from `--seed`,
so a dataset is reproducible regardless of trajectory ordering. Each combination is a
separate prepared dataset — the auto-generated directory name encodes it
(e.g. `..._next_action_tokjl20_layjl3`, or `tokll20` for the logit lens — the abbreviation
comes from the registry).

## Folder Modes

### Single-folder mode

```
activations_dir/
  {trajectory_name}/
    {model_name}/
      layer_{N}/step_{M}/{category}/{token_id}.pt

trajectories_dir/
  {trajectory_name}.json
```

### Multi-size mode

Automatically detected when directories contain `sizeN` subfolders:

```
activations_dir/
  size5/
    {trajectory_name}/...
  size7/
    {trajectory_name}/...

trajectories_dir/
  size5/
    {trajectory_name}.json
  size7/
    {trajectory_name}.json
```

In multi-size mode:
- Each size folder is processed
- Results are merged into a single manifest with per-size subdirs under `activations/`
- `pad_to_size` is auto-set to the maximum size for consistent merging
- Each manifest entry carries a `size: int` field

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `activations_dir` | str | required | Directory containing activation folders |
| `trajectories_dir` | str | required | Directory containing trajectory JSON files |
| `probe_type` | str | "grid_tile" | Type of probe: "grid_tile", "distance", "action_sequence", or "next_action" |
| `layers` | str | "all" | Layer indices to extract |
| `steps` | str | "0" | Step indices (first step used for grid parsing) |
| `pad_to_size` | int \| None | None | Pad grid to this size (auto-detected if None) |
| `max_positions_per_trajectory` | int \| None | None | Max cell positions per trajectory (grid_tile only) |
| `balance_classes_per_trajectory` | bool | False | Balance cell type classes (grid_tile only) |
| `prompt_prefix_indices` | str \| None | None | Token indices for prompt_prefix |
| `prompt_suffix_indices` | str \| None | None | Token indices for prompt_suffix |
| `grid_state_indices` | str \| None | None | Token indices for grid_state |
| `output_indices` | str \| None | None | Token indices for output |
| `output_path` | str \| None | None | Output **directory** (auto-named under `activations_dir` if None). If you pass a path ending in `.pt`, the suffix is stripped with a warning. |
| `verbose` | bool | False | Print detailed progress |
| `seed` | int | 42 | Random seed for reproducibility |
| `token_selection` | str | "all" | `next_action` and `grid_tile` only: "all", "random", "<lens>_direction", "recorded_<method>" — generated from `jlens_utils.METHODS` |
| `layer_selection` | str | "spec" | `next_action` and `grid_tile` only: "spec", "random", or "<lens>_direction" |
| `token_major` | bool | False | `grid_tile` only: one sample per gathered token instead of one per trajectory. Implied by any `token_selection`; the flag is for a tree with no selection record to read |
| `num_tokens` | int \| None | None | N for the non-"all" modes; an optional top-K cap for `recorded_*` |
| `num_layers` | int \| None | None | M for the non-"spec" `layer_selection` modes |
| `direction_tokens_path` | str \| None | None | JSON of direction tokens; required by any `<lens>_direction` mode |
| `direction_classes` | str | "all" | Which direction lists to count (e.g. "UP,DOWN") |
| `direction_score` | str | "count" | How a token's direction evidence scores: "count", "logprob_mass", "logprob_sum" (all from the analysis CSV's top-k columns) or "logprob_mass_full" (from the wide direction-mass table, i.e. over the whole vocabulary rather than a top-20 window) — from `jlens_utils.SCORES`. Only the `<lens>_direction` modes use it; a `recorded_*` selection takes the score the pruning run computed and rejects the flag |
| `jlens_top_k` | int | 20 | How many `top_i` columns of the jlens CSV to scan |

## Examples

### Grid tile probing (single size)

```bash
interp-cli prepare_activations_for_probing \
    --activations-dir /path/to/activations/size5 \
    --trajectories-dir /path/to/trajectories/size5 \
    --probe-type grid_tile \
    --layers all \
    --steps 0 \
    --output-indices -1
```

### Grid tile with class balancing

```bash
interp-cli prepare_activations_for_probing \
    --activations-dir /path/to/activations/size7 \
    --trajectories-dir /path/to/trajectories/size7 \
    --probe-type grid_tile \
    --output-indices all \
    --balance-classes-per-trajectory \
    --max-positions-per-trajectory 100
```

### Distance probing

```bash
interp-cli prepare_activations_for_probing \
    --activations-dir /path/to/activations/size5 \
    --trajectories-dir /path/to/trajectories/size5 \
    --probe-type distance \
    --layers all \
    --output-indices -1
```

### Action sequence probing

```bash
interp-cli prepare_activations_for_probing \
    --activations-dir /path/to/activations/size5 \
    --trajectories-dir /path/to/trajectories/size5 \
    --probe-type action_sequence \
    --layers all \
    --output-indices -1
```

### Next action probing (single EOS token → action)

Gather first with EOS output tokens, then prepare a single layer:

```bash
# 1. gather only sentence-ending output tokens
interp-cli gather_activations ... --output-indices eos

# 2. prepare next_action samples (manifest only; no .pt files copied)
interp-cli prepare_activations_for_probing \
    --activations-dir /path/to/activations/size11 \
    --trajectories-dir /path/to/trajectories/size11 \
    --probe-type next_action \
    --layers 15 \
    --steps 0 \
    --output-indices all \
    --verbose
```

### Next action on the most direction-loaded reasoning tokens (pruned tree)

The normal path. Run `scripts/jlens_reasoning_tokens_filtered.sh` first (or
`scripts/delete_non_jlens_selected.sh` over an existing full tree) — either writes the
`{name}_jlens_selection.json` these modes read.

```bash
# both arms, from the same record; --layers narrows the recorded layers
for arm in jlens random; do
    interp-cli prepare_activations_for_probing \
        --activations-dir /workspace/activations/jlens_reasoning_tokens \
        --trajectories-dir /workspace/trajectories/trajectories_test_full \
        --probe-type next_action \
        --layers 7:23 --steps all --output-indices all \
        --token-selection "recorded_${arm}" \
        --output-path /workspace/prepared/next_action_${arm}
done
```

Narrow to the top-K tokens at *training* time rather than here —
`scripts/split_next_action_manifest.py --tokens-per-trajectory K --layers-per-token 1`
takes top-1/2/3 off a single prepared dataset, so a sweep costs three splits instead of
three prepares.

### The same, scoring the CSV directly (unpruned tree only)

```bash
# top 20 tokens per trajectory, each at its own top 3 layers
interp-cli prepare_activations_for_probing \
    --activations-dir /workspace/activations/jlens_reasoning_tokens \
    --trajectories-dir /workspace/trajectories/trajectories_test_full \
    --probe-type next_action \
    --layers 7:23 \
    --steps all \
    --output-indices all \
    --token-selection jlens_direction \
    --layer-selection jlens_direction \
    --num-tokens 20 \
    --num-layers 3 \
    --direction-tokens-path /workspace/jlens/direction_tokens_full.json

# matched control: same N and M, drawn at random
interp-cli prepare_activations_for_probing \
    ... same dirs/layers/steps ... \
    --token-selection random --layer-selection random \
    --num-tokens 20 --num-layers 3

# conventional wisdom baseline: random tokens, fixed middle layer
interp-cli prepare_activations_for_probing \
    ... same dirs/steps ... \
    --layers 15 \
    --token-selection random --num-tokens 20
```

### Multi-size mode (automatic)

```bash
interp-cli prepare_activations_for_probing \
    --activations-dir /path/to/activations \
    --trajectories-dir /path/to/trajectories \
    --probe-type grid_tile \
    --output-indices all \
    --verbose
```

### Specific layers and custom output path

```bash
interp-cli prepare_activations_for_probing \
    --activations-dir /path/to/activations/size5 \
    --trajectories-dir /path/to/trajectories/size5 \
    --probe-type grid_tile \
    --layers "7,15,23,31" \
    --output-indices -1 \
    --output-path /path/to/output/my_activations
```

## Output Format

The command writes a **directory** (not a single `.pt`):

```
{output_dir}/
  manifest.json
  activations/
    {trajectory_name_1}.pt              # tensor of shape (D,)
    {trajectory_name_2}.pt
    ...
```

In multi-size mode the per-trajectory `.pt` files are namespaced by size:

```
{output_dir}/
  manifest.json
  activations/
    size5/{trajectory_name_a}.pt
    size7/{trajectory_name_b}.pt
    ...
```

Each per-trajectory `.pt` holds a single `(D,)` activation tensor — there is no per-cell replication on disk. For `grid_tile`, the trainer assembles `[activation, row, col]` rows lazily via `GridTileCompactDataset` (see `manifest_loader.py`).

**Exception — token-major modes:** `next_action`, and `grid_tile` with a `token_selection`
or `token_major=True`, write **no** `activations/` directory at all. The output is just
`manifest.json`, whose entries point at the existing gathered token `.pt` files (relative to
`activations_root`). Nothing is copied.

### `manifest.json` schema

```jsonc
{
  "format_version": 3,
  "probe_type": "grid_tile",                 // or "distance" / "action_sequence"
  "activation_dim": 131072,
  "num_cells_per_trajectory": 225,           // grid_tile only
  "max_seq_len": 14,                         // action_sequence only
  "action_to_id": {"LEFT": 0, ...},          // action_sequence only
  "sizes": ["size5", "size7"],               // multi-size only
  "per_size_info": {                         // multi-size only
    "size5": {"num_trajectories": 100, "num_cells_per_trajectory": 225},
    "size7": {"num_trajectories": 80,  "num_cells_per_trajectory": 225}
  },
  "loading_spec": { /* echoes layers/steps/*_indices */ },
  "config":       { /* mirrors v1's "config" */ },
  "trajectories": [
    {
      "name": "traj_0001",
      "size": 5,                             // multi-size only
      "act_path": "activations/size5/traj_0001.pt",
      "positions": [[0,0],[0,1],...],        // grid_tile: list of [row, col]
      "labels":    [3, 1, 7, 3, ...],        // grid_tile: list of cell-ids
      "astar_distance": 12,                  // distance: int
      "actions":   [0, 2, 1, 3]              // action_sequence: list of action ids
    },
    ...
  ]
}
```

`act_path` is relative to `manifest.json` whenever the dataset copied its activations, so
that directory is portable: rename or move it without breaking references. A token-major
manifest instead carries an absolute `activations_root` and resolves `act_path` against it —
it references the gathered tree in place, so it is only as movable as that tree.

#### token-major `grid_tile` manifest schema

With a `token_selection` (or `token_major=True`), `grid_tile` keeps its `trajectories` key
but one entry is now one **(token, layer)**, so a trajectory name repeats across entries:

```jsonc
{
  "format_version": 3,
  "probe_type": "grid_tile",
  "activation_dim": 2880,                       // hidden_dim (one token, one layer)
  "activations_root": "/abs/path/to/activations_dir",
  "num_cells_per_trajectory": 225,
  "selection": { /* same block as next_action */ },
  "cells": {                                    // per-cell payload, stored ONCE per
    "traj_0001|0": {                            // (trajectory, step) rather than repeated
      "positions": [[0, 0], [0, 1]],            // on each of that trajectory's ~20 entries
      "labels": [1, 4]
    }
  },
  "trajectories": [
    {
      "name": "traj_0001",
      "act_path": "traj_0001/<model>/layer_15/step_0/output/532.pt",
      "cells_key": "traj_0001|0",               // which `cells` payload labels this token
      "layer": 15, "step": 0, "token_id": 532, "category": "output",
      "token": " left", "direction_count": 7, "layer_direction_count": 3
    }
  ]
}
```

The cells are keyed by **(trajectory, step)**, not by trajectory: a token selected from step
3 is labeled with the grid the agent saw at step 3, the same way `next_action` labels each
token with its own step's action.

Two consequences for the trainer, both enforced rather than documented-and-hoped:
`train_cognitive_map_probe` refuses an internal `--eval-split` on such a manifest (its
`randperm` is over entries, so a trajectory would land in both halves — pre-split with
`scripts/split_next_action_manifest.py` and pass `--eval-data-path`), and it refuses
`--subset < 1.0` for the same reason.

#### `next_action` manifest schema

`next_action` uses a different, token-level schema (key `samples`, not `trajectories`) and
references existing gathered files rather than copied ones:

```jsonc
{
  "format_version": 3,
  "probe_type": "next_action",
  "activation_dim": 2880,                       // hidden_dim (one token, one layer)
  "activations_root": "/abs/path/to/activations_dir",  // samples are relative to this
  "action_to_id": {"LEFT": 0, "UP": 1, "TOP": 1, "RIGHT": 2, "DOWN": 3},
  "selection": {                                // how tokens/layers were chosen
    "token_selection": "jlens_direction", "layer_selection": "jlens_direction",
    "num_tokens": 20, "num_layers": 3,
    "direction_tokens_path": "/workspace/jlens/direction_tokens_full.json",
    "direction_classes": "all", "direction_score": "logprob_mass",
    "num_direction_tokens": 539,
    "jlens_top_k": 20, "seed": 42
  },
  "loading_spec": { /* ... */ },
  "config":       { /* ... */ },
  "samples": [
    {
      "name": "traj_0001",                      // trajectory the token came from
      "act_path": "traj_0001/<model>/layer_15/step_0/output/532.pt",
      "label": 1,                               // action id (e.g. UP)
      "layer": 15, "step": 0, "token_id": 532, "category": "output",
      "token": "Ġleft",                         // jlens_direction modes only
      "direction_count": 214,                   // token score over the candidate layers
      "layer_direction_count": 12,              // score at this layer alone
                                                // (units are selection.direction_score)
      "size": 11                                // multi-size only
    },
    ...
  ]
}
```

Here `act_path` is relative to `activations_root` (the original gather location), since nothing
is copied into the output directory.

### Format versions

- **v3** (current): manifest dir + per-trajectory `(D,)` `.pt` files. Avoids the per-cell activation replication that made v1 prepare RAM scale as `T × C × D`.
- **v1** (legacy): single monolithic `.pt` containing a flat `(N, D+2)` activations tensor. Trainers still load v1 files via a backward-compatible dispatch.

## Notes

- At least one of `prompt_prefix_indices`, `prompt_suffix_indices`, `grid_state_indices`, or `output_indices` must be specified.
- `next_action` requires `output_indices` to be set and ignores the other category indices. With the default `token_selection="all"` it expects the `output` tokens to have been gathered with `--output-indices eos` and a single `--layers` value; the `<lens>_direction`/`random`/`recorded_*` modes instead pick from whatever reasoning tokens `scripts/jlens_reasoning_tokens.py` saved.
- `token_selection`/`layer_selection` apply to `next_action` and `grid_tile` only; passing them with another probe type is an error. On `grid_tile` they make the manifest **token-major** (see below); `token_major=True` does the same without a selection, and is an error for any other probe type.
- The activation vector itself is stored once per trajectory; per-cell `[row_id, col_id]` is folded in at training time.
- Class balancing finds the minimum count across all cell types and samples equally from each.
- When `balance_classes_per_trajectory` and `max_positions_per_trajectory` are both set, `max_positions` is adjusted to be divisible by the number of classes.
- NaN filtering still happens on the trainer side (a trajectory whose activation contains a NaN is dropped at load time, not at prepare time).
