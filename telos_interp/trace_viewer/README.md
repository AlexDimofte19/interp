# Trace Viewer Input Format

This document describes the JSON input format for the trace viewer, used to visualize probing experiments on language model activations during grid navigation tasks. `example_size5.json` provides a handcrafted example for a valid file containing multiple steps, probabilities and probes outputs for GPT-OSS 20B solving a 5x5 grid. `example_size15.json` provides a larger example with a 15x15 grid with probabilities produced automatically using the command `reveng-cli get_trajectory --grid_size 15 --verbose --grid_complexity 0.7`.

## Top-Level Structure

```json
{
  "grid_params": { ... },
  "model_params": { ... },
  "prompt": { ... },
  "steps": [ ... ]
}
```

## Field Specifications

### `grid_params`
**Type:** `object`

Configuration parameters for the grid environment.

- `grid_width` (integer): Number of rows in the grid
- `grid_height` (integer): Number of columns in the grid
- `grid_complexity` (float): Complexity measure of the grid layout (0.0 to 1.0)
- `fully_observable` (boolean): Whether the entire grid is visible to the agent
- `astar_distance` (integer): Initial A* pathfinding distance to goal
- `agent_start_coordinates` (list of 2 integers): Starting agent coordinates in `[ROW, COLUMN]` format
- `goal_coordinates` (list of 2 integers): Goal coordinates in `[ROW, COLUMN]` format
- `legend` (dict): A map between tile names (e.g., `agent` or `wall`) and a dict containing their `symbol` (e.g., `A` or `#`) and a short `description` used to generate the legend.

### `model_params`
**Type:** `object`

Language model configuration and inference settings.

- `model_id` (string): Model identifier (e.g., "openai/gpt-oss-20b")
- `provider` (string): API provider (e.g., "together", "openai")
- `interface` (string): Interface type (e.g., "playground", "api")
- `n_interactions_in_context` (integer): Number of previous interactions in context
- `max_tokens` (integer | string): Output length limit or "auto"
- `max_steps_per_trajectory` (integer): Max steps the agent can complete before the run is aborted
- `temperature` (string | float): Sampling temperature or "auto"
- `reasoning_effort` (string): Reasoning effort level (e.g., "low", "medium", "high")
- `top_p` (string | float): Nucleus sampling parameter or "auto"
- `top_logprobs` (int): Number of logprobs collected per output token. -1 if absent
- `seed` (integer): Random seed for reproducibility

### `prompt`
**Type:** `object`

Template and tokenized prompt information. The prompt is split into three parts: `prefix` (before placeholder), `placeholder`, and `suffix` (after placeholder). The suffix tokens are stored both here and in each step since they might contain `probes` for activations depending on the preceding grid state.

- `prompt_template` (string): The prompt template text with placeholders between curly brackets (e.g., `{{grid_state}}`)
- `prompt_template_n_tokens` (integer): Number of tokens in the full `prompt_template`
- `prompt_prefix_n_tokens` (integer): Number of tokens in the prefix (before placeholder)
- `prompt_prefix_tokens` (array of [Token](#token-object)): Tokenized prefix portion of the prompt template
- `prompt_placeholder_n_tokens` (integer): Number of tokens in the placeholder itself (can be ignored, as it is replaced with the actual grid_state at every step)
- `prompt_placeholder_tokens` (array of [Token](#token-object)): Tokenized placeholder tokens (e.g., `{{`, `grid`, `_state`, `}}`)
   `prompt_prefix_n_tokens` (integer): Number of tokens in the prefix (before placeholder)
- `prompt_prefix_tokens` (array of [Token](#token-object)): Tokenized prefix portion of the prompt template
- `prompt_suffix_n_tokens` (integer): Number of tokens in the prompt suffix (after the grid state placeholder)
- `prompt_suffix_tokens` (array of [Token](#token-object)): Tokenized suffix portion of the prompt template

### `steps`
**Type:** `array` of [Step](#step-object)

Sequence of agent steps through the grid environment.

## Step Object

Each step represents one timestep in the agent's navigation.

```json
{
  "step_id": NUM,
  "grid_state": [ ... ],
  "grid_state_n_tokens": NUM,
  "grid_state_tokens": [ ... ],
  "prompt_suffix_n_tokens": NUM,
  "prompt_suffix_tokens": [ ... ],
  "astar_actions": [ ... ],
  "agent_action": "...",
  "output_text": "...",
  "output_n_tokens": NUM,
  "output_tokens": [ ... ]
}
```

### Step Fields

- `step_id` (integer): Sequential step identifier starting from 0
- `grid_state` (array of strings): Each string correspond to a row of the input state at this step, including the top coordinate ids.
- `grid_state_n_tokens` (integer): Number of tokens in the grid state
- `grid_state_tokens` (array of [Token](#token-object)): Tokenized grid state
- `prompt_suffix_n_tokens` (integer): Number of tokens in the prompt suffix (after the grid state placeholder)
- `prompt_suffix_tokens` (array of [Token](#token-object)): Tokenized suffix portion of the prompt template, **with optional added information for probes performed on these at this step**, since their activations depend on the preceding grid state.
- `astar_actions` (array of strings): Optimal actions according to A* pathfinding (e.g., `["UP", "RIGHT"]`)
- `agent_action` (string): Action taken by the agent (e.g. one of: "UP", "DOWN", "LEFT", "RIGHT")
- `output_text` (string): Raw text output from the model
- `output_n_tokens` (integer): Number of tokens in the output
- `output_tokens` (array of [Token](#token-object)): Tokenized model output with optional probing data

## Token Object

Represents a single token with optional interpretability metadata.

```json
{
  "id": 0,
  "token": "Hello",
  "token_id": 1234,
  "token_groups": [ ... ],
  "logprobs": { ... },
  "probes": { ... }
}
```

### Token Fields

- `id` (integer): Sequential token position within its context (the last token should have id corresponding to `XXX_n_tokens - 1`)
- `token` (string): Token text (special characters like `Ġ`, `Ċ` should be preserved in serialization)
- `token_id` (integer): Vocabulary index of the token
- `token_groups` (array of strings): Categories this token belongs to (see [Token groups](#token-groups))
- `probabilities` (dict of str:float): Map between several possible output `token` and their probabilities (float, 0-1 range). Returned by `get_trajectory` (control the number by setting `--top_logprobs N` with N>0). May be absent if not requested.
- `probes` (object, optional): Probing results for this token's activations (see [Probes](#probes-object)). Must be added to a trajectory output after creation using the format shown in `example_size5_probes.json`.

## Token groups

Common token group labels used for categorization:

- `"prompt"` - Part of the input prompt
- `"template"` - Template/formatting tokens (e.g. `<|start|>`, `assistant`), or spacing/newlines between those for both inputs and outputs
- `"placeholder"` - Placeholder tokens in the prompt template
- `"grid_state"` - Tokens belonging to the `grid_state` infilled in the prompt template at every step.
- `"grid_tile"` - Tokens corresponding to specific grid cells (i.e. mapped to an x,y coordinate)
- `"output"` - Model output tokens
- `"analysis"` - Reasoning/analysis section of the model output
- `"final"` - Final answer section of the model output
- `"action"` - Mentions of valid actions in model outputs

## Probes Object

Three types of probe object are supported:

### Tile Identity Probe / Cognitive Map Probe

Tile identity probes (also called cognitive map probes) serve the purpose of decoding grid states from model activations. The contents of this probe objects are structured on three levels:

- The first level key format for the probe name must begin with `tile_identity_probe` or `cognitive_map_probe` and end with `r<ROW_IDX>_c<COL_IDX>`, where `<ROW_IDX>` and `<COL_IDX>` are the 0-indexed row and column indices for the corresponding tile in the current state.

- The second level key represent the name of the layer on which the probe was applied, using the real model module names (see e.g. [https://huggingface.co/openai/gpt-oss-20b?show_file_info=model.safetensors.index.json](https://huggingface.co/openai/gpt-oss-20b?show_file_info=model.safetensors.index.json)). A specifier `.input` or `.output` should be added to indicate which activation was selected, following the `nnsight` format.

- The last level is a dictionary mapping elements in the legend to their probability for the given indices according to the probe.

Example:

```json
{
  "tile_identity_probe_example_r0_c0": {
    "model.layers.20.output": {
      "agent": 0.0005,
      "empty": 0.0019,
      "goal": 0.0005,
      "wall": 0.9971
    }
  },
  "cognitive_map_probe_l15_s0_suffix_r1_c3": {
    "model.layers.15.output": {
      "agent": 0.0200,
      "empty": 0.1000,
      "goal": 0.8000,
      "wall": 0.0800
    }
  }
}
```

### Goal Distance Probes

Goal distance probes decode the agent's representation of distances to the goal from its current position. The structure follows the same three-level format:

- The first level key format for the probe name must start with `goal_distance_probe`. The current position of the agent and the goal in `grid_state` are implicitly used.

- The second level key represents the layer name (same format as tile identity probes).

- The last level is a dictionary mapping distance values (as strings) to the float value corresponding to the distance

Example:

```json
{
  "goal_distance_probe_test": {
    "model.layers.15.output": 2.3542
  },
}
```

### Action Sequence Probes

Action sequence probes decode the model's planned multi-step action sequence from its activations.

- The first level key format for the probe name must start with `action_sequence_probe`. The current position of the agent and the goal in `grid_state` are implicitly used.

- The second level key represents the layer name (same format as other probes).

- The last level structure is a list of objects masking possible actions to their respective probabilities for the given step:

```json
{
  "action_sequence_probe_step1": {
    "model.layers.20.output": [
      {
        "UP": 0.7000,
        "RIGHT": 0.2500,
        "DOWN": 0.0300,
        "LEFT": 0.0200,
        "<EOS>": 0.0000
      },
      {
        "UP": 0.750,
        "RIGHT": 0.2000,
        "DOWN": 0.0200,
        "LEFT": 0.0200,
        "<EOS>": 0.0100
      },
      {
        "UP": 0.010,
        "RIGHT": 0.9200,
        "DOWN": 0.0400,
        "LEFT": 0.0200,
        "<EOS>": 0.0100
      },
      {
        "UP": 0.010,
        "RIGHT": 0.8000,
        "DOWN": 0.0600,
        "LEFT": 0.0200,
        "<EOS>": 0.1100
      },
      {
        "UP": 0.010,
        "RIGHT": 0.0100,
        "DOWN": 0.0100,
        "LEFT": 0.0100,
        "<EOS>": 0.9400
      }
    ]
  }
}
```

---

# Fork: jlens per-step files (`trace_viewer_fork.html`)

`trace_viewer_fork.html` is a separate viewer for Jacobian-lens data. It does **not** read the
format above: it loads a **folder** of per-step JSON files produced by
`scripts/jlens_viewer_export.py`, which joins a trajectory JSON with the reasoning-token CSV
written by `scripts/jlens_reasoning_tokens.py`.

```bash
python scripts/jlens_viewer_export.py \
  --csv       /media/alex/D/Uni/northeastern/data/jlens/jlens_reasoning_tokens_own_jlens_1000_traj.csv \
  --trajectory-paths /media/alex/D/Uni/northeastern/data/trajectories/trajectories_test_full \
  --direction-tokens /media/alex/D/Uni/northeastern/data/jlens/direction_tokens.json \
  --sizes 11 --complexities 0.0 --runs 0
# -> /media/alex/D/Uni/northeastern/data/jlens_viewer/size11_comp0.0_run0/{manifest,step_000..N}.json
```

Open `trace_viewer_fork.html` in a browser, press **📂 Load Folder** and pick one trajectory
folder (picking the export root also works — a dropdown then selects the trajectory).

The export streams the CSV once (~75 s for the full 3.2 GB / 16.1M rows at ~215k rows/s, seconds
when the requested trajectories appear early and early exit fires). It never holds more than one
step in memory.

**The CSV is a sample, not the whole trajectory set.** `jlens_reasoning_tokens_own_jlens_1000_traj.csv`
covers 100 of the 300 trajectories in `trajectories_test_full`: sizes 9/11/13, complexities
0.0/0.2/0.8/1.0 only. Requesting anything else still produces a folder, but every step has
`lens: null`. To see what is available without a failed export:

```bash
python scripts/jlens_viewer_export.py --csv <csv> --list-coverage
# prints the size/complexity/run breakdown and writes <out-dir>/csv_coverage.json
```

## Per-step file layout

`step_<index:03d>.json`, index-aligned with `steps[]` of the source trajectory.

```jsonc
{
  "schema": "trace_viewer_fork/1",
  "trajectory": { "id": "size11_comp0.0_run0", "size": 11, "complexity": 0.0, "run": 0,
                  "source": "<trajectory json path>", "n_steps": 6 },
  "grid_params": { ... },            // verbatim from the trajectory (legend included)
  "model_params": { ... },           // verbatim
  "step": {
    "index": 0, "step_id": 0,
    "grid_state": [ ... ],           // row strings, first entry is the coordinate header
    "agent_action": "LEFT",
    "output_text": "...",
    "output_tokens": [ ... ]         // verbatim tokens; `probabilities` unless --drop-probabilities
  },
  "lens": {
    "actions": ["RIGHT","LEFT","UP","DOWN"],
    "layers": [0, ..., 23], "n_pos": 121, "n_layers": 24, "top_k": 20,
    "reasoning_positions": [3,4,5,...],   // index into step.output_tokens, length n_pos
    "reasoning_tokens": ["Agent","Ġat",...],
    "abs_pos": [717,718,...],             // position in prefix+grid+suffix+output
    "vocab": ["-Level"," Side",...],      // interned top-k strings for this step
    "vocab_dir": [0,3,0,...],             // direction class per vocab entry (see below)
    "topk": [ ... ],                      // n_pos*n_layers*top_k vocab indices, -1 = missing
    "rank":    {"UP":[...], "DOWN":[...], "LEFT":[...], "RIGHT":[...]},   // n_pos*n_layers, null = missing
    "logprob": {"UP":[...], ... },
    "density": {"UP":[...], ... },        // 0..20: how many top-k entries are that action's words
    "missing_cells": 0
  },
  "warnings": []
}
```

Rules and conventions:

- **Flat index**: `i = pos * n_layers + layerIdx`; top-k entry `k` of a cell is `topk[i * top_k + k]`.
  `layerIdx` indexes into `lens.layers`, not the raw layer id, so partial-layer exports work.
- **Ranks are 0-based**: `0` = argmax over the ~201k vocabulary, so *lower is better*.
- **`lens` is `null`** when the CSV had no rows for that step. The file is still written so that
  folder order stays aligned with `step_id`.
- **Token conventions must not be mixed**: `step.output_tokens[].token` and
  `lens.reasoning_tokens` keep the trajectory's `Ġ`/`Ċ` byte-level form, while `lens.vocab`
  holds `tokenizer.decode()` output with real spaces and newlines — the same convention as the
  CSV's `top_k` columns and as `direction_tokens.json`. Direction matching happens only on the
  latter.
- **`manifest.json`** sits beside the step files with the trajectory metadata, the layer list and
  a per-step index. The viewer prefers it but works without it.

### Direction encoding and action colors

`vocab_dir` values are assigned from `direction_tokens.json` at export time (produced by
`notebooks/direction_tokens.ipynb`). The encoding and the action palette are duplicated between
`scripts/jlens_viewer_export.py` (`DIR_CODES`, docstring) and `trace_viewer_fork.html`
(`DIR_NAME`, `ACTION_COLOR`) — change both together.

| code | action | color     |
|------|--------|-----------|
| 0    | none   | –         |
| 1    | UP     | `#eb6834` |
| 2    | DOWN   | `#e0559a` |
| 3    | LEFT   | `#008300` |
| 4    | RIGHT  | `#4a3aa7` |

## What the fork shows

1. **Lens predictions by layer** — the top-20 lens predictions at every layer for the selected
   reasoning token, one column per layer; direction words are tinted with their action color.
   Below it, the exact rank of each action token per layer (click a cell to jump to that
   layer/action).
2. **Action rank heatmap** — x = reasoning position, y = layer (L0 at the bottom), brightness from
   the selected action's rank. Scale selector: `log (full vocab)` (default,
   `1 - log10(rank+1)/log10(201089)`), `log capped @ 1k` (much higher contrast — use this to see
   where the model commits) and `top-20 only` (binary).
3. **Top-20 direction density** — same axes, brightness = how many of the 20 lens predictions at
   that cell are words for the selected action (0–20, optionally normalised to the step maximum).
   A fully dark cell means none.

Selection is bidirectional: clicking a heatmap cell selects that reasoning position and layer and
highlights the token in the **Model Output** pane; clicking a reasoning token there moves the
heatmap crosshair. Arrow keys move the selection once a heatmap has focus.
