# Trace Viewer Input Format

This document describes the JSON input format for the trace viewer, used to visualize probing experiments on language model activations during grid navigation tasks. `example_input.json` provides an example for a valid file containing multiple steps, probabilities and probes outputs.

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

- `n_rows` (integer): Number of rows in the grid
- `n_cols` (integer): Number of columns in the grid
- `fully_observable` (boolean): Whether the entire grid is visible to the agent
- `complexity` (float): Complexity measure of the grid layout (0.0 to 1.0)
- `initial_astar_distance` (integer): Initial A* pathfinding distance to goal
- `legend` (dict): A map between tile names (e.g., `agent` or `wall`) and a dict containing their `symbol` (e.g., `A` or `#`) and a short `description` used to generate the legend.

### `model_params`
**Type:** `object`

Language model configuration and inference settings.

- `model_name` (string): Model identifier (e.g., "openai/gpt-oss-20b")
- `provider` (string): API provider (e.g., "together", "openai")
- `interface` (string): Interface type (e.g., "playground", "api")
- `n_interactions_in_context` (integer): Number of previous interactions in context
- `out_length` (string | integer): Output length limit or "auto"
- `temperature` (string | float): Sampling temperature or "auto"
- `reasoning_effort` (string): Reasoning effort level (e.g., "low", "medium", "high")
- `top_p` (string | float): Nucleus sampling parameter or "auto"
- `top_k` (string | integer): Top-k sampling parameter or "auto"
- `seed` (integer): Random seed for reproducibility

### `prompt`
**Type:** `object`

Template and tokenized prompt information. The prompt is split into three parts: `prefix` (before placeholder), `placeholder`, and `suffix` (after placeholder). The suffix tokens are stored both here and in each step since they might contain `probes` for activations depending on the preceding grid state.

- `prompt_template` (string): The prompt template text with placeholders between curly brackets (e.g., `{{grid_state}}`)
- `prompt_template_n_tokens` (string): Number of tokens in the full `prompt_template`
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
- `probabilities` (dict of str:float): Map between several possible output `token` and their probabilities (float, 0-1 range)
- `probes` (object, optional): Probing results for this token's activations (see [Probes](#probes-object))

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

### Tile Identity Probe

Tile identity probes serve the purpose of decoding grid states from model activations. The contents of this probe objects are structured on three levels:

- The first level key format for the probe name must begin with `tile_identity_probe` and end with `r<ROW_IDX>_c<COL_IDX>`, where `<ROW_IDX>` and `<COL_IDX>` are the 0-indexed row and column indices for the corresponding tile in the current state.

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
  "tile_identity_probe_another_example_r1_c3": {
    "20": {
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
