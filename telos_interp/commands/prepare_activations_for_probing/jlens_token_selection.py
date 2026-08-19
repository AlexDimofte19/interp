"""Pick which reasoning tokens and layers a `next_action` probe trains on.

`scripts/jlens_reasoning_tokens.py` writes, per trajectory, both the residual-stream
activations (gather_activations layout) and a `{stem}_jlens_analysis.csv` holding the
top-20 j-space predictions for every (reasoning token, layer). This module turns that
CSV into a selection: the tokens whose j-space is most *direction-loaded* (the
per-trajectory version of notebooks/direction_token_location_analysis.ipynb), plus the
matched random controls.

Stdlib only — no torch — so the selection logic is testable on its own.
"""

import csv
import random
from dataclasses import dataclass, field
from pathlib import Path

# Written next to the activations by scripts/jlens_reasoning_tokens.py.
DEFAULT_JLENS_CSV_SUFFIX = "_jlens_analysis.csv"

DIRECTION_CLASSES = ("UP", "DOWN", "LEFT", "RIGHT")

TokenSelection = str  # "all" | "jlens_direction" | "random"
LayerSelection = str  # "spec" | "jlens_direction" | "random"


@dataclass
class TokenScore:
    """Direction-token counts for one reasoning token, broken down by layer."""

    token: str
    per_layer: dict[int, int] = field(default_factory=dict)

    def total(self, layers: list[int] | None = None) -> int:
        """Sum of counts, restricted to `layers` when given."""
        if layers is None:
            return sum(self.per_layer.values())
        return sum(self.per_layer.get(layer, 0) for layer in layers)


@dataclass
class SelectionConfig:
    """How `next_action` picks its (token, layer) samples.

    Bundled so the seven knobs travel as one argument; the public
    `prepare_activations_for_probing` signature keeps them flat for the CLI.
    See that function's docstring for what each mode means.
    """

    token_selection: TokenSelection = "all"
    layer_selection: LayerSelection = "spec"
    num_tokens: int | None = None
    num_layers: int | None = None
    direction_tokens: set[str] | None = None
    jlens_top_k: int = 20
    seed: int | None = 42
    # Kept for the manifest's provenance record; `direction_tokens` is what is used.
    direction_tokens_path: str | None = None
    direction_classes: str = "all"

    @property
    def is_default(self) -> bool:
        """True when selection reduces to the pre-existing behaviour (no CSV needed)."""
        return self.token_selection == "all" and self.layer_selection == "spec"

    def to_manifest(self) -> dict:
        """JSON-serializable record of how the samples were chosen."""
        return {
            "token_selection": self.token_selection,
            "layer_selection": self.layer_selection,
            "num_tokens": self.num_tokens,
            "num_layers": self.num_layers,
            "direction_tokens_path": self.direction_tokens_path,
            "direction_classes": self.direction_classes,
            "num_direction_tokens": len(self.direction_tokens) if self.direction_tokens else None,
            "jlens_top_k": self.jlens_top_k,
            "seed": self.seed,
        }


@dataclass
class SampleRef:
    """One (token, layer) activation file selected for training."""

    layer: int
    step: int  # step folder index (step_id), i.e. what layer_N/step_M uses
    token_idx: int  # output-relative index, i.e. the .pt filename
    path: Path
    token: str = ""
    direction_count: int | None = None  # token score summed over candidate layers
    layer_direction_count: int | None = None  # score at this layer alone


def jlens_csv_path(trajectory_folder: Path, suffix: str = DEFAULT_JLENS_CSV_SUFFIX) -> Path:
    """`<folder>/<folder name>_jlens_analysis.csv`, as jlens_reasoning_tokens.py writes it."""
    return trajectory_folder / f"{trajectory_folder.name}{suffix}"


def load_direction_tokens(path: str | Path, classes: str = "all") -> set[str]:
    """Load the direction-token vocabulary as a flat set.

    The JSON maps each direction ("UP"/"DOWN"/"LEFT"/"RIGHT") to a list of token strings
    in decoded form (e.g. " left"), matching what the CSV's top_* columns hold. `classes`
    is "all" (the union) or a comma-separated subset such as "UP,DOWN".
    """
    import json

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if classes.strip().lower() == "all":
        wanted = list(data.keys())
    else:
        wanted = [c.strip().upper() for c in classes.split(",") if c.strip()]
        unknown = [c for c in wanted if c not in data]
        if unknown:
            raise ValueError(
                f"Unknown direction class(es) {unknown} in direction_classes='{classes}'; "
                f"available: {sorted(data.keys())}"
            )

    tokens: set[str] = set()
    for name in wanted:
        tokens.update(data[name])
    return tokens


def read_direction_counts(
    csv_path: Path,
    direction_tokens: set[str],
    top_k: int = 20,
) -> dict[tuple[int, int], TokenScore]:
    """Count direction tokens in each row's top-k j-space predictions.

    Returns {(step, abs_pos): TokenScore}, where `step` is the CSV's step column (an index
    into trajectory["steps"]) and `abs_pos` the token's absolute position in the forward
    pass. Uses csv.DictReader rather than pandas so token strings like "NA" or "" survive
    verbatim and memory stays flat over multi-hundred-MB files.
    """
    counts: dict[tuple[int, int], TokenScore] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        top_cols = [f"top_{i}" for i in range(1, top_k + 1) if f"top_{i}" in (reader.fieldnames or [])]
        for row in reader:
            key = (int(row["step"]), int(row["abs_pos"]))
            score = counts.get(key)
            if score is None:
                score = TokenScore(token=row.get("token", ""))
                counts[key] = score
            hits = sum(1 for col in top_cols if row[col] in direction_tokens)
            score.per_layer[int(row["layer"])] = hits
    return counts


def output_start(trajectory_data: dict, step_index: int) -> int:
    """Absolute position of the step's first output token.

    Mirrors how scripts/jlens_reasoning_tokens.py derives the .pt filename:
    out_idx = abs_pos - output_start.
    """
    prompt = trajectory_data["prompt"]
    step = trajectory_data["steps"][step_index]
    return (
        len(prompt["prompt_prefix_tokens"])
        + len(step["grid_state_tokens"])
        + len(prompt["prompt_suffix_tokens"])
    )


def step_folder_index(trajectory_data: dict, step_index: int) -> int:
    """The `step_M` folder number for a CSV step column value.

    gather_activations names folders by `step_id`; the CSV records the list index. They
    agree in the data seen so far, but `step_id` is the authoritative field.
    """
    step = trajectory_data["steps"][step_index]
    return int(step.get("step_id", step_index))


def _pick_layers(
    token_layers: dict[int, int],
    candidate_layers: list[int],
    layer_selection: LayerSelection,
    num_layers: int | None,
    rng: random.Random,
) -> list[int]:
    """Choose which layers of one token become samples."""
    if layer_selection == "spec" or num_layers is None:
        return list(candidate_layers)
    if layer_selection == "random":
        if num_layers >= len(candidate_layers):
            return list(candidate_layers)
        return sorted(rng.sample(candidate_layers, num_layers))
    if layer_selection == "jlens_direction":
        ranked = sorted(candidate_layers, key=lambda layer: (-token_layers.get(layer, 0), layer))
        return sorted(ranked[:num_layers])
    raise ValueError(f"Unknown layer_selection: {layer_selection}")


def select_token_layer_pairs(
    *,
    trajectory_folder: Path,
    model_folder: Path,
    trajectory_data: dict,
    candidate_layers: list[int],
    candidate_steps: list[int],
    available_tokens_by_step: dict[int, list[int]],
    selection: SelectionConfig,
    csv_suffix: str = DEFAULT_JLENS_CSV_SUFFIX,
    verbose: bool = False,
) -> tuple[list[SampleRef], int]:
    """Select the (token, layer) activation files to train on for one trajectory.

    Args:
        trajectory_folder: `<activations_dir>/<trajectory name>`, where the jlens CSV lives.
        model_folder: the `<model>` folder inside it (holds `layer_N/step_M/output`).
        trajectory_data: parsed trajectory JSON, used to map abs_pos -> output-relative index.
        candidate_layers: layers to choose from (the `layers` spec ∩ what is on disk).
        candidate_steps: step folder indices to choose from (the `steps` spec ∩ disk).
        available_tokens_by_step: {step folder index: sorted output token indices on disk}.
        selection: the modes, N, M, direction vocabulary and seed. The random draws are
            seeded per trajectory, so a trajectory's sample is stable regardless of
            iteration order or subsetting.

    Returns:
        (samples, missing) — the selected files, and how many selections pointed at a
        `.pt` that does not exist (skipped). Empty when the trajectory cannot be scored
        (e.g. the jlens CSV is absent).
    """
    rng = random.Random(f"{selection.seed}-{trajectory_folder.name}")
    num_tokens, num_layers = selection.num_tokens, selection.num_layers
    needs_csv = "jlens_direction" in (selection.token_selection, selection.layer_selection)

    scores: dict[tuple[int, int], TokenScore] = {}
    if needs_csv:
        if selection.direction_tokens is None:
            raise ValueError("direction_tokens is required when a jlens_direction mode is used")
        csv_path = jlens_csv_path(trajectory_folder, csv_suffix)
        if not csv_path.exists():
            if verbose:
                print(f"  Skipped: no jlens CSV at {csv_path}")
            return [], 0
        scores = read_direction_counts(csv_path, selection.direction_tokens, top_k=selection.jlens_top_k)

    # The CSV is keyed by (csv step, abs_pos); re-key it onto the on-disk
    # (step folder, output-relative index) coordinates the .pt files use.
    by_disk: dict[tuple[int, int], TokenScore] = {}
    for (csv_step, abs_pos), score in scores.items():
        try:
            key = (step_folder_index(trajectory_data, csv_step), abs_pos - output_start(trajectory_data, csv_step))
        except (IndexError, KeyError):
            continue
        by_disk[key] = score

    # Candidate tokens as (step folder index, output-relative index).
    step_set = set(candidate_steps)
    if selection.token_selection == "jlens_direction":
        candidates = [key for key in by_disk if key[0] in step_set]
        candidates.sort(key=lambda key: (-by_disk[key].total(candidate_layers), key[0], key[1]))
        if num_tokens is not None:
            if verbose and num_tokens > len(candidates):
                print(f"  Only {len(candidates)} scored tokens available (asked for {num_tokens})")
            candidates = candidates[:num_tokens]
    else:
        candidates = [
            (step_idx, token_idx)
            for step_idx in candidate_steps
            for token_idx in available_tokens_by_step.get(step_idx, [])
        ]
        if selection.token_selection == "random" and num_tokens is not None:
            if num_tokens < len(candidates):
                candidates = sorted(rng.sample(candidates, num_tokens))
            elif verbose:
                print(f"  Only {len(candidates)} tokens available (asked for {num_tokens})")

    samples: list[SampleRef] = []
    missing = 0
    for step_idx, token_idx in candidates:
        score = by_disk.get((step_idx, token_idx))
        token = score.token if score else ""
        total = score.total(candidate_layers) if score else None
        token_layers = score.per_layer if score else {}
        for layer in _pick_layers(token_layers, candidate_layers, selection.layer_selection, num_layers, rng):
            path = model_folder / f"layer_{layer}" / f"step_{step_idx}" / "output" / f"{token_idx}.pt"
            if not path.exists():
                missing += 1
                continue
            samples.append(
                SampleRef(
                    layer=layer,
                    step=step_idx,
                    token_idx=token_idx,
                    path=path,
                    token=token,
                    direction_count=total,
                    layer_direction_count=token_layers.get(layer) if score else None,
                )
            )
    return samples, missing
