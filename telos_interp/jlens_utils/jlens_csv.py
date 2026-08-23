"""Read a `{stem}_jlens_analysis.csv` and score its reasoning tokens against a vocabulary.

`scripts/jlens_reasoning_tokens.py` writes, per trajectory, both the residual-stream
activations (gather_activations layout) and a `{stem}_jlens_analysis.csv` holding the
top-20 j-space predictions for every (reasoning token, layer). This module turns that CSV
into per-token, per-layer *direction counts* — how many of a token's top-k lens predictions
are direction words — plus the coordinate math that maps a CSV row onto the `.pt` file it
came from.

Stdlib only, on purpose: the callers are two standalone scripts and one command module, and
none of them should have to import torch to decide which tokens to keep.
"""

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

# Written next to the activations by scripts/jlens_reasoning_tokens.py.
DEFAULT_JLENS_CSV_SUFFIX = "_jlens_analysis.csv"

DIRECTION_CLASSES = ("UP", "DOWN", "LEFT", "RIGHT")


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


def jlens_csv_path(trajectory_folder: Path, suffix: str = DEFAULT_JLENS_CSV_SUFFIX) -> Path:
    """`<folder>/<folder name>_jlens_analysis.csv`, as jlens_reasoning_tokens.py writes it."""
    return trajectory_folder / f"{trajectory_folder.name}{suffix}"


def load_direction_tokens(path: str | Path, classes: str = "all") -> set[str]:
    """Load the direction-token vocabulary as a flat set.

    The JSON maps each direction ("UP"/"DOWN"/"LEFT"/"RIGHT") to a list of token strings
    in decoded form (e.g. " left"), matching what the CSV's top_* columns hold. `classes`
    is "all" (the union) or a comma-separated subset such as "UP,DOWN".
    """
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


def csv_layers(csv_path: Path) -> list[int]:
    """The distinct layer indices a jlens CSV holds rows for, sorted.

    The default candidate pool for a filter: a layer with no jlens matrix produces no rows,
    so it can never be scored and must never be selected.
    """
    layers: set[int] = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            layers.add(int(row["layer"]))
    return sorted(layers)


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
