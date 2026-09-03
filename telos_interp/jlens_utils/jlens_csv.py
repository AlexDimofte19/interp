"""Read a `{stem}_jlens_analysis.csv` and score its reasoning tokens against a vocabulary.

`scripts/jlens_reasoning_tokens.py` writes, per trajectory, both the residual-stream
activations (gather_activations layout) and a `{stem}_jlens_analysis.csv` holding the
top-20 j-space predictions for every (reasoning token, layer). This module turns that CSV
into per-token, per-layer *direction scores* — how direction-loaded a token's top-k lens
predictions are, under one of the modes in `scoring.py` — plus the coordinate math that maps
a CSV row onto the `.pt` file it came from.

The CSV holds a `top_{i}` for each of the top-k predictions and, since the logprob columns
were added, a `top_{i}_logprob` beside it. A `count` score reads only the former, so it
works on every CSV ever written; a logprob score needs the latter and says so loudly rather
than scoring an old CSV as if nothing matched (see `require_logprob_columns`).

The **direction-mass table** is the second artifact and the second half of this module. It
is wide where the analysis CSV is long — one row per reasoning token, one `L{n}` column per
layer — and each cell is the total direction probability at that (token, layer) in log
space, computed over the *whole* direction vocabulary while the logits were still on the
device. A top-k score can only see direction words that reached the top 20; this sees all
of them. What it gives up is late re-scoring: the vocabulary is baked in at gather time, so
every table is written with a `.meta.json` sidecar naming the vocabulary that produced it.

Stdlib only, on purpose: the callers are two standalone scripts and one command module, and
none of them should have to import torch to decide which tokens to keep.
"""

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from .scoring import DEFAULT_SCORE, DirectionScore, get_score

# Written next to the activations by scripts/jlens_reasoning_tokens.py.
DEFAULT_JLENS_CSV_SUFFIX = "_jlens_analysis.csv"

DIRECTION_CLASSES = ("UP", "DOWN", "LEFT", "RIGHT")


@dataclass
class TokenScore:
    """One reasoning token's direction score, broken down by layer.

    `score_mode` names the entry of `scoring.SCORES` the numbers were produced under, and
    is what `total` aggregates by — a count sums across layers, a probability mass
    logsumexps. It travels with the numbers because nothing downstream re-reads the CSV:
    a `TokenScore` handed to `rank_tokens` has to know how to collapse itself.
    """

    token: str
    per_layer: dict[int, float] = field(default_factory=dict)
    score_mode: str = DEFAULT_SCORE

    @property
    def score(self) -> DirectionScore:
        return get_score(self.score_mode)

    def total(self, layers: list[int] | None = None) -> float:
        """The token's score over `layers` (or every layer it has a row for).

        A layer the token has no row for contributes the score's `empty` value rather than
        being skipped, so two tokens covering different layer subsets stay comparable.

        >>> TokenScore("t", {7: 3, 15: 1}).total()
        4
        >>> TokenScore("t", {7: 3, 15: 1}).total([7, 23])
        3
        >>> round(TokenScore("t", {7: -1.0}, "logprob_sum").total([7, 15]), 1)
        -41.0
        """
        score = self.score
        if layers is None:
            return score.across_layers(list(self.per_layer.values()))
        return score.across_layers([self.per_layer.get(layer, score.empty) for layer in layers])


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


def logprob_column(top_column: str) -> str:
    """The logprob column that belongs to a `top_{i}` column.

    >>> logprob_column("top_3")
    'top_3_logprob'
    """
    return f"{top_column}_logprob"


def csv_has_logprobs(csv_path: Path, top_k: int = 20) -> bool:
    """Whether this CSV carries a `top_{i}_logprob` beside every `top_{i}` it has.

    CSVs written before the logprob columns existed have none of them, and there is no
    version field to ask — the header is the only evidence.
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        fields = set(csv.DictReader(f).fieldnames or [])
    present = [f"top_{i}" for i in range(1, top_k + 1) if f"top_{i}" in fields]
    return bool(present) and all(logprob_column(col) in fields for col in present)


def require_logprob_columns(csv_path: Path, score_mode: str, top_k: int = 20) -> None:
    """Raise unless `csv_path` can support `score_mode`.

    A logprob score reading a pre-logprob CSV would find no `top_{i}_logprob` column,
    score every row as "nothing matched", and rank tokens by a tie — a silent wrong answer
    after a multi-hour prepare. Fail at the first CSV instead, and say what fixes it.
    """
    if not get_score(score_mode).needs_logprobs or csv_has_logprobs(csv_path, top_k):
        return
    raise ValueError(
        f"{csv_path} has no top_i_logprob columns, so direction_score='{score_mode}' cannot be "
        f"computed from it. Either use direction_score='count', or re-emit the CSV with "
        f"scripts/jlens_reasoning_tokens.py --overwrite --no-save-activations (a CSV-only pass; "
        f"it touches no .pt file, so a pruned tree stays pruned)."
    )


def read_direction_scores(
    csv_path: Path,
    direction_tokens: set[str],
    top_k: int = 20,
    score_mode: str = DEFAULT_SCORE,
) -> dict[tuple[int, int], TokenScore]:
    """Score every (token, layer) of one lens against the direction vocabulary.

    The single entry point for both artifacts, dispatched on the score's `source`:

    * `"topk"` — `csv_path` is an analysis CSV. Each row contributes the direction words
      among its top-k lens predictions; how those become one number is the score's business
      (a count ignores the logprob columns, a logprob score reads the `top_{i}_logprob`
      beside each hit). `direction_tokens` is what a hit is matched against.
    * `"mass"` — `csv_path` is a direction-mass table, whose cells were already computed
      over the whole vocabulary at gather time. `direction_tokens` and `top_k` are unused;
      the vocabulary that applies is the one in the table's sidecar, not whatever the
      caller happens to be holding.

    Callers should resolve the path with `methods.score_artifact_path`, which makes the
    same decision.

    Returns {(step, abs_pos): TokenScore}, where `step` is the CSV's step column (an index
    into trajectory["steps"]) and `abs_pos` the token's absolute position in the forward
    pass. Uses csv.DictReader rather than pandas so token strings like "NA" or "" survive
    verbatim and memory stays flat over multi-hundred-MB files.

    Raises:
        ValueError: when `score_mode` needs logprobs the CSV does not carry. Scoring it as
            all-zero instead would rank every token identically and look like a result.
    """
    score = get_score(score_mode)
    if score.source == "mass":
        return read_direction_mass(csv_path, score_mode)
    require_logprob_columns(csv_path, score_mode, top_k)

    scores: dict[tuple[int, int], TokenScore] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        top_cols = [f"top_{i}" for i in range(1, top_k + 1) if f"top_{i}" in fields]
        for row in reader:
            key = (int(row["step"]), int(row["abs_pos"]))
            token_score = scores.get(key)
            if token_score is None:
                token_score = TokenScore(token=row.get("token", ""), score_mode=score_mode)
                scores[key] = token_score
            matched = [
                float(row[logprob_column(col)]) if score.needs_logprobs else 0.0
                for col in top_cols
                if row[col] in direction_tokens
            ]
            token_score.per_layer[int(row["layer"])] = score.per_layer(matched)
    return scores


def read_direction_counts(
    csv_path: Path,
    direction_tokens: set[str],
    top_k: int = 20,
) -> dict[tuple[int, int], TokenScore]:
    """`read_direction_scores` in `count` mode — how many top-k predictions are direction words.

    Kept as its own name because it is what every recorded invocation and every pruned tree
    on disk was produced with.
    """
    return read_direction_scores(csv_path, direction_tokens, top_k=top_k, score_mode="count")


# --- the direction-mass table ------------------------------------------------------------

MASS_PREFIX_COLUMNS = (
    "size",
    "complexity",
    "run",
    "step",
    "reasoning_pos",
    "abs_pos",
    "token",
    "agent_action",
)

MASS_META_SUFFIX = ".meta.json"


def mass_layer_column(layer: int) -> str:
    """The mass table's column for one layer.

    >>> mass_layer_column(15)
    'L15'
    """
    return f"L{layer}"


def mass_header(layers: list[int]) -> list[str]:
    """The mass table's header for a given layer set.

    Built here rather than in the writing script so the reader and the writer cannot drift
    on the column names.

    >>> mass_header([7, 15])[-2:]
    ['L7', 'L15']
    """
    return [*MASS_PREFIX_COLUMNS, *(mass_layer_column(layer) for layer in layers)]


def mass_meta_path(mass_path: Path) -> Path:
    """The sidecar naming the vocabulary a mass table was computed against."""
    return Path(str(mass_path) + MASS_META_SUFFIX)


def write_mass_meta(mass_path: Path, meta: dict) -> None:
    """Record which vocabulary produced this table, beside it.

    Not optional bookkeeping: this project points *two* vocabularies
    (`direction_tokens_full.json`, `grid_tokens_full.json`) at the same activation trees,
    and the numbers in a mass table are meaningless without knowing which one. A CSV has
    nowhere to put that, so it goes in a sidecar.
    """
    path = mass_meta_path(mass_path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    tmp.replace(path)


def read_mass_meta(mass_path: Path) -> dict:
    """The sidecar's contents, or `{}` when a table predates it."""
    path = mass_meta_path(mass_path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def mass_csv_layers(mass_path: Path) -> list[int]:
    """The layers a mass table has columns for, sorted."""
    with open(mass_path, newline="", encoding="utf-8") as f:
        fields = csv.DictReader(f).fieldnames or []
    return sorted(int(name[1:]) for name in fields if name.startswith("L") and name[1:].isdigit())


def read_direction_mass(
    mass_path: Path,
    score_mode: str = "logprob_mass_full",
) -> dict[tuple[int, int], TokenScore]:
    """Read a direction-mass table into the same `{(step, abs_pos): TokenScore}` shape.

    One row is one reasoning token and one `L{n}` column is one layer, so unlike the
    analysis CSV there is nothing to score here — the cell already *is* the number. An
    empty cell means the lens covered no such layer for that token (the jlens only has rows
    where it has a fitted `J`) and is left out, so `TokenScore.total` falls back to the
    score's `empty` for it exactly as it does for a missing CSV row.
    """
    get_score(score_mode)  # validate the name before touching the file
    scores: dict[tuple[int, int], TokenScore] = {}
    with open(mass_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        layer_cols = {
            name: int(name[1:]) for name in (reader.fieldnames or []) if name.startswith("L") and name[1:].isdigit()
        }
        if not layer_cols:
            raise ValueError(f"{mass_path} has no L<layer> columns; is it a direction-mass table?")
        for row in reader:
            key = (int(row["step"]), int(row["abs_pos"]))
            token_score = TokenScore(token=row.get("token", ""), score_mode=score_mode)
            for name, layer in layer_cols.items():
                cell = row[name]
                if cell != "":
                    token_score.per_layer[layer] = float(cell)
            scores[key] = token_score
    return scores


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


def artifact_layers(path: Path, score_mode: str = DEFAULT_SCORE) -> list[int]:
    """The layers an artifact covers, whichever artifact this score reads.

    The default candidate pool for a filter: a layer the lens produced nothing for can
    never be scored and must never be selected. The analysis CSV says so by having no rows
    for it; the mass table says so by having no column.
    """
    if get_score(score_mode).source == "mass":
        return mass_csv_layers(path)
    return csv_layers(path)


def output_start(trajectory_data: dict, step_index: int) -> int:
    """Absolute position of the step's first output token.

    Mirrors how scripts/jlens_reasoning_tokens.py derives the .pt filename:
    out_idx = abs_pos - output_start.
    """
    prompt = trajectory_data["prompt"]
    step = trajectory_data["steps"][step_index]
    return len(prompt["prompt_prefix_tokens"]) + len(step["grid_state_tokens"]) + len(prompt["prompt_suffix_tokens"])


def step_folder_index(trajectory_data: dict, step_index: int) -> int:
    """The `step_M` folder number for a CSV step column value.

    gather_activations names folders by `step_id`; the CSV records the list index. They
    agree in the data seen so far, but `step_id` is the authoritative field.
    """
    step = trajectory_data["steps"][step_index]
    return int(step.get("step_id", step_index))
