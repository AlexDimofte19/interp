"""The `{stem}_jlens_selection.json` provenance file.

Once a trajectory's activation tree has been pruned to a selection, the selection itself
becomes the only record of *why* those files are the survivors — and, for an unscored
control arm, the only record of which of the survivors are the control. Re-deriving it from
a CSV would work for a lens arm but not the control: after pruning, a uniform draw over the
surviving tokens is no longer a uniform draw over the reasoning chain.

So both `scripts/jlens_reasoning_tokens.py` (which prunes as it writes) and
`scripts/delete_non_jlens_selected.py` (which prunes after the fact) drop this file next to
the CSVs, and `prepare_activations_for_probing`'s `recorded_*` modes read it back instead of
re-scoring.

Positions are **disk coordinates**: `step` is the `step_M` folder and `token_idx` the `.pt`
filename. `abs_pos` is carried along so a row can still be traced back to a CSV.

Format
------
v2 keys the arms by method name and gives each its **own config block**::

    {"format_version": 2, "stem": ..., "model": ...,
     "arms": {"jlens":     {"config": {...}, "picks": [...]},
              "logitlens": {"config": {...}, "picks": [...]},
              "random":    {"config": {...}, "picks": [...]}}}

Per-arm config is not tidiness: an arm added later by an incremental gather carries that
run's parameters, while the arms already on disk carry the original pruning run's. One
shared block would have to misreport one of them. It is also where `direction_score` lives —
the numbers in `direction_count` / `layer_direction_counts` are meaningless without knowing
whether they are counts or logprobs, and an arm's score mode is fixed at the moment it was
drawn. A record with no `direction_score` predates the logprob scores and is a count.

v1 — a flat `jlens`/`random` pair of lists and a single `config` — is still **read**, and
must stay readable: every trajectory pruned before this change has a v1 record, and it is
the only surviving trace of that trajectory's control arm. `_from_v1` lifts it into the v2
shape in memory. The filename is unchanged for the same reason.
"""

import json
from pathlib import Path

from .scoring import DEFAULT_SCORE
from .top_filter import Arm, KeptTokens, TokenPick

RECORD_SUFFIX = "_jlens_selection.json"
RECORD_FORMAT_VERSION = 2

# The arms a v1 record could hold, in the order it wrote them.
V1_ARMS = ("jlens", "random")


def record_path(trajectory_folder: Path) -> Path:
    """`<folder>/<folder name>_jlens_selection.json`, beside the analysis CSVs."""
    return trajectory_folder / f"{trajectory_folder.name}{RECORD_SUFFIX}"


def _pick_to_json(pick: TokenPick, output_start: int | None) -> dict:
    entry: dict = {
        "step": pick.step,
        "token_idx": pick.pos,
        "layers": list(pick.layers),
    }
    if output_start is not None:
        entry["abs_pos"] = pick.pos + output_start
    # Omitted entirely for an unscored arm -- see TokenPick's docstring for why the absence
    # matters rather than merely being tidy.
    if pick.direction_count is not None:
        entry["token"] = pick.token
        entry["direction_count"] = pick.direction_count
        entry["layer_direction_counts"] = {str(k): v for k, v in (pick.layer_direction_counts or {}).items()}
    return entry


def _pick_from_json(entry: dict, score_mode: str = DEFAULT_SCORE) -> TokenPick:
    """One recorded pick. `score_mode` comes from the arm's config, not the entry.

    The scores are read back **uncast**: a count is an int in the file and stays one, a
    logprob score is a float. Forcing either into `int` would round a logprob to a coarse
    tie and silently change the ranking the record exists to preserve.
    """
    counts = entry.get("layer_direction_counts")
    return TokenPick(
        step=int(entry["step"]),
        pos=int(entry["token_idx"]),
        layers=tuple(int(layer) for layer in entry["layers"]),
        token=entry.get("token", ""),
        direction_count=entry.get("direction_count"),
        layer_direction_counts={int(k): v for k, v in counts.items()} if counts else None,
        score_mode=score_mode,
    )


def build_record(
    kept: KeptTokens,
    *,
    stem: str,
    model: str,
    config: dict,
    output_starts: dict[int, int] | None = None,
) -> dict:
    """Serialise a disk-coordinate `KeptTokens` into the record's JSON shape.

    `config` describes the run that produced *these* arms and is stored against each of
    them; merging two records is what gives a file arms with differing configs.

    `output_starts` maps a step folder index to that step's `output_start`, used only to
    restore the `abs_pos` column; leave it out and the field is omitted.
    """
    return {
        "format_version": RECORD_FORMAT_VERSION,
        "stem": stem,
        "model": model,
        "arms": {
            name: {
                "config": config,
                "picks": [
                    _pick_to_json(pick, (output_starts or {}).get(pick.step)) for _, pick in sorted(arm.items())
                ],
            }
            for name, arm in kept.arms.items()
        },
    }


def merge_records(existing: dict, new: dict) -> dict:
    """Fold `new`'s arms into `existing`, keeping every arm `new` does not mention.

    This is what makes an incremental gather safe. An arm already on disk keeps its
    recorded picks *and* its recorded config verbatim — recomputing a control arm against
    a pruned chain would draw a different, biased sample, and every prepared dataset
    pointing at this tree would silently stop matching it.
    """
    merged = dict(existing)
    merged["format_version"] = RECORD_FORMAT_VERSION
    for key in ("stem", "model"):
        if new.get(key):
            merged[key] = new[key]
    merged["arms"] = {**existing.get("arms", {}), **new.get("arms", {})}
    return merged


def _from_v1(record: dict) -> dict:
    """Lift a v1 record's flat arm lists and single config into the v2 shape."""
    config = record.get("config", {})
    return {
        "format_version": RECORD_FORMAT_VERSION,
        "stem": record.get("stem", ""),
        "model": record.get("model", ""),
        "arms": {name: {"config": config, "picks": record[name]} for name in V1_ARMS if name in record},
    }


def normalize_record(record: dict, path: Path | None = None) -> dict:
    """Return `record` in the v2 shape, upgrading a v1 file in memory.

    Raises ValueError on an unknown `format_version` rather than silently misreading a file
    written by a future version.
    """
    version = record.get("format_version")
    if version == 1:
        return _from_v1(record)
    if version == RECORD_FORMAT_VERSION:
        return record
    raise ValueError(f"{path or '<record>'}: unsupported selection record format_version={version!r}")


def write_selection_record(path: Path, record: dict) -> None:
    """Write the record atomically, so an interrupted run leaves no half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1)
    tmp.replace(path)


def read_raw_record(path: Path) -> dict:
    """The record as JSON, normalized to v2. For callers that want to merge into it."""
    with open(path, encoding="utf-8") as f:
        return normalize_record(json.load(f), path)


def read_selection_record(path: Path) -> tuple[KeptTokens, dict[str, dict]]:
    """Load a record into (KeptTokens in disk coordinates, {arm name: its config})."""
    record = read_raw_record(path)
    arms: dict[str, Arm] = {}
    configs: dict[str, dict] = {}
    for name, block in record.get("arms", {}).items():
        # Which score the recorded numbers are is a property of the *run*, so it lives in
        # the arm's config; a record written before the logprob scores existed has none and
        # is a count by definition.
        config = block.get("config", {})
        score_mode = config.get("direction_score", DEFAULT_SCORE)
        picks = [_pick_from_json(entry, score_mode) for entry in block.get("picks", [])]
        arms[name] = {pick.key: pick for pick in picks}
        configs[name] = block.get("config", {})
    return KeptTokens(arms), configs


__all__ = [
    "RECORD_FORMAT_VERSION",
    "RECORD_SUFFIX",
    "build_record",
    "merge_records",
    "normalize_record",
    "read_raw_record",
    "read_selection_record",
    "record_path",
    "write_selection_record",
]
