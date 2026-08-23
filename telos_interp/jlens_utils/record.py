"""The `{stem}_jlens_selection.json` provenance file.

Once a trajectory's activation tree has been pruned to a selection, the selection itself
becomes the only record of *why* those files are the survivors — and, for the random control
arm, the only record of which of the survivors are the control. Re-deriving it from the CSV
would work for the jlens arm but not the control: after pruning, a uniform draw over the
surviving tokens is no longer a uniform draw over the reasoning chain.

So both `scripts/jlens_reasoning_tokens.py` (which prunes as it writes) and
`scripts/delete_non_jlens_selected.py` (which prunes after the fact) drop this file next to
the CSV, and `prepare_activations_for_probing`'s `recorded_jlens` / `recorded_random` modes
read it back instead of re-scoring.

Positions are **disk coordinates**: `step` is the `step_M` folder and `token_idx` the `.pt`
filename. `abs_pos` is carried along so a row can still be traced back to the CSV.
"""

import json
from pathlib import Path

from .top_filter import KeptTokens, TokenPick

RECORD_SUFFIX = "_jlens_selection.json"
RECORD_FORMAT_VERSION = 1

ARMS = ("jlens", "random")


def record_path(trajectory_folder: Path) -> Path:
    """`<folder>/<folder name>_jlens_selection.json`, beside the jlens CSV."""
    return trajectory_folder / f"{trajectory_folder.name}{RECORD_SUFFIX}"


def _pick_to_json(pick: TokenPick, output_start: int | None) -> dict:
    entry: dict = {
        "step": pick.step,
        "token_idx": pick.pos,
        "layers": list(pick.layers),
    }
    if output_start is not None:
        entry["abs_pos"] = pick.pos + output_start
    # Omitted entirely for the random arm -- see TokenPick's docstring for why the absence
    # matters rather than merely being tidy.
    if pick.direction_count is not None:
        entry["token"] = pick.token
        entry["direction_count"] = pick.direction_count
        entry["layer_direction_counts"] = {str(k): v for k, v in (pick.layer_direction_counts or {}).items()}
    return entry


def _pick_from_json(entry: dict) -> TokenPick:
    counts = entry.get("layer_direction_counts")
    return TokenPick(
        step=int(entry["step"]),
        pos=int(entry["token_idx"]),
        layers=tuple(int(layer) for layer in entry["layers"]),
        token=entry.get("token", ""),
        direction_count=entry.get("direction_count"),
        layer_direction_counts={int(k): int(v) for k, v in counts.items()} if counts else None,
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

    `output_starts` maps a step folder index to that step's `output_start`, used only to
    restore the `abs_pos` column; leave it out and the field is omitted.
    """
    record: dict = {
        "format_version": RECORD_FORMAT_VERSION,
        "stem": stem,
        "model": model,
        "config": config,
    }
    for arm in ARMS:
        picks = getattr(kept, arm)
        record[arm] = [
            _pick_to_json(pick, (output_starts or {}).get(pick.step))
            for _, pick in sorted(picks.items())
        ]
    return record


def write_selection_record(path: Path, record: dict) -> None:
    """Write the record atomically, so an interrupted run leaves no half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1)
    tmp.replace(path)


def read_selection_record(path: Path) -> tuple[KeptTokens, dict]:
    """Load a record back into (KeptTokens in disk coordinates, its config block).

    Raises ValueError on an unknown `format_version` rather than silently misreading a file
    written by a future version.
    """
    with open(path, encoding="utf-8") as f:
        record = json.load(f)

    version = record.get("format_version")
    if version != RECORD_FORMAT_VERSION:
        raise ValueError(f"{path}: unsupported selection record format_version={version!r}")

    arms = {}
    for arm in ARMS:
        picks = [_pick_from_json(entry) for entry in record.get(arm, [])]
        arms[arm] = {pick.key: pick for pick in picks}
    return KeptTokens(**arms), record.get("config", {})
