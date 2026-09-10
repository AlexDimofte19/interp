"""Reading a gathered lens tree: trajectories, analysis CSVs, mass tables, activation folders.

These five helpers -- `trajectory_dirs`, `read_mass_columns`, `load_trajectory`,
`read_lens_tables`, `find_act_folder` -- were byte-identical in `eval_probe_per_token.py` and
`eval_grid_probe_per_token.py`. They are lifted here verbatim so both evaluators, and the
join, read a tree the same way.

THE MASS TABLE IS NOT SELF-DESCRIBING. Its cells are `log P(any signal word)` over *some*
vocabulary, and this repo deliberately points several vocabularies at the same trees. The
vocabulary that produced a table lives only in its `.meta.json` sidecar, so
`check_vocabulary` is the difference between comparing two rulers and comparing two
different questions. `build_token_loudness_x_infered_action_probability.py` was the only
consumer that enforced it; here it is the default.

READ LENS CSVs WITH `csv.DictReader`, NEVER `pandas.read_csv`. Decoded tokens include the
literal string "NA", empty strings, embedded commas and newlines, all of which pandas' NA
handling silently corrupts.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from telos_interp.jlens_utils.jlens_csv import read_direction_scores, read_mass_meta

__all__ = [
    "check_vocabulary",
    "find_act_folder",
    "load_trajectory",
    "read_lens_tables",
    "read_mass_columns",
    "trajectory_dirs",
]


def trajectory_dirs(root: Path) -> list[Path]:
    """Every folder under `root` holding at least one lens artifact, sizeN nesting or flat."""
    seen = {p.parent for p in root.glob("size*/*/*_analysis.csv")}
    seen |= {p.parent for p in root.glob("*/*_analysis.csv")}
    return sorted(seen)


def read_mass_columns(path: Path) -> dict[tuple[int, int], dict[int, float]]:
    """{(step, abs_pos): {layer: log P(signal)}} from a signal-mass table.

    Note the key is `abs_pos`, which is PROMPT-INCLUSIVE. The `output_tokens` index -- what a
    rollout's `eos_token_pos` and a probe's `token_id` mean -- is `token_idx`. Joining the two
    without converting yields an empty or wrong join, never an error.
    """
    out: dict[tuple[int, int], dict[int, float]] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            layers = {}
            for key, val in row.items():
                if key and key.startswith("L") and key[1:].isdigit() and val not in (None, ""):
                    layers[int(key[1:])] = float(val)
            out[(int(row["step"]), int(row["abs_pos"]))] = layers
    return out


def load_trajectory(trajectories_dir: Path, stem: str) -> dict | None:
    hits = list(trajectories_dir.glob(f"size*/{stem}.json")) + list(trajectories_dir.glob(f"{stem}.json"))
    return json.loads(hits[0].read_text()) if hits else None


def read_lens_tables(
    folder: Path, stem: str, lenses: list[str], signal_tokens: dict, mass_suffix: str = "_direction_mass.csv"
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Per-lens (count scores, signal-mass columns) for one trajectory, keyed by (step, abs_pos).

    `mass_suffix` stays `_direction_mass.csv` by default because that is the filename every
    tree on disk uses, whatever vocabulary produced it -- the sidecar, not the filename, says
    which signal a table holds.
    """
    counts: dict[str, dict] = {}
    masses: dict[str, dict] = {}
    for lens in lenses:
        acsv = folder / f"{stem}_{lens}_analysis.csv"
        if acsv.exists():
            counts[lens] = read_direction_scores(acsv, signal_tokens, score_mode="count")
        mcsv = folder / f"{stem}_{lens}{mass_suffix}"
        if mcsv.exists():
            masses[lens] = read_mass_columns(mcsv)
    return counts, masses


def find_act_folder(activations_dir: Path, folder: Path, stem: str) -> Path | None:
    """The `{...}/{model}` folder holding this trajectory's .pt tree, sizeN-nested or flat.

    Falls back to a model folder sitting inside the lens folder itself, for the case where one
    tree holds both artifacts.
    """
    for d in (activations_dir / folder.parent.name / stem, activations_dir / stem):
        if d.exists():
            return next((c for c in d.iterdir() if c.is_dir()), None)
    return next((d for d in folder.iterdir() if d.is_dir()), None)


def check_vocabulary(mass_paths: list[Path], strict: bool = True) -> str | None:
    """The vocabulary every table in `mass_paths` was baked against, or raise if they differ.

    Returns the shared vocabulary fingerprint (or `None` when no table carries a sidecar, which
    is every table written before sidecars existed). With `strict=False` a disagreement is
    returned rather than raised, for callers that want to report and skip.

    A mass table's numbers only mean "loudness of X" relative to the vocabulary that produced
    it. Comparing a `direction`-baked table with a `grid`-baked one is comparing two different
    questions, and nothing in the numbers themselves would reveal it.
    """
    seen: dict[str, list[str]] = {}
    for path in mass_paths:
        meta = read_mass_meta(path)
        key = meta.get("signal_json") or meta.get("direction_mass_json") or meta.get("vocabulary")
        if key:
            seen.setdefault(str(key), []).append(path.name)
    if len(seen) > 1:
        message = "signal-mass tables were baked against different vocabularies: " + "; ".join(
            f"{vocab} <- {sorted(names)}" for vocab, names in sorted(seen.items())
        )
        if strict:
            raise ValueError(message)
        return message
    return next(iter(seen), None)
