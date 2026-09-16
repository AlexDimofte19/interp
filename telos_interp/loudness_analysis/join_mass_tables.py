#!/usr/bin/env python3
"""Concatenate a tree's per-trajectory direction-mass tables into one token-level CSV.

`build_loudness_tables.py` writes one `{stem}_{lens}_direction_mass.csv` per trajectory: one
row per reasoning token, one `L{layer}` column per layer, each cell `log P(any signal word)`
at that (token, layer). Hundreds of those tables are hundreds of files and nothing relates
them. This writes the obvious thing -- all their rows, one file, with a `trajectory` column
saying where each came from -- and does nothing else. No scoring, no aggregation, no
selection. The mean per layer, the argmax, the plots: all of that belongs downstream, to a
notebook or to `scripts/jlens_layer_profile.py`.

    python -m telos_interp.loudness_analysis.join_mass_tables \\
        /workspace/activations/qwen_loudness_profile_p20 --lens jlens --out jlens_tokens.csv

WHY csv AND NOT pandas. Decoded tokens include the literal string "NA", empty strings,
embedded commas and newlines. `pandas.read_csv`'s NA handling turns the first two into
missing values, silently, and a token column is exactly where that bites. Reading and
writing row by row also keeps memory flat: the 20% Qwen sample is a few million rows.

WHY THE VOCABULARY IS CHECKED. A mass table is not self-describing -- its cells are a mass
over *some* vocabulary, named only in the `.meta.json` sidecar beside it, and this repo
deliberately points several vocabularies at the same trees. Concatenating two vocabularies'
tables produces a column of numbers with no referent and no error. The fingerprint is a hash
of the vocabulary's *contents*, so it also catches a file edited in place between two halves
of a gather, which comparing paths never would.

AN EMPTY CELL IS NOT A FLOOR. A layer the lens produced nothing for (the jlens outside its
fitted layers) leaves the cell empty, which means "not covered" -- a different fact from "no
direction mass here", which is a very negative number. Passed through verbatim; do not fill
it with a floor value downstream.

`abs_pos` IS PROMPT-INCLUSIVE. It is the token's position in the whole forward pass, not its
index into `output_tokens` -- that is `reasoning_pos`'s neighbourhood, and the two differ by
the prompt length. Grouping these rows by layer does not care, but joining them to a rollout
or a probe does, and joining on the wrong one yields an empty join rather than an error.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

__all__ = ["find_mass_tables", "check_one_vocabulary", "join_mass_tables"]


def find_mass_tables(tree: Path, lens: str) -> list[Path]:
    """Every `*_{lens}_direction_mass.csv` under `tree`, size-sharded or flat."""
    return sorted(tree.rglob(f"*_{lens}_direction_mass.csv"))


def check_one_vocabulary(paths: list[Path]) -> dict:
    """Return the shared sidecar, or raise naming the first table that disagrees.

    Only the fields that decide whether two tables are the same measurement are compared:
    the vocabulary's name and content hash, and which classes of it were used.
    """
    pinned = ("signal_name", "signal_fingerprint", "direction_classes")
    first_meta, first_path = None, None
    for path in paths:
        sidecar = Path(str(path) + ".meta.json")
        meta = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
        if first_meta is None:
            first_meta, first_path = meta, path
            continue
        for key in pinned:
            if meta.get(key) != first_meta.get(key):
                raise SystemExit(
                    f"{path.name} has {key}={meta.get(key)!r} but {first_path.name} has "
                    f"{first_meta.get(key)!r}. These are different measurements and must not be "
                    "concatenated -- re-gather the odd ones out, or join the two sets separately."
                )
    return first_meta or {}


def join_mass_tables(tree: Path, lens: str, out: Path, *, verbose: bool = False) -> int:
    """Write every table's rows to `out`, prefixed with the trajectory they came from.

    The header is taken from the first table and every later one must match it exactly. A
    table covering a different layer set is a different gather, and letting the columns
    drift would misalign the values without misaligning the column count.

    Returns the number of rows written.
    """
    paths = find_mass_tables(tree, lens)
    if not paths:
        raise SystemExit(f"no *_{lens}_direction_mass.csv under {tree}")
    check_one_vocabulary(paths)

    header: list[str] | None = None
    rows = 0
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        for path in paths:
            # the stem of "foo_jlens_direction_mass.csv" is the trajectory "foo"
            trajectory = path.name.removesuffix(f"_{lens}_direction_mass.csv")
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                columns = next(reader, None)
                if columns is None:
                    continue
                if header is None:
                    header = columns
                    writer.writerow(["trajectory", *header])
                elif columns != header:
                    raise SystemExit(
                        f"{path.name} has columns {columns} but {paths[0].name} has {header}. "
                        "Different layer coverage means a different gather; join them separately."
                    )
                for row in reader:
                    writer.writerow([trajectory, *row])
                    rows += 1
            if verbose:
                print(f"  {trajectory}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tree", type=Path, help="activations tree holding the per-trajectory mass tables")
    parser.add_argument("--lens", default="jlens", help="jlens or logitlens (default jlens)")
    parser.add_argument("--out", type=Path, required=True, help="the concatenated CSV to write")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    paths = find_mass_tables(args.tree, args.lens)
    meta = check_one_vocabulary(paths)
    rows = join_mass_tables(args.tree, args.lens, args.out, verbose=args.verbose)
    print(f"{len(paths)} {args.lens} tables -> {rows} token rows -> {args.out}")
    if meta:
        print(f"  vocabulary: {meta.get('signal_json')} (fingerprint {meta.get('signal_fingerprint')})")
        if meta.get("data_sample_p") is not None:
            print(f"  sampled gather: p={meta['data_sample_p']} (seed {meta.get('data_sample_seed')})")


if __name__ == "__main__":
    sys.exit(main())
