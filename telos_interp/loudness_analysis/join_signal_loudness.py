#!/usr/bin/env python3
"""Add a SECOND signal's loudness to a per-token probe table that already exists.

``score_probes_per_token.py`` writes the probe verdicts and the loudness columns in one
pass, which is right the first time and wrong every time after: asking the same probes
"does *grid* loudness predict you, rather than *direction* loudness?" re-reads 87k
activations and re-runs every probe to change nothing but four columns per lens. That is
hours of GPU to recompute a join.

This is that join on its own. It takes the table as it stands, reads a signal-mass tree
gathered against a DIFFERENT vocabulary, and writes the same table with the new signal's
columns added -- CPU only, minutes not hours. The probe columns are copied through
untouched, so a row's verdict is byte-for-byte the one the evaluator produced and any
difference between two rulers is the ruler.

THE RULER IS THE SIDECAR, NOT THE FILENAME. Every mass table on disk is called
``{stem}_{lens}_direction_mass.csv`` whatever vocabulary produced it, so the name cannot
say which question its numbers answer. ``--signal-name`` states what the caller believes
the tree holds and every table is checked against its ``.meta.json`` before a single cell
is read; a tree with no sidecars is refused outright rather than silently relabelled.

THE KEY IS ``(name, step, abs_pos)``. ``abs_pos`` is prompt-inclusive, and it is what both
the mass table and the evaluator's table carry, so this join needs no offset. A table keyed
on ``token_idx`` would need ``output_start`` and is not what this reads.

Columns written per lens, canonical spelling (``columns.score_columns``):

    {lens}_{signal}_logmass_L{layer}        the loudness: log P(any signal word) at --layer
    {lens}_{signal}_logmass_best_layer      the token's argmax layer
    {lens}_{signal}_logmass_best            its mass there -- a DIFFERENT question from L15

A token the tree has no row for gets empty cells rather than a floor value: absent is not
quiet, and a decile binner must be able to tell them apart.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from telos_interp.jlens_utils import read_mass_meta
from telos_interp.jlens_utils.methods import direction_mass_path
from telos_interp.loudness_analysis import columns as cols
from telos_interp.loudness_analysis import provenance, signals
from telos_interp.loudness_analysis.lens_io import read_mass_columns, trajectory_dirs

# csv.DictReader everywhere, never pandas: decoded tokens include "NA", empty strings,
# embedded commas and newlines, which pandas' NA handling silently corrupts.
csv.field_size_limit(10**9)

# Sidecar fields naming the vocabulary a table was baked against. `signal_name` is the one
# this checks; the rest are reported so a mismatch says what the tree actually holds.
VOCAB_FIELDS = ("signal_name", "signal_json", "signal_fingerprint", "direction_classes")


def mass_columns_for(folder: Path, stem: str, lens: str, signal: str, strict: bool) -> dict:
    """``{(step, abs_pos): {layer: logmass}}`` for one trajectory and one lens.

    Returns `{}` when that lens has no table here, which is normal: `--lens jlens` trees
    carry one and `--lens both` trees carry two.

    Raises:
        SystemExit: when the table's sidecar names a different vocabulary than `signal`,
            or when it has no sidecar at all and `strict`. Reading it anyway would relabel
            one question as another and nothing in the numbers would reveal it.
    """
    path = direction_mass_path(folder, lens)
    if path is None or not path.exists():
        return {}
    meta = read_mass_meta(path)
    if not meta:
        if strict:
            raise SystemExit(
                f"{path} has no .meta.json sidecar, so the vocabulary its cells were "
                f"computed over is unknown. Re-gather it, or pass --allow-unlabelled to "
                f"assert it is {signal!r}."
            )
    elif meta.get("signal_name", "direction") != signal:
        raise SystemExit(
            f"{path} was baked against signal {meta.get('signal_name', 'direction')!r}, not "
            f"{signal!r} -- {{{', '.join(f'{k}: {meta.get(k)!r}' for k in VOCAB_FIELDS)}}}. "
            f"These are different questions; point --lens-root at the right tree."
        )
    return read_mass_columns(path)


def cells(masses: dict, key: tuple[int, int], layer: int) -> list:
    """The three cells for one (token, lens), matching `columns.score_columns`'s last three.

    Empty strings when the tree has no row for this token: absent is not quiet, and a
    floor value here would put unmeasured tokens in the quietest decile.
    """
    per_layer = masses.get(key)
    if not per_layer:
        return ["", "", ""]
    best_layer = max(per_layer, key=lambda ly: per_layer[ly])
    return [per_layer.get(layer, ""), best_layer, per_layer[best_layer]]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", type=Path, required=True, help="Per-token CSV carrying name, step, abs_pos.")
    ap.add_argument("--lens-root", type=Path, required=True, help="Tree holding the signal-mass tables.")
    ap.add_argument("--out", type=Path, required=True, help="Where to write the widened table.")
    ap.add_argument(
        "--signal-name",
        required=True,
        help="Vocabulary the tree was baked against. Names the new columns, and is CHECKED "
        "against every table's sidecar before anything is read.",
    )
    ap.add_argument(
        "--signal-json",
        type=Path,
        default=None,
        help="That vocabulary's JSON, recorded in run_config.json with a hash of its contents.",
    )
    ap.add_argument("--lenses", default="jlens,logitlens", help="Comma-separated; three columns each.")
    ap.add_argument("--layer", type=int, default=15, help="Layer the loudness is read at (default 15).")
    ap.add_argument(
        "--allow-unlabelled",
        action="store_true",
        help="Accept mass tables with no .meta.json sidecar, asserting they hold --signal-name. "
        "Only for trees written before sidecars existed.",
    )
    args = ap.parse_args(argv)

    signal = signals.resolve(args.signal_name, args.signal_json)
    lenses = [x.strip() for x in args.lenses.split(",") if x.strip()]

    with open(args.table, newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    for lens in lenses:
        # Three of the four -- the count comes from an analysis CSV, not a mass table.
        new = cols.score_columns(lens, signal.name, args.layer)[1:]
        clash = [c for c in new if c in header]
        if clash:
            raise SystemExit(f"--table already has {clash}; it was joined against {signal.name!r} already")

    folders = {f.name: f for f in trajectory_dirs(args.lens_root)}
    print(f"{len(folders)} trajectory folder(s) under {args.lens_root}", flush=True)
    if not folders:
        raise SystemExit(f"no lens artifacts under {args.lens_root}")

    new_cols: list[str] = []
    for lens in lenses:
        new_cols += cols.score_columns(lens, signal.name, args.layer)[1:]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cache: tuple[str, dict] = ("", {})
    written = matched = missing_folder = missing_token = 0
    with open(args.table, newline="", encoding="utf-8") as src, open(args.out, "w", newline="") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=list(reader.fieldnames or []) + new_cols)
        writer.writeheader()
        for row in reader:
            stem = row["name"]
            # Rows arrive grouped by trajectory, so a one-entry cache reads each tree once.
            if cache[0] != stem:
                folder = folders.get(stem)
                cache = (
                    stem,
                    {
                        lens: mass_columns_for(folder, stem, lens, signal.name, not args.allow_unlabelled)
                        for lens in lenses
                    }
                    if folder is not None
                    else {},
                )
            per_lens = cache[1]
            if not per_lens:
                missing_folder += 1
            key = (int(row["step"]), int(row["abs_pos"]))
            hit = False
            for lens in lenses:
                values = cells(per_lens.get(lens, {}), key, args.layer)
                hit = hit or values[0] != ""
                for name, value in zip(cols.score_columns(lens, signal.name, args.layer)[1:], values, strict=True):
                    row[name] = value
            if hit:
                matched += 1
            elif per_lens:
                missing_token += 1
            writer.writerow(row)
            written += 1

    print(f"\nwrote {written} row(s) -> {args.out}")
    print(f"  {matched} joined, {missing_token} not in the tree, {missing_folder} in an absent trajectory")
    if not matched:
        raise SystemExit(
            "nothing joined: the key is (name, step, abs_pos) and abs_pos is prompt-inclusive. "
            "A table keyed on token_idx joins to nothing here, without an error."
        )

    cfg = provenance.RunConfig("loudness_analysis/join_signal_loudness.py")
    cfg.measurement(lens=lenses[0], signal=signal.name, layer=args.layer, signal_json=args.signal_json)
    cfg.input("table", args.table)
    cfg.input("lens_tables", args.lens_root)
    cfg.params.update({"lenses": lenses, "layer": args.layer, "added_columns": new_cols})
    cfg.rows("input", written)
    cfg.rows("joined", matched)
    cfg.rows("unjoined", written - matched)
    cfg.outputs.append(args.out)
    cfg.write(args.out.parent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
