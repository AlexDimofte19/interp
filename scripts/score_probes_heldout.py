#!/usr/bin/env python3
"""Balanced accuracy per probe on the held-out 360, against both label definitions.

The numbers table for a round that does not (yet) rebuild the figure report. Reads the
per-token CSV that ``build_probe_loudness_heldout.py`` writes -- one row per reasoning token
of the held-out 360, carrying every probe's prediction plus ``label_local`` (the at-token
belief measured by the ``every_token`` rollout) and ``label_final`` (the trajectory's own
``agent_action``) -- and reduces it to one row per probe.

TWO POPULATIONS, AND THEY ARE NOT COMPARABLE. The balanced accuracy inside a probe's own
checkpoint is measured on the tokens ITS OWN selection picked, so a loud-selected probe is
scored only on loud tokens and the arms do not share a population. What this script computes
is every probe read on the SAME 87,221 tokens with no selection in between. Entry 49's result
is that the two orderings INVERT, so a number is meaningless without naming which one it is.

Balanced accuracy is the mean per-class recall over classes with support, matching
``train_next_action_probe::_evaluate`` so the two columns are the same statistic.

A probe appears once per rowset it was scored in; since entry 48 the three rowsets hold
identical rows, so the per-rowset numbers agree and the default collapses them. ``--by-rowset``
keeps them apart for a CSV that predates that.
"""

import argparse
import collections
import csv
import json
import sys
from pathlib import Path


def balanced_accuracy(pairs: list[tuple[str, str]]) -> float | None:
    """Mean per-class recall over the classes that have support."""
    per: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for truth, pred in pairs:
        cell = per[truth]
        cell[1] += 1
        cell[0] += truth == pred
    recalls = [hit / n for hit, n in per.values() if n]
    return sum(recalls) / len(recalls) if recalls else None


def score(path: Path, by_rowset: bool) -> list[dict]:
    rows: dict[tuple[str, str], dict[str, list[tuple[str, str]]]] = collections.defaultdict(
        lambda: {"local": [], "final": []}
    )
    seen_rows = 0
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        probes = [c[: -len("_pred")] for c in reader.fieldnames or [] if c.endswith("_pred")]
        if not probes:
            raise SystemExit(f"{path}: no *_pred columns -- is this a build_probe_loudness_heldout.py output?")
        for row in reader:
            seen_rows += 1
            rowset = row.get("rowset", "") if by_rowset else ""
            for probe in probes:
                pred = row.get(f"{probe}_pred", "")
                if not pred:
                    continue  # a probe is only scored in the rowset(s) it was registered for
                cell = rows[(probe, rowset)]
                for key, label in (("local", row.get("label_local", "")), ("final", row.get("label_final", ""))):
                    if label:
                        cell[key].append((label, pred))
    out = []
    for (probe, rowset), cell in rows.items():
        out.append(
            {
                "probe": probe,
                "rowset": rowset,
                "n": len(cell["local"]),
                "bal_vs_belief": balanced_accuracy(cell["local"]),
                "bal_vs_final": balanced_accuracy(cell["final"]),
            }
        )
    out.sort(key=lambda r: (-(r["bal_vs_belief"] or 0), r["probe"]))
    print(f"read {seen_rows:,} rows, {len(probes)} probes", file=sys.stderr)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("per_token_csv", type=Path, help="build_probe_loudness_heldout.py's per-token CSV")
    ap.add_argument("--out", type=Path, default=None, help="also write the table as CSV")
    ap.add_argument("--json-out", type=Path, default=None, help="also write the table as JSON")
    ap.add_argument("--by-rowset", action="store_true", help="do not collapse the (identical) rowsets")
    args = ap.parse_args()

    table = score(args.per_token_csv, args.by_rowset)
    width = max(len(r["probe"]) for r in table)
    print(f"{'probe'.ljust(width)}  {'n':>7}  {'vs belief':>9}  {'vs final':>9}")
    for r in table:
        b = f"{r['bal_vs_belief']:.4f}" if r["bal_vs_belief"] is not None else "-"
        f = f"{r['bal_vs_final']:.4f}" if r["bal_vs_final"] is not None else "-"
        print(f"{r['probe'].ljust(width)}  {r['n']:>7,}  {b:>9}  {f:>9}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["probe", "rowset", "n", "bal_vs_belief", "bal_vs_final"])
            w.writeheader()
            w.writerows(table)
        print(f"wrote {args.out}", file=sys.stderr)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(table, indent=1))
        print(f"wrote {args.json_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
