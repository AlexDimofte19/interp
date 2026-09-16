#!/usr/bin/env python3
"""Fold a directory of `eval_binary_cognitive_map_probe` JSONs into one CSV.

One row per (positive class, model type, split, group), so the four binary arms and their
two evaluation sets sit in one table. `group` is `global` for the headline number and
`size{N}` / `comp{X}` for the breakdowns, which is what makes "the wall probe only works on
small grids" visible without opening eight files.

Stdlib only -- this runs on a laptop against pulled-back results, with no torch.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

COLUMNS = [
    "positive_class",
    "model_type",
    "split",
    "group",
    "n_rows",
    "positive_support",
    "positive_rate",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "auroc",
    "tp",
    "fp",
    "tn",
    "fn",
    "n_trajectories",
    "n_entries",
    "probe_path",
    "data_path",
]

#: The grid symbols, in the order the tables should read.
CLASS_ORDER = ["_", "#", "A", "G", "D", "K", "?", "+"]


def _rows(result: dict, split: str):
    """One row per metric block in a single result JSON.

    Args:
        result: A parsed `eval_binary_cognitive_map_probe` result.
        split: Which evaluation set this file scored.

    Yields:
        Dicts keyed by COLUMNS.
    """
    probe = result["probe"]
    data = result["data"]
    blocks = [("global", result["global"])]
    blocks += [(f"size{k}", v) for k, v in result.get("by_size", {}).items()]
    blocks += [(f"comp{k}", v) for k, v in result.get("by_complexity", {}).items()]

    for group, block in blocks:
        row = {
            "positive_class": probe["positive_class"],
            "model_type": probe["model_type"],
            "split": split,
            "group": group,
            "n_trajectories": data["n_trajectories"],
            "n_entries": data["n_entries"],
            "probe_path": probe["path"],
            "data_path": data["path"],
        }
        for key in COLUMNS:
            if key not in row:
                row[key] = block.get(key, "")
        yield row


def _split_of(path: Path) -> str:
    """The evaluation set a result filename names.

    Args:
        path: `eval_{class}_{model}_{split}.json`.

    Returns:
        The split token, or the whole stem if the name does not follow the convention.

    Example:
        >>> _split_of(Path("eval_wall_mlp_heldout360.json"))
        'heldout360'
    """
    parts = path.stem.split("_")
    return parts[-1] if len(parts) >= 2 else path.stem


def main() -> int:
    """Collect the JSONs and write the CSV.

    Returns:
        0 on success, 1 when the directory held no results.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, required=True, help="Directory holding eval_*.json")
    parser.add_argument("--out", type=Path, required=True, help="CSV to write")
    parser.add_argument("--glob", default="eval_*.json", help="Filename pattern (default: eval_*.json)")
    args = parser.parse_args()

    rows = []
    files = sorted(args.results_dir.glob(args.glob))
    for path in files:
        try:
            result = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  skipping {path.name}: {exc}", file=sys.stderr)
            continue
        if "probe" not in result or "positive_class" not in result.get("probe", {}):
            continue  # a multiclass eval JSON sharing the directory
        rows.extend(_rows(result, _split_of(path)))

    if not rows:
        print(f"No binary results matched {args.results_dir}/{args.glob}", file=sys.stderr)
        return 1

    def sort_key(row):
        order = CLASS_ORDER.index(row["positive_class"]) if row["positive_class"] in CLASS_ORDER else 99
        return (order, row["model_type"], row["split"], row["group"] != "global", row["group"])

    rows.sort(key=sort_key)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    header = f"{'class':<7}{'head':<6}{'split':<12}{'rows':>10}{'pos rate':>10}{'bal acc':>10}{'recall':>10}{'prec':>10}{'auroc':>10}"
    print(f"\nRead {len(files)} files -> {len(rows)} rows -> {args.out}\n")
    print(header)
    print("-" * len(header))
    for row in rows:
        if row["group"] != "global":
            continue
        print(
            f"{row['positive_class']:<7}{row['model_type']:<6}{row['split']:<12}"
            f"{row['n_rows']:>10}{float(row['positive_rate']):>10.4f}"
            f"{float(row['balanced_accuracy']):>10.4f}{float(row['recall']):>10.4f}"
            f"{float(row['precision']):>10.4f}{float(row['auroc']):>10.4f}"
        )
    print("-" * len(header))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
