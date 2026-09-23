#!/usr/bin/env python3
"""Confirm that an EXISTING gpt-oss activation tree is the one a wrapper claims to reuse.

The gpt-oss direction line does not gather: every tree it needs was built by an earlier run
(jlens_mass_l15, logitlens_mass_l15, heldout360_l15, heldout360_lens). The wrappers that sit
where the Qwen line's gathers sit call this instead. It checks the tree and writes nothing.
Re-running a gather into a tree that is already pruned is the one thing that must never
happen by accident.

For every trajectory JSON under --trajectories (``size*/<name>.json``) it checks that
``{tree}/{size}/{name}/`` holds:

  * the ``{name}_{lens}_direction_mass.csv`` table AND its ``.meta.json`` sidecar, for each --lens
    (a mass table without its sidecar must not be read, see CLAUDE.md);
  * a sidecar built over the WHOLE chain: ``data_sample_p`` absent or 1.0;
  * a sidecar covering every --layers value and naming the --signal vocabulary by basename;
  * with --arms, a ``{name}_jlens_selection.json`` holding those arms, each with
    ``candidate_layers == [--candidate-layer]``;
  * with --pt-layer, a ``<model>/layer_{L}/`` directory of tensors.

Stdlib only. Exit status 1 on any failure, with the first few per check printed.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def parse_layers(spec: str) -> list[int]:
    """'15' -> [15], '7:23' -> [7..23] inclusive, '7,15' -> [7, 15].

    >>> parse_layers("7:9")
    [7, 8, 9]
    >>> parse_layers("15")
    [15]
    """
    out: list[int] = []
    for part in spec.split(","):
        if ":" in part:
            lo, hi = part.split(":")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", type=Path, required=True)
    ap.add_argument("--trajectories", type=Path, required=True, help="size*/<name>.json; defines the names checked")
    ap.add_argument("--lens", action="append", default=[], help="repeatable: jlens, logitlens")
    ap.add_argument("--layers", default="15", help="layers every mass table must cover")
    ap.add_argument("--signal", default="direction_tokens_full.json", help="vocabulary basename the sidecar must name")
    ap.add_argument("--arms", default="", help="comma-separated arms the selection record must hold")
    ap.add_argument("--candidate-layer", type=int, default=15)
    ap.add_argument("--pt-layer", type=int, default=None, help="require <model>/layer_{L}/ under each trajectory")
    args = ap.parse_args()

    need_layers = set(parse_layers(args.layers))
    arms = [a for a in args.arms.split(",") if a]
    names = sorted(args.trajectories.glob("size*/*.json"))
    if not names:
        print(f"!! no size*/*.json under {args.trajectories}", file=sys.stderr)
        return 1
    if not args.tree.is_dir():
        print(f"!! tree not found: {args.tree}", file=sys.stderr)
        return 1

    problems: dict[str, list[str]] = defaultdict(list)
    for traj in names:
        name, size = traj.stem, traj.parent.name
        folder = args.tree / size / name
        if not folder.is_dir():
            problems["missing trajectory folder"].append(str(folder))
            continue
        for lens in args.lens:
            table = folder / f"{name}_{lens}_direction_mass.csv"
            meta_path = Path(f"{table}.meta.json")
            if not table.is_file() or not meta_path.is_file():
                problems[f"{lens}: missing mass table or sidecar"].append(str(table))
                continue
            meta = json.loads(meta_path.read_text())
            p = meta.get("data_sample_p", 1.0)
            if p is not None and float(p) < 1.0:
                problems[f"{lens}: sampled (data_sample_p < 1)"].append(f"{table} p={p}")
            missing = need_layers - set(meta.get("layers", []))
            if missing:
                problems[f"{lens}: layers not covered"].append(f"{table} missing {sorted(missing)}")
            if Path(meta.get("signal_json", "")).name != args.signal:
                problems[f"{lens}: wrong vocabulary"].append(f"{table} -> {meta.get('signal_json')}")
        if arms:
            rec_path = folder / f"{name}_jlens_selection.json"
            if not rec_path.is_file():
                problems["missing selection record"].append(str(rec_path))
            else:
                rec = json.loads(rec_path.read_text()).get("arms", {})
                for arm in arms:
                    if arm not in rec:
                        problems[f"arm {arm}: not in record"].append(str(rec_path))
                    elif rec[arm].get("config", {}).get("candidate_layers") != [args.candidate_layer]:
                        problems[f"arm {arm}: candidate_layers != [{args.candidate_layer}]"].append(str(rec_path))
        if args.pt_layer is not None and not any(folder.glob(f"*/layer_{args.pt_layer}")):
            problems[f"no layer_{args.pt_layer} tensors"].append(str(folder))

    print(f"tree         {args.tree}")
    print(f"trajectories {args.trajectories}  ({len(names)} names)")
    print(f"lenses       {args.lens or '-'}   layers {sorted(need_layers)}   vocabulary {args.signal}")
    print(f"arms         {arms or '-'}   pt layer {args.pt_layer if args.pt_layer is not None else '-'}")
    if not problems:
        print("OK -- every trajectory is present and matches; nothing to gather.")
        return 0
    for what, items in problems.items():
        print(f"!! {what}: {len(items)}", file=sys.stderr)
        for item in items[:3]:
            print(f"     {item}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
