#!/usr/bin/env python3
"""Merge per-probe `--per-token-out` files for ONE prepared slice into one per-token table.

Each evaluator run (`eval_grid_probe.py` / `rollouts/eval_local_belief.py` with
`--per-token-out`) writes one probe's verdicts on one slice, in count form
(`slice_tokens.write_counts`). This puts the probes side by side, in the column layout of
`score_probes_per_token.py`'s tables, so the loudness notebook reads a slice exactly as it
reads the held-out set:

    name, step, abs_pos, token_idx, token,
    n_true_{c} ...                     shared by every probe -- checked, not assumed
    {probe}_correct_{c} ..., {probe}_n_correct
    {signal}_word, {signal}_word_r{R}  per --signal

and nothing else. Loudness comes after, from `join_signal_loudness.py` on (name, step, abs_pos).

ABS_POS FROM THE TRAJECTORY. A manifest entry names its token by `token_id`, the
category-relative `.pt` index (= `token_idx`); the lens tables key on the prompt-inclusive
`abs_pos`. `abs_pos = token_idx + output_start`, `score_probes_per_token.py`'s own mapping.

THE SIGNAL-WORD WINDOW IS TAKEN OVER THE WHOLE REASONING CHAIN, not over the slice's rows. A
slice holds a few dozen tokens of a chain, so a token's neighbours are almost never rows, and
shifting within the table (what a dense held-out table can do) would find none. The flag is
`any signal word within +-R reasoning tokens` in the step's own chain -- the same definition a
dense table's row shift computes, since there every reasoning token is a row. Tokens are
decoded from byte-level BPE before matching (`slice_tokens.decode_bpe`).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from telos_interp.jlens_utils.jlens_csv import output_start
from telos_interp.loudness_analysis.build_loudness_tables import reasoning_token_positions
from telos_interp.loudness_analysis.slice_tokens import COUNT_KEY, decode_bpe

csv.field_size_limit(10**9)


def read_counts(path: Path) -> tuple[list[tuple], list[str], list[list[int]], list[list[int]]]:
    """(keys, classes, n_true rows, correct rows) from one `write_counts` file."""
    with open(path, newline="", encoding="utf-8") as fh:
        r = csv.reader(fh)
        header = next(r)
        classes = [h[len("n_true_") :] for h in header if h.startswith("n_true_")]
        k = len(COUNT_KEY)
        keys, n_true, correct = [], [], []
        for row in r:
            keys.append((row[0], int(row[1]), int(row[2])))
            n_true.append([int(x) for x in row[k : k + len(classes)]])
            correct.append([int(x) for x in row[k + len(classes) :]])
    return keys, classes, n_true, correct


def step_index(traj: dict, step_folder: int) -> int:
    """List index of the step whose folder number (`step_id`) is `step_folder`."""
    for i, s in enumerate(traj["steps"]):
        if int(s.get("step_id", i)) == step_folder:
            return i
    raise KeyError(f"no step with step_id {step_folder}")


def window_flags(tokens: list[str], vocab: set[str], radius: int) -> tuple[list[bool], list[bool]]:
    """Per reasoning token: is it a signal word, and is any signal word within +-radius."""
    word = [decode_bpe(t) in vocab for t in tokens]
    near = [any(word[max(0, i - radius) : i + radius + 1]) for i in range(len(word))]
    return word, near


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe-counts", type=Path, action="append", required=True,
                    help="One evaluator --per-token-out file; the probe key is its stem (repeatable).")
    ap.add_argument("--trajectories-dir", type=Path, required=True, help="Searched recursively for {name}.json.")
    ap.add_argument("--signal", action="append", default=[], metavar="NAME=JSON",
                    help="Flag this vocabulary's words: writes {NAME}_word and {NAME}_word_r{R} (repeatable).")
    ap.add_argument("--exclude-radius", type=int, default=2, help="R in {NAME}_word_r{R} (default 2).")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    probes = [(p.stem, *read_counts(p)) for p in args.probe_counts]
    key0, classes, true0 = probes[0][1], probes[0][2], probes[0][3]
    for name, keys, cls, n_true, _ in probes[1:]:
        # One slice means one set of rows and one ground truth: a probe read on different
        # entries, or whose cells were drawn differently, cannot share a column block.
        if keys != key0 or cls != classes or n_true != true0:
            raise SystemExit(f"{name}: rows or ground truth differ from {probes[0][0]} -- not the same slice")

    vocabs = {}
    for spec in args.signal:
        sname, path = spec.split("=", 1)
        vocabs[sname] = {t for lst in json.loads(Path(path).read_text()).values() for t in lst}

    wanted = {k[0] for k in key0}
    paths = {p.stem: p for p in args.trajectories_dir.rglob("*.json") if p.stem in wanted}
    missing = wanted - set(paths)
    if missing:
        raise SystemExit(f"{len(missing)} trajectory JSON(s) not under {args.trajectories_dir}, e.g. {sorted(missing)[:3]}")

    R = args.exclude_radius
    header = ["name", "step", "abs_pos", "token_idx", "token"] + [f"n_true_{c}" for c in classes]
    for name, *_ in probes:
        header += [f"{name}_correct_{c}" for c in classes] + [f"{name}_n_correct"]
    for sname in vocabs:
        header += [f"{sname}_word", f"{sname}_word_r{R}"]

    chains: dict[tuple[str, int], tuple[int, dict]] = {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for i, (name, step, token_idx) in enumerate(key0):
            if (name, step) not in chains:
                traj = json.loads(paths[name].read_text())
                si = step_index(traj, step)
                start = output_start(traj, si)
                positions = reasoning_token_positions(traj, traj["steps"][si])  # (rp, abs_pos, token)
                toks = [t for _, _, t in positions]
                flags = {s: window_flags(toks, v, R) for s, v in vocabs.items()}
                by_abs = {ap: j for j, (_, ap, _) in enumerate(positions)}
                chains[(name, step)] = (start, {"by_abs": by_abs, "toks": toks, "flags": flags})
            start, c = chains[(name, step)]
            abs_pos = token_idx + start
            j = c["by_abs"].get(abs_pos)
            if j is None:
                raise SystemExit(f"{name} step {step} token_idx {token_idx}: abs_pos {abs_pos} is not a reasoning token")
            row = [name, step, abs_pos, token_idx, c["toks"][j], *true0[i]]
            for _, _, _, _, correct in probes:
                row += [*correct[i], sum(correct[i])]
            for s in vocabs:
                word, near = c["flags"][s]
                row += [int(word[j]), int(near[j])]
            w.writerow(row)
    print(f"wrote {len(key0)} rows x {len(probes)} probes -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
