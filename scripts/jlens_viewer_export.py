"""Join the jlens reasoning-token CSV with trajectory JSONs into per-step viewer files.

Streams `jlens_reasoning_tokens_own_jlens_1000_traj.csv` (3.2 GB, ~16M rows) exactly
once and writes, for each selected trajectory, a folder of self-contained step files:

    <out-dir>/size11_comp0.0_run0/manifest.json
    <out-dir>/size11_comp0.0_run0/step_000.json
    ...

Those folders are what `telos_interp/trace_viewer/trace_viewer_fork.html` loads (the
fork loads a FOLDER, not a single file). Schema is documented in
`telos_interp/trace_viewer/README.md` ("Fork: jlens per-step files").

The CSV is never held in memory: rows arrive grouped by trajectory -> step -> layer
(see scripts/jlens_reasoning_tokens.py), so we buffer one step at a time and flush.

Usage:
  python scripts/jlens_viewer_export.py \
    --csv /media/alex/D/Uni/northeastern/data/jlens/jlens_reasoning_tokens_own_jlens_1000_traj.csv \
    --trajectory-paths /media/alex/D/Uni/northeastern/data/trajectories/trajectories_test_full \
    --direction-tokens /media/alex/D/Uni/northeastern/data/jlens/direction_tokens.json \
    --sizes 11 --complexities 0.0 --runs 0
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Running this file directly puts scripts/ on sys.path[0], not the repo root, so the
# sibling `scripts.*` imports below would miss. Put the repo root first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.jlens_reasoning_tokens import (  # noqa: E402  (stdlib-only module)
    ACTIONS,
    TOP_K,
    parse_name,
    reasoning_token_positions,
)

SCHEMA = "trace_viewer_fork/1"
DEFAULT_OUT_DIR = Path("/media/alex/D/Uni/northeastern/data/jlens_viewer")
# vocab_dir encoding, mirrored in trace_viewer_fork.html
DIR_CODES = {"UP": 1, "DOWN": 2, "LEFT": 3, "RIGHT": 4}
REQUIRED_COLUMNS = (
    ["size", "complexity", "run", "step", "reasoning_pos", "abs_pos", "token", "layer", "agent_action"]
    + [f"{a}_rank" for a in ACTIONS]
    + [f"{a}_logprob" for a in ACTIONS]
    + [f"top_{i}" for i in range(1, TOP_K + 1)]
)


def raise_field_size_limit() -> None:
    """csv fields hold decoded tokens with embedded newlines; give the parser room."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 2


def load_direction_lookup(path: Path) -> dict[str, int]:
    """direction_tokens.json -> {decoded token string: DIR_CODES value}.

    Keys are `tokenizer.decode()` output (real leading spaces), the same convention as
    the CSV's top_k columns -- never the Ġ-prefixed convention of the `token` column.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    lookup: dict[str, int] = {}
    for action, code in DIR_CODES.items():
        for tok in raw.get(action, []):
            lookup.setdefault(tok, code)
    return lookup


def traj_id(name: dict) -> str:
    return f"size{name['size']}_comp{name['comp']}_run{name['run']}"


def group_key(size: str, comp: str, run: str) -> tuple[int, float, int]:
    """Numeric join key: '1.0' vs '1' and '01' vs '1' must not split a group."""
    return (int(size), float(comp), int(run))


def intern_topk(
    cells: dict[tuple[int, int], dict],
    n_pos: int,
    layers: list[int],
    dir_lookup: dict[str, int],
) -> dict:
    """Flatten buffered rows into interned vocab + flat arrays.

    cells maps (reasoning_pos, layer) -> {"ranks": [...], "logprobs": [...], "topk": [...]}.
    Flat index convention: i = pos * n_layers + layer_idx; top-k entry k is topk[i * TOP_K + k].
    """
    n_layers = len(layers)
    n_cells = n_pos * n_layers
    vocab: list[str] = []
    vocab_dir: list[int] = []
    vocab_index: dict[str, int] = {}

    topk = [-1] * (n_cells * TOP_K)
    rank = {a: [None] * n_cells for a in ACTIONS}
    logprob = {a: [None] * n_cells for a in ACTIONS}
    density = {a: [0] * n_cells for a in ACTIONS}
    code_to_action = {v: k for k, v in DIR_CODES.items()}
    missing = 0

    for pos in range(n_pos):
        for li, layer in enumerate(layers):
            cell = cells.get((pos, layer))
            i = pos * n_layers + li
            if cell is None:
                missing += 1
                continue
            for ai, action in enumerate(ACTIONS):
                rank[action][i] = cell["ranks"][ai]
                logprob[action][i] = cell["logprobs"][ai]
            for k, text in enumerate(cell["topk"]):
                vi = vocab_index.get(text)
                if vi is None:
                    vi = len(vocab)
                    vocab_index[text] = vi
                    vocab.append(text)
                    vocab_dir.append(dir_lookup.get(text, 0))
                topk[i * TOP_K + k] = vi
                code = vocab_dir[vi]
                if code:
                    density[code_to_action[code]][i] += 1

    return {
        "vocab": vocab,
        "vocab_dir": vocab_dir,
        "topk": topk,
        "rank": rank,
        "logprob": logprob,
        "density": density,
        "missing_cells": missing,
    }


def build_step_payload(
    trajectory: dict,
    step_index: int,
    cells: dict[tuple[int, int], dict],
    dir_lookup: dict[str, int],
    meta: dict,
    drop_probabilities: bool = False,
) -> dict:
    """One step file. `cells` empty -> lens is null (the file is still emitted)."""
    step = trajectory["steps"][step_index]
    warnings: list[str] = []

    output_tokens = step.get("output_tokens", [])
    if drop_probabilities:
        output_tokens = [{k: v for k, v in t.items() if k != "probabilities"} for t in output_tokens]

    payload = {
        "schema": SCHEMA,
        "trajectory": {
            "id": meta["id"],
            "size": int(meta["size"]),
            "complexity": float(meta["comp"]),
            "run": int(meta["run"]),
            "source": meta["source"],
            "n_steps": len(trajectory["steps"]),
        },
        "grid_params": trajectory.get("grid_params", {}),
        "model_params": trajectory.get("model_params", {}),
        "step": {
            "index": step_index,
            "step_id": step.get("step_id", step_index),
            "grid_state": step.get("grid_state", []),
            "agent_action": step.get("agent_action", ""),
            "output_text": step.get("output_text", ""),
            "output_tokens": output_tokens,
        },
        "lens": None,
        "warnings": warnings,
    }

    if not cells:
        warnings.append(f"no jlens rows for step {step_index}")
        return payload

    positions = reasoning_token_positions(trajectory, step)
    out = step.get("output_tokens", [])
    analysis_idxs = [i for i, t in enumerate(out) if "analysis" in t.get("token_groups", [])] or list(range(len(out)))
    layers = sorted({layer for _, layer in cells})
    n_pos_csv = max(pos for pos, _ in cells) + 1
    n_pos = min(n_pos_csv, len(positions))
    if n_pos_csv != len(positions):
        warnings.append(
            f"reasoning position count mismatch: csv {n_pos_csv} vs trajectory {len(positions)}; using {n_pos}"
        )

    if n_pos == 0:
        warnings.append("no reasoning tokens for this step")
        return payload

    # Cheap guard against pairing the CSV with the wrong trajectory: compare the first and
    # last reasoning token/abs_pos of the step against what the trajectory says.
    for pos in {0, n_pos - 1}:
        cell = cells.get((pos, layers[0]))
        if cell is None:
            continue
        _, abs_pos, token = positions[pos]
        if cell["token"] != token or cell["abs_pos"] != abs_pos:
            warnings.append(
                f"csv/trajectory mismatch at pos {pos}: csv ({cell['token']!r}, {cell['abs_pos']}) "
                f"vs trajectory ({token!r}, {abs_pos})"
            )

    arrays = intern_topk(cells, n_pos, layers, dir_lookup)
    payload["lens"] = {
        "actions": list(ACTIONS),
        "layers": layers,
        "n_pos": n_pos,
        "n_layers": len(layers),
        "top_k": TOP_K,
        "reasoning_positions": analysis_idxs[:n_pos],
        "reasoning_tokens": [t for _, _, t in positions[:n_pos]],
        "abs_pos": [ap for _, ap, _ in positions[:n_pos]],
        **arrays,
    }
    if arrays["missing_cells"]:
        warnings.append(f"{arrays['missing_cells']} missing (position, layer) cells")
    return payload


def write_step_file(out_dir: Path, payload: dict) -> Path:
    path = out_dir / f"step_{payload['step']['index']:03d}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    return path


def write_manifest(out_dir: Path, trajectory: dict, meta: dict, steps: list[dict], sources: dict) -> None:
    first_lens = next((s["lens"] for s in steps if s["lens"]), None)
    manifest = {
        "schema": SCHEMA,
        "trajectory": steps[0]["trajectory"] if steps else {"id": meta["id"]},
        "grid_params": trajectory.get("grid_params", {}),
        "model_params": trajectory.get("model_params", {}),
        "layers": first_lens["layers"] if first_lens else [],
        "actions": list(ACTIONS),
        "steps": [
            {
                "file": f"step_{s['step']['index']:03d}.json",
                "step_id": s["step"]["step_id"],
                "n_pos": s["lens"]["n_pos"] if s["lens"] else 0,
                "agent_action": s["step"]["agent_action"],
                "has_lens": s["lens"] is not None,
            }
            for s in steps
        ],
        "source_csv": sources["csv"],
        "direction_tokens_source": sources["direction_tokens"],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "warnings": [w for s in steps for w in s["warnings"]],
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def parse_steps_spec(spec: str, n_steps: int) -> list[int]:
    """'all' | '0,2' | '0:3' -> list of step indices (stdlib-only, clamped)."""
    if spec is None or spec.strip().lower() == "all":
        return list(range(n_steps))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            lo, _, hi = part.partition(":")
            lo_i = int(lo) if lo else 0
            hi_i = int(hi) if hi else n_steps
            out.extend(range(max(0, lo_i), min(n_steps, hi_i)))
        else:
            i = int(part)
            if 0 <= i < n_steps:
                out.append(i)
    return sorted(set(out))


class TrajectoryExporter:
    """Buffers one step's rows, flushes a step file, then a manifest per trajectory."""

    def __init__(self, args, dir_lookup: dict[str, int], sources: dict):
        self.args = args
        self.dir_lookup = dir_lookup
        self.sources = sources
        self.reset()

    def reset(self) -> None:
        self.meta = None
        self.trajectory = None
        self.out_dir = None
        self.step_index = None
        self.skip = False
        self.cells: dict[tuple[int, int], dict] = {}
        self.emitted: list[dict] = []
        self.wanted_steps: set[int] = set()

    def open_trajectory(self, path: Path) -> None:
        with open(path, encoding="utf-8") as f:
            self.trajectory = json.load(f)
        name = parse_name(path.stem)
        self.meta = {**name, "id": traj_id(name), "source": str(path)}
        self.out_dir = self.args.out_dir / self.meta["id"]
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.wanted_steps = set(parse_steps_spec(self.args.steps, len(self.trajectory["steps"])))
        self.step_index = None
        self.skip = False
        self.cells = {}
        self.emitted = []

    def flush_step(self) -> None:
        if self.step_index is None or self.skip:
            self.step_index = None
            self.cells = {}
            return
        payload = build_step_payload(
            self.trajectory,
            self.step_index,
            self.cells,
            self.dir_lookup,
            self.meta,
            drop_probabilities=self.args.drop_probabilities,
        )
        write_step_file(self.out_dir, payload)
        self.emitted.append(payload)
        for w in payload["warnings"]:
            print(f"  ! {self.meta['id']} step {self.step_index}: {w}", flush=True)
        self.step_index = None
        self.cells = {}

    def close_trajectory(self) -> None:
        """Flush the buffered step, emit lens-less files for steps the CSV never had."""
        if self.trajectory is None:
            return
        self.flush_step()
        seen = {p["step"]["index"] for p in self.emitted}
        for si in sorted(self.wanted_steps - seen):
            payload = build_step_payload(
                self.trajectory,
                si,
                {},
                self.dir_lookup,
                self.meta,
                drop_probabilities=self.args.drop_probabilities,
            )
            write_step_file(self.out_dir, payload)
            self.emitted.append(payload)
        self.emitted.sort(key=lambda p: p["step"]["index"])
        write_manifest(self.out_dir, self.trajectory, self.meta, self.emitted, self.sources)
        n_lens = sum(1 for p in self.emitted if p["lens"])
        print(f"  wrote {len(self.emitted)} step file(s) ({n_lens} with lens data) -> {self.out_dir}", flush=True)
        self.reset()

    def add_row(self, step: int, pos: int, layer: int, cell: dict) -> None:
        if step != self.step_index:
            self.flush_step()
            self.step_index = step
            self.skip = step not in self.wanted_steps
            self.cells = {}
        if self.skip:
            return
        self.cells[(pos, layer)] = cell


def expand_paths(patterns: list[str]) -> list[Path]:
    """Files, globs or directories -> trajectory JSON files.

    Same contract as ``expand_paths`` in scripts/inference_oss/run_inference.py, but
    stdlib-only: that module imports torch at import time, and this script must run on a
    laptop without it.
    """
    from glob import glob

    out: list[Path] = []
    for pattern in patterns:
        p = Path(pattern)
        if p.is_dir():
            out.extend(p.rglob("*.json"))
            continue
        matched = glob(pattern, recursive=True)
        if matched:
            for m in matched:
                mp = Path(m)
                out.extend(mp.rglob("*.json")) if mp.is_dir() else out.append(mp)
        elif p.exists():
            out.append(p)
    return sorted({p for p in out if p.suffix == ".json"})


def build_wanted(args) -> dict[tuple[int, float, int], Path]:
    paths = expand_paths([str(p) for p in args.trajectory_paths])
    sizes = {s.strip() for s in args.sizes.split(",")} if args.sizes else None
    comps = {c.strip() for c in args.complexities.split(",")} if args.complexities else None
    runs = {r.strip() for r in args.runs.split(",")} if args.runs else None

    wanted: dict[tuple[int, float, int], Path] = {}
    for p in sorted(paths):
        name = parse_name(p.stem)
        if not name["size"]:
            continue
        if sizes and name["size"] not in sizes:
            continue
        if comps and name["comp"] not in comps:
            continue
        if runs and name["run"] not in runs:
            continue
        wanted[group_key(name["size"], name["comp"], name["run"])] = p
    if args.max_trajectories is not None:
        wanted = dict(sorted(wanted.items())[: args.max_trajectories])
    return wanted


def list_coverage(csv_path: Path, out_dir: Path) -> None:
    """Scan the CSV once and report which (size, complexity, run) it actually contains.

    The CSV is a sample: it covers far fewer trajectories than trajectories_test_full holds,
    so this is the cheap way to find out what can be exported before asking for one.
    """
    seen: dict[tuple[str, str, str], set[str]] = {}
    started, n_rows = time.time(), 0
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        col = {name: i for i, name in enumerate(header)}
        i_size, i_comp, i_run, i_step = col["size"], col["complexity"], col["run"], col["step"]
        for row in reader:
            n_rows += 1
            seen.setdefault((row[i_size], row[i_comp], row[i_run]), set()).add(row[i_step])

    print(f"{n_rows:,} rows in {time.time() - started:.0f}s; {len(seen)} distinct trajectories", flush=True)
    for size in sorted({k[0] for k in seen}, key=int):
        comps = sorted({k[1] for k in seen if k[0] == size}, key=float)
        runs = sorted({k[2] for k in seen if k[0] == size}, key=int)
        n = sum(1 for k in seen if k[0] == size)
        print(f"  size{size}: {n} trajectories | complexities {comps} | runs {runs}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "csv_coverage.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {f"size{k[0]}_comp{k[1]}_run{k[2]}": sorted(int(s) for s in v) for k, v in sorted(seen.items())},
            f,
            indent=1,
        )
    print(f"wrote {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, required=True, help="jlens_reasoning_tokens CSV.")
    ap.add_argument("--trajectory-paths", nargs="+", type=Path, help="Trajectory JSON file(s), directory, or glob(s).")
    ap.add_argument("--direction-tokens", type=Path, help="direction_tokens.json.")
    ap.add_argument(
        "--list-coverage",
        action="store_true",
        help="Scan the CSV and report which trajectories it contains, then exit.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Export root; one folder per trajectory (default: {DEFAULT_OUT_DIR}).",
    )
    ap.add_argument("--sizes", default=None, help="Comma-separated grid sizes to keep, e.g. '11,15'.")
    ap.add_argument("--complexities", default=None, help="Comma-separated complexities, e.g. '0.0,1.0'.")
    ap.add_argument("--runs", default=None, help="Comma-separated run indices, e.g. '0,3'.")
    ap.add_argument("--max-trajectories", type=int, default=None, help="Export at most N trajectories.")
    ap.add_argument("--steps", default="all", help="Step spec: 'all', '0,2' or '0:3'.")
    ap.add_argument(
        "--drop-probabilities",
        action="store_true",
        help="Strip per-token next-token probabilities to shrink step files.",
    )
    ap.add_argument(
        "--no-early-exit", action="store_true", help="Keep scanning the CSV after every requested trajectory is done."
    )
    ap.add_argument("--progress-every", type=int, default=2_000_000, help="Rows between progress lines.")
    ap.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty output folder.")
    args = ap.parse_args()

    raise_field_size_limit()
    if args.list_coverage:
        list_coverage(args.csv, args.out_dir)
        return
    missing_args = [n for n in ("trajectory_paths", "direction_tokens") if getattr(args, n) is None]
    if missing_args:
        raise SystemExit(
            "missing required argument(s): " + ", ".join("--" + a.replace("_", "-") for a in missing_args)
        )
    dir_lookup = load_direction_lookup(args.direction_tokens)
    wanted = build_wanted(args)
    if not wanted:
        raise SystemExit("no trajectories matched the filters")
    print(f"{len(wanted)} trajectory file(s) requested", flush=True)

    if not args.overwrite:
        for path in wanted.values():
            out = args.out_dir / traj_id(parse_name(path.stem))
            if out.exists() and any(out.iterdir()):
                raise SystemExit(f"{out} is not empty (pass --overwrite)")

    sources = {"csv": str(args.csv), "direction_tokens": str(args.direction_tokens)}
    exporter = TrajectoryExporter(args, dir_lookup, sources)
    remaining = set(wanted)
    done: set[tuple[int, float, int]] = set()
    rank_cols = [f"{a}_rank" for a in ACTIONS]
    logprob_cols = [f"{a}_logprob" for a in ACTIONS]
    topk_cols = [f"top_{i}" for i in range(1, TOP_K + 1)]
    started = time.time()
    n_rows = 0

    with open(args.csv, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        col = {name: i for i, name in enumerate(header)}
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in col]
        if missing_cols:
            raise SystemExit(f"csv is missing columns: {missing_cols}")
        i_size, i_comp, i_run = col["size"], col["complexity"], col["run"]
        i_step, i_pos, i_abs = col["step"], col["reasoning_pos"], col["abs_pos"]
        i_token, i_layer = col["token"], col["layer"]
        i_ranks = [col[c] for c in rank_cols]
        i_logprobs = [col[c] for c in logprob_cols]
        i_topk = [col[c] for c in topk_cols]

        current_key = None
        for row in reader:
            n_rows += 1
            if n_rows % args.progress_every == 0:
                rate = n_rows / max(1e-9, time.time() - started)
                print(
                    f"  {n_rows / 1e6:.1f}M rows | {rate:,.0f} rows/s | {len(done)}/{len(wanted)} trajectories done",
                    flush=True,
                )
            key = group_key(row[i_size], row[i_comp], row[i_run])
            if key != current_key:
                # rows are contiguous per trajectory, so a key change ends the current one
                if current_key is not None:
                    exporter.close_trajectory()
                    done.add(current_key)
                    remaining.discard(current_key)
                    current_key = None
                    if not remaining and not args.no_early_exit:
                        print("  all requested trajectories exported; stopping early", flush=True)
                        break
                if key not in remaining:
                    if key in done:
                        raise SystemExit(f"csv rows for {key} are not contiguous; cannot stream")
                    continue
                exporter.open_trajectory(wanted[key])
                current_key = key
                print(f"{exporter.meta['id']}: exporting", flush=True)
            cell = {
                "token": row[i_token],
                "abs_pos": int(row[i_abs]),
                "ranks": [int(row[i]) for i in i_ranks],
                "logprobs": [float(row[i]) for i in i_logprobs],
                "topk": [row[i] for i in i_topk],
            }
            exporter.add_row(int(row[i_step]), int(row[i_pos]), int(row[i_layer]), cell)

        if current_key is not None:
            exporter.close_trajectory()
            done.add(current_key)
            remaining.discard(current_key)

    never_seen = {k: p for k, p in wanted.items() if k not in done}
    for _key, path in sorted(never_seen.items()):
        print(f"! {path.stem}: no rows in the CSV; emitting lens-less step files", flush=True)
        exporter.open_trajectory(path)
        exporter.close_trajectory()

    elapsed = time.time() - started
    print(
        f"scanned {n_rows:,} rows in {elapsed / 60:.1f} min; "
        f"{len(done)}/{len(wanted)} trajectories had CSV data -> {args.out_dir}",
        flush=True,
    )


def _self_test() -> None:
    """End-to-end on a synthetic CSV + trajectory. Stdlib only: python ... --self-test."""
    import io
    import tempfile

    dir_lookup = {" left": DIR_CODES["LEFT"], " up": DIR_CODES["UP"]}
    n_layers, n_pos = 2, 3

    def make_trajectory(n_analysis: int) -> dict:
        out = [{"token": "<|channel|>", "token_id": 1, "token_groups": ["output", "template"]}]
        out += [
            {
                "token": f"Ġt{i}",
                "token_id": 100 + i,
                "token_groups": ["output", "analysis"],
                "probabilities": {" a": 1.0},
            }
            for i in range(n_analysis)
        ]
        step = {
            "step_id": 0,
            "grid_state": ["  0  1 ", "0  A  G "],
            "agent_action": "LEFT",
            "output_text": "x",
            "grid_state_tokens": [{}] * 4,
            "output_tokens": out,
        }
        step2 = dict(step, step_id=1, agent_action="UP")
        return {
            "grid_params": {"grid_width": 2, "grid_height": 2, "legend": {}},
            "model_params": {"model_id": "test"},
            "prompt": {"prompt_prefix_tokens": [{}] * 3, "prompt_suffix_tokens": [{}] * 2},
            "steps": [step, step2],
        }

    # position math: output_start = 3 + 4 + 2 = 9, analysis tokens start at output index 1
    traj = make_trajectory(n_pos)
    positions = reasoning_token_positions(traj, traj["steps"][0])
    assert positions == [(0, 10, "Ġt0"), (1, 11, "Ġt1"), (2, 12, "Ġt2")], positions

    def cell(pos: int, layer: int) -> dict:
        # top-k: one " left" everywhere, plus " up" only at layer 1, rest are junk
        top = [" left"] + ([" up"] if layer == 1 else [" x"]) + [f"j{i}" for i in range(TOP_K - 2)]
        return {
            "token": f"Ġt{pos}",
            "abs_pos": 10 + pos,
            "ranks": [10, 0, 5, 200000],
            "logprobs": [-1.0, -2.0, -3.0, -4.0],
            "topk": top,
        }

    cells = {(p, ll): cell(p, ll) for p in range(n_pos) for ll in range(n_layers)}
    del cells[(1, 1)]  # a hole
    meta = {"size": "11", "comp": "0.0", "run": "0", "id": "size11_comp0.0_run0", "source": "x.json"}
    payload = build_step_payload(traj, 0, cells, dir_lookup, meta)
    lens = payload["lens"]
    assert lens["n_pos"] == n_pos and lens["n_layers"] == n_layers
    assert len(lens["topk"]) == n_pos * n_layers * TOP_K
    assert lens["missing_cells"] == 1
    hole = 1 * n_layers + 1
    assert lens["topk"][hole * TOP_K] == -1 and lens["rank"]["UP"][hole] is None
    # density must equal the count of direction-classed vocab entries in each cell
    for i in range(n_pos * n_layers):
        for action, code in DIR_CODES.items():
            expect = sum(
                1
                for k in range(TOP_K)
                if lens["topk"][i * TOP_K + k] >= 0 and lens["vocab_dir"][lens["topk"][i * TOP_K + k]] == code
            )
            assert lens["density"][action][i] == expect, (action, i)
    assert lens["density"]["LEFT"][0] == 1 and lens["density"]["UP"][0] == 0
    assert lens["density"]["UP"][1] == 1  # pos 0, layer 1
    assert lens["reasoning_positions"] == [1, 2, 3]  # output index, skipping <|channel|>
    assert payload["warnings"] == ["1 missing (position, layer) cells"], payload["warnings"]

    # no CSV rows -> lens null, file still describes the step
    empty = build_step_payload(traj, 1, {}, dir_lookup, meta)
    assert empty["lens"] is None and empty["step"]["agent_action"] == "UP"

    # n_pos mismatch: trajectory has 2 analysis tokens, CSV claims 3
    short = make_trajectory(2)
    mismatched = build_step_payload(short, 0, cells, dir_lookup, meta)
    assert mismatched["lens"]["n_pos"] == 2
    assert any("mismatch" in w for w in mismatched["warnings"]), mismatched["warnings"]

    # csv parsing with embedded newlines/commas in top_k fields
    raise_field_size_limit()
    header = ",".join(REQUIRED_COLUMNS)
    body = ['11,0.0,0,0,0,10,Ġt0,0,LEFT,10,0,5,200000,-1,-2,-3,-4,"a,b","c\nd"' + ",x" * (TOP_K - 2)]
    rows = list(csv.reader(io.StringIO(header + "\n" + "\n".join(body) + "\n")))
    assert rows[1][REQUIRED_COLUMNS.index("top_1")] == "a,b"
    assert rows[1][REQUIRED_COLUMNS.index("top_2")] == "c\nd"

    assert parse_steps_spec("all", 3) == [0, 1, 2]
    assert parse_steps_spec("0,2", 5) == [0, 2]
    assert parse_steps_spec("1:3", 5) == [1, 2]
    assert group_key("11", "1.0", "07") == group_key("11", "1.00", "7")

    # write/read round trip
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        write_step_file(out, payload)
        write_manifest(out, traj, meta, [payload, empty], {"csv": "c", "direction_tokens": "d"})
        reloaded = json.loads((out / "step_000.json").read_text(encoding="utf-8"))
        assert reloaded["schema"] == SCHEMA
        assert reloaded["lens"]["vocab"][reloaded["lens"]["topk"][0]] == " left"
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert [s["has_lens"] for s in manifest["steps"]] == [True, False]
        assert manifest["layers"] == [0, 1]

    print("self-test ok")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        main()
