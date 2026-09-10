#!/usr/bin/env python3
"""Does LOUDNESS decode, or does POSITION IN THE REASONING CHAIN decode for it?

Every loudness figure so far bins on one axis at a time: ``analyze_probe_loudness.py`` gives
balanced accuracy per loudness decile, and ``plot_probe_loudness.py::fig_grid`` crosses loudness
against position *within a sentence* at 3x3. Neither separates loudness from where the token sits
in the chain as a whole, which is the standing confound -- loud tokens are not spread uniformly
along a chain, so a monotone loudness curve could be a position curve wearing a hat.

This draws the 10x10 that settles it, one per probe: x is the decile of the token's position in
its reasoning chain, y is the decile of its direction mass, and each cell is that probe's BALANCED
accuracy against the at-token local belief. If loudness were a proxy for position the grid would
vary only along x. It does not: for ``p1_mlp`` the first position decile alone runs .44 (quietest)
to .83 (loudest), and the loudness ordering holds inside every column.

MEASURED ON THE HELD-OUT 360 and on EVERY reasoning token of it -- the tree disjoint from every
probe's training set, with no selection standing between the loudness axis and the label. Both
inputs already exist, so this is a pure CSV read and needs no GPU:

``per_token_jlens_loudness.csv``      ``build_probe_loudness_heldout.py --mass-column jlens_mass_L15``
``per_token_logitlens_loudness.csv``  the same rows, ``--mass-column logitlens_mass_L15``

ONE RULER PER RUN, NEVER PER FIGURE. ``--ruler`` picks the source CSV, the output subfolder
(``{out}/{ruler}_ruler/``) and the text of every title and y-axis label, from one flag. That is
what lets both rounds keep identical filenames: the round binning on jlens loudness writes
``jlens_ruler/``, the logitlens round writes ``logitlens_ruler/``, and neither can overwrite the
other. A figure's FILENAME PREFIX is the lens that selected that probe's TRAINING tokens
(``jlens`` / ``logitlens`` / ``random``) and is NOT the ruler -- under ``--ruler jlens`` the file
``logitlens_p1_local_belief.png`` is the logitlens-selected probe binned by jlens loudness, which
is exactly the cross entry 49 asked for.

THE ROWSET TRAP. The source CSV is 3 x 87,221 rows: since entry 48 its three rowsets hold
IDENTICAL rows, every one of which carries EVERY probe's columns, so a rowset name selects which
probes a report reads and not which tokens. Read all three and every count triples. This filters
to one (``--rowset``, default ``p2``) and reports the survivor count.

The self-check that matters is printed per probe: pooling all 100 cells of a grid must reproduce
that probe's published held-out balanced accuracy (``PUBLISHED`` below, taken from
``build_sixteen_probe_report_page.py``). A mismatch means the rowset dedup or the label column is
wrong, which are the two ways this join fails silently, and the script exits non-zero.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# One definition of balanced accuracy, imported rather than restated so this figure and the
# tables it is read beside can never drift apart. (Mean per-class recall over the classes
# PRESENT in the cell -- a plain accuracy would move with the cell's class mix, and the loud
# cells are direction-word-heavy.)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_probe_loudness import bal_acc  # noqa: E402

# csv.DictReader everywhere, never pandas.read_csv: decoded tokens include "NA", empty strings,
# embedded commas and newlines, which pandas' NA handling silently corrupts.
csv.field_size_limit(10**9)

DEFAULT_SOURCE_DIR = Path("/workspace/reasoning_theatre/probe_loudness_heldout360_16probes")
DEFAULT_OUT = Path("/workspace/reasoning_theatre/360_held_out/loudness_x_reasoning_pos")

RULERS = {
    "jlens": "per_token_jlens_loudness.csv",
    "logitlens": "per_token_logitlens_loudness.csv",
}

# dataviz reference palette, as scripts/jlens_rank_analysis.py.
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#898781"
BLUE_SEQ = ["#cde2fb", "#0d366b"]
MASKED = "#e8e7e0"


class Arm:
    """One heatmap: a probe, the lens family that selected its training tokens, and its slot.

    ``stem`` is the short key ``build_probe_loudness_heldout.py`` gave that probe, so the CSV
    column for a model type is ``f"{stem}_{model_type}_pred"``. An arm with ``stem=None`` has no
    probe yet; it is reported as pending and skipped, which is how the three unbuilt slots cost
    a line of output rather than a crash.

    >>> Arm("logitlens", "p1", "ll1", "x").filename("mlp")
    'logitlens_p1_local_belief.png'
    >>> Arm("jlens", "p1-top20", "p1t20", "x").column("lr")
    'p1t20_lr'
    """

    def __init__(self, family: str, slot: str, stem: str | None, label: str, note: str = ""):
        self.family, self.slot, self.stem, self.label, self.note = family, slot, stem, label, note

    def key(self) -> str:
        return f"{self.family}_{self.slot}"

    def column(self, model_type: str) -> str | None:
        return None if self.stem is None else f"{self.stem}_{model_type}"

    def filename(self, model_type: str) -> str:
        suffix = "" if model_type == "mlp" else f"_{model_type}"
        return f"{self.family}_{self.slot}_local_belief{suffix}.png"


# The nine slots. Three have no probe yet and carry stem=None; filling one in is a single edit
# here once its column exists in the source CSV -- which needs eval_probe_per_token.py and then
# build_probe_loudness_heldout.py --extra-probes, both outside this script.
ARMS = [
    Arm("jlens", "p1", "p1", "P1 - jlens per-sentence loudest"),
    Arm("jlens", "p1-top20", "p1t20", "P1 top-20 - jlens per-sentence, thinned to 20/traj"),
    Arm("jlens", "p2", "p2", "P2 - jlens global top-20"),
    Arm("logitlens", "p1", "ll1", "P1 - logitlens per-sentence loudest"),
    Arm("logitlens", "p1-top20", None, "P1 top-20 - logitlens per-sentence, thinned", "no such probe trained"),
    Arm("logitlens", "p2", "ll2", "P2 - logitlens global top-20"),
    Arm("random", "p1", None, "P1 - random per sentence", "probe still computing"),
    Arm("random", "p1-top20", None, "P1 top-20 - random per sentence, thinned", "probe still computing"),
    Arm("random", "p2", "randb", "P2 - random selection (seeded uniform, replayed picks)"),
]

# Published held-out balanced accuracy per probe column, from the registry in
# build_sixteen_probe_report_page.py. Pooling a grid must reproduce these.
PUBLISHED = {
    "p1_lr": 0.4634,
    "p1_mlp": 0.5276,
    "p1t20_lr": 0.4572,
    "p1t20_mlp": 0.4989,
    "p2_lr": 0.4869,
    "p2_mlp": 0.5243,
    "ll1_lr": 0.4861,
    "ll1_mlp": 0.5483,
    "ll2_lr": 0.5018,
    "ll2_mlp": 0.5498,
    "randb_lr": 0.5274,
    "randb_mlp": 0.5886,
}


def width_bins(values: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Equal-width bins of a [0, 1] variable: a bin index per row, and the k+1 edges.

    ``reasoning_frac`` is a position divided by a chain length, so equal-width bins read as
    "the first tenth of the chain" and are comparable across chains of different lengths. On
    the held-out 360 they come out near-flat anyway (8,630-8,887 rows per bin), so this and
    ``quantile_bins`` give nearly the same picture; this is the interpretable spelling.

    >>> idx, edges = width_bins(np.array([0.0, 0.05, 0.5, 0.99, 1.0]), 10)
    >>> idx.tolist()
    [0, 0, 5, 9, 9]
    >>> edges[:3].round(2).tolist()
    [0.0, 0.1, 0.2]
    """
    edges = np.linspace(0.0, 1.0, k + 1)
    return np.clip((values * k).astype(int), 0, k - 1), edges


def quantile_bins(values: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Equal-count bins: a bin index per row, and the k+1 edges.

    Required for the loudness axis, which is heavy-tailed and floors at ``NO_MATCH_LOGPROB``
    -- equal-width bins there would put almost every token in one row. ``searchsorted`` on the
    interior edges assigns a tie to the higher bin, matching ``pd.qcut``'s right-closed
    intervals, and collapsing duplicate edges is the ``duplicates="drop"`` of ``qbin()``.

    >>> idx, edges = quantile_bins(np.arange(100.0), 10)
    >>> idx[:3].tolist(), idx[-3:].tolist()
    ([0, 0, 0], [9, 9, 9])
    >>> int(np.bincount(idx, minlength=10).min())
    10
    """
    edges = np.quantile(values, np.linspace(0.0, 1.0, k + 1))
    interior = np.unique(edges[1:-1])
    if len(interior) < k - 1:
        print(
            f"  WARNING: loudness bins collapsed to {len(interior) + 1} distinct bins "
            "(tied edges); cells will be uneven.",
            flush=True,
        )
    return np.searchsorted(interior, values, side="right"), edges


def read_rows(path: Path, rowset: str, columns: list[str]) -> dict[str, np.ndarray]:
    """The deduplicated token table: one row per reasoning token of the held-out 360.

    Keeps only ``rowset`` -- the three hold identical rows, each carrying every probe's columns
    -- and drops the rows whose rollout emitted no valid action, so a blank truth is never
    counted as anything.
    """
    name: list[str] = []
    frac: list[float] = []
    mass: list[float] = []
    truth: list[str] = []
    preds: dict[str, list[str]] = {c: [] for c in columns}
    seen: set[str] = set()
    blank = 0
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        have = set(reader.fieldnames or [])
        missing = [f"{c}_pred" for c in columns if f"{c}_pred" not in have]
        if missing:
            raise SystemExit(
                f"{path} is missing {missing}. Was build_probe_loudness_heldout.py run with "
                "the --extra-probes those arms need?"
            )
        for r in reader:
            seen.add(r["rowset"])
            if r["rowset"] != rowset:
                continue
            if not r["label_local"]:
                blank += 1
                continue
            name.append(r["name"])
            frac.append(float(r["reasoning_frac"]))
            mass.append(float(r["dir_logmass"]))
            truth.append(r["label_local"])
            for c in columns:
                preds[c].append(r[f"{c}_pred"])
    if rowset not in seen:
        raise SystemExit(f"--rowset {rowset!r} not in {path}; found {sorted(seen)}")
    print(f"  rowsets in file: {sorted(seen)}; kept {rowset!r}", flush=True)
    print(
        f"  {len(name):,} tokens over {len(set(name))} trajectories ({blank} dropped for a blank local belief)",
        flush=True,
    )
    out: dict[str, np.ndarray] = {
        "name": np.array(name),
        "frac": np.array(frac),
        "mass": np.array(mass),
        "truth": np.array(truth),
    }
    for c in columns:
        out[c] = np.array(preds[c])
    return out


def grid(data: dict[str, np.ndarray], col: str, pos: np.ndarray, loud: np.ndarray, k: int):
    """Per-cell balanced accuracy, counts and cell means, as five (k, k) arrays.

    Row index is the loudness decile (0 quietest) and column index the position decile, the
    orientation ``imshow(origin="lower")`` then draws with loud at the top.
    """
    acc = np.full((k, k), np.nan)
    n = np.zeros((k, k), dtype=int)
    ntraj = np.zeros((k, k), dtype=int)
    logmass = np.full((k, k), np.nan)
    posmean = np.full((k, k), np.nan)
    truth, pred, names = data["truth"], data[col], data["name"]
    for i in range(k):
        for j in range(k):
            m = (loud == i) & (pos == j)
            n[i, j] = int(m.sum())
            if not n[i, j]:
                continue
            acc[i, j] = bal_acc(truth[m], pred[m])
            ntraj[i, j] = len(np.unique(names[m]))
            logmass[i, j] = float(data["mass"][m].mean())
            posmean[i, j] = float(data["frac"][m].mean())
    return acc, n, ntraj, logmass, posmean


def draw(acc, n, arm: Arm, ruler: str, pooled: float, args, out: Path) -> Path:
    """One 10x10 heatmap. Cells below ``--min-n`` are left undrawn but stay in cells.csv."""
    k = acc.shape[0]
    shown = np.where(n >= args.min_n, acc, np.nan)
    fig, ax = plt.subplots(figsize=(args.width, args.height), facecolor=SURFACE)
    ax.set_facecolor(MASKED)
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("blue_seq", BLUE_SEQ)
    im = ax.imshow(shown, origin="lower", cmap=cmap, vmin=args.vmin, vmax=args.vmax, aspect="auto")
    mid = (args.vmin + args.vmax) / 2
    for i in range(k):
        for j in range(k):
            if np.isnan(shown[i, j]):
                ax.text(
                    j,
                    i,
                    f"n<{args.min_n}" if n[i, j] else "-",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=MUTED,
                )
                continue
            txt = f"{shown[i, j]:.2f}" + (f"\n{n[i, j]:,}" if args.annotate_n else "")
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=7 if args.annotate_n else 8,
                linespacing=1.15,
                color="#ffffff" if shown[i, j] > mid else INK,
            )
    ax.set_xticks(
        range(k),
        [f"{int(100 * b / k)}-{int(100 * (b + 1) / k)}%" for b in range(k)],
        rotation=45,
        ha="right",
    )
    ax.set_yticks(range(k), [str(b) for b in range(k)])
    ax.set_xlabel("position in the reasoning chain  (decile of reasoning_frac, 0% = chain start)")
    ax.set_ylabel(f"{ruler} direction-mass decile  (0 = quietest, {k - 1} = loudest)")
    ax.set_title(
        f"{arm.label}, {args.model_type}\n"
        "balanced accuracy vs the at-token local belief   |   "
        f"loudness ruler: {ruler}   |   pooled {pooled:.4f}, n={int(n.sum()):,}",
        color=INK,
        fontsize=10,
        loc="left",
    )
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.set_label("balanced accuracy (chance = .25)", color=MUTED, fontsize=8)
    fig.tight_layout()
    path = out / arm.filename(args.model_type)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {path}", flush=True)
    return path


def check_ruler(out: Path, ruler: str) -> None:
    """Refuse to write a second ruler's figures into a folder that already holds one.

    The per-ruler subfolder makes a collision impossible under normal use; this catches an
    ``--out`` pointed straight at an existing folder, which would otherwise leave two rulers'
    figures under identical filenames and no way to tell them apart.
    """
    cfg = out / "run_config.json"
    if not cfg.exists():
        return
    prev = json.loads(cfg.read_text()).get("ruler")
    if prev and prev != ruler:
        raise SystemExit(
            f"{out} already holds figures built with the {prev!r} ruler; refusing to mix two "
            "rulers in one folder. Use the default --out so each lands in its own "
            "{ruler}_ruler/, or give this run an --out of its own."
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--ruler",
        default="jlens",
        choices=sorted(RULERS),
        help="which lens's layer-15 direction mass is the loudness axis. Picks the source CSV "
        "AND the output subfolder AND the wording of every title.",
    )
    ap.add_argument(
        "--per-token", type=Path, default=None, help="source CSV; defaults to the --ruler one under --source-dir."
    )
    ap.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="figures land in {out}/{ruler}_ruler/.")
    ap.add_argument("--model-type", default="mlp", choices=["mlp", "lr"])
    ap.add_argument("--probes", default="all", help="comma-separated <family>_<slot> subset, or 'all'.")
    ap.add_argument("--rowset", default="p2", help="which rowset to keep; the three hold identical rows.")
    ap.add_argument("--deciles", type=int, default=10)
    ap.add_argument("--pos-bins", default="width", choices=["width", "quantile"])
    ap.add_argument("--min-n", type=int, default=120, help="cells below this go into cells.csv but are not drawn.")
    ap.add_argument("--vmin", type=float, default=0.25, help="chance for four actions; fixed so the panels compare.")
    ap.add_argument("--vmax", type=float, default=0.90)
    ap.add_argument("--width", type=float, default=11.0)
    ap.add_argument("--height", type=float, default=8.0)
    ap.add_argument("--annotate-n", dest="annotate_n", action="store_true", default=True)
    ap.add_argument("--no-annotate-n", dest="annotate_n", action="store_false")
    ap.add_argument("--tol", type=float, default=5e-4, help="max |pooled - published| before the self-check fails.")
    args = ap.parse_args()

    src = args.per_token or (args.source_dir / RULERS[args.ruler])
    ruler = args.ruler
    out = args.out / f"{ruler}_ruler"
    out.mkdir(parents=True, exist_ok=True)
    check_ruler(out, ruler)
    print(f"loudness ruler: {ruler}  ({ruler}_mass_L15)", flush=True)

    wanted = ARMS
    if args.probes != "all":
        keys = {t.strip() for t in args.probes.split(",") if t.strip()}
        unknown = keys - {a.key() for a in ARMS}
        if unknown:
            raise SystemExit(f"unknown --probes {sorted(unknown)}; choose from {[a.key() for a in ARMS]}")
        wanted = [a for a in ARMS if a.key() in keys]

    pending = [a for a in wanted if a.column(args.model_type) is None]
    ready = [a for a in wanted if a.column(args.model_type) is not None]
    cols = [c for c in (a.column(args.model_type) for a in ready) if c is not None]

    print(f"reading {src}", flush=True)
    data = read_rows(src, args.rowset, cols)

    k = args.deciles
    pos, pos_edges = (width_bins if args.pos_bins == "width" else quantile_bins)(data["frac"], k)
    loud, mass_edges = quantile_bins(data["mass"], k)
    print(f"  position bins ({args.pos_bins}): {np.bincount(pos, minlength=k).tolist()}", flush=True)
    print(f"  loudness deciles: {np.bincount(loud, minlength=k).tolist()}", flush=True)

    failures: list[tuple[str, float, float]] = []
    with open(out / "cells.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "probe",
                "family",
                "slot",
                "model_type",
                "ruler",
                "pos_decile",
                "mass_decile",
                "pos_lo",
                "pos_hi",
                "mass_lo",
                "mass_hi",
                "n",
                "n_traj",
                "bal_acc",
                "mean_logmass",
                "mean_reasoning_frac",
            ]
        )
        for arm in ready:
            col = arm.column(args.model_type)
            assert col is not None
            acc, n, ntraj, logmass, posmean = grid(data, col, pos, loud, k)
            pooled = bal_acc(data["truth"], data[col])
            for i in range(k):
                for j in range(k):
                    w.writerow(
                        [
                            col,
                            arm.family,
                            arm.slot,
                            args.model_type,
                            ruler,
                            j,
                            i,
                            f"{pos_edges[j]:.6f}",
                            f"{pos_edges[j + 1]:.6f}",
                            f"{mass_edges[i]:.6f}",
                            f"{mass_edges[i + 1]:.6f}",
                            n[i, j],
                            ntraj[i, j],
                            "" if np.isnan(acc[i, j]) else f"{acc[i, j]:.6f}",
                            "" if np.isnan(logmass[i, j]) else f"{logmass[i, j]:.6f}",
                            "" if np.isnan(posmean[i, j]) else f"{posmean[i, j]:.6f}",
                        ]
                    )
            draw(acc, n, arm, ruler, pooled, args, out)
            ref = PUBLISHED.get(col)
            flag = ""
            if ref is not None and abs(pooled - ref) > args.tol:
                flag = f"  *** MISMATCH vs published {ref:.4f}"
                failures.append((col, pooled, ref))
            print(
                f"    pooled bal_acc {pooled:.4f}"
                + (f" (published {ref:.4f})" if ref is not None else " (no published reference)")
                + f", smallest cell n={int(n.min()):,}, drawn {int((n >= args.min_n).sum())}/{k * k}{flag}",
                flush=True,
            )

    (out / "run_config.json").write_text(
        json.dumps(
            {
                "ruler": ruler,
                "mass_column": f"{ruler}_mass_L15",
                "source": str(src),
                "rowset": args.rowset,
                "model_type": args.model_type,
                "deciles": k,
                "pos_bins": args.pos_bins,
                "min_n": args.min_n,
                "vmin": args.vmin,
                "vmax": args.vmax,
                "n_tokens": int(len(data["name"])),
                "n_trajectories": int(len(set(data["name"].tolist()))),
                "position_edges": [float(x) for x in pos_edges],
                "loudness_edges": [float(x) for x in mass_edges],
                "figures": [a.filename(args.model_type) for a in ready],
                "pending": {a.key(): a.note for a in pending},
            },
            indent=2,
        )
        + "\n"
    )

    print(f"\nwrote {len(ready)} figure(s) + cells.csv + run_config.json -> {out}", flush=True)
    for a in pending:
        print(f"  pending, no figure: {a.key()} ({a.note})", flush=True)
    if failures:
        print("\n  *** pooled balanced accuracy does not match the published value:", flush=True)
        for col, got, ref in failures:
            print(f"      {col}: {got:.4f} vs {ref:.4f}", flush=True)
        print("      the rowset dedup or the label column is wrong.", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
