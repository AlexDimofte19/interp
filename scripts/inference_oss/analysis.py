"""Aggregate and plot reasoning-theatre statistics produced by ``run_inference.py``.

``run_inference.py`` writes one results JSON per trajectory, each with a ``summary`` and
a list of ``steps`` (every step holding per-sentence ``sentence_evals`` with
``answer_prob`` and the commitment indices ``first_correct_sentence_idx`` /
``convinced_sentence_idx``). This script walks the ``<input-folder>/size*/*.json``
layout, pools those records across the dataset, and emits thirteen figures:

1. Mean fraction-of-sentences to first-correct vs. to convinced (whole dataset).
2. Sentence accuracy per size (bar).
3. Mean answer probability before vs. after the convinced point
   (steps convinced at sentence 0 / never convinced are skipped).
4. Mean answer probability before vs. after the first-correct point
   (steps first-correct at sentence 0 / never correct are skipped).
5. Percent of steps first-correct with no reasoning (sentence 0), per size (bar).
6. Percent of steps convinced with no reasoning (sentence 0), per size (bar).
7. Heatmap of mean convinced fraction over (grid complexity x size).
8. Heatmap of mean first-correct fraction over (grid complexity x size).
9. Heatmap of percent convinced with no reasoning over (grid complexity x size).
10. Heatmap of percent first-correct with no reasoning over (grid complexity x size).
11. Percent of steps convinced / first-correct *before the full reasoning chain* (bar,
    whole dataset, SD bars).
12. Heatmap of percent convinced before the full reasoning chain (grid complexity x size).
13. Heatmap of percent first-correct before the full reasoning chain (grid complexity x size).

The heatmaps need each step's grid complexity (density), which lives in the source
trajectory files (``grid_params.grid_complexity``) rather than the results JSON. Pass
``--trajectory-folder`` to read it from there, matched to results by filename stem; if
omitted (or a trajectory is missing) the complexity falls back to the ``comp<X>`` marker
in the filename.

Run on the host with the results, e.g.:
    python scripts/inference_oss/analysis.py \
        --input-folder scripts/inference_oss/inference_results \
        --trajectory-folder data/reveng/trajectories_train_single_step \
        --output-dir scripts/inference_oss/analysis_plots
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless server: render to files, never to a display.
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

DEFAULT_INPUT = str(Path(__file__).with_name("inference_results"))
DEFAULT_OUTPUT = str(Path(__file__).with_name("analysis_plots"))

SIZE_RE = re.compile(r"size(\d+)")
COMP_RE = re.compile(r"comp(\d+(?:\.\d+)?)")


def parse_size(path: Path) -> int | None:
    """Maze size as an int, read from a ``size<N>`` segment of the path.

    Prefers a parent directory named ``size*`` (the documented layout), falling back to
    the filename. Returns ``None`` if no size marker is present.
    """
    for part in (*path.parts[::-1], path.stem):
        m = SIZE_RE.search(part)
        if m:
            return int(m.group(1))
    return None


def parse_complexity(path: Path) -> float | None:
    """Grid complexity (density) read from a ``comp<X>`` segment of the path.

    A fallback for when no trajectory folder is given: the trajectory filenames encode
    the complexity (e.g. ``..._size11_comp1.0_987.json``). Returns ``None`` if absent.
    """
    for part in (*path.parts[::-1], path.stem):
        m = COMP_RE.search(part)
        if m:
            return round(float(m.group(1)), 1)
    return None


def load_complexity_map(trajectory_folder: str) -> dict[str, float]:
    """Map each trajectory file stem to its ``grid_params.grid_complexity``.

    Results files written by ``run_inference.py`` keep the trajectory's filename stem, so
    this stem is the join key back to the source grid complexity.
    """
    root = Path(trajectory_folder)
    mapping: dict[str, float] = {}
    for fp in sorted(root.rglob("*.json")):
        try:
            with open(fp) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        grid_params = data.get("grid_params") if isinstance(data, dict) else None
        if grid_params and "grid_complexity" in grid_params:
            mapping[fp.stem] = round(float(grid_params["grid_complexity"]), 1)
    return mapping


def load_steps(input_folder: str, complexity_map: dict[str, float] | None = None) -> list[dict]:
    """Flatten every results JSON under ``input_folder`` into per-step records.

    Each record carries the step's commitment indices, fractions, and a light view of
    its ``sentence_evals`` (sentence_idx, correct, answer_prob), plus the parsed size and
    grid complexity. Complexity comes from ``complexity_map`` (built from the trajectory
    files, keyed by filename stem) when available, otherwise from the ``comp<X>`` marker
    in the results filename.
    """
    root = Path(input_folder)
    files = sorted(root.rglob("*.json"))
    if not files:
        raise ValueError(f"No JSON files found under {input_folder!r}")

    records: list[dict] = []
    for fp in files:
        with open(fp) as f:
            data = json.load(f)
        if not (isinstance(data, dict) and "steps" in data):
            continue  # not a run_inference results file
        size = parse_size(fp)
        complexity = (complexity_map or {}).get(fp.stem)
        if complexity is None:
            complexity = parse_complexity(fp)
        for step in data["steps"]:
            evals = [
                {
                    "sentence_idx": e["sentence_idx"],
                    "correct": e["correct"],
                    "answer_prob": e.get("answer_prob"),
                }
                for e in step["sentence_evals"]
            ]
            records.append({
                "size": size,
                "complexity": complexity,
                "first_correct_idx": step["first_correct_sentence_idx"],
                "convinced_idx": step["convinced_sentence_idx"],
                "first_correct_fraction": step["first_correct_fraction"],
                "convinced_fraction": step["convinced_fraction"],
                "n_sentences": step["n_reasoning_sentences"],
                "evals": evals,
            })
    if not records:
        raise ValueError(f"No usable results files under {input_folder!r}")
    return records


def _mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return float(np.mean(present)) if present else None


def _mean_sd(values: list[float | None]) -> tuple[float | None, float | None]:
    """Mean and sample standard deviation (``ddof=1``), ignoring ``None``s.

    Works uniformly for continuous values (fractions, probabilities) and 0/1 indicators
    (accuracy, no-reasoning share), where the SD equals the proportion's
    ``sqrt(p(1-p))`` (up to the ``ddof=1`` correction). SD is ``0.0`` for a single
    observation and ``None`` for none.
    """
    arr = np.array([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return None, None
    sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return float(arr.mean()), sd


def _save(fig, output_dir: Path, name: str) -> None:
    out = output_dir / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_fraction_comparison(records: list[dict], output_dir: Path) -> None:
    """Graph 1: mean fraction-of-sentences to first-correct vs. to convinced."""
    first, first_sd = _mean_sd([r["first_correct_fraction"] for r in records])
    convinced, convinced_sd = _mean_sd([r["convinced_fraction"] for r in records])
    labels = ["First correct", "Convinced"]
    values = [first or 0.0, convinced or 0.0]
    errors = [first_sd or 0.0, convinced_sd or 0.0]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(x=labels, y=values, ax=ax, palette="muted")
    ax.errorbar(range(len(values)), values, yerr=errors, fmt="none", ecolor="black", capsize=4)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean fraction of reasoning sentences")
    ax.set_title("Reasoning fraction until first-correct vs. convinced")
    for i, (v, e) in enumerate(zip(values, errors)):
        ax.text(i, v + e + 0.02, f"{v:.2f}", ha="center")
    _save(fig, output_dir, "1_fraction_first_correct_vs_convinced.png")


def plot_accuracy_per_size(records: list[dict], output_dir: Path) -> None:
    """Graph 2: pooled sentence accuracy (correct evals / total evals) per size."""
    by_size: dict[int, list[bool]] = {}
    for r in records:
        if r["size"] is None:
            continue
        by_size.setdefault(r["size"], []).extend(e["correct"] for e in r["evals"])

    sizes = sorted(by_size)
    stats = [_mean_sd([float(b) for b in by_size[s]]) for s in sizes]
    acc = [m or 0.0 for m, _ in stats]
    errors = [se or 0.0 for _, se in stats]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=[str(s) for s in sizes], y=acc, ax=ax, palette="crest")
    ax.errorbar(range(len(acc)), acc, yerr=errors, fmt="none", ecolor="black", capsize=4)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Maze size")
    ax.set_ylabel("Sentence accuracy")
    ax.set_title("Average sentence accuracy per size")
    for i, (v, e) in enumerate(zip(acc, errors)):
        ax.text(i, v + e + 0.02, f"{v:.2f}", ha="center")
    _save(fig, output_dir, "2_sentence_accuracy_per_size.png")


def _before_after_probs(records: list[dict], idx_key: str) -> tuple[list[float], list[float]]:
    """Pooled answer probs before vs. at/after the ``idx_key`` commitment point.

    Steps with the commitment at sentence 0 (no reasoning) or never reached (``None``)
    are skipped, per the request.
    """
    before: list[float] = []
    after: list[float] = []
    for r in records:
        cut = r[idx_key]
        if cut is None or cut == 0:
            continue
        for e in r["evals"]:
            if e["answer_prob"] is None:
                continue
            (before if e["sentence_idx"] < cut else after).append(e["answer_prob"])
    return before, after


def plot_prob_around_commitment(records: list[dict], output_dir: Path, idx_key: str, label: str, name: str) -> None:
    """Graphs 3 & 4: mean answer probability before vs. after a commitment point."""
    before, after = _before_after_probs(records, idx_key)
    before_m, before_sd = _mean_sd(before)
    after_m, after_sd = _mean_sd(after)
    values = [before_m or 0.0, after_m or 0.0]
    errors = [before_sd or 0.0, after_sd or 0.0]
    counts = [len(before), len(after)]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(x=[f"Before {label}", f"After {label}"], y=values, ax=ax, palette="flare")
    ax.errorbar(range(len(values)), values, yerr=errors, fmt="none", ecolor="black", capsize=4)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean answer probability")
    ax.set_title(f"Answer probability before vs. after {label}")
    for i, (v, e, n) in enumerate(zip(values, errors, counts)):
        ax.text(i, v + e + 0.02, f"{v:.2f}\n(n={n})", ha="center")
    _save(fig, output_dir, name)


def plot_no_reasoning_share(records: list[dict], output_dir: Path, idx_key: str, label: str, name: str) -> None:
    """Graphs 5 & 6: percent of steps that hit the commitment at sentence 0, per size."""
    flags_by_size: dict[int, list[float]] = {}
    for r in records:
        if r["size"] is None:
            continue
        flags_by_size.setdefault(r["size"], []).append(1.0 if r[idx_key] == 0 else 0.0)

    sizes = sorted(flags_by_size)
    stats = [_mean_sd(flags_by_size[s]) for s in sizes]
    pct = [100.0 * (m or 0.0) for m, _ in stats]
    errors = [100.0 * (se or 0.0) for _, se in stats]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=[str(s) for s in sizes], y=pct, ax=ax, palette="mako")
    ax.errorbar(range(len(pct)), pct, yerr=errors, fmt="none", ecolor="black", capsize=4)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Maze size")
    ax.set_ylabel(f"% of steps {label} with no reasoning")
    ax.set_title(f"{label.capitalize()} with no reasoning (sentence 0) per size")
    for i, (v, e) in enumerate(zip(pct, errors)):
        ax.text(i, v + e + 1, f"{v:.0f}%", ha="center")
    _save(fig, output_dir, name)


def _heatmap_grid(records: list[dict], cell_fn) -> tuple[list[int], list[float], np.ndarray, np.ndarray]:
    """Bucket records into (complexity, size) cells and reduce each with ``cell_fn``.

    ``cell_fn`` takes the list of records in a cell and returns a ``(value, sd)`` pair
    (either may be ``None``). Returns the sorted sizes (columns), sorted complexities
    (rows), and the value and standard-deviation matrices, with ``np.nan`` for empty cells.
    """
    cells: dict[tuple[float, int], list[dict]] = {}
    for r in records:
        if r["size"] is None or r["complexity"] is None:
            continue
        cells.setdefault((r["complexity"], r["size"]), []).append(r)

    sizes = sorted({s for (_, s) in cells})
    comps = sorted({c for (c, _) in cells})
    value_m = np.full((len(comps), len(sizes)), np.nan)
    sd_m = np.full((len(comps), len(sizes)), np.nan)
    for i, c in enumerate(comps):
        for j, s in enumerate(sizes):
            recs = cells.get((c, s))
            if recs:
                v, sd = cell_fn(recs)
                if v is not None:
                    value_m[i, j] = v
                    if sd is not None:
                        sd_m[i, j] = sd
    return sizes, comps, value_m, sd_m


def plot_heatmap(records: list[dict], output_dir: Path, cell_fn, title: str, cbar_label: str, name: str, precision: int, vmin: float, vmax: float) -> None:
    """Graphs 7-10: a (grid complexity x size) heatmap of a per-cell statistic.

    Each cell is annotated ``value`` over ``±SD`` (sample standard deviation), so the
    spread rides along with the colour-encoded value.
    """
    sizes, comps, value_m, sd_m = _heatmap_grid(records, cell_fn)
    if not sizes or not comps:
        print(f"  skip {name}: no records with both a size and a complexity")
        return

    annot = np.full(value_m.shape, "", dtype=object)
    for i in range(value_m.shape[0]):
        for j in range(value_m.shape[1]):
            if np.isnan(value_m[i, j]):
                continue
            cell = f"{value_m[i, j]:.{precision}f}"
            if not np.isnan(sd_m[i, j]):
                cell += f"\n±{sd_m[i, j]:.{precision}f}"
            annot[i, j] = cell

    fig, ax = plt.subplots(figsize=(1.1 * len(sizes) + 3, 0.9 * len(comps) + 2))
    sns.heatmap(
        value_m,
        ax=ax,
        annot=annot,
        fmt="",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        mask=np.isnan(value_m),  # leave empty cells blank instead of printing "nan"
        xticklabels=[str(s) for s in sizes],
        yticklabels=[f"{c:.1f}" for c in comps],
        cbar_kws={"label": cbar_label},
    )
    ax.set_xlabel("Maze size")
    ax.set_ylabel("Grid complexity (density)")
    ax.set_title(title)
    ax.invert_yaxis()  # lowest complexity at the bottom of the y axis
    _save(fig, output_dir, name)


def _before_full_flag(record: dict, idx_key: str) -> float:
    """1.0 if the step committed *before* the full reasoning chain, else 0.0.

    The last cutoff (index ``n_sentences - 1``) is the full reasoning, so a commitment
    index that is present and strictly below it means the model was already committed
    without needing the whole chain. Never-committed (``None``) counts as 0.0.
    """
    idx = record[idx_key]
    if idx is None:
        return 0.0
    return 1.0 if idx < record["n_sentences"] - 1 else 0.0


def plot_committed_before_full(records: list[dict], output_dir: Path) -> None:
    """Graph 11: percent of steps committed before the full reasoning chain (dataset-wide)."""
    fc_m, fc_sd = _mean_sd([_before_full_flag(r, "first_correct_idx") for r in records])
    cv_m, cv_sd = _mean_sd([_before_full_flag(r, "convinced_idx") for r in records])
    labels = ["First correct", "Convinced"]
    values = [100.0 * (fc_m or 0.0), 100.0 * (cv_m or 0.0)]
    errors = [100.0 * (fc_sd or 0.0), 100.0 * (cv_sd or 0.0)]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(x=labels, y=values, ax=ax, palette="muted")
    ax.errorbar(range(len(values)), values, yerr=errors, fmt="none", ecolor="black", capsize=4)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of steps committed before full reasoning")
    ax.set_title("Committed before the full reasoning chain (whole dataset)")
    for i, (v, e) in enumerate(zip(values, errors)):
        ax.text(i, min(v + e + 1, 98), f"{v:.0f}%", ha="center")
    _save(fig, output_dir, "11_committed_before_full_reasoning.png")


def _cell_pct_before_full(recs: list[dict], key: str) -> tuple[float | None, float | None]:
    mean, sd = _mean_sd([_before_full_flag(r, key) for r in recs])
    return (
        mean * 100.0 if mean is not None else None,
        sd * 100.0 if sd is not None else None,
    )


def _cell_mean_fraction(recs: list[dict], key: str) -> tuple[float | None, float | None]:
    return _mean_sd([r[key] for r in recs])


def _cell_pct_idx_zero(recs: list[dict], key: str) -> tuple[float | None, float | None]:
    mean, sd = _mean_sd([1.0 if r[key] == 0 else 0.0 for r in recs])
    return (
        mean * 100.0 if mean is not None else None,
        sd * 100.0 if sd is not None else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-folder", default=DEFAULT_INPUT, help="Folder with <size*>/*.json results from run_inference.py.")
    parser.add_argument("--trajectory-folder", default=None, help="Folder with the source trajectory JSONs (for grid_complexity); falls back to the comp<X> filename marker if omitted.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT, help="Directory to write the PNG figures into.")
    args = parser.parse_args()

    sns.set_theme(style="whitegrid")

    complexity_map: dict[str, float] | None = None
    if args.trajectory_folder:
        complexity_map = load_complexity_map(args.trajectory_folder)
        print(f"Read grid complexity for {len(complexity_map)} trajectories from {args.trajectory_folder}")

    records = load_steps(args.input_folder, complexity_map)
    n_sized = sum(1 for r in records if r["size"] is not None)
    n_comp = sum(1 for r in records if r["complexity"] is not None)
    print(f"Loaded {len(records)} steps ({n_sized} with a parseable size, {n_comp} with a complexity) from {args.input_folder}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_fraction_comparison(records, output_dir)
    plot_accuracy_per_size(records, output_dir)
    plot_prob_around_commitment(records, output_dir, "convinced_idx", "convinced", "3_answer_prob_around_convinced.png")
    plot_prob_around_commitment(records, output_dir, "first_correct_idx", "first correct", "4_answer_prob_around_first_correct.png")
    plot_no_reasoning_share(records, output_dir, "first_correct_idx", "first correct", "5_first_correct_no_reasoning_per_size.png")
    plot_no_reasoning_share(records, output_dir, "convinced_idx", "convinced", "6_convinced_no_reasoning_per_size.png")
    plot_heatmap(
        records, output_dir, lambda recs: _cell_mean_fraction(recs, "convinced_fraction"),
        "Mean convinced fraction by complexity and size", "Mean convinced fraction",
        "7_convinced_fraction_heatmap.png", precision=2, vmin=0.0, vmax=1.0,
    )
    plot_heatmap(
        records, output_dir, lambda recs: _cell_mean_fraction(recs, "first_correct_fraction"),
        "Mean first-correct fraction by complexity and size", "Mean first-correct fraction",
        "8_first_correct_fraction_heatmap.png", precision=2, vmin=0.0, vmax=1.0,
    )
    plot_heatmap(
        records, output_dir, lambda recs: _cell_pct_idx_zero(recs, "convinced_idx"),
        "Convinced with no reasoning by complexity and size", "% convinced with no reasoning",
        "9_convinced_no_reasoning_heatmap.png", precision=0, vmin=0.0, vmax=100.0,
    )
    plot_heatmap(
        records, output_dir, lambda recs: _cell_pct_idx_zero(recs, "first_correct_idx"),
        "First-correct with no reasoning by complexity and size", "% first-correct with no reasoning",
        "10_first_correct_no_reasoning_heatmap.png", precision=0, vmin=0.0, vmax=100.0,
    )
    plot_committed_before_full(records, output_dir)
    plot_heatmap(
        records, output_dir, lambda recs: _cell_pct_before_full(recs, "convinced_idx"),
        "Convinced before full reasoning by complexity and size", "% convinced before full reasoning",
        "12_convinced_before_full_heatmap.png", precision=0, vmin=0.0, vmax=100.0,
    )
    plot_heatmap(
        records, output_dir, lambda recs: _cell_pct_before_full(recs, "first_correct_idx"),
        "First-correct before full reasoning by complexity and size", "% first-correct before full reasoning",
        "13_first_correct_before_full_heatmap.png", precision=0, vmin=0.0, vmax=100.0,
    )

    print(f"Figures written to {output_dir}/")


if __name__ == "__main__":
    main()
