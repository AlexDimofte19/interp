"""Aggregate and plot reasoning-theatre statistics produced by ``run_inference.py``.

``run_inference.py`` writes one results JSON per trajectory, each with a ``summary`` and
a list of ``steps`` (every step holding per-sentence ``sentence_evals`` with
``answer_prob`` and the commitment indices ``first_correct_sentence_idx`` /
``convinced_sentence_idx``). This script walks the ``<input-folder>/size*/*.json``
layout, pools those records across the dataset, and emits six figures:

1. Mean fraction-of-sentences to first-correct vs. to convinced (whole dataset).
2. Sentence accuracy per size (bar).
3. Mean answer probability before vs. after the convinced point
   (steps convinced at sentence 0 / never convinced are skipped).
4. Mean answer probability before vs. after the first-correct point
   (steps first-correct at sentence 0 / never correct are skipped).
5. Percent of steps first-correct with no reasoning (sentence 0), per size (bar).
6. Percent of steps convinced with no reasoning (sentence 0), per size (bar).

Run on the host with the results, e.g.:
    python scripts/inference_oss/analysis.py \
        --input-folder scripts/inference_oss/inference_results \
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


def load_steps(input_folder: str) -> list[dict]:
    """Flatten every results JSON under ``input_folder`` into per-step records.

    Each record carries the step's commitment indices, fractions, and a light view of
    its ``sentence_evals`` (sentence_idx, correct, answer_prob), plus the parsed size.
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


def _save(fig, output_dir: Path, name: str) -> None:
    out = output_dir / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_fraction_comparison(records: list[dict], output_dir: Path) -> None:
    """Graph 1: mean fraction-of-sentences to first-correct vs. to convinced."""
    first = _mean([r["first_correct_fraction"] for r in records])
    convinced = _mean([r["convinced_fraction"] for r in records])
    labels = ["First correct", "Convinced"]
    values = [first or 0.0, convinced or 0.0]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(x=labels, y=values, ax=ax, palette="muted")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean fraction of reasoning sentences")
    ax.set_title("Reasoning fraction until first-correct vs. convinced")
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    _save(fig, output_dir, "1_fraction_first_correct_vs_convinced.png")


def plot_accuracy_per_size(records: list[dict], output_dir: Path) -> None:
    """Graph 2: pooled sentence accuracy (correct evals / total evals) per size."""
    by_size: dict[int, list[bool]] = {}
    for r in records:
        if r["size"] is None:
            continue
        by_size.setdefault(r["size"], []).extend(e["correct"] for e in r["evals"])

    sizes = sorted(by_size)
    acc = [float(np.mean(by_size[s])) for s in sizes]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=[str(s) for s in sizes], y=acc, ax=ax, palette="crest")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Maze size")
    ax.set_ylabel("Sentence accuracy")
    ax.set_title("Average sentence accuracy per size")
    for i, v in enumerate(acc):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
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
    values = [float(np.mean(before)) if before else 0.0, float(np.mean(after)) if after else 0.0]
    counts = [len(before), len(after)]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(x=[f"Before {label}", f"After {label}"], y=values, ax=ax, palette="flare")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean answer probability")
    ax.set_title(f"Answer probability before vs. after {label}")
    for i, (v, n) in enumerate(zip(values, counts)):
        ax.text(i, v + 0.02, f"{v:.2f}\n(n={n})", ha="center")
    _save(fig, output_dir, name)


def plot_no_reasoning_share(records: list[dict], output_dir: Path, idx_key: str, label: str, name: str) -> None:
    """Graphs 5 & 6: percent of steps that hit the commitment at sentence 0, per size."""
    total: dict[int, int] = {}
    at_zero: dict[int, int] = {}
    for r in records:
        if r["size"] is None:
            continue
        total[r["size"]] = total.get(r["size"], 0) + 1
        if r[idx_key] == 0:
            at_zero[r["size"]] = at_zero.get(r["size"], 0) + 1

    sizes = sorted(total)
    pct = [100.0 * at_zero.get(s, 0) / total[s] for s in sizes]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=[str(s) for s in sizes], y=pct, ax=ax, palette="mako")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Maze size")
    ax.set_ylabel(f"% of steps {label} with no reasoning")
    ax.set_title(f"{label.capitalize()} with no reasoning (sentence 0) per size")
    for i, v in enumerate(pct):
        ax.text(i, v + 1, f"{v:.0f}%", ha="center")
    _save(fig, output_dir, name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-folder", default=DEFAULT_INPUT, help="Folder with <size*>/*.json results from run_inference.py.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT, help="Directory to write the PNG figures into.")
    args = parser.parse_args()

    sns.set_theme(style="whitegrid")

    records = load_steps(args.input_folder)
    n_sized = sum(1 for r in records if r["size"] is not None)
    print(f"Loaded {len(records)} steps ({n_sized} with a parseable size) from {args.input_folder}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_fraction_comparison(records, output_dir)
    plot_accuracy_per_size(records, output_dir)
    plot_prob_around_commitment(records, output_dir, "convinced_idx", "convinced", "3_answer_prob_around_convinced.png")
    plot_prob_around_commitment(records, output_dir, "first_correct_idx", "first correct", "4_answer_prob_around_first_correct.png")
    plot_no_reasoning_share(records, output_dir, "first_correct_idx", "first correct", "5_first_correct_no_reasoning_per_size.png")
    plot_no_reasoning_share(records, output_dir, "convinced_idx", "convinced", "6_convinced_no_reasoning_per_size.png")

    print(f"Figures written to {output_dir}/")


if __name__ == "__main__":
    main()
