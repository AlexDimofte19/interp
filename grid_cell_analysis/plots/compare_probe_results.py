"""Compare two probe result JSON files with bar charts: by size, by complexity, and overall."""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def _load(path):
    with open(path) as f:
        return json.load(f)


def _sorted_keys_numeric(d):
    return sorted(d.keys(), key=lambda k: float(k))


def _bar_chart(ax, labels, values_a, values_b, label_a, label_b, baseline_a=None, baseline_b=None):
    x = np.arange(len(labels))
    width = 0.38

    bars_a = ax.bar(x - width / 2, values_a, width, label=label_a, color="darkmagenta")
    bars_b = ax.bar(x + width / 2, values_b, width, label=label_b, color="teal")

    for bars, values in [(bars_a, values_a), (bars_b, values_b)]:
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.005,
                f"{v * 100:.1f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    if baseline_a is not None and baseline_b is not None:
        for i, (ba, bb) in enumerate(zip(baseline_a, baseline_b)):
            ax.hlines(ba, x[i] - width, x[i], colors="darkmagenta", linestyles="--", alpha=0.6)
            ax.hlines(bb, x[i], x[i] + width, colors="teal", linestyles="--", alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")


def plot_by_size(data_a, data_b, label_a, label_b, out_dir):
    sizes = _sorted_keys_numeric(data_a["by_size"])
    acc_a = [data_a["by_size"][s]["accuracy"] for s in sizes]
    acc_b = [data_b["by_size"][s]["accuracy"] for s in sizes]
    base_a = [data_a["by_size"][s]["baseline_accuracy"] for s in sizes]
    base_b = [data_b["by_size"][s]["baseline_accuracy"] for s in sizes]

    fig, ax = plt.subplots(figsize=(10, 5))
    _bar_chart(ax, sizes, acc_a, acc_b, label_a, label_b, base_a, base_b)
    ax.set_xlabel("Grid Size")
    ax.set_title("Accuracy by Grid Size (dashed = baseline)")
    plt.tight_layout()
    path = os.path.join(out_dir, "compare_by_size.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_by_complexity(data_a, data_b, label_a, label_b, out_dir):
    cxs = _sorted_keys_numeric(data_a["by_complexity"])
    acc_a = [data_a["by_complexity"][c]["accuracy"] for c in cxs]
    acc_b = [data_b["by_complexity"][c]["accuracy"] for c in cxs]
    base_a = [data_a["by_complexity"][c]["baseline_accuracy"] for c in cxs]
    base_b = [data_b["by_complexity"][c]["baseline_accuracy"] for c in cxs]

    fig, ax = plt.subplots(figsize=(10, 5))
    _bar_chart(ax, cxs, acc_a, acc_b, label_a, label_b, base_a, base_b)
    ax.set_xlabel("Complexity")
    ax.set_title("Accuracy by Complexity (dashed = baseline)")
    plt.tight_layout()
    path = os.path.join(out_dir, "compare_by_complexity.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_overall(data_a, data_b, label_a, label_b, out_dir):
    acc_a = data_a["global"]["accuracy"]
    acc_b = data_b["global"]["accuracy"]
    base_a = data_a["global"]["baseline_accuracy"]
    base_b = data_b["global"]["baseline_accuracy"]

    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(2)
    bars = ax.bar(x, [acc_a, acc_b], color=["darkmagenta", "teal"], width=0.5)
    for bar, v in zip(bars, [acc_a, acc_b]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.01,
            f"{v * 100:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.hlines(base_a, x[0] - 0.25, x[0] + 0.25, colors="black", linestyles="--", label="Baseline")
    ax.hlines(base_b, x[1] - 0.25, x[1] + 0.25, colors="black", linestyles="--")

    ax.set_xticks(x)
    ax.set_xticklabels([label_a, label_b])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Overall Accuracy (dashed = baseline)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = os.path.join(out_dir, "compare_overall.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Compare two probe result JSON files.")
    parser.add_argument("file_a", help="Path to first probe results JSON")
    parser.add_argument("file_b", help="Path to second probe results JSON")
    parser.add_argument("--label-a", default="Probe A")
    parser.add_argument("--label-b", default="Probe B")
    parser.add_argument("--out-dir", default="plots")
    args = parser.parse_args()

    data_a = _load(args.file_a)
    data_b = _load(args.file_b)

    os.makedirs(args.out_dir, exist_ok=True)

    plot_overall(data_a, data_b, args.label_a, args.label_b, args.out_dir)
    plot_by_size(data_a, data_b, args.label_a, args.label_b, args.out_dir)
    plot_by_complexity(data_a, data_b, args.label_a, args.label_b, args.out_dir)


if __name__ == "__main__":
    main()
