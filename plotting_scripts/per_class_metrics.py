# Script to plot per-class precision and recall for MLP probes.
# Two stacked bar plots: top is recall (=accuracy), bottom is precision.
# X-axis is grid size, each grid size has 4 bars (one per class).

import json
import os

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

font_path = "../data/Roboto-Regular.ttf"
bold_font_path = "../data/Roboto-Bold.ttf"
fm.fontManager.addfont(str(font_path))
fm.fontManager.addfont(str(bold_font_path))
plt.rcParams.update(
    {
        "font.family": "Roboto",
    }
)

mlp_results_paths = {
    7: "../data/trajectories_test_full_with_probes/layer15/mlp_general/pre_reasoning/size7/_eval_results.json",
    9: "../data/trajectories_test_full_with_probes/layer15/mlp_general/pre_reasoning/size9/_eval_results.json",
    11: "../data/trajectories_test_full_with_probes/layer15/mlp_general/pre_reasoning/size11/_eval_results.json",
    13: "../data/trajectories_test_full_with_probes/layer15/mlp_general/pre_reasoning/size13/_eval_results.json",
    15: "../data/trajectories_test_full_with_probes/layer15/mlp_general/pre_reasoning/size15/_eval_results.json",
}

grid_sizes = [7, 9, 11, 13, 15]
classes = ["wall", "empty", "agent", "goal"]
plotting_dir = "plots"

# Colors for each class
class_colors = {
    "wall": "#5D5D5D",  # dark gray
    "empty": "#3498DB",  # blue
    "agent": "#E74C3C",  # red
    "goal": "#27AE60",  # green
}


def plot_per_class_metrics():
    # 1. Load the data
    # Structure: {class_name: {"precision": [...], "recall": [...]}}
    metrics_data = {cls: {"precision": [], "recall": []} for cls in classes}

    for size in grid_sizes:
        with open(mlp_results_paths[size]) as f:
            data = json.load(f)
            for cls in classes:
                metrics_data[cls]["precision"].append(data["per_class_precision"][cls])
                metrics_data[cls]["recall"].append(data["per_class_recall"][cls])

    # Convert to numpy arrays
    for cls in classes:
        metrics_data[cls]["precision"] = np.array(metrics_data[cls]["precision"])
        metrics_data[cls]["recall"] = np.array(metrics_data[cls]["recall"])

    # 2. Create the figure with two subplots
    os.makedirs(plotting_dir, exist_ok=True)
    fig, (ax_recall, ax_precision) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    # Bar positioning
    x = np.arange(len(grid_sizes))
    bar_width = 0.20
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * bar_width

    # --- Top plot: Recall (= Accuracy) ---
    for i, cls in enumerate(classes):
        bars = ax_recall.bar(
            x + offsets[i],
            metrics_data[cls]["recall"],
            bar_width,
            label=cls.capitalize(),
            color=class_colors[cls],
            edgecolor="white",
            linewidth=1,
        )
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax_recall.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.02,
                f"{height * 100:.0f}",
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
                color=class_colors[cls],
            )

    ax_recall.set_ylabel("Recall (= Accuracy)", fontsize=18)
    ax_recall.set_ylim(0, 1.15)
    ax_recall.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_recall.tick_params(axis="both", labelsize=12)
    ax_recall.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax_recall.set_axisbelow(True)
    ax_recall.legend(
        loc="upper center",
        fontsize=14,
        ncols=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.22),
    )

    ax_recall.spines[["right", "top"]].set_visible(False)

    # --- Bottom plot: Precision ---
    for i, cls in enumerate(classes):
        bars = ax_precision.bar(
            x + offsets[i],
            metrics_data[cls]["precision"],
            bar_width,
            label=cls.capitalize(),
            color=class_colors[cls],
            edgecolor="white",
            linewidth=1,
        )
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax_precision.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.02,
                f"{height * 100:.0f}",
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
                color=class_colors[cls],
            )

    ax_precision.set_ylabel("Precision", fontsize=18)
    ax_precision.set_ylim(0, 1.10)
    ax_precision.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_precision.tick_params(axis="y", labelsize=12)
    ax_precision.tick_params(axis="x", labelsize=15)
    ax_precision.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax_precision.set_axisbelow(True)

    # X-axis labels
    ax_precision.set_xticks(x)
    ax_precision.set_xticklabels(grid_sizes)
    ax_precision.set_xlabel("Grid Size", fontsize=18)

    ax_precision.spines[["right", "top"]].set_visible(False)

    plt.tight_layout()

    results_path = os.path.join(plotting_dir, "per_class_metrics.png")
    plt.savefig(results_path, dpi=300, bbox_inches="tight")
    pdf_path = results_path.replace(".png", ".pdf")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")

    print(f"Saved plot to {results_path}")
    print(f"Saved PDF to {pdf_path}")


if __name__ == "__main__":
    plot_per_class_metrics()
