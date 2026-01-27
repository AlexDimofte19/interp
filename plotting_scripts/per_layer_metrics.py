# Script to plot per-layer precision and recall for MLP probes.
# Two stacked bar plots: top is recall (=accuracy), bottom is precision.
# X-axis is layer, each layer has 5 bars (one per class, and one for overall).

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
    7: "../data/trajectories_test_full_with_probes/layer7/mlp_general/pre_reasoning/size11/_eval_results.json",
    15: "../data/trajectories_test_full_with_probes/layer15/mlp_general/pre_reasoning/size11/_eval_results.json",
    23: "../data/trajectories_test_full_with_probes/layer23/mlp_general/pre_reasoning/size11/_eval_results.json",
}

layers = [7, 15, 23]
classes = ["wall", "empty", "agent", "goal"]
classes_with_overall = ["wall", "empty", "agent", "goal", "overall"]
plotting_dir = "plots"

# Colors for each class
class_colors = {
    "wall": "#5D5D5D",      # dark gray
    "empty": "#3498DB",     # blue
    "agent": "#E74C3C",     # red
    "goal": "#27AE60",      # green
    "overall": "#9B59B6",   # purple
}


def compute_overall_precision(data):
    """Compute overall precision from per_class_counts."""
    counts = data["per_class_counts"]
    total_correct = sum(counts[cls]["correct"] for cls in classes)
    total_predicted = sum(counts[cls]["predicted"] for cls in classes)
    return total_correct / total_predicted if total_predicted > 0 else 0.0


def plot_per_layer_metrics():
    # 1. Load the data
    # Structure: {class_name: {"precision": [...], "recall": [...]}}
    metrics_data = {cls: {"precision": [], "recall": []} for cls in classes_with_overall}

    for layer in layers:
        with open(mlp_results_paths[layer]) as f:
            data = json.load(f)
            for cls in classes:
                metrics_data[cls]["precision"].append(data["per_class_precision"][cls])
                metrics_data[cls]["recall"].append(data["per_class_recall"][cls])
            # Add overall metrics
            metrics_data["overall"]["recall"].append(data["overall_accuracy"])
            metrics_data["overall"]["precision"].append(compute_overall_precision(data))

    # Convert to numpy arrays
    for cls in classes_with_overall:
        metrics_data[cls]["precision"] = np.array(metrics_data[cls]["precision"])
        metrics_data[cls]["recall"] = np.array(metrics_data[cls]["recall"])

    # 2. Create the figure with two subplots
    os.makedirs(plotting_dir, exist_ok=True)
    fig, (ax_recall, ax_precision) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    # Bar positioning (5 bars now)
    x = np.arange(len(layers))
    bar_width = 0.16
    offsets = np.array([-2, -1, 0, 1, 2]) * bar_width

    # --- Top plot: Recall (= Accuracy) ---
    for i, cls in enumerate(classes_with_overall):
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
        ncols=5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.22),
    )

    ax_recall.spines[['right', 'top']].set_visible(False)

    # --- Bottom plot: Precision ---
    for i, cls in enumerate(classes_with_overall):
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
    ax_precision.set_xticklabels(layers)
    ax_precision.set_xlabel("Layer", fontsize=18)

    ax_precision.spines[['right', 'top']].set_visible(False)

    plt.tight_layout()

    results_path = os.path.join(plotting_dir, "per_layer_metrics.png")
    plt.savefig(results_path, dpi=300, bbox_inches="tight")
    pdf_path = results_path.replace(".png", ".pdf")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")

    print(f"Saved plot to {results_path}")
    print(f"Saved PDF to {pdf_path}")


if __name__ == "__main__":
    plot_per_layer_metrics()
