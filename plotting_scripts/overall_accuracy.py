# Script to plot overall accuracy for size-agnostic (general) MLP and LR probes.
# It plots a line plot, y is accuracy, x is grid size 7,9,11,13,15

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

lr_results_paths = {
    7: "../data/trajectories_test_full_with_probes/layer15/lr/pre_reasoning/size7/_eval_results.json",
    9: "../data/trajectories_test_full_with_probes/layer15/lr/pre_reasoning/size9/_eval_results.json",
    11: "../data/trajectories_test_full_with_probes/layer15/lr/pre_reasoning/size11/_eval_results.json",
    13: "../data/trajectories_test_full_with_probes/layer15/lr/pre_reasoning/size13/_eval_results.json",
    15: "../data/trajectories_test_full_with_probes/layer15/lr/pre_reasoning/size15/_eval_results.json",
}

grid_sizes = [7, 9, 11, 13, 15]
plotting_dir = "plots"


def plot_overall_accuracy():
    # 1. Load the data
    mlp_accuracies = []
    lr_accuracies = []

    for size in grid_sizes:
        with open(mlp_results_paths[size]) as f:
            mlp_data = json.load(f)
            mlp_accuracies.append(mlp_data["overall_accuracy"])

        with open(lr_results_paths[size]) as f:
            lr_data = json.load(f)
            lr_accuracies.append(lr_data["overall_accuracy"])

    mlp_accuracies = np.array(mlp_accuracies)
    lr_accuracies = np.array(lr_accuracies)

    # 2. Plot the accuracy curves
    os.makedirs(plotting_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))  # 2:1 aspect ratio

    ax.plot(
        grid_sizes,
        mlp_accuracies,
        label="MLP Probe",
        color="darkmagenta",
        linewidth=4,
        marker="o",
        markersize=10,
        markeredgecolor="white",
        markeredgewidth=2,
    )

    ax.plot(
        grid_sizes,
        lr_accuracies,
        label="Linear Probe",
        color="teal",
        linewidth=4,
        marker="o",
        markersize=10,
        markeredgecolor="white",
        markeredgewidth=2,
    )

   

    # Add accuracy labels to each point
    for i, size in enumerate(grid_sizes):
        # MLP labels (below the point)
        ax.text(
            size, mlp_accuracies[i] - 0.025,
            f"{mlp_accuracies[i] * 100:.1f}",
            ha="center", va="top", fontsize=18, color="darkmagenta", fontweight="bold"
        )

         # LR labels (above the point)
        ax.text(
            size, lr_accuracies[i] + 0.025,
            f"{lr_accuracies[i] * 100:.1f}",
            ha="center", va="bottom", fontsize=18, color="teal", fontweight="bold"
        )

    # Grid lines
    ax.grid(True, linestyle="--", alpha=0.6)

    ax.set_xticks(grid_sizes)
    ax.set_xticklabels(grid_sizes)

    ax.set_yticks([0.2, 0.4, 0.6, 0.8])

    # Set y limits with some padding
    y_min = min(np.min(mlp_accuracies), np.min(lr_accuracies))
    y_max = max(np.max(mlp_accuracies), np.max(lr_accuracies))
    y_min = 0.15
    y_max = 0.85
    ax.set_ylim(y_min - 0.05, y_max + 0.05)

    ax.tick_params(axis="y", labelsize=14)
    ax.tick_params(axis="x", labelsize=16)

    ax.set_xlabel("Grid Size", fontsize=22)
    ax.set_ylabel("Accuracy", fontsize=22)

    ax.legend(
        title_fontsize=16,
        loc="upper center",
        fontsize=20,
        ncols=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.15),
    )

    plt.tight_layout()

    results_path = os.path.join(plotting_dir, "overall_accuracy.png")
    plt.savefig(results_path, dpi=300, bbox_inches="tight")
    pdf_path = results_path.replace(".png", ".pdf")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")

    print(f"Saved plot to {results_path}")
    print(f"Saved PDF to {pdf_path}")


if __name__ == "__main__":
    plot_overall_accuracy()
