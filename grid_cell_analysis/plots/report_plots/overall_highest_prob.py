import argparse
import os

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from highest_prob_goal_and_agent import compute_prediction_metrics

font_path = "../data/Roboto-Regular.ttf"
fm.fontManager.addfont(str(font_path))
plt.rcParams.update(
    {
        "font.family": "Roboto",
    }
)

# Hardcoded template for MLP probe predictions
PREDICTIONS_TEMPLATE = (
    "predictions_multiclass_probe_layer_{layer}_gpu_mlp1024_normalized_balanced.pkl_l{layer}_full_full_prompt.json"
)

# Layers to analyze: 1, 3, 4, 5, ..., 23
LAYERS = list(range(1, 24, 2))

plotting_dir = "plots"


def plot_overall_highest_prob(predictions_dir: str):
    """Plot accuracy and manhattan distance for highest probability goal/agent predictions across layers."""

    # Derive csv_path from predictions_dir
    csv_path = predictions_dir.rstrip("/") + ".csv"

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Collect metrics for each layer
    goal_accuracies = []
    agent_accuracies = []
    goal_distances = []
    agent_distances = []

    for layer in LAYERS:
        predictions_filename = PREDICTIONS_TEMPLATE.format(layer=layer)
        predictions_path = os.path.join(predictions_dir, predictions_filename)

        if not os.path.exists(predictions_path):
            print(f"Warning: Predictions file not found for layer {layer}: {predictions_path}")
            goal_accuracies.append(np.nan)
            agent_accuracies.append(np.nan)
            goal_distances.append(np.nan)
            agent_distances.append(np.nan)
            continue

        print(f"Processing layer {layer}...")
        metrics = compute_prediction_metrics(csv_path, predictions_path, verbose=False)

        goal_accuracies.append(metrics["goal_accuracy"])
        agent_accuracies.append(metrics["agent_accuracy"])
        goal_distances.append(metrics["avg_goal_distance"])
        agent_distances.append(metrics["avg_agent_distance"])

    goal_accuracies = np.array(goal_accuracies)
    agent_accuracies = np.array(agent_accuracies)
    goal_distances = np.array(goal_distances)
    agent_distances = np.array(agent_distances)

    # Plotting
    os.makedirs(plotting_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    x = np.arange(len(LAYERS))
    bar_width = 0.35

    # Top subplot: Bar plots for accuracy
    bars1 = ax1.bar(x - bar_width / 2, goal_accuracies, bar_width, label="Goal", color="steelblue")
    bars2 = ax1.bar(x + bar_width / 2, agent_accuracies, bar_width, label="Agent", color="coral")

    # Add percentage labels on top of bars
    for bar, acc in zip(bars1, goal_accuracies, strict=False):
        if not np.isnan(acc):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{acc * 100:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    for bar, acc in zip(bars2, agent_accuracies, strict=False):
        if not np.isnan(acc):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{acc * 100:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax1.set_ylabel("Accuracy", fontsize=18)
    ax1.set_ylim(0, 1.15)
    ax1.legend(fontsize=14, loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.6, axis="y")
    ax1.set_title("Highest Probability Cell Accuracy by Layer", fontsize=20)

    # Bottom subplot: Line plots for Manhattan distance
    ax2.plot(
        x,
        goal_distances,
        label="Goal",
        color="steelblue",
        linewidth=2,
        marker="d",
        markersize=8,
        markeredgecolor="white",
        markeredgewidth=1,
    )
    ax2.plot(
        x,
        agent_distances,
        label="Agent",
        color="coral",
        linewidth=2,
        marker="s",
        markersize=8,
        markeredgecolor="white",
        markeredgewidth=1,
    )

    ax2.set_xlabel("Layer", fontsize=18)
    ax2.set_ylabel("Avg Manhattan Distance", fontsize=18)
    ax2.set_xticks(x)
    ax2.set_xticklabels(LAYERS)
    ax2.legend(fontsize=14, loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.set_title("Manhattan Distance from Ground Truth by Layer", fontsize=20)

    plt.tight_layout()

    # Save
    dir_basename = os.path.basename(predictions_dir.rstrip("/"))
    results_filename = dir_basename + "_highest_prob.png"
    results_path = os.path.join(plotting_dir, results_filename)
    plt.savefig(results_path, dpi=300)
    print(f"\nPlot saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions_dir",
        type=str,
        required=True,
        help="Directory containing prediction files (e.g., ../data/grids_for_testing/size7_numenvs100_trajectorysteps20)",
    )
    args = parser.parse_args()

    plot_overall_highest_prob(args.predictions_dir)
