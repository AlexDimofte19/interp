import argparse
import json
import os

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

font_path = "../data/Roboto-Regular.ttf"
fm.fontManager.addfont(str(font_path))
plt.rcParams.update(
    {
        "font.family": "Roboto",
    }
)

class_names = ["goal", "agent", "wall", "empty"]


results_filename_template = "predictions_multiclass_probe_layer_{layer}_gpu_{mlp_infix}normalized_balanced.pkl_l{layer}_full_full_prompt_evaluation.json"
layers = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
plotting_dir = "plots"


def plot_probe_accuracy(results_dir: str):
    # 1. Download the data
    linear_results = []
    mlp_results = []
    for layer in layers:
        linear_path = os.path.join(results_dir, results_filename_template.format(layer=layer, mlp_infix=""))
        mlp_path = os.path.join(results_dir, results_filename_template.format(layer=layer, mlp_infix="mlp1024_"))

        linear_results.append(json.load(open(linear_path)))
        mlp_results.append(json.load(open(mlp_path)))

    linear_accuracies = np.array([result["total_accuracy"] for result in linear_results])
    mlp_accuracies = np.array([result["total_accuracy"] for result in mlp_results])

    linear_min = np.min(linear_accuracies)
    linear_max, linear_max_layer = np.max(linear_accuracies), layers[np.argmax(linear_accuracies)]
    mlp_max, mlp_max_layer = np.max(mlp_accuracies), layers[np.argmax(mlp_accuracies)]

    linear_best_results = linear_results[np.argmax(linear_accuracies)]
    mlp_best_results = mlp_results[np.argmax(mlp_accuracies)]

    linear_best_results_string = f"Best: {linear_best_results['total_accuracy'] * 100:.1f}%"
    for class_name in class_names:
        accuracy = linear_best_results["total_per_class_accuracies"][class_name]
        linear_best_results_string += f"\n{class_name.capitalize()}: {accuracy * 100:.1f}%"

    mlp_best_results_string = f"Best: {mlp_best_results['total_accuracy'] * 100:.1f}%"
    for class_name in class_names:
        accuracy = mlp_best_results["total_per_class_accuracies"][class_name]
        mlp_best_results_string += f"\n{class_name.capitalize()}: {accuracy * 100:.1f}%"

    results_filename = os.path.basename(results_dir) + "_probe_acc.png"
    results_path = os.path.join(plotting_dir, results_filename)

    # 2. Plot the accuracy curves
    os.makedirs(plotting_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        layers,
        linear_accuracies,
        label="Linear Probe",
        color="teal",
        linewidth=2,
        marker="o",
        markersize=8,
        markeredgecolor="white",
        markeredgewidth=2,
    )

    ax.plot(
        layers,
        mlp_accuracies,
        label="MLP Probe",
        color="darkmagenta",
        linewidth=2,
        marker="o",
        markersize=8,
        markeredgecolor="white",
        markeredgewidth=2,
    )

    # write best results at argmax for each line
    # linear
    ax.text(linear_max_layer, linear_max - 0.20, linear_best_results_string, fontsize=7, ha="center", va="bottom")
    # mlp
    ax.text(mlp_max_layer, mlp_max + 0.03, mlp_best_results_string, fontsize=7, ha="center", va="bottom")

    # Grid lines
    ax.grid(True, linestyle="--", alpha=0.6)

    ax.set_xticks(layers)
    ax.set_xticklabels(layers)

    ax.set_ylim(linear_min - 0.25, mlp_max + 0.25)

    # set size of yticks
    ax.tick_params(axis="y", labelsize=14)

    ax.set_xlabel("Layer", fontsize=22)
    ax.set_ylabel("Accuracy", fontsize=22)
    legend_title = f"{os.path.basename(results_dir)}"
    ax.legend(
        title=legend_title,
        title_fontsize=16,
        loc="upper center",
        fontsize=20,
        ncols=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.30),
    )
    plt.tight_layout()

    plt.savefig(results_path, dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)

    args = parser.parse_args()

    plot_probe_accuracy(args.results_dir)
