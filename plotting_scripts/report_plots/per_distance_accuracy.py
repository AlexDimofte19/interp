import argparse
import json
import os

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prediction_visualization import get_predicted_class
from tqdm import tqdm

font_path = "../data/Roboto-Regular.ttf"
fm.fontManager.addfont(str(font_path))
plt.rcParams.update(
    {
        "font.family": "Roboto",
    }
)

plotting_dir = "plots"


def get_coordinates_within_radius(point: tuple[int, int], radius: int, grid_size: int) -> list[tuple[int, int]]:
    x_point, y_point = point
    if x_point < 0 or x_point >= grid_size or y_point < 0 or y_point >= grid_size:
        raise ValueError(f"Point {point} is out of bounds for grid size {grid_size}")
    for x in range(x_point - radius, x_point + radius + 1):
        for y in range(y_point - radius, y_point + radius + 1):
            if x < 0 or x >= grid_size or y < 0 or y >= grid_size:
                continue
            yield (x, y)


def get_accuracy_at_radius_from_point(
    correctness_matrix: np.ndarray, point: tuple[int, int], radius: int, grid_size: int
) -> float:
    correct_count = 0
    total_count = 0
    for x, y in get_coordinates_within_radius(point, radius, grid_size):
        if correctness_matrix[x, y] == 1:
            correct_count += 1
        total_count += 1
    return correct_count / total_count


def plot_per_distance_accuracy(csv_path: str, predictions_path: str, grid_size: int):
    with open(predictions_path) as f:
        predictions = json.load(f)

    original_df = pd.read_csv(csv_path)

    assert len(original_df) == len(predictions), "Number of rows in csv and predictions must match"

    radii = range(0, grid_size)
    acc_at_radius_from_agent = dict.fromkeys(radii, 0)
    acc_at_radius_from_goal = dict.fromkeys(radii, 0)

    for row_idx, row in tqdm(original_df.iterrows(), total=len(original_df)):
        true_fo_cell_types = eval(row.fo_cell_types)
        row_predictions = predictions[row_idx]["predictions"]

        # First, identify which cells were correctly predicted.
        row_correctness_matrix = np.zeros((grid_size, grid_size))
        # In the meantime, also identify the (x,y) of the agent and the goal.
        agent_pos = None
        goal_pos = None
        for x, y, true_cell_type in true_fo_cell_types:
            xy_key = f"({x}, {y})"

            if xy_key not in row_predictions:
                raise ValueError(f"xy_key {xy_key} not in predictions")
            pred_xy_probabilities = row_predictions[xy_key]
            predicted_class, max_probability = get_predicted_class(pred_xy_probabilities)

            if predicted_class == true_cell_type:
                row_correctness_matrix[x, y] = 1

            if true_cell_type == "agent":
                agent_pos = (x, y)
            elif true_cell_type == "goal":
                goal_pos = (x, y)

        if agent_pos is None or goal_pos is None:
            raise ValueError(f"Agent or goal position not found for row {row_idx}")

        for radius in radii:
            acc_at_radius_from_agent_for_row = get_accuracy_at_radius_from_point(
                row_correctness_matrix, agent_pos, radius, grid_size
            )
            acc_at_radius_from_goal_for_row = get_accuracy_at_radius_from_point(
                row_correctness_matrix, goal_pos, radius, grid_size
            )

            acc_at_radius_from_agent[radius] += acc_at_radius_from_agent_for_row
            acc_at_radius_from_goal[radius] += acc_at_radius_from_goal_for_row

    acc_at_radius_from_agent = {radius: acc / len(original_df) for radius, acc in acc_at_radius_from_agent.items()}
    acc_at_radius_from_goal = {radius: acc / len(original_df) for radius, acc in acc_at_radius_from_goal.items()}

    print(f"Accuracy at radius from agent: {acc_at_radius_from_agent}")
    print(f"Accuracy at radius from goal: {acc_at_radius_from_goal}")

    # Plotting
    radii_list = list(radii)
    agent_accuracies = np.array([acc_at_radius_from_agent[r] for r in radii_list])
    goal_accuracies = np.array([acc_at_radius_from_goal[r] for r in radii_list])

    os.makedirs(plotting_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        radii_list,
        agent_accuracies,
        label="From Agent",
        color="coral",
        linewidth=2,
        marker="s",
        markersize=8,
        markeredgecolor="white",
        markeredgewidth=1,
    )

    ax.plot(
        radii_list,
        goal_accuracies,
        label="From Goal",
        color="steelblue",
        linewidth=2,
        marker="d",
        markersize=8,
        markeredgecolor="white",
        markeredgewidth=1,
    )

    # Grid lines
    ax.grid(True, linestyle="--", alpha=0.6)

    ax.set_xticks(radii_list)
    ax.set_xticklabels(radii_list)

    y_min = min(np.min(agent_accuracies), np.min(goal_accuracies))
    y_max = max(np.max(agent_accuracies), np.max(goal_accuracies))
    ax.set_ylim(y_min - 0.05, y_max + 0.05)

    # set size of yticks
    ax.tick_params(axis="y", labelsize=14)

    ax.set_xlabel("Radius", fontsize=22)
    ax.set_ylabel("Accuracy", fontsize=22)
    legend_title = os.path.basename(predictions_path)
    ax.legend(
        title=legend_title,
        title_fontsize=10,
        loc="upper center",
        fontsize=20,
        ncols=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.30),
    )
    plt.tight_layout()

    csv_basename = os.path.splitext(os.path.basename(csv_path))[0]
    results_filename = csv_basename + "_per_distance_acc.png"
    results_path = os.path.join(plotting_dir, results_filename)
    plt.savefig(results_path, dpi=300)
    print(f"Plot saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--predictions_path", type=str, required=True)
    parser.add_argument("--grid_size", type=int, required=True)
    args = parser.parse_args()

    plot_per_distance_accuracy(args.csv_path, args.predictions_path, args.grid_size)
