# Script to plot highest probability agent/goal metrics across grid sizes.
# Two subplots:
# 1. Bar plot: Accuracy of highest probability goal and agent cell
# 2. Line plot: Manhattan distance between ground truth and highest prob cell
#
# Both plots fit into one column of a 2-column ICML template.

import json
import os
from pathlib import Path

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

# Data paths (same structure as per_class_metrics.py)
base_path = Path("../data/trajectories_test_full_with_probes/layer15/mlp_general/pre_reasoning")
grid_sizes = [7, 9, 11, 13, 15]
plotting_dir = "plots"

# Colors from per_class_metrics.py
class_colors = {
    "agent": "#E74C3C",  # red
    "goal": "#27AE60",   # green
}


def manhattan_distance(pos1: tuple[int, int], pos2: tuple[int, int]) -> int:
    """Calculate Manhattan distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def find_cell_with_highest_class_probability(
    probes: dict, target_class: str, grid_size: int
) -> tuple[tuple[int, int] | None, float]:
    """Find the cell with highest probability for a given class.

    Args:
        probes: Dictionary with keys like "{probe_name}_r{row}_c{col}"
        target_class: The class to find (e.g., "agent" or "goal")
        grid_size: Size of the grid

    Returns:
        Tuple of ((row, col), probability) for the cell with highest probability
    """
    max_prob = -1.0
    best_cell = None

    for probe_key, layer_data in probes.items():
        # Parse row and column from key like "cognitive_map_probe_..._r0_c0"
        if "_r" not in probe_key or "_c" not in probe_key:
            continue

        try:
            parts = probe_key.rsplit("_r", 1)[1]
            row_part, col_part = parts.split("_c")
            row = int(row_part)
            col = int(col_part)
        except (ValueError, IndexError):
            continue

        # Get class probabilities from layer data
        for layer_name, class_probs in layer_data.items():
            if target_class in class_probs:
                prob = class_probs[target_class]
                if prob > max_prob:
                    max_prob = prob
                    best_cell = (row, col)

    return best_cell, max_prob


def extract_probes_from_trajectory(trajectory_data: dict) -> dict | None:
    """Extract probe predictions from a trajectory.

    Probes are stored on tokens. We look for the first token that has probes.
    """
    # Check prompt prefix tokens
    prefix_tokens = trajectory_data.get("prompt", {}).get("prompt_prefix_tokens", [])
    for token in prefix_tokens:
        if "probes" in token and token["probes"]:
            return token["probes"]

    # Check step tokens
    for step in trajectory_data.get("steps", []):
        for token_key in ["prompt_suffix_tokens", "grid_state_tokens", "output_tokens"]:
            tokens = step.get(token_key, [])
            for token in tokens:
                if "probes" in token and token["probes"]:
                    return token["probes"]

    return None


def compute_metrics_for_size(size_folder: Path) -> dict:
    """Compute agent and goal accuracy and distance for a grid size.

    Returns:
        Dictionary with keys: agent_accuracy, goal_accuracy, avg_agent_distance, avg_goal_distance
    """
    json_files = list(size_folder.glob("*.json"))
    # Exclude eval results file
    json_files = [f for f in json_files if not f.name.startswith("_")]

    agent_correct = 0
    goal_correct = 0
    agent_distances = []
    goal_distances = []
    total = 0

    for json_file in json_files:
        with open(json_file) as f:
            trajectory_data = json.load(f)

        # Get ground truth positions
        grid_params = trajectory_data.get("grid_params", {})
        agent_coords = grid_params.get("agent_start_coordinates")
        goal_coords = grid_params.get("goal_coordinates")

        if agent_coords is None or goal_coords is None:
            continue

        # Ground truth as (row, col) - coordinates are already in [row, col] format
        agent_pos = (agent_coords[0], agent_coords[1])  # (row, col)
        goal_pos = (goal_coords[0], goal_coords[1])     # (row, col)

        # Get grid size
        grid_size = max(
            grid_params.get("grid_width", 0),
            grid_params.get("grid_height", 0)
        )

        # Extract probe predictions
        probes = extract_probes_from_trajectory(trajectory_data)
        if probes is None:
            continue

        # Find predicted positions
        predicted_agent_pos, _ = find_cell_with_highest_class_probability(probes, "agent", grid_size)
        predicted_goal_pos, _ = find_cell_with_highest_class_probability(probes, "goal", grid_size)

        if predicted_agent_pos is None or predicted_goal_pos is None:
            continue

        total += 1

        # Calculate accuracy
        if predicted_agent_pos == agent_pos:
            agent_correct += 1
        if predicted_goal_pos == goal_pos:
            goal_correct += 1

        # Calculate Manhattan distances
        agent_distances.append(manhattan_distance(predicted_agent_pos, agent_pos))
        goal_distances.append(manhattan_distance(predicted_goal_pos, goal_pos))

    if total == 0:
        return {
            "agent_accuracy": 0.0,
            "goal_accuracy": 0.0,
            "avg_agent_distance": 0.0,
            "avg_goal_distance": 0.0,
            "total": 0,
        }

    return {
        "agent_accuracy": agent_correct / total,
        "goal_accuracy": goal_correct / total,
        "avg_agent_distance": np.mean(agent_distances) if agent_distances else 0.0,
        "avg_goal_distance": np.mean(goal_distances) if goal_distances else 0.0,
        "total": total,
    }


def plot_highest_prob_metrics():
    # 1. Compute metrics for each grid size
    metrics_by_size = {}
    for size in grid_sizes:
        size_folder = base_path / f"size{size}"
        if size_folder.exists():
            metrics_by_size[size] = compute_metrics_for_size(size_folder)
            print(f"Size {size}: {metrics_by_size[size]}")
        else:
            print(f"Warning: Folder not found: {size_folder}")

    if not metrics_by_size:
        print("No data found!")
        return

    # Extract data for plotting
    sizes = sorted(metrics_by_size.keys())
    agent_accuracies = [metrics_by_size[s]["agent_accuracy"] for s in sizes]
    goal_accuracies = [metrics_by_size[s]["goal_accuracy"] for s in sizes]
    agent_distances = [metrics_by_size[s]["avg_agent_distance"] for s in sizes]
    goal_distances = [metrics_by_size[s]["avg_goal_distance"] for s in sizes]

    # 2. Create figure with two horizontal subplots
    # ICML single column width is ~3.25 inches
    os.makedirs(plotting_dir, exist_ok=True)
    fig, (ax_accuracy, ax_distance) = plt.subplots(
        1, 2, figsize=(8, 3), gridspec_kw={"width_ratios": [1.3, 1]}
    )
    plt.subplots_adjust(wspace=0.4)

    # --- Top plot: Accuracy bar plot ---
    x = np.arange(len(sizes))
    bar_width = 0.40
    offsets = np.array([-0.5, 0.5]) * bar_width

    # Agent bars
    bars_agent = ax_accuracy.bar(
        x + offsets[0],
        agent_accuracies,
        bar_width,
        label="Agent",
        color=class_colors["agent"],
        edgecolor="white",
        linewidth=1.5,
    )

    # Goal bars
    bars_goal = ax_accuracy.bar(
        x + offsets[1],
        goal_accuracies,
        bar_width,
        label="Goal",
        color=class_colors["goal"],
        edgecolor="white",
        linewidth=1.5,
    )

    # Add value labels on top of bars
    for bars, color in [(bars_agent, class_colors["agent"]), (bars_goal, class_colors["goal"])]:
        for bar in bars:
            height = bar.get_height()
            ax_accuracy.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.02,
                f"{height * 100:.0f}",
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
                color=color,
            )

    ax_accuracy.set_ylabel("Accuracy", fontsize=16)
    ax_accuracy.set_ylim(0, 1.15)
    ax_accuracy.set_yticks([0.0, 0.5, 1.0])
    ax_accuracy.tick_params(axis="y", labelsize=12)
    ax_accuracy.tick_params(axis="x", labelsize=14)
    ax_accuracy.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax_accuracy.set_axisbelow(True)
    ax_accuracy.legend(
        loc="upper center",
        fontsize=14,
        ncols=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.20),
    )
    ax_accuracy.spines[["right", "top"]].set_visible(False)

    # X-axis labels for accuracy plot
    ax_accuracy.set_xticks(x)
    ax_accuracy.set_xticklabels(sizes)
    ax_accuracy.set_xlabel("Grid Size", fontsize=16)

    # --- Right plot: Manhattan distance line plot ---
    ax_distance.plot(
        sizes,
        agent_distances,
        label="Agent",
        color=class_colors["agent"],
        linewidth=3,
        marker="o",
        markersize=8,
        markeredgecolor="white",
        markeredgewidth=1.5,
    )

    ax_distance.plot(
        sizes,
        goal_distances,
        label="Goal",
        color=class_colors["goal"],
        linewidth=3,
        marker="o",
        markersize=8,
        markeredgecolor="white",
        markeredgewidth=1.5,
    )

    ax_distance.set_ylabel("Manhattan Dist.", fontsize=16)
    ax_distance.set_xlabel("Grid Size", fontsize=16)
    ax_distance.tick_params(axis="y", labelsize=12)
    ax_distance.tick_params(axis="x", labelsize=14)
    ax_distance.grid(True, linestyle="--", alpha=0.6)
    ax_distance.set_axisbelow(True)

    # Set y limits
    y_max = max(max(agent_distances), max(goal_distances))
    ax_distance.set_ylim(0, y_max + 0.3)

    # X-axis labels
    ax_distance.set_xticks(sizes)
    ax_distance.set_xticklabels(sizes)

    ax_distance.spines[["right", "top"]].set_visible(False)

    plt.tight_layout()

    # Save plots
    results_path = os.path.join(plotting_dir, "highest_prob_metrics.png")
    plt.savefig(results_path, dpi=300, bbox_inches="tight")
    pdf_path = results_path.replace(".png", ".pdf")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")

    print(f"\nSaved plot to {results_path}")
    print(f"Saved PDF to {pdf_path}")


if __name__ == "__main__":
    plot_highest_prob_metrics()

