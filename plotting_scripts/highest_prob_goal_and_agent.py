import argparse
import json

import numpy as np
import pandas as pd
from tqdm import tqdm


def manhattan_distance(pos1: tuple[int, int], pos2: tuple[int, int]) -> int:
    """Calculate Manhattan distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def find_cell_with_highest_class_probability(predictions: dict, target_class: str) -> tuple[tuple[int, int], float]:
    """Find the cell with highest probability for a given class.

    Args:
        predictions: Dictionary mapping "(x, y)" to {class_name: probability, ...}
        target_class: The class to find (e.g., "agent" or "goal")

    Returns:
        Tuple of ((x, y), probability) for the cell with highest probability for target_class
    """
    max_prob = -1
    best_cell = None

    for xy_key, class_probs in predictions.items():
        if target_class in class_probs:
            prob = class_probs[target_class]
            if prob > max_prob:
                max_prob = prob
                # Parse "(x, y)" string to tuple
                best_cell = eval(xy_key)

    return best_cell, max_prob


def compute_prediction_metrics(csv_path: str, predictions_path: str, verbose: bool = True) -> dict:
    """Compute accuracy and Manhattan distance metrics for agent and goal predictions.

    Returns:
        Dictionary with keys: agent_accuracy, goal_accuracy, avg_agent_distance, avg_goal_distance
    """

    with open(predictions_path) as f:
        predictions = json.load(f)

    original_df = pd.read_csv(csv_path)

    assert len(original_df) == len(predictions), "Number of rows in csv and predictions must match"

    # Metrics to track
    agent_correct = 0
    goal_correct = 0
    agent_distances = []
    goal_distances = []

    for row_idx, row in tqdm(original_df.iterrows(), total=len(original_df), disable=not verbose):
        true_fo_cell_types = eval(row.fo_cell_types)
        row_predictions = predictions[row_idx]["predictions"]

        # Find ground truth positions
        agent_pos = None
        goal_pos = None
        for x, y, cell_type in true_fo_cell_types:
            if cell_type == "agent":
                agent_pos = (x, y)
            elif cell_type == "goal":
                goal_pos = (x, y)

        if agent_pos is None or goal_pos is None:
            raise ValueError(f"Agent or goal position not found for row {row_idx}")

        # Find predicted positions (cells with highest probability for each class)
        predicted_agent_pos, _ = find_cell_with_highest_class_probability(row_predictions, "agent")
        predicted_goal_pos, _ = find_cell_with_highest_class_probability(row_predictions, "goal")

        # Calculate accuracy
        if predicted_agent_pos == agent_pos:
            agent_correct += 1
        if predicted_goal_pos == goal_pos:
            goal_correct += 1

        # Calculate Manhattan distances
        if predicted_agent_pos is not None:
            agent_distances.append(manhattan_distance(predicted_agent_pos, agent_pos))
        if predicted_goal_pos is not None:
            goal_distances.append(manhattan_distance(predicted_goal_pos, goal_pos))

    # Compute final metrics
    total_rows = len(original_df)
    agent_accuracy = agent_correct / total_rows
    goal_accuracy = goal_correct / total_rows
    avg_agent_distance = np.mean(agent_distances) if agent_distances else float("nan")
    avg_goal_distance = np.mean(goal_distances) if goal_distances else float("nan")

    if verbose:
        print("\n=== Results ===")
        print(f"Total samples: {total_rows}")
        print("\nAgent Predictions:")
        print(f"  Accuracy: {agent_accuracy:.4f} ({agent_correct}/{total_rows})")
        print(f"  Avg Manhattan Distance: {avg_agent_distance:.4f}")
        print("\nGoal Predictions:")
        print(f"  Accuracy: {goal_accuracy:.4f} ({goal_correct}/{total_rows})")
        print(f"  Avg Manhattan Distance: {avg_goal_distance:.4f}")

    return {
        "agent_accuracy": agent_accuracy,
        "goal_accuracy": goal_accuracy,
        "avg_agent_distance": avg_agent_distance,
        "avg_goal_distance": avg_goal_distance,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--predictions_path", type=str, required=True)
    args = parser.parse_args()

    compute_prediction_metrics(args.csv_path, args.predictions_path)
