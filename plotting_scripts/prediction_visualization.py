import argparse
import json

import numpy as np
import pandas as pd
from telos_interp import probing_gpu


# For inspiration from reveng repo
def _get_grid(
    self,
    seen_mask: np.ndarray | None = None,
    add_indices: bool = True,
) -> str:
    env: MiniGridEnv = self.unwrapped  # type: ignore
    if add_indices:
        tot_height = env.height + 1
        tot_width = env.width + 1
    else:
        tot_height = env.height
        tot_width = env.width
    grid = np.full((tot_height, tot_width), "unknown_obj", dtype=object)

    # Pad appropriately for column indices
    padding = len(str(env.width)) + 1

    for j in range(tot_height):
        for i in range(tot_width):
            if add_indices:
                if j == 0:
                    if i == 0:
                        grid[j, i] = " " * padding  # Top-left corner
                    else:
                        grid[j, i] = f"{i - 1:<{padding}}"  # Column indices
                elif i == 0:
                    grid[j, i] = f"{j - 1:<{padding}}"  # Row indices
                elif i == tot_width - 1:
                    grid[j, i] = self._get_cell_at_position(env, i - 1, j - 1, seen_mask) + " "
                else:
                    grid[j, i] = f"{self._get_cell_at_position(env, i - 1, j - 1, seen_mask):<{padding}}"
            else:
                grid[j, i] = self._get_cell_at_position(env, i, j, seen_mask)
    return "\n".join("".join(row) for row in grid)


class_to_symbol = {
    "agent": "A",
    "goal": "G",
    "wall": "#",
    "empty": "_",
}


def get_predicted_cell_at_position(predictions: dict, xy_key: str) -> str:
    pred_xy_probabilities = predictions[xy_key]
    predicted_class, max_probability = probing_gpu.get_predicted_class(pred_xy_probabilities)
    return class_to_symbol[predicted_class]


def show_prediction_visualization(csv_path: str, predictions_path: str, grid_size: int, index: int):
    with open(predictions_path) as f:
        predictions = json.load(f)

    original_df = pd.read_csv(csv_path)

    assert len(original_df) == len(predictions), "Number of rows in csv and predictions must match"

    original_row = original_df.iloc[index]
    predictions = predictions[index]["predictions"]

    print(f"Ground-truth observation:\n{original_row.fo_observation}")

    tot_width = grid_size + 1
    tot_height = grid_size + 1

    padding = len(str(grid_size)) + 1

    grid = np.full((tot_height, tot_width), "unknown_obj", dtype=object)

    for y in range(tot_height):
        for x in range(tot_width):
            if y == 0:
                if x == 0:
                    grid[y, x] = " " * padding  # Top-left corner
                else:
                    grid[y, x] = f"{x - 1:<{padding}}"  # Column indices
            elif x == 0:
                grid[y, x] = f"{y - 1:<{padding}}"  # Row indices
            else:
                xy_key = f"({x - 1}, {y - 1})"
                predicted_cell = get_predicted_cell_at_position(predictions, xy_key)
                if x == tot_width - 1:
                    grid[y, x] = predicted_cell + " "
                else:
                    grid[y, x] = f"{predicted_cell:<{padding}}"

    predicted_grid_string = "\n".join("".join(row) for row in grid)

    print(f"Predicted observation:\n{predicted_grid_string}")

    # Create predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--predictions_path", type=str, required=True)
    parser.add_argument("--grid_size", type=int, required=True)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    show_prediction_visualization(args.csv_path, args.predictions_path, args.grid_size, args.index)
