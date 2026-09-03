#!/usr/bin/env python3
"""
Visualize side-by-side comparison of true grid state vs decoded (predicted) grid from probes.

This script takes a trajectory JSON file with probe outputs and displays the ground truth
grid alongside the grid reconstructed from probe predictions.
"""

import argparse
import json
import re
from pathlib import Path

# Mapping from class names to symbols for display
CLASS_TO_SYMBOL = {
    "agent": "A",
    "goal": "G",
    "wall": "#",
    "empty": "_",
    "door": "D",
    "key": "K",
    "padding": "+",
    "unknown": "?",
}


def find_probes_in_tokens(tokens: list[dict]) -> dict | None:
    """
    Find the first token with a 'probes' key in the token list.
    """
    for token in tokens:
        if "probes" in token:
            return token["probes"]
    return None


def extract_probe_predictions(probes: dict) -> dict[tuple[int, int], tuple[str, float]]:
    """
    Extract predicted class and probability for each cell from probe results.

    Probe names follow pattern: ...r{row}_c{col}
    Each probe has a dict with class probabilities.
    Returns dict mapping (row, col) -> (predicted_class, max_probability).
    """
    predictions = {}

    # Pattern to extract row and column from probe name
    cell_pattern = re.compile(r"r(\d+)_c(\d+)$")

    for probe_name, probe_data in probes.items():
        match = cell_pattern.search(probe_name)
        if not match:
            continue

        row = int(match.group(1))
        col = int(match.group(2))

        # Get the probabilities from the first (and only) key in probe_data
        for layer_key, probs in probe_data.items():
            # Find the class with highest probability, excluding 'padding'
            best_class = None
            best_prob = -1
            for class_name, prob in probs.items():
                if class_name != "padding" and prob > best_prob:
                    best_prob = prob
                    best_class = class_name

            if best_class:
                predictions[(row, col)] = (best_class, best_prob)
            break  # Only use first layer

    return predictions


def get_grid_size_from_state(grid_state: list[str]) -> int:
    """
    Infer grid size from the grid state format.
    """
    # Find the first data row (skip header line)
    # Header line looks like "  0 1 2 3 4 5 6 " - all parts are digits
    # Data line looks like "0 # # # # # # # " - has row number then symbols
    for line in grid_state:
        parts = line.split()
        if not parts:
            continue
        try:
            int(parts[0])  # First part is row number
            # Check if this is a data row (second part should be a symbol, not a digit)
            if len(parts) > 1 and not parts[1].isdigit():
                return len(parts) - 1  # Number of columns
        except ValueError:
            continue
    return 0


def build_predicted_grid(
    predictions: dict[tuple[int, int], tuple[str, float]],
    grid_size: int,
) -> list[str]:
    """
    Build a grid representation from probe predictions.

    Returns list of strings in the same format as grid_state.
    """
    # Create header line
    padding = len(str(grid_size - 1)) + 1
    header = " " * padding + " ".join(f"{i:<{padding - 1}}" for i in range(grid_size))

    lines = [header + " "]

    for row in range(grid_size):
        row_parts = [f"{row:<{padding - 1}}"]
        for col in range(grid_size):
            if (row, col) in predictions:
                pred_class, _ = predictions[(row, col)]
                symbol = CLASS_TO_SYMBOL.get(pred_class, "?")
            else:
                symbol = "?"
            row_parts.append(symbol)
        lines.append(" ".join(row_parts) + " ")

    return lines


def visualize_grids(
    trajectory_path: Path,
    step: int = 0,
    token_key: str = "prompt_suffix_tokens",
) -> None:
    """
    Load trajectory and visualize true vs decoded grid side by side.

    Args:
        trajectory_path: Path to trajectory JSON file
        step: Step index to visualize (default: 0)
        token_key: Key in step data containing tokens with probes
    """
    with open(trajectory_path) as f:
        data = json.load(f)

    steps = data.get("steps", [])

    if not steps:
        raise ValueError(f"No steps found in trajectory file: {trajectory_path}")

    if step >= len(steps):
        raise ValueError(f"Step {step} out of range. Trajectory has {len(steps)} steps.")

    step_data = steps[step]
    grid_state = step_data.get("grid_state")
    tokens = step_data.get(token_key)

    if not grid_state:
        raise ValueError(f"No grid_state found in step {step}")

    if not tokens:
        raise ValueError(f"No {token_key} found in step {step}")

    # Find probes in tokens
    probes = find_probes_in_tokens(tokens)
    if not probes:
        raise ValueError(f"No probes found in {token_key} for step {step}")

    # Extract predictions
    predictions = extract_probe_predictions(probes)

    # Determine grid size
    grid_size = get_grid_size_from_state(grid_state)

    # Build predicted grid
    predicted_grid = build_predicted_grid(predictions, grid_size)

    # Print header info
    print(f"\nTrajectory: {trajectory_path.name}")
    print(f"Step: {step}")
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Predictions extracted: {len(predictions)}")
    print()

    # Calculate display widths
    true_width = max(len(line) for line in grid_state)
    pred_width = max(len(line) for line in predicted_grid)
    col_width = max(true_width, pred_width) + 4

    # Print side by side
    print(f"{'True Grid':<{col_width}} | {'Decoded Grid':<{col_width}}")
    print(f"{'-' * (col_width - 2):<{col_width}} | {'-' * (col_width - 2):<{col_width}}")

    max_lines = max(len(grid_state), len(predicted_grid))
    for i in range(max_lines):
        true_line = grid_state[i] if i < len(grid_state) else ""
        pred_line = predicted_grid[i] if i < len(predicted_grid) else ""
        print(f"{true_line:<{col_width}} | {pred_line:<{col_width}}")

    # Compute and display accuracy
    correct = 0
    total = 0

    # Parse true grid for comparison
    from collections import defaultdict

    symbol_to_class = {
        "#": "wall",
        "_": "empty",
        "A": "agent",
        "G": "goal",
        "D": "door",
        "K": "key",
        "?": "unknown",
    }

    per_class_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for line in grid_state:
        parts = line.split()
        if not parts:
            continue
        try:
            row = int(parts[0])
        except ValueError:
            continue

        for col, symbol in enumerate(parts[1:]):
            if symbol not in symbol_to_class:
                continue
            true_class = symbol_to_class[symbol]

            if (row, col) in predictions:
                pred_class, _ = predictions[(row, col)]
                total += 1
                per_class_stats[true_class]["total"] += 1

                if pred_class == true_class:
                    correct += 1
                    per_class_stats[true_class]["correct"] += 1

    print()
    if total > 0:
        accuracy = correct / total
        print(f"Overall Accuracy: {accuracy:.2%} ({correct}/{total})")
        print()
        print("Per-class accuracy:")
        for cls in sorted(per_class_stats.keys()):
            stats = per_class_stats[cls]
            if stats["total"] > 0:
                cls_acc = stats["correct"] / stats["total"]
                print(f"  {cls:<10}: {cls_acc:.2%} ({stats['correct']}/{stats['total']})")
    else:
        print("No predictions to evaluate.")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize true vs decoded grid from probe predictions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python visualize_decoded_grid.py path/to/trajectory.json
  python visualize_decoded_grid.py path/to/trajectory.json --step 5
  python visualize_decoded_grid.py path/to/trajectory.json --step 0 --token-key prompt_suffix_tokens
        """,
    )
    parser.add_argument(
        "trajectory",
        type=Path,
        help="Path to trajectory JSON file with probe outputs",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=0,
        help="Step index to visualize (default: 0)",
    )
    parser.add_argument(
        "--token-key",
        type=str,
        default="prompt_suffix_tokens",
        help="Key in step data containing tokens with probes (default: prompt_suffix_tokens)",
    )

    args = parser.parse_args()

    if not args.trajectory.exists():
        raise FileNotFoundError(f"Trajectory file not found: {args.trajectory}")

    visualize_grids(args.trajectory, args.step, args.token_key)


if __name__ == "__main__":
    main()
