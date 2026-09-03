#!/usr/bin/env python3
"""
Compute overall and per-class accuracy of probe predictions.

This script processes trajectory JSON files from a given directory,
computes probe prediction accuracy against ground truth grid states,
and saves results to _eval_results.json in the same directory.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# Mapping from grid symbols to class names
SYMBOL_TO_CLASS = {
    "#": "wall",
    "_": "empty",
    "A": "agent",
    "G": "goal",
    "+": "padding",
    "D": "door",
    "K": "key",
    "?": "unknown",
}


def parse_grid_state(grid_state: list[str]) -> dict[tuple[int, int], str]:
    """
    Parse grid_state strings into a dict mapping (row, col) -> class_name.

    Grid format:
        "  0 1 2 3 4 5 6 "
        "0 # # # # # # # "
        "1 # _ _ _ _ _ # "
        ...
    """
    cell_labels = {}

    for line in grid_state:
        # Skip header line (starts with spaces then column numbers)
        if line.strip().startswith("0 1") or not line.strip():
            continue

        # Parse row: format is "row_num symbol symbol symbol ..."
        parts = line.split()
        if not parts:
            continue

        try:
            row = int(parts[0])
        except ValueError:
            continue

        # Rest are cell symbols
        for col, symbol in enumerate(parts[1:]):
            if symbol in SYMBOL_TO_CLASS:
                cell_labels[(row, col)] = SYMBOL_TO_CLASS[symbol]

    return cell_labels


def extract_probe_predictions(probes: dict) -> dict[tuple[int, int], str]:
    """
    Extract predicted class for each cell from probe results.

    Probe names follow pattern: ...r{row}_c{col}
    Each probe has a dict with class probabilities.
    Returns dict mapping (row, col) -> predicted_class.
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
                predictions[(row, col)] = best_class
            break  # Only use first layer

    return predictions


def find_probes_in_tokens(tokens: list[dict]) -> dict | None:
    """
    Find the first token with a 'probes' key in the token list.
    """
    for token in tokens:
        if "probes" in token:
            return token["probes"]
    return None


def compute_accuracy(
    trajectories_dir: Path,
    token_key: str = "prompt_suffix_tokens",
    include_per_file: bool = False,
) -> dict:
    """
    Compute overall and per-class accuracy across all trajectories.

    Args:
        trajectories_dir: Directory containing trajectory JSON files
        token_key: Key in step data containing tokens with probes
        include_per_file: Whether to include per-file results in output

    Returns:
        Dictionary with accuracy results
    """
    # Counters for accuracy computation
    total_correct = 0
    total_count = 0

    # Per-class counters: class -> {correct: int, total: int, predicted: int}
    # total = actual instances of class (for recall)
    # predicted = times class was predicted (for precision)
    per_class = defaultdict(lambda: {"correct": 0, "total": 0, "predicted": 0})

    # Per-file results
    per_file_results = {}

    # Process all JSON files in directory
    json_files = sorted(trajectories_dir.glob("*.json"))
    json_files = [f for f in json_files if not f.name.startswith("_")]  # Skip result files

    if not json_files:
        raise ValueError(f"No JSON files found in {trajectories_dir}")

    for json_file in json_files:
        file_correct = 0
        file_total = 0
        file_per_class = defaultdict(lambda: {"correct": 0, "total": 0, "predicted": 0})

        with open(json_file) as f:
            data = json.load(f)

        steps = data.get("steps", [])

        for step in steps:
            grid_state = step.get("grid_state")
            tokens = step.get(token_key)

            if not grid_state or not tokens:
                continue

            # Parse ground truth
            ground_truth = parse_grid_state(grid_state)

            # Find probes in tokens
            probes = find_probes_in_tokens(tokens)
            if not probes:
                continue

            # Extract predictions
            predictions = extract_probe_predictions(probes)

            # Compare predictions to ground truth
            for (row, col), true_class in ground_truth.items():
                if (row, col) not in predictions:
                    continue

                pred_class = predictions[(row, col)]

                # Update counters
                total_count += 1
                file_total += 1
                per_class[true_class]["total"] += 1
                file_per_class[true_class]["total"] += 1
                per_class[pred_class]["predicted"] += 1
                file_per_class[pred_class]["predicted"] += 1

                if pred_class == true_class:
                    total_correct += 1
                    file_correct += 1
                    per_class[true_class]["correct"] += 1
                    file_per_class[true_class]["correct"] += 1

        # Compute per-file metrics
        file_accuracy = file_correct / file_total if file_total > 0 else 0
        file_per_class_accuracy = {
            cls: data["correct"] / data["total"] if data["total"] > 0 else 0 for cls, data in file_per_class.items()
        }
        file_per_class_recall = {
            cls: data["correct"] / data["total"] if data["total"] > 0 else 0 for cls, data in file_per_class.items()
        }
        file_per_class_precision = {
            cls: data["correct"] / data["predicted"] if data["predicted"] > 0 else 0
            for cls, data in file_per_class.items()
        }

        per_file_results[json_file.name] = {
            "overall_accuracy": file_accuracy,
            "total_predictions": file_total,
            "correct_predictions": file_correct,
            "per_class_accuracy": file_per_class_accuracy,
            "per_class_recall": file_per_class_recall,
            "per_class_precision": file_per_class_precision,
            "per_class_counts": dict(file_per_class),
        }

    # Compute overall accuracy
    overall_accuracy = total_correct / total_count if total_count > 0 else 0

    # Compute per-class metrics
    # Accuracy and Recall are the same: correct / total_actual
    per_class_accuracy = {
        cls: data["correct"] / data["total"] if data["total"] > 0 else 0 for cls, data in per_class.items()
    }
    per_class_recall = {
        cls: data["correct"] / data["total"] if data["total"] > 0 else 0 for cls, data in per_class.items()
    }
    per_class_precision = {
        cls: data["correct"] / data["predicted"] if data["predicted"] > 0 else 0 for cls, data in per_class.items()
    }

    results = {
        "overall_accuracy": overall_accuracy,
        "total_predictions": total_count,
        "correct_predictions": total_correct,
        "per_class_accuracy": per_class_accuracy,
        "per_class_recall": per_class_recall,
        "per_class_precision": per_class_precision,
        "per_class_counts": {cls: dict(data) for cls, data in per_class.items()},
        "token_key_used": token_key,
    }

    if include_per_file:
        results["per_file_results"] = per_file_results

    return results


def main():
    parser = argparse.ArgumentParser(description="Compute probe prediction accuracy from trajectory files.")
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing trajectory JSON files",
    )
    parser.add_argument(
        "--token-key",
        type=str,
        default="prompt_suffix_tokens",
        help="Key in step data containing tokens with probes (default: prompt_suffix_tokens)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: _eval_results.json in input directory)",
    )
    parser.add_argument(
        "--per-file-results",
        action="store_true",
        default=False,
        help="Include per-file accuracy results in output",
    )

    args = parser.parse_args()

    if not args.directory.is_dir():
        raise ValueError(f"Not a directory: {args.directory}")

    # Compute accuracy
    results = compute_accuracy(args.directory, args.token_key, args.per_file_results)

    # Determine output path
    output_path = args.output if args.output else args.directory / "_eval_results.json"

    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_path}")
    print(f"\nOverall accuracy: {results['overall_accuracy']:.4f}")
    print(f"Total predictions: {results['total_predictions']}")
    print("\nPer-class metrics:")
    print(f"  {'class':<10} {'accuracy':>10} {'recall':>10} {'precision':>10} {'support':>10}")
    print(f"  {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")
    for cls in sorted(results["per_class_accuracy"].keys()):
        acc = results["per_class_accuracy"][cls]
        recall = results["per_class_recall"][cls]
        precision = results["per_class_precision"][cls]
        counts = results["per_class_counts"][cls]
        print(f"  {cls:<10} {acc:>10.4f} {recall:>10.4f} {precision:>10.4f} {counts['total']:>10}")


if __name__ == "__main__":
    main()
