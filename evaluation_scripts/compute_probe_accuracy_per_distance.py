#!/usr/bin/env python3
"""
Compute probe-prediction metrics bucketed by distance to goal / agent.

A fork of ``compute_probe_accuracy.py``. In addition to the overall and
per-class metrics, it reports the same metrics (accuracy, recall, precision, F1)
bucketed by each cell's Chebyshev ("square-ring") distance to the goal and,
separately, to the agent. Distance of cell (r, c) from a point (pr, pc) is
``max(|r - pr|, |c - pc|)``.

Everything is aggregated globally, per grid size, and per difficulty. Results
are saved to ``_eval_results_per_distance.json`` in the input directory.

Like the original, this is pure post-processing: it reads probe predictions
already stored in each trajectory token (``token["probes"]``, written upstream by
``apply_cognitive_map_probe``). No model or probe is loaded here.
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
        for _layer_key, probs in probe_data.items():
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


def chebyshev_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Square-ring (Chebyshev) distance: max(|dr|, |dc|)."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def find_position(ground_truth: dict[tuple[int, int], str], class_name: str) -> tuple[int, int] | None:
    """Return the (row, col) of the (first) cell with the given class, or None."""
    for pos, cls in ground_truth.items():
        if cls == class_name:
            return pos
    return None


# ---------------------------------------------------------------------------
# Accumulators
# ---------------------------------------------------------------------------


def _new_class_counters():
    # total = actual instances of class (recall denom)
    # predicted = times class was predicted (precision denom)
    return {"correct": 0, "total": 0, "predicted": 0}


def _new_dist_entry() -> dict:
    """Counters for a single distance ring: overall + per-class."""
    return {
        "correct": 0,
        "total": 0,
        "per_class": defaultdict(_new_class_counters),
    }


def new_bucket() -> dict:
    """An accumulator for one aggregation cell (global / a size / a difficulty)."""
    return {
        "correct": 0,
        "total": 0,
        "per_class": defaultdict(_new_class_counters),
        "dist_goal": defaultdict(_new_dist_entry),
        "dist_agent": defaultdict(_new_dist_entry),
    }


def _update_counters(per_class: dict, true_cls: str, pred_cls: str) -> int:
    """Update overall/per-class counters for one cell. Returns 1 if correct else 0."""
    per_class[true_cls]["total"] += 1
    per_class[pred_cls]["predicted"] += 1
    if pred_cls == true_cls:
        per_class[true_cls]["correct"] += 1
        return 1
    return 0


def update_bucket(bucket: dict, true_cls: str, pred_cls: str, d_goal: int, d_agent: int) -> None:
    """Record one cell prediction into overall, per-class and both distance maps."""
    correct = _update_counters(bucket["per_class"], true_cls, pred_cls)
    bucket["correct"] += correct
    bucket["total"] += 1

    for dist_map, dist in (("dist_goal", d_goal), ("dist_agent", d_agent)):
        entry = bucket[dist_map][dist]
        entry_correct = _update_counters(entry["per_class"], true_cls, pred_cls)
        entry["correct"] += entry_correct
        entry["total"] += 1


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def metrics_from_counters(correct: int, total: int, per_class: dict) -> dict:
    """Compute overall + per-class accuracy/recall/precision/f1 from counters."""
    overall_accuracy = correct / total if total > 0 else 0

    per_class_accuracy = {}
    per_class_recall = {}
    per_class_precision = {}
    per_class_f1 = {}
    per_class_counts = {}

    for cls, data in per_class.items():
        recall = data["correct"] / data["total"] if data["total"] > 0 else 0
        precision = data["correct"] / data["predicted"] if data["predicted"] > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        # Accuracy and recall coincide here: correct / actual_instances.
        per_class_accuracy[cls] = recall
        per_class_recall[cls] = recall
        per_class_precision[cls] = precision
        per_class_f1[cls] = f1
        per_class_counts[cls] = dict(data)

    return {
        "overall_accuracy": overall_accuracy,
        "total_predictions": total,
        "correct_predictions": correct,
        "per_class_accuracy": per_class_accuracy,
        "per_class_recall": per_class_recall,
        "per_class_precision": per_class_precision,
        "per_class_f1": per_class_f1,
        "per_class_counts": per_class_counts,
    }


def _summarize_dist_map(dist_map: dict) -> dict:
    """Summarize a distance->entry map into sorted string-keyed metric blocks."""
    out = {}
    for dist in sorted(dist_map.keys()):
        entry = dist_map[dist]
        out[str(dist)] = metrics_from_counters(entry["correct"], entry["total"], entry["per_class"])
    return out


def summarize_bucket(bucket: dict) -> dict:
    """Full metric block for one aggregation cell, including per-distance breakdowns."""
    summary = metrics_from_counters(bucket["correct"], bucket["total"], bucket["per_class"])
    summary["per_distance_to_goal"] = _summarize_dist_map(bucket["dist_goal"])
    summary["per_distance_to_agent"] = _summarize_dist_map(bucket["dist_agent"])
    return summary


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------


def compute_metrics(
    trajectories_dir: Path,
    token_key: str = "prompt_suffix_tokens",
) -> dict:
    """
    Compute distance-bucketed metrics across all trajectories.

    Args:
        trajectories_dir: Directory containing trajectory JSON files
        token_key: Key in step data containing tokens with probes

    Returns:
        Dictionary with global, per-size and per-difficulty metric blocks.
    """
    global_bucket = new_bucket()
    by_size: dict[int, dict] = defaultdict(new_bucket)
    by_difficulty: dict[float, dict] = defaultdict(new_bucket)

    json_files = sorted(trajectories_dir.glob("*.json"))
    json_files = [f for f in json_files if not f.name.startswith("_")]  # Skip result files

    if not json_files:
        raise ValueError(f"No JSON files found in {trajectories_dir}")

    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)

        grid_params = data.get("grid_params", {})
        size = grid_params.get("grid_width")
        difficulty = grid_params.get("grid_complexity")

        for step in data.get("steps", []):
            grid_state = step.get("grid_state")
            tokens = step.get(token_key)

            if not grid_state or not tokens:
                continue

            ground_truth = parse_grid_state(grid_state)

            probes = find_probes_in_tokens(tokens)
            if not probes:
                continue

            predictions = extract_probe_predictions(probes)

            # Agent moves per step; goal is fixed. Derive both from this step's grid.
            goal_pos = find_position(ground_truth, "goal")
            agent_pos = find_position(ground_truth, "agent")
            if goal_pos is None or agent_pos is None:
                continue

            buckets = [global_bucket]
            if size is not None:
                buckets.append(by_size[size])
            if difficulty is not None:
                buckets.append(by_difficulty[difficulty])

            for (row, col), true_class in ground_truth.items():
                if (row, col) not in predictions:
                    continue

                pred_class = predictions[(row, col)]
                d_goal = chebyshev_distance((row, col), goal_pos)
                d_agent = chebyshev_distance((row, col), agent_pos)

                for bucket in buckets:
                    update_bucket(bucket, true_class, pred_class, d_goal, d_agent)

    results = {
        "global": summarize_bucket(global_bucket),
        "per_size": {str(size): summarize_bucket(b) for size, b in sorted(by_size.items())},
        "per_difficulty": {
            str(diff): summarize_bucket(b) for diff, b in sorted(by_difficulty.items())
        },
        "token_key_used": token_key,
    }
    return results


def _print_per_class_table(block: dict) -> None:
    print(f"  {'class':<10} {'accuracy':>10} {'recall':>10} {'precision':>10} {'f1':>10} {'support':>10}")
    print(f"  {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")
    for cls in sorted(block["per_class_accuracy"].keys()):
        acc = block["per_class_accuracy"][cls]
        recall = block["per_class_recall"][cls]
        precision = block["per_class_precision"][cls]
        f1 = block["per_class_f1"][cls]
        support = block["per_class_counts"][cls]["total"]
        print(f"  {cls:<10} {acc:>10.4f} {recall:>10.4f} {precision:>10.4f} {f1:>10.4f} {support:>10}")


def _print_distance_curve(label: str, dist_blocks: dict) -> None:
    print(f"\n{label}:")
    print(f"  {'distance':>10} {'accuracy':>10} {'support':>10}")
    print(f"  {'-' * 10} {'-' * 10} {'-' * 10}")
    for dist in sorted(dist_blocks.keys(), key=int):
        block = dist_blocks[dist]
        print(f"  {dist:>10} {block['overall_accuracy']:>10.4f} {block['total_predictions']:>10}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute probe-prediction metrics bucketed by distance to goal/agent."
    )
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
        help="Output file path (default: _eval_results_per_distance.json in input directory)",
    )

    args = parser.parse_args()

    if not args.directory.is_dir():
        raise ValueError(f"Not a directory: {args.directory}")

    results = compute_metrics(args.directory, args.token_key)

    output_path = args.output if args.output else args.directory / "_eval_results_per_distance.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_path}")

    global_block = results["global"]
    print(f"\nOverall accuracy: {global_block['overall_accuracy']:.4f}")
    print(f"Total predictions: {global_block['total_predictions']}")
    print("\nGlobal per-class metrics:")
    _print_per_class_table(global_block)

    _print_distance_curve("Accuracy per distance to goal", global_block["per_distance_to_goal"])
    _print_distance_curve("Accuracy per distance to agent", global_block["per_distance_to_agent"])


if __name__ == "__main__":
    main()
