"""
Generate LaTeX tables for distance probe evaluation results.

This script creates two tables:
1. Layer 15 only - comparing MLP and Linear probes on pre/post reasoning
2. All layers (7, 15, 23) - comprehensive comparison
"""

import json
from pathlib import Path

# Base path for results
BASE_PATH = Path(__file__).parent.parent / "data" / "distance_probe_results"

# Define paths to all JSON files
JSON_PATHS = {
    7: {
        "mlp": {
            "pre_reasoning": BASE_PATH / "layer7/mlp/pre_reasoning/eval_distance_probe_layer7_mlp_pre_reasoning.json",
            "post_reasoning": BASE_PATH / "layer7/mlp/post_reasoning/eval_distance_probe_layer7_mlp_post_reasoning.json",
        },
        "lr": {
            "pre_reasoning": BASE_PATH / "layer7/lr/pre_reasoning/eval_distance_probe_layer7_lr_pre_reasoning.json",
            "post_reasoning": BASE_PATH / "layer7/lr/post_reasoning/eval_distance_probe_layer7_lr_post_reasoning.json",
        },
    },
    15: {
        "mlp": {
            "pre_reasoning": BASE_PATH / "layer15/mlp/pre_reasoning/eval_distance_probe_layer15_mlp_pre_reasoning.json",
            "post_reasoning": BASE_PATH / "layer15/mlp/post_reasoning/eval_distance_probe_layer15_mlp_post_reasoning.json",
        },
        "lr": {
            "pre_reasoning": BASE_PATH / "layer15/lr/pre_reasoning/eval_distance_probe_layer15_lr_pre_reasoning.json",
            "post_reasoning": BASE_PATH / "layer15/lr/post_reasoning/eval_distance_probe_layer15_lr_post_reasoning.json",
        },
    },
    23: {
        "mlp": {
            "pre_reasoning": BASE_PATH / "layer23/mlp/pre_reasoning/eval_distance_probe_layer23_mlp_pre_reasoning.json",
            "post_reasoning": BASE_PATH / "layer23/mlp/post_reasoning/eval_distance_probe_layer23_mlp_post_reasoning.json",
        },
        "lr": {
            "pre_reasoning": BASE_PATH / "layer23/lr/pre_reasoning/eval_distance_probe_layer23_lr_pre_reasoning.json",
            "post_reasoning": BASE_PATH / "layer23/lr/post_reasoning/eval_distance_probe_layer23_lr_post_reasoning.json",
        },
    },
}


def load_results() -> dict:
    """Load results from JSON files."""
    results = {}
    for layer, probe_types in JSON_PATHS.items():
        results[layer] = {}
        for probe_type, stages in probe_types.items():
            results[layer][probe_type] = {}
            for stage, path in stages.items():
                with open(path) as f:
                    data = json.load(f)
                results[layer][probe_type][stage] = {
                    "mae": data["global"]["mae"],
                    "r2": data["global"]["r2"],
                }
    return results


def format_value(value: float, is_best: bool, decimals: int = 3) -> str:
    """Format a value, making it bold if it's the best."""
    formatted = f"{value:.{decimals}f}"
    if is_best:
        return f"\\textbf{{{formatted}}}"
    return formatted


def generate_layer15_table(results: dict) -> str:
    """Generate LaTeX table for Layer 15 only."""
    # Collect all values to find best ones
    mae_values = []
    r2_values = []

    for probe_type in ["mlp", "lr"]:
        for stage in ["pre_reasoning", "post_reasoning"]:
            mae_values.append(results[15][probe_type][stage]["mae"])
            r2_values.append(results[15][probe_type][stage]["r2"])

    min_mae = min(mae_values)
    max_r2 = max(r2_values)

    # Build table
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Distance Probe Performance at Layer 15}",
        r"\label{tab:distance_probes_layer15}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Probe Type & Reasoning Stage & MAE $\downarrow$ & $R^2$ $\uparrow$ \\",
        r"\midrule",
    ]

    probe_names = {"mlp": "MLP Probe", "lr": "Linear Probe"}
    stage_names = {"pre_reasoning": "Pre-Reasoning", "post_reasoning": "Post-Reasoning"}

    for probe_type in ["mlp", "lr"]:
        for i, stage in enumerate(["pre_reasoning", "post_reasoning"]):
            mae = results[15][probe_type][stage]["mae"]
            r2 = results[15][probe_type][stage]["r2"]

            mae_str = format_value(mae, mae == min_mae, decimals=2)
            r2_str = format_value(r2, r2 == max_r2, decimals=2)

            # Use multirow for probe type name
            if i == 0:
                probe_name = probe_names[probe_type]
            else:
                probe_name = ""

            lines.append(f"{probe_name} & {stage_names[stage]} & {mae_str} & {r2_str} \\\\")

        if probe_type == "mlp":
            lines.append(r"\midrule")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def generate_all_layers_table(results: dict) -> str:
    """Generate LaTeX table for all layers (7, 15, 23)."""
    # Collect all values to find best ones
    mae_values = []
    r2_values = []

    for layer in [7, 15, 23]:
        for probe_type in ["mlp", "lr"]:
            for stage in ["pre_reasoning", "post_reasoning"]:
                mae_values.append(results[layer][probe_type][stage]["mae"])
                r2_values.append(results[layer][probe_type][stage]["r2"])

    min_mae = min(mae_values)
    max_r2 = max(r2_values)

    # Build table
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Distance Probe Performance Across Layers}",
        r"\label{tab:distance_probes_all_layers}",
        r"\begin{tabular}{cllcc}",
        r"\toprule",
        r"Layer & Probe Type & Reasoning Stage & MAE $\downarrow$ & $R^2$ $\uparrow$ \\",
        r"\midrule",
    ]

    probe_names = {"mlp": "MLP Probe", "lr": "Linear Probe"}
    stage_names = {"pre_reasoning": "Pre-Reasoning", "post_reasoning": "Post-Reasoning"}

    for layer in [7, 15, 23]:
        first_in_layer = True
        for probe_type in ["mlp", "lr"]:
            for i, stage in enumerate(["pre_reasoning", "post_reasoning"]):
                mae = results[layer][probe_type][stage]["mae"]
                r2 = results[layer][probe_type][stage]["r2"]

                mae_str = format_value(mae, mae == min_mae, decimals=2)
                r2_str = format_value(r2, r2 == max_r2, decimals=2)

                # Show layer only for first row in group
                if first_in_layer:
                    layer_str = str(layer)
                    first_in_layer = False
                else:
                    layer_str = ""

                # Show probe name only for first row in probe group
                if i == 0:
                    probe_name = probe_names[probe_type]
                else:
                    probe_name = ""

                lines.append(
                    f"{layer_str} & {probe_name} & {stage_names[stage]} & {mae_str} & {r2_str} \\\\"
                )

        if layer != 23:
            lines.append(r"\midrule")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def main():
    results = load_results()

    print("=" * 60)
    print("TABLE 1: Layer 15 Only")
    print("=" * 60)
    print(generate_layer15_table(results))
    print()
    print("=" * 60)
    print("TABLE 2: All Layers (7, 15, 23)")
    print("=" * 60)
    print(generate_all_layers_table(results))


if __name__ == "__main__":
    main()
