#!/usr/bin/env python3
"""
Visualize side-by-side comparison of true grid state vs decoded grid as PDF.

This script renders grids in the same style as trace_viewer.html using matplotlib,
and outputs to PDF (and optionally PNG).
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


# Color scheme matching trace_viewer.html
TILE_COLORS = {
    "wall": "#666666",
    "empty": "#E8F5E9",
    "agent": "#F44336",
    "goal": "#4CAF50",
    "start": "#2196F3",
    "fog": "#444444",
    "door": "#9C27B0",
    "key": "#FFEB3B",
    "unknown": "#FF9800",
    "padding": "#888888",
}

# Symbol mapping
CLASS_TO_SYMBOL = {
    "agent": "A",
    "goal": "G",
    "wall": "#",
    "empty": "_",
    "door": "D",
    "key": "K",
    "padding": "+",
    "unknown": "?",
    "fog": "*",
}

SYMBOL_TO_CLASS = {v: k for k, v in CLASS_TO_SYMBOL.items()}


def find_probes_in_tokens(tokens: list[dict]) -> dict | None:
    """Find the first token with a 'probes' key in the token list."""
    for token in tokens:
        if "probes" in token:
            return token["probes"]
    return None


def extract_probe_predictions(probes: dict) -> dict[tuple[int, int], tuple[str, float]]:
    """
    Extract predicted class and probability for each cell from probe results.
    
    Returns dict mapping (row, col) -> (predicted_class, max_probability).
    """
    predictions = {}
    cell_pattern = re.compile(r"r(\d+)_c(\d+)$")
    
    for probe_name, probe_data in probes.items():
        match = cell_pattern.search(probe_name)
        if not match:
            continue
        
        row = int(match.group(1))
        col = int(match.group(2))
        
        for layer_key, probs in probe_data.items():
            best_class = None
            best_prob = -1
            for class_name, prob in probs.items():
                if class_name != "padding" and prob > best_prob:
                    best_prob = prob
                    best_class = class_name
            
            if best_class:
                predictions[(row, col)] = (best_class, best_prob)
            break
    
    return predictions


def parse_grid_state(grid_state: list[str]) -> tuple[dict[tuple[int, int], str], int]:
    """
    Parse grid state into cell class mapping and determine grid size.
    
    Returns: (cell_classes dict, grid_size)
    """
    cell_classes = {}
    grid_size = 0
    
    for line in grid_state:
        parts = line.split()
        if not parts:
            continue
        try:
            row = int(parts[0])
            # Check this is a data row (second part should be a symbol)
            if len(parts) > 1 and not parts[1].isdigit():
                grid_size = max(grid_size, len(parts) - 1)
                for col, symbol in enumerate(parts[1:]):
                    if symbol in SYMBOL_TO_CLASS:
                        cell_classes[(row, col)] = SYMBOL_TO_CLASS[symbol]
        except ValueError:
            continue
    
    return cell_classes, grid_size


def draw_grid(
    ax: plt.Axes,
    cell_classes: dict[tuple[int, int], str],
    grid_size: int,
    title: str,
    show_missing_as_x: bool = False,
) -> None:
    """
    Draw a grid on the given matplotlib axes.
    
    Args:
        ax: Matplotlib axes to draw on
        cell_classes: Dict mapping (row, col) -> class_name
        grid_size: Size of the grid (assumes square)
        title: Title for the grid
        show_missing_as_x: If True, show missing cells with X pattern
    """
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # Row 0 at top
    # ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    
    # Set tick positions to center of cells (0.5, 1.5, 2.5, ...)
    ax.set_xticks([i + 0.5 for i in range(grid_size)])
    ax.set_yticks([i + 0.5 for i in range(grid_size)])
    ax.set_xticklabels(range(grid_size), fontsize=11)
    ax.set_yticklabels(range(grid_size), fontsize=11)
    ax.tick_params(length=0)
    
    for row in range(grid_size):
        for col in range(grid_size):
            if (row, col) in cell_classes:
                cell_class = cell_classes[(row, col)]
                color = TILE_COLORS.get(cell_class, "#CCCCCC")
                symbol = CLASS_TO_SYMBOL.get(cell_class, "?")
            elif show_missing_as_x:
                # Missing probe data
                color = "#888888"
                symbol = None
            else:
                # Default for unknown cells
                color = "#CCCCCC"
                symbol = "?"
            
            # Draw cell background
            rect = patches.Rectangle(
                (col, row), 1, 1,
                linewidth=1,
                edgecolor=(0, 0, 0, 0.3),
                facecolor=color,
            )
            ax.add_patch(rect)
            
            # Draw symbol or X
            if symbol and symbol != "_":
                ax.text(
                    col + 0.5, row + 0.5, symbol,
                    ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="white",
                )
            elif show_missing_as_x and (row, col) not in cell_classes:
                # Draw X for missing data
                ax.plot(
                    [col + 0.15, col + 0.85], [row + 0.15, row + 0.85],
                    color=(1, 1, 1, 0.5), linewidth=2,
                )
                ax.plot(
                    [col + 0.85, col + 0.15], [row + 0.15, row + 0.85],
                    color=(1, 1, 1, 0.5), linewidth=2,
                )
    
    # Draw grid lines
    for i in range(grid_size + 1):
        ax.axhline(y=i, color="black", linewidth=0.5, alpha=0.3)
        ax.axvline(x=i, color="black", linewidth=0.5, alpha=0.3)


def create_visualization(
    trajectory_path: Path,
    step: int = 0,
    token_key: str = "prompt_suffix_tokens",
    output_path: Path | None = None,
    show_plot: bool = False,
) -> Path:
    """
    Create PDF visualization of true vs decoded grid.
    
    Args:
        trajectory_path: Path to trajectory JSON file
        step: Step index to visualize
        token_key: Key in step data containing tokens with probes
        output_path: Output PDF path (default: same as input with _step{N}.pdf suffix)
        show_plot: Whether to display the plot interactively
        
    Returns:
        Path to the generated PDF file
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
    
    # Parse true grid
    true_cell_classes, grid_size = parse_grid_state(grid_state)
    
    # Find probes and extract predictions
    probes = find_probes_in_tokens(tokens)
    if not probes:
        raise ValueError(f"No probes found in {token_key} for step {step}")
    
    predictions = extract_probe_predictions(probes)
    
    # Convert predictions to cell classes format
    decoded_cell_classes = {pos: pred[0] for pos, pred in predictions.items()}
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # fig.suptitle(
    #     f"{trajectory_path.name}\nStep {step}",
    #     fontsize=12,
    #     y=0.98,
    # )
    
    # Draw true grid
    draw_grid(ax1, true_cell_classes, grid_size, "True Grid")
    
    # Draw decoded grid
    draw_grid(ax2, decoded_cell_classes, grid_size, "Decoded Grid", show_missing_as_x=True)
    
    # # Add legend (commented out for now)
    # legend_elements = []
    # for class_name in ["wall", "empty", "agent", "goal"]:
    #     color = TILE_COLORS.get(class_name, "#CCCCCC")
    #     symbol = CLASS_TO_SYMBOL.get(class_name, "?")
    #     legend_elements.append(
    #         patches.Patch(
    #             facecolor=color,
    #             edgecolor="black",
    #             label=f"{symbol} = {class_name}",
    #         )
    #     )
    # 
    # fig.legend(
    #     handles=legend_elements,
    #     loc="lower center",
    #     ncol=len(legend_elements),
    #     fontsize=10,
    #     frameon=True,
    #     bbox_to_anchor=(0.5, 0.02),
    # )
    
    plt.tight_layout()
    
    # Determine output path
    if output_path is None:
        output_path = trajectory_path.parent / f"{trajectory_path.stem}_step{step}.pdf"
    
    # Save to PDF
    fig.savefig(output_path, format="pdf", bbox_inches="tight", dpi=150)
    
    # Also save PNG
    png_path = output_path.with_suffix(".png")
    fig.savefig(png_path, format="png", bbox_inches="tight", dpi=150)
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Visualize true vs decoded grid as PDF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python visualize_decoded_grid_pdf.py path/to/trajectory.json
  python visualize_decoded_grid_pdf.py path/to/trajectory.json --step 5
  python visualize_decoded_grid_pdf.py path/to/trajectory.json --output my_visualization.pdf
  python visualize_decoded_grid_pdf.py path/to/trajectory.json --show
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
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output PDF path (default: <trajectory>_step<N>.pdf)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively",
    )
    
    args = parser.parse_args()
    
    if not args.trajectory.exists():
        raise FileNotFoundError(f"Trajectory file not found: {args.trajectory}")
    
    output_path = create_visualization(
        args.trajectory,
        args.step,
        args.token_key,
        args.output,
        args.show,
    )
    
    print(f"PDF saved to: {output_path}")
    print(f"PNG saved to: {output_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()

