#!/usr/bin/env python3
"""
Visualize side-by-side comparison of true grid state vs decoded grid as PDF.

Uses the plotting style from reveng/src/reveng/environment_generator/env_plots.py
for consistent visualization with the environment generator.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


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

# Color scheme: black empty, gray walls, red triangle agent, green goal
TILE_COLORS = {
    "empty": "#000000",      # Black
    "wall": "#808080",       # Gray
    "agent": "#000000",      # Black background (red triangle drawn on top)
    "goal": "#00FF00",       # Bright green
    "door": "#8B4513",       # Brown
    "key": "#FFD700",        # Gold
    "unknown": "#444444",    # Dark gray
    "padding": "#333333",    # Dark gray
    "fog": "#222222",        # Very dark
}


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


def draw_agent_triangle(ax: plt.Axes, x: float, y: float, width: float = 0.5, height: float = 0.35) -> None:
    """
    Draw a red triangle pointing right (agent) at the given position.
    
    Args:
        ax: Matplotlib axes
        x: Center x position
        y: Center y position  
        width: Width of the triangle (horizontal extent)
        height: Height of the triangle (vertical extent)
    """
    # Triangle pointing right (stretched horizontally)
    triangle_points = np.array([
        [x - width * 0.5, y - height * 0.5],  # Bottom left
        [x - width * 0.5, y + height * 0.5],  # Top left
        [x + width * 0.5, y],                  # Right point
    ])
    triangle = patches.Polygon(triangle_points, closed=True, facecolor="#FF0000", edgecolor="#FF0000", linewidth=1)
    ax.add_patch(triangle)


def draw_grid_reveng_style(
    ax: plt.Axes,
    cell_classes: dict[tuple[int, int], str],
    grid_size: int,
    show_missing_as_x: bool = False,
    show_numbers: bool = False,
    title: str | None = None,
) -> None:
    """
    Draw a grid with colored tiles on the given matplotlib axes.
    
    Colors:
    - empty: black
    - wall: gray
    - agent: red triangle on black background
    - goal: bright green square
    
    Args:
        ax: Matplotlib axes to draw on
        cell_classes: Dict mapping (row, col) -> class_name
        grid_size: Size of the grid (assumes square)
        show_missing_as_x: If True, show missing cells with X pattern
        show_numbers: If True, show row and column numbers
        title: Optional title to display above the grid
    """
    num_rows = grid_size
    num_cols = grid_size
    
    # Adjust limits for number labels if needed
    if show_numbers:
        ax.set_xlim(-1.0, num_cols - 0.5)
        ax.set_ylim(-0.5, num_rows + 0.5)
    else:
        ax.set_xlim(-0.5, num_cols - 0.5)
        ax.set_ylim(-0.5, num_rows - 0.5)
    ax.set_aspect("equal")
    
    # Add title if provided
    if title:
        ax.set_title(title, fontsize=30, fontweight="bold", pad=1, y=0.98)
    
    # Draw each cell
    for row in range(num_rows):
        for col in range(num_cols):
            # Flipped y position (row 0 at top)
            y_pos = num_rows - 1 - row
            
            if (row, col) in cell_classes:
                cell_class = cell_classes[(row, col)]
                color = TILE_COLORS.get(cell_class, "#444444")
            elif show_missing_as_x:
                cell_class = "unknown"
                color = "#444444"
            else:
                cell_class = "unknown"
                color = "#444444"
            
            # Draw cell background
            rect = patches.Rectangle(
                (col - 0.5, y_pos - 0.5), 1, 1,
                linewidth=1.0,
                edgecolor="#808080",  # Same color as walls
                facecolor=color,
            )
            ax.add_patch(rect)
            
            # Draw special elements
            if cell_class == "agent":
                # Draw red triangle on black background (stretched horizontally)
                draw_agent_triangle(ax, col, y_pos, width=0.65, height=0.55)
            elif show_missing_as_x and (row, col) not in cell_classes:
                # Draw X for missing data
                ax.plot(
                    [col - 0.3, col + 0.3], [y_pos - 0.3, y_pos + 0.3],
                    color=(1, 1, 1, 0.5), linewidth=2,
                )
                ax.plot(
                    [col + 0.3, col - 0.3], [y_pos - 0.3, y_pos + 0.3],
                    color=(1, 1, 1, 0.5), linewidth=2,
                )
    
    # Draw row and column numbers if requested
    if show_numbers:
        for col in range(num_cols):
            # Column numbers at top (slightly closer to grid)
            ax.text(
                col, num_rows - 0.15,
                str(col),
                ha="center", va="center",
                fontsize=20, fontweight="bold",
                color="black",
            )
        for row in range(num_rows):
            # Row numbers on left (flipped y position)
            y_pos = num_rows - 1 - row
            ax.text(
                -0.85, y_pos,
                str(row),
                ha="center", va="center",
                fontsize=20, fontweight="bold",
                color="black",
            )
    
    # Remove ticks and spines
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)


def create_visualization(
    trajectory_path: Path,
    step: int = 0,
    token_key: str = "prompt_suffix_tokens",
    output_path: Path | None = None,
    show_plot: bool = False,
    show_numbers: bool = False,
) -> Path:
    """
    Create PDF visualization of true vs decoded grid using reveng style.
    
    Args:
        trajectory_path: Path to trajectory JSON file
        step: Step index to visualize
        token_key: Key in step data containing tokens with probes
        output_path: Output PDF path (default: same as input with _step{N}.pdf suffix)
        show_plot: Whether to display the plot interactively
        show_numbers: Whether to show row and column numbers
        
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
    
    # Create figure with two subplots (reveng style: 0.6 per cell)
    cell_size = 0.6
    fig_width = grid_size * cell_size * 2 + 2  # Two grids side by side with gap
    fig_height = grid_size * cell_size + 1
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
    
    # Draw true grid
    draw_grid_reveng_style(
        ax1, true_cell_classes, grid_size,
        show_numbers=show_numbers, title="True Grid"
    )
    
    # Draw decoded grid
    draw_grid_reveng_style(
        ax2, decoded_cell_classes, grid_size,
        show_missing_as_x=True, show_numbers=show_numbers, title="Decoded Grid"
    )
    
    plt.tight_layout()
    
    # Determine output path
    if output_path is None:
        output_path = trajectory_path.parent / f"{trajectory_path.stem}_step{step}_reveng.pdf"
    
    # Save to PDF (reveng style: dpi=300, bbox_inches="tight", pad_inches=0)
    fig.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0)
    
    # Also save PNG
    png_path = output_path.with_suffix(".png")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight", pad_inches=0)
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Visualize true vs decoded grid as PDF (reveng style).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python reveng_visualization.py path/to/trajectory.json
  python reveng_visualization.py path/to/trajectory.json --step 5
  python reveng_visualization.py path/to/trajectory.json --output my_visualization.pdf
  python reveng_visualization.py path/to/trajectory.json --show
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
        help="Output PDF path (default: <trajectory>_step<N>_reveng.pdf)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively",
    )
    parser.add_argument(
        "--show-numbers",
        action="store_true",
        help="Show row and column numbers on grids",
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
        args.show_numbers,
    )
    
    print(f"PDF saved to: {output_path}")
    print(f"PNG saved to: {output_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()

