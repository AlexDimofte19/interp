#!/usr/bin/env python3
"""
Test script for gathering activations from grid CSV data.
"""

import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from telos_interp.activations import TokenPosition, gather_activations_from_grid_csv  # noqa: E402


def main():
    """Test the grid activation gathering functionality."""

    # Path to the grid CSV file
    grid_csv_path = "/Users/mokshnirvaan/Desktop/Goal-directed/reveng/reveng/grids_for_probing/grids_for_probing.csv"

    # Check if the file exists
    if not os.path.exists(grid_csv_path):
        print(f"Error: Grid CSV file not found at {grid_csv_path}")
        print("Please make sure you've generated the grid data first using the reveng script.")
        return

    print("🚀 Starting grid activation gathering...")
    print(f"📁 Input CSV: {grid_csv_path}")

    # Gather activations
    try:
        results_path = gather_activations_from_grid_csv(
            model_name_or_path="microsoft/DialoGPT-medium",  # Using a smaller model for testing
            csv_path=grid_csv_path,
            token_position=TokenPosition.prompt_last,
            layer=12,
        )

        print(f"✅ Success! Activations saved to: {results_path}")

        # List the generated files
        if os.path.exists(results_path):
            print("\n📋 Generated files:")
            for file in os.listdir(results_path):
                if file.endswith(".pt"):
                    file_path = os.path.join(results_path, file)
                    file_size = os.path.getsize(file_path)
                    print(f"  - {file} ({file_size:,} bytes)")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
