#!/usr/bin/env python3
"""
Debug script for grid activation extraction.
"""

import pandas as pd
import torch
from nnsight import LanguageModel


def find_token_position_for_grid_cell(grid_text, x, y, tokenizer):
    lines = grid_text.strip().split("\n")

    # Find grid start (skip mission, legend, etc.)
    grid_start = None
    for i, line in enumerate(lines):
        if line.startswith("#"):
            grid_start = i
            break

    if grid_start is None:
        return None

    # Calculate character position for (x, y)
    # First, count characters before the grid starts
    char_position = 0
    for line_idx in range(grid_start):
        char_position += len(lines[line_idx]) + 1  # +1 for newline

    # Now add characters within the grid
    grid_lines = lines[grid_start:]

    # Count characters before the target line within the grid
    for line_idx in range(y):
        if line_idx < len(grid_lines):
            char_position += len(grid_lines[line_idx]) + 1  # +1 for newline

    # Add x position within the target line
    if y < len(grid_lines):
        char_position += x

    # Use a different approach: tokenize and find the token that contains this character
    try:
        # Get token-to-character mapping
        tokens = tokenizer.encode(grid_text)
        token_ranges = []

        # Decode each token to find its character range
        for i, token in enumerate(tokens):
            decoded = tokenizer.decode([token])
            if i == 0:
                start = 0
            else:
                start = token_ranges[i - 1][1] if token_ranges else 0

            end = start + len(decoded)
            token_ranges.append((start, end))

        # Find which token contains our character position
        for i, (start, end) in enumerate(token_ranges):
            if start <= char_position < end:
                return i

        return None
    except Exception:
        return None


def main():
    # Load the data
    df = pd.read_csv("/Users/mokshnirvaan/Desktop/Goal-directed/reveng/reveng/grids_for_probing/grids_for_probing.csv")
    env_data = df[df["env_idx"] == 0]
    grid_text = env_data["observation"].iloc[0]

    # Load model
    model = LanguageModel("microsoft/DialoGPT-medium", device_map="auto")

    print("Testing activation extraction...")
    print(f"Grid text length: {len(grid_text)}")

    # Get activations
    layer_activations = None
    with torch.no_grad():
        with model.trace(grid_text):
            layer_activations = model.transformer.h[12].output[0]

    print(f"Layer activations shape: {layer_activations.shape}")

    # Test a few cells
    cell_activations = {}
    for i in range(10):
        row = env_data.iloc[i]
        x, y = row["x"], row["y"]
        cell_type = row["cell_type"]

        token_pos = find_token_position_for_grid_cell(grid_text, x, y, model.tokenizer)
        print(f"Cell ({x},{y}) - {cell_type}: token_pos = {token_pos}")

        if token_pos is not None and layer_activations is not None and token_pos < len(layer_activations):
            cell_activation = layer_activations[token_pos]
            cell_activations[(x, y)] = cell_activation
            print(f"  -> Extracted activation with shape: {cell_activation.shape}")
        else:
            print("  -> No activation extracted")

    print(f"Total cells with activations: {len(cell_activations)}")


if __name__ == "__main__":
    main()
