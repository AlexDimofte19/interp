#!/usr/bin/env python3
"""
Example script showing how to use the grid activation gathering functionality
for probing experiments.
"""

import os
import sys
from pathlib import Path

import torch

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from telos_interp.activations import TokenPosition, gather_activations_from_grid_csv  # noqa: E402
from telos_interp.probing import ProbingClassifier  # noqa: E402


def main():
    """Complete example of grid probing workflow."""

    print("🎯 Grid Probing Example")
    print("=" * 50)

    # Step 1: Gather activations from grid CSV
    print("\n📊 Step 1: Gathering activations from grid data...")

    grid_csv_path = "/Users/mokshnirvaan/Desktop/Goal-directed/reveng/reveng/grids_for_probing/grids_for_probing.csv"

    if not os.path.exists(grid_csv_path):
        print(f"❌ Grid CSV not found at {grid_csv_path}")
        print("Please run the reveng script first to generate grid data.")
        return

    # Gather activations for different layers
    layers_to_test = [6, 12, 18]  # Test different layers

    for layer in layers_to_test:
        print(f"\n🔍 Processing layer {layer}...")

        results_path = gather_activations_from_grid_csv(
            model_name_or_path="microsoft/DialoGPT-medium",
            csv_path=grid_csv_path,
            token_position=TokenPosition.prompt_last,
            layer=layer,
        )

        print(f"✅ Layer {layer} activations saved to: {results_path}")

        # Step 2: Train probes on the gathered activations
        print(f"\n🧠 Step 2: Training probes for layer {layer}...")

        # Load the activations
        activations_dir = results_path

        # Load activations for each cell type
        cell_types = ["wall", "empty", "agent", "goal"]
        activations_data = {}

        for cell_type in cell_types:
            file_path = os.path.join(activations_dir, f"acts_{cell_type}.pt")
            if os.path.exists(file_path):
                activations_data[cell_type] = torch.load(file_path)
                print(f"  Loaded {len(activations_data[cell_type])} {cell_type} activations")
            else:
                print(f"  ⚠️  No {cell_type} activations found")

        # Step 3: Train probes for each cell type
        print(f"\n🎯 Step 3: Training probes for layer {layer}...")

        probes = {}

        for target_cell_type in cell_types:
            if target_cell_type not in activations_data:
                continue

            print(f"\n  Training {target_cell_type} probe...")

            # Prepare positive and negative examples
            positive_acts = activations_data[target_cell_type]

            # Get negative examples (all other cell types)
            negative_acts = []
            for other_type, acts in activations_data.items():
                if other_type != target_cell_type:
                    negative_acts.append(acts)

            if negative_acts:
                negative_acts = torch.cat(negative_acts, dim=0)

                # Train the probe
                probe = ProbingClassifier(reg_coeff=1e3, normalize=True)
                probe.fit(positive_acts, negative_acts)
                probes[target_cell_type] = probe

                print(f"    ✅ Trained on {len(positive_acts)} positive, {len(negative_acts)} negative examples")
            else:
                print(f"    ❌ No negative examples found for {target_cell_type}")

        # Step 4: Test the probes
        print(f"\n🧪 Step 4: Testing probes for layer {layer}...")

        for cell_type, probe in probes.items():
            # Test on the same data (in practice, you'd use held-out data)
            test_acts = activations_data[cell_type]

            # Get predictions
            scores = probe.score_tokens(test_acts)
            predictions = (scores > 0).float()

            # Calculate accuracy
            accuracy = predictions.mean().item()
            print(f"  {cell_type} probe accuracy: {accuracy:.3f}")

        print(f"\n✅ Layer {layer} probing complete!")
        print("-" * 50)


if __name__ == "__main__":
    main()
