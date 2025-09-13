"""
Generate Integrated Gradients explanations for temporal attribution.

This script creates Integrated Gradients heat-maps that show which
time-steps (packets) drove the model's decision.

Usage:
    python scripts/make_ig.py [--samples N] [--steps N]
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from captum.attr import IntegratedGradients
from arx_nid.explain.load_model import create_pytorch_model_wrapper


def main():
    parser = argparse.ArgumentParser(
        description="Generate Integrated Gradients explanations"
    )
    parser.add_argument(
        "--samples", type=int, default=1, help="Number of samples to analyze"
    )
    parser.add_argument(
        "--steps", type=int, default=50, help="Number of integration steps"
    )
    parser.add_argument(
        "--output-dir", default="reports", help="Output directory for IG results"
    )
    parser.add_argument(
        "--data-path",
        default="data/processed/tensor_v0.npy",
        help="Path to tensor data",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load PyTorch model
    print("Loading PyTorch model...")
    model = create_pytorch_model_wrapper()

    # Load data
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: Data not found at {data_path}")
        return

    print(f"Loading data from {data_path}")
    X = np.load(data_path)

    # Limit samples
    if args.samples < len(X):
        X = X[: args.samples]

    print(f"Loaded data shape: {X.shape}")

    # Convert to PyTorch tensor
    X_tensor = torch.from_numpy(X).float()

    # Initialize Integrated Gradients
    print("Initializing Integrated Gradients...")
    ig = IntegratedGradients(model)

    # Generate attributions
    print(
        f"Computing attributions for {args.samples} samples with {args.steps} steps..."
    )

    # Use zero baseline (neutral input)
    baseline = torch.zeros_like(X_tensor)

    # Compute attributions (no target for regression-like outputs)
    attributions = ig.attribute(
        X_tensor, baseline, n_steps=args.steps, return_convergence_delta=False
    )

    # Convert back to numpy
    attributions_np = attributions.detach().numpy()

    # Get model predictions for context
    with torch.no_grad():
        predictions = model(X_tensor).numpy()

    print(f"Attribution tensor shape: {attributions_np.shape}")

    # Save raw attributions
    attr_path = output_dir / "ig_attr.npy"
    print(f"Saving attributions to {attr_path}")
    np.save(attr_path, attributions_np)

    # Create metadata
    metadata = {
        "shape": attributions_np.shape,
        "samples_analyzed": args.samples,
        "integration_steps": args.steps,
        "predictions": predictions.tolist(),
        "data_source": str(data_path),
        "attribution_range": {
            "min": float(attributions_np.min()),
            "max": float(attributions_np.max()),
            "mean": float(attributions_np.mean()),
            "std": float(attributions_np.std()),
        },
    }

    # Save metadata
    metadata_path = output_dir / "ig_attr_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Compute summary statistics
    print("\n✓ Integrated Gradients analysis complete!")
    print(f"  Samples analyzed: {args.samples}")
    print(f"  Integration steps: {args.steps}")
    print(
        f"  Attribution range: [{attributions_np.min():.6f}, {attributions_np.max():.6f}]"
    )
    print(f"  Mean absolute attribution: {np.abs(attributions_np.mean()):.6f}")
    print(f"  Saved to: {attr_path}")

    # Show sample-level statistics
    for i in range(min(3, args.samples)):
        sample_attr = attributions_np[i]
        temporal_importance = np.sum(np.abs(sample_attr), axis=1)  # Sum over features
        most_important_timestep = np.argmax(temporal_importance)

        print(f"\n  Sample {i}:")
        print(f"    Prediction: {predictions[i]:.4f}")
        print(f"    Most important timestep: {most_important_timestep}")
        print(f"    Temporal importance: {temporal_importance}")

    return attributions_np, metadata


if __name__ == "__main__":
    main()
