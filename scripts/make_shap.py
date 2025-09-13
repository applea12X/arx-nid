"""
Generate SHAP explanations for network flow predictions.

This script creates SHAP force plots and explanations that highlight
which features push a flow toward "attack" classification.

Usage:
    python scripts/make_shap.py [--sample synthetic] [--limit N]
"""

import argparse
import json
import numpy as np
import pandas as pd
import shap
from pathlib import Path
from arx_nid.explain.load_model import ONNXModel


def main():
    parser = argparse.ArgumentParser(description="Generate SHAP explanations")
    parser.add_argument(
        "--sample",
        choices=["normal", "synthetic"],
        default="normal",
        help="Type of sample to explain",
    )
    parser.add_argument(
        "--limit", type=int, default=50, help="Number of samples to explain"
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=200,
        help="Number of background samples for SHAP",
    )
    parser.add_argument(
        "--output-dir", default="reports", help="Output directory for SHAP results"
    )

    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading ONNX model...")
    model = ONNXModel()

    # Load data
    if args.sample == "synthetic":
        print("Loading synthetic data...")
        data_path = Path("data/processed/synthetic_ddos.npy")
        if not data_path.exists():
            print(f"Error: Synthetic data not found at {data_path}")
            print("Run 'python scripts/make_synthetic.py' first")
            return

        X = np.load(data_path)
        # Add batch dimension if needed
        if X.ndim == 2:
            X = X[np.newaxis, :, :]
        output_prefix = "shap_synthetic"
    else:
        print("Loading normal tensor data...")
        data_path = Path("data/processed/tensor_v0.npy")
        if not data_path.exists():
            print(f"Error: Tensor data not found at {data_path}")
            print("Run DVC pipeline to generate tensor data")
            return

        X = np.load(data_path)
        if args.limit and args.limit < len(X):
            X = X[: args.limit]
        output_prefix = f"shap_{args.limit}"

    print(f"Loaded data with shape: {X.shape}")

    # Flatten for SHAP (B, T, F) -> (B, T*F)
    batch_size, time_steps, features = X.shape
    X_flat = X.reshape(batch_size, -1)
    print(f"Flattened data shape: {X_flat.shape}")

    # Create SHAP explainer
    print("Creating SHAP explainer...")

    # Use a subset of data as background for KernelExplainer
    background_size = min(100, len(X_flat))
    background_indices = np.random.choice(len(X_flat), background_size, replace=False)
    background_data = X_flat[background_indices]

    explainer = shap.KernelExplainer(model.predict, background_data)

    # Generate explanations
    explain_size = min(args.limit, len(X_flat))
    X_explain = X_flat[:explain_size]

    print(f"Generating SHAP values for {explain_size} samples...")
    shap_values = explainer.shap_values(X_explain, nsamples=args.nsamples)

    # Get predictions for context
    predictions = model.predict(X_explain)

    # Prepare output data
    output_data = {
        "shap_values": (
            shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values
        ),
        "base_value": float(explainer.expected_value),
        "sample_data": X_explain.tolist(),
        "predictions": predictions.tolist(),
        "feature_dimensions": {
            "time_steps": time_steps,
            "features_per_step": features,
            "total_features": time_steps * features,
        },
        "metadata": {
            "model_type": "onnx",
            "background_samples": background_size,
            "explained_samples": explain_size,
            "nsamples": args.nsamples,
        },
    }

    # Save JSON data
    json_path = output_dir / f"{output_prefix}.json"
    print(f"Saving SHAP data to {json_path}")
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)

    # Generate HTML visualization
    html_path = output_dir / f"{output_prefix}.html"
    print(f"Generating HTML visualization: {html_path}")

    try:
        # Create force plot for the first sample
        force_plot = shap.force_plot(
            explainer.expected_value,
            shap_values[0],
            X_explain[0],
            show=False,
            matplotlib=False,
        )

        # Generate HTML with embedded JavaScript
        shap_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SHAP Force Plot - {output_prefix}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metadata {{ background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                .sample-info {{ background: #e8f4f8; padding: 10px; margin: 10px 0; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>SHAP Explanation - {output_prefix.replace('_', ' ').title()}</h1>
            
            <div class="metadata">
                <h3>Model Information</h3>
                <ul>
                    <li>Base Value (Expected): {explainer.expected_value:.4f}</li>
                    <li>Prediction for Sample 0: {predictions[0]:.4f}</li>
                    <li>Background Samples: {background_size}</li>
                    <li>Explained Samples: {explain_size}</li>
                </ul>
            </div>
            
            <div class="sample-info">
                <h3>Data Shape Information</h3>
                <ul>
                    <li>Time Steps: {time_steps}</li>
                    <li>Features per Time Step: {features}</li>
                    <li>Total Features: {time_steps * features}</li>
                </ul>
                <p><strong>Note:</strong> Features are flattened for SHAP analysis. 
                Features 0-{features-1} correspond to time step 0, 
                features {features}-{2*features-1} to time step 1, etc.</p>
            </div>
            
            <h3>Force Plot (Sample 0)</h3>
            {shap.getjs()}
            {force_plot.html()}
            
            <div class="metadata">
                <h3>Interpretation</h3>
                <p>
                    <strong>Red features</strong> push the prediction toward "attack" (higher probability).<br>
                    <strong>Blue features</strong> push the prediction toward "benign" (lower probability).<br>
                    The final prediction is the sum of all feature contributions plus the base value.
                </p>
            </div>
        </body>
        </html>
        """

        with open(html_path, "w") as f:
            f.write(shap_html)

    except Exception as e:
        print(f"Warning: Could not generate HTML visualization: {e}")
        # Create a simple HTML file with the data
        simple_html = f"""
        <!DOCTYPE html>
        <html>
        <head><title>SHAP Results - {output_prefix}</title></head>
        <body>
            <h1>SHAP Results - {output_prefix}</h1>
            <p>SHAP analysis completed successfully.</p>
            <p>Check the JSON file for detailed results: {json_path.name}</p>
            <h3>Summary</h3>
            <ul>
                <li>Base Value: {explainer.expected_value:.4f}</li>
                <li>Samples Analyzed: {explain_size}</li>
                <li>Prediction for first sample: {predictions[0]:.4f}</li>
            </ul>
        </body>
        </html>
        """
        with open(html_path, "w") as f:
            f.write(simple_html)

    # Print summary
    print("\n✓ SHAP analysis complete!")
    print(f"  Analyzed {explain_size} samples")
    print(f"  Base value (expected): {explainer.expected_value:.4f}")
    print(f"  First sample prediction: {predictions[0]:.4f}")
    print(f"  JSON data: {json_path}")
    print(f"  HTML visualization: {html_path}")

    # Show top contributing features for first sample
    if isinstance(shap_values, np.ndarray) and len(shap_values) > 0:
        abs_shap = np.abs(shap_values[0])
        top_features = np.argsort(abs_shap)[-10:][::-1]  # Top 10 features

        print(f"\n  Top 10 most important features for sample 0:")
        for i, feat_idx in enumerate(top_features):
            contribution = shap_values[0][feat_idx]
            time_step = feat_idx // features
            feat_in_step = feat_idx % features
            print(
                f"    {i+1}. Feature {feat_idx} (t={time_step}, f={feat_in_step}): {contribution:+.4f}"
            )


if __name__ == "__main__":
    main()
