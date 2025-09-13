"""
Convert Integrated Gradients attributions to HTML heatmap visualization.

This script takes IG attribution data and creates an HTML heatmap
that's easier for SOC analysts to interpret than raw numbers.

Usage:
    python scripts/ig_to_html.py [--sample N]
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import base64
import io


def create_heatmap_html(attributions, metadata, sample_idx=0, output_path=None):
    """
    Create an HTML heatmap visualization of IG attributions.

    Args:
        attributions: IG attribution array of shape (samples, time, features)
        metadata: Metadata dictionary
        sample_idx: Which sample to visualize
        output_path: Where to save the HTML file
    """
    if sample_idx >= attributions.shape[0]:
        raise ValueError(
            f"Sample {sample_idx} not available (only {attributions.shape[0]} samples)"
        )

    # Get the attribution data for this sample
    sample_attr = attributions[sample_idx]  # Shape: (time, features)

    # Create the heatmap
    plt.figure(figsize=(12, 6))

    # Use a diverging colormap centered at 0
    ax = sns.heatmap(
        sample_attr,
        cmap="coolwarm",
        center=0,
        cbar_kws={"label": "Attribution Score"},
        xticklabels=False,  # Too many features to show labels
        yticklabels=True,
    )

    ax.set_xlabel("Features")
    ax.set_ylabel("Time (packet index)")
    ax.set_title(f"Integrated Gradients Attribution - Sample {sample_idx}")

    # Add grid for better readability
    ax.grid(False)

    # Save to a bytes buffer to embed in HTML
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png", bbox_inches="tight", dpi=150)
    img_buffer.seek(0)
    img_data = base64.b64encode(img_buffer.read()).decode()
    plt.close()

    # Create temporal summary (sum over features)
    temporal_importance = np.sum(np.abs(sample_attr), axis=1)

    # Create feature summary (sum over time)
    feature_importance = np.sum(np.abs(sample_attr), axis=0)

    # Get prediction if available
    prediction = (
        metadata.get("predictions", [None])[sample_idx]
        if sample_idx < len(metadata.get("predictions", []))
        else None
    )

    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Integrated Gradients Heatmap - Sample {sample_idx}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f9f9f9;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .metadata {{
                background: #e8f4f8;
                padding: 15px;
                margin: 15px 0;
                border-radius: 5px;
                border-left: 5px solid #2196F3;
            }}
            .summary {{
                background: #f0f8e8;
                padding: 15px;
                margin: 15px 0;
                border-radius: 5px;
                border-left: 5px solid #4CAF50;
            }}
            .interpretation {{
                background: #fff3e0;
                padding: 15px;
                margin: 15px 0;
                border-radius: 5px;
                border-left: 5px solid #FF9800;
            }}
            .heatmap {{
                text-align: center;
                margin: 20px 0;
            }}
            .stats-table {{
                border-collapse: collapse;
                width: 100%;
                margin: 10px 0;
            }}
            .stats-table th, .stats-table td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            .stats-table th {{
                background-color: #f2f2f2;
            }}
            .highlight {{
                background-color: #ffeb3b;
                padding: 2px 4px;
                border-radius: 3px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 Integrated Gradients Analysis</h1>
            <h2>Sample {sample_idx} Attribution Heatmap</h2>
            
            <div class="metadata">
                <h3>📊 Sample Information</h3>
                <table class="stats-table">
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Sample Index</td><td>{sample_idx}</td></tr>
                    <tr><td>Prediction Score</td><td>{'%.4f' % prediction if prediction is not None else 'N/A'}</td></tr>
                    <tr><td>Data Shape</td><td>{sample_attr.shape[0]} time steps × {sample_attr.shape[1]} features</td></tr>
                    <tr><td>Attribution Range</td><td>[{sample_attr.min():.6f}, {sample_attr.max():.6f}]</td></tr>
                    <tr><td>Total Attribution</td><td>{np.sum(sample_attr):.6f}</td></tr>
                    <tr><td>Integration Steps</td><td>{metadata.get('integration_steps', 'N/A')}</td></tr>
                </table>
            </div>
            
            <div class="heatmap">
                <img src="data:image/png;base64,{img_data}" alt="IG Attribution Heatmap" style="max-width: 100%; height: auto;">
            </div>
            
            <div class="summary">
                <h3>⏱️ Temporal Importance Analysis</h3>
                <p>Which time steps (packets) contributed most to the prediction:</p>
                <table class="stats-table">
                    <tr><th>Time Step</th><th>Total Attribution</th><th>Relative Importance</th></tr>
    """

    # Add temporal importance table
    sorted_temporal = sorted(
        enumerate(temporal_importance), key=lambda x: abs(x[1]), reverse=True
    )
    for i, (time_step, importance) in enumerate(sorted_temporal[:10]):  # Top 10
        relative = importance / np.max(np.abs(temporal_importance)) * 100
        style = 'class="highlight"' if i < 3 else ""
        html_content += f"""
                    <tr {style}><td>{time_step}</td><td>{importance:.6f}</td><td>{relative:.1f}%</td></tr>
        """

    html_content += f"""
                </table>
                <p><em>Most important time step: <span class="highlight">Time Step {np.argmax(np.abs(temporal_importance))}</span></em></p>
            </div>
            
            <div class="summary">
                <h3>🎯 Feature Importance Analysis</h3>
                <p>Which features contributed most across all time steps:</p>
                <table class="stats-table">
                    <tr><th>Feature Index</th><th>Total Attribution</th><th>Relative Importance</th></tr>
    """

    # Add feature importance table
    sorted_features = sorted(
        enumerate(feature_importance), key=lambda x: abs(x[1]), reverse=True
    )
    for i, (feat_idx, importance) in enumerate(sorted_features[:15]):  # Top 15
        relative = importance / np.max(np.abs(feature_importance)) * 100
        style = 'class="highlight"' if i < 5 else ""
        html_content += f"""
                    <tr {style}><td>{feat_idx}</td><td>{importance:.6f}</td><td>{relative:.1f}%</td></tr>
        """

    html_content += f"""
                </table>
                <p><em>Most important feature: <span class="highlight">Feature {np.argmax(np.abs(feature_importance))}</span></em></p>
            </div>
            
            <div class="interpretation">
                <h3>💡 How to Read This Heatmap</h3>
                <ul>
                    <li><strong>🔴 Red areas</strong>: Features that <em>increase</em> the attack probability</li>
                    <li><strong>🔵 Blue areas</strong>: Features that <em>decrease</em> the attack probability</li>
                    <li><strong>⚪ White areas</strong>: Features with little influence on the decision</li>
                    <li><strong>Y-axis (Time)</strong>: Represents packet sequence (0 = first packet)</li>
                    <li><strong>X-axis (Features)</strong>: Network flow features (bytes, packets, duration, etc.)</li>
                </ul>
                
                <h4>🎯 Key Insights for Sample {sample_idx}:</h4>
                <ul>
                    <li>Most critical time step: <strong>Packet {np.argmax(np.abs(temporal_importance))}</strong></li>
                    <li>Most important feature: <strong>Feature {np.argmax(np.abs(feature_importance))}</strong></li>
                    <li>Attack probability: <strong>{prediction:.1%}</strong> {('(High Risk 🔴)' if prediction and prediction > 0.7 else '(Medium Risk 🟡)' if prediction and prediction > 0.3 else '(Low Risk 🟢)') if prediction else ''}</li>
                </ul>
            </div>
            
            <div class="metadata">
                <h3>📋 Technical Details</h3>
                <ul>
                    <li><strong>Method:</strong> Integrated Gradients with {metadata.get('integration_steps', 'N/A')} integration steps</li>
                    <li><strong>Baseline:</strong> Zero tensor (neutral input)</li>
                    <li><strong>Model:</strong> BiLSTM sequence classifier</li>
                    <li><strong>Total samples analyzed:</strong> {metadata.get('samples_analyzed', 'N/A')}</li>
                </ul>
            </div>
            
            <footer style="text-align: center; margin-top: 30px; color: #666; font-size: 0.9em;">
                <p>Generated by arx-nid explainability module | Time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </footer>
        </div>
    </body>
    </html>
    """

    if output_path:
        with open(output_path, "w") as f:
            f.write(html_content)
        print(f"HTML visualization saved to {output_path}")

    return html_content


def main():
    parser = argparse.ArgumentParser(
        description="Create HTML heatmap from IG attributions"
    )
    parser.add_argument(
        "--sample", type=int, default=0, help="Which sample to visualize (default: 0)"
    )
    parser.add_argument(
        "--input-dir",
        default="reports",
        help="Directory containing IG attribution files",
    )
    parser.add_argument(
        "--output",
        help="Output HTML file path (default: reports/ig_heatmap_{sample}.html)",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    # Load attribution data
    attr_path = input_dir / "ig_attr.npy"
    if not attr_path.exists():
        print(f"Error: Attribution file not found: {attr_path}")
        print("Run 'python scripts/make_ig.py' first")
        return

    # Load metadata
    metadata_path = input_dir / "ig_attr_metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    print(f"Loading attributions from {attr_path}")
    attributions = np.load(attr_path)
    print(f"Attribution shape: {attributions.shape}")

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_dir / f"ig_heatmap_{args.sample}.html"

    # Create visualization
    print(f"Creating heatmap for sample {args.sample}")
    create_heatmap_html(attributions, metadata, args.sample, output_path)

    print(f"\n✓ HTML heatmap created successfully!")
    print(f"  Sample: {args.sample}")
    print(f"  Output: {output_path}")
    print(f"  Open in browser: file://{output_path.absolute()}")


if __name__ == "__main__":
    main()
