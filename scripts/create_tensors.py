#!/usr/bin/env python3
"""
Convert preprocessed flow data to time-series tensors for deep learning.

This script:
1. Loads preprocessed Parquet flow data
2. Groups flows by connection tuples
3. Creates sliding window sequences
4. Outputs tensors in [batch, time, features] format
"""

import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from typing import List, Optional

from arx_nid.features.transformers import FlowPreprocessor


def create_sliding_windows(
    df: pd.DataFrame, window_size: int = 20, stride: int = 1, min_flow_length: int = 5
) -> np.ndarray:
    """
    Create sliding window sequences from flow data.

    Args:
        df: Preprocessed flow DataFrame
        window_size: Number of time steps per sequence
        stride: Step size between windows
        min_flow_length: Minimum flow length to include

    Returns:
        Tensor of shape [num_samples, window_size, num_features]
    """
    # Sort by timestamp
    df = df.sort_values("ts")

    # Group by flow tuple
    flow_groups = df.groupby(["id.orig_h", "id.resp_h"])

    sequences = []
    flow_metadata = []

    for (src_ip, dst_ip), flow_data in flow_groups:
        if len(flow_data) < min_flow_length:
            continue

        # Get numeric features only (exclude timestamps and IPs)
        feature_cols = []
        for col in flow_data.columns:
            if col not in ["ts", "id.orig_h", "id.resp_h", "uid"]:
                if flow_data[col].dtype in ["int64", "float64", "int32", "float32"]:
                    feature_cols.append(col)

        if not feature_cols:
            continue

        flow_features = flow_data[feature_cols].values

        # Create sliding windows
        for start_idx in range(0, len(flow_features) - window_size + 1, stride):
            end_idx = start_idx + window_size
            sequence = flow_features[start_idx:end_idx]

            # Handle any remaining NaN values
            if np.isnan(sequence).any():
                sequence = np.nan_to_num(sequence, 0.0)

            sequences.append(sequence)
            flow_metadata.append(
                {
                    "src_ip": src_ip,
                    "dst_ip": dst_ip,
                    "start_time": flow_data.iloc[start_idx]["ts"],
                    "end_time": flow_data.iloc[end_idx - 1]["ts"],
                }
            )

    if not sequences:
        print("Warning: No valid sequences generated")
        return np.array([]), []

    tensor = np.stack(sequences)
    print(f"Generated {len(sequences)} sequences of shape {tensor.shape}")
    print(
        f"Feature columns ({len(feature_cols)}): {feature_cols[:5]}{'...' if len(feature_cols) > 5 else ''}"
    )

    return tensor, flow_metadata


def prepare_features_for_tensors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare flow data for tensor conversion.

    This includes categorical encoding and feature scaling.
    """
    # Define categorical and numeric columns
    categorical_cols = ["proto", "service", "conn_state"]

    # Initialize preprocessor
    preprocessor = FlowPreprocessor(
        rolling_window_s=5, rolling_stats=["mean"], categorical_cols=categorical_cols
    )

    # Apply preprocessing
    df_processed = preprocessor.fit_transform(df)

    return df_processed


def save_tensor_data(
    tensor: np.ndarray,
    metadata: List[dict],
    output_path: str,
    metadata_path: Optional[str] = None,
):
    """Save tensor and metadata to files."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save tensor
    np.save(output_path, tensor)
    print(f"✓ Saved tensor of shape {tensor.shape} to {output_path}")

    # Save metadata if requested
    if metadata_path and metadata:
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_parquet(metadata_path, compression="zstd")
        print(f"✓ Saved metadata ({len(metadata)} records) to {metadata_path}")


def analyze_tensor_properties(tensor: np.ndarray):
    """Print analysis of tensor properties."""
    print("\n=== Tensor Analysis ===")
    print(f"Shape: {tensor.shape}")
    print(f"Data type: {tensor.dtype}")
    print(f"Memory usage: {tensor.nbytes / (1024**2):.2f} MB")
    print(f"Value range: [{tensor.min():.4f}, {tensor.max():.4f}]")
    print(f"Mean: {tensor.mean():.4f}")
    print(f"Std: {tensor.std():.4f}")
    print(f"NaN count: {np.isnan(tensor).sum()}")
    print(f"Inf count: {np.isinf(tensor).sum()}")


def main():
    parser = argparse.ArgumentParser(description="Convert flow data to tensors")
    parser.add_argument(
        "--input",
        default="data/processed/flows_v0.parquet",
        help="Input Parquet file with preprocessed flows",
    )
    parser.add_argument(
        "--output",
        default="data/processed/tensor_v0.npy",
        help="Output tensor file path",
    )
    parser.add_argument(
        "--metadata",
        default="data/processed/tensor_metadata_v0.parquet",
        help="Output metadata file path",
    )
    parser.add_argument(
        "--window-size", type=int, default=20, help="Sequence length for each sample"
    )
    parser.add_argument(
        "--stride", type=int, default=1, help="Step size between windows"
    )
    parser.add_argument(
        "--min-flow-length", type=int, default=5, help="Minimum flow length to include"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to generate (for testing)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        print("Run 'python scripts/zeek2parquet.py' first to generate flow data")
        return

    # Load flow data
    print(f"Loading flow data from {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} flow records")

    if len(df) == 0:
        print("No data to process")
        return

    # Limit data for testing if requested
    if args.max_samples:
        df = df.head(args.max_samples * args.window_size)
        print(f"Limited to {len(df)} records for testing")

    # Preprocess features
    print("Preprocessing features...")
    try:
        df_processed = prepare_features_for_tensors(df)
        print(f"Preprocessed data shape: {df_processed.shape}")
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        print("Attempting basic tensor creation without preprocessing...")
        df_processed = df

    # Create tensor sequences
    print(
        f"Creating sequences (window_size={args.window_size}, stride={args.stride})..."
    )
    tensor, metadata = create_sliding_windows(
        df_processed,
        window_size=args.window_size,
        stride=args.stride,
        min_flow_length=args.min_flow_length,
    )

    if tensor.size == 0:
        print("No sequences generated. Check input data and parameters.")
        return

    # Analyze tensor properties
    analyze_tensor_properties(tensor)

    # Save outputs
    save_tensor_data(tensor, metadata, args.output, args.metadata)

    print("\n✓ Tensor generation complete!")
    print(f"  Input: {len(df)} flow records")
    print(f"  Output: {tensor.shape[0]} sequences")
    print(f"  Shape: {tensor.shape}")


if __name__ == "__main__":
    main()
