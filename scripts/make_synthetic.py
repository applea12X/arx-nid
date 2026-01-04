"""
Create synthetic attack flows for explainability demonstration.

This script generates artificial network flows with specific patterns
that should trigger IDS alerts, providing clear explainability examples.

Usage:
    python scripts/make_synthetic.py [--attack-type ddos] [--output-dir data/processed]
"""

import argparse
import json
import numpy as np
from pathlib import Path


def create_ddos_flow(num_packets=20, num_features=34):
    """
    Create a synthetic DDoS attack flow.

    This flow has characteristics typical of DDoS attacks:
    - High packet rate
    - Large byte counts
    - Consistent timing
    - Minimal response traffic
    """
    flow = np.zeros((num_packets, num_features))

    # Assuming feature indices based on typical network flow features
    # Feature 0-1: orig_bytes, resp_bytes
    # Feature 2-3: orig_pkts, resp_pkts
    # Feature 4: duration
    # Feature 5-7: protocol features
    # etc. (adjust based on actual feature engineering)

    for i in range(num_packets):
        # Massive outgoing bytes (typical of DDoS flood)
        flow[i, 0] = 50000 + np.random.normal(10000, 2000)  # orig_bytes
        flow[i, 1] = np.random.normal(100, 50)  # resp_bytes (minimal response)

        # High packet counts
        flow[i, 2] = 500 + np.random.normal(100, 20)  # orig_pkts
        flow[i, 3] = np.random.normal(2, 1)  # resp_pkts (minimal)

        # Short duration per packet (rapid fire)
        flow[i, 4] = np.random.uniform(0.001, 0.01)  # duration

        # TCP protocol indicator (assuming one-hot encoding)
        if num_features > 10:
            flow[i, 10] = 1.0  # TCP protocol

        # Connection state (assuming failed/reset connections)
        if num_features > 15:
            flow[i, 15] = 1.0  # Connection reset/failed

        # High flow ratio (one-way traffic)
        if num_features > 20:
            flow[i, 20] = flow[i, 0] / (flow[i, 1] + 1)  # orig/resp ratio

        # High rate
        if num_features > 25:
            flow[i, 25] = flow[i, 0] / (flow[i, 4] + 0.001)  # bytes/second rate

    # Ensure no NaN or inf values
    flow = np.nan_to_num(flow, nan=0.0, posinf=100000.0, neginf=0.0)

    return flow


def create_port_scan_flow(num_packets=20, num_features=34):
    """
    Create a synthetic port scan attack flow.

    Characteristics:
    - Many small connections
    - Different destination ports
    - Minimal data transfer
    - Quick connection attempts
    """
    flow = np.zeros((num_packets, num_features))

    for i in range(num_packets):
        # Small data amounts
        flow[i, 0] = np.random.normal(100, 30)  # orig_bytes
        flow[i, 1] = np.random.normal(50, 20)  # resp_bytes

        # Few packets per connection
        flow[i, 2] = np.random.randint(1, 5)  # orig_pkts
        flow[i, 3] = np.random.randint(0, 2)  # resp_pkts

        # Very short durations
        flow[i, 4] = np.random.uniform(0.001, 0.1)  # duration

        # TCP SYN scan indicators
        if num_features > 10:
            flow[i, 10] = 1.0  # TCP protocol

        # Connection state variations (many failed attempts)
        if num_features > 15:
            if i % 3 == 0:
                flow[i, 15] = 1.0  # Reset
            elif i % 3 == 1:
                flow[i, 16] = 1.0  # Timeout
            else:
                flow[i, 17] = 1.0  # Failed

        # Different destination ports (simulated as feature variation)
        if num_features > 30:
            port_variation = np.sin(i * 0.5) * 1000  # Simulate port scanning
            flow[i, 30] = port_variation

    flow = np.nan_to_num(flow, nan=0.0, posinf=10000.0, neginf=0.0)
    return flow


def create_data_exfiltration_flow(num_packets=20, num_features=34):
    """
    Create a synthetic data exfiltration flow.

    Characteristics:
    - Large outbound data transfer
    - Steady, sustained connection
    - High byte-to-packet ratio
    - Long duration
    """
    flow = np.zeros((num_packets, num_features))

    base_bytes = 10000
    for i in range(num_packets):
        # Large, increasing outbound data
        flow[i, 0] = base_bytes + i * 5000 + np.random.normal(1000, 200)  # orig_bytes
        flow[i, 1] = np.random.normal(200, 50)  # resp_bytes (minimal ack)

        # Moderate packet counts (large packets)
        flow[i, 2] = np.random.normal(100, 20)  # orig_pkts
        flow[i, 3] = np.random.normal(10, 5)  # resp_pkts

        # Longer durations (sustained connection)
        flow[i, 4] = np.random.uniform(1.0, 10.0)  # duration

        # TCP protocol
        if num_features > 10:
            flow[i, 10] = 1.0

        # Established connections
        if num_features > 15:
            flow[i, 18] = 1.0  # Established connection

        # High bytes per packet ratio
        if num_features > 25:
            flow[i, 26] = flow[i, 0] / flow[i, 2]  # bytes per packet

    flow = np.nan_to_num(flow, nan=0.0, posinf=50000.0, neginf=0.0)
    return flow


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic attack flows")
    parser.add_argument(
        "--attack-type",
        choices=["ddos", "port_scan", "exfiltration", "all"],
        default="ddos",
        help="Type of attack to simulate",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Output directory for synthetic flows",
    )
    parser.add_argument(
        "--num-packets", type=int, default=5, help="Number of packets per flow (default: 5 to match tensor_v0.npy)"
    )
    parser.add_argument(
        "--num-features", type=int, default=34, help="Number of features per packet"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    attack_types = (
        ["ddos", "port_scan", "exfiltration"]
        if args.attack_type == "all"
        else [args.attack_type]
    )

    for attack_type in attack_types:
        print(f"Creating {attack_type} synthetic flow...")

        # Generate the appropriate flow
        if attack_type == "ddos":
            flow = create_ddos_flow(args.num_packets, args.num_features)
            description = "DDoS attack with massive byte counts and high packet rates"
        elif attack_type == "port_scan":
            flow = create_port_scan_flow(args.num_packets, args.num_features)
            description = "Port scan with many small, failed connections"
        elif attack_type == "exfiltration":
            flow = create_data_exfiltration_flow(args.num_packets, args.num_features)
            description = "Data exfiltration with large, sustained outbound traffic"
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")

        # Save the flow data
        flow_path = output_dir / f"synthetic_{attack_type}.npy"
        np.save(flow_path, flow)
        print(f"Saved synthetic {attack_type} flow to {flow_path}")

        # Create metadata
        metadata = {
            "attack_type": attack_type,
            "description": description,
            "shape": flow.shape,
            "num_packets": args.num_packets,
            "num_features": args.num_features,
            "statistics": {
                "mean": float(np.mean(flow)),
                "std": float(np.std(flow)),
                "min": float(np.min(flow)),
                "max": float(np.max(flow)),
                "total_bytes_orig": (
                    float(np.sum(flow[:, 0])) if flow.shape[1] > 0 else 0
                ),
                "total_bytes_resp": (
                    float(np.sum(flow[:, 1])) if flow.shape[1] > 1 else 0
                ),
                "avg_duration": float(np.mean(flow[:, 4])) if flow.shape[1] > 4 else 0,
            },
            "key_characteristics": get_attack_characteristics(attack_type),
            "generation_timestamp": (
                pd.Timestamp.now().isoformat() if "pd" in globals() else "unknown"
            ),
        }

        # Import pandas for timestamp
        try:
            import pandas as pd

            metadata["generation_timestamp"] = pd.Timestamp.now().isoformat()
        except ImportError:
            metadata["generation_timestamp"] = "unknown"

        # Save metadata
        metadata_path = output_dir / f"synthetic_{attack_type}.meta.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {metadata_path}")

        # Print summary
        print(f"\n{attack_type.upper()} Flow Summary:")
        print(f"  Shape: {flow.shape}")
        print(f"  Total orig bytes: {np.sum(flow[:, 0]):.0f}")
        print(
            f"  Total resp bytes: {np.sum(flow[:, 1]):.0f}" if flow.shape[1] > 1 else ""
        )
        print(
            f"  Avg duration: {np.mean(flow[:, 4]):.3f}s" if flow.shape[1] > 4 else ""
        )
        print(f"  Max feature value: {np.max(flow):.2f}")
        print()

    print("✓ Synthetic attack flow generation complete!")
    print(f"Generated flows: {', '.join(attack_types)}")
    print(f"Output directory: {output_dir}")

    # Show usage instructions
    print("\nTo analyze these flows with explainability tools:")
    for attack_type in attack_types:
        print(
            f"  python scripts/make_shap.py --sample synthetic --limit 1 --data-path data/processed/synthetic_{attack_type}.npy"
        )


def get_attack_characteristics(attack_type):
    """Get human-readable characteristics for each attack type."""
    characteristics = {
        "ddos": [
            "Extremely high byte counts (50,000+ bytes per packet)",
            "High packet rates with minimal responses",
            "One-way traffic pattern (high orig/resp ratio)",
            "Short connection durations",
            "TCP protocol with connection resets",
        ],
        "port_scan": [
            "Many small connections to different ports",
            "Minimal data transfer per connection",
            "Very short connection durations",
            "High rate of failed/reset connections",
            "Sequential port scanning pattern",
        ],
        "exfiltration": [
            "Large outbound data transfers",
            "Sustained, long-duration connections",
            "High bytes-per-packet ratio",
            "Minimal inbound traffic",
            "Established TCP connections",
        ],
    }
    return characteristics.get(attack_type, [])


if __name__ == "__main__":
    main()
