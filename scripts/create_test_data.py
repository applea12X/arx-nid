#!/usr/bin/env python3
"""
Generate synthetic network flow data for testing the arx-nid pipeline.

This creates realistic Zeek connection log data that can be used to test
the feature transformers and tensor creation without requiring real PCAPs.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
from pathlib import Path


def generate_flow_data(num_flows=1000, duration_hours=1):
    """Generate synthetic network flow data."""

    # Base timestamp
    base_time = datetime.now() - timedelta(hours=duration_hours)

    # Generate realistic IP addresses
    internal_ips = [f"192.168.1.{i}" for i in range(10, 50)]
    external_ips = [f"10.0.{i//256}.{i%256}" for i in range(1, 100)]

    # Common protocols and services
    protocols = ["tcp", "udp", "icmp"]
    conn_states = ["SF", "S0", "REJ", "RSTO", "SH", "S1"]

    flows = []

    # Create some persistent connections that will have multiple flows
    persistent_connections = []
    for _ in range(
        min(50, num_flows // 20)
    ):  # ~5% of flows will be part of persistent connections
        orig_h = np.random.choice(internal_ips)
        resp_h = np.random.choice(external_ips)
        resp_p = np.random.choice([80, 443, 22])  # Common persistent services
        persistent_connections.append((orig_h, resp_h, resp_p))

    for i in range(num_flows):
        # Generate timestamps with some realistic patterns
        if i == 0:
            ts = base_time.timestamp()
        else:
            # Some clustering of connections
            if np.random.random() < 0.3:  # 30% chance of clustered timing
                ts = flows[-1]["ts"] + np.random.exponential(1)  # Close in time
            else:
                ts = (
                    base_time.timestamp()
                    + np.random.exponential(10) * i / num_flows * 3600
                )

        # Generate connection tuple - sometimes reuse persistent connections
        if (
            persistent_connections and np.random.random() < 0.4
        ):  # 40% chance to use persistent connection
            conn_idx = np.random.randint(len(persistent_connections))
            orig_h, resp_h, resp_p = persistent_connections[conn_idx]
            orig_p = np.random.randint(32768, 65536)  # Ephemeral port varies
        elif np.random.random() < 0.7:  # 70% outbound traffic
            orig_h = np.random.choice(internal_ips)
            resp_h = np.random.choice(external_ips)
            orig_p = np.random.randint(32768, 65536)  # Ephemeral port
            resp_p = np.random.choice([80, 443, 22, 53, 25])  # Common services
        else:  # 30% inbound traffic
            orig_h = np.random.choice(external_ips)
            resp_h = np.random.choice(internal_ips)
            orig_p = np.random.choice([80, 443, 22, 53, 25])
            resp_p = np.random.randint(32768, 65536)

        if isinstance(ts, datetime):
            ts = ts.timestamp()

        # Generate protocol and service
        proto = np.random.choice(protocols, p=[0.7, 0.25, 0.05])
        service = None
        if proto == "tcp":
            if resp_p == 80:
                service = "http"
            elif resp_p == 443:
                service = "https"
            elif resp_p == 22:
                service = "ssh"
            elif resp_p == 25:
                service = "smtp"
        elif proto == "udp" and resp_p == 53:
            service = "dns"

        # Generate realistic traffic patterns
        duration = np.random.exponential(5.0)  # Average 5 second connections

        # Traffic volume based on service type
        if service == "dns":
            orig_bytes = np.random.randint(50, 200)
            resp_bytes = np.random.randint(100, 500)
            orig_pkts = np.random.randint(1, 3)
            resp_pkts = np.random.randint(1, 3)
        elif service in ["http", "https"]:
            orig_bytes = np.random.randint(500, 5000)
            resp_bytes = np.random.randint(1000, 50000)
            orig_pkts = np.random.randint(5, 50)
            resp_pkts = np.random.randint(10, 100)
        else:
            orig_bytes = np.random.randint(100, 2000)
            resp_bytes = np.random.randint(100, 2000)
            orig_pkts = np.random.randint(2, 20)
            resp_pkts = np.random.randint(2, 20)

        # Connection state based on protocol and success probability
        if proto == "icmp":
            conn_state = "SF"
        else:
            conn_state = np.random.choice(
                conn_states, p=[0.8, 0.05, 0.05, 0.03, 0.05, 0.02]
            )

        flow = {
            "ts": ts,
            "uid": f"C{i:06d}",
            "id.orig_h": orig_h,
            "id.orig_p": orig_p,
            "id.resp_h": resp_h,
            "id.resp_p": resp_p,
            "proto": proto,
            "service": service,
            "duration": duration,
            "orig_bytes": orig_bytes,
            "resp_bytes": resp_bytes,
            "conn_state": conn_state,
            "orig_pkts": orig_pkts,
            "resp_pkts": resp_pkts,
            "orig_ip_bytes": orig_bytes + orig_pkts * 40,  # Add IP header overhead
            "resp_ip_bytes": resp_bytes + resp_pkts * 40,
            "history": "Dd" if conn_state == "SF" else "S",
            "tunnel_parents": None,
        }

        flows.append(flow)

    df = pd.DataFrame(flows)
    df = df.sort_values("ts")

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic flow data")
    parser.add_argument(
        "--output", default="data/processed/flows_v0.parquet", help="Output file path"
    )
    parser.add_argument(
        "--num-flows", type=int, default=1000, help="Number of flows to generate"
    )
    parser.add_argument(
        "--duration-hours",
        type=float,
        default=1.0,
        help="Time span of generated data in hours",
    )
    parser.add_argument(
        "--format", choices=["parquet", "csv"], default="parquet", help="Output format"
    )

    args = parser.parse_args()

    print(f"Generating {args.num_flows} synthetic flows...")
    df = generate_flow_data(args.num_flows, args.duration_hours)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "parquet":
        df.to_parquet(output_path, compression="zstd")
    else:
        df.to_csv(output_path, index=False)

    print(f"✓ Generated {len(df)} flows saved to {output_path}")
    print(
        f"  Time range: {pd.to_datetime(df['ts'].min(), unit='s')} to {pd.to_datetime(df['ts'].max(), unit='s')}"
    )
    print(f"  Protocols: {dict(df['proto'].value_counts())}")
    print(f"  Top services: {dict(df['service'].value_counts().head())}")


if __name__ == "__main__":
    main()
