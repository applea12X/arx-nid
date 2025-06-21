#!/usr/bin/env python3
"""
Convert Zeek logs to Parquet format for efficient processing.

This script:
1. Converts PCAP files to Zeek logs (if needed)
2. Parses JSON-formatted Zeek logs
3. Combines all connection logs into a single Parquet file
"""

import json
import glob
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import argparse

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd


def convert_pcap_to_zeek(pcap_path: str, output_dir: str) -> None:
    """Convert PCAP file to Zeek logs."""
    pcap_path = Path(pcap_path)
    output_dir = Path(output_dir)
    
    if not pcap_path.exists():
        raise FileNotFoundError(f"PCAP file not found: {pcap_path}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run Zeek to process the PCAP
    cmd = [
        "zeek", "-r", str(pcap_path),
        f"Log::default_writer=JSON",
        f"Log::default_path={output_dir}"
    ]
    
    print(f"Converting {pcap_path.name} to Zeek logs...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✓ Zeek logs saved to {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error running Zeek: {e}")
        print(f"Command: {' '.join(cmd)}")
        print(f"Stderr: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: Zeek not found. Install with: brew install zeek")
        sys.exit(1)


def parse_zeek_logs(log_dir: str, log_type: str = "conn") -> List[Dict[str, Any]]:
    """Parse Zeek logs from JSON format."""
    log_dir = Path(log_dir)
    log_pattern = f"{log_type}*.json"
    
    rows = []
    for log_file in log_dir.glob(log_pattern):
        print(f"Parsing {log_file.name}...")
        
        with open(log_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    record = json.loads(line)
                    rows.append(record)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed JSON on line {line_num}: {e}")
                    continue
    
    print(f"Parsed {len(rows)} records from {log_type} logs")
    return rows


def convert_to_parquet(data: List[Dict[str, Any]], output_path: str) -> None:
    """Convert parsed data to Parquet format."""
    if not data:
        print("No data to convert")
        return
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to PyArrow table for efficient serialization
    table = pa.Table.from_pylist(data)
    
    # Write with compression
    pq.write_table(
        table, 
        output_path, 
        compression="zstd",
        use_dictionary=True,  # Better compression for string columns
        write_statistics=True
    )
    
    # Print some stats
    file_size = output_path.stat().st_size / (1024 * 1024)  # MB
    print(f"✓ Saved {len(data)} records to {output_path}")
    print(f"  File size: {file_size:.2f} MB")
    print(f"  Columns: {len(table.column_names)}")


def main():
    parser = argparse.ArgumentParser(description="Convert PCAP files to Parquet via Zeek")
    parser.add_argument("--input-dir", default="data/raw", 
                       help="Directory containing PCAP files")
    parser.add_argument("--interim-dir", default="data/interim/zeek",
                       help="Directory for intermediate Zeek logs")
    parser.add_argument("--output", default="data/processed/flows_v0.parquet",
                       help="Output Parquet file path")
    parser.add_argument("--log-type", default="conn",
                       help="Zeek log type to process (conn, dns, http, etc.)")
    parser.add_argument("--dataset", 
                       help="Specific dataset to process (e.g., hitar-2024)")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    interim_dir = Path(args.interim_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Find PCAP files to process
    pcap_files = []
    if args.dataset:
        # Process specific dataset
        dataset_dir = input_dir / args.dataset
        if dataset_dir.exists():
            pcap_files.extend(dataset_dir.glob("*.pcap"))
            pcap_files.extend(dataset_dir.glob("*.pcapng"))
    else:
        # Process all datasets
        for dataset_dir in input_dir.iterdir():
            if dataset_dir.is_dir():
                pcap_files.extend(dataset_dir.glob("*.pcap"))
                pcap_files.extend(dataset_dir.glob("*.pcapng"))
    
    if not pcap_files:
        print(f"No PCAP files found in {input_dir}")
        # Try to process existing Zeek logs instead
        all_rows = []
        for log_dir in interim_dir.iterdir():
            if log_dir.is_dir():
                rows = parse_zeek_logs(str(log_dir), args.log_type)
                all_rows.extend(rows)
        
        if all_rows:
            convert_to_parquet(all_rows, args.output)
        else:
            print("No data found to process")
        return
    
    # Process each PCAP file
    all_rows = []
    for pcap_file in pcap_files:
        dataset_name = pcap_file.parent.name
        zeek_output_dir = interim_dir / dataset_name
        
        # Convert PCAP to Zeek logs
        convert_pcap_to_zeek(str(pcap_file), str(zeek_output_dir))
        
        # Parse the generated logs
        rows = parse_zeek_logs(str(zeek_output_dir), args.log_type)
        all_rows.extend(rows)
    
    # Convert all data to Parquet
    if all_rows:
        convert_to_parquet(all_rows, args.output)
    else:
        print("No data extracted from PCAP files")


if __name__ == "__main__":
    main()
