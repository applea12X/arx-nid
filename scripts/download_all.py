#!/usr/bin/env python3
"""
Utility script to download all datasets or specific ones.
Usage: 
    python download_all.py                    # Download all datasets
    python download_all.py hitar-2024        # Download specific dataset
    python download_all.py --list            # List available datasets
"""
import argparse
import csv
import sys
from pathlib import Path
import subprocess

MANIFEST = "data/datasets.csv"
SCRIPTS_DIR = Path("scripts")

def list_datasets():
    """List all available datasets"""
    print("Available datasets:")
    print("-" * 50)
    with open(MANIFEST) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            dataset = row["dataset"]
            url = row["url"]
            print(f"  {dataset}")
            print(f"    URL: {url}")
            print()

def get_available_datasets():
    """Get list of available datasets from manifest"""
    datasets = []
    with open(MANIFEST) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            datasets.append(row["dataset"])
    return datasets

def download_dataset(dataset_name, output_dir="data/raw"):
    """Download a specific dataset"""
    script_name = f"download_{dataset_name.replace('-', '_')}.py"
    script_path = SCRIPTS_DIR / script_name
    
    if not script_path.exists():
        print(f"❌ Download script not found: {script_path}")
        return False
    
    print(f"📥 Downloading {dataset_name}...")
    try:
        result = subprocess.run([
            "python3", str(script_path), "--out", output_dir
        ], check=True, capture_output=True, text=True)
        print(f"✅ Successfully downloaded {dataset_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download {dataset_name}")
        print(f"Error: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download datasets for arx-nid project")
    parser.add_argument("datasets", nargs="*", help="Specific datasets to download")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--out", default="data/raw", help="Output directory")
    
    args = parser.parse_args()
    
    if args.list:
        list_datasets()
        return
    
    available_datasets = get_available_datasets()
    
    if not args.datasets:
        # Download all datasets
        print("📦 Downloading all datasets...")
        success_count = 0
        for dataset in available_datasets:
            if download_dataset(dataset, args.out):
                success_count += 1
        
        print(f"\n✅ Successfully downloaded {success_count}/{len(available_datasets)} datasets")
    else:
        # Download specific datasets
        for dataset in args.datasets:
            if dataset not in available_datasets:
                print(f"❌ Unknown dataset: {dataset}")
                print(f"Available datasets: {', '.join(available_datasets)}")
                continue
            
            download_dataset(dataset, args.out)

if __name__ == "__main__":
    main()
