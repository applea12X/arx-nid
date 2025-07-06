#!/usr/bin/env python3
"""
Download CIC IoV-2024 and verify checksum.
Usage: python download_cic_iov.py --out data/raw
"""
from pathlib import Path
import argparse
import hashlib
import requests
import tqdm
import csv

MANIFEST = "data/datasets.csv"  # url, sha256 stored here


def get_meta(name):
    """Get URL and expected SHA256 from manifest CSV"""
    with open(MANIFEST) as fh:
        for row in csv.DictReader(fh):
            if row["dataset"] == name:
                return row["url"], row["sha256"]
    raise KeyError(f"{name} not in manifest")


def sha256(fp, buf=1 << 20):
    """Calculate SHA256 hash of file"""
    h = hashlib.sha256()
    while chunk := fp.read(buf):
        h.update(chunk)
    fp.seek(0)
    return h.hexdigest()


def main(ds_name, out_dir):
    """Download and verify dataset"""
    url, expected = get_meta(ds_name)
    out_path = Path(out_dir) / f"{ds_name}.zip"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {ds_name} from {url}")

    # Skip download if hash is placeholder
    if expected == "placeholder_hash_to_be_updated_after_download":
        print(f"⚠️  Placeholder hash detected for {ds_name}")
        print(f"Please update {MANIFEST} with actual URL and SHA256")
        return

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(out_path, "wb") as f, tqdm.tqdm(
            total=total, unit="B", unit_scale=True, desc=ds_name
        ) as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))

    # Verify checksum
    with open(out_path, "rb") as f:
        actual_hash = sha256(f)

    if actual_hash != expected:
        print("❌ Checksum mismatch!")
        print(f"Expected: {expected}")
        print(f"Actual:   {actual_hash}")
        raise ValueError("Checksum verification failed")

    print("✓ Download and verification OK ➜", out_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Download CIC IoV-2024 dataset")
    p.add_argument("--out", default="data/raw", help="Output directory")
    args = p.parse_args()
    main("cic-iov-2024", args.out)
