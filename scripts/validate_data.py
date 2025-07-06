#!/usr/bin/env python3
"""
Validate data integrity and structure for arx-nid project.
Usage: python validate_data.py
"""
import csv
import hashlib
import sys
from pathlib import Path


def validate_manifest():
    """Validate the datasets manifest"""
    manifest_path = Path("data/datasets.csv")
    if not manifest_path.exists():
        print("❌ Manifest file not found: data/datasets.csv")
        return False

    print("📋 Validating data manifest...")
    try:
        with open(manifest_path) as f:
            reader = csv.DictReader(f)
            datasets = list(reader)

        print(f"✅ Found {len(datasets)} datasets in manifest")

        for ds in datasets:
            print(f"  📦 {ds['dataset']}")
            print(f"     URL: {ds['url']}")
            print(f"     SHA256: {ds['sha256']}")

        return True
    except Exception as e:
        print(f"❌ Error reading manifest: {e}")
        return False


def validate_scripts():
    """Validate download scripts exist and are syntactically correct"""
    print("\n🐍 Validating download scripts...")
    scripts_dir = Path("scripts")

    if not scripts_dir.exists():
        print("❌ Scripts directory not found")
        return False

    script_files = list(scripts_dir.glob("download_*.py"))
    if not script_files:
        print("❌ No download scripts found")
        return False

    success = True
    for script in script_files:
        try:
            compile(open(script).read(), script, "exec")
            print(f"  ✅ {script.name}")
        except SyntaxError as e:
            print(f"  ❌ {script.name}: Syntax error - {e}")
            success = False
        except Exception as e:
            print(f"  ❌ {script.name}: Error - {e}")
            success = False

    return success


def validate_structure():
    """Validate directory structure"""
    print("\n📁 Validating directory structure...")

    required_dirs = ["data", "data/raw", "data/processed", "scripts"]

    success = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  ✅ {dir_path}/")
        else:
            print(f"  ❌ {dir_path}/ (missing)")
            success = False

    return success


def validate_downloaded_data():
    """Check for downloaded data and validate checksums"""
    print("\n💾 Checking downloaded data...")

    raw_dir = Path("data/raw")
    if not raw_dir.exists():
        print("  ℹ️  No raw data directory")
        return True

    zip_files = list(raw_dir.glob("*.zip"))
    if not zip_files:
        print("  ℹ️  No downloaded data files found")
        return True

    # Load expected checksums
    manifest = {}
    try:
        with open("data/datasets.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                manifest[f"{row['dataset']}.zip"] = row["sha256"]
    except Exception as e:
        print(f"  ❌ Could not read manifest: {e}")
        return False

    success = True
    for zip_file in zip_files:
        expected_hash = manifest.get(zip_file.name)
        if (
            not expected_hash
            or expected_hash == "placeholder_hash_to_be_updated_after_download"
        ):
            print(f"  ⚠️  {zip_file.name}: No checksum to verify")
            continue

        print(f"  🔍 Verifying {zip_file.name}...")
        try:
            actual_hash = calculate_sha256(zip_file)
            if actual_hash == expected_hash:
                print("    ✅ Checksum verified")
            else:
                print("    ❌ Checksum mismatch!")
                print(f"       Expected: {expected_hash}")
                print(f"       Actual:   {actual_hash}")
                success = False
        except Exception as e:
            print(f"    ❌ Error calculating checksum: {e}")
            success = False

    return success


def calculate_sha256(file_path, buf_size=65536):
    """Calculate SHA256 hash of a file"""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(buf_size):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def main():
    """Run all validation checks"""
    print("🔍 arx-nid Data Validation")
    print("=" * 50)

    checks = [
        ("Manifest", validate_manifest),
        ("Scripts", validate_scripts),
        ("Structure", validate_structure),
        ("Downloaded Data", validate_downloaded_data),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Error in {name} validation: {e}")
            results.append((name, False))

    print("\n" + "=" * 50)
    print("📊 Validation Summary:")

    all_passed = True
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name:20} {status}")
        if not result:
            all_passed = False

    if all_passed:
        print("\n🎉 All validations passed!")
        return 0
    else:
        print("\n⚠️  Some validations failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
