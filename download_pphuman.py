#!/usr/bin/env python3
"""
PP-Human Attribute Model Downloader

Downloads the pre-trained PP-Human attribute model for comparison
with the PA-100k fine-tuned model.

Model: PPLCNet_x1_0_person_attribute_945_infer
Architecture: PPLCNet x1.0 (Medium)
Attributes: 26 (without Male)
mAP: 95.4%
Parameters: 6.7M

Usage:
    python download_pphuman.py
"""

import os
import sys
import urllib.request
import tarfile
from pathlib import Path

# Model configuration
MODEL_NAME = "PP-Human Attribute (PPLCNet x1.0)"
MODEL_URL = "https://bj.bcebos.com/v1/paddledet/models/pipeline/PPLCNet_x1_0_person_attribute_945_infer.tar"
OUTPUT_DIR = Path(__file__).parent / "pretrained_models" / "pphuman_attribute"

def download_progress(count, block_size, total_size):
    """Show download progress"""
    percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
    percent = min(percent, 100)
    bar_length = 50
    filled = int(bar_length * percent / 100)
    bar = '█' * filled + '-' * (bar_length - filled)
    sys.stdout.write(f'\r  [{bar}] {percent}%')
    sys.stdout.flush()

def download_file(url: str, output_path: Path) -> bool:
    """Download a file with progress bar"""
    try:
        print(f"  URL: {url}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"  Downloading...")
        urllib.request.urlretrieve(url, output_path, reporthook=download_progress)
        print(f"\n  Downloaded: {output_path}")
        print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        return True
    except Exception as e:
        print(f"\n  Error: {e}")
        return False

def extract_tar(tar_path: Path, extract_to: Path) -> bool:
    """Extract tar archive"""
    try:
        print(f"\n  Extracting...")
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=extract_to)
        print(f"  Extracted to: {extract_to}")

        # List extracted files
        print(f"\n  Files extracted:")
        for item in extract_to.rglob("*"):
            if item.is_file():
                size_mb = item.stat().st_size / 1024 / 1024
                print(f"    {item.name}: {size_mb:.2f} MB")

        return True
    except Exception as e:
        print(f"  Extract error: {e}")
        return False

def main():
    print("=" * 70)
    print(f"Downloading: {MODEL_NAME}")
    print("=" * 70)
    print(f"\nModel Details:")
    print(f"  Name: PP-Human Attribute Recognition")
    print(f"  Architecture: PPLCNet x1.0 (Medium)")
    print(f"  Attributes: 26 (Female, Age, Bags, Clothing, etc.)")
    print(f"  Note: Does NOT include 'Male' as separate attribute")
    print(f"  mAP: 95.4%")
    print(f"  Parameters: 6.7M")
    print(f"\n  This is for COMPARISON with the PA-100k fine-tuned model")
    print(f"  (which has 27 attributes including Male)")

    # Check if already downloaded
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        print(f"\n  Model already exists: {OUTPUT_DIR}")
        overwrite = input("\n  Download again? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("\n  Skipping download.")
            return

    # Download
    tar_filename = MODEL_URL.split("/")[-1]
    tar_path = OUTPUT_DIR / tar_filename

    print(f"\n[1/3] Downloading model...")
    if not download_file(MODEL_URL, tar_path):
        print("\n❌ Download failed")
        sys.exit(1)

    # Extract
    print(f"\n[2/3] Extracting archive...")
    if not extract_tar(tar_path, OUTPUT_DIR):
        print("\n❌ Extraction failed")
        sys.exit(1)

    # Save info
    print(f"\n[3/3] Saving model info...")
    info_file = OUTPUT_DIR / "model_info.txt"
    with open(info_file, "w") as f:
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Architecture: PPLCNet x1.0\n")
        f.write(f"Attributes: 26\n")
        f.write(f"mAP: 95.4%\n")
        f.write(f"Parameters: 6.7M\n")
        f.write(f"URL: {MODEL_URL}\n")
        f.write(f"Downloaded: {Path(tar_filename).name}\n")

    print(f"  Info saved: {info_file}")

    # Cleanup tar file
    if tar_path.exists():
        tar_path.unlink()
        print(f"  Cleaned up: {tar_filename}")

    print("\n" + "=" * 70)
    print("✅ PP-Human model downloaded successfully!")
    print("=" * 70)
    print(f"\nModel location: {OUTPUT_DIR}")
    print(f"\nTo use this model:")
    print(f"  1. Convert to ONNX (if needed)")
    print(f"  2. Compare with PA-100k fine-tuned model")
    print(f"  3. Test on video with test_attributes_cpu.py")

if __name__ == "__main__":
    main()
