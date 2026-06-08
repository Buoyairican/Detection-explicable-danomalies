"""
Download CICIDS2017 dataset from Kaggle into data/raw/.

Prerequisites:
  1. pip install kaggle
  2. Place your kaggle.json at ~/.kaggle/kaggle.json
       (download from https://www.kaggle.com/settings -> API -> Create New Token)
     OR set environment variables:
       export KAGGLE_USERNAME=your_username
       export KAGGLE_KEY=your_api_key

Usage:
  python scripts/download_data.py
"""

import pathlib
import sys

DATA_DIR = pathlib.Path("data/raw")
DATASET = "cicdataset/cicids2017"


def main():
    try:
        import kaggle
    except ImportError:
        print("Error: kaggle package not installed. Run: pip install kaggle")
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET} into {DATA_DIR} ...")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(DATASET, path=str(DATA_DIR), unzip=True)
    files = list(DATA_DIR.glob("*.csv"))
    print(f"Done. {len(files)} CSV file(s) in {DATA_DIR}:")
    for f in sorted(files):
        size_mb = f.stat().st_size / 1_048_576
        print(f"  {f.name}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
