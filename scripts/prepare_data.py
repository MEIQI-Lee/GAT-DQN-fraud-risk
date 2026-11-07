"""Step 1: build the processed graph from raw Elliptic CSVs.

Usage:  python scripts/prepare_data.py [--force]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.data_utils import build_dataset, describe
from src.utils import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="rebuild even if cached")
    args = parser.parse_args()

    set_seed(config.SEED)
    t0 = time.time()
    data = build_dataset(force_rebuild=args.force)
    print(f"[prepare_data] done in {time.time() - t0:.1f}s")
    print(describe(data))
    print(f"cache -> {config.DATA_PROCESSED / 'elliptic_graph.pt'}")


if __name__ == "__main__":
    main()
