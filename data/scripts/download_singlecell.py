#!/usr/bin/env python
"""
download_singlecell.py
----------------------
Download GSE131907 Lung Cancer single-cell annotation file from NCBI GEO.

Source: Kim et al. 2020, Nature Communications
  "Single-cell RNA sequencing demonstrates the molecular and cellular
   reprogramming of metastatic lung adenocarcinoma"

File: GSE131907_Lung_Cancer_cell_annotation.txt.gz
  - 208,506 cells × metadata columns
  - Cell type labels, sample IDs, tissue type, etc.
  - ~1.5 GB compressed

Output:
  modules/04_single_cell/data/input/GSE131907/
    GSE131907_Lung_Cancer_cell_annotation.txt.gz

Usage:
  python download_singlecell.py
"""

import logging
import sys
import time
from pathlib import Path

import requests

# ── Config ────────────────────────────────────────────────────────────────────
GEO_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE131nnn/GSE131907/suppl/"
    "GSE131907_Lung_Cancer_cell_annotation.txt.gz"
)
CHUNK_SIZE = 1024 * 1024  # 1 MB

logging.basicConfig(
    format="[%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("download_singlecell.log"),
    ],
)
log = logging.getLogger("download-singlecell")


def download_with_progress(url: str, dest: Path):
    """Stream download with progress display."""
    log.info(f"Downloading: {url}")
    log.info(f"Destination: {dest}")

    dest.parent.mkdir(parents=True, exist_ok=True)

    # Skip if already exists and non-empty
    if dest.exists() and dest.stat().st_size > 1_000_000:
        log.info(f"File already exists ({dest.stat().st_size / 1e9:.2f} GB), skipping.")
        return

    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    start = time.time()

    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    mb  = downloaded / 1e6
                    elapsed = time.time() - start
                    speed = mb / elapsed if elapsed > 0 else 0
                    print(
                        f"\r  {pct:5.1f}%  {mb:6.0f} MB / {total/1e6:.0f} MB"
                        f"  ({speed:.1f} MB/s)",
                        end="", flush=True,
                    )
    print()  # newline after progress
    size_gb = dest.stat().st_size / 1e9
    log.info(f"Done. File size: {size_gb:.2f} GB → {dest}")


def main():
    script_dir  = Path(__file__).parent
    project_dir = script_dir.parent
    dest = project_dir.parent / "modules/04_single_cell/data/input/GSE131907" / \
           "GSE131907_Lung_Cancer_cell_annotation.txt.gz"

    download_with_progress(GEO_URL, dest)
    log.info("Single-cell annotation file ready for M04.")


if __name__ == "__main__":
    main()
