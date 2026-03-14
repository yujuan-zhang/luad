#!/usr/bin/env python
"""
download_wsi.py
---------------
Download TCGA-LUAD whole slide image (WSI) thumbnails from GDC.

Two modes:
  --thumbnail   Download low-res PNG thumbnails (~100–500 KB each)  [default]
  --full        Download full SVS files (~500 MB–2 GB each)         [requires storage]

Thumbnails are sufficient for M08 pathology report generation (H&E color
analysis, tissue segmentation, TIL density scoring) without GPU.

Output:
  modules/08_pathology/data/input/thumbnails/{case_id}.png   (thumbnail mode)
  modules/08_pathology/data/input/svs/{case_id}.svs          (full mode)

Usage:
  python download_wsi.py --thumbnail               # all patients, thumbnails
  python download_wsi.py --thumbnail --test        # first 5 patients
  python download_wsi.py --thumbnail --resume      # skip already downloaded
  python download_wsi.py --full --sample TCGA-49-4507  # single full SVS
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import requests

# ── Config ────────────────────────────────────────────────────────────────────
GDC_FILES_URL = "https://api.gdc.cancer.gov/files"
GDC_DATA_URL  = "https://api.gdc.cancer.gov/data"
PROJECT_ID    = "TCGA-LUAD"
THUMBNAIL_SIZE = 1000   # max dimension in pixels for GDC thumbnail API
BATCH_SIZE    = 20
MAX_RETRIES   = 3

logging.basicConfig(
    format="[%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("download_wsi.log"),
    ],
)
log = logging.getLogger("download-wsi")


# ── Step 1: Query GDC for diagnostic slide files ──────────────────────────────

def query_slide_files(sample_id: str = None) -> list:
    """Return list of {file_id, case_id, file_name} for diagnostic slides."""
    filters = {
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "cases.project.project_id", "value": PROJECT_ID}},
            {"op": "=", "content": {"field": "data_type",                "value": "Slide Image"}},
            {"op": "=", "content": {"field": "experimental_strategy",    "value": "Diagnostic Slide"}},
        ]
    }
    if sample_id:
        filters["content"].append(
            {"op": "=", "content": {"field": "cases.submitter_id", "value": sample_id}}
        )

    payload = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,cases.submitter_id,cases.samples.sample_type",
        "format": "json",
        "size": "2000",
    }
    log.info(f"Querying GDC for {PROJECT_ID} diagnostic slide files...")
    resp = requests.get(GDC_FILES_URL, params=payload, timeout=60)
    resp.raise_for_status()
    hits = resp.json()["data"]["hits"]
    log.info(f"  Found {len(hits)} slide files")

    rows = []
    seen = set()
    for h in hits:
        fid   = h["file_id"]
        fname = h["file_name"]
        cases = h.get("cases", [])
        if not cases:
            continue
        case_id = cases[0].get("submitter_id", "")
        if case_id in seen:          # one slide per patient
            continue
        seen.add(case_id)
        rows.append({"file_id": fid, "case_id": case_id, "file_name": fname})

    log.info(f"  Unique patients with slides: {len(rows)}")
    return rows


# ── Step 2a: Download thumbnail via GDC API ───────────────────────────────────

def extract_thumbnail_from_svs(svs_path: Path, png_path: Path) -> bool:
    """
    Extract low-resolution thumbnail from SVS pyramid using tifffile.
    SVS files contain multiple resolution levels; Level 1 is the thumbnail.
    """
    try:
        import tifffile
        from PIL import Image

        with tifffile.TiffFile(str(svs_path)) as tif:
            # Level 1 is the thumbnail (~768×983 px); Level 0 is full resolution
            if len(tif.series) >= 2:
                thumb_array = tif.series[1].asarray()
            else:
                # Only one level — downsample level 0 to 1000px wide
                full = tif.series[0].asarray()
                h, w = full.shape[:2]
                scale = 1000 / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                img = Image.fromarray(full)
                img = img.resize((new_w, new_h), Image.LANCZOS)
                img.save(png_path)
                return True

        img = Image.fromarray(thumb_array)
        img.save(png_path)
        return True
    except Exception as e:
        log.error(f"  Thumbnail extraction failed: {e}")
        return False


def download_thumbnail(file_id: str, case_id: str, out_dir: Path,
                       svs_dir: Path, keep_svs: bool = False) -> bool:
    """
    Download SVS from GDC, extract PNG thumbnail, optionally delete SVS.
    GDC does not provide standalone thumbnail endpoints for WSI files.
    """
    png_path = out_dir / f"{case_id}.png"
    if png_path.exists() and png_path.stat().st_size > 10_000:
        log.info(f"  Already exists: {case_id}.png")
        return True

    svs_path = svs_dir / f"{case_id}.svs"

    # Download SVS if not already present
    if not (svs_path.exists() and svs_path.stat().st_size > 1_000_000):
        log.info(f"  Downloading SVS for {case_id}...")
        ok = _stream_download(f"{GDC_DATA_URL}/{file_id}", svs_path)
        if not ok:
            return False
        size_mb = svs_path.stat().st_size / 1e6
        log.info(f"  SVS downloaded ({size_mb:.0f} MB)")

    # Extract thumbnail
    log.info(f"  Extracting thumbnail from SVS...")
    ok = extract_thumbnail_from_svs(svs_path, png_path)
    if ok:
        size_kb = png_path.stat().st_size / 1024
        log.info(f"  ✓ {case_id}.png  ({size_kb:.0f} KB)")
        if not keep_svs:
            svs_path.unlink()
            log.info(f"  SVS deleted to save space")
        return True
    return False


def _stream_download(url: str, dest: Path) -> bool:
    """Stream download a file, return True on success."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=300, stream=True)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
            return True
        except Exception as e:
            log.warning(f"  Download attempt {attempt}/{MAX_RETRIES}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(5 * attempt)
    return False


# ── Step 2b: Download full SVS ────────────────────────────────────────────────

def download_svs(file_id: str, case_id: str, out_dir: Path) -> bool:
    """Download full SVS file (large, ~500MB–2GB each)."""
    out_path = out_dir / f"{case_id}.svs"
    if out_path.exists() and out_path.stat().st_size > 1_000_000:
        log.info(f"  Already exists: {case_id} ({out_path.stat().st_size / 1e9:.2f} GB)")
        return True

    url = f"{GDC_DATA_URL}/{file_id}"
    log.info(f"  Downloading SVS for {case_id}...")
    try:
        resp = requests.get(url, timeout=300, stream=True)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {pct:5.1f}%  {downloaded/1e9:.2f} GB", end="", flush=True)
        print()
        log.info(f"  ✓ {case_id} ({out_path.stat().st_size / 1e9:.2f} GB)")
        return True
    except Exception as e:
        log.error(f"  ✗ {case_id}: {e}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download TCGA-LUAD WSI files")
    parser.add_argument("--thumbnail", action="store_true", default=True,
                        help="Download low-res PNG thumbnails (default)")
    parser.add_argument("--full",      action="store_true",
                        help="Download full SVS files (large)")
    parser.add_argument("--test",      action="store_true",
                        help="First 5 patients only")
    parser.add_argument("--resume",    action="store_true",
                        help="Skip already-downloaded files")
    parser.add_argument("--sample",    type=str,
                        help="Single patient ID (e.g. TCGA-49-4507)")
    parser.add_argument("--keep-svs",  action="store_true",
                        help="Keep SVS file after extracting thumbnail")
    args = parser.parse_args()

    script_dir  = Path(__file__).parent
    project_dir = script_dir.parent
    module_dir  = project_dir.parent / "modules/08_pathology/data/input"
    thumb_dir   = module_dir / "thumbnails"
    svs_dir     = module_dir / "svs"
    thumb_dir.mkdir(parents=True, exist_ok=True)
    svs_dir.mkdir(parents=True, exist_ok=True)

    slides = query_slide_files(args.sample)

    if args.test:
        slides = slides[:5]
        log.info(f"[TEST MODE] {len(slides)} slides")

    if args.resume and not args.full:
        existing = {f.stem for f in thumb_dir.glob("*.png")}
        slides = [s for s in slides if s["case_id"] not in existing]
        log.info(f"[RESUME] {len(slides)} slides remaining")

    total   = len(slides)
    success = 0
    failed  = []

    for i, slide in enumerate(slides, 1):
        log.info(f"[{i}/{total}] {slide['case_id']}")
        if args.full:
            ok = download_svs(slide["file_id"], slide["case_id"], svs_dir)
        else:
            ok = download_thumbnail(
                slide["file_id"], slide["case_id"],
                thumb_dir, svs_dir,
                keep_svs=args.full or args.keep_svs,
            )
        if ok:
            success += 1
        else:
            failed.append(slide["case_id"])

    log.info(f"\n{'='*50}")
    log.info(f"Done. Success: {success}/{total}  Failed: {len(failed)}")
    if failed:
        log.warning(f"Failed: {failed}")


if __name__ == "__main__":
    main()
