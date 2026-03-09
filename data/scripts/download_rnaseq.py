#!/usr/bin/env python
"""
download_rnaseq.py
------------------
Download TCGA-LUAD RNA-seq STAR Counts from GDC, extract TPM column,
and save per-patient files to data/rnaseq/{case_id}.tsv.gz

Output format (matches M03 luad_expression.py expectation):
  TargetID   TPM
  ENSG...    12.34
  ...

Usage:
  python download_rnaseq.py              # all TCGA-LUAD samples
  python download_rnaseq.py --test       # first 3 samples only
  python download_rnaseq.py --resume     # skip already-downloaded files
"""

import argparse
import gzip
import io
import json
import logging
import re
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import pandas as pd
import requests

# ── Config ────────────────────────────────────────────────────────────────────
GDC_FILES_URL = "https://api.gdc.cancer.gov/files"
GDC_DATA_URL  = "https://api.gdc.cancer.gov/data"
PROJECT_ID    = "TCGA-LUAD"
BATCH_SIZE    = 50
MAX_RETRIES   = 3

logging.basicConfig(
    format="[%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("download_rnaseq.log"),
    ],
)
log = logging.getLogger("download-rnaseq")


# ── Step 1: Query GDC for all TCGA-LUAD STAR Counts files ────────────────────

def query_rnaseq_files() -> pd.DataFrame:
    """Return DataFrame with file_id, case_id, file_name for all STAR Counts files."""
    log.info(f"Querying GDC for {PROJECT_ID} RNA-seq STAR Counts files...")
    payload = {
        "filters": json.dumps({
            "op": "and",
            "content": [
                {"op": "=", "content": {"field": "cases.project.project_id",   "value": PROJECT_ID}},
                {"op": "=", "content": {"field": "data_type",                  "value": "Gene Expression Quantification"}},
                {"op": "=", "content": {"field": "experimental_strategy",      "value": "RNA-Seq"}},
                {"op": "=", "content": {"field": "analysis.workflow_type",     "value": "STAR - Counts"}},
                {"op": "=", "content": {"field": "data_format",                "value": "TSV"}},
            ]
        }),
        "fields": "file_id,file_name,cases.submitter_id,cases.samples.sample_type",
        "format": "json",
        "size": "2000",
    }
    resp = requests.get(GDC_FILES_URL, params=payload, timeout=60)
    resp.raise_for_status()
    hits = resp.json()["data"]["hits"]
    log.info(f"  Found {len(hits)} RNA-seq files")

    rows = []
    for h in hits:
        file_id   = h["file_id"]
        file_name = h["file_name"]
        cases     = h.get("cases", [])
        if not cases:
            continue
        case      = cases[0]
        case_id   = case.get("submitter_id", "")

        # Prefer Primary Tumor; skip blood derived normal
        samples   = case.get("samples", [])
        sample_types = [s.get("sample_type", "") for s in samples]
        if any("Normal" in st for st in sample_types) and not any("Tumor" in st for st in sample_types):
            continue

        rows.append({"file_id": file_id, "case_id": case_id, "file_name": file_name})

    df = pd.DataFrame(rows)

    # One file per case — keep first occurrence (primary tumor preferred by GDC ordering)
    df = df.drop_duplicates(subset="case_id", keep="first").reset_index(drop=True)
    log.info(f"  Unique cases after dedup: {len(df)}")
    return df


# ── Step 2: Download in batches ───────────────────────────────────────────────

def download_batch(file_ids: list, out_dir: Path) -> dict:
    """POST batch of file IDs to GDC /data endpoint, return {file_id: bytes}."""
    payload = {"ids": file_ids}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                GDC_DATA_URL,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=300,
                stream=True,
            )
            resp.raise_for_status()
            break
        except Exception as e:
            log.warning(f"  Batch attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(5 * attempt)

    # Response is a tar archive
    content = resp.content
    results = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(content)) as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                # Extract file_id from path: {file_id}/{filename}
                parts = Path(member.name).parts
                if len(parts) >= 2:
                    fid = parts[0]
                else:
                    continue
                f = tar.extractfile(member)
                if f:
                    results[fid] = (Path(member.name).name, f.read())
    except Exception as e:
        log.error(f"  Failed to parse tar response: {e}")
    return results


# ── Step 3: Parse STAR Counts TSV → TargetID + TPM ───────────────────────────

def parse_star_counts(raw_bytes: bytes, file_name: str) -> pd.DataFrame:
    """
    Parse GDC STAR Counts TSV (possibly gzipped).
    Columns include: gene_id, gene_name, ..., tpm_unstranded, ...
    Returns DataFrame with columns: TargetID, TPM
    """
    is_gz = file_name.endswith(".gz")
    if is_gz:
        data = gzip.decompress(raw_bytes)
    else:
        data = raw_bytes

    text = data.decode("utf-8")

    # Skip comment lines (#) and GDC summary rows (N_unmapped etc.)
    lines = [l for l in text.splitlines()
             if not l.startswith("#") and not l.startswith("N_")]
    content = "\n".join(lines)

    df = pd.read_csv(io.StringIO(content), sep="\t", low_memory=False)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Find gene_id column
    gene_col = next((c for c in df.columns if "gene_id" in c), None)
    # Find TPM column
    tpm_col  = next((c for c in df.columns if "tpm" in c and "unstranded" in c), None)
    if tpm_col is None:
        tpm_col = next((c for c in df.columns if "tpm" in c), None)

    if gene_col is None or tpm_col is None:
        raise ValueError(f"Cannot find gene_id or tpm column. Columns: {list(df.columns)}")

    out = df[[gene_col, tpm_col]].copy()
    out.columns = ["TargetID", "TPM"]
    out = out[out["TargetID"].notna()].reset_index(drop=True)
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download TCGA-LUAD RNA-seq TPM files")
    parser.add_argument("--test",   action="store_true", help="Download first 3 samples only")
    parser.add_argument("--resume", action="store_true", help="Skip already-downloaded files")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    script_dir  = Path(__file__).parent
    project_dir = script_dir.parent
    out_dir     = Path(args.out_dir) if args.out_dir else project_dir / "rnaseq"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Query file manifest
    manifest = query_rnaseq_files()

    if args.test:
        manifest = manifest.head(3)
        log.info(f"[TEST MODE] Downloading {len(manifest)} samples")

    # Skip already downloaded
    if args.resume:
        existing = {f.stem.replace(".tsv", "") for f in out_dir.glob("*.tsv.gz")}
        before = len(manifest)
        manifest = manifest[~manifest["case_id"].isin(existing)].reset_index(drop=True)
        log.info(f"[RESUME] Skipping {before - len(manifest)} already-downloaded files, {len(manifest)} remaining")

    total    = len(manifest)
    n_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    log.info(f"\nDownloading {total} files in {n_batches} batches of {BATCH_SIZE}...")

    success = 0
    failed  = []

    for batch_idx in range(n_batches):
        batch = manifest.iloc[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]
        file_ids = batch["file_id"].tolist()
        id_to_case = dict(zip(batch["file_id"], batch["case_id"]))

        log.info(f"\nBatch {batch_idx + 1}/{n_batches} ({len(file_ids)} files)...")

        try:
            results = download_batch(file_ids, out_dir)
        except Exception as e:
            log.error(f"  Batch {batch_idx + 1} failed entirely: {e}")
            failed.extend(batch["case_id"].tolist())
            continue

        for file_id, (file_name, raw_bytes) in results.items():
            case_id = id_to_case.get(file_id, file_id)
            out_path = out_dir / f"{case_id}.tsv.gz"
            try:
                df = parse_star_counts(raw_bytes, file_name)
                df.to_csv(out_path, sep="\t", index=False, compression="gzip")
                success += 1
                log.info(f"  ✓ {case_id}  ({len(df):,} genes)")
            except Exception as e:
                log.error(f"  ✗ {case_id} parse error: {e}")
                failed.append(case_id)

    log.info(f"\n{'='*50}")
    log.info(f"Done. Success: {success}/{total}  Failed: {len(failed)}")
    if failed:
        log.warning(f"Failed cases: {failed}")


if __name__ == "__main__":
    main()
