#!/usr/bin/env python
"""
build_gtex_reference.py
-----------------------
Downloads GTEx v8 normal lung RNA-seq from UCSC Xena Toil (same pipeline as
TCGA Toil, so TCGA tumors and GTEx normals are directly comparable).

Strategy:
  1. Download GTEx sample attributes from GTEx portal (public, reliable)
     → identify lung sample IDs (SMTS == "Lung", ~570 samples)
  2. Use xenaPython API to fetch expression for lung samples from Xena Toil
     (uses GraphQL API, not /download/ which is now restricted)
  3. Compute per-gene stats: mean, std, median (log2 TPM scale)
  4. Save as data/databases/gtex_lung_normal_tpm_stats.tsv.gz

Requirements:
  pip install xenaPython requests numpy pandas

Output columns:
  ENSEMBL_GENE_ID | gtex_mean_log2tpm | gtex_std_log2tpm |
  gtex_median_log2tpm | gtex_n_samples

Note:
  Xena Toil values are log2(TPM + 0.001). Both TCGA and GTEx are processed
  with STAR + RSEM against GRCh38, making them directly comparable.

Usage:
  python data/scripts/build_gtex_reference.py
  python data/scripts/build_gtex_reference.py --force   # re-download
"""

import argparse
import gzip
import io
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
REF_DIR     = PROJECT_DIR / "databases"
OUT_PATH    = REF_DIR / "gtex_lung_normal_tpm_stats.tsv.gz"

# ── GTEx portal: sample attributes (public, no auth required) ─────────────────
GTEX_SAMPLE_ATTR_URL = (
    "https://storage.googleapis.com/adult-gtex/annotations/v8/"
    "metadata-files/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt"
)

# ── Xena Toil hub ─────────────────────────────────────────────────────────────
XENA_HOST    = "https://toil.xenahubs.net"
XENA_DATASET = "gtex_RSEM_gene_tpm"

logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
log = logging.getLogger("build-gtex-ref")


# ── Check xenaPython ──────────────────────────────────────────────────────────

def _require_xena_python():
    try:
        import xenaPython  # noqa: F401
    except ImportError:
        log.error(
            "xenaPython is required but not installed.\n"
            "  Install: pip install xenaPython\n"
            "  Then re-run this script."
        )
        sys.exit(1)


# ── Step 1: GTEx sample attributes → lung sample IDs ─────────────────────────

def load_lung_sample_ids(force: bool = False) -> list:
    """Download GTEx v8 sample attributes and return lung sample IDs."""
    cache = REF_DIR / "_cache_gtex_sample_attrs.tsv.gz"

    if cache.exists() and not force:
        log.info(f"Loading cached sample attributes: {cache}")
        attrs = pd.read_csv(cache, sep="\t", compression="gzip",
                            usecols=["SAMPID", "SMTS", "SMTSD"])
    else:
        log.info("Downloading GTEx v8 sample attributes from GTEx portal ...")
        log.info(f"  URL: {GTEX_SAMPLE_ATTR_URL}")
        resp = requests.get(GTEX_SAMPLE_ATTR_URL, timeout=120)
        resp.raise_for_status()
        log.info(f"  Done — {len(resp.content) / 1024:.0f} KB")

        attrs = pd.read_csv(io.StringIO(resp.text), sep="\t",
                            usecols=["SAMPID", "SMTS", "SMTSD"])
        REF_DIR.mkdir(parents=True, exist_ok=True)
        attrs.to_csv(cache, sep="\t", index=False, compression="gzip")

    lung = attrs[attrs["SMTS"].str.strip() == "Lung"]
    sample_ids = lung["SAMPID"].tolist()
    log.info(f"  Found {len(sample_ids)} GTEx lung samples (SMTS == 'Lung')")
    return sample_ids


# ── Step 2: Xena Toil expression for lung samples (xenaPython API) ────────────

def fetch_expression_xena(lung_samples: list, force: bool = False) -> pd.DataFrame:
    """
    Use xenaPython API to fetch GTEx expression for lung samples.
    xenaPython uses GraphQL/REST API (not /download/), so it works even
    when direct download is restricted.

    Returns: DataFrame with genes as rows, samples as columns (log2 TPM).
    """
    cache = REF_DIR / "_cache_gtex_lung_expression.tsv.gz"

    if cache.exists() and not force:
        log.info(f"Loading cached lung expression: {cache}")
        return pd.read_csv(cache, sep="\t", compression="gzip", index_col=0)

    import xenaPython as xena

    log.info(f"Fetching all gene IDs from Xena Toil dataset ...")
    all_genes = xena.dataset_field(XENA_HOST, XENA_DATASET)
    log.info(f"  {len(all_genes):,} genes in dataset")

    # Match lung samples against what's actually in Xena dataset
    log.info("Getting sample list from Xena to verify overlap ...")
    xena_samples = xena.dataset_samples(XENA_HOST, XENA_DATASET, None)
    xena_sample_set = set(xena_samples)
    matched = [s for s in lung_samples if s in xena_sample_set]
    log.info(f"  {len(matched)}/{len(lung_samples)} GTEx lung samples found in Xena")

    if not matched:
        log.error("No lung samples matched Xena dataset. Check sample ID format.")
        sys.exit(1)

    # Fetch expression in gene chunks to avoid timeout
    CHUNK = 2000
    n_chunks = (len(all_genes) + CHUNK - 1) // CHUNK
    log.info(f"Fetching expression in {n_chunks} chunks of {CHUNK} genes ...")

    expr_dict = {}  # gene_id → list of values per sample

    for i in range(0, len(all_genes), CHUNK):
        chunk_genes = all_genes[i : i + CHUNK]
        chunk_idx = i // CHUNK + 1
        log.info(f"  Chunk {chunk_idx}/{n_chunks} ...")

        try:
            # xenaPython returns list of lists: one per gene, values per sample
            data = xena.dataset_fetch(XENA_HOST, XENA_DATASET, matched, chunk_genes)
            for gene, vals in zip(chunk_genes, data):
                expr_dict[gene] = vals
        except Exception as e:
            log.warning(f"  Chunk {chunk_idx} failed: {e}, skipping")
            continue

    if not expr_dict:
        log.error("No expression data fetched.")
        sys.exit(1)

    log.info(f"  Fetched {len(expr_dict):,} genes for {len(matched)} samples")

    expr = pd.DataFrame(expr_dict, index=matched).T  # genes × samples
    expr = expr.apply(pd.to_numeric, errors="coerce")

    REF_DIR.mkdir(parents=True, exist_ok=True)
    expr.to_csv(cache, sep="\t", compression="gzip")
    log.info(f"  Cached → {cache}")
    return expr


# ── Alternative: Xena /data/ POST API (no xenaPython needed) ─────────────────

def fetch_expression_api(lung_samples: list, force: bool = False) -> pd.DataFrame:
    """
    Fetch GTEx expression via Xena /data/ POST API (no xenaPython needed).
    Falls back to this if xenaPython is unavailable.
    """
    cache = REF_DIR / "_cache_gtex_lung_expression.tsv.gz"

    if cache.exists() and not force:
        log.info(f"Loading cached lung expression: {cache}")
        return pd.read_csv(cache, sep="\t", compression="gzip", index_col=0)

    api_url = f"{XENA_HOST}/data/{XENA_DATASET}"
    log.info(f"Fetching expression via Xena /data/ API ...")
    log.info(f"  URL: {api_url}")
    log.info(f"  Samples: {len(lung_samples)}")

    try:
        resp = requests.post(api_url, json={"samples": lung_samples}, timeout=600)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        log.error(
            f"Xena API request failed: {e}\n\n"
            "MANUAL ALTERNATIVE:\n"
            "  1. Install xenaPython: pip install xenaPython\n"
            "  2. Re-run: python data/scripts/build_gtex_reference.py\n"
            "  OR download manually from:\n"
            "     https://xenabrowser.net/datapages/?dataset=gtex_RSEM_gene_tpm"
            "&host=https://toil.xenahubs.net\n"
            "  and place the file at:\n"
            f"     {cache}"
        )
        sys.exit(1)

    content = resp.text
    expr = pd.read_csv(io.StringIO(content), sep="\t", index_col=0)

    # Handle orientation: genes as rows (expected) or samples as rows
    lung_set = set(lung_samples)
    if not any(c in lung_set for c in expr.columns[:5]):
        # Try rows
        lung_rows = [r for r in expr.index if r in lung_set]
        if lung_rows:
            expr = expr.loc[lung_rows].T

    lung_cols = [c for c in expr.columns if c in lung_set]
    if lung_cols:
        expr = expr[lung_cols]

    log.info(f"  Matrix: {len(expr):,} genes × {len(expr.columns)} samples")

    REF_DIR.mkdir(parents=True, exist_ok=True)
    expr.to_csv(cache, sep="\t", compression="gzip")
    return expr


# ── Step 3: Compute per-gene statistics ───────────────────────────────────────

def compute_reference_stats(expr: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-gene mean, std, median across GTEx normal lung samples.
    Input:  genes × samples DataFrame (log2 TPM values)
    Output: DataFrame with ENSEMBL_GENE_ID + stats
    """
    log.info(f"Computing per-gene statistics ({len(expr):,} genes × "
             f"{len(expr.columns)} samples) ...")

    gene_ids_clean = expr.index.astype(str).str.split(".").str[0]
    vals = expr.values.astype(np.float32)

    stats = pd.DataFrame({
        "ENSEMBL_GENE_ID":      gene_ids_clean,
        "gtex_mean_log2tpm":    np.nanmean(vals, axis=1).round(4),
        "gtex_std_log2tpm":     np.nanstd(vals, axis=1, ddof=1).round(4),
        "gtex_median_log2tpm":  np.nanmedian(vals, axis=1).round(4),
        "gtex_n_samples":       np.sum(~np.isnan(vals), axis=1).astype(int),
    })

    n_before = len(stats)
    stats = stats[
        (stats["gtex_std_log2tpm"] > 0) &
        (stats["gtex_n_samples"] >= 10)
    ].reset_index(drop=True)
    log.info(f"  Retained {len(stats):,} genes "
             f"(removed {n_before - len(stats):,} with zero std or <10 samples)")
    return stats


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build GTEx normal lung reference for M03")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if cached files exist")
    parser.add_argument("--no-xena-python", action="store_true",
                        help="Use Xena /data/ API instead of xenaPython")
    args = parser.parse_args()

    if OUT_PATH.exists() and not args.force:
        log.info(f"Reference already exists: {OUT_PATH}")
        log.info("Use --force to re-download.")
        ref = pd.read_csv(OUT_PATH, sep="\t", compression="gzip")
        log.info(f"  {len(ref):,} genes | columns: {list(ref.columns)}")
        return

    REF_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: GTEx sample attributes → lung sample IDs
    lung_samples = load_lung_sample_ids(args.force)
    if not lung_samples:
        log.error("No lung samples found.")
        sys.exit(1)

    # Step 2: expression matrix
    use_xena_python = not args.no_xena_python
    if use_xena_python:
        try:
            import xenaPython  # noqa: F401
        except ImportError:
            log.warning("xenaPython not installed, falling back to /data/ API")
            log.warning("  Install with: pip install xenaPython")
            use_xena_python = False

    if use_xena_python:
        expr = fetch_expression_xena(lung_samples, args.force)
    else:
        expr = fetch_expression_api(lung_samples, args.force)

    # Step 3: compute stats
    stats = compute_reference_stats(expr)

    # Step 4: save
    stats.to_csv(OUT_PATH, sep="\t", index=False, compression="gzip")
    log.info(f"\nSaved GTEx normal lung reference → {OUT_PATH}")
    log.info(f"  {len(stats):,} genes | n_samples={stats['gtex_n_samples'].iloc[0]}")

    # Sanity check: EGFR
    egfr = stats[stats["ENSEMBL_GENE_ID"] == "ENSG00000146648"]
    if not egfr.empty:
        row = egfr.iloc[0]
        log.info(f"\nSanity check EGFR: mean={row['gtex_mean_log2tpm']:.2f} "
                 f"std={row['gtex_std_log2tpm']:.2f} (log2 TPM, normal lung)")


if __name__ == "__main__":
    main()
