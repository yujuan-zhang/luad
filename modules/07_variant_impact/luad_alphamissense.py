#!/usr/bin/env python
"""
luad_alphamissense.py
---------------------
Module 07: AlphaMissense pathogenicity scoring for LUAD missense mutations.

For each patient:
  1. Load missense mutations from M02 (SYMBOL + HGVSp_Short)
  2. Map gene symbol → UniProt ID  (UniProt REST API, cached)
  3. Parse HGVSp_Short → protein_variant string  (e.g. p.L858R → L858R)
  4. Lookup AlphaMissense table → am_pathogenicity, am_class
  5. Filter: COSMIC cancer genes + Pathogenic (am_class = likely_pathogenic)
  6. Save per-patient TSV + cohort summary

Output: data/output/07_variant_impact/{sample}/{sample}_alphamissense.tsv
        data/output/07_variant_impact/alphamissense_summary.tsv

Usage:
  python luad_alphamissense.py                        # all samples
  python luad_alphamissense.py --sample TCGA-05-4244  # single sample
"""

import argparse
import gzip
import json
import logging
import re
import time
from pathlib import Path

import pandas as pd
import requests

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
VARIANT_DIR = PROJECT_DIR / "data/output/02_variants"
OUT_DIR     = PROJECT_DIR / "data/output/07_variant_impact"
AM_FILE     = PROJECT_DIR / "data/input/alphamissense/AlphaMissense_aa_substitutions.tsv.gz"
CACHE_DIR   = SCRIPT_DIR / "data/input"
UNIPROT_CACHE = CACHE_DIR / "uniprot_id_cache.json"

# ── LUAD core driver genes (always included regardless of COSMIC) ──────────────
LUAD_DRIVERS = {
    "EGFR":   "P00533",
    "KRAS":   "P01116",
    "TP53":   "P04637",
    "STK11":  "Q15831",
    "KEAP1":  "Q14145",
    "BRAF":   "P15056",
    "MET":    "P08581",
    "ERBB2":  "P04626",
    "NF1":    "P21359",
    "CDKN2A": "P42771",
    "RB1":    "P06400",
    "PIK3CA": "P42336",
    "PTEN":   "P60484",
    "ALK":    "Q9UM73",
    "ROS1":   "P08922",
    "RET":    "P07949",
    "MAP2K1": "Q02750",
    "NRAS":   "P01111",
    "HRAS":   "P01112",
    "MYC":    "P01106",
    "SMARCA4":"P51532",
    "ARID1A": "O14497",
    "SETD2":  "Q9BYW2",
    "RBM10":  "P98175",
    "U2AF1":  "Q01081",
}


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
    return logging.getLogger(name)


# ── UniProt ID lookup ──────────────────────────────────────────────────────────

def load_uniprot_cache() -> dict:
    if UNIPROT_CACHE.exists():
        return json.loads(UNIPROT_CACHE.read_text())
    return dict(LUAD_DRIVERS)   # seed with known IDs


def save_uniprot_cache(cache: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    UNIPROT_CACHE.write_text(json.dumps(cache, indent=2))


def fetch_uniprot_id(gene_symbol: str, cache: dict, logger) -> str | None:
    """Return canonical UniProt accession for human gene, with caching."""
    if gene_symbol in cache:
        return cache[gene_symbol]

    url = (
        "https://rest.uniprot.org/uniprotkb/search"
        f"?query=gene_exact:{gene_symbol}+AND+organism_id:9606+AND+reviewed:true"
        "&fields=accession&format=json&size=1"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        results = r.json().get("results", [])
        if results:
            uid = results[0]["primaryAccession"]
            cache[gene_symbol] = uid
            return uid
    except Exception as e:
        logger.debug(f"UniProt lookup failed for {gene_symbol}: {e}")
    cache[gene_symbol] = None
    return None


# ── AlphaMissense table ────────────────────────────────────────────────────────

_AM_TABLE: dict | None = None   # lazy-loaded {(uniprot_id, variant): (score, class)}
AM_SUBSET_CACHE = CACHE_DIR / "am_subset.tsv.gz"  # pre-filtered subset


def build_am_subset(relevant_uids: set, logger) -> None:
    """
    One-time: scan full AlphaMissense table, keep only rows for relevant_uids.
    Saves a small subset file for fast future loads.
    """
    if not AM_FILE.exists():
        logger.error(f"AlphaMissense file not found: {AM_FILE}")
        return
    logger.info(
        f"Building AM subset for {len(relevant_uids)} UniProt IDs "
        f"(scanning {AM_FILE.name}) ..."
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    kept = 0
    with gzip.open(AM_FILE, "rt") as fin, \
         gzip.open(AM_SUBSET_CACHE, "wt") as fout:
        for line in fin:
            if line.startswith("#"):
                continue
            if line.startswith("uniprot"):
                fout.write(line)
                continue
            uid = line.split("\t", 1)[0]
            if uid in relevant_uids:
                fout.write(line)
                kept += 1
    logger.info(f"  Subset: {kept:,} entries → {AM_SUBSET_CACHE}")


def load_am_table(relevant_uids: set | None, logger) -> dict:
    """
    Load AlphaMissense lookup dict {(uniprot_id, variant): (score, class)}.
    If subset cache exists, load from it (fast).
    Otherwise build subset first (slow, one-time).
    """
    global _AM_TABLE
    if _AM_TABLE is not None:
        return _AM_TABLE

    # Rebuild subset if new UIDs are not covered
    if not AM_SUBSET_CACHE.exists() and relevant_uids:
        build_am_subset(relevant_uids, logger)
    elif AM_SUBSET_CACHE.exists() and relevant_uids:
        # Check if any UIDs are missing from current subset
        existing_uids: set = set()
        with gzip.open(AM_SUBSET_CACHE, "rt") as f:
            for line in f:
                if line.startswith("#") or line.startswith("uniprot"):
                    continue
                existing_uids.add(line.split("\t", 1)[0])
        missing = relevant_uids - existing_uids
        if missing:
            logger.info(f"  {len(missing)} new UniProt IDs — rebuilding subset")
            build_am_subset(relevant_uids, logger)

    src = AM_SUBSET_CACHE if AM_SUBSET_CACHE.exists() else AM_FILE
    logger.info(f"Loading AlphaMissense from {src.name} ...")
    am = {}
    with gzip.open(src, "rt") as f:
        for line in f:
            if line.startswith("#") or line.startswith("uniprot"):
                continue
            parts = line.rstrip().split("\t")
            if len(parts) < 4:
                continue
            uid, variant, score, cls = parts[0], parts[1], parts[2], parts[3]
            am[(uid, variant)] = (float(score), cls)
    logger.info(f"  Loaded {len(am):,} entries")
    _AM_TABLE = am
    return am


# ── Variant parsing ────────────────────────────────────────────────────────────

_HGVSP_RE = re.compile(r"^p\.([A-Z])(\d+)([A-Z])$")


def parse_hgvsp(hgvsp: str) -> str | None:
    """
    Convert HGVSp_Short to AlphaMissense variant string.
    p.L858R  →  L858R
    Returns None if not a simple missense substitution.
    """
    m = _HGVSP_RE.match(str(hgvsp).strip())
    if not m:
        return None
    return m.group(1) + m.group(2) + m.group(3)


# ── Per-sample pipeline ────────────────────────────────────────────────────────

def run_sample(
    sample_id: str,
    am_table: dict,
    uniprot_cache: dict,
    logger,
) -> dict:
    out_dir = OUT_DIR / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sample_id}_alphamissense.tsv"

    # Load M02 variants
    var_path = VARIANT_DIR / sample_id / f"{sample_id}_variants.tsv.gz"
    if not var_path.exists():
        logger.info(f"  {sample_id}: no variants file — skip")
        return {"sample_id": sample_id, "n_missense": 0, "n_scored": 0, "n_pathogenic": 0}

    df = pd.read_csv(var_path, sep="\t", compression="gzip", low_memory=False)

    cons_col = next((c for c in ["CONSEQUENCE", "Consequence"] if c in df.columns), None)
    if cons_col is None or "HGVSp_Short" not in df.columns:
        return {"sample_id": sample_id, "n_missense": 0, "n_scored": 0, "n_pathogenic": 0}

    missense = df[df[cons_col].str.contains("missense", case=False, na=False)].copy()
    if missense.empty:
        return {"sample_id": sample_id, "n_missense": 0, "n_scored": 0, "n_pathogenic": 0}

    logger.info(f"  {sample_id}: {len(missense)} missense mutations")

    # Build lookup for each gene
    genes = missense["SYMBOL"].unique()
    for gene in genes:
        if gene not in uniprot_cache:
            uid = fetch_uniprot_id(gene, uniprot_cache, logger)
            time.sleep(0.15)

    # Score each mutation
    records = []
    for _, row in missense.iterrows():
        gene    = str(row["SYMBOL"])
        hgvsp   = str(row.get("HGVSp_Short", ""))
        variant = parse_hgvsp(hgvsp)
        if variant is None:
            continue

        uid = uniprot_cache.get(gene)
        if uid is None:
            continue

        key = (uid, variant)
        if key in am_table:
            score, cls = am_table[key]
        else:
            score, cls = None, "not_found"

        records.append({
            "sample_id":        sample_id,
            "gene":             gene,
            "hgvsp":            hgvsp,
            "uniprot_id":       uid,
            "am_variant":       variant,
            "am_pathogenicity": score,
            "am_class":         cls,
            "is_luad_driver":   gene in LUAD_DRIVERS,
            "impact":           row.get("IMPACT", ""),
            "existing_class":   row.get("FUNCTIONAL_IMPACT", ""),
            "civic_level":      row.get("CIVIC_LEVEL", ""),
        })

    if not records:
        return {"sample_id": sample_id, "n_missense": len(missense), "n_scored": 0, "n_pathogenic": 0}

    result_df = pd.DataFrame(records)
    result_df = result_df.sort_values("am_pathogenicity", ascending=False, na_position="last")
    result_df.to_csv(out_path, sep="\t", index=False)

    n_scored     = result_df["am_pathogenicity"].notna().sum()
    n_pathogenic = (result_df["am_class"] == "pathogenic").sum()
    n_driver_path = (
        (result_df["am_class"] == "pathogenic") &
        result_df["is_luad_driver"]
    ).sum()

    logger.info(
        f"    scored={n_scored}, pathogenic={n_pathogenic}, "
        f"driver+pathogenic={n_driver_path}"
    )
    if n_driver_path > 0:
        top = result_df[
            (result_df["am_class"] == "likely_pathogenic") &
            result_df["is_luad_driver"]
        ].head(5)
        for _, r in top.iterrows():
            logger.info(
                f"      {r['gene']} {r['hgvsp']}  score={r['am_pathogenicity']:.3f}"
            )

    return {
        "sample_id":        sample_id,
        "n_missense":       len(missense),
        "n_scored":         int(n_scored),
        "n_pathogenic":     int(n_pathogenic),
        "n_driver_pathogenic": int(n_driver_path),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=str, help="Single sample ID")
    args   = parser.parse_args()
    logger = get_logger("luad-alphamissense")

    # Load UniProt cache
    uniprot_cache = load_uniprot_cache()
    logger.info(f"UniProt cache: {len(uniprot_cache)} entries pre-loaded")

    samples = (
        [args.sample] if args.sample
        else sorted(p.name for p in VARIANT_DIR.glob("*/") if p.is_dir())
    )
    logger.info(f"Processing {len(samples)} samples")

    # Step 1: collect all unique gene symbols across all samples
    logger.info("Scanning variant files for unique gene symbols ...")
    all_genes: set = set(LUAD_DRIVERS.keys())
    for s in samples:
        var_path = VARIANT_DIR / s / f"{s}_variants.tsv.gz"
        if not var_path.exists():
            continue
        try:
            df = pd.read_csv(var_path, sep="\t", compression="gzip",
                             usecols=["SYMBOL", "CONSEQUENCE"], low_memory=False)
            missense = df[df["CONSEQUENCE"].str.contains("missense", case=False, na=False)]
            all_genes.update(missense["SYMBOL"].dropna().unique())
        except Exception:
            pass
    logger.info(f"  {len(all_genes)} unique genes across all samples")

    # Step 2: resolve UniProt IDs for all genes
    logger.info("Resolving UniProt IDs ...")
    for gene in sorted(all_genes):
        if gene not in uniprot_cache:
            fetch_uniprot_id(gene, uniprot_cache, logger)
            time.sleep(0.15)
    save_uniprot_cache(uniprot_cache)
    relevant_uids = {v for v in uniprot_cache.values() if v}
    logger.info(f"  {len(relevant_uids)} UniProt IDs resolved")

    # Step 3: load AlphaMissense subset for relevant UniProt IDs
    am_table = load_am_table(relevant_uids, logger)
    if not am_table:
        logger.error("Cannot proceed without AlphaMissense table")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for s in samples:
        try:
            r = run_sample(s, am_table, uniprot_cache, logger)
        except Exception as e:
            logger.error(f"{s} failed: {e}")
            r = {"sample_id": s, "n_missense": 0, "n_scored": 0, "n_pathogenic": 0}
        results.append(r)
        # Save UniProt cache periodically
        if len(results) % 50 == 0:
            save_uniprot_cache(uniprot_cache)

    save_uniprot_cache(uniprot_cache)

    summary = pd.DataFrame(results)
    summary_path = OUT_DIR / "alphamissense_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    logger.info(f"\nDone. Summary → {summary_path}")
    print(summary.describe().to_string())


if __name__ == "__main__":
    main()
