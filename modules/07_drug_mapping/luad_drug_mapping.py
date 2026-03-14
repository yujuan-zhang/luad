#!/usr/bin/env python
"""
luad_drug_mapping.py
--------------------
Module 07: Mutation-to-drug mapping for LUAD precision oncology.

For each sample, integrates:
  - Somatic mutations (module 02): driver genes, hotspot flags, TMB
  - Clinical data   (module 01): stage, smoking, gender
  - Embedded NCCN/FDA drug knowledge base (LUAD-specific, curated)
  - Optional CIViC API enrichment (free, no token required)

Outputs per sample:
  {sample}_drug_report.tsv      — actionable mutations × drug recommendations
  {sample}_drug_report.png      — 4-panel clinical actionability figure

All-sample summary:
  drug_actionability_heatmap.png — drugs × samples heatmap
  all_samples_drug_summary.tsv

Knowledge base covers:
  EGFR, KRAS (G12C), BRAF (V600E), MET (exon14), RET, ALK, ROS1, ERBB2,
  NTRK1/2/3, STK11, KEAP1, TP53, FGFR1, MAP2K1, PIK3CA
  + Immunotherapy (PD-L1/TMB) and platinum-based chemotherapy

Usage:
  python luad_drug_mapping.py                        # all samples
  python luad_drug_mapping.py --sample TCGA-86-A4D0  # single sample
  python luad_drug_mapping.py --dry_run              # validate inputs only
"""

import argparse
import logging
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

import requests

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
VARIANT_DIR = PROJECT_DIR / "data/output/02_variants"
CLINICAL_DIR= PROJECT_DIR / "data/output/01_patients"
OUT_DIR     = PROJECT_DIR / "data/output/07_drug_mapping"

# ══════════════════════════════════════════════════════════════════════════════
# Drug Knowledge Base  (curated from NCCN NSCLC 2024, FDA approvals, CIViC)
# ══════════════════════════════════════════════════════════════════════════════

DRUG_KB = {
    # ── Oncogenes ────────────────────────────────────────────────────────────
    "EGFR": {
        "category": "Oncogene",
        "color": "#e74c3c",
        "drugs": [
            # Standard sensitizing mutations (exon 19 del, L858R, G719X, S768I, L861Q)
            {"drug": "Osimertinib",   "class": "3rd-gen EGFR TKI",      "line": "1L",
             "level": "FDA-Approved", "pattern": r"L858R|exon19|sensitiz|G719|S768|L861"},
            {"drug": "Erlotinib",     "class": "1st-gen EGFR TKI",      "line": "1L",
             "level": "FDA-Approved", "pattern": r"L858R|exon19|sensitiz|G719|S768|L861"},
            {"drug": "Afatinib",      "class": "2nd-gen EGFR TKI",      "line": "1L",
             "level": "FDA-Approved", "pattern": r"L858R|exon19|sensitiz|G719|S768|L861"},
            {"drug": "Dacomitinib",   "class": "2nd-gen EGFR TKI",      "line": "1L",
             "level": "FDA-Approved", "pattern": r"L858R|exon19|sensitiz"},
            # T790M resistance
            {"drug": "Osimertinib",   "class": "3rd-gen EGFR TKI",      "line": "2L",
             "level": "FDA-Approved", "pattern": r"T790M"},
            # Exon 20 insertions
            {"drug": "Amivantamab",   "class": "EGFR+MET bispecific Ab", "line": "1L",
             "level": "FDA-Approved", "pattern": r"exon20ins|exon_20_ins"},
            {"drug": "Mobocertinib",  "class": "EGFR TKI (exon20)",     "line": "2L",
             "level": "FDA-Approved", "pattern": r"exon20ins|exon_20_ins"},
            # Uncommon sensitizing mutations (G719X, S768I, L861Q)
            {"drug": "Lazertinib",    "class": "3rd-gen EGFR TKI",      "line": "1L",
             "level": "Clinical",     "pattern": r"G719|S768|L861"},
        ],
        "io_note": "Checkpoint inhibitors have limited benefit in EGFR-mutant LUAD",
    },
    "KRAS": {
        "category": "Oncogene",
        "color": "#e67e22",
        "drugs": [
            {"drug": "Sotorasib",          "class": "KRAS G12C inhibitor",       "line": "2L",
             "level": "FDA-Approved",      "pattern": r"G12C"},
            {"drug": "Adagrasib",          "class": "KRAS G12C inhibitor",       "line": "2L",
             "level": "FDA-Approved",      "pattern": r"G12C"},
            {"drug": "Adagrasib+Cetuximab","class": "KRAS G12C + anti-EGFR",     "line": "2L",
             "level": "FDA-Approved",      "pattern": r"G12C"},
        ],
        "non_g12c_note": "G12D/V/A: MRTX-1133, RMC-6236 in clinical trials (no FDA approval yet)",
    },
    "BRAF": {
        "category": "Oncogene",
        "color": "#9b59b6",
        "drugs": [
            {"drug": "Dabrafenib+Trametinib", "class": "BRAF+MEK inhibitor", "line": "1L",
             "level": "FDA-Approved",          "pattern": r"V600E"},
        ],
    },
    "MET": {
        "category": "Oncogene",
        "color": "#1abc9c",
        "drugs": [
            {"drug": "Tepotinib",   "class": "MET inhibitor", "line": "1L",
             "level": "FDA-Approved", "pattern": r"exon14|Y1003|splice"},
            {"drug": "Capmatinib",  "class": "MET inhibitor", "line": "1L",
             "level": "FDA-Approved", "pattern": r"exon14|Y1003|splice"},
            {"drug": "Crizotinib",  "class": "ALK/ROS1/MET",  "line": "1L",
             "level": "FDA-Approved", "pattern": r"exon14|amplif"},
        ],
    },
    "RET": {
        "category": "Oncogene",
        "color": "#3498db",
        "drugs": [
            {"drug": "Selpercatinib", "class": "RET-selective inhibitor", "line": "1L",
             "level": "FDA-Approved",  "pattern": r"fusion|.*"},
            {"drug": "Pralsetinib",   "class": "RET-selective inhibitor", "line": "1L",
             "level": "FDA-Approved",  "pattern": r"fusion|.*"},
        ],
    },
    "ALK": {
        "category": "Oncogene",
        "color": "#2ecc71",
        "drugs": [
            {"drug": "Alectinib",   "class": "2nd-gen ALK TKI", "line": "1L",
             "level": "FDA-Approved", "pattern": r"fusion|.*"},
            {"drug": "Brigatinib",  "class": "2nd-gen ALK TKI", "line": "1L",
             "level": "FDA-Approved", "pattern": r"fusion|.*"},
            {"drug": "Lorlatinib",  "class": "3rd-gen ALK TKI", "line": "1L",
             "level": "FDA-Approved", "pattern": r"fusion|.*"},
        ],
    },
    "ROS1": {
        "category": "Oncogene",
        "color": "#16a085",
        "drugs": [
            {"drug": "Entrectinib",  "class": "ROS1/NTRK inhibitor", "line": "1L",
             "level": "FDA-Approved", "pattern": r"fusion|.*"},
            {"drug": "Crizotinib",   "class": "ALK/ROS1/MET",        "line": "1L",
             "level": "FDA-Approved", "pattern": r"fusion|.*"},
            {"drug": "Lorlatinib",   "class": "3rd-gen ALK/ROS1 TKI","line": "2L",
             "level": "FDA-Approved", "pattern": r"fusion|.*"},
        ],
    },
    "ERBB2": {
        "category": "Oncogene",
        "color": "#c0392b",
        "drugs": [
            {"drug": "Trastuzumab deruxtecan", "class": "HER2 ADC",        "line": "2L",
             "level": "FDA-Approved",            "pattern": r".*"},
            {"drug": "Ado-trastuzumab emtansine","class": "HER2 ADC (T-DM1)","line": "2L",
             "level": "Clinical",               "pattern": r"amplif|overexp"},
        ],
    },
    "NTRK1": {
        "category": "Oncogene",
        "color": "#8e44ad",
        "drugs": [
            {"drug": "Larotrectinib","class": "pan-NTRK inhibitor",     "line": "1L",
             "level": "FDA-Approved", "pattern": r"fusion|.*"},
            {"drug": "Entrectinib",  "class": "NTRK/ROS1/ALK inhibitor","line": "1L",
             "level": "FDA-Approved", "pattern": r"fusion|.*"},
        ],
    },
    "FGFR1": {
        "category": "Oncogene",
        "color": "#d35400",
        "drugs": [
            {"drug": "Erdafitinib",   "class": "pan-FGFR inhibitor", "line": "2L",
             "level": "Clinical",      "pattern": r"amplif|fusion|mutation"},
            {"drug": "Infigratinib",  "class": "FGFR1-3 inhibitor",  "line": "2L",
             "level": "Clinical",      "pattern": r"amplif|fusion"},
        ],
    },
    "MAP2K1": {
        "category": "Oncogene",
        "color": "#7f8c8d",
        "drugs": [
            {"drug": "Trametinib",  "class": "MEK inhibitor", "line": "Investigational",
             "level": "Phase II",    "pattern": r".*"},
            {"drug": "Cobimetinib", "class": "MEK inhibitor", "line": "Investigational",
             "level": "Phase I/II",  "pattern": r".*"},
        ],
    },
    "PIK3CA": {
        "category": "Oncogene",
        "color": "#27ae60",
        "drugs": [
            {"drug": "Alpelisib",   "class": "PI3Kα inhibitor",   "line": "Investigational",
             "level": "Phase II",   "pattern": r"H1047|E545|E542"},
            {"drug": "Copanlisib",  "class": "pan-PI3K inhibitor", "line": "Investigational",
             "level": "Phase I/II", "pattern": r".*"},
        ],
    },
    # ── Tumor Suppressors ────────────────────────────────────────────────────
    "TP53": {
        "category": "Tumor Suppressor",
        "color": "#95a5a6",
        "drugs": [
            {"drug": "Eprenetapopt (APR-246)", "class": "TP53 reactivator", "line": "Investigational",
             "level": "Phase II",               "pattern": r"R175H|R248W|R248Q|R273H|R273C|G245S|R249S"},
        ],
        "note": (
            "TP53 hotspot mutations (R175H, R248W/Q, R273H/C, G245S, R249S): "
            "APR-246 in Phase II trials. Generally, TP53 loss = co-mutation target selection."
        ),
    },
    "STK11": {
        "category": "Tumor Suppressor",
        "color": "#bdc3c7",
        "drugs": [],
        "note": (
            "STK11 loss-of-function: no direct targeted therapy approved. "
            "Associated with PD-1/PD-L1 inhibitor resistance, especially with KRAS co-mutation. "
            "Platinum-based chemotherapy remains standard."
        ),
        "io_resistance": True,
    },
    "KEAP1": {
        "category": "Tumor Suppressor",
        "color": "#ecf0f1",
        "drugs": [],
        "note": (
            "KEAP1 loss activates NRF2/ARE pathway → oxidative stress resistance, "
            "poor response to platinum chemotherapy. "
            "NRF2 pathway inhibitors in early clinical trials."
        ),
    },
    "NF1": {
        "category": "Tumor Suppressor",
        "color": "#7f8c8d",
        "drugs": [
            {"drug": "Trametinib",     "class": "MEK inhibitor", "line": "Investigational",
             "level": "Phase II",       "pattern": r".*"},
            {"drug": "Binimetinib",    "class": "MEK inhibitor", "line": "Investigational",
             "level": "Phase I/II",    "pattern": r".*"},
        ],
        "note": "NF1 loss → RAS/MAPK pathway activation. MEK inhibitors in trials.",
    },
    "CDKN2A": {
        "category": "Tumor Suppressor",
        "color": "#bdc3c7",
        "drugs": [
            {"drug": "Palbociclib",   "class": "CDK4/6 inhibitor", "line": "Investigational",
             "level": "Phase II",      "pattern": r".*"},
            {"drug": "Abemaciclib",   "class": "CDK4/6 inhibitor", "line": "Investigational",
             "level": "Phase II",      "pattern": r".*"},
        ],
        "note": "CDKN2A loss → CDK4/6 hyperactivation. CDK4/6 inhibitors in NSCLC trials.",
    },
    "RB1": {
        "category": "Tumor Suppressor",
        "color": "#bdc3c7",
        "drugs": [],
        "note": "RB1 loss: no direct targeted therapy. May affect CDK4/6 inhibitor efficacy.",
    },
    "PTEN": {
        "category": "Tumor Suppressor",
        "color": "#bdc3c7",
        "drugs": [
            {"drug": "Copanlisib",   "class": "pan-PI3K inhibitor", "line": "Investigational",
             "level": "Phase II",     "pattern": r".*"},
            {"drug": "Everolimus",   "class": "mTOR inhibitor",     "line": "Investigational",
             "level": "Phase II",     "pattern": r".*"},
        ],
        "note": "PTEN loss → PI3K/AKT/mTOR pathway activation.",
    },
}

# ── General therapy (immunotherapy + chemo, TMB/stage based) ─────────────────
IMMUNOTHERAPY = [
    {"drug": "Pembrolizumab",       "class": "PD-1 inhibitor",      "line": "1L",
     "level": "FDA-Approved",       "condition": "TMB ≥ 10 mut/Mb, no EGFR/ALK"},
    {"drug": "Pembrolizumab+chemo", "class": "PD-1 + platinum",     "line": "1L",
     "level": "FDA-Approved",       "condition": "PD-L1 any, stage IV NSCLC"},
    {"drug": "Atezolizumab",        "class": "PD-L1 inhibitor",     "line": "1L",
     "level": "FDA-Approved",       "condition": "PD-L1 high, no EGFR/ALK"},
    {"drug": "Nivolumab",           "class": "PD-1 inhibitor",      "line": "2L",
     "level": "FDA-Approved",       "condition": "After platinum-based chemo"},
    {"drug": "Durvalumab",          "class": "PD-L1 inhibitor",     "line": "Consolidation",
     "level": "FDA-Approved",       "condition": "Stage III, post-chemoradiation"},
]

CHEMOTHERAPY = [
    {"drug": "Carboplatin+Pemetrexed",         "class": "Platinum doublet", "line": "1L",
     "level": "Standard of care",              "condition": "Non-squamous NSCLC"},
    {"drug": "Cisplatin+Pemetrexed",           "class": "Platinum doublet", "line": "1L",
     "level": "Standard of care",              "condition": "Non-squamous NSCLC"},
    {"drug": "Carboplatin+Pemetrexed+Pembro",  "class": "Chemo-immuno",    "line": "1L",
     "level": "FDA-Approved",                  "condition": "Non-squamous, stage IV"},
    {"drug": "Docetaxel",                      "class": "Taxane",           "line": "2L",
     "level": "Standard of care",              "condition": "After platinum doublet"},
    {"drug": "Pemetrexed maintenance",         "class": "Antimetabolite",   "line": "Maintenance",
     "level": "Standard of care",              "condition": "Non-squamous, no progression after 1L"},
]

# Evidence level colors for plotting
LEVEL_COLORS = {
    "FDA-Approved":    "#27ae60",
    "Clinical":        "#2980b9",
    "Phase II":        "#8e44ad",
    "Phase I/II":      "#c0392b",
    "Investigational": "#e67e22",
    "Standard of care":"#16a085",
}

# ── LUAD driver gene set ──────────────────────────────────────────────────────
LUAD_DRIVERS = set(DRUG_KB.keys()) | {"MYC", "MYCL", "DDR2", "NRAS", "HRAS"}


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
    return logging.getLogger(name)


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_variants(sample_id: str) -> pd.DataFrame:
    """Load all variants (not just missense) from module 02."""
    path = VARIANT_DIR / sample_id / f"{sample_id}_variants.tsv.gz"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t", compression="gzip", low_memory=False)


def load_tmb(sample_id: str) -> float:
    """Load TMB estimate (coding+nonsilent) from module 02. Returns NaN if missing."""
    path = VARIANT_DIR / sample_id / f"{sample_id}_tmb.tsv"
    if not path.exists():
        return float("nan")
    df = pd.read_csv(path, sep="\t")
    row = df[df["tmb_measure"] == "TMB_coding_non_silent"]
    return float(row["tmb_estimate"].iloc[0]) if not row.empty else float("nan")


def load_clinical(sample_id: str) -> dict:
    """Load clinical data from module 01 clinical_summary.tsv."""
    path = CLINICAL_DIR / "clinical_summary.tsv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, sep="\t")
    row = df[df["sample_id"] == sample_id]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


# ══════════════════════════════════════════════════════════════════════════════
# Drug mapping logic
# ══════════════════════════════════════════════════════════════════════════════

def extract_driver_hits(variants: pd.DataFrame) -> pd.DataFrame:
    """
    Extract mutations in driver genes from the variant table.
    Returns rows with: SYMBOL, CONSEQUENCE, HGVSp_Short, hotspot, IMPACT.
    """
    if variants.empty:
        return pd.DataFrame()

    driver_hits = variants[
        variants["SYMBOL"].isin(DRUG_KB.keys())
    ].copy()

    cols = ["SYMBOL", "CONSEQUENCE", "HGVSp_Short", "hotspot", "IMPACT"]
    available = [c for c in cols if c in driver_hits.columns]
    return driver_hits[available].reset_index(drop=True)


def match_drugs(gene: str, hgvsp: str, consequence: str) -> list:
    """
    Match gene + variant to drug recommendations in DRUG_KB.
    Returns list of drug dicts with match confidence.
    """
    if gene not in DRUG_KB:
        return []

    entry = DRUG_KB[gene]
    matched = []
    hgvsp_str = str(hgvsp) if pd.notna(hgvsp) else ""
    csq_str   = str(consequence) if pd.notna(consequence) else ""

    for d in entry.get("drugs", []):
        pattern = d.get("pattern", "")
        if not pattern or pattern == r".*":
            continue   # skip catch-all patterns in primary scan
        if re.search(pattern, hgvsp_str, re.IGNORECASE) or \
           re.search(pattern, csq_str,   re.IGNORECASE):
            matched.append({
                "gene":       gene,
                "mutation":   hgvsp_str,
                "drug":       d["drug"],
                "drug_class": d["class"],
                "line":       d["line"],
                "level":      d["level"],
                "category":   entry["category"],
            })

    # Fallback: RET / ALK / ROS1 drugs apply to any detected mutation in those genes
    if not matched and entry.get("category") == "Oncogene":
        for d in entry.get("drugs", []):
            if d.get("pattern") == r".*":
                matched.append({
                    "gene":       gene,
                    "mutation":   hgvsp_str,
                    "drug":       d["drug"],
                    "drug_class": d["class"],
                    "line":       d["line"],
                    "level":      d["level"],
                    "category":   entry["category"],
                })

    return matched


def assess_immunotherapy(tmb: float, driver_hits: pd.DataFrame) -> list:
    """
    Assess immunotherapy eligibility based on TMB and driver gene profile.
    Returns list of IO recommendations with eligibility notes.
    """
    results = []

    # Check for IO-resistance genes
    io_resistant_genes = [
        g for g in driver_hits.get("SYMBOL", pd.Series()).tolist()
        if DRUG_KB.get(g, {}).get("io_resistance", False)
    ]
    has_egfr = "EGFR" in driver_hits.get("SYMBOL", pd.Series()).tolist()
    has_alk  = "ALK"  in driver_hits.get("SYMBOL", pd.Series()).tolist()

    for io in IMMUNOTHERAPY:
        eligible = True
        note = io["condition"]

        if has_egfr or has_alk:
            if "no EGFR/ALK" in io["condition"]:
                eligible = False
                note = f"NOT recommended: EGFR/ALK driver present"

        if "TMB ≥ 10" in io["condition"]:
            if pd.isna(tmb) or tmb < 10:
                eligible = False
                note = f"TMB = {tmb:.1f} mut/Mb (threshold: ≥10)"
            else:
                note = f"TMB = {tmb:.1f} mut/Mb ✓ — eligible"

        if io_resistant_genes and "PD-1" in io["class"]:
            note += f" | Caution: {', '.join(io_resistant_genes)} may confer IO resistance"

        results.append({
            "drug":       io["drug"],
            "drug_class": io["class"],
            "line":       io["line"],
            "level":      io["level"],
            "eligible":   eligible,
            "note":       note,
        })
    return results


def query_civic(gene: str, hgvsp: str, logger) -> list:
    """
    Query CIViC API for variant-level drug evidence (open, no token required).
    Returns list of evidence summaries, or empty list on failure.
    """
    results = []
    try:
        # Search gene
        r = requests.get(
            f"https://civicdb.org/api/genes/{gene}",
            timeout=8, params={"identifier_type": "name"}
        )
        if r.status_code != 200:
            return []
        gene_data = r.json()

        # Filter variants matching hgvsp (or name contains key mutation)
        short_mut = re.sub(r"^p\.", "", str(hgvsp))  # "L858R" from "p.L858R"
        for v in gene_data.get("variants", [])[:15]:
            vname = str(v.get("name", ""))
            if not re.search(short_mut[:4], vname, re.IGNORECASE):
                continue
            # Fetch variant evidence
            rv = requests.get(f"https://civicdb.org/api/variants/{v['id']}", timeout=5)
            if rv.status_code != 200:
                continue
            vdata = rv.json()
            for ev in vdata.get("evidence_items", [])[:3]:
                if ev.get("disease", {}) and ev.get("drugs"):
                    for drug_info in ev["drugs"]:
                        results.append({
                            "source":     "CIViC",
                            "gene":       gene,
                            "mutation":   vname,
                            "drug":       drug_info.get("name", ""),
                            "evidence_level": ev.get("evidence_level", ""),
                            "evidence_type":  ev.get("evidence_type", ""),
                            "disease":    ev.get("disease", {}).get("name", ""),
                        })
    except Exception as e:
        logger.debug(f"  CIViC query failed for {gene}: {e}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Per-sample pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_sample(sample_id: str, logger) -> dict:
    logger.info(f"\n{'='*55}")
    logger.info(f"Sample: {sample_id}")

    out_dir = OUT_DIR / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    variants     = load_variants(sample_id)
    tmb          = load_tmb(sample_id)
    clinical     = load_clinical(sample_id)
    driver_hits  = extract_driver_hits(variants)

    logger.info(
        f"  {len(variants)} variants | TMB={tmb:.2f} mut/Mb | "
        f"{len(driver_hits)} driver-gene hits"
    )

    # ── Map mutations to drugs ──
    drug_rows = []
    for _, row in driver_hits.iterrows():
        gene   = row["SYMBOL"]
        hgvsp  = row.get("HGVSp_Short", "")
        csq    = row.get("CONSEQUENCE", "")
        hits   = match_drugs(gene, hgvsp, csq)
        drug_rows.extend(hits)

        # Optional CIViC enrichment
        civic = query_civic(gene, hgvsp, logger)
        for c in civic:
            drug_rows.append({
                "gene":       gene,
                "mutation":   str(hgvsp),
                "drug":       c["drug"],
                "drug_class": f"CIViC ({c['evidence_type']})",
                "line":       "—",
                "level":      f"CIViC-{c['evidence_level']}",
                "category":   DRUG_KB.get(gene, {}).get("category", "Unknown"),
            })

    drug_df = pd.DataFrame(drug_rows).drop_duplicates(
        subset=["gene", "mutation", "drug"]
    ).reset_index(drop=True) if drug_rows else pd.DataFrame()

    # ── Immunotherapy ──
    io_df = pd.DataFrame(assess_immunotherapy(tmb, driver_hits))

    # ── Chemotherapy (always applicable) ──
    chemo_df = pd.DataFrame(CHEMOTHERAPY)

    # ── Save report TSV ──
    report_sections = []
    if not drug_df.empty:
        drug_df["sample_id"] = sample_id
        report_sections.append(drug_df)

    # Build gene-level summary notes
    gene_notes = []
    for gene in driver_hits["SYMBOL"].unique():
        note = DRUG_KB.get(gene, {}).get("note", "")
        if note:
            gene_notes.append({"gene": gene, "note": note})

    report_path = out_dir / f"{sample_id}_drug_report.tsv"
    if not drug_df.empty:
        drug_df.to_csv(report_path, sep="\t", index=False)
        logger.info(f"  {len(drug_df)} drug recommendations → {report_path}")
    else:
        logger.info("  No targeted drug matches found (see IO/chemo recommendations)")

    # ── Visualization ──
    plot_drug_report(
        sample_id, drug_df, driver_hits, io_df, chemo_df,
        tmb, clinical, gene_notes, out_dir
    )

    return {
        "sample_id":        sample_id,
        "n_driver_genes":   len(driver_hits["SYMBOL"].unique()) if not driver_hits.empty else 0,
        "n_targeted_drugs": len(drug_df),
        "n_fda_approved":   len(drug_df[drug_df["level"] == "FDA-Approved"]) if not drug_df.empty else 0,
        "tmb":              round(tmb, 2),
        "top_drugs":        "; ".join(drug_df["drug"].head(3).tolist()) if not drug_df.empty else "—",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Visualization
# ══════════════════════════════════════════════════════════════════════════════

def plot_drug_report(
    sample_id, drug_df, driver_hits, io_df, chemo_df,
    tmb, clinical, gene_notes, out_dir
):
    """
    4-panel clinical drug actionability figure per sample.

    Panel layout (2×2):
      [top-left]    Actionable mutations table
      [top-right]   Drug class donut chart (targeted + IO)
      [bottom-left] Immunotherapy eligibility + TMB gauge
      [bottom-right]Therapy options by evidence level + line
    """
    fig = plt.figure(figsize=(16, 12))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    ax_tbl  = fig.add_subplot(gs[0, 0])
    ax_pie  = fig.add_subplot(gs[0, 1])
    ax_io   = fig.add_subplot(gs[1, 0])
    ax_lvl  = fig.add_subplot(gs[1, 1])

    # Clinical header
    age   = clinical.get("age_at_diagnosis", "N/A")
    stage = clinical.get("stage", "N/A")
    gender= clinical.get("gender", "N/A")
    smoke = clinical.get("pack_years", "N/A")
    fig.suptitle(
        f"Drug Actionability Report — {sample_id}\n"
        f"Age: {age}  |  Stage: {stage}  |  Gender: {gender}  |  "
        f"Pack-years: {smoke}  |  TMB: {tmb:.1f} mut/Mb",
        fontsize=12, fontweight="bold", y=1.01,
    )

    # ── [0,0] Actionable mutations table ────────────────────────────────────
    ax_tbl.axis("off")

    if not drug_df.empty:
        tbl_data = []
        for _, r in drug_df.iterrows():
            tbl_data.append([
                r["gene"],
                r["mutation"][:15] if len(str(r["mutation"])) > 15 else r["mutation"],
                r["drug"][:25] if len(str(r["drug"])) > 25 else r["drug"],
                r["line"],
                r["level"],
            ])
        # Add gene notes (truncated)
        for n in gene_notes[:3]:
            tbl_data.append([n["gene"], "—", f"Note: {n['note'][:35]}...", "—", "—"])

        col_labels = ["Gene", "Mutation", "Drug / Therapy", "Line", "Evidence"]
        tbl = ax_tbl.table(
            cellText=tbl_data[:18],
            colLabels=col_labels,
            cellLoc="left",
            loc="center",
            bbox=[0.0, 0.0, 1.0, 1.0],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.5)

        # Color-code rows by evidence level
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor("#2c3e50")
                cell.set_text_props(color="white", fontweight="bold")
            elif r <= len(tbl_data):
                level = tbl_data[r - 1][4]
                color = LEVEL_COLORS.get(level, "#ffffff")
                cell.set_facecolor(color + "33")  # 20% opacity
        ax_tbl.set_title(
            f"Actionable Mutations ({len(drug_df)} drug recommendations)",
            fontsize=9, pad=4,
        )
    else:
        # Show gene-level notes even when no targeted drug matches
        note_rows = []
        for n in gene_notes:
            note_rows.append([n["gene"], n["note"][:60]])
        if note_rows:
            note_tbl = ax_tbl.table(
                cellText=note_rows,
                colLabels=["Gene", "Clinical Note"],
                cellLoc="left",
                loc="center",
                bbox=[0.0, 0.1, 1.0, 0.85],
            )
            note_tbl.auto_set_font_size(False)
            note_tbl.set_fontsize(7.5)
            for (r, c), cell in note_tbl.get_celld().items():
                if r == 0:
                    cell.set_facecolor("#2c3e50")
                    cell.set_text_props(color="white", fontweight="bold")
                elif r % 2 == 0:
                    cell.set_facecolor("#fef9e7")
            ax_tbl.text(0.5, 0.02,
                "No FDA-approved targeted therapy for these variants.\n"
                "→ Immunotherapy eligibility: see panel 3  |  → Platinum doublet: standard",
                ha="center", va="bottom", fontsize=8, style="italic",
                transform=ax_tbl.transAxes, color="#7f8c8d")
        else:
            ax_tbl.text(
                0.5, 0.5,
                "No direct targeted drug matches.\n"
                "No known oncogenic driver mutations\nin EGFR / KRAS / BRAF / MET / RET / ALK.\n\n"
                "→ Consider immunotherapy (see panel 3)\n"
                "→ Platinum-based chemotherapy",
                ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffeeba", alpha=0.7),
            )
        ax_tbl.set_title(
            f"Driver Mutations ({len(driver_hits)} hits) — No direct targeted match",
            fontsize=9,
        )

    # ── [0,1] Drug class donut chart ─────────────────────────────────────────
    all_drugs = []
    if not drug_df.empty:
        all_drugs = drug_df["drug_class"].tolist()
    # Always include IO and chemo as options
    all_drugs += [io["drug_class"] for io in io_df.to_dict("records")
                  if io.get("eligible", False)]
    all_drugs += ["Platinum doublet"]  # standard chemo always applicable

    from collections import Counter
    class_counts = Counter(all_drugs)

    if class_counts:
        labels  = list(class_counts.keys())
        sizes   = list(class_counts.values())
        # Colour palette
        palette = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        wedges, texts, autotexts = ax_pie.pie(
            sizes, labels=None, autopct="%1.0f%%",
            startangle=90, colors=palette,
            pctdistance=0.8, wedgeprops=dict(width=0.55),
        )
        for at in autotexts:
            at.set_fontsize(8)
        ax_pie.legend(
            wedges, labels,
            loc="lower center", fontsize=7,
            bbox_to_anchor=(0.5, -0.3),
            ncol=2,
        )
        ax_pie.set_title("Drug Class Breakdown", fontsize=9)
    else:
        ax_pie.text(0.5, 0.5, "No drug classes identified",
                    ha="center", va="center", fontsize=10)

    # ── [1,0] Immunotherapy eligibility ──────────────────────────────────────
    ax_io.axis("off")

    # TMB gauge bar
    tmb_ax = ax_io.inset_axes([0.05, 0.8, 0.9, 0.12])
    tmb_norm = min(tmb / 20.0, 1.0) if not np.isnan(tmb) else 0.0
    tmb_color = "#27ae60" if tmb >= 10 else ("#e67e22" if tmb >= 5 else "#e74c3c")
    tmb_ax.barh(0, tmb_norm, color=tmb_color, alpha=0.8)
    tmb_ax.barh(0, 1.0, color="#ecf0f1", alpha=0.5)
    tmb_ax.set_xlim(0, 1.0)
    tmb_ax.set_yticks([])
    tmb_ax.set_xticks([0, 0.25, 0.5, 1.0])
    tmb_ax.set_xticklabels(["0", "5", "10", "≥20"], fontsize=7)
    tmb_ax.axvline(0.5, color="#e74c3c", lw=1.5, linestyle="--")
    tmb_ax.set_title(f"TMB = {tmb:.1f} mut/Mb", fontsize=8, pad=2)

    # IO recommendation table
    if not io_df.empty:
        io_show = io_df[io_df["eligible"]].head(4)
        if io_show.empty:
            io_show = io_df.head(4)
        io_data = [[r["drug"][:22], r["line"], "✓" if r["eligible"] else "✗", r["note"][:28]]
                   for _, r in io_show.iterrows()]
        io_tbl = ax_io.table(
            cellText=io_data,
            colLabels=["Immunotherapy", "Line", "Eligible", "Note"],
            cellLoc="left",
            loc="center",
            bbox=[0.0, 0.02, 1.0, 0.72],
        )
        io_tbl.auto_set_font_size(False)
        io_tbl.set_fontsize(7)
        for (r, c), cell in io_tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor("#2c3e50")
                cell.set_text_props(color="white", fontweight="bold")
            elif c == 2 and r > 0:
                val = io_data[r - 1][2]
                cell.set_facecolor("#d5f5e3" if val == "✓" else "#fadbd8")
    ax_io.set_title("Immunotherapy Eligibility", fontsize=9, y=1.04)

    # ── [1,1] Evidence level + therapy line bar chart ────────────────────────
    all_recs = []
    if not drug_df.empty:
        for _, r in drug_df.iterrows():
            all_recs.append({"drug": r["drug"], "level": r["level"], "line": r["line"]})
    for _, r in io_df[io_df["eligible"]].iterrows():
        all_recs.append({"drug": r["drug"], "level": r["level"], "line": r["line"]})
    for _, r in chemo_df.iterrows():
        all_recs.append({"drug": r["drug"], "level": r["level"], "line": r["line"]})

    if all_recs:
        rec_df = pd.DataFrame(all_recs).drop_duplicates(subset=["drug", "line"])
        # Group by line
        line_order = ["1L", "Consolidation", "Maintenance", "2L", "Investigational", "—"]
        rec_df["line_order"] = rec_df["line"].apply(
            lambda x: line_order.index(x) if x in line_order else len(line_order)
        )
        rec_df = rec_df.sort_values("line_order")

        drugs_plot  = rec_df["drug"].tolist()[:18]
        levels_plot = rec_df["level"].tolist()[:18]
        lines_plot  = rec_df["line"].tolist()[:18]

        colors = [LEVEL_COLORS.get(lv, "#95a5a6") for lv in levels_plot]
        y_pos  = range(len(drugs_plot))
        ax_lvl.barh(y_pos, [1] * len(drugs_plot), color=colors, alpha=0.85)
        ax_lvl.set_yticks(y_pos)
        yticklabels = [f"{d[:22]}  [{ln}]" for d, ln in zip(drugs_plot, lines_plot)]
        ax_lvl.set_yticklabels(yticklabels, fontsize=7)
        ax_lvl.set_xlim(0, 1.5)
        ax_lvl.set_xticks([])
        ax_lvl.set_title("Recommended Therapies by Evidence Level", fontsize=9)

        # Legend
        legend_patches = [
            mpatches.Patch(color=v, label=k, alpha=0.85)
            for k, v in LEVEL_COLORS.items()
        ]
        ax_lvl.legend(
            handles=legend_patches, loc="lower right",
            fontsize=7, ncol=1,
        )
    else:
        ax_lvl.text(0.5, 0.5, "No recommendations",
                    ha="center", va="center", fontsize=10)

    plt.tight_layout()
    out_png = out_dir / f"{sample_id}_drug_report.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    get_logger("luad-drug").info(f"  Plot → {out_png}")


# ══════════════════════════════════════════════════════════════════════════════
# All-sample heatmap
# ══════════════════════════════════════════════════════════════════════════════

def plot_all_samples_heatmap(summary_rows: list, out_dir: Path):
    """
    Drug × sample heatmap showing which drugs apply to each sample,
    color-coded by evidence level.
    """
    if not summary_rows:
        return

    # Collect all drug data per sample
    drug_sample = {}   # drug → {sample → level}
    for row in summary_rows:
        sid = row["sample_id"]
        path = out_dir / sid / f"{sid}_drug_report.tsv"
        if not path.exists():
            continue
        df = pd.read_csv(path, sep="\t")
        for _, r in df.iterrows():
            d = r["drug"]
            if d not in drug_sample:
                drug_sample[d] = {}
            drug_sample[d][sid] = r["level"]

    if not drug_sample:
        return

    # Add IO/chemo options
    samples = sorted({r["sample_id"] for r in summary_rows})
    for io in IMMUNOTHERAPY[:3]:
        d = io["drug"]
        for sid in samples:
            if d not in drug_sample:
                drug_sample[d] = {}
            if sid not in drug_sample[d]:
                drug_sample[d][sid] = io["level"]

    drugs  = sorted(drug_sample.keys())
    n_drugs, n_samples = len(drugs), len(samples)

    # Encode level to numeric
    level_order = {
        "FDA-Approved": 5, "Standard of care": 4,
        "Clinical": 3, "Phase II": 2, "Phase I/II": 1,
        "Investigational": 1, "—": 0,
    }
    matrix = np.zeros((n_drugs, n_samples))
    for i, drug in enumerate(drugs):
        for j, sid in enumerate(samples):
            level = drug_sample.get(drug, {}).get(sid, "—")
            matrix[i, j] = level_order.get(level, 0)

    fig, ax = plt.subplots(figsize=(max(8, n_samples * 2.5), max(8, n_drugs * 0.45)))
    cmap = plt.cm.RdYlGn
    im   = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=5)

    ax.set_xticks(range(n_samples))
    ax.set_xticklabels(samples, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(n_drugs))
    ax.set_yticklabels(drugs, fontsize=8)
    ax.set_title(
        "Drug Actionability Heatmap — LUAD Samples\n"
        "(Green = FDA-Approved, Yellow = Clinical, Red/None = Investigational/None)",
        fontsize=11,
    )

    # Annotate cells
    for i in range(n_drugs):
        for j in range(n_samples):
            drug_name = drugs[i]
            sid       = samples[j]
            level     = drug_sample.get(drug_name, {}).get(sid, "")
            if level:
                short = {"FDA-Approved": "FDA✓", "Standard of care": "SoC",
                         "Clinical": "Clin", "Phase II": "P2",
                         "Phase I/II": "P1/2", "Investigational": "Inv"}.get(level, level[:4])
                ax.text(j, i, short, ha="center", va="center",
                        fontsize=7, color="black", fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Evidence Strength", fontsize=9)
    cbar.set_ticks([0, 1, 2, 3, 4, 5])
    cbar.set_ticklabels(["None", "Invest.", "Phase I/II", "Phase II", "SoC", "FDA ✓"])

    plt.tight_layout()
    out_png = out_dir / "drug_actionability_heatmap.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    get_logger("luad-drug").info(f"  Heatmap → {out_png}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def get_all_samples() -> list:
    return sorted(p.name for p in VARIANT_DIR.glob("*/") if p.is_dir())


def main():
    parser = argparse.ArgumentParser(
        description="LUAD mutation-to-drug mapping (module 07)"
    )
    parser.add_argument("--sample",  type=str, help="Single sample ID")
    parser.add_argument("--dry_run", action="store_true")
    args   = parser.parse_args()
    logger = get_logger("luad-drug")

    if args.dry_run:
        samples = [args.sample] if args.sample else get_all_samples()
        for s in samples:
            v = load_variants(s)
            d = extract_driver_hits(v)
            logger.info(f"  {s}: {len(v)} variants, {len(d)} driver hits "
                        f"({list(d['SYMBOL'].unique()) if not d.empty else []})")
        return

    samples = [args.sample] if args.sample else get_all_samples()
    if not samples:
        print("[ERROR] No samples found")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for s in samples:
        try:
            r = run_sample(s, logger)
            results.append(r)
        except Exception as e:
            logger.error(f"Sample {s} failed: {e}")
            results.append({"sample_id": s})

    if results:
        summary = pd.DataFrame(results)
        summary_path = OUT_DIR / "all_samples_drug_summary.tsv"
        summary.to_csv(summary_path, sep="\t", index=False)
        print(f"\n[Summary] {len(results)} samples → {summary_path}")
        print(summary.to_string(index=False))

        # All-sample heatmap
        plot_all_samples_heatmap(results, OUT_DIR)


if __name__ == "__main__":
    main()
