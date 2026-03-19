#!/usr/bin/env python
"""
luad_clinical_rec.py
--------------------
Module 10: Clinical trial matching + MDT report generation.

Two components:
  1. Trial matching: pre-curated LUAD clinical trial database (~25 trials),
     match patient molecular profile → eligible / potentially eligible trials.
  2. MDT report: structured one-page summary integrating all modules
     (M02 variants, M03 expression, M04/M08 TME, M05 TIL, M09 recommendations).

Data inputs:
  M02  → somatic variants, TMB, driver mutations
  M03  → PD-L1 RNA, IFN-γ signature
  M04  → ssGSEA TME phenotype
  M08  → immune activity score, io_group
  M09  → ranked treatment recommendations
  clinical → stage, age

Usage:
  python luad_clinical_rec.py --sample TCGA-05-4244
  python luad_clinical_rec.py               # all patients
"""

import argparse
import json
import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
PROJECT_DIR  = SCRIPT_DIR.parent.parent
VARIANT_DIR  = PROJECT_DIR / "data/output/02_variants"
EXPR_DIR     = PROJECT_DIR / "data/output/03_expression"
TME_DIR      = PROJECT_DIR / "data/output/04_single_cell"
PATH_DIR     = PROJECT_DIR / "data/output/05_pathology"
IO_DIR       = PROJECT_DIR / "data/output/08_io_ml"
INT_DIR      = PROJECT_DIR / "data/output/09_integration"
CLIN_FILE    = PROJECT_DIR / "data/clinical/tcga_luad_survival.tsv"
OUT_DIR      = PROJECT_DIR / "data/output/10_clinical_rec"

# ── Clinical trial database ────────────────────────────────────────────────────
# Each entry: nct_id, name, drug, target, phase, status, line,
#             eligibility (required_mutations, tmb_min, tme_required, exclusions),
#             key_result, reference
TRIAL_DB = [
    # ── KRAS G12C ──────────────────────────────────────────────────────────────
    {
        "nct_id": "NCT04303780",
        "name": "CodeBreaK 200",
        "drug": "Sotorasib vs Docetaxel",
        "drug_class": "KRAS G12C inhibitor",
        "target": "KRAS G12C",
        "phase": "III",
        "status": "Completed",
        "line": "2L+",
        "eligibility": {
            "mutations": [{"gene": "KRAS", "variant_contains": "G12C"}],
            "tmb_min": None,
            "tme": None,
            "exclusions": [],
        },
        "key_result": "PFS 5.6 vs 4.5 mo (HR 0.66), ORR 28.1%",
        "reference": "de Langen AJ, Lancet 2023",
    },
    {
        "nct_id": "NCT04613596",
        "name": "KRYSTAL-12",
        "drug": "Adagrasib vs Docetaxel",
        "drug_class": "KRAS G12C inhibitor",
        "target": "KRAS G12C",
        "phase": "III",
        "status": "Completed",
        "line": "2L+",
        "eligibility": {
            "mutations": [{"gene": "KRAS", "variant_contains": "G12C"}],
            "tmb_min": None,
            "tme": None,
            "exclusions": [],
        },
        "key_result": "PFS 5.5 vs 3.8 mo (HR 0.58), ORR 32%",
        "reference": "Jänne PA, NEJM 2024",
    },
    {
        "nct_id": "NCT04933955",
        "name": "KRYSTAL-7",
        "drug": "Adagrasib + Pembrolizumab",
        "drug_class": "KRAS G12C + PD-1 inhibitor",
        "target": "KRAS G12C + PD-L1 high",
        "phase": "II",
        "status": "Active",
        "line": "1L",
        "eligibility": {
            "mutations": [{"gene": "KRAS", "variant_contains": "G12C"}],
            "tmb_min": None,
            "tme": ["Inflamed"],
            "exclusions": [],
        },
        "key_result": "ORR 49% (PD-L1 ≥50% cohort)",
        "reference": "Riely GJ, ASCO 2023",
    },
    # ── EGFR ───────────────────────────────────────────────────────────────────
    {
        "nct_id": "NCT04035486",
        "name": "FLAURA2",
        "drug": "Osimertinib + Chemotherapy",
        "drug_class": "3rd-gen EGFR TKI + platinum doublet",
        "target": "EGFR exon19del / L858R",
        "phase": "III",
        "status": "Completed",
        "line": "1L",
        "eligibility": {
            "mutations": [
                {"gene": "EGFR", "variant_contains": "19"},
                {"gene": "EGFR", "variant_contains": "858"},
            ],
            "mutation_logic": "OR",
            "tmb_min": None,
            "tme": None,
            "exclusions": [],
        },
        "key_result": "PFS 25.5 vs 16.7 mo (HR 0.62)",
        "reference": "Planchard D, NEJM 2023",
    },
    {
        "nct_id": "NCT04538664",
        "name": "PAPILLON",
        "drug": "Amivantamab + Carboplatin + Pemetrexed",
        "drug_class": "EGFR/MET bispecific + chemotherapy",
        "target": "EGFR exon20 insertion",
        "phase": "III",
        "status": "Completed",
        "line": "1L",
        "eligibility": {
            "mutations": [{"gene": "EGFR", "variant_contains": "20"}],
            "tmb_min": None,
            "tme": None,
            "exclusions": [],
        },
        "key_result": "PFS 11.4 vs 6.7 mo (HR 0.40)",
        "reference": "Zhou C, NEJM 2023",
    },
    {
        "nct_id": "NCT04558853",
        "name": "MARIPOSA",
        "drug": "Amivantamab + Lazertinib",
        "drug_class": "EGFR/MET bispecific + 3rd-gen EGFR TKI",
        "target": "EGFR exon19del / L858R",
        "phase": "III",
        "status": "Completed",
        "line": "1L",
        "eligibility": {
            "mutations": [
                {"gene": "EGFR", "variant_contains": "19"},
                {"gene": "EGFR", "variant_contains": "858"},
            ],
            "mutation_logic": "OR",
            "tmb_min": None,
            "tme": None,
            "exclusions": [],
        },
        "key_result": "PFS 23.7 vs 16.6 mo (HR 0.70)",
        "reference": "Cho BC, NEJM 2024",
    },
    # ── ALK ────────────────────────────────────────────────────────────────────
    {
        "nct_id": "NCT03052608",
        "name": "CROWN",
        "drug": "Lorlatinib",
        "drug_class": "3rd-gen ALK inhibitor",
        "target": "ALK fusion",
        "phase": "III",
        "status": "Completed",
        "line": "1L",
        "eligibility": {
            "mutations": [{"gene": "ALK", "variant_contains": "fusion"}],
            "tmb_min": None,
            "tme": None,
            "exclusions": [],
        },
        "key_result": "5-yr PFS 60% vs 8% (HR 0.19)",
        "reference": "Shaw AT, NEJM 2020 / ASCO 2024",
    },
    {
        "nct_id": "NCT02075840",
        "name": "ALEX",
        "drug": "Alectinib",
        "drug_class": "2nd-gen ALK inhibitor",
        "target": "ALK fusion",
        "phase": "III",
        "status": "Completed",
        "line": "1L",
        "eligibility": {
            "mutations": [{"gene": "ALK", "variant_contains": "fusion"}],
            "tmb_min": None,
            "tme": None,
            "exclusions": [],
        },
        "key_result": "PFS 34.8 vs 10.9 mo (HR 0.43)",
        "reference": "Peters S, NEJM 2017",
    },
    # ── RET ────────────────────────────────────────────────────────────────────
    {
        "nct_id": "NCT03157962",
        "name": "LIBRETTO-431",
        "drug": "Selpercatinib",
        "drug_class": "Selective RET inhibitor",
        "target": "RET fusion",
        "phase": "III",
        "status": "Completed",
        "line": "1L",
        "eligibility": {
            "mutations": [{"gene": "RET", "variant_contains": "fusion"}],
            "tmb_min": None,
            "tme": None,
            "exclusions": [],
        },
        "key_result": "PFS 24.8 vs 11.2 mo (HR 0.46)",
        "reference": "Ramalingam SS, NEJM 2023",
    },
    # ── MET exon14 ─────────────────────────────────────────────────────────────
    {
        "nct_id": "NCT02414139",
        "name": "GEOMETRY mono-1",
        "drug": "Capmatinib",
        "drug_class": "Selective MET inhibitor",
        "target": "MET exon14 skipping",
        "phase": "II",
        "status": "Completed",
        "line": "1L/2L",
        "eligibility": {
            "mutations": [{"gene": "MET", "variant_contains": "exon14"}],
            "tmb_min": None,
            "tme": None,
            "exclusions": [],
        },
        "key_result": "ORR 68% (treatment-naive), 41% (pretreated)",
        "reference": "Wolf J, NEJM 2020",
    },
    {
        "nct_id": "NCT02864213",
        "name": "VISION",
        "drug": "Tepotinib",
        "drug_class": "Selective MET inhibitor",
        "target": "MET exon14 skipping",
        "phase": "II",
        "status": "Completed",
        "line": "1L/2L",
        "eligibility": {
            "mutations": [{"gene": "MET", "variant_contains": "exon14"}],
            "tmb_min": None,
            "tme": None,
            "exclusions": [],
        },
        "key_result": "ORR 46%, mDOR 11.1 mo",
        "reference": "Paik PK, NEJM 2020",
    },
    # ── ERBB2/HER2 ─────────────────────────────────────────────────────────────
    {
        "nct_id": "NCT04644004",
        "name": "DESTINY-Lung02",
        "drug": "Trastuzumab deruxtecan (T-DXd)",
        "drug_class": "HER2-targeted ADC",
        "target": "ERBB2/HER2 mutation",
        "phase": "II",
        "status": "Completed",
        "line": "2L+",
        "eligibility": {
            "mutations": [{"gene": "ERBB2", "variant_contains": ""}],
            "tmb_min": None,
            "tme": None,
            "exclusions": [],
        },
        "key_result": "ORR 49.0%, mPFS 9.9 mo",
        "reference": "Li BT, NEJM 2023",
    },
    # ── NTRK ───────────────────────────────────────────────────────────────────
    {
        "nct_id": "NCT02122913",
        "name": "LOXO-TRK-14001",
        "drug": "Larotrectinib",
        "drug_class": "Selective TRK inhibitor",
        "target": "NTRK1/2/3 fusion",
        "phase": "I/II",
        "status": "Completed",
        "line": "Any",
        "eligibility": {
            "mutations": [{"gene": "NTRK1", "variant_contains": "fusion"},
                          {"gene": "NTRK2", "variant_contains": "fusion"},
                          {"gene": "NTRK3", "variant_contains": "fusion"}],
            "mutation_logic": "OR",
            "tmb_min": None,
            "tme": None,
            "exclusions": [],
        },
        "key_result": "ORR 75% (pooled TRK fusion solid tumors)",
        "reference": "Hong DS, NEJM 2020",
    },
    # ── BRAF V600E ─────────────────────────────────────────────────────────────
    {
        "nct_id": "NCT01336634",
        "name": "BRF113928",
        "drug": "Dabrafenib + Trametinib",
        "drug_class": "BRAF + MEK inhibitor",
        "target": "BRAF V600E",
        "phase": "II",
        "status": "Completed",
        "line": "1L/2L",
        "eligibility": {
            "mutations": [{"gene": "BRAF", "variant_contains": "V600E"}],
            "tmb_min": None,
            "tme": None,
            "exclusions": [],
        },
        "key_result": "ORR 64% (treatment-naive), 63% (pretreated)",
        "reference": "Planchard D, Lancet Oncol 2017",
    },
    # ── IO: TMB-high ───────────────────────────────────────────────────────────
    {
        "nct_id": "NCT02453282",
        "name": "KEYNOTE-158",
        "drug": "Pembrolizumab",
        "drug_class": "PD-1 inhibitor",
        "target": "TMB-high (≥10 mut/Mb)",
        "phase": "II",
        "status": "Completed",
        "line": "2L+",
        "eligibility": {
            "mutations": [],
            "tmb_min": 10,
            "tme": None,
            "exclusions": [],
        },
        "key_result": "ORR 29% (TMB ≥10), FDA-approved 2020",
        "reference": "Marabelle A, JAMA Oncol 2020",
    },
    {
        "nct_id": "NCT02477826",
        "name": "CheckMate 227",
        "drug": "Nivolumab + Ipilimumab",
        "drug_class": "PD-1 + CTLA-4 dual blockade",
        "target": "TMB-high (≥10 mut/Mb)",
        "phase": "III",
        "status": "Completed",
        "line": "1L",
        "eligibility": {
            "mutations": [],
            "tmb_min": 10,
            "tme": None,
            "exclusions": ["EGFR", "ALK"],
        },
        "key_result": "PFS 7.2 vs 5.5 mo (HR 0.58) in TMB-high",
        "reference": "Hellmann MD, NEJM 2019",
    },
    # ── IO: Inflamed TME ───────────────────────────────────────────────────────
    {
        "nct_id": "NCT02220894",
        "name": "KEYNOTE-010",
        "drug": "Pembrolizumab",
        "drug_class": "PD-1 inhibitor",
        "target": "PD-L1 ≥1% / Inflamed TME",
        "phase": "II/III",
        "status": "Completed",
        "line": "2L+",
        "eligibility": {
            "mutations": [],
            "tmb_min": None,
            "tme": ["Inflamed"],
            "exclusions": [],
        },
        "key_result": "OS 10.4 vs 8.5 mo (PD-L1 ≥1%)",
        "reference": "Herbst RS, Lancet 2016",
    },
    {
        "nct_id": "NCT02142738",
        "name": "KEYNOTE-024",
        "drug": "Pembrolizumab",
        "drug_class": "PD-1 inhibitor",
        "target": "PD-L1 ≥50% / Inflamed TME",
        "phase": "III",
        "status": "Completed",
        "line": "1L",
        "eligibility": {
            "mutations": [],
            "tmb_min": None,
            "tme": ["Inflamed"],
            "exclusions": ["EGFR", "ALK"],
        },
        "key_result": "PFS 10.3 vs 6.0 mo (HR 0.50)",
        "reference": "Reck M, NEJM 2016",
    },
    # ── Novel / combination ────────────────────────────────────────────────────
    {
        "nct_id": "NCT04656652",
        "name": "TROPION-Lung08",
        "drug": "Datopotamab deruxtecan + Pembrolizumab",
        "drug_class": "TROP2-targeted ADC + PD-1 inhibitor",
        "target": "Non-squamous NSCLC, no driver mutation",
        "phase": "III",
        "status": "Active",
        "line": "1L",
        "eligibility": {
            "mutations": [],
            "tmb_min": None,
            "tme": None,
            "no_driver": True,
            "exclusions": ["EGFR", "ALK", "ROS1", "RET", "KRAS", "MET", "BRAF", "NTRK"],
        },
        "key_result": "PFS 7.0 vs 5.6 mo (HR 0.73); OS benefit in PD-L1 ≥50%",
        "reference": "Mok TSK, NEJM 2024",
    },
    {
        "nct_id": "NCT04294810",
        "name": "SKYSCRAPER-01",
        "drug": "Tiragolumab + Atezolizumab",
        "drug_class": "TIGIT + PD-L1 dual blockade",
        "target": "PD-L1 high / Inflamed TME",
        "phase": "III",
        "status": "Completed",
        "line": "1L",
        "eligibility": {
            "mutations": [],
            "tmb_min": None,
            "tme": ["Inflamed"],
            "exclusions": ["EGFR", "ALK"],
        },
        "key_result": "PFS 5.6 vs 5.4 mo (HR 0.94; ns) — negative trial",
        "reference": "Cho BC, NEJM 2024",
    },
    {
        "nct_id": "NCT04116541",
        "name": "CodeBreaK 101 (STK11 cohort)",
        "drug": "Sotorasib + PD-1/CTLA-4",
        "drug_class": "KRAS G12C inhibitor + immunotherapy",
        "target": "KRAS G12C + STK11 co-mutation",
        "phase": "Ib",
        "status": "Active",
        "line": "2L+",
        "eligibility": {
            "mutations": [{"gene": "KRAS", "variant_contains": "G12C"},
                          {"gene": "STK11", "variant_contains": ""}],
            "mutation_logic": "AND",
            "tmb_min": None,
            "tme": None,
            "exclusions": [],
        },
        "key_result": "Ongoing — addressing STK11-driven IO resistance",
        "reference": "Skoulidis F, Cancer Discov 2024",
    },
]

STATUS_COLORS = {
    "Completed": "#2ecc71",
    "Active":    "#3498db",
    "Recruiting":"#e67e22",
}

PHASE_RANK = {"I": 1, "I/II": 2, "II": 3, "II/III": 3.5, "III": 4, "Ib": 2}


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S", level=logging.INFO)
    return logging.getLogger(name)


# ══════════════════════════════════════════════════════════════════════════════
# Data loading helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_patient_data(sample_id: str) -> dict:
    """Aggregate all available data for one patient across modules."""
    data: dict = {"sample_id": sample_id}

    # M09 recommendation
    rec_path = INT_DIR / sample_id / f"{sample_id}_recommendation.tsv"
    if rec_path.exists():
        data["recommendations"] = pd.read_csv(rec_path, sep="\t")

    # M09 summary row
    sum_path = INT_DIR / "integration_summary.tsv"
    if sum_path.exists():
        df_sum = pd.read_csv(sum_path, sep="\t")
        row = df_sum[df_sum["sample_id"] == sample_id]
        if not row.empty:
            data["summary"] = row.iloc[0].to_dict()

    # M02 variants (try .tsv.gz first, then .tsv)
    var_path = VARIANT_DIR / sample_id / f"{sample_id}_variants.tsv.gz"
    if not var_path.exists():
        var_path = VARIANT_DIR / sample_id / f"{sample_id}_variants.tsv"
    if var_path.exists():
        data["variants"] = pd.read_csv(var_path, sep="\t")

    # M08 io scores
    io_path = IO_DIR / "io_scores.tsv"
    if io_path.exists():
        df_io = pd.read_csv(io_path, sep="\t", index_col=0)
        if sample_id in df_io.index:
            data["io"] = df_io.loc[sample_id].to_dict()

    # M04 TME
    tme_path = TME_DIR / sample_id / f"{sample_id}_tme_scores.tsv"
    if tme_path.exists():
        data["tme"] = pd.read_csv(tme_path, sep="\t").iloc[0].to_dict()

    # Clinical
    if CLIN_FILE.exists():
        df_clin = pd.read_csv(CLIN_FILE, sep="\t")
        row = df_clin[df_clin["sample_id"] == sample_id]
        if not row.empty:
            data["clinical"] = row.iloc[0].to_dict()

    return data


# ══════════════════════════════════════════════════════════════════════════════
# Clinical trial matching
# ══════════════════════════════════════════════════════════════════════════════

def _get_driver_mutations(data: dict) -> list[dict]:
    """Extract driver gene mutations from variant data."""
    DRIVER_GENES = {"EGFR", "KRAS", "ALK", "ROS1", "RET", "MET", "BRAF",
                    "ERBB2", "NTRK1", "NTRK2", "NTRK3", "STK11", "KEAP1",
                    "TP53", "SMAD4"}
    if "variants" not in data:
        return []
    df = data["variants"]
    gene_col = next((c for c in ["SYMBOL", "gene", "Gene"] if c in df.columns), None)
    hgvsp_col = next((c for c in ["HGVSp_Short", "hgvsp", "aa_change"] if c in df.columns), None)
    if gene_col is None:
        return []
    result = []
    for _, row in df.iterrows():
        gene = str(row.get(gene_col, ""))
        if gene in DRIVER_GENES:
            hgvsp = str(row.get(hgvsp_col, "")) if hgvsp_col else ""
            result.append({"gene": gene, "variant": hgvsp})
    return result


def _get_tmb(data: dict) -> float | None:
    """Get TMB from summary or variants."""
    s = data.get("summary", {})
    if "tmb" in s and pd.notna(s["tmb"]):
        return float(s["tmb"])
    return None


def _get_tme_phenotype(data: dict) -> str:
    """Get TME phenotype; prefer M08 io_group, fallback to M04."""
    io = data.get("io", {})
    if "io_group" in io:
        g = str(io["io_group"])
        if "High" in g:
            return "Inflamed"
        elif "Low" in g:
            return "Desert"
    tme = data.get("tme", {})
    if "immune_phenotype" in tme:
        return str(tme["immune_phenotype"])
    s = data.get("summary", {})
    return str(s.get("tme_phenotype", "Unknown"))


def match_trials(data: dict) -> pd.DataFrame:
    """
    Match patient to clinical trials.

    Returns DataFrame with columns:
      nct_id, name, drug, drug_class, target, phase, status, line,
      match_level, match_basis, key_result, reference
    """
    mutations  = _get_driver_mutations(data)
    tmb        = _get_tmb(data)
    tme        = _get_tme_phenotype(data)
    mut_genes  = {m["gene"] for m in mutations}
    mut_strs   = {f"{m['gene']} {m['variant']}" for m in mutations}

    rows = []
    for trial in TRIAL_DB:
        elig     = trial["eligibility"]
        req_muts = elig.get("mutations", [])
        logic    = elig.get("mutation_logic", "OR")
        tmb_min  = elig.get("tmb_min")
        req_tme  = elig.get("tme")
        excl     = set(elig.get("exclusions", []))
        no_driver = elig.get("no_driver", False)

        # Exclusion check
        if excl & mut_genes:
            continue

        # No-driver requirement
        DRIVER_GENES_TARGETED = {"EGFR", "ALK", "ROS1", "RET", "KRAS",
                                   "MET", "BRAF", "NTRK1", "NTRK2", "NTRK3"}
        if no_driver and (mut_genes & DRIVER_GENES_TARGETED):
            continue

        match_level = None
        basis_parts = []

        # Mutation match
        mut_matches = []
        for req in req_muts:
            gene = req["gene"]
            vc   = req.get("variant_contains", "").lower()
            for m in mutations:
                if m["gene"] == gene:
                    if not vc or vc in m["variant"].lower():
                        mut_matches.append(f"{m['gene']} {m['variant']}")

        mut_ok = (len(mut_matches) > 0) if logic == "OR" else (
            len(mut_matches) >= len(req_muts) if req_muts else True
        )

        if req_muts:
            if mut_ok:
                match_level = "Eligible"
                basis_parts.append(f"Mutation: {', '.join(set(mut_matches))}")
            else:
                continue  # required mutation not present

        # TMB check
        if tmb_min is not None:
            if tmb is not None and tmb >= tmb_min:
                if match_level is None:
                    match_level = "Eligible"
                basis_parts.append(f"TMB: {tmb:.1f} mut/Mb ≥ {tmb_min}")
            elif tmb is None:
                if match_level is None:
                    match_level = "Potentially Eligible"
                basis_parts.append(f"TMB: unknown (threshold ≥{tmb_min})")
            else:
                if match_level is None:
                    continue  # TMB too low, no other basis

        # TME check
        if req_tme:
            if tme in req_tme:
                if match_level is None:
                    match_level = "Eligible"
                basis_parts.append(f"TME: {tme}")
            else:
                if match_level == "Eligible":
                    match_level = "Potentially Eligible"
                    basis_parts.append(f"TME: {tme} (preferred: {'/'.join(req_tme)})")
                elif match_level is None:
                    continue

        # No-driver match
        if no_driver and not (mut_genes & DRIVER_GENES_TARGETED):
            if match_level is None:
                match_level = "Eligible"
            basis_parts.append("No actionable driver mutation")

        if match_level is None:
            continue

        rows.append({
            "nct_id":      trial["nct_id"],
            "name":        trial["name"],
            "drug":        trial["drug"],
            "drug_class":  trial["drug_class"],
            "target":      trial["target"],
            "phase":       trial["phase"],
            "status":      trial["status"],
            "line":        trial["line"],
            "match_level": match_level,
            "match_basis": " | ".join(basis_parts),
            "key_result":  trial["key_result"],
            "reference":   trial["reference"],
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Sort: Eligible first, then Phase III > II > I
    df["_phase_rank"] = df["phase"].map(lambda p: PHASE_RANK.get(p, 1))
    df["_level_rank"] = df["match_level"].map({"Eligible": 0, "Potentially Eligible": 1})
    df = df.sort_values(["_level_rank", "_phase_rank"], ascending=[True, False])
    return df.drop(columns=["_phase_rank", "_level_rank"]).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# MDT report figure
# ══════════════════════════════════════════════════════════════════════════════

def _fmt(val, default="—"):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return str(val)


def generate_mdt_figure(sample_id: str, data: dict,
                        matched_trials: pd.DataFrame) -> Path | None:
    """Generate a one-page MDT summary figure."""
    out_dir = OUT_DIR / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sample_id}_mdt_report.png"

    clin  = data.get("clinical", {})
    io    = data.get("io", {})
    tme   = data.get("tme", {})
    summ  = data.get("summary", {})
    recs  = data.get("recommendations", pd.DataFrame())

    fig   = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#f8f9fa")
    gs    = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35,
                             top=0.92, bottom=0.04, left=0.04, right=0.97)

    # ── Header ───────────────────────────────────────────────────────────────
    fig.text(0.5, 0.965, f"MDT Clinical Report — {sample_id}",
             ha="center", va="top", fontsize=16, fontweight="bold", color="#2c3e50")
    fig.text(0.5, 0.948, "TCGA-LUAD · Multi-Omics Precision Oncology Platform",
             ha="center", va="top", fontsize=10, color="#7f8c8d")

    PANEL_STYLE = dict(facecolor="white", edgecolor="#dfe6e9",
                       boxstyle="round,pad=0.5", linewidth=1)

    def panel_title(ax, title, color="#2c3e50"):
        ax.text(0.0, 1.04, title, transform=ax.transAxes,
                fontsize=10, fontweight="bold", color=color, va="bottom")

    # ── Panel A: Patient overview ─────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.axis("off")
    panel_title(ax_a, "A · Patient Overview")
    stage  = _fmt(clin.get("stage", summ.get("ts_alterations", "—")))
    age    = _fmt(clin.get("age_at_diagnosis"))
    gender = _fmt(clin.get("gender"))
    os_mo  = _fmt(clin.get("os_months"))
    event  = "Deceased" if clin.get("event") == 1 else "Alive/Censored"
    tmb_v  = _fmt(summ.get("tmb"))

    lines = [
        ("Stage",       stage),
        ("Age",         f"{age} yrs" if age != "—" else "—"),
        ("Gender",      gender),
        ("OS",          f"{os_mo} mo · {event}"),
        ("TMB",         f"{tmb_v} mut/Mb"),
        ("Data layers", _fmt(summ.get("data_layers"))),
    ]
    for i, (k, v) in enumerate(lines):
        ax_a.text(0.02, 0.93 - i * 0.15, f"{k}:", fontsize=9,
                  fontweight="bold", color="#555", transform=ax_a.transAxes)
        ax_a.text(0.38, 0.93 - i * 0.15, v, fontsize=9,
                  color="#2c3e50", transform=ax_a.transAxes)

    # ── Panel B: Immune profile ───────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.axis("off")
    panel_title(ax_b, "B · Immune Activity Profile")

    io_score   = io.get("io_score", np.nan)
    io_group   = _fmt(io.get("io_group"))
    tme_ph     = _fmt(tme.get("immune_phenotype") or summ.get("tme_phenotype"))
    til_d      = _fmt(io.get("til_density"))
    cd8        = _fmt(tme.get("CD8_T_cytotoxic"))
    m1         = _fmt(tme.get("Macrophage_M1"))
    treg       = _fmt(tme.get("Treg"))

    im_lines = [
        ("Immune Activity Score", f"{io_score:.1f} / 100 ({io_group})"
         if isinstance(io_score, (int, float)) and not np.isnan(io_score) else "—"),
        ("TME Phenotype",  tme_ph),
        ("TIL Density",    til_d),
        ("CD8 cytotoxic",  cd8),
        ("M1 macrophage",  m1),
        ("Treg",           treg),
    ]
    for i, (k, v) in enumerate(im_lines):
        ax_b.text(0.02, 0.93 - i * 0.15, f"{k}:", fontsize=9,
                  fontweight="bold", color="#555", transform=ax_b.transAxes)
        ax_b.text(0.52, 0.93 - i * 0.15, v, fontsize=9,
                  color="#1a9850", transform=ax_b.transAxes)

    # ── Panel C: Immune activity score gauge ──────────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.axis("off")
    panel_title(ax_c, "C · Immune Activity Score")
    if isinstance(io_score, (int, float)) and not np.isnan(io_score):
        theta   = np.linspace(np.pi, 0, 200)
        r_out, r_in = 1.0, 0.6
        cmap    = plt.cm.RdYlGn
        n_segs  = 100
        for j in range(n_segs):
            t0 = np.pi + j * np.pi / n_segs
            t1 = np.pi + (j + 1) * np.pi / n_segs
            t  = np.array([t0, t1, t1, t0])
            r  = np.array([r_out, r_out, r_in, r_in])
            x  = r * np.cos(t)
            y  = r * np.sin(t)
            ax_c.fill(x, y, color=cmap(j / n_segs), alpha=0.9)
        # Needle
        angle   = np.pi * (1 - io_score / 100)
        ax_c.plot([0, 0.85 * np.cos(angle)], [0, 0.85 * np.sin(angle)],
                  "k-", linewidth=2.5, zorder=5)
        ax_c.plot(0, 0, "ko", markersize=7, zorder=6)
        ax_c.text(0, -0.25, f"{io_score:.1f}", ha="center", fontsize=18,
                  fontweight="bold", color="#2c3e50")
        ax_c.text(-0.95, -0.05, "Low", ha="left",  fontsize=8, color="#e74c3c")
        ax_c.text( 0.95, -0.05, "High", ha="right", fontsize=8, color="#27ae60")
        ax_c.set_xlim(-1.15, 1.15)
        ax_c.set_ylim(-0.45, 1.15)

    # ── Panel D: Top recommendations ──────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, :2])
    ax_d.axis("off")
    panel_title(ax_d, "D · Top Treatment Recommendations (M09)")

    ICONS = {"Targeted Therapy": "🔵", "Immunotherapy": "🟢",
             "Combination Therapy": "🟣", "Chemotherapy": "🔴"}
    CAT_COLORS = {"Targeted Therapy": "#2980b9", "Immunotherapy": "#27ae60",
                  "Combination Therapy": "#8e44ad", "Chemotherapy": "#c0392b"}

    if not recs.empty:
        show = recs.head(5)
        cols_map = [("rank","#"), ("category","Category"), ("drug","Drug / Regimen"),
                    ("oncokb_level","OncoKB"), ("evidence","Evidence"),
                    ("confidence","Conf.")]
        headers = [c[1] for c in cols_map]
        col_keys = [c[0] for c in cols_map]
        col_x = [0.01, 0.07, 0.23, 0.62, 0.73, 0.90]

        for xi, h in zip(col_x, headers):
            ax_d.text(xi, 0.93, h, transform=ax_d.transAxes, fontsize=8,
                      fontweight="bold", color="#555")
        ax_d.axhline(y=0.88, xmin=0.0, xmax=1.0, color="#bdc3c7", linewidth=0.8)

        for ri, (_, rec) in enumerate(show.iterrows()):
            y = 0.80 - ri * 0.16
            cat = str(rec.get("category", ""))
            col = CAT_COLORS.get(cat, "#555")
            vals = [str(rec.get(k, "—")) for k in col_keys]
            vals[1] = f"{ICONS.get(cat, '⚪')} {vals[1]}"
            for xi, v in zip(col_x, vals):
                ax_d.text(xi, y, v[:38], transform=ax_d.transAxes,
                          fontsize=8, color=col if xi == col_x[2] else "#2c3e50",
                          fontweight="bold" if xi == col_x[2] else "normal")

    # ── Panel E: Matched trials ───────────────────────────────────────────────
    ax_e = fig.add_subplot(gs[1, 2])
    ax_e.axis("off")
    panel_title(ax_e, "E · Matched Clinical Trials")

    if not matched_trials.empty:
        show_t = matched_trials.head(5)
        for ri, (_, t) in enumerate(show_t.iterrows()):
            y = 0.90 - ri * 0.18
            lvl_color = "#27ae60" if t["match_level"] == "Eligible" else "#e67e22"
            ax_e.text(0.0, y, t["name"][:22], transform=ax_e.transAxes,
                      fontsize=8, fontweight="bold", color="#2c3e50")
            ax_e.text(0.0, y - 0.07, t["nct_id"], transform=ax_e.transAxes,
                      fontsize=7, color="#7f8c8d")
            ax_e.text(0.55, y, f"Ph{t['phase']}", transform=ax_e.transAxes,
                      fontsize=8, color="#555")
            ax_e.text(0.72, y, t["match_level"], transform=ax_e.transAxes,
                      fontsize=7, color=lvl_color, fontweight="bold")
    else:
        ax_e.text(0.5, 0.5, "No matched trials", ha="center", va="center",
                  transform=ax_e.transAxes, fontsize=9, color="#7f8c8d")

    # ── Panel F: Confidence bar chart ─────────────────────────────────────────
    ax_f = fig.add_subplot(gs[2, :2])
    panel_title(ax_f, "F · Recommendation Confidence Scores")
    if not recs.empty and "confidence" in recs.columns:
        show_r = recs.head(7)
        drugs  = [str(d)[:35] for d in show_r["drug"]]
        confs  = show_r["confidence"].astype(float).tolist()
        cats   = show_r["category"].tolist()
        colors = [CAT_COLORS.get(c, "#aaa") for c in cats]
        bars = ax_f.barh(range(len(drugs)), confs, color=colors,
                         edgecolor="white", linewidth=0.5, height=0.6)
        ax_f.set_yticks(range(len(drugs)))
        ax_f.set_yticklabels(drugs[::-1] if False else drugs, fontsize=8)
        ax_f.set_xlabel("Confidence score (0–100)", fontsize=8)
        ax_f.set_xlim(0, 105)
        for bar, val in zip(bars, confs):
            ax_f.text(val + 1, bar.get_y() + bar.get_height() / 2,
                      f"{int(val)}", va="center", fontsize=8)
        ax_f.invert_yaxis()
        ax_f.spines["top"].set_visible(False)
        ax_f.spines["right"].set_visible(False)

        legend_h = [mpatches.Patch(color=c, label=k)
                    for k, c in CAT_COLORS.items()]
        ax_f.legend(handles=legend_h, fontsize=7, loc="lower right", framealpha=0.8)

    # ── Panel G: Trial phases ─────────────────────────────────────────────────
    ax_g = fig.add_subplot(gs[2, 2])
    panel_title(ax_g, "G · Trial Match Summary")
    if not matched_trials.empty:
        elig_n  = (matched_trials["match_level"] == "Eligible").sum()
        pot_n   = (matched_trials["match_level"] == "Potentially Eligible").sum()
        ax_g.bar(["Eligible", "Potential"], [elig_n, pot_n],
                 color=["#27ae60", "#e67e22"], edgecolor="white", width=0.5)
        ax_g.set_ylabel("Trials", fontsize=8)
        for xi, v in enumerate([elig_n, pot_n]):
            ax_g.text(xi, v + 0.1, str(v), ha="center", fontsize=10,
                      fontweight="bold")
        ax_g.set_ylim(0, max(elig_n, pot_n) + 2)
        ax_g.spines["top"].set_visible(False)
        ax_g.spines["right"].set_visible(False)
    else:
        ax_g.axis("off")
        ax_g.text(0.5, 0.5, "No trials matched", ha="center", va="center",
                  transform=ax_g.transAxes, fontsize=9, color="#7f8c8d")

    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def run_patient(sample_id: str, logger) -> dict:
    data          = load_patient_data(sample_id)
    matched       = match_trials(data)
    out_path      = generate_mdt_figure(sample_id, data, matched)

    # Save matched trials
    if not matched.empty:
        (OUT_DIR / sample_id).mkdir(parents=True, exist_ok=True)
        matched.to_csv(OUT_DIR / sample_id / f"{sample_id}_matched_trials.tsv",
                       sep="\t", index=False)

    n_elig = (matched["match_level"] == "Eligible").sum() if not matched.empty else 0
    n_pot  = (matched["match_level"] == "Potentially Eligible").sum() if not matched.empty else 0
    logger.info(f"  {sample_id}: {n_elig} eligible, {n_pot} potential trials → {out_path.name}")
    return {"sample_id": sample_id, "n_eligible": n_elig, "n_potential": n_pot}


def main():
    parser = argparse.ArgumentParser(description="LUAD M10: Clinical trial matching + MDT report")
    parser.add_argument("--sample", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    logger = get_logger("luad-m10")
    logger.info(f"{'='*55}")
    logger.info("Module 10: Clinical Trial Matching + MDT Report")
    logger.info(f"  Trial database: {len(TRIAL_DB)} curated LUAD trials")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Discover samples
    if args.sample:
        samples = [args.sample]
    else:
        samples = sorted([
            d.name for d in INT_DIR.iterdir()
            if d.is_dir() and d.name.startswith("TCGA")
            and (INT_DIR / d.name / f"{d.name}_recommendation.tsv").exists()
        ]) if INT_DIR.exists() else []

    logger.info(f"  Processing {len(samples)} patients ...")
    if args.dry_run:
        logger.info("[DRY RUN] Done")
        return

    results = []
    for sid in samples:
        try:
            r = run_patient(sid, logger)
            results.append(r)
        except Exception as e:
            logger.warning(f"  {sid}: FAILED — {e}")

    if results:
        df_res = pd.DataFrame(results)
        df_res.to_csv(OUT_DIR / "m10_summary.tsv", sep="\t", index=False)
        logger.info(f"\n[Module 10 Complete] {len(results)} patients processed")
        logger.info(f"  Mean eligible trials: {df_res['n_eligible'].mean():.1f}")
        logger.info(f"  Mean potential trials: {df_res['n_potential'].mean():.1f}")


if __name__ == "__main__":
    main()
