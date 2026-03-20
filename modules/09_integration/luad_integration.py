#!/usr/bin/env python
"""
luad_integration.py (v3)
------------------------
Module 09: Multi-omics integration & treatment recommendation engine.

Design principles:
  - ALL recommendations shown and scored (not just top priority)
  - Evidence grading: OncoKB Levels + AMP/ASCO/CAP Tiers + ESCAT
  - Targeted therapy: All drugs from M07 (full 24-drug list), enriched with
    VAF clonality (M02), target expression (M03), CIViC resistance penalty
  - Immunotherapy: Multi-dimensional scoring (TMB + TME + PD-L1 + IFN-γ sig.
    + TIDE-like dysfunction/exclusion/CYT scores from M03)
  - Chemotherapy: Fixed fallback for all patients
  - Combination therapy: Co-mutation + TME driven (ESCAT Tier V logic)
  - Confidence 0-100 per recommendation, data-layer aware
  - M07 is now an internal data layer only; M09 is the sole user-facing output

Data inputs (all optional except M02):
  M02  → somatic variants, TMB            (272 patients)
  M07  → all targeted drug mappings       (103 patients)
  M03  → gene expression, IFN-γ, PD-L1   (517 patients)
  M04  → ssGSEA TME phenotype             (517 patients)
  M05  → H&E TIL density, TME phenotype  (478 patients)
  CIViC→ variant-drug resistance evidence (local: data/databases/civic_evidence.tsv.gz)

Key references:
  OncoKB:       Chakravarty et al., JCO Precis Oncol 2017 (PMID: 28890946)
  AMP/ASCO/CAP: Li et al., J Mol Diagnostics 2017       (PMID: 27993330)
  ESCAT:        Mateo et al., Ann Oncol 2018            (PMID: 30137196)
  IFN-γ sig.:   Ayers et al., JCI 2017                 (PMID: 28650338)
  TIDE:         Jiang et al., Nat Med 2018              (PMID: 30127393)
  CYT score:    Rooney et al., Cell 2015               (PMID: 25594174)
  TMB cutoff:   Marabelle et al., JAMA Oncol 2020

Usage:
  python luad_integration.py                        # all patients
  python luad_integration.py --sample TCGA-05-4244  # single patient
  python luad_integration.py --dry_run              # validate inputs only
"""

import argparse
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

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).parent
PROJECT_DIR   = SCRIPT_DIR.parent.parent
VARIANT_DIR   = PROJECT_DIR / "data/output/02_variants"
DRUG_DIR      = PROJECT_DIR / "data/output/07_drug_mapping"
EXPR_DIR      = PROJECT_DIR / "data/output/03_expression"
PATHOLOGY_DIR = PROJECT_DIR / "data/output/05_pathology"
SC_TME_DIR    = PROJECT_DIR / "data/output/04_single_cell"
OUT_DIR       = PROJECT_DIR / "data/output/09_integration"
CIVIC_PATH    = PROJECT_DIR / "data/databases/civic_evidence.tsv.gz"

# ── Evidence / level mappings ─────────────────────────────────────────────────
# Map M07 evidence levels → OncoKB levels
ONCOKB_LEVEL_MAP = {
    "FDA-Approved": "Level 1",
    "Phase II":     "Level 3A",
    "Phase I/II":   "Level 3A",
    "Clinical":     "Level 3B",
    "Preclinical":  "Level 4",
}

# OncoKB level → AMP/ASCO/CAP tier
AMP_TIER_MAP = {
    "Level 1":  "Tier I-A",
    "Level 2":  "Tier I-B",
    "Level 3A": "Tier II-C",
    "Level 3B": "Tier II-D",
    "Level 4":  "Tier II-D",
}

# Base confidence score by evidence level
BASE_SCORE_MAP = {
    "FDA-Approved":     85,
    "Phase II":         60,
    "Phase I/II":       52,
    "Clinical":         45,
    "Preclinical":      30,
    "Standard of Care": 40,
}

# ── Immunotherapy constants ───────────────────────────────────────────────────
TMB_HIGH_THRESHOLD  = 10.0   # mut/Mb — FDA threshold for pembrolizumab
TMB_INTER_THRESHOLD =  5.0   # mut/Mb — intermediate

# IFN-γ 10-gene signature (Ayers et al., JCI 2017, PMID: 28650338)
IFNG_SIGNATURE_GENES = [
    "IFNG", "STAT1", "CCR5", "CXCL9", "CXCL10",
    "IDO1", "PRF1", "GZMA", "HLA-DRA", "TIGIT",
]

# TIDE-like T-cell dysfunction markers (checkpoint/exhaustion genes)
# Jiang et al., Nature Medicine 2018 (PMID: 30127393)
TCELL_DYSFUNCTION_GENES = [
    "HAVCR2", "LAG3", "TIGIT", "PDCD1", "CTLA4",
    "BTLA", "TOX", "ENTPD1",
]

# TIDE-like T-cell exclusion markers
# Cancer-associated fibroblast (CAF) signature
TCELL_EXCLUSION_CAF = [
    "ACTA2", "FAP", "PDPN", "COL1A1", "COL3A1", "TGFB1", "POSTN",
]
# MDSC / myeloid exclusion signature
TCELL_EXCLUSION_MDSC = [
    "S100A8", "S100A9", "ARG1", "CEACAM8", "CXCR2",
]

# Cytolytic activity (CYT) — Rooney et al., Cell 2015 (PMID: 25594174)
CYT_GENES = ["GZMA", "PRF1"]

# Co-mutations associated with poor immunotherapy response (IO resistance)
IO_RESISTANCE_GENES = {"STK11", "KEAP1", "NFE2L2"}

# Tumor suppressor genes to flag
TUMOR_SUPPRESSORS = {"TP53", "STK11", "KEAP1", "RB1", "CDKN2A", "NFE2L2"}

# ── Colours ───────────────────────────────────────────────────────────────────
CATEGORY_COLORS = {
    "Targeted Therapy":   "#2166ac",
    "Immunotherapy":      "#1a9850",
    "Combination Therapy":"#7b2d8b",
    "Chemotherapy":       "#d73027",
}
TME_COLORS = {
    "Inflamed": "#2ca02c",
    "Excluded": "#ff7f0e",
    "Desert":   "#d62728",
    "Unknown":  "#999999",
}


# ── Logging ───────────────────────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
    return logging.getLogger(name)


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_variants(case_id: str) -> pd.DataFrame:
    """Load M02 somatic variants."""
    path = VARIANT_DIR / case_id / f"{case_id}_variants.tsv.gz"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def load_tmb(case_id: str) -> float:
    """Load TMB (missense only) from M02."""
    path = VARIANT_DIR / case_id / f"{case_id}_tmb.tsv"
    if not path.exists():
        return float("nan")
    df = pd.read_csv(path, sep="\t")
    row = df[df["tmb_measure"] == "TMB_missense_only"]
    if row.empty:
        return float("nan")
    return float(row.iloc[0]["tmb_estimate"])


def load_m07_drugs(case_id: str) -> pd.DataFrame:
    """Load all M07 drug mappings for a patient (full drug list)."""
    path = DRUG_DIR / case_id / f"{case_id}_drug_report.tsv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def load_expression(case_id: str) -> pd.DataFrame:
    """Load M03 gene expression. Returns DataFrame indexed by SYMBOL."""
    path = EXPR_DIR / case_id / f"{case_id}_gene_expression.tsv.gz"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, sep="\t")
    if "SYMBOL" not in df.columns:
        return pd.DataFrame()
    # For duplicated symbols keep highest TPM
    df = df.sort_values("TPM_GENE", ascending=False).drop_duplicates("SYMBOL")
    return df.set_index("SYMBOL")


def load_tme(case_id: str) -> dict:
    """Load TME phenotype and TIL scores from M05 (H&E pathology)."""
    path = PATHOLOGY_DIR / case_id / f"{case_id}_pathology_scores.tsv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, sep="\t")
    if df.empty:
        return {}
    row = df.iloc[0]
    return {
        "tme_phenotype": str(row.get("tme_phenotype", "Unknown")),
        "til_score":     float(row.get("til_score",   0)),
        "til_density":   float(row.get("til_density", 0)),
    }


def load_ssgsea_phenotype(case_id: str) -> str | None:
    """Load immune phenotype from M04 ssGSEA deconvolution."""
    path = SC_TME_DIR / case_id / f"{case_id}_tme_scores.tsv"
    if not path.exists():
        return None
    df = pd.read_csv(path, sep="\t")
    if df.empty or "immune_phenotype" not in df.columns:
        return None
    return str(df["immune_phenotype"].iloc[0])


def combine_tme_evidence(m05_tme: dict, ssgsea_phenotype: str | None) -> dict:
    """
    Combine M05 H&E pathology TME and M04 ssGSEA phenotype into a consensus call.

    Scoring rationale (TME component, 0-35):
      Both sources agree    → high confidence (+5 bonus)
        Inflamed + Inflamed → 35
        Excluded + Excluded → 20
        Desert   + Desert   → 0
      Sources disagree      → M05 preferred (has spatial info), score reduced by 5
      Only M05 available    → standard scores (30/15/0/8)
      Only M04 available    → reduced scores (20/10/0/5) — no spatial validation
      Neither available     → Unknown → 8
    """
    m05_phenotype = m05_tme.get("tme_phenotype", "Unknown") if m05_tme else "Unknown"
    has_m05 = m05_phenotype not in ("Unknown", "")
    has_m04 = ssgsea_phenotype is not None

    BASE_M05  = {"Inflamed": 30, "Excluded": 15, "Desert": 0, "Unknown": 8}
    BASE_M04  = {"Inflamed": 20, "Excluded": 10, "Desert": 0, "Unknown": 5}
    CONCORDANT_BONUS = 5

    if has_m05 and has_m04:
        if m05_phenotype == ssgsea_phenotype:
            # Concordant: high confidence
            base = BASE_M05.get(m05_phenotype, 8)
            tme_score = min(35, base + CONCORDANT_BONUS)
            consensus = m05_phenotype
            confidence_note = f"M04+M05 concordant: {consensus}"
            source = "M04+M05"
        else:
            # Discordant: M05 preferred, penalise confidence
            base = BASE_M05.get(m05_phenotype, 8)
            tme_score = max(0, base - 5)
            consensus = m05_phenotype           # M05 wins (spatial info)
            confidence_note = (
                f"M04/M05 discordant (M05={m05_phenotype}, M04={ssgsea_phenotype}; "
                f"M05 preferred — spatial data)"
            )
            source = "M04+M05"
    elif has_m05:
        tme_score = BASE_M05.get(m05_phenotype, 8)
        consensus = m05_phenotype
        confidence_note = f"M05 only: {m05_phenotype}"
        source = "M05"
    elif has_m04:
        tme_score = BASE_M04.get(ssgsea_phenotype, 5)
        consensus = ssgsea_phenotype
        confidence_note = f"M04 only (ssGSEA): {ssgsea_phenotype}"
        source = "M04"
    else:
        tme_score = 8
        consensus = "Unknown"
        confidence_note = "TME: no data"
        source = ""

    result = dict(m05_tme) if m05_tme else {}
    result.update({
        "tme_phenotype":    consensus,
        "tme_score":        tme_score,
        "tme_confidence":   confidence_note,
        "tme_source":       source,
        "ssgsea_phenotype": ssgsea_phenotype or "N/A",
    })
    return result


# ── Helper utilities ──────────────────────────────────────────────────────────

def get_gene_tpm(expr_df: pd.DataFrame, gene: str) -> float | None:
    """Return TPM for a gene symbol; None if not available."""
    if expr_df.empty or gene not in expr_df.index:
        return None
    return float(expr_df.loc[gene, "TPM_GENE"])


def get_driver_vaf(variants: pd.DataFrame, gene: str) -> float:
    """Return peak VAF for a gene across all somatic variants."""
    if variants.empty:
        return 0.0
    gv = variants[variants["SYMBOL"] == gene]
    if gv.empty:
        return 0.0
    return float(gv["VAF_TUMOR"].max())


def find_tumor_suppressors(variants: pd.DataFrame) -> list:
    """Flag high-impact tumor suppressor alterations."""
    if variants.empty:
        return []
    ts_vars = variants[
        (variants["SYMBOL"].isin(TUMOR_SUPPRESSORS)) &
        (variants["IMPACT"].isin(["HIGH", "MODERATE"]))
    ]
    results = []
    for gene, grp in ts_vars.groupby("SYMBOL"):
        top = grp.iloc[0]
        results.append({
            "gene":   gene,
            "hgvsp":  str(top.get("HGVSp_Short", "")),
            "impact": str(top.get("IMPACT", "")),
            "vaf":    float(top.get("VAF_TUMOR", 0)),
        })
    return results


# ── CIViC evidence (module-level cache) ──────────────────────────────────────
_CIVIC_CACHE: dict = {}

def _load_civic_luad() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and cache CIViC LUAD predictive evidence (Levels A/B/C).
    Returns (sensitivity_df, resistance_df).
    """
    global _CIVIC_CACHE
    if "sens" in _CIVIC_CACHE:
        return _CIVIC_CACHE["sens"], _CIVIC_CACHE["res"]

    if not CIVIC_PATH.exists():
        empty = pd.DataFrame()
        _CIVIC_CACHE["sens"] = empty
        _CIVIC_CACHE["res"]  = empty
        return empty, empty

    df = pd.read_csv(CIVIC_PATH, sep="\t")
    luad = df[
        (df["evidence_type"]   == "Predictive") &
        (df["evidence_status"] == "accepted") &
        (df["evidence_level"].isin(["A", "B", "C"])) &
        (df["significance"].isin(["Sensitivity/Response", "Resistance"])) &
        (df["disease"].str.contains("Lung|NSCLC|Non-Small", case=False, na=False))
    ].copy()

    # Parse simple molecular profiles (skip AND / fusion profiles)
    import re as _re
    def _parse_mp(mp: str) -> tuple[str, str]:
        mp = str(mp).strip()
        if " AND " in mp or "::" in mp:
            return "", ""
        parts = mp.split()
        return (parts[0], " ".join(parts[1:])) if len(parts) >= 2 else (parts[0] if parts else "", "")

    parsed = luad["molecular_profile"].apply(_parse_mp)
    luad["civic_gene"]     = parsed.apply(lambda x: x[0])
    luad["civic_mutation"] = parsed.apply(lambda x: x[1])
    luad = luad[luad["civic_gene"] != ""]

    sens = luad[luad["significance"] == "Sensitivity/Response"].copy()
    res  = luad[luad["significance"] == "Resistance"].copy()
    _CIVIC_CACHE["sens"] = sens
    _CIVIC_CACHE["res"]  = res
    return sens, res


def get_civic_resistance_penalty(
    drug: str,
    gene: str,
    mutation: str,
    civic_res: pd.DataFrame,
) -> tuple[int, str]:
    """
    Check CIViC resistance evidence for a specific drug-mutation pair.
    Returns (penalty_int, note_str).
    Penalty: Level A → -30, Level B → -20, Level C → -10.
    """
    if civic_res.empty:
        return 0, ""

    gene_res = civic_res[civic_res["civic_gene"] == gene]
    if gene_res.empty:
        return 0, ""

    drug_lower = drug.lower()
    mut_short  = str(mutation).lstrip("p.").upper() if mutation else ""

    matches = []
    for _, row in gene_res.iterrows():
        therapies  = str(row.get("therapies", "")).lower()
        civic_mut  = str(row.get("civic_mutation", "")).upper()
        drug_match = any(
            d.strip() in drug_lower or drug_lower in d.strip()
            for d in therapies.split(",")
        )
        mut_match = bool(mut_short) and (mut_short in civic_mut or civic_mut in mut_short)
        if drug_match and mut_match:
            matches.append((row.get("evidence_level", "C"),
                            str(row.get("molecular_profile", ""))))

    if not matches:
        return 0, ""

    lvl_penalty = {"A": -30, "B": -20, "C": -10}
    best = min(matches, key=lambda x: x[0])
    penalty = lvl_penalty.get(best[0], -5)
    note    = f"CIViC resistance Level {best[0]}: {best[1]}"
    return penalty, note


# ── TIDE-like immune scoring ──────────────────────────────────────────────────

def compute_tide_like_score(expr_df: pd.DataFrame) -> dict:
    """
    Compute TIDE-like T-cell dysfunction, exclusion, and CYT scores.
    Based on Jiang et al., Nature Medicine 2018 (PMID: 30127393) and
    Rooney et al., Cell 2015 (PMID: 25594174).

    T-cell dysfunction:  mean log2-TPM of 8 exhaustion/checkpoint genes
    T-cell exclusion:    mean of CAF + MDSC exclusion signature genes
    CYT score:           mean log2-TPM of GZMA + PRF1

    Returns dict with scores (None if insufficient genes available).
    """
    if expr_df.empty:
        return {"dysfunction": None, "exclusion": None, "cyt": None, "tide_note": "No M03"}

    def mean_log2(genes: list) -> float | None:
        avail = [g for g in genes if g in expr_df.index]
        if len(avail) < max(2, len(genes) // 2):
            return None
        return float(np.mean([float(expr_df.loc[g, "TPM_LOG2_GENE"]) for g in avail]))

    dys  = mean_log2(TCELL_DYSFUNCTION_GENES)
    caf  = mean_log2(TCELL_EXCLUSION_CAF)
    mdsc = mean_log2(TCELL_EXCLUSION_MDSC)
    cyt  = mean_log2(CYT_GENES)

    excl = None
    if caf is not None and mdsc is not None:
        excl = (caf + mdsc) / 2
    elif caf is not None:
        excl = caf
    elif mdsc is not None:
        excl = mdsc

    note_parts = []
    if dys  is not None: note_parts.append(f"Dysfunc={dys:.2f}")
    if excl is not None: note_parts.append(f"Excl={excl:.2f}")
    if cyt  is not None: note_parts.append(f"CYT={cyt:.2f}")

    return {
        "dysfunction": round(dys,  2) if dys  is not None else None,
        "exclusion":   round(excl, 2) if excl is not None else None,
        "cyt":         round(cyt,  2) if cyt  is not None else None,
        "tide_note":   " | ".join(note_parts) if note_parts else "Insufficient genes",
    }


def compute_ifng_score(expr_df: pd.DataFrame) -> float | None:
    """
    Compute IFN-γ signature score (mean log2-TPM of 10-gene panel).
    Reference: Ayers et al., JCI 2017 (PMID: 28650338).
    Returns None if fewer than 5 of 10 signature genes are available.
    """
    if expr_df.empty:
        return None
    available = [g for g in IFNG_SIGNATURE_GENES if g in expr_df.index]
    if len(available) < 5:
        return None
    vals = [float(expr_df.loc[g, "TPM_LOG2_GENE"]) for g in available]
    return float(np.mean(vals))


# ── Scoring functions ─────────────────────────────────────────────────────────

def score_targeted_drug(
    drug_row: pd.Series,
    variants: pd.DataFrame,
    expr_df: pd.DataFrame,
    ts_genes: list,
    civic_res: pd.DataFrame,
) -> dict:
    """
    Score a single targeted therapy recommendation (0-100).

    Components:
      Base score       — evidence level (FDA=85, Phase II=60, Phase I/II=52, Clinical=45)
      VAF bonus        — +8 clonal (VAF≥0.3), +3 moderate, -10 subclonal (<0.05)
      Expr bonus       — +5 if target gene TPM≥100, -3 if TPM<10
      CIViC resistance — -30 Level A, -20 Level B, -10 Level C
    """
    level    = str(drug_row.get("level",    "Preclinical"))
    gene     = str(drug_row.get("gene",     ""))
    mutation = str(drug_row.get("mutation", ""))
    drug     = str(drug_row.get("drug",     ""))
    base     = BASE_SCORE_MAP.get(level, 40)
    oncokb   = ONCOKB_LEVEL_MAP.get(level, "Level 4")
    amp_tier = AMP_TIER_MAP.get(oncokb, "Tier II-D")

    # VAF component
    vaf = get_driver_vaf(variants, gene)
    if vaf >= 0.30:
        vaf_bonus, vaf_note = 8,   f"VAF {vaf:.0%} (clonal)"
    elif vaf >= 0.05:
        vaf_bonus, vaf_note = 3,   f"VAF {vaf:.0%}"
    elif vaf > 0:
        vaf_bonus, vaf_note = -10, f"VAF {vaf:.0%} (subclonal)"
    else:
        vaf_bonus, vaf_note = 0,   ""

    # Expression component (M03)
    expr_bonus, expr_note = 0, ""
    tpm = get_gene_tpm(expr_df, gene)
    if tpm is not None:
        if tpm >= 100:
            expr_bonus, expr_note = 5,  f"{gene} high (TPM={tpm:.0f})"
        elif tpm >= 10:
            expr_bonus, expr_note = 0,  f"{gene} expressed (TPM={tpm:.0f})"
        else:
            expr_bonus, expr_note = -3, f"{gene} low (TPM={tpm:.1f})"

    # CIViC resistance penalty
    civic_penalty, civic_note = get_civic_resistance_penalty(drug, gene, mutation, civic_res)

    # IO resistance co-mutation note (informational only)
    ts_gene_names = {t["gene"] for t in ts_genes}
    io_res  = IO_RESISTANCE_GENES & ts_gene_names
    io_note = f"Co-mutation {'/'.join(sorted(io_res))} (IO resistance)" if io_res else ""

    final_score = max(5, min(98, base + vaf_bonus + expr_bonus + civic_penalty))

    layers = ["M02", "M07"]
    if not expr_df.empty:         layers.append("M03")
    if not civic_res.empty:       layers.append("CIViC")
    notes = [n for n in [vaf_note, expr_note, civic_note, io_note] if n]

    return {
        "oncokb_level":   oncokb,
        "amp_tier":       amp_tier,
        "confidence":     round(final_score),
        "civic_penalty":  civic_penalty,
        "vaf":            round(vaf, 3) if vaf > 0 else None,
        "data_layers":    "+".join(layers),
        "score_notes":    " | ".join(notes),
    }


def score_immunotherapy(
    tmb: float,
    tme: dict,
    expr_df: pd.DataFrame,
    ts_genes: list,
) -> dict:
    """
    Multi-dimensional immunotherapy composite score (0-100).

    Components (max 100 before penalty):
      TMB   (0-30): ≥10=30, ≥5=18, ≥1=8, <1=0
      TME   (0-30): Inflamed=30, Excluded=15, Desert=0, Unknown=8
      PD-L1 (0-10): CD274 expression (M03)
      IFN-γ (0-15): 10-gene IFN-γ signature (Ayers 2017, M03)
      TIDE  (0-15): dysfunction (+8), exclusion (-8), CYT (+7) — Jiang 2018 / Rooney 2015
    Modifier:
      IO resistance (-20): STK11/KEAP1/NFE2L2 co-mutation
    """
    tme_phenotype = tme.get("tme_phenotype", "Unknown")

    # TMB component (0-30)
    if not np.isnan(tmb):
        if tmb >= 10:
            tmb_score, tmb_label = 30, f"High ({tmb:.1f} mut/Mb)"
        elif tmb >= 5:
            tmb_score, tmb_label = 18, f"Intermediate ({tmb:.1f} mut/Mb)"
        elif tmb >= 1:
            tmb_score, tmb_label = 8,  f"Low ({tmb:.1f} mut/Mb)"
        else:
            tmb_score, tmb_label = 0,  f"Very low ({tmb:.1f} mut/Mb)"
    else:
        tmb_score, tmb_label = 0, "N/A"

    # TME component (0-35): use pre-computed combined score if available
    if "tme_score" in tme:
        tme_score = int(tme["tme_score"])
    else:
        tme_score = {"Inflamed": 30, "Excluded": 15, "Desert": 0, "Unknown": 8}.get(
            tme_phenotype, 8
        )

    # PD-L1 component (CD274 mRNA, 0-10)
    # Note: mRNA ≠ IHC protein; clinical decisions require IHC.
    pdl1_tpm = get_gene_tpm(expr_df, "CD274")
    if pdl1_tpm is not None:
        if pdl1_tpm >= 20:
            pdl1_score, pdl1_label = 10, f"High (TPM={pdl1_tpm:.1f}, RNA proxy)"
        elif pdl1_tpm >= 5:
            pdl1_score, pdl1_label = 5,  f"Intermediate (TPM={pdl1_tpm:.1f}, RNA proxy)"
        else:
            pdl1_score, pdl1_label = 0,  f"Low (TPM={pdl1_tpm:.1f}, RNA proxy)"
    else:
        pdl1_score, pdl1_label = 0, "N/A"

    # IFN-γ signature component (0-15)
    ifng_val = compute_ifng_score(expr_df)
    if ifng_val is not None:
        if ifng_val >= 8:
            ifng_score, ifng_label = 15, f"High ({ifng_val:.2f})"
        elif ifng_val >= 5:
            ifng_score, ifng_label = 8,  f"Medium ({ifng_val:.2f})"
        else:
            ifng_score, ifng_label = 2,  f"Low ({ifng_val:.2f})"
    else:
        ifng_score, ifng_label = 0, "N/A"

    # TIDE-like component (0-15): dysfunction + CYT bonus, exclusion penalty
    # Ref: Jiang et al., Nat Med 2018; Rooney et al., Cell 2015
    tide = compute_tide_like_score(expr_df)
    tide_score, tide_label = 0, "N/A"
    if tide["dysfunction"] is not None or tide["exclusion"] is not None:
        dys  = tide["dysfunction"] or 0
        excl = tide["exclusion"]   or 0
        cyt  = tide["cyt"]         or 0
        # High dysfunction + high CYT = active but exhausted immune response → IO responsive
        # High exclusion = T cells blocked → IO less likely to work
        dys_pts  = min(8, max(0, int((dys  - 4) * 2)))   # range ~[0,8]
        cyt_pts  = min(7, max(0, int((cyt  - 3) * 1.5))) # range ~[0,7]
        excl_pts = min(8, max(0, int((excl - 3) * 2)))   # penalty
        tide_score = max(0, dys_pts + cyt_pts - excl_pts)
        tide_label = tide["tide_note"]

    # IO resistance penalty
    ts_gene_names = {t["gene"] for t in ts_genes}
    io_res_genes  = IO_RESISTANCE_GENES & ts_gene_names
    io_penalty    = -20 if io_res_genes else 0
    io_res_label  = f"IO resistance: {'/'.join(sorted(io_res_genes))}" if io_res_genes else ""

    total = max(0, min(100,
        tmb_score + tme_score + pdl1_score + ifng_score + tide_score + io_penalty
    ))

    layers = []
    if not np.isnan(tmb):          layers.append("M02")
    tme_source = tme.get("tme_source", "")
    for src in ("M04", "M05"):
        if src in tme_source:
            layers.append(src)
    if pdl1_tpm is not None:       layers.append("M03")

    tme_label = tme.get("tme_confidence", tme_phenotype)

    return {
        "total_score":     round(total),
        "tmb_component":   tmb_score,
        "tme_component":   tme_score,
        "pdl1_component":  pdl1_score,
        "ifng_component":  ifng_score,
        "tide_component":  tide_score,
        "io_penalty":      io_penalty,
        "tmb_label":       tmb_label,
        "tme_label":       tme_label,
        "pdl1_label":      pdl1_label,
        "ifng_label":      ifng_label,
        "tide_label":      tide_label,
        "io_res_label":    io_res_label,
        "data_layers":     "+".join(layers) if layers else "M02",
    }


# ── Combination therapy evaluation (ESCAT Tier V) ────────────────────────────

def evaluate_combination_therapy(
    m07_drugs: pd.DataFrame,
    variants: pd.DataFrame,
    ts_genes: list,
    tme: dict,
    tmb: float,
    expr_df: pd.DataFrame,
) -> list:
    """
    Evaluate combination therapy options (ESCAT Tier V logic).

    Three biological logics:
      1. Vertical blockade:   same pathway upstream + downstream
      2. Horizontal blockade: co-mutation activates parallel/compensatory pathway
      3. Targeted + Immune:   driver mutation + TME-based immunotherapy synergy

    Confidence:
      High (≥60): FDA-approved individual components + prospective data
      Medium (40-59): preclinical/early clinical rationale + M02+M05 support
      Low (<40): purely biological rationale, limited data
    """
    combos = []
    tme_phenotype = tme.get("tme_phenotype", "Unknown")
    ts_gene_names = {t["gene"] for t in ts_genes}

    if m07_drugs.empty:
        return combos

    driver_genes = set(m07_drugs["gene"].unique())

    # ── Logic 2: Horizontal blockade ─────────────────────────────────────────
    # KRAS G12C + PIK3CA co-mutation → KRAS inh + PI3K inh
    if "KRAS" in driver_genes and not variants.empty:
        if not variants[variants["SYMBOL"] == "PIK3CA"].empty:
            combos.append({
                "drugs":      "Sotorasib + Alpelisib",
                "drug_class": "KRAS G12C inhibitor + PI3Kα inhibitor",
                "basis":      "Horizontal blockade",
                "logic":      "KRAS G12C + PIK3CA co-mutation: PI3K compensatory pathway activation",
                "escat_tier": "Tier V",
                "conf_score": 38,
                "data_layers":"M02+M07",
            })

    # KRAS G12C + MEK/ERK high expression → KRAS inh + MEK inh
    if "KRAS" in driver_genes and not expr_df.empty:
        mapk1 = get_gene_tpm(expr_df, "MAPK1")
        if mapk1 is not None and mapk1 > 50:
            combos.append({
                "drugs":      "Sotorasib + Trametinib",
                "drug_class": "KRAS G12C inhibitor + MEK inhibitor",
                "basis":      "Vertical blockade",
                "logic":      f"KRAS G12C + high MEK/ERK expression (MAPK1 TPM={mapk1:.0f}) → downstream blockade",
                "escat_tier": "Tier V",
                "conf_score": 40,
                "data_layers":"M02+M03+M07",
            })

    # EGFR + MET co-alteration → EGFR TKI + MET inh (resistance bypass)
    if "EGFR" in driver_genes and not variants.empty:
        if not variants[variants["SYMBOL"] == "MET"].empty:
            combos.append({
                "drugs":      "Osimertinib + Capmatinib",
                "drug_class": "3rd-gen EGFR TKI + MET inhibitor",
                "basis":      "Resistance bypass",
                "logic":      "EGFR sensitizing mutation + MET co-alteration: prevent MET-driven bypass resistance",
                "escat_tier": "Tier V",
                "conf_score": 52,
                "data_layers":"M02+M07",
            })

    # ── Logic 3: Targeted + Immune synergy ───────────────────────────────────
    # Any driver + Excluded TME → targeted then immunotherapy
    fda_drugs = m07_drugs[m07_drugs["level"] == "FDA-Approved"]
    if tme_phenotype == "Excluded" and not fda_drugs.empty:
        top = fda_drugs.iloc[0]
        combos.append({
            "drugs":      f"{top['drug']} + Pembrolizumab",
            "drug_class": f"{top['drug_class']} + PD-1 inhibitor",
            "basis":      "Targeted + Immune synergy",
            "logic":      (f"Excluded TME: {top['gene']} targeted therapy may remodel TME "
                           "to Inflamed → sequential PD-1 blockade"),
            "escat_tier": "Tier V",
            "conf_score": 48,
            "data_layers":"M02+M07+M05",
        })

    # KRAS G12C + Inflamed TME (no IO resistance) → dual targeting
    if "KRAS" in driver_genes and tme_phenotype == "Inflamed":
        if not (IO_RESISTANCE_GENES & ts_gene_names):
            combos.append({
                "drugs":      "Sotorasib + Pembrolizumab",
                "drug_class": "KRAS G12C inhibitor + PD-1 inhibitor",
                "basis":      "Targeted + Immune synergy",
                "logic":      "KRAS G12C + Inflamed TME (no IO resistance genes): concurrent dual targeting",
                "escat_tier": "Tier V",
                "conf_score": 55,
                "data_layers":"M02+M07+M05",
            })

    return combos


# ── Main recommendation generator ────────────────────────────────────────────

def generate_all_recommendations(
    case_id: str,
    m07_drugs: pd.DataFrame,
    variants: pd.DataFrame,
    tmb: float,
    tme: dict,
    expr_df: pd.DataFrame,
) -> tuple[list, list, dict]:
    """
    Generate ALL treatment recommendations for a patient.

    Returns:
      recs       — list of recommendation dicts (sorted by confidence desc.)
      ts_genes   — list of tumor suppressor alteration dicts
      io_scoring — immunotherapy scoring breakdown dict
    """
    recs     = []
    ts_genes = find_tumor_suppressors(variants)
    tme_phenotype = tme.get("tme_phenotype", "Unknown")

    # Load CIViC evidence (cached after first call)
    _, civic_res = _load_civic_luad()

    # ── 1. TARGETED THERAPY (all drugs from M07) ──────────────────────────────
    for _, drug_row in m07_drugs.iterrows():
        scoring = score_targeted_drug(drug_row, variants, expr_df, ts_genes, civic_res)
        mutation = str(drug_row.get("mutation", ""))
        gene     = str(drug_row.get("gene", ""))
        rationale = f"{gene} {mutation}"
        if scoring["score_notes"]:
            rationale += f" | {scoring['score_notes']}"
        recs.append({
            "category":         "Targeted Therapy",
            "drug":             drug_row["drug"],
            "drug_class":       drug_row["drug_class"],
            "oncokb_level":     scoring["oncokb_level"],
            "amp_tier":         scoring["amp_tier"],
            "escat_tier":       "Tier I-A" if drug_row["level"] == "FDA-Approved" else "Tier II-C",
            "evidence":         drug_row["level"],
            "line":             str(drug_row.get("line", "")),
            "confidence":       scoring["confidence"],
            "tmb_support":      "N/A",
            "tme_support":      "N/A",
            "data_layers":      scoring["data_layers"],
            "rationale":        rationale,
            "combination_basis":"",
        })

    # ── 2. IMMUNOTHERAPY (multi-dimensional scoring) ──────────────────────────
    io_scoring = score_immunotherapy(tmb, tme, expr_df, ts_genes)
    io_score   = io_scoring["total_score"]

    # Determine evidence level for pembrolizumab
    tmb_high = (not np.isnan(tmb)) and (tmb >= TMB_HIGH_THRESHOLD)
    if tmb_high:
        io_evidence, io_oncokb, io_amp = "FDA-Approved (TMB-H)", "Level 1", "Tier I-A"
    elif tme_phenotype == "Inflamed":
        io_evidence, io_oncokb, io_amp = "FDA-Approved",          "Level 1", "Tier I-A"
    elif tme_phenotype == "Excluded":
        io_evidence, io_oncokb, io_amp = "FDA-Approved (± chemo)","Level 1", "Tier I-B"
    else:
        io_evidence, io_oncokb, io_amp = "Clinical Evidence",     "Level 3A","Tier II-C"

    rationale_parts = [
        f"TMB: {io_scoring['tmb_label']}",
        f"TME: {io_scoring['tme_label']}",
    ]
    if io_scoring["pdl1_label"] != "N/A":
        rationale_parts.append(f"PD-L1(RNA): {io_scoring['pdl1_label']}")
    if io_scoring["ifng_label"] != "N/A":
        rationale_parts.append(f"IFN-γ: {io_scoring['ifng_label']}")
    if io_scoring["tide_label"] != "N/A":
        rationale_parts.append(f"TIDE: {io_scoring['tide_label']}")
    if io_scoring["io_res_label"]:
        rationale_parts.append(f"⚠ {io_scoring['io_res_label']}")

    recs.append({
        "category":         "Immunotherapy",
        "drug":             "Pembrolizumab",
        "drug_class":       "PD-1 inhibitor",
        "oncokb_level":     io_oncokb,
        "amp_tier":         io_amp,
        "escat_tier":       "Tier I-A" if io_oncokb == "Level 1" else "Tier II-C",
        "evidence":         io_evidence,
        "line":             "1L/2L",
        "confidence":       io_score,
        "tmb_support":      io_scoring["tmb_label"],
        "tme_support":      io_scoring["tme_label"],
        "data_layers":      io_scoring["data_layers"],
        "rationale":        " | ".join(rationale_parts),
        "combination_basis":"",
    })

    # ── 3. COMBINATION THERAPY (ESCAT Tier V) ────────────────────────────────
    for combo in evaluate_combination_therapy(
        m07_drugs, variants, ts_genes, tme, tmb, expr_df
    ):
        recs.append({
            "category":         "Combination Therapy",
            "drug":             combo["drugs"],
            "drug_class":       combo["drug_class"],
            "oncokb_level":     "Level 3A",
            "amp_tier":         "Tier II-C",
            "escat_tier":       combo["escat_tier"],
            "evidence":         "Investigational",
            "line":             "Investigational",
            "confidence":       combo["conf_score"],
            "tmb_support":      "N/A",
            "tme_support":      tme_phenotype if "M05" in combo["data_layers"] else "N/A",
            "data_layers":      combo["data_layers"],
            "rationale":        combo["logic"],
            "combination_basis":combo["basis"],
        })

    # ── 4. CHEMOTHERAPY (fallback — always included) ──────────────────────────
    chemo_parts = ["Standard platinum-based doublet"]
    if not m07_drugs.empty:
        chemo_parts.append("consider alongside targeted therapy")
    if tme_phenotype == "Desert":
        chemo_parts.append("Desert TME — immunotherapy less effective")
    ts_str = ", ".join(t["gene"] for t in ts_genes)
    if ts_str:
        chemo_parts.append(f"TS alterations: {ts_str}")

    recs.append({
        "category":         "Chemotherapy",
        "drug":             "Carboplatin + Pemetrexed",
        "drug_class":       "Platinum doublet",
        "oncokb_level":     "N/A",
        "amp_tier":         "N/A",
        "escat_tier":       "N/A",
        "evidence":         "Standard of Care",
        "line":             "1L/2L",
        "confidence":       40,
        "tmb_support":      "N/A",
        "tme_support":      "N/A",
        "data_layers":      "Standard",
        "rationale":        " | ".join(chemo_parts),
        "combination_basis":"",
    })

    # Sort by confidence (descending) and assign ranks
    recs.sort(key=lambda x: x["confidence"], reverse=True)
    for i, r in enumerate(recs, 1):
        r["rank"] = i

    return recs, ts_genes, io_scoring


# ── Visualization ─────────────────────────────────────────────────────────────

def make_integration_report(
    case_id: str,
    recs: list,
    ts_genes: list,
    tmb: float,
    tme: dict,
    io_scoring: dict,
    m07_drugs: pd.DataFrame,
    out_path: Path,
):
    """Generate comprehensive 2-panel integration report."""
    n_recs  = len(recs)
    fig_h   = max(10, min(20, 3.5 + n_recs * 0.85))

    fig = plt.figure(figsize=(17, fig_h), facecolor="#fafafa")
    fig.suptitle(
        f"Multi-Omics Treatment Recommendation Report  —  {case_id}",
        fontsize=13, fontweight="bold", y=0.99,
    )

    gs = fig.add_gridspec(
        1, 2, width_ratios=[1, 2.5],
        left=0.02, right=0.98, top=0.95, bottom=0.03, wspace=0.06,
    )
    ax_l = fig.add_subplot(gs[0])
    ax_r = fig.add_subplot(gs[1])
    ax_l.axis("off")
    ax_r.axis("off")

    tme_phenotype = tme.get("tme_phenotype", "Unknown")
    tme_color     = TME_COLORS.get(tme_phenotype, "#999999")
    tmb_str       = f"{tmb:.1f} mut/Mb" if not np.isnan(tmb) else "N/A"

    # ── LEFT PANEL: Molecular Profile ─────────────────────────────────────────
    ax_l.set_title("Molecular Profile", fontsize=11, fontweight="bold", pad=8)
    y = 0.97

    def add_block(title, items, color="#333333"):
        nonlocal y
        ax_l.text(0.04, y, title, transform=ax_l.transAxes,
                  fontsize=9, fontweight="bold", color=color)
        y -= 0.05
        for item in items:
            ax_l.text(0.07, y, f"• {item}", transform=ax_l.transAxes,
                      fontsize=8.5, color="#555555")
            y -= 0.048
        y -= 0.012

    # Targetable drivers
    if not m07_drugs.empty:
        driver_items = [
            f"{row['gene']} {row['mutation']}"
            for _, row in m07_drugs.drop_duplicates("gene").iterrows()
        ]
        add_block("Targetable Drivers (M02+M07)", driver_items, "#2166ac")
    else:
        add_block("Targetable Drivers", ["No targetable mutations detected"], "#2166ac")

    # Tumor suppressors
    ts_items = [f"{t['gene']} {t['hgvsp']} [{t['impact']}]" for t in ts_genes] or ["None detected"]
    add_block("Tumor Suppressors (M02)", ts_items, "#8c510a")

    # TMB
    tmb_flag = "  [HIGH ✓]" if (not np.isnan(tmb)) and tmb >= TMB_HIGH_THRESHOLD else ""
    add_block("TMB (M02)", [f"{tmb_str}{tmb_flag}"], "#444444")

    # TME
    add_block(
        "TME Phenotype (M05)",
        [f"{tme_phenotype}  (TIL score: {tme.get('til_score', 0):.2f})"],
        tme_color,
    )

    # IO score breakdown
    io_items = [
        f"TMB:   {io_scoring['tmb_label']}  (+{io_scoring['tmb_component']})",
        f"TME:   {io_scoring['tme_label']}  (+{io_scoring['tme_component']})",
    ]
    if io_scoring["pdl1_label"] != "N/A":
        io_items.append(f"PD-L1(RNA): {io_scoring['pdl1_label']}  (+{io_scoring['pdl1_component']})")
    if io_scoring["ifng_label"] != "N/A":
        io_items.append(f"IFN-γ sig:  {io_scoring['ifng_label']}  (+{io_scoring['ifng_component']})")
    if io_scoring.get("tide_component", 0) > 0 and io_scoring["tide_label"] != "N/A":
        io_items.append(f"TIDE-like:  {io_scoring['tide_label']}  (+{io_scoring['tide_component']})")
    if io_scoring["io_penalty"] < 0:
        io_items.append(f"⚠ IO resistance penalty: {io_scoring['io_penalty']}")
    io_items.append(f"→  IO total score: {io_scoring['total_score']} / 100")
    add_block("Immunotherapy Scoring (M02+M05+M03+TIDE)", io_items, "#1a9850")

    # TME colour badge
    if y > 0.07:
        by = max(0.03, y - 0.01)
        ax_l.add_patch(mpatches.FancyBboxPatch(
            (0.08, by), 0.84, 0.065, boxstyle="round,pad=0.02",
            linewidth=1.5, edgecolor=tme_color, facecolor=tme_color + "22",
            transform=ax_l.transAxes, clip_on=False,
        ))
        ax_l.text(0.50, by + 0.032, f"TME: {tme_phenotype}",
                  transform=ax_l.transAxes, fontsize=11, fontweight="bold",
                  color=tme_color, ha="center", va="center")

    # ── RIGHT PANEL: Recommendations Table ────────────────────────────────────
    ax_r.set_title(
        "All Treatment Recommendations  (sorted by confidence score)",
        fontsize=11, fontweight="bold", pad=8,
    )

    # Column positions
    cx = [0.00, 0.18, 0.43, 0.59, 0.72, 0.84]
    headers = ["Category", "Drug / Class", "OncoKB / AMP", "Evidence", "Score", "Data"]
    for xi, h in zip(cx, headers):
        ax_r.text(xi, 0.975, h, transform=ax_r.transAxes,
                  fontsize=8.5, fontweight="bold", color="#333333")
    ax_r.axhline(y=0.968, xmin=0, xmax=1, color="#cccccc", linewidth=0.8)

    row_h = min(0.085, 0.93 / max(len(recs), 1))
    y2    = 0.955

    for rec in recs:
        if y2 < 0.02:
            break
        cat   = rec["category"]
        color = CATEGORY_COLORS.get(cat, "#555555")
        conf  = rec["confidence"]

        # Category badge
        cat_short = {
            "Targeted Therapy":   "Targeted",
            "Immunotherapy":      "Immuno",
            "Combination Therapy":"Combo",
            "Chemotherapy":       "Chemo",
        }.get(cat, cat)
        ax_r.add_patch(mpatches.FancyBboxPatch(
            (cx[0], y2 - row_h * 0.58), 0.165, row_h * 0.68,
            boxstyle="round,pad=0.01", facecolor=color + "22", edgecolor=color,
            transform=ax_r.transAxes, clip_on=False, linewidth=1.0,
        ))
        ax_r.text(cx[0] + 0.083, y2 - row_h * 0.2, cat_short,
                  transform=ax_r.transAxes, fontsize=7.5, fontweight="bold",
                  color=color, ha="center", va="center")

        # Drug name + class
        drug_display = rec["drug"]
        if len(drug_display) > 32:
            drug_display = drug_display[:30] + "…"
        ax_r.text(cx[1], y2, drug_display, transform=ax_r.transAxes,
                  fontsize=9, fontweight="bold", color="#111111", va="top")
        ax_r.text(cx[1], y2 - row_h * 0.44, rec["drug_class"][:32],
                  transform=ax_r.transAxes, fontsize=7.5, color="#777777", va="top")

        # OncoKB + AMP tier
        ax_r.text(cx[2], y2, rec["oncokb_level"],
                  transform=ax_r.transAxes, fontsize=8.5, color="#333333", va="top")
        ax_r.text(cx[2], y2 - row_h * 0.44, rec["amp_tier"],
                  transform=ax_r.transAxes, fontsize=7.5, color="#999999", va="top")

        # Evidence + line
        ax_r.text(cx[3], y2, rec["evidence"],
                  transform=ax_r.transAxes, fontsize=8.5, color="#333333", va="top")
        ax_r.text(cx[3], y2 - row_h * 0.44, rec.get("line", ""),
                  transform=ax_r.transAxes, fontsize=7.5, color="#999999", va="top")

        # Confidence score bar
        bar_w = 0.10
        bar_h = row_h * 0.45
        ax_r.add_patch(mpatches.FancyBboxPatch(
            (cx[4], y2 - row_h * 0.58), bar_w, bar_h,
            boxstyle="round,pad=0.005", facecolor="#eeeeee", edgecolor="#cccccc",
            transform=ax_r.transAxes, clip_on=False, linewidth=0.5,
        ))
        ax_r.add_patch(mpatches.FancyBboxPatch(
            (cx[4], y2 - row_h * 0.58), bar_w * conf / 100, bar_h,
            boxstyle="round,pad=0.005", facecolor=color, edgecolor="none",
            transform=ax_r.transAxes, clip_on=False,
        ))
        ax_r.text(cx[4] + bar_w + 0.01, y2 - row_h * 0.30,
                  str(conf), transform=ax_r.transAxes,
                  fontsize=8.5, fontweight="bold", color=color, va="center")

        # Data layers
        ax_r.text(cx[5], y2 - row_h * 0.22, rec["data_layers"],
                  transform=ax_r.transAxes, fontsize=7.5, color="#888888", va="center")

        # Rationale (truncated)
        rat = rec["rationale"]
        if len(rat) > 72:
            rat = rat[:70] + "…"
        ax_r.text(cx[1], y2 - row_h * 0.76,
                  f"↳ {rat}", transform=ax_r.transAxes,
                  fontsize=7, color="#aaaaaa", va="top", style="italic")

        # Row separator
        ax_r.plot([0, 1], [y2 - row_h * 0.97, y2 - row_h * 0.97],
                  color="#eeeeee", linewidth=0.6, transform=ax_r.transAxes)
        y2 -= row_h

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


# ── Per-patient pipeline ──────────────────────────────────────────────────────

def run_patient(case_id: str, logger) -> dict | None:
    """Run full integration pipeline for one patient."""
    logger.info(f"[{case_id}]")

    variants  = load_variants(case_id)
    if variants.empty:
        logger.warning(f"  No M02 variant data — skipping")
        return None

    tmb            = load_tmb(case_id)
    m05_tme        = load_tme(case_id)
    ssgsea_pheno   = load_ssgsea_phenotype(case_id)
    tme            = combine_tme_evidence(m05_tme, ssgsea_pheno)
    m07_drugs      = load_m07_drugs(case_id)
    expr_df        = load_expression(case_id)

    has_m07 = not m07_drugs.empty
    has_m05 = bool(m05_tme)
    has_m04 = ssgsea_pheno is not None
    has_m03 = not expr_df.empty
    logger.info(
        f"  Data: M02✓ M07{'✓' if has_m07 else '✗'} "
        f"M05{'✓' if has_m05 else '✗'} M04{'✓' if has_m04 else '✗'} "
        f"M03{'✓' if has_m03 else '✗'} | TME={tme['tme_phenotype']} "
        f"({tme.get('tme_confidence','')[:40]})"
    )

    recs, ts_genes, io_scoring = generate_all_recommendations(
        case_id, m07_drugs, variants, tmb, tme, expr_df
    )

    # Save TSV
    case_out = OUT_DIR / case_id
    case_out.mkdir(parents=True, exist_ok=True)

    rec_df = pd.DataFrame(recs)
    col_order = [
        "rank", "category", "drug", "drug_class",
        "oncokb_level", "amp_tier", "escat_tier",
        "evidence", "line", "confidence",
        "tmb_support", "tme_support", "data_layers",
        "rationale", "combination_basis",
    ]
    rec_df = rec_df[[c for c in col_order if c in rec_df.columns]]
    rec_df.to_csv(case_out / f"{case_id}_recommendation.tsv", sep="\t", index=False)

    # Save figure
    make_integration_report(
        case_id, recs, ts_genes, tmb, tme, io_scoring, m07_drugs,
        case_out / f"{case_id}_integration_report.png",
    )

    n_targeted = sum(1 for r in recs if r["category"] == "Targeted Therapy")
    n_combo    = sum(1 for r in recs if r["category"] == "Combination Therapy")
    logger.info(
        f"  → {len(recs)} recs  "
        f"(targeted={n_targeted}, combo={n_combo})  "
        f"IO score={io_scoring['total_score']}"
    )

    top_t = next((r for r in recs if r["category"] == "Targeted Therapy"), None)
    return {
        "sample_id":          case_id,
        "data_layers":        sum([True, has_m07, has_m05, has_m03]),
        "n_targeted":         n_targeted,
        "n_immunotherapy":    sum(1 for r in recs if r["category"] == "Immunotherapy"),
        "n_combination":      n_combo,
        "n_recommendations":  len(recs),
        "top_targeted_drug":  top_t["drug"]       if top_t else "None",
        "top_targeted_score": top_t["confidence"] if top_t else None,
        "io_score":           io_scoring["total_score"],
        "tmb":                round(tmb, 2) if not np.isnan(tmb) else None,
        "tmb_high":           bool((not np.isnan(tmb)) and tmb >= TMB_HIGH_THRESHOLD),
        "tme_phenotype":      tme.get("tme_phenotype", "Unknown"),
        "ts_alterations":     "; ".join(t["gene"] for t in ts_genes) or "None",
    }


def discover_patients() -> list:
    """Find all patients with M02 variant output."""
    if not VARIANT_DIR.exists():
        return []
    return sorted([
        d.name for d in VARIANT_DIR.iterdir()
        if d.is_dir() and (d / f"{d.name}_variants.tsv.gz").exists()
    ])


def main():
    parser = argparse.ArgumentParser(description="LUAD multi-omics integration v2 (M09)")
    parser.add_argument("--sample",  type=str,  help="Single patient ID")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    logger   = get_logger("luad-integration")
    logger.info(f"\n{'='*60}")
    logger.info("Module 09: Multi-Omics Integration & Treatment Recommendation (v2)")

    patients = [args.sample] if args.sample else discover_patients()
    logger.info(f"Patients to process: {len(patients)}")

    if args.dry_run:
        logger.info(f"[DRY RUN] First 5: {patients[:5]}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows, success, failed = [], 0, 0
    for case_id in patients:
        try:
            row = run_patient(case_id, logger)
            if row:
                rows.append(row)
                success += 1
        except Exception as e:
            logger.error(f"  [{case_id}] Failed: {e}")
            failed += 1

    if rows:
        summary = pd.DataFrame(rows)
        summary_path = OUT_DIR / "integration_summary.tsv"
        summary.to_csv(summary_path, sep="\t", index=False)
        logger.info(f"\nCohort summary → {summary_path}")

        logger.info("\nTME distribution:")
        for phe, cnt in summary["tme_phenotype"].value_counts().items():
            logger.info(f"  {phe}: {cnt} ({cnt/len(summary):.1%})")

        logger.info("\nTop targeted drugs:")
        for drug, cnt in summary["top_targeted_drug"].value_counts().head(8).items():
            logger.info(f"  {drug}: {cnt}")

        logger.info(f"\nMean IO score: {summary['io_score'].mean():.1f}")

    logger.info(f"\n[Module 09 Complete] Success: {success}  Failed: {failed}")


if __name__ == "__main__":
    main()
