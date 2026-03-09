#!/usr/bin/env python
"""
luad_integration.py
-------------------
Module 09: Multi-omics integration & treatment recommendation engine.

Integrates outputs from M02, M07, M08 using evidence-based rules
(NCCN NSCLC 2024, FDA approvals) to generate per-patient treatment summaries.

Integration logic (priority order):
  1. Targetable driver mutation (M02) → matched targeted therapy (M07)
  2. TME phenotype (M08) → immunotherapy eligibility
  3. TMB (M02) → immunotherapy support
  4. Fallback → platinum-based chemotherapy

Outputs per patient:
  data/output/09_integration/{case_id}/
    {case_id}_recommendation.tsv   — ranked treatment recommendations
    {case_id}_integration_report.png — summary card figure

Cohort summary:
  data/output/09_integration/integration_summary.tsv

Usage:
  python luad_integration.py                         # all patients
  python luad_integration.py --sample TCGA-49-4507   # single patient
  python luad_integration.py --dry_run               # validate inputs only
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
SCRIPT_DIR   = Path(__file__).parent
PROJECT_DIR  = SCRIPT_DIR.parent.parent
VARIANT_DIR  = PROJECT_DIR / "data/output/02_variation_annotation"
PATHOLOGY_DIR= PROJECT_DIR / "data/output/08_pathology"
DRUG_DIR     = PROJECT_DIR / "data/output/07_drug_mapping"
OUT_DIR      = PROJECT_DIR / "data/output/09_integration"

# ── LUAD Driver Gene Rules ────────────────────────────────────────────────────
# Each rule: gene → {hgvsp_pattern, drugs, rationale}
# Based on NCCN NSCLC v2.2024 + FDA approvals

DRIVER_RULES = [
    {
        "gene": "EGFR",
        "pattern": r"L858R|p\.L858|del19|delE746|exon19|G719|S768|L861",
        "mutation_label": "EGFR sensitizing mutation",
        "drugs": ["Osimertinib"],
        "drug_class": "3rd-gen EGFR TKI",
        "evidence": "FDA-Approved (1L)",
        "priority": 1,
    },
    {
        "gene": "EGFR",
        "pattern": r"T790M|p\.T790",
        "mutation_label": "EGFR T790M (resistance)",
        "drugs": ["Osimertinib"],
        "drug_class": "3rd-gen EGFR TKI",
        "evidence": "FDA-Approved (2L)",
        "priority": 1,
    },
    {
        "gene": "EGFR",
        "pattern": r"exon20ins|exon_20|ins.*exon20|C797",
        "mutation_label": "EGFR exon 20 insertion",
        "drugs": ["Amivantamab", "Mobocertinib"],
        "drug_class": "EGFR exon20 inhibitor",
        "evidence": "FDA-Approved",
        "priority": 1,
    },
    {
        "gene": "KRAS",
        "pattern": r"G12C|p\.G12C",
        "mutation_label": "KRAS G12C",
        "drugs": ["Sotorasib", "Adagrasib"],
        "drug_class": "KRAS G12C inhibitor",
        "evidence": "FDA-Approved",
        "priority": 1,
    },
    {
        "gene": "BRAF",
        "pattern": r"V600E|p\.V600E",
        "mutation_label": "BRAF V600E",
        "drugs": ["Dabrafenib + Trametinib"],
        "drug_class": "BRAF+MEK inhibitor",
        "evidence": "FDA-Approved",
        "priority": 1,
    },
    {
        "gene": "MET",
        "pattern": r"exon14|splice|skipping|Y1003|D1010",
        "mutation_label": "MET exon 14 skipping",
        "drugs": ["Capmatinib", "Tepotinib"],
        "drug_class": "MET inhibitor",
        "evidence": "FDA-Approved",
        "priority": 1,
    },
    {
        "gene": "RET",
        "pattern": r"fusion|rearrang|KIF5B|CCDC6",
        "mutation_label": "RET fusion",
        "drugs": ["Selpercatinib", "Pralsetinib"],
        "drug_class": "RET inhibitor",
        "evidence": "FDA-Approved",
        "priority": 1,
    },
    {
        "gene": "ALK",
        "pattern": r"fusion|rearrang|EML4",
        "mutation_label": "ALK fusion",
        "drugs": ["Alectinib", "Brigatinib", "Lorlatinib"],
        "drug_class": "ALK inhibitor",
        "evidence": "FDA-Approved (1L)",
        "priority": 1,
    },
    {
        "gene": "ROS1",
        "pattern": r"fusion|rearrang|CD74",
        "mutation_label": "ROS1 fusion",
        "drugs": ["Entrectinib", "Crizotinib"],
        "drug_class": "ROS1 inhibitor",
        "evidence": "FDA-Approved",
        "priority": 1,
    },
    {
        "gene": "ERBB2",
        "pattern": r"exon20|ins|amp|overexpress|p\.Y772|p\.V777|p\.G778",
        "mutation_label": "ERBB2 (HER2) mutation/amplification",
        "drugs": ["Trastuzumab deruxtecan (T-DXd)"],
        "drug_class": "HER2-directed ADC",
        "evidence": "FDA-Approved",
        "priority": 1,
    },
    {
        "gene": "NTRK1",
        "pattern": r"fusion|rearrang",
        "mutation_label": "NTRK1 fusion",
        "drugs": ["Larotrectinib", "Entrectinib"],
        "drug_class": "TRK inhibitor",
        "evidence": "FDA-Approved",
        "priority": 1,
    },
    {
        "gene": "NTRK2",
        "pattern": r"fusion|rearrang",
        "mutation_label": "NTRK2 fusion",
        "drugs": ["Larotrectinib", "Entrectinib"],
        "drug_class": "TRK inhibitor",
        "evidence": "FDA-Approved",
        "priority": 1,
    },
    {
        "gene": "NTRK3",
        "pattern": r"fusion|rearrang",
        "mutation_label": "NTRK3 fusion",
        "drugs": ["Larotrectinib", "Entrectinib"],
        "drug_class": "TRK inhibitor",
        "evidence": "FDA-Approved",
        "priority": 1,
    },
]

# Genes to flag as tumor suppressors (no targeted therapy, but clinically relevant)
TUMOR_SUPPRESSORS = {"TP53", "STK11", "KEAP1", "RB1", "CDKN2A"}

TMB_HIGH_THRESHOLD = 10.0  # mutations/Mb


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
    return logging.getLogger(name)


# ── Data loaders ─────────────────────────────────────────────────────────────

def load_variants(case_id: str) -> pd.DataFrame:
    """Load M02 variant table for one patient."""
    path = VARIANT_DIR / case_id / f"{case_id}_variants.tsv.gz"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def load_tmb(case_id: str) -> float:
    """Load TMB (missense only) from M02 output."""
    path = VARIANT_DIR / case_id / f"{case_id}_tmb.tsv"
    if not path.exists():
        return float("nan")
    df = pd.read_csv(path, sep="\t")
    row = df[df["tmb_measure"] == "TMB_missense_only"]
    if row.empty:
        return float("nan")
    return float(row.iloc[0]["tmb_estimate"])


def load_tme(case_id: str) -> dict:
    """Load TME phenotype from M08 output."""
    path = PATHOLOGY_DIR / case_id / f"{case_id}_pathology_scores.tsv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, sep="\t")
    if df.empty:
        return {}
    row = df.iloc[0]
    return {
        "tme_phenotype": row.get("tme_phenotype", "Unknown"),
        "til_score":     float(row.get("til_score", 0)),
        "til_density":   float(row.get("til_density", 0)),
    }


# ── Rule engine ───────────────────────────────────────────────────────────────

def find_driver_mutations(variants: pd.DataFrame) -> list:
    """
    Match somatic variants against DRIVER_RULES.
    Returns list of matched rule dicts with added 'hgvsp' field.
    """
    if variants.empty:
        return []

    matches = []
    seen_genes = set()

    for rule in DRIVER_RULES:
        gene = rule["gene"]
        if gene in seen_genes:
            continue

        gene_vars = variants[variants["SYMBOL"] == gene]
        if gene_vars.empty:
            continue

        import re
        for _, row in gene_vars.iterrows():
            hgvsp = str(row.get("HGVSp_Short", ""))
            csq   = str(row.get("CONSEQUENCE", ""))
            text  = f"{hgvsp} {csq}"

            if re.search(rule["pattern"], text, re.IGNORECASE):
                match = rule.copy()
                match["hgvsp"] = hgvsp
                match["vaf"]   = float(row.get("VAF_TUMOR", 0))
                matches.append(match)
                seen_genes.add(gene)
                break

    return matches


def find_tumor_suppressors(variants: pd.DataFrame) -> list:
    """Flag high-impact tumor suppressor mutations."""
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
        })
    return results


def generate_recommendations(
    case_id: str,
    drivers: list,
    ts_genes: list,
    tmb: float,
    tme: dict,
) -> list:
    """
    Apply integration rules to generate ranked treatment recommendations.
    Returns list of recommendation dicts.
    """
    recs = []
    tme_phenotype = tme.get("tme_phenotype", "Unknown")
    tmb_high = (not np.isnan(tmb)) and (tmb >= TMB_HIGH_THRESHOLD)

    # ── Priority 1: Targeted therapy for driver mutations ─────────────────────
    for driver in drivers:
        for drug in driver["drugs"]:
            recs.append({
                "rank":        len(recs) + 1,
                "category":    "Targeted Therapy",
                "drug":        drug,
                "drug_class":  driver["drug_class"],
                "evidence":    driver["evidence"],
                "rationale":   f"{driver['mutation_label']} detected ({driver['hgvsp']})",
                "priority":    1,
            })

    # ── Priority 2: Immunotherapy based on TME + TMB ──────────────────────────
    if tme_phenotype == "Inflamed":
        io_rationale = "Inflamed TME (TIL-rich)"
        if tmb_high:
            io_rationale += f" + High TMB ({tmb:.1f} mut/Mb)"
        recs.append({
            "rank":       len(recs) + 1,
            "category":   "Immunotherapy",
            "drug":       "Pembrolizumab",
            "drug_class": "PD-1 inhibitor",
            "evidence":   "FDA-Approved",
            "rationale":  io_rationale,
            "priority":   2,
        })
        # If also has KRAS G12C → combination
        if any(d["gene"] == "KRAS" for d in drivers):
            recs.append({
                "rank":       len(recs) + 1,
                "category":   "Combination",
                "drug":       "Sotorasib + Pembrolizumab",
                "drug_class": "KRAS G12C inhibitor + PD-1 inhibitor",
                "evidence":   "Clinical Trial",
                "rationale":  "KRAS G12C mutation + Inflamed TME",
                "priority":   2,
            })

    elif tme_phenotype == "Excluded":
        recs.append({
            "rank":       len(recs) + 1,
            "category":   "Immunotherapy",
            "drug":       "Pembrolizumab ± chemotherapy",
            "drug_class": "PD-1 inhibitor",
            "evidence":   "FDA-Approved",
            "rationale":  "Excluded TME — consider combination to overcome exclusion",
            "priority":   2,
        })

    elif tme_phenotype == "Desert" and tmb_high:
        recs.append({
            "rank":       len(recs) + 1,
            "category":   "Immunotherapy",
            "drug":       "Pembrolizumab",
            "drug_class": "PD-1 inhibitor",
            "evidence":   "FDA-Approved (TMB-H)",
            "rationale":  f"High TMB ({tmb:.1f} mut/Mb) despite Desert TME",
            "priority":   2,
        })

    # ── Priority 3: Fallback chemotherapy ────────────────────────────────────
    chemo_rationale = "No targetable driver detected"
    if tme_phenotype == "Desert":
        chemo_rationale += "; Desert TME (immunotherapy less likely effective)"
    if ts_genes:
        ts_str = ", ".join(f"{t['gene']} {t['hgvsp']}" for t in ts_genes)
        chemo_rationale += f"; Tumor suppressor alterations: {ts_str}"

    recs.append({
        "rank":       len(recs) + 1,
        "category":   "Chemotherapy",
        "drug":       "Carboplatin + Pemetrexed",
        "drug_class": "Platinum-based doublet",
        "evidence":   "Standard of Care",
        "rationale":  chemo_rationale,
        "priority":   3,
    })

    # Renumber ranks
    for i, r in enumerate(recs, 1):
        r["rank"] = i

    return recs


# ── Visualization ─────────────────────────────────────────────────────────────

CATEGORY_COLORS = {
    "Targeted Therapy": "#2166ac",
    "Immunotherapy":    "#1a9850",
    "Combination":      "#7b2d8b",
    "Chemotherapy":     "#d73027",
}

TME_COLORS = {
    "Inflamed": "#2ca02c",
    "Excluded": "#ff7f0e",
    "Desert":   "#d62728",
    "Unknown":  "#999999",
}


def make_integration_report(
    case_id: str,
    drivers: list,
    ts_genes: list,
    tmb: float,
    tme: dict,
    recs: list,
    out_path: Path,
):
    """Generate integration summary figure (2-panel)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor("#fafafa")
    fig.suptitle(
        f"Multi-Omics Integration Report — {case_id}",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # ── Left panel: Molecular profile summary ────────────────────────────────
    ax = axes[0]
    ax.axis("off")
    ax.set_title("Molecular Profile", fontsize=12, fontweight="bold", pad=10)

    tme_phenotype = tme.get("tme_phenotype", "Unknown")
    tme_color = TME_COLORS.get(tme_phenotype, "#999999")
    tmb_str = f"{tmb:.1f} mut/Mb" if not np.isnan(tmb) else "N/A"
    tmb_flag = " [HIGH]" if (not np.isnan(tmb)) and tmb >= TMB_HIGH_THRESHOLD else ""

    # Build profile text blocks
    y = 0.92
    def add_section(title, items, color="#333333"):
        nonlocal y
        ax.text(0.05, y, title, transform=ax.transAxes,
                fontsize=10, fontweight="bold", color=color)
        y -= 0.06
        for item in items:
            ax.text(0.10, y, f"• {item}", transform=ax.transAxes,
                    fontsize=9, color="#555555")
            y -= 0.055
        y -= 0.02

    # Genomic section
    driver_items = [f"{d['mutation_label']}  ({d['hgvsp']})" for d in drivers] or ["No targetable driver detected"]
    add_section("Targetable Drivers (M02)", driver_items, "#2166ac")

    ts_items = [f"{t['gene']} {t['hgvsp']} [{t['impact']}]" for t in ts_genes] or ["None detected"]
    add_section("Tumor Suppressors (M02)", ts_items, "#8c510a")

    add_section("Tumor Mutational Burden (M02)", [f"TMB = {tmb_str}{tmb_flag}"], "#555555")

    add_section(
        "TME Phenotype (M08)",
        [f"{tme_phenotype}  (TIL score: {tme.get('til_score', 0):.2f})"],
        tme_color,
    )

    # TME badge
    badge_x, badge_y = 0.72, 0.12
    ax.add_patch(mpatches.FancyBboxPatch(
        (badge_x - 0.01, badge_y - 0.04), 0.28, 0.10,
        boxstyle="round,pad=0.02", linewidth=1.5,
        edgecolor=tme_color, facecolor=tme_color + "22",
        transform=ax.transAxes, clip_on=False,
    ))
    ax.text(badge_x + 0.13, badge_y + 0.01, tme_phenotype,
            transform=ax.transAxes, fontsize=13, fontweight="bold",
            color=tme_color, ha="center", va="center")

    # ── Right panel: Treatment recommendations ───────────────────────────────
    ax2 = axes[1]
    ax2.axis("off")
    ax2.set_title("Treatment Recommendations", fontsize=12, fontweight="bold", pad=10)

    if not recs:
        ax2.text(0.5, 0.5, "No recommendations generated",
                 transform=ax2.transAxes, ha="center", va="center",
                 fontsize=11, color="#888888")
    else:
        y2 = 0.93
        for rec in recs:
            cat   = rec["category"]
            color = CATEGORY_COLORS.get(cat, "#555555")

            # Category badge
            ax2.add_patch(mpatches.FancyBboxPatch(
                (0.02, y2 - 0.025), 0.18, 0.05,
                boxstyle="round,pad=0.01",
                facecolor=color + "33", edgecolor=color,
                transform=ax2.transAxes, clip_on=False, linewidth=1.2,
            ))
            ax2.text(0.11, y2, cat, transform=ax2.transAxes,
                     fontsize=8, fontweight="bold", color=color,
                     ha="center", va="center")

            # Drug name
            ax2.text(0.23, y2, rec["drug"],
                     transform=ax2.transAxes, fontsize=10,
                     fontweight="bold", color="#222222", va="center")

            # Drug class + evidence
            ax2.text(0.23, y2 - 0.035,
                     f"{rec['drug_class']}  |  {rec['evidence']}",
                     transform=ax2.transAxes, fontsize=8,
                     color="#777777", va="center")

            # Rationale
            rationale = rec["rationale"]
            if len(rationale) > 60:
                rationale = rationale[:58] + "…"
            ax2.text(0.23, y2 - 0.065,
                     f"↳ {rationale}",
                     transform=ax2.transAxes, fontsize=7.5,
                     color="#999999", va="center", style="italic")

            # Separator line
            ax2.axhline(y=y2 - 0.085, xmin=0.02, xmax=0.98,
                        color="#eeeeee", linewidth=0.8)

            y2 -= 0.14
            if y2 < 0.05:
                break

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def run_patient(case_id: str, logger) -> dict | None:
    """Run full integration pipeline for one patient."""
    logger.info(f"\n[{case_id}]")

    variants  = load_variants(case_id)
    tmb       = load_tmb(case_id)
    tme       = load_tme(case_id)

    if variants.empty:
        logger.warning(f"  No variant data — skipping")
        return None

    drivers  = find_driver_mutations(variants)
    ts_genes = find_tumor_suppressors(variants)
    recs     = generate_recommendations(case_id, drivers, ts_genes, tmb, tme)

    logger.info(f"  Drivers: {[d['mutation_label'] for d in drivers] or 'None'}")
    logger.info(f"  TMB: {tmb:.1f} mut/Mb" if not np.isnan(tmb) else "  TMB: N/A")
    logger.info(f"  TME: {tme.get('tme_phenotype', 'N/A')}")
    logger.info(f"  Recommendations: {len(recs)}")

    # Save outputs
    case_out = OUT_DIR / case_id
    case_out.mkdir(parents=True, exist_ok=True)

    rec_path    = case_out / f"{case_id}_recommendation.tsv"
    report_path = case_out / f"{case_id}_integration_report.png"

    pd.DataFrame(recs).to_csv(rec_path, sep="\t", index=False)
    make_integration_report(case_id, drivers, ts_genes, tmb, tme, recs, report_path)
    logger.info(f"  Report → {report_path}")

    # Summary row
    return {
        "sample_id":        case_id,
        "n_drivers":        len(drivers),
        "driver_genes":     "; ".join(d["gene"] for d in drivers) or "None",
        "driver_mutations": "; ".join(d["mutation_label"] for d in drivers) or "None",
        "ts_alterations":   "; ".join(t["gene"] for t in ts_genes) or "None",
        "tmb":              round(tmb, 2) if not np.isnan(tmb) else None,
        "tmb_high":         bool((not np.isnan(tmb)) and tmb >= TMB_HIGH_THRESHOLD),
        "tme_phenotype":    tme.get("tme_phenotype", "Unknown"),
        "til_score":        tme.get("til_score", None),
        "top_recommendation": recs[0]["drug"] if recs else "None",
        "top_category":     recs[0]["category"] if recs else "None",
        "n_recommendations": len(recs),
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
    parser = argparse.ArgumentParser(description="LUAD multi-omics integration (M09)")
    parser.add_argument("--sample",  type=str,  help="Single patient ID")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    logger = get_logger("luad-integration")
    logger.info(f"\n{'='*55}")
    logger.info("Module 09: Multi-Omics Integration")

    if args.sample:
        patients = [args.sample]
    else:
        patients = discover_patients()

    logger.info(f"Patients to process: {len(patients)}")

    if args.dry_run:
        logger.info(f"[DRY RUN] Would process: {patients[:5]}...")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    success = failed = 0

    for case_id in patients:
        try:
            row = run_patient(case_id, logger)
            if row:
                all_rows.append(row)
                success += 1
        except Exception as e:
            logger.error(f"  [{case_id}] Failed: {e}")
            failed += 1

    # Cohort summary
    if all_rows:
        summary_path = OUT_DIR / "integration_summary.tsv"
        df = pd.DataFrame(all_rows)
        df.to_csv(summary_path, sep="\t", index=False)
        logger.info(f"\nCohort summary → {summary_path}")

        # Print distribution
        logger.info("\nTop recommendation distribution:")
        for cat, cnt in df["top_category"].value_counts().items():
            logger.info(f"  {cat}: {cnt} ({cnt/len(df):.1%})")

        logger.info("\nTME phenotype distribution:")
        for phe, cnt in df["tme_phenotype"].value_counts().items():
            logger.info(f"  {phe}: {cnt} ({cnt/len(df):.1%})")

    logger.info(f"\n[Module 09 Complete] Success: {success}  Failed: {failed}")


if __name__ == "__main__":
    main()
