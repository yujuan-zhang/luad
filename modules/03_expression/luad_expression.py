#!/usr/bin/env python
"""
luad_expression.py
------------------
LUAD bulk RNA-seq expression analysis using pcgr Python API directly.

Steps:
  1. Parse RNA-seq TPM files via pcgr.expression.parse_expression()
  2. Correlate sample expression with TCGA-LUAD reference cohort
     via pcgr.expression.correlate_sample_expression()
  3. Find expression outlier genes via pcgr.expression.find_expression_outliers()
  4. Integrate variant + expression data (requires module 02 output)
     via pcgr.expression.integrate_variant_expression()
  5. Save results: correlation table + outlier genes + integrated table

Input:
  data/rnaseq/{case_id}.tsv.gz          — TargetID + TPM columns
  data/output/02_variation_annotation/  — variant TSV from module 02

Output:
  data/output/03_expression/{case_id}/
    {case_id}_expression_outliers.tsv   — top over/under-expressed genes
    {case_id}_expr_correlation.tsv      — similarity to TCGA subtypes
    {case_id}_variant_expression.tsv    — variants with expression integrated

Usage:
  python luad_expression.py                        # all samples in data/rnaseq/
  python luad_expression.py --sample TCGA-86-A4D0  # single sample
  python luad_expression.py --dry_run              # validate inputs only
"""

import argparse
import logging
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# pcgr Python API
from pcgr.expression import (
    parse_expression,
    correlate_sample_expression,
    find_expression_outliers,
    integrate_variant_expression,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
PROJECT_DIR  = SCRIPT_DIR.parent.parent
RNASEQ_DIR   = PROJECT_DIR / "data/rnaseq"
VARIANT_DIR  = PROJECT_DIR / "data/output/02_variation_annotation"
OUT_DIR      = PROJECT_DIR / "data/output/03_expression"
REFDATA_DIR  = Path.home() / "pcgr_refdata/data/grch38"

# ── Parameters ────────────────────────────────────────────────────────────────
GENOME_ASSEMBLY = "grch38"
EXPRESSION_SIMILARITY_DB = "tcga"   # compare against TCGA cohort


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
    return logging.getLogger(name)


def get_rnaseq_files(sample_id: str = None):
    """Return RNA-seq TSV files."""
    if sample_id:
        return [RNASEQ_DIR / f"{sample_id}.tsv.gz"]
    return sorted(RNASEQ_DIR.glob("*.tsv.gz"))


def load_variant_table(sample_id: str) -> pd.DataFrame:
    """Load variant table from module 02 output if available."""
    variant_path = VARIANT_DIR / sample_id / f"{sample_id}_variants.tsv.gz"
    if variant_path.exists():
        return pd.read_csv(variant_path, sep="\t", compression="gzip", low_memory=False)
    return None


# ── Visualization ─────────────────────────────────────────────────────────────

# Genes of known importance in LUAD — highlighted in expression plots
LUAD_GENES = {
    "EGFR", "KRAS", "TP53", "STK11", "KEAP1", "BRAF", "MET", "RET",
    "ALK", "ROS1", "ERBB2", "NKX2-1", "TTF1", "CD274", "PDCD1LG2",
    "MYC", "CDKN2A", "RB1", "PTEN", "PIK3CA", "MDM2",
}


def _plot_expression_summary(sample_id: str, out_dir: Path, logger):
    """
    4-panel expression summary figure (pcgr-inspired):
      [0,0] Outlier volcano: Z-score vs Percentile (TCGA background)
      [0,1] Top over-expressed genes (bar)
      [1,0] Top under-expressed genes (bar)
      [1,1] TCGA correlation distribution (histogram of Spearman r)
    """
    outlier_path = out_dir / f"{sample_id}_expression_outliers.tsv"
    corr_path    = out_dir / f"{sample_id}_expr_correlation.tsv"

    if not outlier_path.exists() and not corr_path.exists():
        logger.warning("  No outlier/correlation files found, skipping visualization")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#f8f9fa")
    fig.suptitle(f"Expression Summary — {sample_id}", fontsize=14,
                 fontweight="bold", y=1.01)

    # ── Load outliers ─────────────────────────────────────────────────────────
    outliers = pd.DataFrame()
    if outlier_path.exists():
        outliers = pd.read_csv(outlier_path, sep="\t")
        # Outlier file has ENSEMBL_GENE_ID but no SYMBOL — join from gene_expression
        expr_path = out_dir / f"{sample_id}_gene_expression.tsv.gz"
        if expr_path.exists() and "SYMBOL" not in outliers.columns:
            expr_df = pd.read_csv(expr_path, sep="\t", compression="gzip",
                                  usecols=["ENSEMBL_GENE_ID", "SYMBOL", "BIOTYPE"])
            outliers = outliers.merge(expr_df, on="ENSEMBL_GENE_ID", how="left")
        # Keep protein-coding only
        if "BIOTYPE" in outliers.columns:
            outliers = outliers[outliers["BIOTYPE"] == "protein_coding"]

    # ── Panel [0,0]: Outlier scatter (Z-score vs TPM_LOG2) ───────────────────
    ax = axes[0, 0]
    if not outliers.empty and {"Z_SCORE", "TPM_LOG2_GENE", "PERCENTILE"}.issubset(outliers.columns):
        z  = outliers["Z_SCORE"].clip(-10, 10)
        tpm = outliers["TPM_LOG2_GENE"]

        # Color by expression direction
        colors = np.where(z > 2, "#d62728",
                 np.where(z < -2, "#4393c3", "#bbbbbb"))

        ax.scatter(tpm, z, c=colors, s=6, alpha=0.5, linewidths=0)

        # Highlight LUAD cancer genes
        if "SYMBOL" in outliers.columns:
            luad_mask = outliers["SYMBOL"].isin(LUAD_GENES)
            luad_df   = outliers[luad_mask]
            ax.scatter(luad_df["TPM_LOG2_GENE"], luad_df["Z_SCORE"].clip(-10, 10),
                       c="#ff7f0e", s=60, zorder=5, edgecolors="white", linewidths=0.5)
            for _, row in luad_df.iterrows():
                ax.annotate(row["SYMBOL"],
                            xy=(row["TPM_LOG2_GENE"], np.clip(row["Z_SCORE"], -10, 10)),
                            fontsize=7, color="#ff7f0e",
                            xytext=(3, 3), textcoords="offset points")

        ax.axhline(2, color="#d62728", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.axhline(-2, color="#4393c3", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("log₂(TPM+1)", fontsize=9)
        ax.set_ylabel("Z-score vs TCGA-LUAD", fontsize=9)
        ax.set_title("Expression Outliers\n(orange = LUAD driver genes)", fontsize=10, fontweight="bold")

        n_up   = (z > 2).sum()
        n_down = (z < -2).sum()
        ax.text(0.02, 0.97, f"↑ {n_up} over-expressed",
                transform=ax.transAxes, fontsize=8, color="#d62728", va="top")
        ax.text(0.02, 0.91, f"↓ {n_down} under-expressed",
                transform=ax.transAxes, fontsize=8, color="#4393c3", va="top")
    else:
        ax.text(0.5, 0.5, "No outlier data", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="gray")
        ax.axis("off")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel [0,1]: Top over-expressed genes ────────────────────────────────
    ax = axes[0, 1]
    if not outliers.empty and "Z_SCORE" in outliers.columns and "SYMBOL" in outliers.columns:
        top_over = (outliers[outliers["Z_SCORE"] > 0]
                    .nlargest(15, "Z_SCORE")[["SYMBOL", "Z_SCORE", "TPM_LOG2_GENE"]])
        if not top_over.empty:
            bar_colors = ["#ff7f0e" if g in LUAD_GENES else "#d62728"
                          for g in top_over["SYMBOL"]]
            ax.barh(top_over["SYMBOL"], top_over["Z_SCORE"],
                    color=bar_colors, alpha=0.85, edgecolor="white")
            ax.set_xlabel("Z-score", fontsize=9)
            ax.set_title("Top Over-expressed Genes\n(orange = LUAD drivers)", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)

    # ── Panel [1,0]: Top under-expressed genes ───────────────────────────────
    ax = axes[1, 0]
    if not outliers.empty and "Z_SCORE" in outliers.columns and "SYMBOL" in outliers.columns:
        top_under = (outliers[outliers["Z_SCORE"] < 0]
                     .nsmallest(15, "Z_SCORE")[["SYMBOL", "Z_SCORE", "TPM_LOG2_GENE"]])
        if not top_under.empty:
            bar_colors = ["#ff7f0e" if g in LUAD_GENES else "#4393c3"
                          for g in top_under["SYMBOL"]]
            ax.barh(top_under["SYMBOL"], top_under["Z_SCORE"],
                    color=bar_colors, alpha=0.85, edgecolor="white")
            ax.set_xlabel("Z-score", fontsize=9)
            ax.set_title("Top Under-expressed Genes\n(orange = LUAD drivers)", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)

    # ── Panel [1,1]: TCGA correlation distribution ───────────────────────────
    ax = axes[1, 1]
    if corr_path.exists():
        corr_df = pd.read_csv(corr_path, sep="\t")
        if "CORR" in corr_df.columns:
            corr_vals = pd.to_numeric(corr_df["CORR"], errors="coerce").dropna()
            ax.hist(corr_vals, bins=40, color="#74c476",
                    edgecolor="white", linewidth=0.5, alpha=0.85)
            ax.axvline(corr_vals.median(), color="#d62728", linestyle="--",
                       linewidth=1.5, label=f"Median r={corr_vals.median():.3f}")

            # Top 3 most similar samples
            top3 = corr_df.nlargest(3, "CORR")
            label_col = "EXT_SAMPLE_ID" if "EXT_SAMPLE_ID" in corr_df.columns else corr_df.columns[1]
            for _, row in top3.iterrows():
                r = float(row["CORR"])
                ax.axvline(r, color="#ff7f0e", linestyle=":", linewidth=1, alpha=0.8)
                ax.text(r, ax.get_ylim()[1] * 0.5,
                        f" {row[label_col]}", fontsize=6, color="#ff7f0e",
                        rotation=90, va="center")

            ax.set_xlabel("Spearman correlation with TCGA-LUAD sample", fontsize=9)
            ax.set_ylabel("Count", fontsize=9)
            ax.set_title(f"TCGA-LUAD Similarity\n(n={len(corr_vals):,} reference samples)",
                         fontsize=10, fontweight="bold")
            ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_png = out_dir / f"{sample_id}_expression_summary.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"  Expression summary figure → {out_png}")


def run_sample(rnaseq_path: Path, dry_run: bool = False):
    """Process a single sample's RNA-seq data."""
    logger = get_logger("luad-expression")
    sample_id = rnaseq_path.name.replace(".tsv.gz", "")

    logger.info(f"\n{'='*55}")
    logger.info(f"Sample: {sample_id}")

    if dry_run:
        logger.info(f"[DRY RUN] Would process: {rnaseq_path}")
        return

    out_dir = OUT_DIR / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parse expression data
    logger.info("Step 1: Parsing expression data...")
    expression_map = parse_expression(
        expression_fname_tsv = str(rnaseq_path),
        sample_id            = sample_id,
        refdata_assembly_dir = str(REFDATA_DIR),
        logger               = logger,
    )

    if expression_map is None or expression_map.get("gene") is None:
        logger.warning(f"  Expression parsing failed for {sample_id}, skipping.")
        return None

    gene_expr = expression_map["gene"]
    logger.info(f"  Parsed {len(gene_expr)} genes")

    # Save gene-level expression table
    expr_out = out_dir / f"{sample_id}_gene_expression.tsv.gz"
    gene_expr.to_csv(expr_out, sep="\t", index=False, compression="gzip")
    logger.info(f"  Gene expression saved → {expr_out}")

    # Build sample_expression dict and minimal yaml_data for pcgr API
    sample_expression = {
        "gene":       gene_expr,
        "transcript": expression_map.get("transcript"),
    }

    # Minimal yaml_data structure required by pcgr expression functions
    yaml_data = {
        "sample_id": sample_id,
        "genome_assembly": GENOME_ASSEMBLY,
        "conf": {
            "sample_properties": {
                "site": "Lung"   # PCGR primary site → maps to TCGA-LUAD cohort
            },
            "expression": {
                "similarity_db": {
                    "tcga": {"luad": True}   # compare against TCGA-LUAD cohort
                },
                "outlier_analysis_db": "tcga",
                "expression_outlier_pvalue": 0.01,
            }
        }
    }

    # 2. Correlate with TCGA reference cohort
    logger.info("Step 2: Correlating with TCGA-LUAD reference cohort...")
    try:
        corr_result = correlate_sample_expression(
            sample_expression    = sample_expression,
            yaml_data            = yaml_data,
            refdata_assembly_dir = str(REFDATA_DIR),
            protein_coding_only  = True,
            logger               = logger,
        )
        if corr_result and "tcga" in corr_result:
            corr_df = corr_result["tcga"]
            if not corr_df.empty:
                corr_out = out_dir / f"{sample_id}_expr_correlation.tsv"
                corr_df.to_csv(corr_out, sep="\t", index=False)
                logger.info(f"  Correlation saved ({len(corr_df)} samples) → {corr_out}")
    except Exception as e:
        logger.warning(f"  Correlation step skipped: {e}")

    # 3. Find expression outliers
    logger.info("Step 3: Finding expression outlier genes...")
    try:
        outliers = find_expression_outliers(
            sample_expression    = sample_expression,
            yaml_data            = yaml_data,
            refdata_assembly_dir = str(REFDATA_DIR),
            protein_coding_only  = True,
            logger               = logger,
        )
        if outliers is not None and isinstance(outliers, pd.DataFrame) and not outliers.empty:
            outlier_out = out_dir / f"{sample_id}_expression_outliers.tsv"
            outliers.to_csv(outlier_out, sep="\t", index=False)
            logger.info(f"  Outliers saved ({len(outliers)} genes) → {outlier_out}")
    except Exception as e:
        logger.warning(f"  Outlier detection skipped: {e}")

    # 4. Integrate variant + expression (requires module 02 output)
    logger.info("Step 4: Integrating variants with expression...")
    variant_df = load_variant_table(sample_id)
    if variant_df is not None:
        try:
            integrated = integrate_variant_expression(
                variant_set   = variant_df,
                expression_data = sample_expression,
                logger        = logger,
            )
            if integrated is not None and not integrated.empty:
                integ_out = out_dir / f"{sample_id}_variant_expression.tsv.gz"
                integrated.to_csv(integ_out, sep="\t", index=False, compression="gzip")
                logger.info(f"  Integrated table saved → {integ_out}")
        except Exception as e:
            logger.warning(f"  Variant-expression integration skipped: {e}")
    else:
        logger.info(f"  No variant table found for {sample_id}, skipping integration")

    # 5. Visualization
    logger.info("Step 5: Generating expression summary figure...")
    out_dir_path = out_dir if isinstance(out_dir, Path) else Path(out_dir)
    _plot_expression_summary(sample_id, out_dir_path, logger)

    return {"sample_id": sample_id, "n_genes": len(gene_expr)}


def main():
    parser = argparse.ArgumentParser(description="LUAD expression analysis (pcgr Python API)")
    parser.add_argument("--sample",  type=str, help="Single sample ID (e.g. TCGA-86-A4D0)")
    parser.add_argument("--dry_run", action="store_true", help="Validate inputs without running")
    args = parser.parse_args()

    if not REFDATA_DIR.exists():
        print(f"[ERROR] PCGR refdata not found: {REFDATA_DIR}")
        return

    rnaseq_files = get_rnaseq_files(args.sample)
    if not rnaseq_files:
        print(f"[ERROR] No RNA-seq files found in {RNASEQ_DIR}")
        return

    print(f"[INFO] Found {len(rnaseq_files)} sample(s) to process")
    results = []
    for f in rnaseq_files:
        r = run_sample(f, dry_run=args.dry_run)
        if r:
            results.append(r)

    if results:
        summary = pd.DataFrame(results)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        summary_path = OUT_DIR / "all_samples_summary.tsv"
        summary.to_csv(summary_path, sep="\t", index=False)
        print(f"\n[Summary] {len(results)} samples → {summary_path}")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
