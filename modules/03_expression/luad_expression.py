#!/usr/bin/env python
"""
luad_expression.py
------------------
LUAD bulk RNA-seq expression analysis using pcgr Python API + GTEx baseline.

Steps:
  1. Parse RNA-seq TPM files via pcgr.expression.parse_expression()
  2. Compute per-gene Z-scores vs GTEx normal lung tissue baseline
     (scientifically correct: tumor vs normal, not tumor vs tumor median)
  3. Correlate sample expression with TCGA-LUAD reference cohort
     via pcgr.expression.correlate_sample_expression()
  4. Integrate variant + expression data (requires module 02 output)
     via pcgr.expression.integrate_variant_expression()
  5. Save results: GTEx-normalised outlier genes + correlation + integrated table

GTEx Reference:
  data/databases/gtex_lung_normal_tpm_stats.tsv.gz
  Build with: python data/scripts/build_gtex_reference.py
  ~570 normal lung samples from GTEx v8, processed by UCSC Toil pipeline
  (same pipeline as TCGA — directly comparable without extra normalisation).

Input:
  data/rnaseq/{case_id}.tsv.gz          — TargetID + TPM columns
  data/output/02_variants/  — variant TSV from module 02

Output:
  data/output/03_expression/{case_id}/
    {case_id}_expression_outliers.tsv   — outlier genes (Z vs GTEx normal lung)
    {case_id}_gene_expression.tsv.gz    — all genes with GTEx Z-scores
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
VARIANT_DIR  = PROJECT_DIR / "data/output/02_variants"
OUT_DIR      = PROJECT_DIR / "data/output/03_expression"
REFDATA_DIR  = Path.home() / "pcgr_refdata/data/grch38"
GTEX_REF_PATH = PROJECT_DIR / "data/databases/gtex_lung_normal_tpm_stats.tsv.gz"

# ── Parameters ────────────────────────────────────────────────────────────────
GENOME_ASSEMBLY = "grch38"
EXPRESSION_SIMILARITY_DB = "tcga"   # compare against TCGA cohort (for subtype similarity)
GTEX_ZSCORE_THRESHOLD = 2.0         # |Z| > 2 → outlier vs normal lung


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
    return logging.getLogger(name)


# ── GTEx Reference Baseline ───────────────────────────────────────────────────

_gtex_ref_cache: pd.DataFrame = None   # module-level cache, loaded once


def load_gtex_reference(logger=None) -> pd.DataFrame | None:
    """
    Load pre-computed GTEx normal lung reference stats.
    Returns None if reference file not found (with a warning).
    """
    global _gtex_ref_cache
    if _gtex_ref_cache is not None:
        return _gtex_ref_cache

    if not GTEX_REF_PATH.exists():
        msg = (
            f"GTEx reference not found: {GTEX_REF_PATH}\n"
            f"  Run: python data/scripts/build_gtex_reference.py\n"
            f"  Falling back to TCGA-cohort-median Z-scores (less accurate)."
        )
        if logger:
            logger.warning(msg)
        else:
            print(f"[WARNING] {msg}")
        return None

    ref = pd.read_csv(GTEX_REF_PATH, sep="\t", compression="gzip")
    # Ensure ENSEMBL_GENE_ID is string, strip any version suffixes
    ref["ENSEMBL_GENE_ID"] = ref["ENSEMBL_GENE_ID"].astype(str).str.split(".").str[0]
    _gtex_ref_cache = ref
    if logger:
        logger.info(f"  GTEx reference loaded: {len(ref):,} genes "
                    f"(n={ref['gtex_n_samples'].iloc[0]} normal lung samples)")
    return ref


def compute_gtex_zscores(gene_expr: pd.DataFrame,
                          gtex_ref: pd.DataFrame) -> pd.DataFrame:
    """
    Merge gene expression with GTEx reference stats and compute Z-scores.

    Z = (patient_log2tpm − gtex_mean) / gtex_std

    Positive Z: gene is OVER-expressed relative to normal lung
    Negative Z: gene is UNDER-expressed relative to normal lung

    Returns gene_expr with added columns:
      GTEX_ZSCORE, GTEX_PERCENTILE, GTEX_DIRECTION
    """
    # Ensure ID column is clean (no version numbers)
    expr = gene_expr.copy()
    expr["ENSEMBL_GENE_ID"] = (
        expr["ENSEMBL_GENE_ID"].astype(str).str.split(".").str[0]
    )

    merged = expr.merge(
        gtex_ref[["ENSEMBL_GENE_ID", "gtex_mean_log2tpm",
                  "gtex_std_log2tpm", "gtex_median_log2tpm", "gtex_n_samples"]],
        on="ENSEMBL_GENE_ID",
        how="left",
    )

    # Compute Z-score; guard against zero std
    std = merged["gtex_std_log2tpm"].replace(0, np.nan)
    merged["GTEX_ZSCORE"] = (
        (merged["TPM_LOG2_GENE"] - merged["gtex_mean_log2tpm"]) / std
    ).round(3)

    # Rank-based percentile within this sample (relative to GTEx baseline direction)
    valid = merged["GTEX_ZSCORE"].notna()
    merged.loc[valid, "GTEX_PERCENTILE"] = (
        merged.loc[valid, "GTEX_ZSCORE"].rank(pct=True) * 100
    ).round(1)

    merged["GTEX_DIRECTION"] = np.where(
        merged["GTEX_ZSCORE"] > GTEX_ZSCORE_THRESHOLD, "OVER",
        np.where(merged["GTEX_ZSCORE"] < -GTEX_ZSCORE_THRESHOLD, "UNDER", "NORMAL")
    )

    return merged


def _plot_clinical_gene_panel(sample_id: str, out_dir: Path, scored: pd.DataFrame,
                               logger, gtex_n: int = 287):
    """
    Lollipop chart: expression of LUAD clinical genes vs GTEx normal lung.
    Each gene shows the patient's Z-score as a dot; shaded band = normal range (±2 SD).
    """
    if scored is None or "SYMBOL" not in scored.columns:
        return

    panel = scored[scored["SYMBOL"].isin(CLINICAL_PANEL_GENES)].copy()
    if panel.empty:
        logger.warning("  No clinical panel genes found — skipping clinical gene panel")
        return

    panel = panel.drop_duplicates("SYMBOL").set_index("SYMBOL")
    # Reindex to fixed panel order, keeping only genes present
    genes_present = [g for g in CLINICAL_PANEL_GENES if g in panel.index]
    panel = panel.loc[genes_present]
    # Sort by Z-score so most extreme are at top/bottom
    panel = panel.sort_values("GTEX_ZSCORE")

    n = len(panel)
    fig, ax = plt.subplots(figsize=(11, max(5, n * 0.42 + 1.5)))
    fig.patch.set_facecolor("#f8f9fa")

    y_pos = np.arange(n)

    # Normal range band
    ax.axvspan(-2, 2, alpha=0.07, color="#888888")
    ax.axvline(0, color="#888888", linewidth=1.0, alpha=0.5, linestyle="-")
    ax.axvline(-2, color="#4393c3", linewidth=0.8, linestyle="--", alpha=0.55)
    ax.axvline( 2, color="#d62728", linewidth=0.8, linestyle="--", alpha=0.55)

    for i, (gene, row) in enumerate(panel.iterrows()):
        z   = float(np.clip(row.get("GTEX_ZSCORE", 0), -15, 15))
        tpm = row.get("TPM_LOG2_GENE", np.nan)
        direction = row.get("GTEX_DIRECTION", "NORMAL")
        color = "#d62728" if direction == "OVER" else \
                "#4393c3" if direction == "UNDER" else "#888888"

        ax.plot([0, z], [i, i], color=color, linewidth=1.8, alpha=0.55, solid_capstyle="round")
        ax.scatter(z, i, color=color, s=90, zorder=5,
                   edgecolors="white", linewidths=0.6)

        # Annotation: Z-score + patient TPM
        if pd.notna(tpm):
            tpm_linear = max(0, 2 ** float(tpm) - 0.001)
            label = f"Z={z:+.1f}  (TPM {tpm_linear:.1f})"
        else:
            label = f"Z={z:+.1f}"
        x_offset = 0.25 if z >= 0 else -0.25
        ha = "left" if z >= 0 else "right"
        ax.text(z + x_offset, i, label, va="center", ha=ha,
                fontsize=7.5, color=color)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(panel.index.tolist(), fontsize=9, fontweight="bold")
    ax.set_xlabel("GTEx Z-score  (tumor vs normal lung)", fontsize=10)
    ax.set_title(f"Clinical Gene Panel — {sample_id}\n"
                 f"Expression vs GTEx Normal Lung (n={gtex_n} samples)",
                 fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d62728", label="Over-expressed  (Z > +2)"),
        Patch(facecolor="#4393c3", label="Under-expressed  (Z < −2)"),
        Patch(facecolor="#888888", alpha=0.25, label="Normal range  (−2 to +2)"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right",
              framealpha=0.8, edgecolor="#cccccc")

    plt.tight_layout()
    out_png = out_dir / f"{sample_id}_clinical_genes.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"  Clinical gene panel → {out_png}")


def _plot_cohort_heatmap(clinical_data: dict, out_dir: Path, logger):
    """
    Cohort-level heatmap: rows = patients, cols = clinical panel genes.
    Color = GTEx Z-score (red over-expressed, blue under-expressed).
    """
    if not clinical_data:
        return

    df = pd.DataFrame(clinical_data).T   # samples × genes
    genes_present = [g for g in CLINICAL_PANEL_GENES if g in df.columns]
    df = df[genes_present].astype(float)

    # Sort patients: descending by mean Z-score across panel
    df = df.loc[df.mean(axis=1).sort_values(ascending=False).index]

    n_samples, n_genes = df.shape
    fig_h = max(8, min(n_samples * 0.08 + 3, 28))
    fig, ax = plt.subplots(figsize=(n_genes * 0.65 + 2.5, fig_h))
    fig.patch.set_facecolor("#f8f9fa")

    cmap = plt.cm.RdBu_r
    im = ax.imshow(df.clip(-6, 6).values, aspect="auto",
                   cmap=cmap, vmin=-6, vmax=6, interpolation="nearest")

    ax.set_xticks(range(n_genes))
    ax.set_xticklabels(genes_present, rotation=45, ha="right", fontsize=9, fontweight="bold")
    ax.set_yticks([])
    ax.set_ylabel(f"Patients  (n={n_samples})", fontsize=10)
    ax.set_title("LUAD Clinical Gene Expression vs GTEx Normal Lung — Full Cohort\n"
                 "(Z-score: red = over-expressed, blue = under-expressed; clipped at ±6)",
                 fontsize=11, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.55, pad=0.02)
    cbar.set_label("GTEx Z-score", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    out_png = out_dir / "cohort_clinical_genes_heatmap.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"  Cohort clinical gene heatmap ({n_samples} patients × {n_genes} genes) → {out_png}")


def _plot_mutation_expression_scatter(sample_id: str, out_dir: Path,
                                       all_genes_df: pd.DataFrame,
                                       mutated_genes: set, logger):
    """
    Scatter: GTEx Z-score vs log2TPM for all protein-coding genes.
    Gray = normal, red/blue = outlier, orange = LUAD driver, purple star = mutated gene.
    """
    if all_genes_df is None or all_genes_df.empty:
        return
    if not mutated_genes:
        logger.info("  No mutation data — skipping mutation-expression scatter")
        return

    df = all_genes_df.copy()
    if "BIOTYPE" in df.columns:
        df = df[df["BIOTYPE"] == "protein_coding"]
    if "GTEX_ZSCORE" not in df.columns or "TPM_LOG2_GENE" not in df.columns:
        return

    df = df.dropna(subset=["GTEX_ZSCORE", "TPM_LOG2_GENE"])
    z   = df["GTEX_ZSCORE"].clip(-12, 12)
    tpm = df["TPM_LOG2_GENE"]

    colors = np.where(z > GTEX_ZSCORE_THRESHOLD, "#d62728",
             np.where(z < -GTEX_ZSCORE_THRESHOLD, "#4393c3", "#cccccc"))

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#f8f9fa")
    ax.scatter(tpm, z, c=colors, s=4, alpha=0.3, linewidths=0, rasterized=True)

    has_sym = "SYMBOL" in df.columns
    n_mut_outlier = 0

    if has_sym:
        # LUAD drivers (not mutated)
        luad_only = df[df["SYMBOL"].isin(LUAD_GENES) & ~df["SYMBOL"].isin(mutated_genes)]
        if not luad_only.empty:
            ax.scatter(luad_only["TPM_LOG2_GENE"],
                       luad_only["GTEX_ZSCORE"].clip(-12, 12),
                       c="#ff7f0e", s=55, zorder=5,
                       edgecolors="white", linewidths=0.5, label="LUAD driver (not mutated)")
            for _, row in luad_only.iterrows():
                ax.annotate(row["SYMBOL"],
                            xy=(row["TPM_LOG2_GENE"], np.clip(row["GTEX_ZSCORE"], -12, 12)),
                            fontsize=6.5, color="#ff7f0e",
                            xytext=(3, 3), textcoords="offset points")

        # Mutated genes
        mut_df = df[df["SYMBOL"].isin(mutated_genes)].copy()
        if not mut_df.empty:
            mut_z = mut_df["GTEX_ZSCORE"].clip(-12, 12)
            n_mut_outlier = int((mut_z.abs() > GTEX_ZSCORE_THRESHOLD).sum())
            mut_colors = np.where(mut_z > GTEX_ZSCORE_THRESHOLD, "#d62728",
                         np.where(mut_z < -GTEX_ZSCORE_THRESHOLD, "#4393c3", "#9467bd"))
            ax.scatter(mut_df["TPM_LOG2_GENE"], mut_z,
                       c=mut_colors, s=130, zorder=6, marker="*",
                       edgecolors="white", linewidths=0.4, label="Mutated gene (★)")
            for _, row in mut_df.iterrows():
                z_val = float(np.clip(row["GTEX_ZSCORE"], -12, 12))
                col = "#d62728" if z_val > GTEX_ZSCORE_THRESHOLD else \
                      "#4393c3" if z_val < -GTEX_ZSCORE_THRESHOLD else "#9467bd"
                ax.annotate(row["SYMBOL"],
                            xy=(row["TPM_LOG2_GENE"], z_val),
                            fontsize=8.5, color=col, fontweight="bold",
                            xytext=(4, 4), textcoords="offset points")

    ax.axhline( GTEX_ZSCORE_THRESHOLD, color="#d62728", linestyle="--",
                linewidth=0.8, alpha=0.6)
    ax.axhline(-GTEX_ZSCORE_THRESHOLD, color="#4393c3", linestyle="--",
                linewidth=0.8, alpha=0.6)
    ax.set_xlabel("log₂(TPM+0.001)", fontsize=10)
    ax.set_ylabel("GTEx Z-score  (tumor vs normal lung)", fontsize=10)
    n_mut = len(mut_df) if has_sym else 0
    ax.set_title(f"Mutation–Expression Integration — {sample_id}\n"
                 f"(★ = mutated gene;  {n_mut} mutated genes,  {n_mut_outlier} are expression outliers)",
                 fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.85)

    plt.tight_layout()
    out_png = out_dir / f"{sample_id}_mutation_expression.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"  Mutation-expression scatter → {out_png}")


def _plot_cohort_outlier_distribution(outlier_counts: dict, out_dir: Path, logger):
    """
    Cohort-level 2-panel figure:
      Left:  scatter n_up vs n_down per patient (color = total outliers)
      Right: overlapping histograms of n_up and n_down counts
    """
    if not outlier_counts:
        return

    df = pd.DataFrame(outlier_counts).T.dropna()
    df["n_up"]   = df["n_up"].astype(int)
    df["n_down"] = df["n_down"].astype(int)
    df["total"]  = df["n_up"] + df["n_down"]
    df = df.sort_values("total", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#f8f9fa")
    fig.suptitle(f"Cohort Expression Outlier Distribution vs GTEx Normal Lung  "
                 f"(n={len(df)} patients)",
                 fontsize=13, fontweight="bold")

    # ── Panel 1: scatter n_up vs n_down ──────────────────────────────────────
    ax = axes[0]
    sc = ax.scatter(df["n_up"], df["n_down"],
                    c=df["total"], cmap="YlOrRd",
                    s=18, alpha=0.65, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="Total outlier genes", shrink=0.75)
    # Annotate top-5 most disrupted patients
    for sid in df.head(5).index:
        ax.annotate(sid,
                    xy=(df.loc[sid, "n_up"], df.loc[sid, "n_down"]),
                    fontsize=5.5, color="#d62728",
                    xytext=(3, 3), textcoords="offset points")
    med_up, med_down = int(df["n_up"].median()), int(df["n_down"].median())
    ax.axvline(med_up,   color="#d62728", linestyle="--", linewidth=1, alpha=0.6,
               label=f"Median over = {med_up}")
    ax.axhline(med_down, color="#4393c3", linestyle="--", linewidth=1, alpha=0.6,
               label=f"Median under = {med_down}")
    ax.set_xlabel("Over-expressed genes per patient  (Z > +2)", fontsize=10)
    ax.set_ylabel("Under-expressed genes per patient  (Z < −2)", fontsize=10)
    ax.set_title("Per-patient Outlier Counts", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel 2: histograms ────────────────────────────────────────────────
    ax = axes[1]
    max_val = max(df["n_up"].max(), df["n_down"].max())
    bins = np.linspace(0, max_val, 45)
    ax.hist(df["n_up"],   bins=bins, alpha=0.65, color="#d62728",
            label=f"Over-expressed  (median={med_up})")
    ax.hist(df["n_down"], bins=bins, alpha=0.65, color="#4393c3",
            label=f"Under-expressed  (median={med_down})")
    ax.set_xlabel("Number of outlier genes per patient", fontsize=10)
    ax.set_ylabel("Number of patients", fontsize=10)
    ax.set_title("Distribution of Outlier Gene Counts", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_png = out_dir / "cohort_outlier_distribution.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"  Cohort outlier distribution ({len(df)} patients) → {out_png}")


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

# Clinical gene panel — shown in per-patient lollipop chart and cohort heatmap
CLINICAL_PANEL_GENES = [
    # Targetable drivers
    "EGFR", "KRAS", "BRAF", "MET", "ERBB2", "ALK", "ROS1", "RET", "NTRK1", "NTRK3",
    # Immune checkpoint
    "CD274",
    # Tumor suppressors
    "TP53", "STK11", "KEAP1", "CDKN2A", "RB1", "PTEN",
    # Oncogenes / lineage marker
    "MYC", "PIK3CA", "MDM2", "NKX2-1",
]


def _plot_expression_summary(sample_id: str, out_dir: Path, logger,
                              use_gtex: bool = True):
    """
    4-panel expression summary figure.
      [0,0] Outlier scatter: Z-score (vs GTEx normal lung) vs log2 TPM
      [0,1] Top over-expressed genes vs normal lung (bar)
      [1,0] Top under-expressed genes vs normal lung (bar)
      [1,1] TCGA-LUAD similarity distribution (Spearman r histogram)
    """
    outlier_path = out_dir / f"{sample_id}_expression_outliers.tsv"
    corr_path    = out_dir / f"{sample_id}_expr_correlation.tsv"

    if not outlier_path.exists() and not corr_path.exists():
        logger.warning("  No outlier/correlation files found, skipping visualization")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#f8f9fa")

    baseline_label = "GTEx Normal Lung" if use_gtex else "TCGA-LUAD Cohort"
    fig.suptitle(f"Expression Summary — {sample_id}\n"
                 f"(Z-score reference: {baseline_label})",
                 fontsize=13, fontweight="bold", y=1.01)

    # ── Load outliers ─────────────────────────────────────────────────────────
    outliers = pd.DataFrame()
    if outlier_path.exists():
        outliers = pd.read_csv(outlier_path, sep="\t")
        # Keep protein-coding only
        if "BIOTYPE" in outliers.columns:
            outliers = outliers[outliers["BIOTYPE"] == "protein_coding"]

    # Determine which Z-score column to use
    zscore_col = "GTEX_ZSCORE" if (use_gtex and "GTEX_ZSCORE" in outliers.columns) else "Z_SCORE"
    has_zscore = not outliers.empty and zscore_col in outliers.columns

    # ── Panel [0,0]: Outlier scatter (Z-score vs log2 TPM) ───────────────────
    ax = axes[0, 0]
    if has_zscore and "TPM_LOG2_GENE" in outliers.columns:
        z   = outliers[zscore_col].clip(-10, 10)
        tpm = outliers["TPM_LOG2_GENE"]

        colors = np.where(z > GTEX_ZSCORE_THRESHOLD, "#d62728",
                 np.where(z < -GTEX_ZSCORE_THRESHOLD, "#4393c3", "#cccccc"))
        ax.scatter(tpm, z, c=colors, s=5, alpha=0.4, linewidths=0)

        # Highlight LUAD driver genes
        if "SYMBOL" in outliers.columns:
            luad_mask = outliers["SYMBOL"].isin(LUAD_GENES)
            luad_df   = outliers[luad_mask]
            ax.scatter(luad_df["TPM_LOG2_GENE"],
                       luad_df[zscore_col].clip(-10, 10),
                       c="#ff7f0e", s=70, zorder=5,
                       edgecolors="white", linewidths=0.6)
            for _, row in luad_df.iterrows():
                ax.annotate(
                    row["SYMBOL"],
                    xy=(row["TPM_LOG2_GENE"], np.clip(row[zscore_col], -10, 10)),
                    fontsize=7, color="#ff7f0e",
                    xytext=(3, 3), textcoords="offset points",
                )

        ax.axhline(GTEX_ZSCORE_THRESHOLD,  color="#d62728", linestyle="--",
                   linewidth=0.8, alpha=0.7)
        ax.axhline(-GTEX_ZSCORE_THRESHOLD, color="#4393c3", linestyle="--",
                   linewidth=0.8, alpha=0.7)
        ax.set_xlabel("log₂(TPM+0.001)", fontsize=9)
        ax.set_ylabel(f"Z-score vs {baseline_label}", fontsize=9)
        ax.set_title(f"Expression Outliers vs {baseline_label}\n"
                     "(orange = LUAD driver genes)", fontsize=10, fontweight="bold")

        n_up   = (z > GTEX_ZSCORE_THRESHOLD).sum()
        n_down = (z < -GTEX_ZSCORE_THRESHOLD).sum()
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
    if has_zscore and "SYMBOL" in outliers.columns:
        top_over = (outliers[outliers[zscore_col] > 0]
                    .nlargest(15, zscore_col)[["SYMBOL", zscore_col, "TPM_LOG2_GENE"]])
        if not top_over.empty:
            bar_colors = ["#ff7f0e" if g in LUAD_GENES else "#d62728"
                          for g in top_over["SYMBOL"]]
            ax.barh(top_over["SYMBOL"], top_over[zscore_col],
                    color=bar_colors, alpha=0.85, edgecolor="white")
            ax.set_xlabel("Z-score", fontsize=9)
            ax.set_title(f"Top Over-expressed vs {baseline_label}\n"
                         "(orange = LUAD drivers)", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)

    # ── Panel [1,0]: Top under-expressed genes ───────────────────────────────
    ax = axes[1, 0]
    if has_zscore and "SYMBOL" in outliers.columns:
        top_under = (outliers[outliers[zscore_col] < 0]
                     .nsmallest(15, zscore_col)[["SYMBOL", zscore_col, "TPM_LOG2_GENE"]])
        if not top_under.empty:
            bar_colors = ["#ff7f0e" if g in LUAD_GENES else "#4393c3"
                          for g in top_under["SYMBOL"]]
            ax.barh(top_under["SYMBOL"], top_under[zscore_col],
                    color=bar_colors, alpha=0.85, edgecolor="white")
            ax.set_xlabel("Z-score", fontsize=9)
            ax.set_title(f"Top Under-expressed vs {baseline_label}\n"
                         "(orange = LUAD drivers)", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)

    # ── Panel [1,1]: TCGA-LUAD subtype similarity ────────────────────────────
    ax = axes[1, 1]
    if corr_path.exists():
        corr_df = pd.read_csv(corr_path, sep="\t")
        if "CORR" in corr_df.columns:
            corr_vals = pd.to_numeric(corr_df["CORR"], errors="coerce").dropna()
            ax.hist(corr_vals, bins=40, color="#74c476",
                    edgecolor="white", linewidth=0.5, alpha=0.85)
            ax.axvline(corr_vals.median(), color="#d62728", linestyle="--",
                       linewidth=1.5, label=f"Median r={corr_vals.median():.3f}")

            top3 = corr_df.nlargest(3, "CORR")
            label_col = ("EXT_SAMPLE_ID" if "EXT_SAMPLE_ID" in corr_df.columns
                         else corr_df.columns[1])
            for _, row in top3.iterrows():
                r = float(row["CORR"])
                ax.axvline(r, color="#ff7f0e", linestyle=":", linewidth=1, alpha=0.8)
                ax.text(r, ax.get_ylim()[1] * 0.5,
                        f" {row[label_col]}", fontsize=6, color="#ff7f0e",
                        rotation=90, va="center")

            ax.set_xlabel("Spearman r with TCGA-LUAD sample", fontsize=9)
            ax.set_ylabel("Count", fontsize=9)
            ax.set_title(f"TCGA-LUAD Subtype Similarity\n"
                         f"(n={len(corr_vals):,} reference samples)",
                         fontsize=10, fontweight="bold")
            ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_png = out_dir / f"{sample_id}_expression_summary.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"  Expression summary figure → {out_png}")


def run_sample(rnaseq_path: Path, dry_run: bool = False, figures_only: bool = False):
    """Process a single sample's RNA-seq data."""
    logger = get_logger("luad-expression")
    sample_id = rnaseq_path.name.replace(".tsv.gz", "")

    logger.info(f"\n{'='*55}")
    logger.info(f"Sample: {sample_id}")

    if dry_run:
        logger.info(f"[DRY RUN] Would process: {rnaseq_path}")
        return

    out_dir = OUT_DIR / sample_id

    # ── Figures-only mode: read existing files, regenerate figures ────────────
    if figures_only:
        expr_path = out_dir / f"{sample_id}_gene_expression.tsv.gz"
        if not expr_path.exists():
            logger.warning(f"  No gene expression file — skipping {sample_id}")
            return None
        all_genes = pd.read_csv(expr_path, sep="\t", compression="gzip", low_memory=False)

        clinical_zscores = {}
        if "SYMBOL" in all_genes.columns and "GTEX_ZSCORE" in all_genes.columns:
            for gene in CLINICAL_PANEL_GENES:
                rows = all_genes[all_genes["SYMBOL"] == gene]
                clinical_zscores[gene] = (
                    float(rows["GTEX_ZSCORE"].iloc[0]) if not rows.empty else np.nan
                )

        n_up, n_down = 0, 0
        outlier_path = out_dir / f"{sample_id}_expression_outliers.tsv"
        if outlier_path.exists():
            od = pd.read_csv(outlier_path, sep="\t")
            if "GTEX_ZSCORE" in od.columns:
                n_up   = int((od["GTEX_ZSCORE"] >  GTEX_ZSCORE_THRESHOLD).sum())
                n_down = int((od["GTEX_ZSCORE"] < -GTEX_ZSCORE_THRESHOLD).sum())

        variant_df = load_variant_table(sample_id)
        mutated_genes = set()
        if variant_df is not None and "SYMBOL" in variant_df.columns:
            mutated_genes = set(variant_df["SYMBOL"].dropna().unique())

        _plot_mutation_expression_scatter(sample_id, out_dir, all_genes,
                                          mutated_genes, logger)
        # Regenerate clinical panel only if missing
        if not (out_dir / f"{sample_id}_clinical_genes.png").exists():
            _plot_clinical_gene_panel(sample_id, out_dir, all_genes, logger)

        return {"sample_id": sample_id, "n_genes": len(all_genes),
                "clinical_zscores": clinical_zscores,
                "n_up": n_up, "n_down": n_down}
    # ── End figures-only ─────────────────────────────────────────────────────

    out_dir = OUT_DIR / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parse expression data
    # Strip Ensembl version numbers (ENSG00000000003.15 → ENSG00000000003)
    # because PCGR refdata maps against unversioned IDs
    logger.info("Step 1: Parsing expression data...")
    import tempfile
    _rna_df = pd.read_csv(rnaseq_path, sep="\t")
    if "TargetID" in _rna_df.columns:
        _rna_df["TargetID"] = _rna_df["TargetID"].str.split(".").str[0]
    _tmp = tempfile.NamedTemporaryFile(suffix=".tsv.gz", delete=False)
    _rna_df.to_csv(_tmp.name, sep="\t", index=False, compression="gzip")
    _tmp.close()
    expression_map = parse_expression(
        expression_fname_tsv = _tmp.name,
        sample_id            = sample_id,
        refdata_assembly_dir = str(REFDATA_DIR),
        logger               = logger,
    )
    import os; os.unlink(_tmp.name)

    if expression_map is None or expression_map.get("gene") is None:
        logger.warning(f"  Expression parsing failed for {sample_id}, skipping.")
        return None

    gene_expr = expression_map["gene"]
    logger.info(f"  Parsed {len(gene_expr)} genes")
    # Note: gene_expression.tsv.gz saved in Step 3 after GTEx Z-scores are added

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

    # 3. Compute GTEx-normalised Z-scores (tumor vs normal lung baseline)
    logger.info("Step 3: Computing Z-scores vs GTEx normal lung baseline...")
    gtex_ref = load_gtex_reference(logger)
    use_gtex = gtex_ref is not None
    scored = None          # set inside if use_gtex; used by visualization
    clinical_zscores = {}  # per-patient clinical gene Z-scores for cohort heatmap
    n_up, n_down = 0, 0    # outlier counts for cohort distribution figure

    if use_gtex:
        # Filter to protein-coding genes for outlier reporting
        pc_mask = gene_expr.get("BIOTYPE", pd.Series(["protein_coding"] * len(gene_expr))) == "protein_coding"
        gene_expr_pc = gene_expr[pc_mask] if "BIOTYPE" in gene_expr.columns else gene_expr

        scored = compute_gtex_zscores(gene_expr_pc, gtex_ref)

        # Save all genes with GTEx Z-scores (update gene_expression file)
        # Merge Z-scores back into the full gene_expr table
        gene_expr = gene_expr.copy()
        gene_expr["ENSEMBL_GENE_ID"] = gene_expr["ENSEMBL_GENE_ID"].astype(str).str.split(".").str[0]
        zcols = ["ENSEMBL_GENE_ID", "GTEX_ZSCORE", "GTEX_PERCENTILE",
                 "GTEX_DIRECTION", "gtex_mean_log2tpm", "gtex_std_log2tpm"]
        available_zcols = [c for c in zcols if c in scored.columns]
        gene_expr = gene_expr.merge(scored[available_zcols], on="ENSEMBL_GENE_ID", how="left")
        expr_out = out_dir / f"{sample_id}_gene_expression.tsv.gz"
        gene_expr.to_csv(expr_out, sep="\t", index=False, compression="gzip")
        logger.info(f"  Gene expression (with GTEx Z-scores) → {expr_out}")

        # Save outlier table: genes with |GTEX_ZSCORE| > threshold
        outlier_mask = scored["GTEX_ZSCORE"].abs() > GTEX_ZSCORE_THRESHOLD
        outliers_df = (scored[outlier_mask]
                       .sort_values("GTEX_ZSCORE", ascending=False)
                       .reset_index(drop=True))
        outlier_out = out_dir / f"{sample_id}_expression_outliers.tsv"
        outliers_df.to_csv(outlier_out, sep="\t", index=False)
        n_up   = int((outliers_df["GTEX_ZSCORE"] >  GTEX_ZSCORE_THRESHOLD).sum())
        n_down = int((outliers_df["GTEX_ZSCORE"] < -GTEX_ZSCORE_THRESHOLD).sum())
        logger.info(f"  GTEx outliers: ↑{n_up} over-expressed, ↓{n_down} under-expressed "
                    f"→ {outlier_out}")

        # Collect clinical gene Z-scores for cohort heatmap
        if "SYMBOL" in scored.columns:
            for gene in CLINICAL_PANEL_GENES:
                rows = scored[scored["SYMBOL"] == gene]
                clinical_zscores[gene] = (
                    float(rows["GTEX_ZSCORE"].iloc[0]) if not rows.empty else np.nan
                )
    else:
        # Save gene_expression without GTEx columns
        expr_out = out_dir / f"{sample_id}_gene_expression.tsv.gz"
        gene_expr.to_csv(expr_out, sep="\t", index=False, compression="gzip")
        logger.info(f"  Gene expression saved → {expr_out}")

        # Fallback: PCGR TCGA-cohort-based outlier detection
        logger.info("  [Fallback] Using PCGR TCGA-cohort outlier detection ...")
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
                logger.info(f"  TCGA-cohort outliers ({len(outliers)} genes) → {outlier_out}")
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
    _plot_expression_summary(sample_id, out_dir_path, logger, use_gtex=use_gtex)

    if use_gtex and scored is not None:
        _plot_clinical_gene_panel(sample_id, out_dir_path, scored, logger)
        mutated_genes = set()
        if variant_df is not None and "SYMBOL" in variant_df.columns:
            mutated_genes = set(variant_df["SYMBOL"].dropna().unique())
        _plot_mutation_expression_scatter(sample_id, out_dir_path, scored,
                                          mutated_genes, logger)

    return {"sample_id": sample_id, "n_genes": len(gene_expr),
            "clinical_zscores": clinical_zscores,
            "n_up": n_up, "n_down": n_down}


def main():
    parser = argparse.ArgumentParser(description="LUAD expression analysis (pcgr Python API)")
    parser.add_argument("--sample",       type=str, help="Single sample ID (e.g. TCGA-86-A4D0)")
    parser.add_argument("--dry_run",      action="store_true", help="Validate inputs without running")
    parser.add_argument("--figures-only", action="store_true",
                        help="Read existing output files and regenerate figures only (no heavy recomputation)")
    args = parser.parse_args()

    figures_only = getattr(args, "figures_only", False)

    if not figures_only and not REFDATA_DIR.exists():
        print(f"[ERROR] PCGR refdata not found: {REFDATA_DIR}")
        return

    rnaseq_files = get_rnaseq_files(args.sample)
    if not rnaseq_files:
        print(f"[ERROR] No RNA-seq files found in {RNASEQ_DIR}")
        return

    print(f"[INFO] Found {len(rnaseq_files)} sample(s) to process"
          + ("  [figures-only mode]" if figures_only else ""))
    results = []
    for f in rnaseq_files:
        r = run_sample(f, dry_run=args.dry_run, figures_only=figures_only)
        if r:
            results.append(r)

    if results:
        exclude = {"clinical_zscores", "n_up", "n_down"}
        summary = pd.DataFrame([{k: v for k, v in r.items() if k not in exclude}
                                 for r in results])
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        summary_path = OUT_DIR / "all_samples_summary.tsv"
        summary.to_csv(summary_path, sep="\t", index=False)
        print(f"\n[Summary] {len(results)} samples → {summary_path}")

        logger = get_logger("luad-expression")

        # Cohort-level clinical gene heatmap
        clinical_data = {r["sample_id"]: r["clinical_zscores"]
                         for r in results if r.get("clinical_zscores")}
        if clinical_data:
            logger.info("\nGenerating cohort clinical gene heatmap...")
            _plot_cohort_heatmap(clinical_data, OUT_DIR, logger)

        # Cohort-level outlier distribution
        outlier_counts = {r["sample_id"]: {"n_up": r["n_up"], "n_down": r["n_down"]}
                          for r in results if "n_up" in r}
        if outlier_counts:
            logger.info("Generating cohort outlier distribution figure...")
            _plot_cohort_outlier_distribution(outlier_counts, OUT_DIR, logger)


if __name__ == "__main__":
    main()
