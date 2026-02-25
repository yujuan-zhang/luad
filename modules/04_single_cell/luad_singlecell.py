#!/usr/bin/env python
"""
luad_singlecell.py
------------------
LUAD single-cell RNA-seq tumor microenvironment (TME) characterization.
Uses the publicly available GSE131907 Lung Cancer Atlas (Kim et al. 2020).

Steps:
  1. Load cell annotation (208k cells, 58 samples, pre-labeled by authors)
  2. Compute TME cell-type fractions per sample (tLung = tumor lung tissue)
  3. Calculate immune phenotype metrics:
       - CD8/Treg ratio (cytotoxic vs regulatory)
       - Exhausted CD8 T-cell fraction (immune exhaustion marker)
       - Myeloid suppression index (mo-Mac + CD163+ DCs)
       - B-cell fraction (tertiary lymphoid structures proxy)
  4. Classify samples into immune phenotypes:
       - Inflamed / Excluded / Desert
  5. Save outputs:
       - per_sample_tme_fractions.tsv   — cell type % per sample
       - per_sample_immune_metrics.tsv  — derived immune scores
       - luad_tme_summary.tsv           — dataset-level summary

Input:
  modules/04_single_cell/data/input/GSE131907/
    GSE131907_Lung_Cancer_cell_annotation.txt.gz

Output:
  data/output/04_single_cell/
    per_sample_tme_fractions.tsv
    per_sample_immune_metrics.tsv
    luad_tme_summary.tsv

Usage:
  python luad_singlecell.py             # full dataset
  python luad_singlecell.py --dry_run   # validate inputs only
"""

import argparse
import logging
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR    = SCRIPT_DIR / "data/input/GSE131907"
ANN_FILE    = DATA_DIR / "GSE131907_Lung_Cancer_cell_annotation.txt.gz"
OUT_DIR     = PROJECT_DIR / "data/output/04_single_cell"

# ── Cell type groupings ────────────────────────────────────────────────────────
# Map Cell_subtype → broad immune lineage for TME analysis
LINEAGE_MAP = {
    # T cells
    "CD4+ Th":             "CD4+ T",
    "CD8+/CD4+ Mixed Th":  "CD4+ T",
    "Naive CD4+ T":        "CD4+ T",
    "Treg":                "Treg",
    "Exhausted CD8+ T":    "Exhausted CD8+ T",
    "Cytotoxic CD8+ T":    "Cytotoxic CD8+ T",
    "CD8 low T":           "CD8+ T (other)",
    "NK":                  "NK",
    # Myeloid
    "mo-Mac":              "Macrophage",
    "Alveolar Mac":        "Macrophage",
    "CD163+CD14+ DCs":     "Immunosuppressive myeloid",
    "CD1c+ DCs":           "DC",
    "Monocytes":           "Monocyte",
    # B cells / plasma
    "Follicular B cells":  "B cell",
    "MALT B cells":        "B cell",
    "GC B cells":          "B cell",
    "Plasma cells":        "Plasma cell",
    # Other immune
    "MAST":                "Mast cell",
    # More T cell subtypes
    "Naive CD8+ T":        "CD8+ T (other)",
    "Exhausted Tfh":       "CD4+ T",
    # More myeloid
    "Activated DCs":       "DC",
    "CD141+ DCs":          "DC",
    "pDCs":                "DC",
    "CD207+CD1a+ LCs":     "DC",
    "Pleural Mac":         "Macrophage",
    # More B / plasma
    "GC B cells in the LZ":"B cell",
    "GC B cells in the DZ":"B cell",
    # Stromal / tumor
    "tS1":                 "Tumor epithelial",
    "tS2":                 "Tumor epithelial",
    "tS3":                 "Tumor epithelial",
    "tS4":                 "Tumor epithelial",
    "COL13A1+ matrix FBs": "Fibroblast",
    "COL14A1+ matrix FBs": "Fibroblast",
    "Myofibroblasts":      "Fibroblast",
    "Pericytes":           "Pericyte",
    "Tip-like ECs":        "Endothelial",
    "Stalk-like ECs":      "Endothelial",
    "Tumor ECs":           "Endothelial",
    "Lymphatic ECs":       "Endothelial",
    "EPCs":                "Endothelial",
    "Smooth muscle cells": "Stromal other",
    "Mesothelial cells":   "Stromal other",
}

# Cells included in TME (exclude tumor epithelial for immune TME metrics)
IMMUNE_LINEAGES = {
    "CD4+ T", "Treg", "Exhausted CD8+ T", "Cytotoxic CD8+ T",
    "CD8+ T (other)", "NK", "Macrophage", "Immunosuppressive myeloid",
    "DC", "Monocyte", "B cell", "Plasma cell", "Mast cell",
}


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
    return logging.getLogger(name)


def load_annotation(logger) -> pd.DataFrame:
    """Load and validate the GSE131907 cell annotation file."""
    if not ANN_FILE.exists():
        raise FileNotFoundError(f"Annotation file not found: {ANN_FILE}")
    logger.info(f"Loading cell annotation: {ANN_FILE.name}")
    ann = pd.read_csv(ANN_FILE, sep="\t", low_memory=False)
    logger.info(f"  Total cells: {len(ann):,}")
    logger.info(f"  Columns: {ann.columns.tolist()}")
    return ann


def map_lineage(ann: pd.DataFrame) -> pd.DataFrame:
    """Add Lineage column from Cell_subtype via LINEAGE_MAP."""
    ann = ann.copy()
    ann["Lineage"] = ann["Cell_subtype"].map(LINEAGE_MAP).fillna("Other")
    logger = get_logger("luad-sc")
    unmapped = ann[ann["Lineage"] == "Other"]["Cell_subtype"].value_counts()
    if len(unmapped) > 0:
        logger.info(f"  Unmapped subtypes ({len(unmapped)} types) → 'Other'")
    return ann


def compute_tme_fractions(ann: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Compute cell-type fraction per sample for tLung (tumor lung) cells.
    Returns: rows = samples, columns = lineage categories, values = fraction 0-1.
    """
    tlung = ann[ann["Sample_Origin"] == "tLung"].copy()
    logger.info(f"  Tumor lung cells: {len(tlung):,} across {tlung['Sample'].nunique()} samples")

    # Count per sample × lineage
    counts = (
        tlung.groupby(["Sample", "Lineage"])
        .size()
        .unstack(fill_value=0)
    )

    # Fraction of total cells per sample
    fractions = counts.div(counts.sum(axis=1), axis=0).round(4)
    fractions.index.name = "sample_id"
    return fractions


def compute_immune_metrics(fractions: pd.DataFrame, ann: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate derived immune phenotype metrics per sample.
    """
    df = fractions.copy()
    metrics = pd.DataFrame(index=df.index)

    # Helper: get column safely
    def col(name, default=0.0):
        return df[name] if name in df.columns else pd.Series(default, index=df.index)

    # CD8 all = cytotoxic + exhausted + other
    cd8_total = (
        col("Cytotoxic CD8+ T") +
        col("Exhausted CD8+ T") +
        col("CD8+ T (other)")
    )
    treg = col("Treg")

    metrics["cd8_fraction"]      = cd8_total.round(4)
    metrics["treg_fraction"]     = treg.round(4)
    metrics["exhausted_cd8_frac"]= col("Exhausted CD8+ T").round(4)
    metrics["cytotoxic_cd8_frac"]= col("Cytotoxic CD8+ T").round(4)

    # CD8/Treg ratio (higher = more cytotoxic, better prognosis)
    metrics["cd8_treg_ratio"]    = (cd8_total / treg.replace(0, np.nan)).round(3)

    # Macrophage burden
    metrics["macrophage_frac"]   = col("Macrophage").round(4)
    metrics["immunosupp_myeloid"]= col("Immunosuppressive myeloid").round(4)

    # B cell / TLS proxy
    metrics["b_cell_frac"]       = (col("B cell") + col("Plasma cell")).round(4)

    # NK cells
    metrics["nk_frac"]           = col("NK").round(4)

    # Tumor epithelial fraction
    metrics["tumor_frac"]        = col("Tumor epithelial").round(4)

    # Immune cell fraction (all immune lineages)
    immune_cols = [c for c in df.columns if c in IMMUNE_LINEAGES]
    metrics["immune_frac"]       = df[immune_cols].sum(axis=1).round(4)

    # Exhaustion index (exhausted CD8 / total CD8)
    metrics["exhaustion_index"]  = (
        col("Exhausted CD8+ T") / cd8_total.replace(0, np.nan)
    ).round(4)

    # Immune phenotype classification
    #   Inflamed:  high CD8, high immune_frac, low exhaustion
    #   Excluded:  high immune but suppressed (high Treg/Macro, low CD8 ratio)
    #   Desert:    low immune_frac overall
    def classify(row):
        if row["immune_frac"] < 0.15:
            return "Desert"
        elif row["cd8_treg_ratio"] > 3.0 and row["exhaustion_index"] < 0.4:
            return "Inflamed"
        elif row["macrophage_frac"] > 0.15 or row["treg_fraction"] > 0.05:
            return "Excluded"
        else:
            return "Inflamed"

    metrics["immune_phenotype"] = metrics.apply(classify, axis=1)

    return metrics


def compute_dataset_summary(metrics: pd.DataFrame, logger) -> pd.DataFrame:
    """Compute dataset-level summary statistics."""
    numeric = metrics.select_dtypes(include="number")
    summary = numeric.describe().T[["mean", "std", "min", "50%", "max"]].round(4)
    summary.columns = ["mean", "std", "min", "median", "max"]

    phenotype_counts = metrics["immune_phenotype"].value_counts()
    logger.info("  Immune phenotype distribution:")
    for ptype, n in phenotype_counts.items():
        logger.info(f"    {ptype}: {n} samples ({100*n/len(metrics):.1f}%)")

    return summary


# ── Visualization ─────────────────────────────────────────────────────────────

# Fixed color palette for cell lineages
LINEAGE_COLORS = {
    "Tumor epithelial":        "#e6194b",
    "CD4+ T":                  "#4363d8",
    "Treg":                    "#911eb4",
    "Exhausted CD8+ T":        "#f58231",
    "Cytotoxic CD8+ T":        "#3cb44b",
    "CD8+ T (other)":          "#a9c9a4",
    "NK":                      "#42d4f4",
    "Macrophage":              "#ffe119",
    "Immunosuppressive myeloid":"#9a6324",
    "DC":                      "#ffd8b1",
    "Monocyte":                "#808000",
    "B cell":                  "#469990",
    "Plasma cell":             "#aaffc3",
    "Mast cell":               "#dcbeff",
    "Fibroblast":              "#bfef45",
    "Endothelial":             "#fabed4",
    "Pericyte":                "#fffac8",
    "Stromal other":           "#e6beff",
    "Other":                   "#a9a9a9",
}

PHENOTYPE_COLORS = {
    "Inflamed": "#d73027",
    "Excluded": "#fc8d59",
    "Desert":   "#4575b4",
}


def plot_tme_overview(fractions: pd.DataFrame, metrics: pd.DataFrame,
                      out_dir: Path, logger):
    """
    4-panel TME overview figure:
      [0,0] Stacked bar: cell-type fractions per sample
      [0,1] CD8/Treg ratio scatter (immune phenotype landscape)
      [1,0] Key immune metrics heatmap
      [1,1] Immune phenotype donut chart
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.patch.set_facecolor("#f8f9fa")
    fig.suptitle("LUAD Tumor Microenvironment (GSE131907)", fontsize=14,
                 fontweight="bold", y=1.01)

    samples = fractions.index.tolist()
    lineage_cols = fractions.columns.tolist()

    # ── Panel [0,0]: Stacked bar ──────────────────────────────────────────────
    ax = axes[0, 0]
    bottom = np.zeros(len(samples))
    for lin in lineage_cols:
        vals = fractions[lin].values
        color = LINEAGE_COLORS.get(lin, "#aaaaaa")
        ax.bar(samples, vals, bottom=bottom, color=color,
               label=lin, width=0.75, edgecolor="white", linewidth=0.4)
        bottom += vals

    ax.set_ylabel("Cell fraction", fontsize=9)
    ax.set_title("TME Cell Composition\n(tLung samples)", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(samples)))
    ax.set_xticklabels([s.replace("LUNG_", "") for s in samples],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.01)

    # Color-code x-labels by immune phenotype
    if "immune_phenotype" in metrics.columns:
        for tick, sample in zip(ax.get_xticklabels(), samples):
            ptype = metrics.loc[sample, "immune_phenotype"] if sample in metrics.index else "Unknown"
            tick.set_color(PHENOTYPE_COLORS.get(ptype, "black"))

    handles = [mpatches.Patch(color=LINEAGE_COLORS.get(l, "#aaa"), label=l)
               for l in lineage_cols if l in LINEAGE_COLORS]
    ax.legend(handles=handles, fontsize=6, loc="upper left",
              bbox_to_anchor=(1.01, 1), ncol=1, framealpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel [0,1]: CD8 vs Treg scatter (immune phenotype) ─────────────────
    ax = axes[0, 1]
    for sample in samples:
        if sample not in metrics.index:
            continue
        row    = metrics.loc[sample]
        ptype  = row.get("immune_phenotype", "Unknown")
        color  = PHENOTYPE_COLORS.get(ptype, "#aaaaaa")
        ax.scatter(row["treg_fraction"], row["cd8_fraction"],
                   c=color, s=120, zorder=5, edgecolors="white", linewidths=0.8)
        ax.annotate(sample.replace("LUNG_", ""),
                    xy=(row["treg_fraction"], row["cd8_fraction"]),
                    xytext=(3, 3), textcoords="offset points", fontsize=7)

    # Reference lines
    ax.axhline(0.1, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(0.05, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Treg fraction", fontsize=9)
    ax.set_ylabel("CD8+ T-cell fraction", fontsize=9)
    ax.set_title("Immune Phenotype Landscape\n(CD8 vs Treg)", fontsize=10, fontweight="bold")

    legend_handles = [mpatches.Patch(color=c, label=p)
                      for p, c in PHENOTYPE_COLORS.items()]
    ax.legend(handles=legend_handles, fontsize=8, framealpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel [1,0]: Immune metrics heatmap ──────────────────────────────────
    ax = axes[1, 0]
    heatmap_cols = ["immune_frac", "cd8_fraction", "treg_fraction",
                    "exhausted_cd8_frac", "cytotoxic_cd8_frac",
                    "macrophage_frac", "b_cell_frac", "nk_frac"]
    heatmap_cols = [c for c in heatmap_cols if c in metrics.columns]
    hm_data = metrics[heatmap_cols].copy()
    # Normalize each column 0-1 for display
    hm_norm = (hm_data - hm_data.min()) / (hm_data.max() - hm_data.min() + 1e-9)

    im = ax.imshow(hm_norm.values, aspect="auto", cmap="RdYlBu_r",
                   vmin=0, vmax=1)
    ax.set_xticks(range(len(heatmap_cols)))
    ax.set_xticklabels([c.replace("_frac", "").replace("_", " ")
                        for c in heatmap_cols], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(samples)))
    ax.set_yticklabels([s.replace("LUNG_", "") for s in samples], fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Normalized value")
    ax.set_title("Immune Metric Heatmap\n(row-normalized)", fontsize=10, fontweight="bold")

    # Annotate phenotype on y-axis labels
    if "immune_phenotype" in metrics.columns:
        for ytick, sample in zip(ax.get_yticklabels(), samples):
            ptype = metrics.loc[sample, "immune_phenotype"] if sample in metrics.index else "Unknown"
            ytick.set_color(PHENOTYPE_COLORS.get(ptype, "black"))

    # ── Panel [1,1]: Immune phenotype donut ──────────────────────────────────
    ax = axes[1, 1]
    if "immune_phenotype" in metrics.columns:
        phenotype_counts = metrics["immune_phenotype"].value_counts()
        colors = [PHENOTYPE_COLORS.get(p, "#aaaaaa") for p in phenotype_counts.index]
        wedges, texts, autotexts = ax.pie(
            phenotype_counts.values,
            labels=phenotype_counts.index,
            colors=colors,
            autopct="%1.0f%%",
            pctdistance=0.75,
            startangle=90,
            wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
        )
        for text in texts:
            text.set_fontsize(10)
            text.set_fontweight("bold")
        for at in autotexts:
            at.set_fontsize(9)

        ax.set_title(f"Immune Phenotype Distribution\n(n={len(metrics)} tLung samples)",
                     fontsize=10, fontweight="bold")

        # Add legend with counts
        legend_labels = [f"{p} (n={n})" for p, n in phenotype_counts.items()]
        handles = [mpatches.Patch(color=PHENOTYPE_COLORS.get(p, "#aaa"), label=lbl)
                   for p, lbl in zip(phenotype_counts.index, legend_labels)]
        ax.legend(handles=handles, fontsize=9, loc="lower center",
                  bbox_to_anchor=(0.5, -0.12), ncol=len(phenotype_counts))
    else:
        ax.axis("off")

    plt.tight_layout()
    out_png = out_dir / "luad_tme_overview.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"  TME overview figure → {out_png}")


def main():
    parser = argparse.ArgumentParser(description="LUAD single-cell TME characterization")
    parser.add_argument("--dry_run", action="store_true", help="Validate inputs only")
    args = parser.parse_args()

    logger = get_logger("luad-sc")
    logger.info(f"\n{'='*55}")
    logger.info("Module 04: Single-cell TME characterization (GSE131907)")

    if not ANN_FILE.exists():
        print(f"[ERROR] Annotation file not found: {ANN_FILE}")
        return

    if args.dry_run:
        logger.info(f"[DRY RUN] Would process: {ANN_FILE}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load annotation
    ann = load_annotation(logger)

    # 2. Map to broad lineages
    logger.info("Step 1: Mapping cell subtypes to lineages...")
    ann = map_lineage(ann)

    # 3. Compute TME fractions (tLung only)
    logger.info("Step 2: Computing TME cell-type fractions...")
    fractions = compute_tme_fractions(ann, logger)

    fractions_out = OUT_DIR / "per_sample_tme_fractions.tsv"
    fractions.to_csv(fractions_out, sep="\t")
    logger.info(f"  Fractions saved ({fractions.shape[0]} samples × {fractions.shape[1]} lineages) → {fractions_out}")

    # 4. Compute immune phenotype metrics
    logger.info("Step 3: Computing immune phenotype metrics...")
    metrics = compute_immune_metrics(fractions, ann)

    metrics_out = OUT_DIR / "per_sample_immune_metrics.tsv"
    metrics.to_csv(metrics_out, sep="\t")
    logger.info(f"  Metrics saved → {metrics_out}")

    # 5. Dataset summary
    logger.info("Step 4: Generating dataset summary...")
    summary = compute_dataset_summary(metrics, logger)

    summary_out = OUT_DIR / "luad_tme_summary.tsv"
    summary.to_csv(summary_out, sep="\t")
    logger.info(f"  Summary saved → {summary_out}")

    # 6. Visualization
    logger.info("Step 5: Generating TME visualization...")
    plot_tme_overview(fractions, metrics, OUT_DIR, logger)

    print(f"\n[Module 04 Complete] Outputs in {OUT_DIR}")
    print(f"  Samples analyzed: {fractions.shape[0]}")
    print(f"\n  Immune metrics preview:")
    print(metrics[["immune_frac", "cd8_fraction", "treg_fraction",
                    "cd8_treg_ratio", "exhaustion_index", "immune_phenotype"]].head(10).to_string())


if __name__ == "__main__":
    main()
