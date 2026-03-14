#!/usr/bin/env python
"""
luad_singlecell.py
------------------
LUAD TME characterization — two-stage pipeline.

Stage 1  GSE131907 single-cell reference analysis (Kim et al. 2020)
         - Cell-type fractions per tLung sample (58 samples)
         - Immune phenotype classification (Inflamed/Excluded/Desert)
         - Reference TME overview figure

Stage 2  ssGSEA deconvolution of TCGA-LUAD bulk RNA-seq (517 patients)
         - Curated cell-type signature gene sets (literature-based)
         - Per-patient ssGSEA enrichment scores → relative TME composition
         - Immune phenotype classification for each TCGA patient
         - Per-patient _tme_scores.tsv + cohort heatmap

Input:
  modules/04_single_cell/data/input/GSE131907/
    GSE131907_Lung_Cancer_cell_annotation.txt.gz  (Stage 1)
  data/output/03_expression/{sample}/{sample}_gene_expression.tsv.gz (Stage 2)

Output:
  data/output/04_single_cell/
    per_sample_tme_fractions.tsv      GSE131907 reference
    per_sample_immune_metrics.tsv     GSE131907 reference
    luad_tme_summary.tsv              GSE131907 reference
    luad_tme_overview.png             GSE131907 reference figure
    tme_cohort_scores.tsv             TCGA 517 patients × 10 cell types
    tme_cohort_heatmap.png            TCGA cohort TME heatmap
    tme_phenotype_distribution.png    TCGA phenotype donut
  data/output/04_single_cell/{sample}/
    {sample}_tme_scores.tsv           per-TCGA-patient scores

Usage:
  python luad_singlecell.py             # full pipeline (Stage 1 + 2)
  python luad_singlecell.py --skip_sc   # skip Stage 1 (no GSE131907 data)
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
EXPR_DIR    = PROJECT_DIR / "data/output/03_expression"
OUT_DIR     = PROJECT_DIR / "data/output/04_single_cell"

# ── Stage 1: Cell type groupings (GSE131907) ──────────────────────────────────
LINEAGE_MAP = {
    "CD4+ Th":             "CD4+ T",
    "CD8+/CD4+ Mixed Th":  "CD4+ T",
    "Naive CD4+ T":        "CD4+ T",
    "Treg":                "Treg",
    "Exhausted CD8+ T":    "Exhausted CD8+ T",
    "Cytotoxic CD8+ T":    "Cytotoxic CD8+ T",
    "CD8 low T":           "CD8+ T (other)",
    "NK":                  "NK",
    "mo-Mac":              "Macrophage",
    "Alveolar Mac":        "Macrophage",
    "CD163+CD14+ DCs":     "Immunosuppressive myeloid",
    "CD1c+ DCs":           "DC",
    "Monocytes":           "Monocyte",
    "Follicular B cells":  "B cell",
    "MALT B cells":        "B cell",
    "GC B cells":          "B cell",
    "Plasma cells":        "Plasma cell",
    "MAST":                "Mast cell",
    "Naive CD8+ T":        "CD8+ T (other)",
    "Exhausted Tfh":       "CD4+ T",
    "Activated DCs":       "DC",
    "CD141+ DCs":          "DC",
    "pDCs":                "DC",
    "CD207+CD1a+ LCs":     "DC",
    "Pleural Mac":         "Macrophage",
    "GC B cells in the LZ":"B cell",
    "GC B cells in the DZ":"B cell",
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

IMMUNE_LINEAGES = {
    "CD4+ T", "Treg", "Exhausted CD8+ T", "Cytotoxic CD8+ T",
    "CD8+ T (other)", "NK", "Macrophage", "Immunosuppressive myeloid",
    "DC", "Monocyte", "B cell", "Plasma cell", "Mast cell",
}

# ── Stage 2: curated TME signature gene sets (literature-based) ───────────────
TME_SIGNATURES = {
    "CD8_T_cytotoxic":  ["CD8A", "CD8B", "GZMB", "PRF1", "GZMA", "IFNG",
                         "NKG7", "GNLY", "GZMK"],
    "Treg":             ["FOXP3", "IL2RA", "CTLA4", "IKZF2", "TNFRSF18",
                         "ICOS", "IL10", "ENTPD1"],
    "CD8_T_exhausted":  ["PDCD1", "LAG3", "HAVCR2", "TIGIT", "TOX",
                         "ENTPD1", "CXCL13", "VCAM1"],
    "CD4_T":            ["CD4", "IL7R", "TCF7", "CCR7", "SELL", "LDHB"],
    "NK":               ["NCAM1", "KLRB1", "NKG7", "KLRD1", "KLRF1",
                         "XCL1", "XCL2", "FCGR3A"],
    "Macrophage_M1":    ["CD68", "CD80", "TNF", "CXCL10", "IL1B",
                         "IL6", "CXCL9", "CXCL11"],
    "Macrophage_M2":    ["CD163", "MRC1", "ARG1", "TGFB1", "IL10",
                         "VEGFA", "CCL18", "FOLR2"],
    "B_cell":           ["CD19", "CD79A", "MS4A1", "PAX5", "CD22",
                         "BLK", "FCRL5"],
    "Fibroblast":       ["COL1A1", "COL1A2", "FAP", "ACTA2", "FN1",
                         "PDGFRA", "PDPN"],
    "Endothelial":      ["PECAM1", "VWF", "ENG", "CLDN5", "CDH5", "KDR"],
}

PHENOTYPE_COLORS = {
    "Inflamed": "#d73027",
    "Excluded": "#fc8d59",
    "Desert":   "#4575b4",
}

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

CELL_TYPE_COLORS = {
    "CD8_T_cytotoxic":  "#3cb44b",
    "Treg":             "#911eb4",
    "CD8_T_exhausted":  "#f58231",
    "CD4_T":            "#4363d8",
    "NK":               "#42d4f4",
    "Macrophage_M1":    "#e6194b",
    "Macrophage_M2":    "#ffe119",
    "B_cell":           "#469990",
    "Fibroblast":       "#bfef45",
    "Endothelial":      "#fabed4",
}


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
    return logging.getLogger(name)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1: GSE131907 single-cell reference analysis
# ══════════════════════════════════════════════════════════════════════════════

def load_annotation(logger) -> pd.DataFrame:
    logger.info(f"Loading cell annotation: {ANN_FILE.name}")
    ann = pd.read_csv(ANN_FILE, sep="\t", low_memory=False)
    logger.info(f"  Total cells: {len(ann):,}")
    return ann


def map_lineage(ann: pd.DataFrame) -> pd.DataFrame:
    ann = ann.copy()
    ann["Lineage"] = ann["Cell_subtype"].map(LINEAGE_MAP).fillna("Other")
    return ann


def compute_tme_fractions(ann: pd.DataFrame, logger) -> pd.DataFrame:
    tlung = ann[ann["Sample_Origin"] == "tLung"].copy()
    logger.info(f"  Tumor lung cells: {len(tlung):,} across {tlung['Sample'].nunique()} samples")
    counts = (
        tlung.groupby(["Sample", "Lineage"])
        .size()
        .unstack(fill_value=0)
    )
    fractions = counts.div(counts.sum(axis=1), axis=0).round(4)
    fractions.index.name = "sample_id"
    return fractions


def compute_immune_metrics(fractions: pd.DataFrame, ann: pd.DataFrame) -> pd.DataFrame:
    df = fractions.copy()
    metrics = pd.DataFrame(index=df.index)

    def col(name, default=0.0):
        return df[name] if name in df.columns else pd.Series(default, index=df.index)

    cd8_total = (
        col("Cytotoxic CD8+ T") + col("Exhausted CD8+ T") + col("CD8+ T (other)")
    )
    treg = col("Treg")

    metrics["cd8_fraction"]       = cd8_total.round(4)
    metrics["treg_fraction"]      = treg.round(4)
    metrics["exhausted_cd8_frac"] = col("Exhausted CD8+ T").round(4)
    metrics["cytotoxic_cd8_frac"] = col("Cytotoxic CD8+ T").round(4)
    metrics["cd8_treg_ratio"]     = (cd8_total / treg.replace(0, np.nan)).round(3)
    metrics["macrophage_frac"]    = col("Macrophage").round(4)
    metrics["immunosupp_myeloid"] = col("Immunosuppressive myeloid").round(4)
    metrics["b_cell_frac"]        = (col("B cell") + col("Plasma cell")).round(4)
    metrics["nk_frac"]            = col("NK").round(4)
    metrics["tumor_frac"]         = col("Tumor epithelial").round(4)
    immune_cols = [c for c in df.columns if c in IMMUNE_LINEAGES]
    metrics["immune_frac"]        = df[immune_cols].sum(axis=1).round(4)
    metrics["exhaustion_index"]   = (
        col("Exhausted CD8+ T") / cd8_total.replace(0, np.nan)
    ).round(4)

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
    numeric = metrics.select_dtypes(include="number")
    summary = numeric.describe().T[["mean", "std", "min", "50%", "max"]].round(4)
    summary.columns = ["mean", "std", "min", "median", "max"]
    phenotype_counts = metrics["immune_phenotype"].value_counts()
    logger.info("  Immune phenotype distribution (GSE131907):")
    for ptype, n in phenotype_counts.items():
        logger.info(f"    {ptype}: {n} samples ({100*n/len(metrics):.1f}%)")
    return summary


def plot_tme_overview(fractions: pd.DataFrame, metrics: pd.DataFrame,
                      out_dir: Path, logger):
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.patch.set_facecolor("#f8f9fa")
    fig.suptitle("LUAD Tumor Microenvironment Reference — GSE131907 (Kim et al. 2020)",
                 fontsize=13, fontweight="bold", y=1.01)

    samples = fractions.index.tolist()
    lineage_cols = fractions.columns.tolist()

    # Panel [0,0]: Stacked bar
    ax = axes[0, 0]
    bottom = np.zeros(len(samples))
    for lin in lineage_cols:
        vals = fractions[lin].values
        color = LINEAGE_COLORS.get(lin, "#aaaaaa")
        ax.bar(samples, vals, bottom=bottom, color=color,
               label=lin, width=0.75, edgecolor="white", linewidth=0.4)
        bottom += vals
    ax.set_ylabel("Cell fraction", fontsize=9)
    ax.set_title("TME Cell Composition (tLung samples)", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(samples)))
    ax.set_xticklabels([s.replace("LUNG_", "") for s in samples],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.01)
    if "immune_phenotype" in metrics.columns:
        for tick, sample in zip(ax.get_xticklabels(), samples):
            ptype = metrics.loc[sample, "immune_phenotype"] if sample in metrics.index else "Unknown"
            tick.set_color(PHENOTYPE_COLORS.get(ptype, "black"))
    handles = [mpatches.Patch(color=LINEAGE_COLORS.get(l, "#aaa"), label=l)
               for l in lineage_cols if l in LINEAGE_COLORS]
    ax.legend(handles=handles, fontsize=6, loc="upper left",
              bbox_to_anchor=(1.01, 1), ncol=1, framealpha=0.8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Panel [0,1]: CD8 vs Treg scatter
    ax = axes[0, 1]
    for sample in samples:
        if sample not in metrics.index:
            continue
        row = metrics.loc[sample]
        ptype = row.get("immune_phenotype", "Unknown")
        color = PHENOTYPE_COLORS.get(ptype, "#aaaaaa")
        ax.scatter(row["treg_fraction"], row["cd8_fraction"],
                   c=color, s=120, zorder=5, edgecolors="white", linewidths=0.8)
        ax.annotate(sample.replace("LUNG_", ""),
                    xy=(row["treg_fraction"], row["cd8_fraction"]),
                    xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.axhline(0.1, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(0.05, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Treg fraction", fontsize=9)
    ax.set_ylabel("CD8+ T-cell fraction", fontsize=9)
    ax.set_title("Immune Phenotype Landscape (CD8 vs Treg)", fontsize=10, fontweight="bold")
    legend_handles = [mpatches.Patch(color=c, label=p) for p, c in PHENOTYPE_COLORS.items()]
    ax.legend(handles=legend_handles, fontsize=8, framealpha=0.8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Panel [1,0]: Immune metrics heatmap
    ax = axes[1, 0]
    heatmap_cols = ["immune_frac", "cd8_fraction", "treg_fraction",
                    "exhausted_cd8_frac", "cytotoxic_cd8_frac",
                    "macrophage_frac", "b_cell_frac", "nk_frac"]
    heatmap_cols = [c for c in heatmap_cols if c in metrics.columns]
    hm_data = metrics[heatmap_cols].copy()
    hm_norm = (hm_data - hm_data.min()) / (hm_data.max() - hm_data.min() + 1e-9)
    im = ax.imshow(hm_norm.values, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(heatmap_cols)))
    ax.set_xticklabels([c.replace("_frac", "").replace("_", " ")
                        for c in heatmap_cols], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(samples)))
    ax.set_yticklabels([s.replace("LUNG_", "") for s in samples], fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Normalized value")
    ax.set_title("Immune Metric Heatmap (row-normalized)", fontsize=10, fontweight="bold")
    if "immune_phenotype" in metrics.columns:
        for ytick, sample in zip(ax.get_yticklabels(), samples):
            ptype = metrics.loc[sample, "immune_phenotype"] if sample in metrics.index else "Unknown"
            ytick.set_color(PHENOTYPE_COLORS.get(ptype, "black"))

    # Panel [1,1]: Immune phenotype donut
    ax = axes[1, 1]
    if "immune_phenotype" in metrics.columns:
        phenotype_counts = metrics["immune_phenotype"].value_counts()
        colors = [PHENOTYPE_COLORS.get(p, "#aaaaaa") for p in phenotype_counts.index]
        wedges, texts, autotexts = ax.pie(
            phenotype_counts.values, labels=phenotype_counts.index,
            colors=colors, autopct="%1.0f%%", pctdistance=0.75, startangle=90,
            wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
        )
        for t in texts: t.set_fontsize(10); t.set_fontweight("bold")
        for at in autotexts: at.set_fontsize(9)
        ax.set_title(f"Immune Phenotype Distribution (n={len(metrics)} tLung samples)",
                     fontsize=10, fontweight="bold")
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
    logger.info(f"  GSE131907 TME overview → {out_png}")


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2: ssGSEA deconvolution on TCGA bulk RNA-seq
# ══════════════════════════════════════════════════════════════════════════════

def _ssgsea_score(expr_ranked: pd.Series, gene_set: list) -> float:
    """
    Single-sample GSEA enrichment score (Barbie et al. 2009).
    expr_ranked: Series indexed by gene symbol, sorted descending by expression.
    Returns normalized enrichment score in [-1, 1].
    """
    genes = [g for g in gene_set if g in expr_ranked.index]
    if len(genes) < 2:
        return 0.0

    n = len(expr_ranked)
    n_hit = len(genes)
    hit_set = set(genes)

    # Rank positions (0-based)
    ranks = {g: i for i, g in enumerate(expr_ranked.index)}

    # Running sum
    running = 0.0
    max_dev = 0.0
    min_dev = 0.0

    for i, gene in enumerate(expr_ranked.index):
        if gene in hit_set:
            running += 1.0 / n_hit
        else:
            running -= 1.0 / (n - n_hit)
        if running > max_dev:
            max_dev = running
        if running < min_dev:
            min_dev = running

    es = max_dev if abs(max_dev) >= abs(min_dev) else min_dev
    return round(es, 6)


def run_ssgsea_patient(sample_id: str, logger) -> dict | None:
    """
    Load one patient's gene expression, run ssGSEA for all TME cell types.
    Returns dict {cell_type: score} or None if expression file not found.
    """
    expr_path = EXPR_DIR / sample_id / f"{sample_id}_gene_expression.tsv.gz"
    if not expr_path.exists():
        return None

    expr = pd.read_csv(expr_path, sep="\t", compression="gzip",
                       usecols=["SYMBOL", "TPM_GENE"], low_memory=False)
    expr = expr.dropna(subset=["SYMBOL", "TPM_GENE"])
    expr = expr.drop_duplicates(subset="SYMBOL")

    # Rank by TPM (descending)
    expr_ranked = expr.set_index("SYMBOL")["TPM_GENE"].sort_values(ascending=False)

    scores = {}
    for cell_type, gene_set in TME_SIGNATURES.items():
        scores[cell_type] = _ssgsea_score(expr_ranked, gene_set)

    return scores


def classify_phenotype_ssgsea(zscores: dict) -> str:
    """
    Classify immune phenotype from z-score-normalized ssGSEA scores.
    Z-scores are normalized across patients (column-wise), so values reflect
    relative enrichment vs. the cohort median.

    Inflamed:  CD8 z-score high (>0.3) AND CD8 > suppressive cells
    Desert:    all immune z-scores below average (<-0.3)
    Excluded:  stromal/suppressive cells elevated relative to CD8
    """
    cd8  = zscores.get("CD8_T_cytotoxic", 0)
    treg = zscores.get("Treg", 0)
    m2   = zscores.get("Macrophage_M2", 0)
    fib  = zscores.get("Fibroblast", 0)
    nk   = zscores.get("NK", 0)
    b    = zscores.get("B_cell", 0)
    cd4  = zscores.get("CD4_T", 0)

    # Overall immune infiltration (z-score average of effector cells)
    immune_z = (cd8 + nk + b + cd4) / 4.0

    if immune_z < -0.5:
        return "Desert"
    elif cd8 > 0.2 and cd8 >= treg and cd8 >= m2:
        return "Inflamed"
    elif treg > cd8 + 0.3 or m2 > cd8 + 0.3 or fib > cd8 + 0.5:
        return "Excluded"
    else:
        return "Inflamed"


def run_stage2(logger, sample_id: str | None = None):
    """Run ssGSEA deconvolution for all TCGA patients (or one sample)."""
    import gseapy  # noqa: F401 — confirm available

    # Discover samples
    if sample_id:
        samples = [sample_id]
    else:
        samples = sorted([
            d.name for d in EXPR_DIR.iterdir()
            if d.is_dir() and d.name.startswith("TCGA")
        ])

    logger.info(f"Stage 2: ssGSEA deconvolution — {len(samples)} patients")
    logger.info(f"  Cell-type signatures: {list(TME_SIGNATURES.keys())}")

    cohort_rows = []
    n_ok = 0

    for i, sid in enumerate(samples):
        scores = run_ssgsea_patient(sid, logger)
        if scores is None:
            continue

        scores["sample_id"] = sid
        cohort_rows.append(scores)
        n_ok += 1

        if (i + 1) % 50 == 0:
            logger.info(f"  Processed {i+1}/{len(samples)} ...")

    if not cohort_rows:
        logger.warning("No expression files found — Stage 2 skipped")
        return None

    # ── Z-score normalize each cell-type column across the cohort ──────────────
    # Raw ssGSEA scores differ in absolute magnitude (signature size effect).
    # Column-wise z-scoring makes scores comparable for phenotype classification.
    cell_cols = list(TME_SIGNATURES.keys())
    cohort_df = pd.DataFrame(cohort_rows).set_index("sample_id")
    means = cohort_df[cell_cols].mean()
    stds  = cohort_df[cell_cols].std().replace(0, 1)
    z_df  = (cohort_df[cell_cols] - means) / stds

    # Add phenotype based on z-scores
    def _phenotype(row):
        return classify_phenotype_ssgsea(row.to_dict())
    cohort_df["immune_phenotype"] = z_df.apply(_phenotype, axis=1)

    cohort_out = OUT_DIR / "tme_cohort_scores.tsv"
    cohort_df.to_csv(cohort_out, sep="\t")
    logger.info(f"  Cohort TME scores → {cohort_out} ({n_ok} patients)")

    # Write per-patient files with raw scores + phenotype
    for sid in cohort_df.index:
        sample_out_dir = OUT_DIR / sid
        sample_out_dir.mkdir(parents=True, exist_ok=True)
        row_df = cohort_df.loc[[sid]]
        row_df.to_csv(sample_out_dir / f"{sid}_tme_scores.tsv", sep="\t")

    # Phenotype distribution
    phenotype_counts = cohort_df["immune_phenotype"].value_counts()
    logger.info("  TCGA-LUAD immune phenotype distribution:")
    for ptype, n in phenotype_counts.items():
        logger.info(f"    {ptype}: {n} ({100*n/n_ok:.1f}%)")

    return cohort_df


def plot_cohort_tme(cohort_df: pd.DataFrame, out_dir: Path, logger):
    """Two-panel figure: cohort TME heatmap + phenotype distribution."""
    score_cols = [c for c in TME_SIGNATURES.keys() if c in cohort_df.columns]
    scores = cohort_df[score_cols].copy()

    # Sort patients by CD8_T_cytotoxic score
    if "CD8_T_cytotoxic" in scores.columns:
        scores = scores.sort_values("CD8_T_cytotoxic", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8),
                             gridspec_kw={"width_ratios": [3, 1]})
    fig.patch.set_facecolor("#f8f9fa")
    fig.suptitle(f"TCGA-LUAD TME Deconvolution — ssGSEA (n={len(scores)} patients)",
                 fontsize=13, fontweight="bold")

    # Panel 1: heatmap (samples × cell types)
    ax = axes[0]
    norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    im = ax.imshow(norm.values.T, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1)
    ax.set_yticks(range(len(score_cols)))
    ax.set_yticklabels([c.replace("_", " ") for c in score_cols], fontsize=9)
    ax.set_xlabel(f"Patients (n={len(scores)}, sorted by CD8 cytotoxic score)", fontsize=9)
    ax.set_title("TME Cell-Type Scores (ssGSEA)", fontsize=11, fontweight="bold")
    ax.set_xticks([])
    plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05,
                 shrink=0.6, label="Normalized ssGSEA score")

    # Color-code y-axis labels by cell type
    for ytick, ct in zip(ax.get_yticklabels(), score_cols):
        ytick.set_color(CELL_TYPE_COLORS.get(ct, "#333333"))

    # Phenotype color bar on top
    if "immune_phenotype" in cohort_df.columns:
        phenotypes = cohort_df.loc[scores.index, "immune_phenotype"].values
        ptype_colors = [PHENOTYPE_COLORS.get(p, "#aaaaaa") for p in phenotypes]
        for xi, c in enumerate(ptype_colors):
            ax.add_patch(plt.Rectangle((xi - 0.5, -1.5), 1, 1,
                                       color=c, transform=ax.transData, clip_on=False))
        legend_handles = [mpatches.Patch(color=c, label=p)
                          for p, c in PHENOTYPE_COLORS.items()]
        ax.legend(handles=legend_handles, fontsize=8, loc="upper right",
                  title="Immune phenotype", framealpha=0.9)

    # Panel 2: phenotype donut
    ax2 = axes[1]
    if "immune_phenotype" in cohort_df.columns:
        pcounts = cohort_df["immune_phenotype"].value_counts()
        colors = [PHENOTYPE_COLORS.get(p, "#aaaaaa") for p in pcounts.index]
        wedges, texts, autotexts = ax2.pie(
            pcounts.values, labels=pcounts.index, colors=colors,
            autopct="%1.0f%%", pctdistance=0.75, startangle=90,
            wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
        )
        for t in texts: t.set_fontsize(10); t.set_fontweight("bold")
        for at in autotexts: at.set_fontsize(9)
        ax2.set_title("Immune Phenotype\nDistribution", fontsize=11, fontweight="bold")
        legend_labels = [f"{p} (n={n})" for p, n in pcounts.items()]
        handles = [mpatches.Patch(color=PHENOTYPE_COLORS.get(p, "#aaa"), label=lbl)
                   for p, lbl in zip(pcounts.index, legend_labels)]
        ax2.legend(handles=handles, fontsize=9, loc="lower center",
                   bbox_to_anchor=(0.5, -0.15), ncol=1)
    else:
        ax2.axis("off")

    plt.tight_layout()
    out_png = out_dir / "tme_cohort_heatmap.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"  Cohort TME heatmap → {out_png}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="LUAD TME characterization (2-stage)")
    parser.add_argument("--dry_run",  action="store_true", help="Validate inputs only")
    parser.add_argument("--skip_sc",  action="store_true",
                        help="Skip Stage 1 (GSE131907 analysis)")
    parser.add_argument("--sample",   type=str, default=None,
                        help="Run Stage 2 for a single sample only")
    args = parser.parse_args()

    logger = get_logger("luad-tme")
    logger.info(f"\n{'='*55}")
    logger.info("Module 04: TME Characterization (GSE131907 + ssGSEA)")

    if args.dry_run:
        logger.info(f"[DRY RUN] GSE131907 annotation: {'FOUND' if ANN_FILE.exists() else 'MISSING'}")
        n_expr = sum(1 for d in EXPR_DIR.iterdir()
                     if d.is_dir() and (d / f"{d.name}_gene_expression.tsv.gz").exists()) \
                 if EXPR_DIR.exists() else 0
        logger.info(f"[DRY RUN] TCGA expression files available: {n_expr}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: GSE131907 reference ─────────────────────────────────────────
    if not args.skip_sc and not args.sample:
        if ANN_FILE.exists():
            logger.info("\n--- Stage 1: GSE131907 single-cell reference ---")
            ann = load_annotation(logger)
            logger.info("Mapping cell subtypes to lineages...")
            ann = map_lineage(ann)
            logger.info("Computing TME fractions...")
            fractions = compute_tme_fractions(ann, logger)
            fractions.to_csv(OUT_DIR / "per_sample_tme_fractions.tsv", sep="\t")
            logger.info("Computing immune metrics...")
            metrics = compute_immune_metrics(fractions, ann)
            metrics.to_csv(OUT_DIR / "per_sample_immune_metrics.tsv", sep="\t")
            summary = compute_dataset_summary(metrics, logger)
            summary.to_csv(OUT_DIR / "luad_tme_summary.tsv", sep="\t")
            logger.info("Generating GSE131907 TME overview figure...")
            plot_tme_overview(fractions, metrics, OUT_DIR, logger)
        else:
            logger.warning(f"GSE131907 annotation not found — Stage 1 skipped")
            logger.warning(f"  Expected: {ANN_FILE}")

    # ── Stage 2: ssGSEA on TCGA ──────────────────────────────────────────────
    logger.info("\n--- Stage 2: ssGSEA deconvolution on TCGA bulk RNA-seq ---")
    try:
        cohort_df = run_stage2(logger, sample_id=args.sample)
        if cohort_df is not None and not args.sample:
            logger.info("Generating cohort TME heatmap...")
            plot_cohort_tme(cohort_df, OUT_DIR, logger)
    except ImportError:
        logger.error("gseapy not installed. Run: pip install gseapy")
        raise

    logger.info(f"\n[Module 04 Complete] Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
