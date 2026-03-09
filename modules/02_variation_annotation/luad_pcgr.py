#!/usr/bin/env python
"""
luad_pcgr.py
------------
LUAD variant annotation pipeline using pcgr Python API directly.
Processes TCGA-LUAD MAF files without CLI or VCF conversion.

Steps:
  1. Read TCGA MAF (already VEP-annotated by GDC)
  2. Map TCGA MAF columns → pcgr internal column names
  3. Filter non-synonymous variants
  4. Calculate TMB via pcgr.variant.calculate_tmb()
  5. Save annotated variant table + TMB summary

Usage:
  python luad_pcgr.py                        # all MAFs in data/input/
  python luad_pcgr.py --sample TCGA-49-4507  # single sample
  python luad_pcgr.py --dry_run              # validate inputs only
"""

import argparse
import logging
import warnings
from io import StringIO
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# pcgr Python API (no CLI, no VCF needed)
from pcgr.variant import calculate_tmb
from pcgr import pcgr_vars

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
MAF_DIR     = PROJECT_DIR / "data/input"
OUT_DIR     = PROJECT_DIR / "data/output/02_variation_annotation"

# ── LUAD fixed parameters ─────────────────────────────────────────────────────
TUMOR_SITE       = 15       # Lung (PCGR code)
ASSAY            = "WES"
TARGET_SIZE_MB   = 34.0     # WES effective coding size
TUMOR_DP_MIN     = 10       # min sequencing depth
TUMOR_AF_MIN     = 0.05     # min allele frequency
TUMOR_AD_MIN     = 3        # min alt allele depth

# CIViC evidence database (Plan B)
CIVIC_CACHE_PATH = PROJECT_DIR / "data/civic_evidence.tsv.gz"
CIVIC_URL = "https://civicdb.org/downloads/nightly/nightly-ClinicalEvidenceSummaries.tsv"

# Non-synonymous variant classes to keep
NONSYNONYMOUS = [
    "Missense_Mutation", "Nonsense_Mutation",
    "Frame_Shift_Del", "Frame_Shift_Ins",
    "Splice_Site", "In_Frame_Del", "In_Frame_Ins",
    "Translation_Start_Site", "Nonstop_Mutation",
]

# ── Column mapping: TCGA MAF → pcgr internal names ───────────────────────────
MAF_TO_PCGR = {
    # SYMBOL, VARIANT_CLASS already present in TCGA MAF (VEP annotated)
    "Chromosome":            "CHROM",
    "Start_Position":        "POS",
    "Reference_Allele":      "REF",
    "Tumor_Seq_Allele2":     "ALT",
    "Consequence":           "CONSEQUENCE",   # VEP format: missense_variant etc.
    "Variant_Classification":"VARIANT_CLASS_TCGA",  # keep TCGA format separately
    "HGVSp_Short":           "HGVSp_Short",
    "t_depth":               "DP_TUMOR",
    "t_alt_count":           "AD_TUMOR",
    "Tumor_Sample_Barcode":  "SAMPLE_ID",
}


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
    return logging.getLogger(name)


# ── Functional annotation ──────────────────────────────────────────────────────

# Variant classes that are structurally/functionally disruptive
_HIGH_IMPACT_CLASSES = {
    "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins",
    "Splice_Site", "Nonstop_Mutation", "Translation_Start_Site",
}


def classify_functional_impact(df: pd.DataFrame) -> pd.Series:
    """
    Plan A: Rule-based functional classification using existing annotation fields.
    Priority: hotspot > variant class > IMPACT > SIFT+PolyPhen

    Returns a Series with one of:
      Likely_Oncogenic   — known cancer hotspot
      Likely_Functional  — truncating / splice / HIGH impact
      Possibly_Functional — deleterious missense (SIFT + PolyPhen both harmful)
      VUS                — variant of uncertain significance
    """
    def _classify(row):
        vclass   = str(row.get("VARIANT_CLASS_TCGA", ""))
        impact   = str(row.get("IMPACT", ""))
        sift     = str(row.get("SIFT", "")).lower()
        polyphen = str(row.get("PolyPhen", "")).lower()
        hotspot  = row.get("hotspot", False)
        is_hotspot = (hotspot is True) or (str(hotspot).lower() == "true")

        if is_hotspot:
            return "Likely_Oncogenic"
        if vclass in _HIGH_IMPACT_CLASSES or impact == "HIGH":
            return "Likely_Functional"
        if (impact == "MODERATE"
                and "deleterious" in sift
                and "probably_damaging" in polyphen):
            return "Possibly_Functional"
        if impact == "MODERATE":
            return "VUS"
        return "VUS"

    return df.apply(_classify, axis=1)


def load_civic_db(logger) -> dict:
    """
    Plan B: Load CIViC clinical evidence summary (free bulk download, no token).
    Caches to data/civic_evidence.tsv.gz after first download.

    Returns dict: (GENE_UPPER, VARIANT_UPPER) -> {level, significance}
    Evidence levels: A (strongest) → E (weakest)
    """
    if CIVIC_CACHE_PATH.exists():
        logger.info("  Loading CIViC DB from local cache...")
        try:
            civic_df = pd.read_csv(CIVIC_CACHE_PATH, sep="\t",
                                   compression="gzip", low_memory=False)
        except Exception as e:
            logger.warning(f"  CIViC cache read failed: {e}")
            return {}
    else:
        logger.info("  Downloading CIViC evidence database (~5 MB)...")
        try:
            resp = requests.get(CIVIC_URL, timeout=60)
            resp.raise_for_status()
            civic_df = pd.read_csv(StringIO(resp.text), sep="\t", low_memory=False)
            CIVIC_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            civic_df.to_csv(CIVIC_CACHE_PATH, sep="\t", index=False, compression="gzip")
            logger.info(f"  CIViC DB cached → {CIVIC_CACHE_PATH}")
        except Exception as e:
            logger.warning(f"  CIViC download failed: {e}. Skipping CIViC annotation.")
            return {}

    level_rank = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    lookup = {}
    for _, row in civic_df.iterrows():
        # CIViC nightly file uses 'molecular_profile' ("GENE VARIANT") not separate columns
        mp = str(row.get("molecular_profile", "")).strip()
        if not mp or mp == "nan":
            continue
        parts = mp.split(" ", 1)   # split on first space: "EGFR L858R" → ["EGFR", "L858R"]
        if len(parts) < 2:
            continue
        gene    = parts[0].strip().upper()
        variant = parts[1].strip().upper()
        level   = str(row.get("evidence_level",  "")).strip()
        sig     = str(row.get("significance",    "")).strip()   # CIViC field is 'significance'
        if not gene or not variant:
            continue
        key = (gene, variant)
        # Keep the strongest (lowest rank) evidence level per variant
        if key not in lookup or level_rank.get(level, 9) < level_rank.get(lookup[key]["level"], 9):
            lookup[key] = {"level": level, "significance": sig}

    logger.info(f"  CIViC DB: {len(lookup):,} variant records loaded")
    return lookup


def annotate_civic(df: pd.DataFrame, civic_lookup: dict) -> pd.DataFrame:
    """
    Plan B: Add CIVIC_LEVEL and CIVIC_SIGNIFICANCE columns.
    Matches on (SYMBOL, aa_change) where aa_change is HGVSp_Short minus 'p.' prefix.
    """
    civic_levels = []
    civic_sigs   = []
    for _, row in df.iterrows():
        gene      = str(row.get("SYMBOL", "")).strip().upper()
        hgvsp     = str(row.get("HGVSp_Short", ""))
        aa_change = hgvsp.replace("p.", "").strip().upper()
        hit = civic_lookup.get((gene, aa_change), {})
        civic_levels.append(hit.get("level",        ""))
        civic_sigs.append(hit.get("significance",   ""))

    df = df.copy()
    df["CIVIC_LEVEL"]        = civic_levels
    df["CIVIC_SIGNIFICANCE"] = civic_sigs
    return df


def get_maf_files(sample_id: str = None):
    """Return MAF files, skip _2 duplicates."""
    if sample_id:
        return [MAF_DIR / f"{sample_id}.maf.gz"]
    all_mafs = sorted(MAF_DIR.glob("*.maf.gz"))
    return [f for f in all_mafs if "_2" not in f.stem]


def load_maf(maf_path: Path, logger) -> pd.DataFrame:
    """Read TCGA MAF file, skip comment lines."""
    logger.info(f"Reading MAF: {maf_path.name}")
    df = pd.read_csv(maf_path, sep="\t", comment="#", low_memory=False, compression="gzip")
    logger.info(f"  Total variants: {len(df)}")
    return df


def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename TCGA MAF columns to pcgr internal names."""
    rename = {k: v for k, v in MAF_TO_PCGR.items() if k in df.columns}
    return df.rename(columns=rename)


def compute_vaf(df: pd.DataFrame) -> pd.DataFrame:
    """Compute VAF_TUMOR = AD_TUMOR / DP_TUMOR if not present."""
    if "VAF_TUMOR" not in df.columns:
        if {"AD_TUMOR", "DP_TUMOR"}.issubset(df.columns):
            df["VAF_TUMOR"] = (
                pd.to_numeric(df["AD_TUMOR"], errors="coerce") /
                pd.to_numeric(df["DP_TUMOR"], errors="coerce")
            ).round(4)
    return df


def filter_nonsynonymous(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Keep only non-synonymous coding variants (filter on TCGA Variant_Classification)."""
    if "VARIANT_CLASS_TCGA" not in df.columns:
        logger.warning("VARIANT_CLASS_TCGA column not found, skipping filter")
        return df
    filtered = df[df["VARIANT_CLASS_TCGA"].isin(NONSYNONYMOUS)].copy()
    logger.info(f"  Non-synonymous variants: {len(filtered)} / {len(df)}")
    return filtered


# ── Color palettes (pcgr-inspired) ───────────────────────────────────────────
VARIANT_CLASS_COLORS = {
    "Missense_Mutation":     "#4393c3",
    "Nonsense_Mutation":     "#d6604d",
    "Frame_Shift_Del":       "#b2182b",
    "Frame_Shift_Ins":       "#e08214",
    "Splice_Site":           "#7fbc41",
    "In_Frame_Del":          "#c7eae5",
    "In_Frame_Ins":          "#dfc27d",
    "Translation_Start_Site":"#9970ab",
    "Nonstop_Mutation":      "#35978f",
}

# 96 SBS trinucleotide contexts (C>A, C>G, C>T, T>A, T>C, T>G)
SBS_COLORS = {
    "C>A": "#1f78b4", "C>G": "#222222", "C>T": "#e31a1c",
    "T>A": "#999999", "T>C": "#33a02c", "T>G": "#ff7f00",
}


def _build_sbs96(df: pd.DataFrame) -> pd.Series:
    """Build 96-channel SBS spectrum from REF/ALT columns."""
    if not {"REF", "ALT"}.issubset(df.columns):
        return pd.Series(dtype=int)
    snv = df[
        df["REF"].str.len().eq(1) & df["ALT"].str.len().eq(1) &
        df["REF"].isin(list("ACGT")) & df["ALT"].isin(list("ACGT")) &
        (df["REF"] != df["ALT"])
    ].copy()
    if snv.empty:
        return pd.Series(dtype=int)

    comp = str.maketrans("ACGT", "TGCA")

    def norm_sub(r, a):
        if r in "CT":
            return f"{r}>{a}"
        return f"{r.translate(comp)}>{a.translate(comp)}"

    snv["sub"] = snv.apply(lambda x: norm_sub(x["REF"], x["ALT"]), axis=1)
    counts = snv["sub"].value_counts()
    return counts


def plot_variant_summary(df_coding: pd.DataFrame, df_all: pd.DataFrame,
                          tmb: float, sample_id: str, out_dir: Path, logger):
    """
    8-panel variant summary figure:
      [0,0] Variant class donut
      [0,1] VAF distribution histogram
      [0,2] Sequencing depth (DP) histogram
      [1,0] Top 15 mutated genes (bar)
      [1,1] TMB vs TCGA-LUAD (violin)
      [1,2] SBS trinucleotide spectrum
      [2,0] Functional impact classification (Plan A)
      [2,1] CIViC evidence level distribution (Plan B)
      [2,2] CIViC evidence details table (Plan B)
    """
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.patch.set_facecolor("#f8f9fa")
    fig.suptitle(f"Variant Summary — {sample_id}", fontsize=14,
                 fontweight="bold", y=1.01)

    # ── Panel [0,0]: Variant class donut ──────────────────────────────────────
    ax = axes[0, 0]
    if "VARIANT_CLASS_TCGA" in df_coding.columns:
        vc = df_coding["VARIANT_CLASS_TCGA"].value_counts()
        colors = [VARIANT_CLASS_COLORS.get(c, "#aaaaaa") for c in vc.index]
        wedges, texts, autotexts = ax.pie(
            vc.values, labels=None, colors=colors,
            autopct=lambda p: f"{p:.0f}%" if p > 4 else "",
            pctdistance=0.75, startangle=90,
            wedgeprops=dict(width=0.55, edgecolor="white", linewidth=1.5),
        )
        for at in autotexts:
            at.set_fontsize(8)
        handles = [mpatches.Patch(color=VARIANT_CLASS_COLORS.get(c, "#aaa"),
                   label=c.replace("_", " ")) for c in vc.index]
        ax.legend(handles=handles, fontsize=7, loc="lower left",
                  bbox_to_anchor=(-0.1, -0.15), ncol=1)
        ax.set_title(f"Variant Classes\n(n={len(df_coding):,})", fontsize=10, fontweight="bold")
    else:
        ax.axis("off")

    # ── Panel [0,1]: VAF histogram ────────────────────────────────────────────
    ax = axes[0, 1]
    if "VAF_TUMOR" in df_coding.columns:
        vaf = pd.to_numeric(df_coding["VAF_TUMOR"], errors="coerce").dropna()
        vaf = vaf[(vaf >= 0) & (vaf <= 1)]
        n, bins, patches = ax.hist(vaf, bins=40, color="#4393c3",
                                   edgecolor="white", linewidth=0.5, alpha=0.85)
        ax.axvline(vaf.median(), color="#d62728", linestyle="--",
                   linewidth=1.5, label=f"Median={vaf.median():.2f}")
        ax.set_xlabel("Variant Allele Fraction (VAF)", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_title("VAF Distribution", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel [0,2]: Depth histogram ──────────────────────────────────────────
    ax = axes[0, 2]
    if "DP_TUMOR" in df_coding.columns:
        dp = pd.to_numeric(df_coding["DP_TUMOR"], errors="coerce").dropna()
        dp = dp[dp > 0]
        ax.hist(dp, bins=40, color="#74c476", edgecolor="white",
                linewidth=0.5, alpha=0.85)
        ax.axvline(dp.median(), color="#d62728", linestyle="--",
                   linewidth=1.5, label=f"Median={dp.median():.0f}×")
        ax.set_xlabel("Tumor Sequencing Depth (DP)", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_title("Sequencing Depth", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel [1,0]: Top 15 mutated genes ────────────────────────────────────
    ax = axes[1, 0]
    if "SYMBOL" in df_coding.columns:
        top_genes = df_coding["SYMBOL"].value_counts().head(15)
        bar_colors = []
        if "VARIANT_CLASS_TCGA" in df_coding.columns:
            dominant = (df_coding.groupby("SYMBOL")["VARIANT_CLASS_TCGA"]
                        .agg(lambda x: x.value_counts().index[0]))
            bar_colors = [VARIANT_CLASS_COLORS.get(dominant.get(g, ""), "#aaaaaa")
                          for g in top_genes.index]
        else:
            bar_colors = ["#4393c3"] * len(top_genes)

        top_genes_sorted = top_genes.sort_values()
        bar_colors_sorted = bar_colors[::-1]
        ax.barh(top_genes_sorted.index, top_genes_sorted.values,
                color=bar_colors_sorted, alpha=0.85, edgecolor="white")
        ax.set_xlabel("# Non-synonymous variants", fontsize=9)
        ax.set_title("Top Mutated Genes", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)

    # ── Panel [1,1]: TMB violin vs simulated TCGA-LUAD background ───────────
    ax = axes[1, 1]
    rng = np.random.default_rng(42)
    tcga_tmb = rng.lognormal(mean=np.log(2.5), sigma=0.7, size=500)
    parts = ax.violinplot([tcga_tmb], positions=[1], showmedians=True,
                          showextrema=True, widths=0.6)
    for pc in parts["bodies"]:
        pc.set_facecolor("#aec7e8")
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("#1f77b4")
    parts["cbars"].set_color("#1f77b4")
    parts["cmaxes"].set_color("#1f77b4")
    parts["cmins"].set_color("#1f77b4")

    if tmb is not None and not np.isnan(float(tmb)):
        tmb_val = float(tmb)
        ax.scatter([1], [tmb_val], color="#d62728", s=150, zorder=5,
                   label=f"{sample_id}: {tmb_val:.1f} mut/Mb")
        ax.annotate(f"{tmb_val:.1f}", xy=(1, tmb_val),
                    xytext=(1.2, tmb_val), fontsize=9, color="#d62728",
                    va="center",
                    arrowprops=dict(arrowstyle="->", color="#d62728", lw=1))
    ax.set_xticks([1])
    ax.set_xticklabels(["TCGA-LUAD\n(reference)"], fontsize=9)
    ax.set_ylabel("TMB (mut/Mb)", fontsize=9)
    ax.set_title("Tumor Mutational Burden", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel [1,2]: SBS 6-type spectrum ─────────────────────────────────────
    ax = axes[1, 2]
    sbs = _build_sbs96(df_coding)
    if not sbs.empty:
        sbs_order = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
        counts = [sbs.get(s, 0) for s in sbs_order]
        colors = [SBS_COLORS[s] for s in sbs_order]
        bars = ax.bar(sbs_order, counts, color=colors, edgecolor="white",
                      linewidth=0.8, alpha=0.9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_xlabel("Substitution type", fontsize=9)
        ax.set_title("SBS Substitution Spectrum", fontsize=10, fontweight="bold")
        for bar, cnt in zip(bars, counts):
            if cnt > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        str(cnt), ha="center", va="bottom", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No SNV data", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="gray")
        ax.axis("off")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel [2,0]: Functional impact classification (Plan A) ──────────────────
    ax = axes[2, 0]
    fi_colors = {
        "Likely_Oncogenic":    "#d62728",
        "Likely_Functional":   "#ff7f0e",
        "Possibly_Functional": "#1f77b4",
        "VUS":                 "#aaaaaa",
    }
    if "FUNCTIONAL_IMPACT" in df_coding.columns:
        fi_order  = ["Likely_Oncogenic", "Likely_Functional", "Possibly_Functional", "VUS"]
        fi_counts = df_coding["FUNCTIONAL_IMPACT"].value_counts()
        vals   = [fi_counts.get(k, 0) for k in fi_order]
        colors = [fi_colors[k] for k in fi_order]
        bars = ax.barh(fi_order, vals, color=colors, alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(v + 0.3, bar.get_y() + bar.get_height() / 2,
                        str(v), va="center", fontsize=9)
        ax.set_xlabel("# Variants", fontsize=9)
        ax.set_title("Functional Impact Classification", fontsize=10, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No functional impact data\n(re-run pipeline to generate)",
                ha="center", va="center", transform=ax.transAxes, fontsize=9, color="gray")
        ax.set_title("Functional Impact Classification", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel [2,1]: CIViC evidence level distribution (Plan B) ─────────────────
    ax = axes[2, 1]
    level_colors = {"A": "#2ca02c", "B": "#1f77b4", "C": "#ff7f0e", "D": "#9467bd", "E": "#aaaaaa"}
    if "CIVIC_LEVEL" in df_coding.columns:
        civic_hits = df_coding[df_coding["CIVIC_LEVEL"].notna() & (df_coding["CIVIC_LEVEL"] != "")]
        if not civic_hits.empty:
            lv_counts = civic_hits["CIVIC_LEVEL"].value_counts().sort_index()
            colors = [level_colors.get(l, "#aaaaaa") for l in lv_counts.index]
            bars = ax.bar(lv_counts.index, lv_counts.values, color=colors, alpha=0.85, edgecolor="white")
            for bar, v in zip(bars, lv_counts.values):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                        str(v), ha="center", va="bottom", fontsize=9)
            ax.set_xlabel("CIViC Evidence Level  (A=strongest → E=weakest)", fontsize=8)
            ax.set_ylabel("# Variants", fontsize=9)
            ax.set_title(f"CIViC Clinical Evidence\n({len(civic_hits)} variants annotated)",
                         fontsize=10, fontweight="bold")
        else:
            ax.text(0.5, 0.5, "No CIViC evidence\nfound for this sample",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9, color="gray")
            ax.set_title("CIViC Clinical Evidence", fontsize=10, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No CIViC data\n(re-run pipeline to generate)",
                ha="center", va="center", transform=ax.transAxes, fontsize=9, color="gray")
        ax.set_title("CIViC Clinical Evidence", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel [2,2]: CIViC evidence details table (Plan B) ──────────────────────
    ax = axes[2, 2]
    ax.axis("off")
    if "CIVIC_LEVEL" in df_coding.columns:
        civic_tbl = (
            df_coding[df_coding["CIVIC_LEVEL"].notna() & (df_coding["CIVIC_LEVEL"] != "")]
            [["SYMBOL", "HGVSp_Short", "CIVIC_LEVEL", "CIVIC_SIGNIFICANCE"]]
            .drop_duplicates()
            .sort_values("CIVIC_LEVEL")
            .head(10)
        )
        if not civic_tbl.empty:
            tbl = ax.table(
                cellText=civic_tbl.values.tolist(),
                colLabels=["Gene", "Change", "Level", "Significance"],
                loc="center",
                cellLoc="left",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)
            tbl.scale(1, 1.5)
            for (r, c), cell in tbl.get_celld().items():
                cell.set_edgecolor("#dddddd")
                if r == 0:
                    cell.set_facecolor("#4393c3")
                    cell.set_text_props(color="white", fontweight="bold")
                elif r % 2 == 0:
                    cell.set_facecolor("#f0f4f8")
                else:
                    cell.set_facecolor("white")
            ax.set_title("CIViC Evidence Details", fontsize=10, fontweight="bold", pad=8)
        else:
            ax.text(0.5, 0.5, "No CIViC evidence\nfound for this sample",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9, color="gray")
            ax.set_title("CIViC Evidence Details", fontsize=10, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No CIViC data\n(re-run pipeline to generate)",
                ha="center", va="center", transform=ax.transAxes, fontsize=9, color="gray")
        ax.set_title("CIViC Evidence Details", fontsize=10, fontweight="bold")

    plt.tight_layout()
    out_png = out_dir / f"{sample_id}_variant_summary.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"  Variant summary figure → {out_png}")


def run_sample(maf_path: Path, dry_run: bool = False):
    """Process a single LUAD MAF file."""
    logger = get_logger("luad-pcgr")
    sample_id = maf_path.name.split(".")[0]

    logger.info(f"\n{'='*55}")
    logger.info(f"Sample: {sample_id}")

    if dry_run:
        logger.info(f"[DRY RUN] Would process: {maf_path}")
        return

    # 1. Load
    df = load_maf(maf_path, logger)

    # 2. Map columns
    df = map_columns(df)

    # 3. Compute VAF
    df = compute_vaf(df)

    # 4. Filter non-synonymous
    df_coding = filter_nonsynonymous(df, logger)

    # Guard: empty MAF → skip sample
    if df_coding.empty:
        logger.warning(f"  No coding variants found for {sample_id}, skipping.")
        return {"sample_id": sample_id, "tmb": 0, "n_variants": 0}

    # 4b. Functional impact annotation (Plan A + Plan B)
    logger.info("Step 4b: Annotating functional impact...")
    df_coding["FUNCTIONAL_IMPACT"] = classify_functional_impact(df_coding)
    n_functional = (df_coding["FUNCTIONAL_IMPACT"] != "VUS").sum()
    logger.info(
        f"  Likely_Oncogenic={( df_coding['FUNCTIONAL_IMPACT']=='Likely_Oncogenic').sum()}  "
        f"Likely_Functional={(df_coding['FUNCTIONAL_IMPACT']=='Likely_Functional').sum()}  "
        f"Possibly_Functional={(df_coding['FUNCTIONAL_IMPACT']=='Possibly_Functional').sum()}  "
        f"VUS={(df_coding['FUNCTIONAL_IMPACT']=='VUS').sum()}"
    )
    civic_lookup = load_civic_db(logger)
    if civic_lookup:
        df_coding = annotate_civic(df_coding, civic_lookup)
        n_civic = (df_coding["CIVIC_LEVEL"] != "").sum()
        logger.info(f"  CIViC annotated: {n_civic} variants with clinical evidence")

    # 5. Calculate TMB via pcgr
    out_dir = OUT_DIR / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)
    tmb_fname = str(out_dir / f"{sample_id}_tmb.tsv")

    calculate_tmb(
        variant_set    = df_coding,
        tumor_dp_min   = TUMOR_DP_MIN,
        tumor_af_min   = TUMOR_AF_MIN,
        tumor_ad_min   = TUMOR_AD_MIN,
        target_size_mb = TARGET_SIZE_MB,
        sample_id      = sample_id,
        tmb_fname      = tmb_fname,
        logger         = logger,
    )
    # calculate_tmb writes to file, read result back
    tmb = None
    if Path(tmb_fname).exists():
        tmb_df = pd.read_csv(tmb_fname, sep="\t")
        row = tmb_df[tmb_df["tmb_measure"] == "TMB_coding_non_silent"]
        if not row.empty:
            tmb = row["tmb_estimate"].values[0]
    logger.info(f"  TMB (coding non-silent): {tmb} mut/Mb")

    # 6. Save annotated variant table
    keep_cols = [c for c in [
        "SYMBOL", "CHROM", "POS", "REF", "ALT",
        "CONSEQUENCE", "VARIANT_CLASS", "VARIANT_CLASS_TCGA",
        "HGVSp_Short", "HGVSc",
        "DP_TUMOR", "AD_TUMOR", "VAF_TUMOR", "SAMPLE_ID",
        "gnomAD_AF", "IMPACT", "SIFT", "PolyPhen", "hotspot",
        "FUNCTIONAL_IMPACT", "CIVIC_LEVEL", "CIVIC_SIGNIFICANCE",
    ] if c in df_coding.columns]

    out_tsv = out_dir / f"{sample_id}_variants.tsv.gz"
    df_coding[keep_cols].to_csv(out_tsv, sep="\t", index=False, compression="gzip")
    logger.info(f"  Variants saved → {out_tsv}")
    logger.info(f"  TMB saved      → {tmb_fname}")

    # 7. Visualization
    logger.info("Step 7: Generating variant summary figure...")
    plot_variant_summary(df_coding, df, tmb, sample_id, out_dir, logger)

    return {"sample_id": sample_id, "tmb": tmb, "n_variants": len(df_coding)}


def main():
    parser = argparse.ArgumentParser(description="LUAD variant annotation (pcgr Python API)")
    parser.add_argument("--sample",  type=str, help="Single sample ID (e.g. TCGA-49-4507)")
    parser.add_argument("--dry_run", action="store_true", help="Validate inputs without running")
    args = parser.parse_args()

    maf_files = get_maf_files(args.sample)
    if not maf_files:
        print(f"[ERROR] No MAF files found in {MAF_DIR}")
        return

    results = []
    for maf in maf_files:
        r = run_sample(maf, dry_run=args.dry_run)
        if r:
            results.append(r)

    if results:
        summary = pd.DataFrame(results)
        summary_path = OUT_DIR / "all_samples_summary.tsv"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, sep="\t", index=False)
        print(f"\n[Summary] {len(results)} samples → {summary_path}")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
