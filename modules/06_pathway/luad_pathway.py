#!/usr/bin/env python
"""
luad_pathway.py
---------------
LUAD pathway analysis combining:
  A) ORA (Over-Representation Analysis) — mutated genes from module 02
  B) GSEA prerank — bulk RNA-seq expression from module 03

Steps:
  1. ORA: for each sample in 02_variants output,
     run Enrichr on non-synonymous mutated genes
  2. GSEA: for each sample in 03_expression output,
     run prerank GSEA using log2(TPM+1) as ranking metric
  3. Save enriched pathway tables + bar charts per sample

Input:
  data/output/02_variants/{sample_id}/{sample_id}_variants.tsv.gz
  data/output/03_expression/{sample_id}/{sample_id}_gene_expression.tsv.gz

Output:
  data/output/06_pathway/{sample_id}/
    {sample_id}_ora.tsv          — ORA enriched pathways (FDR < 0.05)
    {sample_id}_ora.png          — ORA dot plot
    {sample_id}_gsea.tsv         — GSEA enriched pathways (FDR < 0.25)
    {sample_id}_gsea.png         — GSEA NES bar chart

Usage:
  python luad_pathway.py                        # all available samples
  python luad_pathway.py --sample TCGA-86-A4D0  # single sample
  python luad_pathway.py --dry_run              # validate inputs only
"""

import argparse
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import gseapy as gp

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
PROJECT_DIR  = SCRIPT_DIR.parent.parent
VARIANT_DIR  = PROJECT_DIR / "data/output/02_variants"
EXPR_DIR     = PROJECT_DIR / "data/output/03_expression"
OUT_DIR      = PROJECT_DIR / "data/output/06_pathway"

# ── Gene set databases ─────────────────────────────────────────────────────────
ORA_GENE_SETS  = ["KEGG_2021_Human", "Reactome_2022"]
GSEA_GENE_SETS = ["MSigDB_Hallmark_2020", "KEGG_2021_Human"]

# LUAD-relevant pathway keywords (highlighted in green in plots)
LUAD_KEYWORDS = [
    "EGFR", "MAPK", "RAS", "PI3K", "AKT", "MTOR",
    "P53", "CELL_CYCLE", "HYPOXIA", "MYC",
    "EMT", "EPITHELIAL_MESENCHYMAL", "TGF",
    "INTERFERON", "IMMUNE", "PD", "ANGIOGENESIS",
    "WNT", "NOTCH", "KRAS",
]


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
    return logging.getLogger(name)


# ── ORA ────────────────────────────────────────────────────────────────────────

def load_mutated_genes(sample_id: str) -> list:
    """Load non-synonymous mutated SYMBOL list from module 02 output."""
    var_path = VARIANT_DIR / sample_id / f"{sample_id}_variants.tsv.gz"
    if not var_path.exists():
        return []
    df = pd.read_csv(var_path, sep="\t", compression="gzip", low_memory=False)
    if "SYMBOL" not in df.columns:
        return []
    return df["SYMBOL"].dropna().unique().tolist()


def run_ora(sample_id: str, genes: list, out_dir: Path, logger) -> pd.DataFrame:
    """Run Enrichr ORA on mutated gene list."""
    logger.info(f"  ORA: {len(genes)} mutated genes")
    all_results = []
    for gs in ORA_GENE_SETS:
        try:
            enr = gp.enrichr(
                gene_list=genes,
                gene_sets=gs,
                outdir=None,
                cutoff=0.05,
                verbose=False,
            )
            res = enr.results.copy()
            res["gene_set_db"] = gs
            all_results.append(res)
        except Exception as e:
            logger.warning(f"  ORA {gs} failed: {e}")

    if not all_results:
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)
    combined = combined[combined["Adjusted P-value"] < 0.05].sort_values("Adjusted P-value")

    out_tsv = out_dir / f"{sample_id}_ora.tsv"
    combined.to_csv(out_tsv, sep="\t", index=False)
    logger.info(f"  ORA: {len(combined)} enriched pathways → {out_tsv}")

    plot_ora(combined.head(20), sample_id, out_dir)
    return combined


def plot_ora(df: pd.DataFrame, sample_id: str, out_dir: Path):
    if df.empty:
        return
    df = df.copy()
    df["-log10(FDR)"] = -np.log10(df["Adjusted P-value"].clip(lower=1e-10))
    df["Overlap_n"] = df["Overlap"].apply(
        lambda x: int(x.split("/")[0]) if "/" in str(x) else 0
    )
    df = df.sort_values("-log10(FDR)")

    fig, ax = plt.subplots(figsize=(9, max(4, len(df) * 0.4)))
    sc = ax.scatter(
        df["-log10(FDR)"], range(len(df)),
        s=df["Overlap_n"] * 20 + 30,
        c=df["-log10(FDR)"], cmap="RdYlBu_r", alpha=0.85,
    )
    ax.set_yticks(range(len(df)))
    yticklabels = ax.set_yticklabels(df["Term"], fontsize=8)
    # Highlight LUAD-relevant
    for i, term in enumerate(df["Term"]):
        if any(kw in term.upper() for kw in LUAD_KEYWORDS):
            yticklabels[i].set_color("darkgreen")
            yticklabels[i].set_fontweight("bold")
    ax.set_xlabel("-log10(FDR)", fontsize=10)
    ax.set_title(f"ORA — {sample_id} (mutated gene enrichment)", fontsize=11)
    plt.colorbar(sc, label="-log10(FDR)")
    plt.tight_layout()
    out_png = out_dir / f"{sample_id}_ora.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


# ── GSEA ───────────────────────────────────────────────────────────────────────

def load_expression_ranking(sample_id: str) -> pd.DataFrame:
    """Load gene expression from module 03, return gene × log2TPM for ranking."""
    expr_path = EXPR_DIR / sample_id / f"{sample_id}_gene_expression.tsv.gz"
    if not expr_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(expr_path, sep="\t", compression="gzip")
    # Use SYMBOL + TPM_LOG2_GENE (already log2(TPM+1))
    df = df[["SYMBOL", "TPM_LOG2_GENE"]].dropna()
    df = df[df["SYMBOL"] != ""]
    # Deduplicate by keeping highest TPM per symbol
    df = df.sort_values("TPM_LOG2_GENE", ascending=False).drop_duplicates("SYMBOL")
    df.columns = ["gene", "score"]
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def run_gsea(sample_id: str, ranked: pd.DataFrame, out_dir: Path, logger) -> pd.DataFrame:
    """Run GSEA prerank on expression-ranked gene list."""
    logger.info(f"  GSEA: {len(ranked)} ranked genes")
    all_results = []
    for gs in GSEA_GENE_SETS:
        try:
            pre = gp.prerank(
                rnk=ranked,
                gene_sets=gs,
                outdir=None,
                min_size=10,
                max_size=500,
                permutation_num=200,
                seed=42,
                verbose=False,
            )
            res = pre.res2d.copy()
            res["gene_set_db"] = gs
            all_results.append(res)
        except Exception as e:
            logger.warning(f"  GSEA {gs} failed: {e}")

    if not all_results:
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)
    combined = combined[combined["FDR q-val"] < 0.25].sort_values("NES", ascending=False)

    # Flag LUAD-relevant
    combined["luad_relevant"] = combined["Term"].str.upper().apply(
        lambda t: any(kw in t for kw in LUAD_KEYWORDS)
    )

    out_tsv = out_dir / f"{sample_id}_gsea.tsv"
    combined.to_csv(out_tsv, sep="\t", index=False)
    logger.info(f"  GSEA: {len(combined)} enriched pathways (FDR<0.25) → {out_tsv}")

    plot_gsea(combined, sample_id, out_dir)
    return combined


def plot_gsea(df: pd.DataFrame, sample_id: str, out_dir: Path):
    if df.empty:
        return
    top = pd.concat([
        df[df["NES"] > 0].head(15),
        df[df["NES"] < 0].tail(15),
    ]).drop_duplicates().sort_values("NES")

    colors = ["#d73027" if n > 0 else "#4575b4" for n in top["NES"]]
    fig, ax = plt.subplots(figsize=(10, max(5, len(top) * 0.35)))
    ax.barh(range(len(top)), top["NES"], color=colors, alpha=0.85)
    ax.set_yticks(range(len(top)))
    yticklabels = ax.set_yticklabels(top["Term"], fontsize=8)
    for i, row in enumerate(top.itertuples()):
        if getattr(row, "luad_relevant", False):
            yticklabels[i].set_color("darkgreen")
            yticklabels[i].set_fontweight("bold")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Normalized Enrichment Score (NES)", fontsize=10)
    ax.set_title(
        f"GSEA — {sample_id}\n"
        f"Red=activated | Blue=suppressed | Green=LUAD-relevant",
        fontsize=10,
    )
    plt.tight_layout()
    out_png = out_dir / f"{sample_id}_gsea.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


# ── Per-sample runner ──────────────────────────────────────────────────────────

def run_sample(sample_id: str, logger):
    logger.info(f"\n{'='*55}")
    logger.info(f"Sample: {sample_id}")

    out_dir = OUT_DIR / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {"sample_id": sample_id}

    # ORA
    genes = load_mutated_genes(sample_id)
    if genes:
        ora_df = run_ora(sample_id, genes, out_dir, logger)
        result["n_ora_pathways"] = len(ora_df)
    else:
        logger.info("  ORA: no variant data — skipping")
        result["n_ora_pathways"] = 0

    # GSEA
    ranked = load_expression_ranking(sample_id)
    if not ranked.empty:
        gsea_df = run_gsea(sample_id, ranked, out_dir, logger)
        result["n_gsea_pathways"] = len(gsea_df)
    else:
        logger.info("  GSEA: no expression data — skipping")
        result["n_gsea_pathways"] = 0

    return result


def get_all_samples() -> list:
    """Collect all sample IDs that have variant or expression data."""
    samples = set()
    for p in VARIANT_DIR.glob("*/"):
        if p.is_dir():
            samples.add(p.name)
    for p in EXPR_DIR.glob("*/"):
        if p.is_dir():
            samples.add(p.name)
    return sorted(samples)


# ── Cross-sample pathway frequency ────────────────────────────────────────────

def build_cross_sample_summary(n_samples: int) -> pd.DataFrame:
    """
    Aggregate GSEA results across all processed samples.

    For each pathway, count:
      - n_activated : patients with NES > 0 & FDR < 0.25
      - n_suppressed: patients with NES < 0 & FDR < 0.25
    Then rank by n_activated descending.
    """
    from collections import defaultdict
    counts: dict = defaultdict(lambda: {"n_activated": 0, "n_suppressed": 0,
                                         "nes_sum": 0.0, "luad_relevant": False})

    for gsea_path in OUT_DIR.glob("*/*_gsea.tsv"):
        try:
            df = pd.read_csv(gsea_path, sep="\t")
            for _, row in df.iterrows():
                term = row.get("Term", "")
                nes  = float(row.get("NES", 0))
                fdr  = float(row.get("FDR q-val", 1))
                if fdr >= 0.25:
                    continue
                if nes > 0:
                    counts[term]["n_activated"] += 1
                else:
                    counts[term]["n_suppressed"] += 1
                counts[term]["nes_sum"] += nes
                if any(kw in term.upper() for kw in LUAD_KEYWORDS):
                    counts[term]["luad_relevant"] = True
        except Exception:
            continue

    rows = []
    for term, c in counts.items():
        n_tot = c["n_activated"] + c["n_suppressed"]
        rows.append({
            "pathway":       term,
            "n_activated":   c["n_activated"],
            "n_suppressed":  c["n_suppressed"],
            "pct_patients":  round((c["n_activated"] / n_samples) * 100, 1),
            "mean_nes":      round(c["nes_sum"] / n_tot, 3) if n_tot else 0,
            "luad_relevant": c["luad_relevant"],
        })

    df = pd.DataFrame(rows).sort_values("n_activated", ascending=False)
    df = df.reset_index(drop=True)

    # Remove "always-on" housekeeping pathways (activated in >90% of patients)
    # These are universally enriched due to absolute TPM ranking, not informative
    df = df[df["pct_patients"] <= 90].reset_index(drop=True)

    return df


def main():
    parser = argparse.ArgumentParser(description="LUAD pathway analysis (ORA + GSEA)")
    parser.add_argument("--sample",       type=str, help="Single sample ID")
    parser.add_argument("--dry_run",      action="store_true", help="Validate inputs only")
    parser.add_argument("--summarize",    action="store_true",
                        help="Only rebuild cross-sample summary from existing results")
    args = parser.parse_args()

    logger = get_logger("luad-pathway")

    if args.dry_run:
        samples = [args.sample] if args.sample else get_all_samples()
        logger.info(f"[DRY RUN] Would process {len(samples)} samples: {samples}")
        return

    if args.summarize:
        # Rebuild summary from already-processed results without re-running GSEA
        n = len(list(OUT_DIR.glob("*/*_gsea.tsv")))
        summary = build_cross_sample_summary(n_samples=max(n, 1))
        summary.to_csv(OUT_DIR / "all_samples_summary.tsv", sep="\t", index=False)
        print(f"Cross-sample summary rebuilt: {len(summary)} pathways from {n} samples")
        print(summary.head(20).to_string(index=False))
        return

    samples = [args.sample] if args.sample else get_all_samples()
    if not samples:
        print("[ERROR] No samples found in variant or expression output directories")
        return

    print(f"[INFO] Processing {len(samples)} samples")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for s in samples:
        run_sample(s, logger)

    # Build cross-sample pathway frequency summary
    summary = build_cross_sample_summary(n_samples=len(samples))
    summary.to_csv(OUT_DIR / "all_samples_summary.tsv", sep="\t", index=False)
    print(f"\n[Cross-sample summary] {len(summary)} pathways → {OUT_DIR / 'all_samples_summary.tsv'}")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
