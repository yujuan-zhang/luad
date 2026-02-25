"""
gsea_analysis.py
----------------
Gene Set Enrichment Analysis (GSEA): which pathways are activated or
suppressed based on bulk RNA-seq expression profiles (LUAD samples)?

Input:
  RNA-seq TPM files from data/rnaseq/{case_id}.tsv.gz
  Format: TargetID (Ensembl gene ID) + TPM columns per sample

Output:
  data/output/{case_id}_gsea.tsv    — enriched pathway table
  data/output/{case_id}_gsea.png    — GSEA enrichment dot plot

Key pathways highlighted for LUAD:
  - EGFR / RAS / MAPK signaling
  - PI3K / AKT / mTOR
  - Cell cycle (TP53, RB1, CDKN2A)
  - EMT (Epithelial-Mesenchymal Transition)
  - Immune / PD-L1 / IFN-gamma response

Requirements:
  pip install gseapy matplotlib pandas

Usage:
  python gsea_analysis.py --sample TCGA-49-4507
  python gsea_analysis.py --all
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import gseapy as gp

SCRIPT_DIR  = Path(__file__).parent
RNASEQ_DIR  = SCRIPT_DIR / "../../../data/rnaseq"
OUT_DIR     = SCRIPT_DIR / "../data/output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GENE_SETS   = ["KEGG_2021_Human", "Reactome_2022", "MSigDB_Hallmark_2020"]

# LUAD-relevant pathway keywords for highlighting
LUAD_KEYWORDS = [
    "EGFR", "MAPK", "RAS", "PI3K", "AKT", "MTOR",
    "CELL_CYCLE", "P53", "EMT", "EPITHELIAL_MESENCHYMAL",
    "INTERFERON", "PD", "IMMUNE", "ANGIOGENESIS", "HYPOXIA",
    "MYC", "WNT", "NOTCH", "TGF"
]


def load_expression(sample_id: str) -> pd.Series:
    """Load TPM values for a sample, return as gene→TPM Series."""
    tpm_file = RNASEQ_DIR / f"{sample_id}.tsv.gz"
    if not tpm_file.exists():
        raise FileNotFoundError(f"TPM file not found: {tpm_file}")
    df = pd.read_csv(tpm_file, sep="\t", compression="gzip")
    # Columns: TargetID, TPM
    df = df.dropna(subset=["TargetID", "TPM"])
    df = df[df["TPM"] > 0]
    return df.set_index("TargetID")["TPM"]


def run_gsea_prerank(expr: pd.Series, case_id: str) -> pd.DataFrame:
    """
    Use log2(TPM+1) as ranking metric for prerank GSEA.
    For a single sample, ranking by expression level (high → activated pathways).
    """
    ranked = np.log2(expr + 1).sort_values(ascending=False)
    ranked_df = ranked.reset_index()
    ranked_df.columns = ["gene", "score"]

    results = []
    for gene_set in GENE_SETS:
        pre_res = gp.prerank(
            rnk=ranked_df,
            gene_sets=gene_set,
            outdir=None,
            min_size=10,
            max_size=500,
            permutation_num=100,
            seed=42,
            verbose=False,
        )
        res = pre_res.res2d.copy()
        res["gene_set_db"] = gene_set
        results.append(res)

    combined = pd.concat(results, ignore_index=True)
    combined = combined[combined["FDR q-val"] < 0.25].sort_values("NES", ascending=False)

    # Flag LUAD-relevant pathways
    combined["luad_relevant"] = combined["Term"].str.upper().apply(
        lambda t: any(kw in t for kw in LUAD_KEYWORDS)
    )

    out_tsv = OUT_DIR / f"{case_id}_gsea.tsv"
    combined.to_csv(out_tsv, sep="\t", index=False)
    print(f"[Done] {len(combined)} enriched pathways → {out_tsv}")
    return combined


def plot_gsea(df: pd.DataFrame, case_id: str):
    if df.empty:
        return

    top = pd.concat([
        df[df["NES"] > 0].head(15),   # top activated
        df[df["NES"] < 0].tail(15),   # top suppressed
    ]).drop_duplicates()

    top = top.sort_values("NES")
    colors = ["#d73027" if n > 0 else "#4575b4" for n in top["NES"]]

    fig, ax = plt.subplots(figsize=(9, max(5, len(top) * 0.3)))
    bars = ax.barh(range(len(top)), top["NES"], color=colors, alpha=0.8)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["Term"], fontsize=7)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Normalized Enrichment Score (NES)")
    ax.set_title(f"GSEA Pathway Activity — {case_id}\n(red=activated, blue=suppressed)")

    # Mark LUAD-relevant
    for i, (_, row) in enumerate(top.iterrows()):
        if row.get("luad_relevant"):
            ax.get_yticklabels()[i].set_color("darkgreen")
            ax.get_yticklabels()[i].set_fontweight("bold")

    plt.tight_layout()
    out_png = OUT_DIR / f"{case_id}_gsea.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] → {out_png}")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample", type=str, help="Case ID (e.g. TCGA-49-4507)")
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.sample:
        expr = load_expression(args.sample)
        df = run_gsea_prerank(expr, args.sample)
        plot_gsea(df, args.sample)
    else:
        for f in sorted(RNASEQ_DIR.glob("*.tsv.gz")):
            sample_id = f.stem.replace(".tsv", "")
            print(f"\n=== {sample_id} ===")
            try:
                expr = load_expression(sample_id)
                df = run_gsea_prerank(expr, sample_id)
                plot_gsea(df, sample_id)
            except Exception as e:
                print(f"[Error] {sample_id}: {e}")


if __name__ == "__main__":
    main()
