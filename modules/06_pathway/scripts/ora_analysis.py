"""
ora_analysis.py
---------------
Over-Representation Analysis (ORA): which KEGG/Reactome pathways are
enriched in the mutated genes of a LUAD sample?

Input:
  MAF file from data/input/ (somatic mutations)

Output:
  data/output/{case_id}_ora.tsv     — enriched pathways table
  data/output/{case_id}_ora.png     — dot plot visualization

Requirements:
  pip install gseapy matplotlib pandas

Usage:
  python ora_analysis.py --maf ../../data/input/TCGA-49-4507.maf
  python ora_analysis.py --all
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import gseapy as gp

SCRIPT_DIR = Path(__file__).parent
MAF_DIR    = SCRIPT_DIR / "../../data/input"
OUT_DIR    = SCRIPT_DIR / "../data/output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Gene sets to query
GENE_SETS = ["KEGG_2021_Human", "Reactome_2022"]


def run_ora(maf_path: Path):
    case_id = maf_path.stem
    df = pd.read_csv(maf_path, sep="\t", comment="#", low_memory=False)

    # Extract unique mutated genes (non-synonymous)
    nonsynon = ["Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del",
                "Frame_Shift_Ins", "Splice_Site", "In_Frame_Del", "In_Frame_Ins"]
    genes = df[df["Variant_Classification"].isin(nonsynon)]["Hugo_Symbol"].dropna().unique().tolist()
    print(f"[ORA] {case_id}: {len(genes)} mutated genes")

    results = []
    for gene_set in GENE_SETS:
        enr = gp.enrichr(
            gene_list=genes,
            gene_sets=gene_set,
            outdir=None,
            cutoff=0.05,
        )
        res = enr.results
        res["gene_set"] = gene_set
        results.append(res)

    combined = pd.concat(results, ignore_index=True)
    combined = combined[combined["Adjusted P-value"] < 0.05].sort_values("Adjusted P-value")

    # Save table
    out_tsv = OUT_DIR / f"{case_id}_ora.tsv"
    combined.to_csv(out_tsv, sep="\t", index=False)
    print(f"[Done] {len(combined)} enriched pathways → {out_tsv}")

    # Dot plot (top 20)
    plot_ora(combined.head(20), case_id)
    return combined


def plot_ora(df: pd.DataFrame, case_id: str):
    if df.empty:
        return
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.35)))
    df = df.copy()
    df["-log10(FDR)"] = -np.log10(df["Adjusted P-value"].clip(lower=1e-10))
    df["Overlap_ratio"] = df["Overlap"].apply(
        lambda x: int(x.split("/")[0]) / int(x.split("/")[1]) if "/" in str(x) else 0
    )
    df = df.sort_values("-log10(FDR)")

    sc = ax.scatter(
        df["-log10(FDR)"], range(len(df)),
        s=df["Overlap_ratio"] * 500,
        c=df["-log10(FDR)"], cmap="RdYlBu_r", alpha=0.8
    )
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["Term"], fontsize=8)
    ax.set_xlabel("-log10(FDR)")
    ax.set_title(f"ORA Enriched Pathways — {case_id}")
    plt.colorbar(sc, label="-log10(FDR)")
    plt.tight_layout()

    out_png = OUT_DIR / f"{case_id}_ora.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] → {out_png}")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--maf", type=Path)
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.maf:
        run_ora(args.maf)
    else:
        for maf in sorted(MAF_DIR.glob("*.maf")):
            run_ora(maf)


if __name__ == "__main__":
    main()
