"""
kegg_viewer.py
--------------
Per-patient KEGG pathway viewer for LUAD.

For a given patient:
  1. Find which of the 8 core LUAD pathways have ≥1 mutated gene
  2. Color genes on the official KEGG pathway image:
       red    = mutated
       orange = high expression (GTEx z-score > 1.5), not mutated
       blue   = low expression  (GTEx z-score < -1.5), not mutated
  3. Return colored KEGG URL + small gene hit table

Gene membership comes from gseapy's built-in KEGG_2021_Human gene sets
(no network dependency). Only the pathway image display hits KEGG servers.
"""

import requests
import pandas as pd
from pathlib import Path
from urllib.parse import quote

import gseapy as gp

SCRIPT_DIR  = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
VARIANT_DIR = PROJECT_DIR / "data/output/02_variants"
EXPR_DIR    = PROJECT_DIR / "data/output/03_expression"

# ── 8 core LUAD pathways ──────────────────────────────────────────────────────
# display name → (KEGG pathway ID, gseapy KEGG_2021_Human key)

LUAD_PATHWAYS = {
    "MAPK Signaling":   ("hsa04010", "MAPK signaling pathway"),
    "PI3K-AKT":         ("hsa04151", "PI3K-Akt signaling pathway"),
    "ErbB / EGFR":      ("hsa04012", "ErbB signaling pathway"),
    "p53 Signaling":    ("hsa04115", "p53 signaling pathway"),
    "Cell Cycle":       ("hsa04110", "Cell cycle"),
    "TGF-β Signaling":  ("hsa04350", "TGF-beta signaling pathway"),
    "Wnt Signaling":    ("hsa04310", "Wnt signaling pathway"),
    "VEGF Signaling":   ("hsa04370", "VEGF signaling pathway"),
}


# ── Gene membership via gseapy (no network needed) ───────────────────────────

def load_all_pathway_genes() -> dict:
    """
    Return {pathway_id: [symbols]} for all 8 LUAD pathways.
    Uses gseapy KEGG_2021_Human gene sets — no REST API calls needed.
    """
    kegg_sets = gp.get_library("KEGG_2021_Human")
    result = {}
    for name, (pid, gsea_key) in LUAD_PATHWAYS.items():
        genes = kegg_sets.get(gsea_key, [])
        result[pid] = [g.upper() for g in genes]
    return result


# ── Patient data loaders ──────────────────────────────────────────────────────

def load_mutations(sample_id: str) -> list:
    """Load non-synonymous mutated gene symbols from M02."""
    path = VARIANT_DIR / sample_id / f"{sample_id}_variants.tsv.gz"
    if not path.exists():
        return []
    df = pd.read_csv(path, sep="\t", compression="gzip", low_memory=False)
    if "SYMBOL" not in df.columns:
        return []
    return df["SYMBOL"].dropna().str.upper().unique().tolist()


def load_expression_colors(sample_id: str, pathway_genes: list) -> tuple:
    """
    Return (high_expr, low_expr) gene lists for pathway genes.
    Uses GTEx z-score from M03 expression outlier table.
      high_expr: GTEX_ZSCORE > 1.5
      low_expr : GTEX_ZSCORE < -1.5
    """
    path = EXPR_DIR / sample_id / f"{sample_id}_expression_outliers.tsv"
    if not path.exists():
        # Try gzip version
        gz = path.with_suffix(".tsv.gz")
        if gz.exists():
            df = pd.read_csv(gz, sep="\t", compression="gzip")
        else:
            return [], []
    else:
        df = pd.read_csv(path, sep="\t")

    if "SYMBOL" not in df.columns or "GTEX_ZSCORE" not in df.columns:
        return [], []

    pg = {g.upper() for g in pathway_genes}
    df = df[df["SYMBOL"].str.upper().isin(pg)].copy()

    high = df[df["GTEX_ZSCORE"] >  1.5]["SYMBOL"].str.upper().tolist()
    low  = df[df["GTEX_ZSCORE"] < -1.5]["SYMBOL"].str.upper().tolist()
    return high, low


# ── Hit detection ─────────────────────────────────────────────────────────────

def get_hit_pathways(mutations: list, pathway_genes: dict) -> list:
    """
    Return list of (name, pathway_id, hit_genes) for pathways with ≥1
    mutated gene, sorted by number of hits descending.
    """
    hits = []
    mut_set = {m.upper() for m in mutations}
    for name, (pid, _) in LUAD_PATHWAYS.items():
        genes  = {g.upper() for g in pathway_genes.get(pid, [])}
        hit    = sorted(mut_set & genes)
        if hit:
            hits.append((name, pid, hit))
    hits.sort(key=lambda x: len(x[2]), reverse=True)
    return hits


# ── KEGG URL builder ──────────────────────────────────────────────────────────

def build_kegg_url(pathway_id: str,
                   mutations: list,
                   high_expr: list,
                   low_expr: list) -> str:
    """
    Build KEGG pathway URL with gene coloring.
      red    = mutated (priority)
      orange = high expression, not mutated
      blue   = low expression,  not mutated
    """
    mut_set = {g.upper() for g in mutations}
    lines = []
    for g in mutations:
        lines.append(f"{g}\tred")
    for g in high_expr:
        if g.upper() not in mut_set:
            lines.append(f"{g}\torange")
    for g in low_expr:
        if g.upper() not in mut_set:
            lines.append(f"{g}\tblue")

    if not lines:
        return f"https://www.kegg.jp/pathway/{pathway_id}"

    multi_query = quote("\n".join(lines))
    return (
        f"https://www.kegg.jp/kegg-bin/show_pathway"
        f"?{pathway_id}&multi_query={multi_query}"
    )


# ── Gene hit table ────────────────────────────────────────────────────────────

def build_gene_table(sample_id: str,
                     pathway_genes: list,
                     mutations: list,
                     high_expr: list,
                     low_expr: list) -> pd.DataFrame:
    """
    Build a small table of pathway genes that are mutated or
    have significant expression changes for this patient.
    """
    mut_set  = {g.upper() for g in mutations}
    high_set = {g.upper() for g in high_expr}
    low_set  = {g.upper() for g in low_expr}

    # Load z-scores for all pathway genes
    path = EXPR_DIR / sample_id / f"{sample_id}_expression_outliers.tsv"
    gz   = path.with_suffix(".tsv.gz")
    expr_df = pd.DataFrame()
    if path.exists():
        expr_df = pd.read_csv(path, sep="\t")
    elif gz.exists():
        expr_df = pd.read_csv(gz, sep="\t", compression="gzip")

    zscore_map = {}
    if not expr_df.empty and "SYMBOL" in expr_df.columns and "GTEX_ZSCORE" in expr_df.columns:
        for _, row in expr_df.iterrows():
            zscore_map[str(row["SYMBOL"]).upper()] = round(float(row["GTEX_ZSCORE"]), 2)

    rows = []
    for g in sorted(pathway_genes):
        gu = g.upper()
        mutated  = gu in mut_set
        high     = gu in high_set
        low_e    = gu in low_set
        if not (mutated or high or low_e):
            continue
        z = zscore_map.get(gu, None)
        rows.append({
            "Gene":      g,
            "Mutated":   "●" if mutated else "",
            "Expr Z":    f"{z:+.2f}" if z is not None else "—",
            "Direction": ("↑ Over"  if high else
                          "↓ Under" if low_e else "—"),
        })

    return pd.DataFrame(rows)
