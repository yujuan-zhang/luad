#!/usr/bin/env python
"""
luad_esm2.py
------------
Module 06: ESM2-based per-site characterization of LUAD missense mutations.

Pipeline (two phases):

  Phase 1 — ESM2 Inference (SLOW, GPU recommended, saves results to disk):
    Input : missense mutations from module 02 (gene, protein position, wt_aa, mut_aa)
    Steps :
      1. Fetch canonical WT protein sequence from UniProt by protein/gene ID
      2. For each mutation site, run ESM2 to extract:
           (a) Per-site embedding (1280-dim) at mutation position — WT sequence
           (b) Per-site embedding (1280-dim) at mutation position — mutant sequence
           (c) ESM2 masked-marginal log-odds:
               log P(mut_aa | context) − log P(wt_aa | context)
               → negative = WT conserved = mutation likely pathogenic
               → positive = mutant tolerated = likely benign
      3. Compute delta embedding: mut_emb − wt_emb (structural perturbation)
    Output: {sample}_site_info.csv       — per-site metadata (mirrors test_info.csv)
            {sample}_wt_features.npy     — (N × 1280) WT embeddings
            {sample}_mut_features.npy    — (N × 1280) mutant embeddings
            {sample}_delta_features.npy  — (N × 1280) delta embeddings

  Phase 2 — Scoring + Visualization (FAST, loads saved .npy features):
    Scores:
      esm2_log_odds    — primary pathogenicity signal (ESM2 masked marginal)
      delta_magnitude  — L2‖mut_emb − wt_emb‖  (structural perturbation)
      cosine_distance  — 1 − cos(wt, mut)       (semantic shift)
      oncogenicity_score — combined ranking score
    Output: {sample}_mutation_scores.tsv
            {sample}_esm2_summary.png

Relationship to ESM2_AMP-Ubiquitination project:
  - Same feature format: per-site 1280-dim embedding saved as .npy
  - Same site_info.csv structure (mutation_id, gene, protein_id, mut_site …)
  - Extended with log-odds scoring (not in ubiquitination project)
  - Adapted for oncogenicity instead of ubiquitination

Input:  data/output/02_variants/{sample}/{sample}_variants.tsv.gz
Output: data/output/07_esm2/{sample}/

Usage:
  python luad_esm2.py                        # all samples
  python luad_esm2.py --sample TCGA-86-A4D0  # single sample
  python luad_esm2.py --dry_run              # validate inputs, count mutations
"""

import argparse
import logging
import re
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import requests
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
VARIANT_DIR = PROJECT_DIR / "data/output/02_variants"
OUT_DIR     = PROJECT_DIR / "data/output/07_esm2"
CACHE_DIR   = SCRIPT_DIR / "data/input"   # local HuggingFace model cache

ESM2_MODEL   = "facebook/esm2_t33_650M_UR50D"
ESM2_MAX_LEN = 1022   # max residues (1024 tokens − CLS − EOS)

# ── Known LUAD biology ────────────────────────────────────────────────────────
LUAD_DRIVERS = {
    "EGFR", "KRAS", "TP53", "STK11", "KEAP1", "BRAF", "MET", "RET",
    "ALK", "ROS1", "ERBB2", "NF1", "CDKN2A", "RB1", "PIK3CA", "PTEN",
    "MYC", "MYCL", "FGFR1", "DDR2", "MAP2K1", "NRAS", "HRAS",
}

# Curated LUAD hotspot mutations {gene: {position: set_of_oncogenic_mutant_aas}}
LUAD_HOTSPOTS = {
    "EGFR": {858: {"R"}, 790: {"M"}, 746: {"A","S","T","V","P","K"}, 747: {"G","S","A","P"}},
    "KRAS": {12: {"D","C","V","A","R","S"}, 13: {"D","C","V"}},
    "BRAF": {600: {"E","K"}},
    "TP53": {175: {"H"}, 248: {"W","Q"}, 249: {"S"}, 273: {"H","R"}, 245: {"S"}},
    "PIK3CA": {1047: {"R","L"}, 545: {"E","K"}, 542: {"E","K"}},
}


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
    return logging.getLogger(name)


# ═════════════════════════════════════════════════════════════════════════════
# Data preparation helpers
# ═════════════════════════════════════════════════════════════════════════════

def load_missense_mutations(sample_id: str) -> pd.DataFrame:
    """
    Load missense mutations from module 02 output.

    Column schema (module 02 uppercase):
      SYMBOL, CONSEQUENCE, HGVSp_Short (p.L858R), SIFT, PolyPhen, hotspot

    Returns DataFrame: mutation_id, SYMBOL, Protein_position (int),
                       wt_aa, mut_aa, sample_id, [SIFT, PolyPhen, hotspot]
    """
    var_path = VARIANT_DIR / sample_id / f"{sample_id}_variants.tsv.gz"
    if not var_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(var_path, sep="\t", compression="gzip", low_memory=False)

    cons_col = next(
        (c for c in ["CONSEQUENCE", "Consequence"] if c in df.columns), None
    )
    if cons_col is None or "HGVSp_Short" not in df.columns:
        return pd.DataFrame()

    missense = df[
        df[cons_col].str.contains("missense", case=False, na=False)
    ].copy()

    if missense.empty:
        return pd.DataFrame()

    # Parse p.L858R → wt_aa=L, position=858, mut_aa=R
    hgvsp_re = re.compile(r"^p\.([A-Z])(\d+)([A-Z])$")
    parsed   = missense["HGVSp_Short"].str.extract(hgvsp_re, expand=True)
    parsed.columns = ["wt_aa", "Protein_position", "mut_aa"]

    missense = pd.concat(
        [missense.reset_index(drop=True), parsed.reset_index(drop=True)], axis=1
    ).dropna(subset=["wt_aa", "Protein_position", "mut_aa"])

    missense["Protein_position"] = missense["Protein_position"].astype(int)
    missense["mutation_id"] = (
        missense["SYMBOL"] + "_" +
        missense["wt_aa"] +
        missense["Protein_position"].astype(str) +
        missense["mut_aa"]
    )
    missense["sample_id"] = sample_id

    keep = ["mutation_id", "SYMBOL", "Protein_position", "wt_aa", "mut_aa", "sample_id"]
    for extra in ["SIFT", "PolyPhen", "IMPACT", "hotspot"]:
        if extra in missense.columns:
            keep.append(extra)

    return missense[keep].drop_duplicates("mutation_id").reset_index(drop=True)


def fetch_uniprot_sequence(gene_symbol: str, logger) -> tuple:
    """
    Fetch canonical human protein sequence from UniProt REST API.
    Returns (uniprot_id, sequence_str) or (None, None) on failure.
    """
    url = (
        "https://rest.uniprot.org/uniprotkb/search"
        f"?query=gene_exact:{gene_symbol}+AND+organism_id:9606+AND+reviewed:true"
        "&fields=accession,sequence&format=json&size=1"
    )
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return None, None
        entry = results[0]
        return entry["primaryAccession"], entry["sequence"]["value"]
    except Exception as e:
        logger.warning(f"  UniProt fetch failed for {gene_symbol}: {e}")
        return None, None


def window_sequence(seq: str, position: int) -> tuple:
    """
    For proteins longer than ESM2_MAX_LEN, extract a window centred at position.
    Returns (windowed_seq, adjusted_position_1indexed).
    """
    if len(seq) <= ESM2_MAX_LEN:
        return seq, position
    half  = ESM2_MAX_LEN // 2
    start = max(0, position - 1 - half)
    end   = min(len(seq), start + ESM2_MAX_LEN)
    start = max(0, end - ESM2_MAX_LEN)
    adj   = position - start           # 1-indexed in window
    return seq[start:end], adj


def build_mutant_seq(wt_seq: str, position: int, mut_aa: str) -> str:
    """Substitute residue at 1-indexed position with mut_aa."""
    p = position - 1
    return wt_seq[:p] + mut_aa + wt_seq[p + 1:]


def is_hotspot(gene: str, pos: int, mut_aa: str) -> bool:
    if gene not in LUAD_HOTSPOTS:
        return False
    pos_dict = LUAD_HOTSPOTS[gene]
    if pos not in pos_dict:
        return False
    expected = pos_dict[pos]
    return expected is None or mut_aa in expected


# ═════════════════════════════════════════════════════════════════════════════
# Phase 1: ESM2 inference engine
# ═════════════════════════════════════════════════════════════════════════════

class MutationSiteExtractor:
    """
    ESM2 feature extractor for per-site mutation characterization.

    Mirrors ESM2_AMP-Ubiquitination project:
      • Per-site embedding: last hidden state at the mutation-position token
        (analogous to get_last_hidden_phosphorylation_position_feature_fast)
      • Extended: masked-marginal log-odds for zero-shot pathogenicity scoring

    Model: facebook/esm2_t33_650M_UR50D (33-layer, hidden_size=1280)
    ESM2 token layout: [CLS=0] [aa1=1] [aa2=2] … [aaN=N] [EOS=N+1]
    Residue at 1-indexed position p → token index p (same index, CLS at 0).
    """

    def __init__(self, model_name: str = ESM2_MODEL, cache_dir: Path = CACHE_DIR):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=str(cache_dir)
        )
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            output_hidden_states=True,
            cache_dir=str(cache_dir),
        )
        self.model.to(self.device)
        self.model.eval()
        self.hidden_size = self.model.config.hidden_size  # 1280

    @torch.no_grad()
    def site_embedding(self, sequence: str, position: int) -> np.ndarray:
        """
        Extract 1280-dim per-site embedding at mutation position.
        'position' is 1-indexed (matches HGVSp_Short convention).
        """
        inputs      = self.tokenizer(sequence, return_tensors="pt").to(self.device)
        outputs     = self.model(**inputs)
        last_hidden = outputs.hidden_states[-1]       # (1, L+2, 1280)
        n_residues  = last_hidden.shape[1] - 2
        tok_pos     = max(1, min(int(position), n_residues))
        emb = last_hidden[0, tok_pos, :].cpu().float().numpy()
        del inputs, outputs
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return emb

    @torch.no_grad()
    def masked_log_odds(
        self, sequence: str, position: int, wt_aa: str, mut_aa: str
    ) -> float:
        """
        ESM2 masked-marginal log-odds (zero-shot variant effect prediction).

        Score = log P(mut_aa | masked context) − log P(wt_aa | masked context)

        Negative score → WT is evolutionarily conserved → mutation likely pathogenic.
        Positive score → mutation tolerated → likely benign.

        Reference: Meier et al. 2021 (ESM-1v); works analogously with ESM2.
        """
        pos0       = position - 1
        masked_seq = sequence[:pos0] + "<mask>" + sequence[pos0 + 1:]
        inputs     = self.tokenizer(masked_seq, return_tensors="pt").to(self.device)
        outputs    = self.model(**inputs)

        n_residues = outputs.logits.shape[1] - 2
        tok_pos    = max(1, min(int(position), n_residues))
        log_probs  = F.log_softmax(outputs.logits[0, tok_pos, :], dim=-1)

        wt_id  = self.tokenizer.convert_tokens_to_ids(wt_aa)
        mut_id = self.tokenizer.convert_tokens_to_ids(mut_aa)

        if self.tokenizer.unk_token_id in (wt_id, mut_id):
            return float("nan")

        log_odds = log_probs[mut_id].item() - log_probs[wt_id].item()
        del inputs, outputs
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return log_odds


# ═════════════════════════════════════════════════════════════════════════════
# Per-sample pipeline
# ═════════════════════════════════════════════════════════════════════════════

def run_sample(sample_id: str, extractor: MutationSiteExtractor, logger) -> dict:
    logger.info(f"\n{'='*55}")
    logger.info(f"Sample: {sample_id}")

    out_dir = OUT_DIR / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load mutations from module 02 ──
    muts = load_missense_mutations(sample_id)
    if muts.empty:
        logger.info("  No missense mutations — skipping")
        return {"sample_id": sample_id, "n_sites": 0}
    logger.info(f"  {len(muts)} missense mutations")

    # ── Step 2: Fetch WT protein sequences from UniProt ──
    gene_seqs = {}
    for gene in sorted(muts["SYMBOL"].unique()):
        uid, seq = fetch_uniprot_sequence(gene, logger)
        if uid and seq:
            gene_seqs[gene] = (uid, seq)
        time.sleep(0.1)   # polite rate limiting
    logger.info(
        f"  UniProt: {len(gene_seqs)}/{muts['SYMBOL'].nunique()} sequences retrieved"
    )

    # ── Step 3: Build validated site table ──
    records = []
    for _, row in muts.iterrows():
        gene = row["SYMBOL"]
        if gene not in gene_seqs:
            continue
        uid, wt_seq = gene_seqs[gene]
        pos    = int(row["Protein_position"])
        wt_aa  = str(row["wt_aa"])
        mut_aa = str(row["mut_aa"])

        # Validate: position in range and WT amino acid matches
        if pos < 1 or pos > len(wt_seq):
            continue
        if wt_seq[pos - 1].upper() != wt_aa.upper():
            continue

        # Window long proteins for ESM2 token limit
        wt_win, adj_pos = window_sequence(wt_seq, pos)
        mut_win, _      = window_sequence(build_mutant_seq(wt_seq, pos, mut_aa), pos)

        rec = {
            "mutation_id": row["mutation_id"],
            "gene":        gene,
            "uniprot_id":  uid,
            "sample_id":   row["sample_id"],
            "mut_site":    pos,        # 1-indexed, full protein (mirrors ubi_site)
            "adj_pos":     adj_pos,    # 1-indexed in windowed sequence (ESM2 input)
            "wt_aa":       wt_aa,
            "mut_aa":      mut_aa,
            "wt_win":      wt_win,     # sequence fed to ESM2
            "mut_win":     mut_win,
            "seq_length":  len(wt_seq),
            "is_driver":   gene in LUAD_DRIVERS,
            "is_hotspot":  is_hotspot(gene, pos, mut_aa),
        }
        for col in ["SIFT", "PolyPhen", "IMPACT", "hotspot"]:
            if col in row.index:
                rec[f"m02_{col}"] = row[col]
        records.append(rec)

    if not records:
        logger.info("  No sites passed sequence validation — skipping")
        return {"sample_id": sample_id, "n_sites": 0}

    site_df = (
        pd.DataFrame(records)
        .drop_duplicates("mutation_id")
        .reset_index(drop=True)
    )
    logger.info(
        f"  {len(site_df)} unique validated sites "
        f"({site_df['is_driver'].sum()} driver, {site_df['is_hotspot'].sum()} hotspot)"
    )

    # Save site info — mirrors test_info.csv from ubiquitination project
    meta_cols = [
        "mutation_id", "gene", "uniprot_id", "sample_id",
        "mut_site", "wt_aa", "mut_aa", "seq_length", "is_driver", "is_hotspot",
    ] + [c for c in site_df.columns if c.startswith("m02_")]
    site_df[meta_cols].to_csv(out_dir / f"{sample_id}_site_info.csv", index=False)

    # ── Step 4: ESM2 inference (Phase 1 — SLOW) ──
    n = len(site_df)
    wt_features   = np.zeros((n, extractor.hidden_size), dtype=np.float32)
    mut_features  = np.zeros((n, extractor.hidden_size), dtype=np.float32)
    log_odds_arr  = np.full(n, np.nan, dtype=np.float32)

    logger.info(f"  ESM2 inference: {n} sites × 3 passes (wt_emb, mut_emb, log_odds) ...")
    for i, (_, row) in enumerate(site_df.iterrows()):
        if i % 10 == 0:
            logger.info(f"    [{i+1}/{n}] {row['mutation_id']}")
        pos = int(row["adj_pos"])

        # (a) WT per-site embedding
        wt_features[i]  = extractor.site_embedding(row["wt_win"], pos)
        # (b) Mutant per-site embedding
        mut_features[i] = extractor.site_embedding(row["mut_win"], pos)
        # (c) ESM2 masked-marginal log-odds (pathogenicity proxy)
        log_odds_arr[i] = extractor.masked_log_odds(
            row["wt_win"], pos, row["wt_aa"], row["mut_aa"]
        )

    # Delta embedding = mut − wt (structural perturbation representation)
    delta_features = mut_features - wt_features

    # ── Step 5: Save feature arrays (.npy, same format as ubiquitination project) ──
    np.save(out_dir / f"{sample_id}_wt_features.npy",    wt_features)
    np.save(out_dir / f"{sample_id}_mut_features.npy",   mut_features)
    np.save(out_dir / f"{sample_id}_delta_features.npy", delta_features)
    logger.info(
        f"  Saved: wt/mut/delta features shape={wt_features.shape} → {out_dir}/"
    )

    # ── Step 6: Phase 2 — Compute scores and rank ──
    scores = site_df[meta_cols].copy()
    scores["esm2_log_odds"]   = log_odds_arr
    scores["delta_magnitude"] = np.linalg.norm(delta_features, axis=1)
    scores["cosine_distance"] = 1.0 - (
        np.sum(wt_features * mut_features, axis=1)
        / (
            np.linalg.norm(wt_features, axis=1)
            * np.linalg.norm(mut_features, axis=1)
            + 1e-10
        )
    )
    # Combined oncogenicity score (higher = more likely disease-related):
    #   - Negative log_odds means WT is conserved → pathogenic → negate
    #   - Large delta_magnitude means structural perturbation → pathogenic
    scores["oncogenicity_score"] = (
        -scores["esm2_log_odds"].fillna(0)
        + 0.3 * scores["delta_magnitude"]
    )
    scores = scores.sort_values("oncogenicity_score", ascending=False).reset_index(drop=True)

    scores_path = out_dir / f"{sample_id}_mutation_scores.tsv"
    scores.to_csv(scores_path, sep="\t", index=False)
    logger.info(f"  Scores → {scores_path}")
    logger.info(f"  Top 5 oncogenic mutations:")
    for _, r in scores.head(5).iterrows():
        logger.info(
            f"    {r['mutation_id']:25s}  log_odds={r['esm2_log_odds']:+.3f}  "
            f"delta={r['delta_magnitude']:.3f}  hotspot={r['is_hotspot']}"
        )

    # ── Step 7: Visualization ──
    plot_esm2_summary(scores, delta_features, sample_id, out_dir)

    return {
        "sample_id":       sample_id,
        "n_sites":         n,
        "n_driver_sites":  int(site_df["is_driver"].sum()),
        "n_hotspot_sites": int(site_df["is_hotspot"].sum()),
        "mean_log_odds":   float(np.nanmean(log_odds_arr)),
        "min_log_odds":    float(np.nanmin(log_odds_arr)),
        "n_pathogenic":    int((log_odds_arr < -2).sum()),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Visualization
# ═════════════════════════════════════════════════════════════════════════════

def plot_esm2_summary(
    scores: pd.DataFrame,
    delta_features: np.ndarray,
    sample_id: str,
    out_dir: Path,
):
    """
    4-panel ESM2 mutation characterization summary.

    [0,0] ESM2 log-odds distribution — conservation-based pathogenicity
    [0,1] Oncogenicity scatter — log_odds vs delta_magnitude
    [1,0] Top 20 mutations by oncogenicity score
    [1,1] Summary statistics table
    """
    if scores.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(
        f"ESM2 Mutation Site Characterization — {sample_id}\n"
        "(red = LUAD driver gene  |  ★ = known hotspot)",
        fontsize=12,
    )

    driver_mask  = scores["is_driver"].values
    hotspot_mask = scores["is_hotspot"].values
    log_odds     = scores["esm2_log_odds"].values
    delta_mag    = scores["delta_magnitude"].values

    # ── [0,0] Log-odds distribution ──────────────────────────────────────────
    ax = axes[0, 0]
    ax.hist(log_odds[~driver_mask & ~np.isnan(log_odds)],
            bins=30, color="#4575b4", alpha=0.7, label="Other")
    ax.hist(log_odds[driver_mask & ~np.isnan(log_odds)],
            bins=15, color="#d73027", alpha=0.8, label="Driver gene")
    ax.axvline(0, color="black", lw=1.0, linestyle="--")
    ax.axvline(-2, color="darkorange", lw=1.0, linestyle=":", label="Pathogenic threshold (−2)")
    ax.set_xlabel("ESM2 Log-odds  [log P(mut) − log P(wt)]", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title("ESM2 Masked-Marginal Log-odds\n(negative = WT conserved = pathogenic)",
                 fontsize=9)
    ax.legend(fontsize=8)

    # ── [0,1] Perturbation landscape scatter ─────────────────────────────────
    ax = axes[0, 1]
    valid = ~np.isnan(log_odds)
    # non-driver, non-hotspot
    m = valid & ~driver_mask & ~hotspot_mask
    ax.scatter(log_odds[m], delta_mag[m], s=20, alpha=0.5, color="#4575b4", label="Other")
    # driver genes (non-hotspot)
    m = valid & driver_mask & ~hotspot_mask
    ax.scatter(log_odds[m], delta_mag[m], s=45, alpha=0.85, color="#d73027", label="Driver")
    # confirmed hotspots
    m = valid & hotspot_mask
    ax.scatter(log_odds[m], delta_mag[m], s=90, alpha=0.95,
               color="gold", edgecolors="#d73027", linewidths=0.8,
               marker="*", label="Hotspot", zorder=5)
    # Annotate hotspot mutations
    for _, row in scores[scores["is_hotspot"]].iterrows():
        if not np.isnan(row["esm2_log_odds"]):
            ax.annotate(row["mutation_id"], (row["esm2_log_odds"], row["delta_magnitude"]),
                        fontsize=6, xytext=(4, 4), textcoords="offset points")
    ax.axvline(0, color="gray", lw=0.7, linestyle="--")
    ax.axvline(-2, color="darkorange", lw=0.7, linestyle=":")
    ax.set_xlabel("ESM2 Log-odds", fontsize=9)
    ax.set_ylabel("Delta Magnitude  ‖mut − wt‖", fontsize=9)
    ax.set_title("Perturbation Landscape\n(bottom-left = most oncogenic)", fontsize=9)
    ax.legend(fontsize=8)

    # ── [1,0] Top 20 mutations by oncogenicity score ──────────────────────────
    ax = axes[1, 0]
    top = scores.head(20).copy()
    # colour: hotspot=gold, driver=red, other=blue
    colors = []
    for _, r in top.iterrows():
        if r["is_hotspot"]:
            colors.append("gold")
        elif r["is_driver"]:
            colors.append("#d73027")
        else:
            colors.append("#4575b4")

    bars = ax.barh(range(len(top)), top["oncogenicity_score"][::-1].values,
                   color=colors[::-1], alpha=0.85)
    ax.set_yticks(range(len(top)))
    labels = [
        ("★ " if r["is_hotspot"] else "") + r["mutation_id"]
        for _, r in top[::-1].iterrows()
    ]
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Oncogenicity Score  [−log_odds + 0.3×delta_mag]", fontsize=9)
    ax.set_title("Top 20 Ranked Mutations", fontsize=9)

    # ── [1,1] Summary table ───────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.axis("off")

    n_total    = len(scores)
    n_driver   = int(scores["is_driver"].sum())
    n_hotspot  = int(scores["is_hotspot"].sum())
    n_pathogen = int((log_odds < -2).sum())
    top5_list  = scores.head(5)["mutation_id"].tolist()

    rows = [
        ["Total mutation sites",      str(n_total)],
        ["Driver gene sites",         f"{n_driver} ({100*n_driver/n_total:.0f}%)"],
        ["Known hotspot sites",       str(n_hotspot)],
        ["Pathogenic (log-odds < −2)",f"{n_pathogen} ({100*n_pathogen/n_total:.0f}%)"],
        ["Mean ESM2 log-odds",        f"{np.nanmean(log_odds):+.3f}"],
        ["Min ESM2 log-odds",         f"{np.nanmin(log_odds):+.3f}"],
    ] + [
        [f"#{i+1} oncogenic", top5_list[i] if i < len(top5_list) else "N/A"]
        for i in range(5)
    ]

    tbl = ax.table(
        cellText=rows,
        colLabels=["Metric", "Value"],
        cellLoc="left",
        loc="center",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f0f0f0")
    ax.set_title("Summary Statistics", fontsize=9)

    plt.tight_layout()
    out_png = out_dir / f"{sample_id}_esm2_summary.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    logging.getLogger("luad-esm2").info(f"  Plot → {out_png}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def get_all_samples() -> list:
    return sorted(
        p.name for p in VARIANT_DIR.glob("*/") if p.is_dir()
    )


def main():
    parser = argparse.ArgumentParser(
        description="LUAD ESM2 per-site mutation characterization"
    )
    parser.add_argument("--sample",  type=str, help="Single sample ID")
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Count mutations and validate inputs without running ESM2"
    )
    args   = parser.parse_args()
    logger = get_logger("luad-esm2")

    if args.dry_run:
        samples = [args.sample] if args.sample else get_all_samples()
        logger.info(f"[DRY RUN] {len(samples)} samples")
        for s in samples:
            muts = load_missense_mutations(s)
            logger.info(f"  {s}: {len(muts)} missense mutations")
        return

    samples = [args.sample] if args.sample else get_all_samples()
    if not samples:
        print("[ERROR] No samples found")
        return

    # Load ESM2 model once — reused across all samples
    logger.info(f"Loading ESM2: {ESM2_MODEL}")
    extractor = MutationSiteExtractor(model_name=ESM2_MODEL, cache_dir=CACHE_DIR)
    logger.info(
        f"  Device={extractor.device}  hidden_size={extractor.hidden_size}"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for s in samples:
        try:
            r = run_sample(s, extractor, logger)
        except Exception as e:
            logger.error(f"Sample {s} failed: {e}")
            r = {"sample_id": s, "n_sites": -1}
        results.append(r)

    if results:
        summary = pd.DataFrame(results)
        summary_path = OUT_DIR / "all_samples_esm2_summary.tsv"
        summary.to_csv(summary_path, sep="\t", index=False)
        print(f"\n[Summary] {len(results)} samples → {summary_path}")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
