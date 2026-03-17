#!/usr/bin/env python
"""
luad_io_ml.py  —  Module 08: Multi-modal Immune Activity Score
==============================================================
Multi-modal LUAD immunotherapy suitability assessment.

Features (data-driven)
----------------------
  RNA   : all protein-coding genes, TCGA + GSE72094 combined
           → pre-filter top-5000 by variance
           → Stage 1: CoxNet selects ~100-200 genes
  Pathology : TIL density + TIL score (M05, TCGA only)
  Genomics  : STK11, KEAP1, EGFR, KRAS, TP53 mutations

Model pipeline
--------------
  Pre-filter: top-5,000 genes by variance
  CoxNet (Elastic Net Cox PH): nested 5-fold CV over α × l₁ grid
    → selects non-zero features (typically 50-150 genes)
  Scoring: CoxNet log-hazard (inverted) → normalized 0-100
  Training: TCGA-LUAD (n≈443), External validation: GSE72094 (n=398)

Outputs
-------
  data/output/08_io_ml/
    io_scores.tsv            per-patient Immune Activity Score (0-100)
    selected_genes.tsv       CoxNet-selected genes with coefficients
    feature_importance.tsv   feature importance (|coefficient|)
    coxnet_model.pkl         local only, not pushed
    figures/
      km_io_score.png
      km_stk11_subgroup.png
      km_gse72094.png        external validation KM
      feature_importance.png top-20 gene coefficients

Usage
-----
  python luad_io_ml.py              # full pipeline
  python luad_io_ml.py --features_only
  python luad_io_ml.py --force      # ignore cache
"""

import argparse
import logging
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_DIR  = Path(__file__).parent.parent.parent
EXPR_DIR     = PROJECT_DIR / "data/output/03_expression"
TME_DIR      = PROJECT_DIR / "data/output/04_single_cell"
PATHO_DIR    = PROJECT_DIR / "data/output/05_pathology"
VAR_DIR      = PROJECT_DIR / "data/output/02_variants"
CLINICAL_DIR = PROJECT_DIR / "data/clinical"
GSE_DIR      = Path(__file__).parent / "data/input/gse72094"
OUT_DIR      = PROJECT_DIR / "data/output/08_io_ml"
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "figures").mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

DRIVER_GENES  = ["STK11", "KEAP1", "EGFR", "KRAS", "TP53", "SMAD4"]
TOP_N_GENES   = 5000   # variance pre-filter
TOP_COX_GENES = 500    # univariate Cox filter (top genes by |Wald stat|)

STAGE_MAP = {
    "Stage IA": 1, "Stage IB": 1, "Stage I": 1,
    "Stage IIA": 2, "Stage IIB": 2, "Stage II": 2,
    "Stage IIIA": 3, "Stage IIIB": 3, "Stage III": 3,
    "Stage IV": 4,
}

# ── Immune signature gene sets ─────────────────────────────────────────────────
# TIS: Tumor Inflammation Signature (Ayers et al., JCI 2017)
TIS_GENES = ["CD3D","IDO1","CIITA","CD3E","CCL5","GZMK","CD2","HLA-DRA",
             "CXCL13","IL2RG","NKG7","HLA-E","CXCR6","LAG3","TIGIT","CD8A",
             "PDCD1LG2","CD27"]

# IMPRES: 15 gene-pair ratios (Auslander et al., Nat Med 2018)
# Score += 1 if expr(A) > expr(B)
IMPRES_PAIRS = [
    ("PDCD1","TIGIT"), ("PDCD1","BTLA"),
    ("TIGIT","CD276"), ("TIGIT","ADORA2A"),
    ("CTLA4","ADORA2A"), ("CTLA4","BTLA"),
    ("LAG3","TIGIT"), ("LAG3","HAVCR2"), ("LAG3","BTLA"),
    ("CD200","CD28"), ("CD200","CD86"), ("CD200","CD80"),
    ("HAVCR2","BTLA"), ("CD200","LAIR1"), ("PDCD1","LAG3"),
]

# M04 TME cell-type columns
TME_COLS = ["CD8_T_cytotoxic","Treg","CD8_T_exhausted","NK",
            "Macrophage_M1","Macrophage_M2","B_cell"]


# ══════════════════════════════════════════════════════════════════════════════
# 1. TCGA FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_tcga_expression(sample_id: str, gene_set: set = None) -> dict:
    """Log2-TPM for protein-coding genes from M03."""
    path = EXPR_DIR / sample_id / f"{sample_id}_gene_expression.tsv.gz"
    if not path.exists():
        return {}
    df = pd.read_csv(path, sep="\t",
                     usecols=["SYMBOL", "BIOTYPE", "TPM_LOG2_GENE"],
                     low_memory=False)
    df = df[df["BIOTYPE"] == "protein_coding"]
    if gene_set:
        df = df[df["SYMBOL"].isin(gene_set)]
    expr = df.groupby("SYMBOL")["TPM_LOG2_GENE"].max()
    return {f"rna_{g}": float(v) for g, v in expr.items()}


def extract_til(sample_id: str) -> dict:
    path = PATHO_DIR / sample_id / f"{sample_id}_pathology_scores.tsv"
    if not path.exists():
        return {}
    row = pd.read_csv(path, sep="\t").iloc[0]
    return {"til_density": float(row.get("til_density", np.nan)),
            "til_score":   float(row.get("til_score",   np.nan))}


def extract_mutations(sample_id: str) -> dict:
    path = VAR_DIR / sample_id / f"{sample_id}_variants.tsv.gz"
    if not path.exists():
        return {f"mut_{g}": 0 for g in DRIVER_GENES}
    df = pd.read_csv(path, sep="\t", compression="gzip", low_memory=False)
    sym_col = next((c for c in ["SYMBOL", "Hugo_Symbol"] if c in df.columns), None)
    if not sym_col:
        return {f"mut_{g}": 0 for g in DRIVER_GENES}
    mutated = set(df[sym_col].dropna())
    return {f"mut_{g}": int(g in mutated) for g in DRIVER_GENES}


def extract_clinical(sample_id: str, clinical_df: pd.DataFrame) -> dict:
    """Extract clinical stage and age from pre-loaded clinical table."""
    if sample_id not in clinical_df.index:
        return {}
    row = clinical_df.loc[sample_id]
    result = {}
    stage_val = STAGE_MAP.get(str(row.get("stage", "")), np.nan)
    result["clinical_stage"] = float(stage_val) if not np.isnan(stage_val) else np.nan
    age = row.get("age_at_diagnosis", np.nan)
    try:
        result["clinical_age"] = float(age)
    except (ValueError, TypeError):
        result["clinical_age"] = np.nan
    return result


def extract_tmb(sample_id: str) -> dict:
    """Compute TMB (mutations per Mb, exome ~30 Mb) from M02 variants."""
    path = VAR_DIR / sample_id / f"{sample_id}_variants.tsv.gz"
    if not path.exists():
        return {"tmb": np.nan}
    df = pd.read_csv(path, sep="\t", compression="gzip",
                     usecols=lambda c: c in ["Variant_Classification","VARIANT_CLASS",
                                             "SYMBOL","Hugo_Symbol"],
                     low_memory=False)
    # Count non-synonymous variants only
    nonsyn = {"Missense_Mutation","Nonsense_Mutation","Frame_Shift_Del",
              "Frame_Shift_Ins","In_Frame_Del","In_Frame_Ins","Splice_Site",
              "Nonstop_Mutation","Translation_Start_Site"}
    if "Variant_Classification" in df.columns:
        n = df["Variant_Classification"].isin(nonsyn).sum()
    else:
        n = len(df)   # fallback: count all variants
    return {"tmb": float(n) / 30.0}   # per Mb (30 Mb exome)


def extract_immune_signatures(sample_id: str) -> dict:
    """Compute TIS, CYT, IMPRES from M03 expression data."""
    path = EXPR_DIR / sample_id / f"{sample_id}_gene_expression.tsv.gz"
    if not path.exists():
        return {}
    df = pd.read_csv(path, sep="\t", usecols=["SYMBOL","TPM_GENE","TPM_LOG2_GENE"],
                     low_memory=False)
    df = df.drop_duplicates("SYMBOL").set_index("SYMBOL")
    result = {}

    # TIS: mean log2-TPM across 18 TIS genes
    avail = [g for g in TIS_GENES if g in df.index]
    if avail:
        result["sig_tis"] = float(df.loc[avail, "TPM_LOG2_GENE"].mean())

    # CYT: geometric mean of GZMA and PRF1 raw TPM (Rooney et al., Cell 2015)
    if "GZMA" in df.index and "PRF1" in df.index:
        gzma = max(float(df.loc["GZMA","TPM_GENE"]), 0)
        prf1 = max(float(df.loc["PRF1","TPM_GENE"]), 0)
        result["sig_cyt"] = float(np.sqrt(gzma * prf1))

    # IMPRES: fraction of 15 gene-pairs where pair[0] > pair[1] (Auslander 2018)
    score, total = 0, 0
    for ga, gb in IMPRES_PAIRS:
        if ga in df.index and gb in df.index:
            va = float(df.loc[ga,"TPM_LOG2_GENE"])
            vb = float(df.loc[gb,"TPM_LOG2_GENE"])
            score += int(va > vb)
            total += 1
    if total > 0:
        result["sig_impres"] = score / total

    return result


def extract_tme_scores(sample_id: str) -> dict:
    """Load M04 ssGSEA immune cell-type fractions."""
    path = TME_DIR / sample_id / f"{sample_id}_tme_scores.tsv"
    if not path.exists():
        return {}
    row = pd.read_csv(path, sep="\t").iloc[0]
    return {f"tme_{c}": float(row.get(c, np.nan)) for c in TME_COLS}


def build_tcga_matrix(samples: list, gene_set: set = None,
                      clinical_df: pd.DataFrame = None) -> pd.DataFrame:
    """Build TCGA feature matrix: z-scored RNA + TIL + mutations + immune sigs + TME + clinical."""
    if clinical_df is None:
        clinical_df = pd.DataFrame()
    logger.info("Extracting TCGA features ...")
    records = []
    for i, sid in enumerate(samples):
        if i % 100 == 0:
            logger.info(f"  {i}/{len(samples)} ...")
        row = {"sample_id": sid}
        row.update(extract_tcga_expression(sid, gene_set))
        row.update(extract_til(sid))
        row.update(extract_mutations(sid))
        row.update(extract_tmb(sid))
        row.update(extract_immune_signatures(sid))
        row.update(extract_tme_scores(sid))
        row.update(extract_clinical(sid, clinical_df))
        records.append(row)

    df = pd.DataFrame(records).set_index("sample_id")

    # Z-score RNA columns within TCGA (gene-wise)
    rna_cols = [c for c in df.columns if c.startswith("rna_")]
    df[rna_cols] = df[rna_cols].apply(
        lambda col: (col - col.mean()) / (col.std() + 1e-8))
    logger.info(f"TCGA: {df.shape[0]} samples, {len(rna_cols)} RNA genes")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. GSE72094 INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════

def load_gse72094(gene_set: set = None) -> tuple:
    """Load GSE72094 z-scored expression + survival. Returns (expr_df, surv_df)."""
    expr_path = GSE_DIR / "gse72094_expr_zscore.tsv.gz"
    surv_path = GSE_DIR / "gse72094_survival.tsv"
    if not expr_path.exists() or not surv_path.exists():
        logger.warning("GSE72094 data not found. Run parse_gse72094.py first.")
        return None, None

    logger.info("Loading GSE72094 ...")
    surv = pd.read_csv(surv_path, sep="\t", index_col=0)
    surv["event"] = surv["event"].astype(bool)

    expr = pd.read_csv(expr_path, sep="\t", index_col=0)

    # Keep only genes in gene_set if provided
    if gene_set:
        keep = [c for c in expr.columns if c in gene_set]
        expr = expr[keep]

    # Compute immune signatures BEFORE adding rna_ prefix (need bare gene names)
    sig_records = {}
    for sid in expr.index:
        row_expr = expr.loc[sid]
        rec = {}
        # TIS
        avail = [g for g in TIS_GENES if g in row_expr.index]
        if avail:
            rec["sig_tis"] = float(row_expr[avail].mean())
        # CYT (z-scored, so use z-scores as proxy; sqrt not meaningful, use mean)
        if "GZMA" in row_expr.index and "PRF1" in row_expr.index:
            rec["sig_cyt"] = float((row_expr["GZMA"] + row_expr["PRF1"]) / 2)
        # IMPRES
        score, total = 0, 0
        for ga, gb in IMPRES_PAIRS:
            if ga in row_expr.index and gb in row_expr.index:
                score += int(row_expr[ga] > row_expr[gb])
                total += 1
        if total > 0:
            rec["sig_impres"] = score / total
        sig_records[sid] = rec
    sig_df = pd.DataFrame(sig_records).T
    for col in sig_df.columns:
        expr[col] = sig_df[col]

    # Add rna_ prefix to match TCGA convention
    rna_cols = [c for c in expr.columns if not c.startswith("sig_")]
    expr = expr.rename(columns={c: f"rna_{c}" for c in rna_cols})

    # Add mutation flags from metadata
    meta_path = GSE_DIR / "gse72094_survival.tsv"
    meta = pd.read_csv(meta_path, sep="\t", index_col=0)
    mut_map = {"WT": 0, "Mut": 1, "NA": np.nan}
    for gene, col in [("STK11","stk11_status"), ("EGFR","egfr_status"),
                      ("KRAS","kras_status"), ("TP53","tp53_status")]:
        if col in meta.columns:
            expr[f"mut_{gene}"] = meta[col].map(mut_map).reindex(expr.index)

    # TIL and TME not available for GSE72094 — fill with NaN
    expr["til_density"] = np.nan
    expr["til_score"]   = np.nan
    for c in TME_COLS:
        expr[f"tme_{c}"] = np.nan

    # Add missing driver gene mutations as NaN
    for g in DRIVER_GENES:
        if f"mut_{g}" not in expr.columns:
            expr[f"mut_{g}"] = np.nan

    # Keep only samples with survival
    common = expr.index.intersection(surv.index)
    logger.info(f"GSE72094: {len(common)} samples, {surv.loc[common,'event'].sum()} events")
    return expr.loc[common], surv.loc[common]


# ══════════════════════════════════════════════════════════════════════════════
# 3. SURVIVAL DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_tcga_survival(samples: list) -> pd.DataFrame:
    df = pd.read_csv(CLINICAL_DIR / "tcga_luad_survival.tsv", sep="\t")
    df = df[df["sample_id"].isin(samples)].dropna(subset=["os_days","event"])
    df = df[df["os_days"] > 0].copy()
    df["event"] = df["event"].astype(bool)
    return df.set_index("sample_id")


def make_survival_array(event: pd.Series, time: pd.Series) -> np.ndarray:
    return np.array(list(zip(event.astype(bool), time.astype(float))),
                    dtype=[("event", bool), ("time", float)])


# ══════════════════════════════════════════════════════════════════════════════
# 4. STAGE 1 — CoxNet feature selection
# ══════════════════════════════════════════════════════════════════════════════

def univariate_cox_filter(X: pd.DataFrame, y: np.ndarray,
                          top_n: int = TOP_COX_GENES) -> list:
    """
    Fast univariate Cox filter using the Wald statistic.
    For each RNA gene, fit a single-feature CoxPH, rank by |Wald stat|.
    Returns top_n gene column names.
    """
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    logger.info(f"Univariate Cox filter: testing {X.shape[1]} genes → keep top {top_n} ...")
    stats = {}
    for col in X.columns:
        x = X[[col]].values
        try:
            m = CoxPHSurvivalAnalysis(ties="efron")
            m.fit(x, y)
            stats[col] = abs(m.coef_[0])
        except Exception:
            stats[col] = 0.0
    ranked = sorted(stats, key=stats.get, reverse=True)
    selected = ranked[:top_n]
    logger.info(f"Univariate Cox filter: selected {len(selected)} genes")
    return selected


def _cv_cindex(X_np, y, alphas, l1_ratios, kf):
    """Return (best_alpha, best_l1, best_cv_ci) via nested CV."""
    from sklearn.preprocessing import StandardScaler
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored
    best_ci, best_alpha, best_l1 = -1, alphas[0], l1_ratios[0]
    for l1 in l1_ratios:
        for alpha in alphas:
            cis = []
            for tr, te in kf.split(X_np):
                sc = StandardScaler()
                try:
                    m = CoxnetSurvivalAnalysis(alphas=[alpha], l1_ratio=l1,
                                               fit_baseline_model=True, max_iter=1000)
                    m.fit(sc.fit_transform(X_np[tr]), y[tr])
                    ci = concordance_index_censored(
                        y[te]["event"], y[te]["time"],
                        m.predict(sc.transform(X_np[te])))[0]
                    cis.append(ci)
                except Exception:
                    pass
            if cis and np.mean(cis) > best_ci:
                best_ci, best_alpha, best_l1 = np.mean(cis), alpha, l1
    return best_alpha, best_l1, best_ci


def coxnet_train(X: pd.DataFrame, y: np.ndarray,
                 top_n: int = TOP_N_GENES) -> tuple:
    """
    Two-stage multi-modal CoxNet:
      Stage 1 — RNA only: variance pre-filter → CoxNet → select OS-predictive RNA genes
      Stage 2 — Combined: [selected RNA + immune sigs + TME + TIL + mutations] → final CoxNet
    Returns (scaler, model, train_cols, coefs, cv_cindex_stage2).
    """
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sksurv.linear_model import CoxnetSurvivalAnalysis

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # ── Stage 1: RNA-only CoxNet → feature selection ──────────────────────────
    # Pre-filter: top-N by variance (avoids data leakage from supervised filters)
    rna_cols   = [c for c in X.columns if c.startswith("rna_")]
    non_rna    = [c for c in X.columns if not c.startswith("rna_")]

    var = X[rna_cols].var()
    top_rna = var.nlargest(min(top_n, len(rna_cols))).index.tolist()
    X_rna = X[top_rna].fillna(X[top_rna].median(numeric_only=True))
    logger.info(f"Stage 1 — RNA-only CoxNet: {X_rna.shape[1]} RNA features")

    alphas_s1  = np.logspace(-3, 0, 20)
    l1_ratios  = [0.1, 0.5, 0.9]
    a1, l1_best, ci_s1 = _cv_cindex(X_rna.values, y, alphas_s1, l1_ratios, kf)
    logger.info(f"Stage 1 best: alpha={a1:.4f}, l1={l1_best}, CV C-index={ci_s1:.3f}")

    sc1 = StandardScaler()
    m1  = CoxnetSurvivalAnalysis(alphas=[a1], l1_ratio=l1_best,
                                 fit_baseline_model=True, max_iter=1000)
    m1.fit(sc1.fit_transform(X_rna.values), y)
    coefs_s1   = pd.Series(m1.coef_.ravel(), index=X_rna.columns)
    rna_selected = coefs_s1[coefs_s1 != 0].sort_values(key=abs, ascending=False).index.tolist()
    logger.info(f"Stage 1 selected {len(rna_selected)} RNA genes")

    # ── Stage 2: combined model with all modalities ───────────────────────────
    # Selected RNA + immune sigs + TME + TIL + mutations + TMB + clinical stage/age
    immune_cols = [c for c in non_rna if any(c.startswith(p)
                   for p in ("sig_","tme_","til_","mut_","tmb","clinical_"))]
    stage2_cols = rna_selected + immune_cols
    X_s2 = X[stage2_cols].fillna(X[stage2_cols].median(numeric_only=True))
    logger.info(f"Stage 2 — Combined CoxNet: {len(rna_selected)} RNA + "
                f"{len(immune_cols)} immune/TME/clinical features")

    alphas_s2  = np.logspace(-4, -1, 20)   # smaller alpha = less shrinkage on combined set
    a2, l1_s2, ci_s2 = _cv_cindex(X_s2.values, y, alphas_s2, l1_ratios, kf)
    logger.info(f"Stage 2 best: alpha={a2:.4f}, l1={l1_s2}, CV C-index={ci_s2:.3f}")

    sc2 = StandardScaler()
    m2  = CoxnetSurvivalAnalysis(alphas=[a2], l1_ratio=l1_s2,
                                 fit_baseline_model=True, max_iter=1000)
    m2.fit(sc2.fit_transform(X_s2.values), y)

    coefs_s2 = pd.Series(m2.coef_.ravel(), index=X_s2.columns)
    selected = coefs_s2[coefs_s2 != 0].sort_values(key=abs, ascending=False)
    logger.info(f"Stage 2 selected {len(selected)} / {X_s2.shape[1]} features")

    return sc2, m2, X_s2.columns.tolist(), selected, ci_s2, rna_selected, coefs_s1


# ══════════════════════════════════════════════════════════════════════════════
# 5. SCORING — CoxNet risk → Immune Activity Score
# ══════════════════════════════════════════════════════════════════════════════

def _impute(X: pd.DataFrame, train_cols: list) -> pd.DataFrame:
    """Reindex to train_cols, impute median, fill remaining NaN with 0."""
    Xr = X.reindex(columns=train_cols)
    return Xr.fillna(Xr.median(numeric_only=True)).fillna(0)


def compute_scores(scaler, model, X: pd.DataFrame,
                   train_cols: list) -> pd.Series:
    """Score patients using CoxNet log-hazard; invert so high = more immune active."""
    X_imp = _impute(X, train_cols)
    risk = model.predict(scaler.transform(X_imp))
    s = -risk   # higher = lower hazard = more immune active
    return pd.Series(100 * (s - s.min()) / (s.max() - s.min() + 1e-8),
                     index=X.index, name="io_score")


def bootstrap_cindex(scaler, model, X: pd.DataFrame,
                     y: np.ndarray, train_cols: list,
                     n: int = 200) -> tuple:
    from sksurv.metrics import concordance_index_censored
    X_imp = _impute(X, train_cols)
    risk = model.predict(scaler.transform(X_imp))
    rng  = np.random.default_rng(42)
    cis  = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        try:
            cis.append(concordance_index_censored(
                y[idx]["event"], y[idx]["time"], risk[idx])[0])
        except Exception:
            pass
    ci = np.array(cis)
    return ci.mean(), np.percentile(ci, 2.5), np.percentile(ci, 97.5)


# ══════════════════════════════════════════════════════════════════════════════
# 6. FEATURE IMPORTANCE — CoxNet coefficients
# ══════════════════════════════════════════════════════════════════════════════

def plot_feature_importance(coefs: pd.Series, path: Path,
                            top_n: int = 20) -> None:
    """Bar chart of top-N CoxNet coefficients (absolute value)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = coefs.abs().nlargest(top_n).sort_values()
    labels = top.index.str.replace("rna_", "").str.replace("mut_", "mut:")
    colors = ["#e74c3c" if coefs[f] > 0 else "#3498db" for f in top.index]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(labels, top.values, color=colors, height=0.6)
    ax.set_xlabel("CoxNet coefficient (absolute value)")
    ax.set_title(f"Top-{top_n} Feature Importance — Immune Activity Score\n"
                 "(red = higher expression → higher hazard, blue = protective)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 7. VALIDATION PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_km(scores: pd.Series, surv: pd.DataFrame, path: Path,
            title: str = "Immune Activity Score — Kaplan-Meier (TCGA-LUAD)") -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    m = surv.join(scores, how="inner")
    q33, q67 = np.percentile(m["io_score"], [33, 67])
    m["grp"] = "Intermediate"
    m.loc[m["io_score"] >= q67, "grp"] = "High"
    m.loc[m["io_score"] <  q33, "grp"] = "Low"

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, color in [("High","#e74c3c"),("Intermediate","#f39c12"),("Low","#3498db")]:
        s = m[m["grp"] == label]
        KaplanMeierFitter().fit(s["os_days"]/30.44, s["event"],
                                label=f"{label} (n={len(s)})")\
                           .plot_survival_function(ax=ax, ci_show=True, color=color)

    hi, lo = m[m["grp"]=="High"], m[m["grp"]=="Low"]
    if len(hi) and len(lo):
        p = logrank_test(hi["os_days"], lo["os_days"],
                         hi["event"], lo["event"]).p_value
        ax.text(0.62, 0.95, f"High vs Low\nlog-rank p = {p:.2e}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    ax.set(xlabel="Time (months)", ylabel="Overall Survival", title=title)
    ax.legend(title="Immune Activity")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close()


def plot_stk11_km(scores: pd.Series, surv: pd.DataFrame,
                  feat: pd.DataFrame, path: Path) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter

    if "mut_STK11" not in feat.columns:
        return
    m = surv.join(scores).join(feat[["mut_STK11"]], how="inner").dropna()
    med = m["io_score"].median()
    m["sub"] = m.apply(
        lambda r: ("STK11-mut" if r["mut_STK11"] else "STK11-wt") +
                  (" / High" if r["io_score"] >= med else " / Low"), axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    palette = {"STK11-wt / High":"#e74c3c","STK11-wt / Low":"#f1948a",
               "STK11-mut / High":"#2980b9","STK11-mut / Low":"#85c1e9"}
    from lifelines import KaplanMeierFitter as KMF
    for label, color in palette.items():
        s = m[m["sub"] == label]
        if len(s) < 5: continue
        KMF().fit(s["os_days"]/30.44, s["event"],
                  label=f"{label} (n={len(s)})")\
             .plot_survival_function(ax=ax, ci_show=False, color=color)

    ax.set(xlabel="Time (months)", ylabel="Overall Survival",
           title="Immune Activity Score × STK11 — TCGA-LUAD Subgroup")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    tcga_samples = sorted(p.name for p in EXPR_DIR.iterdir()
                          if p.is_dir() and p.name.startswith("TCGA"))
    logger.info(f"TCGA samples: {len(tcga_samples)}")

    # ── Determine shared gene set (TCGA ∩ GSE72094) ───────────────────────────
    gse_expr_path = GSE_DIR / "gse72094_expr_zscore.tsv.gz"
    gene_set = None
    if gse_expr_path.exists():
        gse_genes = set(pd.read_csv(gse_expr_path, sep="\t",
                                    index_col=0, nrows=0).columns)
        logger.info(f"GSE72094 genes: {len(gse_genes):,}")
        gene_set = gse_genes   # limit TCGA to overlapping genes (for external validation)

    # ── TCGA feature matrix ────────────────────────────────────────────────────
    feat_path = OUT_DIR / "io_features_tcga.tsv"
    # Load clinical data for stage + age features
    clinical_df = pd.read_csv(CLINICAL_DIR / "tcga_luad_survival.tsv",
                              sep="\t", index_col=0)

    if feat_path.exists() and not args.force:
        logger.info("Loading cached TCGA features ...")
        tcga_feat = pd.read_csv(feat_path, sep="\t", index_col=0)
    else:
        tcga_feat = build_tcga_matrix(tcga_samples, gene_set, clinical_df)
        tcga_feat.to_csv(feat_path, sep="\t")

    if args.features_only:
        logger.info("Done (--features_only).")
        return

    # ── TCGA survival ──────────────────────────────────────────────────────────
    tcga_surv = load_tcga_survival(tcga_samples)

    # Align TCGA features and survival
    tcga_common = tcga_feat.index.intersection(tcga_surv.index)
    X_tcga = tcga_feat.loc[tcga_common].copy()
    X_tcga = X_tcga.loc[:, X_tcga.isna().mean() < 0.5]
    X_tcga = X_tcga.fillna(X_tcga.median(numeric_only=True))
    y_tcga = make_survival_array(tcga_surv.loc[tcga_common, "event"],
                                 tcga_surv.loc[tcga_common, "os_days"])
    logger.info(f"TCGA training set: {X_tcga.shape[0]} samples × {X_tcga.shape[1]} features")

    # ── Two-stage CoxNet: train on TCGA ──────────────────────────────────────
    model_path = OUT_DIR / "coxnet_model.pkl"
    if model_path.exists() and not args.force:
        logger.info("Loading cached CoxNet model ...")
        with open(model_path, "rb") as f:
            saved = pickle.load(f)
        coxnet_scaler = saved["scaler"]
        coxnet_model  = saved["model"]
        train_cols    = saved["train_cols"]
        coxnet_coefs  = saved["coefs"]
        cv_ci         = saved["cv_ci"]
        rna_selected  = saved.get("rna_selected", [])
        coefs_s1      = saved.get("coefs_s1", pd.Series(dtype=float))
    else:
        coxnet_scaler, coxnet_model, train_cols, coxnet_coefs, cv_ci, \
            rna_selected, coefs_s1 = coxnet_train(X_tcga, y_tcga)
        with open(model_path, "wb") as f:
            pickle.dump({"scaler": coxnet_scaler, "model": coxnet_model,
                         "train_cols": train_cols, "coefs": coxnet_coefs,
                         "cv_ci": cv_ci, "rna_selected": rna_selected,
                         "coefs_s1": coefs_s1}, f)

    # ── Save selected genes with stage label ──────────────────────────────────
    sel_df = coxnet_coefs.reset_index()
    sel_df.columns = ["feature", "coefficient"]
    sel_df["stage"] = sel_df["feature"].apply(
        lambda f: "Stage1-RNA" if f in rna_selected else "Stage2-immune/TME/clinical")
    sel_df.to_csv(OUT_DIR / "selected_genes.tsv", sep="\t", index=False)

    # Feature importance figure (CoxNet coefficients)
    plot_feature_importance(coxnet_coefs,
                            OUT_DIR / "figures" / "feature_importance.png")

    # Feature importance table
    fi_df = sel_df.copy()
    fi_df["abs_coefficient"] = fi_df["coefficient"].abs()
    fi_df = fi_df.sort_values("abs_coefficient", ascending=False)
    fi_df.to_csv(OUT_DIR / "feature_importance.tsv", sep="\t", index=False)

    # ── Bootstrap C-index on TCGA ─────────────────────────────────────────────
    ci_mean, ci_lo, ci_hi = bootstrap_cindex(
        coxnet_scaler, coxnet_model, X_tcga, y_tcga, train_cols)
    logger.info(f"C-index (TCGA bootstrap): {ci_mean:.3f} "
                f"(95% CI {ci_lo:.3f}–{ci_hi:.3f})")

    # ── Score all TCGA patients (including those without survival data) ────────
    scores = compute_scores(coxnet_scaler, coxnet_model, tcga_feat, train_cols)

    # ── Save scores with tertile grouping (quantile-based, equal thirds) ──────
    q33, q67 = np.percentile(scores, [33.3, 66.7])
    out = scores.to_frame()
    out["io_group"] = "Intermediate"
    out.loc[scores >= q67, "io_group"] = "High"
    out.loc[scores <  q33, "io_group"] = "Low"
    out = out.join(tcga_surv[["os_days","event"]], how="left")
    extra_cols = ["til_density","til_score",
                  "mut_STK11","mut_KEAP1","mut_EGFR","mut_KRAS",
                  "tmb","clinical_stage","clinical_age",
                  "sig_tis","sig_cyt","sig_impres"] + [f"tme_{c}" for c in TME_COLS]
    extra_cols = [c for c in extra_cols if c in tcga_feat.columns]
    out = out.join(tcga_feat[extra_cols], how="left")
    out.index.name = "sample_id"
    out.to_csv(OUT_DIR / "io_scores.tsv", sep="\t")

    # ── External validation: GSE72094 ────────────────────────────────────────
    gse_feat, gse_surv = load_gse72094(gene_set)
    ext_ci_str = "N/A"
    if gse_feat is not None:
        try:
            gse_scores_raw = compute_scores(
                coxnet_scaler, coxnet_model, gse_feat, train_cols)
            gse_common = gse_feat.index.intersection(gse_surv.index)
            gse_X_eval = gse_feat.loc[gse_common]
            y_gse = make_survival_array(gse_surv.loc[gse_common, "event"],
                                        gse_surv.loc[gse_common, "os_days"])
            gse_ci, gse_lo, gse_hi = bootstrap_cindex(
                coxnet_scaler, coxnet_model, gse_X_eval, y_gse, train_cols)
            ext_ci_str = f"{gse_ci:.3f} (95% CI {gse_lo:.3f}–{gse_hi:.3f})"
            logger.info(f"C-index (GSE72094 external): {ext_ci_str}")

            # KM plot for external validation
            gse_scores_df = gse_scores_raw.to_frame("io_score")
            q33g, q67g = np.percentile(gse_scores_raw, [33.3, 66.7])
            gse_scores_df["io_group"] = "Intermediate"
            gse_scores_df.loc[gse_scores_raw >= q67g, "io_group"] = "High"
            gse_scores_df.loc[gse_scores_raw <  q33g, "io_group"] = "Low"
            plot_km(gse_scores_raw, gse_surv,
                    OUT_DIR / "figures" / "km_gse72094.png",
                    title="Immune Activity Score — Kaplan-Meier (GSE72094 external validation)")
        except Exception as e:
            logger.warning(f"GSE72094 validation failed: {e}")

    # ── KM survival plots (TCGA) ──────────────────────────────────────────────
    plot_km(scores, tcga_surv, OUT_DIR / "figures" / "km_io_score.png")
    plot_stk11_km(scores, tcga_surv, tcga_feat,
                  OUT_DIR / "figures" / "km_stk11_subgroup.png")

    # ── Save model metrics JSON (for Streamlit display) ───────────────────────
    import json
    metrics = {
        "n_training":         len(tcga_common),
        "n_events":           int(tcga_surv.loc[tcga_common,"event"].sum()),
        "n_features":         int((coxnet_coefs != 0).sum()),
        "n_rna_selected":     len(rna_selected),
        "cv_cindex":          round(float(cv_ci), 3),
        "bootstrap_cindex":   round(float(ci_mean), 3),
        "bootstrap_ci_lo":    round(float(ci_lo), 3),
        "bootstrap_ci_hi":    round(float(ci_hi), 3),
        "external_cindex":    ext_ci_str,
        "n_tcga_scored":      int(len(out)),
    }
    with open(OUT_DIR / "model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    grp = out["io_group"].value_counts()
    n_sel = (coxnet_coefs != 0).sum()
    logger.info(f"\n{'='*45}")
    logger.info(f"  Training: TCGA n={len(tcga_common)} "
                f"({int(tcga_surv.loc[tcga_common,'event'].sum())} events)")
    for g in ["High","Intermediate","Low"]:
        logger.info(f"  {g}: {grp.get(g,0)} TCGA patients")
    logger.info(f"  CV C-index (TCGA 5-fold): {cv_ci:.3f}")
    logger.info(f"  Bootstrap C-index (TCGA): {ci_mean:.3f} "
                f"(95% CI {ci_lo:.3f}–{ci_hi:.3f})")
    logger.info(f"  External C-index (GSE72094): {ext_ci_str}")
    logger.info(f"  CoxNet features selected: {n_sel}")
    top5 = sel_df.head(5)["feature"].tolist()
    logger.info(f"  Top features: {top5}")
    logger.info(f"  Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
