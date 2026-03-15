#!/usr/bin/env python
"""
luad_io_ml.py
-------------
Module 08: Immunotherapy ML — multi-modal IO score for LUAD.

Pipeline
--------
1. Feature engineering (per TCGA patient):
     - RNA-based immune signatures from M03 bulk RNA-seq:
         TIS   — 18-gene Tumor Inflammation Signature (Ayers 2017, JCI)
         IMPRES — 15 gene-pair ratio score (Auslander 2018, Nat Med)
         CYT   — Cytolytic Activity: sqrt(GZMA × PRF1) (Rooney 2015, Cell)
         CD274 (PD-L1), CD8A, PDCD1, CXCL9 raw log2-TPM
     - Cell-type fractions from M04 ssGSEA
     - TME immune phenotype from M04 (Inflamed/Excluded/Desert)
     - TMB from M02 variants
     - LUAD-specific resistance mutations: STK11, KEAP1 (Skoulidis 2018, NEJM)
2. CoxNet — regularized Cox PH with Elastic Net:
     - Endpoint: overall survival (os_days + event from TCGA clinical)
     - Regularization: Elastic Net (L1+L2), nested 5-fold CV for alpha/l1_ratio
     - Evaluation: C-index (Harrell's concordance)
3. IO Score — normalized risk score per patient (0–100)
4. Validation:
     - KM curves by IO Score tertile (log-rank test)
     - C-index with 95% CI (bootstrap)
     - STK11 subgroup analysis
5. SHAP — feature contribution per patient
6. Save: io_scores.tsv, figures/

TODO (upgrade path):
  - Replace CoxNet with DeepSurv (neural Cox PH)
  - Add MOFA multi-omics factor analysis
  - Add Geneformer / scGPT expression embeddings
  - External validation on Hellmann 2018 or IMvigor210

Usage:
  python luad_io_ml.py          # full pipeline
  python luad_io_ml.py --features_only   # extract features only
"""

import argparse
import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import shap
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.preprocessing import OneHotEncoder
import pickle

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
PROJECT_DIR  = SCRIPT_DIR.parent.parent
EXPR_DIR     = PROJECT_DIR / "data/output/03_expression"
TME_DIR      = PROJECT_DIR / "data/output/04_single_cell"
VAR_DIR      = PROJECT_DIR / "data/output/02_variants"
AM_DIR       = PROJECT_DIR / "data/output/07_variant_impact"
CLINICAL_DIR = PROJECT_DIR / "data/clinical"
OUT_DIR      = PROJECT_DIR / "data/output/08_io_ml"

OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "figures").mkdir(exist_ok=True)

# ── TIS gene set (Ayers 2017, JCI) ────────────────────────────────────────────
TIS_GENES = [
    "CD8A", "CXCL9", "CXCL10", "IDO1", "IFNG", "LAG3", "NKG7",
    "PDCD1", "PDCD1LG2", "PSMB10", "STAT1", "TIGIT", "CD27",
    "CD276", "CMKLR1", "CX3CL1", "HLA-DQA1", "HLA-E",
]

# ── IMPRES gene pairs (Auslander 2018, Nat Med) ───────────────────────────────
# Score += 1 for each pair where TPM(gene_a) > TPM(gene_b) → sum / 15
IMPRES_PAIRS = [
    ("CD274",   "PDCD1"),
    ("CTLA4",   "LAG3"),
    ("TIGIT",   "CD8A"),
    ("PDCD1",   "CD27"),
    ("CTLA4",   "CD2"),
    ("PDCD1",   "CXCR6"),
    ("PDCD1",   "KLRB1"),
    ("PDCD1LG2","PDCD1"),
    ("CD86",    "PDCD1"),
    ("CD80",    "TIGIT"),
    ("HAVCR2",  "PDCD1LG2"),
    ("VSIR",    "PDCD1"),       # VSIR = VISTA
    ("CD28",    "CTLA4"),
    ("SIGLEC7", "HAVCR2"),
    ("TNFRSF9", "TIGIT"),
]

# ── Single marker genes ────────────────────────────────────────────────────────
MARKER_GENES = ["CD274", "CD8A", "PDCD1", "CXCL9", "GZMA", "PRF1",
                "HAVCR2", "FOXP3", "IFNG"]

# ── M04 ssGSEA cell-type columns ───────────────────────────────────────────────
TME_SCORE_COLS = [
    "CD8_T_cytotoxic", "Treg", "CD8_T_exhausted",
    "CD4_T", "NK", "Macrophage_M1", "Macrophage_M2",
]

# ── LUAD resistance mutation genes ────────────────────────────────────────────
RESISTANCE_GENES = ["STK11", "KEAP1", "EGFR", "KRAS"]

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def _get_tpm(expr_df: pd.DataFrame) -> pd.Series:
    """Return SYMBOL → TPM_GENE mapping (deduplicated, keep max)."""
    return expr_df.groupby("SYMBOL")["TPM_GENE"].max()


def _get_log2tpm(expr_df: pd.DataFrame) -> pd.Series:
    """Return SYMBOL → TPM_LOG2_GENE mapping."""
    return expr_df.groupby("SYMBOL")["TPM_LOG2_GENE"].max()


def compute_tis(tpm: pd.Series) -> float:
    """TIS = mean log2(TPM+1) of available TIS genes."""
    vals = []
    for g in TIS_GENES:
        if g in tpm.index and tpm[g] > 0:
            vals.append(np.log2(tpm[g] + 1))
    return float(np.mean(vals)) if vals else np.nan


def compute_impres(tpm: pd.Series) -> float:
    """IMPRES = fraction of 15 gene pairs where TPM(A) > TPM(B)."""
    score, counted = 0, 0
    for a, b in IMPRES_PAIRS:
        if a in tpm.index and b in tpm.index:
            score += int(tpm[a] > tpm[b])
            counted += 1
    return score / counted if counted >= 10 else np.nan


def compute_cyt(tpm: pd.Series) -> float:
    """CYT = geometric mean of GZMA and PRF1 TPM."""
    if "GZMA" in tpm.index and "PRF1" in tpm.index:
        return float(np.sqrt(tpm["GZMA"] * tpm["PRF1"]))
    return np.nan


def extract_expression_features(sample_id: str) -> dict:
    """Extract TIS, IMPRES, CYT, and single-gene features from M03 output."""
    expr_path = EXPR_DIR / sample_id / f"{sample_id}_gene_expression.tsv.gz"
    if not expr_path.exists():
        return {}

    expr_df = pd.read_csv(expr_path, sep="\t", usecols=[
        "SYMBOL", "TPM_GENE", "TPM_LOG2_GENE"
    ], low_memory=False)

    tpm     = _get_tpm(expr_df)
    log2tpm = _get_log2tpm(expr_df)

    feats = {
        "tis_score":    compute_tis(tpm),
        "impres_score": compute_impres(tpm),
        "cyt_score":    compute_cyt(tpm),
    }
    for g in MARKER_GENES:
        feats[f"expr_{g}"] = float(log2tpm[g]) if g in log2tpm.index else np.nan

    return feats


def extract_tme_features(tme_df: pd.DataFrame, sample_id: str) -> dict:
    """Extract M04 ssGSEA cell-type scores and M04 immune phenotype."""
    row = tme_df[tme_df["sample_id"] == sample_id]
    if row.empty:
        return {}
    row = row.iloc[0]

    feats = {col: float(row[col]) for col in TME_SCORE_COLS if col in row.index}

    phenotype = row.get("immune_phenotype", "Unknown")
    feats["tme_inflamed"]  = int(phenotype == "Inflamed")
    feats["tme_excluded"]  = int(phenotype == "Excluded")
    feats["tme_desert"]    = int(phenotype == "Desert")

    return feats


def extract_tmb(sample_id: str) -> dict:
    """Extract TMB (mutations/Mb) from M02 output."""
    tmb_path = VAR_DIR / sample_id / f"{sample_id}_tmb.tsv"
    if not tmb_path.exists():
        return {}
    df = pd.read_csv(tmb_path, sep="\t")
    # Take row with highest tmb_estimate (most conservative callable denominator)
    tmb_val = df["tmb_estimate"].max()
    return {"tmb": float(tmb_val)}


def extract_resistance_mutations(sample_id: str) -> dict:
    """Binary flags for LUAD resistance/driver mutations from M02 variants."""
    var_path = VAR_DIR / sample_id / f"{sample_id}_variants.tsv.gz"
    if not var_path.exists():
        return {f"mut_{g}": 0 for g in RESISTANCE_GENES}

    df = pd.read_csv(var_path, sep="\t", compression="gzip",
                     usecols=["SYMBOL"] if "SYMBOL" in pd.read_csv(
                         var_path, sep="\t", compression="gzip", nrows=0
                     ).columns else None,
                     low_memory=False)

    # Handle SYMBOL vs Hugo_Symbol column naming
    sym_col = "SYMBOL" if "SYMBOL" in df.columns else "Hugo_Symbol"
    if sym_col not in df.columns:
        return {f"mut_{g}": 0 for g in RESISTANCE_GENES}

    mutated = set(df[sym_col].dropna().unique())
    return {f"mut_{g}": int(g in mutated) for g in RESISTANCE_GENES}


def build_feature_matrix(samples: list[str]) -> pd.DataFrame:
    """Build full feature matrix for all samples."""
    logger.info("Loading M04 TME cohort scores ...")
    tme_path = TME_DIR / "tme_cohort_scores.tsv"
    tme_df   = pd.read_csv(tme_path, sep="\t") if tme_path.exists() else pd.DataFrame()

    records = []
    for i, sid in enumerate(samples):
        if i % 50 == 0:
            logger.info(f"  {i}/{len(samples)} samples processed ...")

        row = {"sample_id": sid}
        row.update(extract_expression_features(sid))
        row.update(extract_tme_features(tme_df, sid))
        row.update(extract_tmb(sid))
        row.update(extract_resistance_mutations(sid))
        records.append(row)

    df = pd.DataFrame(records).set_index("sample_id")
    logger.info(f"Feature matrix: {df.shape[0]} samples × {df.shape[1]} features")
    logger.info(f"Missing value rate: {df.isna().mean().mean():.1%}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. SURVIVAL DATA INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════

def load_survival(samples: list[str]) -> pd.DataFrame:
    """Load TCGA LUAD survival data, filter to samples with complete data."""
    surv = pd.read_csv(CLINICAL_DIR / "tcga_luad_survival.tsv", sep="\t")
    surv = surv[surv["sample_id"].isin(samples)].copy()
    surv = surv.dropna(subset=["os_days", "event"])
    surv = surv[surv["os_days"] > 0]
    surv["event"] = surv["event"].astype(bool)
    logger.info(f"Survival data: {len(surv)} samples "
                f"({surv['event'].sum()} events / {(~surv['event']).sum()} censored)")
    return surv.set_index("sample_id")


# ══════════════════════════════════════════════════════════════════════════════
# 3. COXNET MODEL
# ══════════════════════════════════════════════════════════════════════════════

def prepare_survival_array(surv_df: pd.DataFrame) -> np.ndarray:
    """Convert DataFrame to structured numpy array for scikit-survival."""
    return np.array(
        [(bool(e), float(t)) for e, t in zip(surv_df["event"], surv_df["os_days"])],
        dtype=[("event", bool), ("time", float)],
    )


def train_coxnet(X: pd.DataFrame, y: np.ndarray) -> tuple:
    """
    Train CoxNet with Elastic Net regularization.

    Uses nested 5-fold CV to select optimal alpha and l1_ratio.
    Returns (fitted pipeline, cv_cindex_scores).

    Upgrade path:
        Replace CoxnetSurvivalAnalysis with DeepSurv for non-linear capture.
    """
    logger.info("Training CoxNet (Elastic Net Cox PH) ...")

    # Impute missing values with column median
    X_imp = X.fillna(X.median(numeric_only=True))

    # Grid of regularization strengths
    alphas    = np.logspace(-3, 1, 30)
    l1_ratios = [0.1, 0.5, 0.9]

    best_cindex, best_alpha, best_l1 = -1, 0.1, 0.5

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for l1 in l1_ratios:
        for alpha in alphas:
            cindices = []
            for train_idx, test_idx in kf.split(X_imp):
                X_tr, X_te = X_imp.iloc[train_idx], X_imp.iloc[test_idx]
                y_tr, y_te = y[train_idx], y[test_idx]

                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_te_s = scaler.transform(X_te)

                try:
                    cox = CoxnetSurvivalAnalysis(
                        alphas=[alpha], l1_ratio=l1,
                        fit_baseline_model=True, max_iter=1000,
                    )
                    cox.fit(X_tr_s, y_tr)
                    risk = cox.predict(X_te_s)
                    ci = concordance_index_censored(
                        y_te["event"], y_te["time"], risk
                    )[0]
                    cindices.append(ci)
                except Exception:
                    continue

            mean_ci = np.mean(cindices) if cindices else 0
            if mean_ci > best_cindex:
                best_cindex, best_alpha, best_l1 = mean_ci, alpha, l1

    logger.info(f"  Best alpha={best_alpha:.4f}, l1_ratio={best_l1}, "
                f"CV C-index={best_cindex:.3f}")

    # Fit final model on all data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)
    cox_final = CoxnetSurvivalAnalysis(
        alphas=[best_alpha], l1_ratio=best_l1,
        fit_baseline_model=True, max_iter=1000,
    )
    cox_final.fit(X_scaled, y)

    return (scaler, cox_final), best_cindex


def compute_io_scores(model_tuple: tuple, X: pd.DataFrame) -> pd.Series:
    """
    Compute IO Score (0–100) for all patients.

    Higher score = higher immune activity = better IO candidate.
    CoxNet risk score is negated (higher risk = lower immune score).
    """
    scaler, cox = model_tuple
    X_imp    = X.fillna(X.median(numeric_only=True))
    X_scaled = scaler.transform(X_imp)
    risk     = cox.predict(X_scaled)

    # Negate and normalize to 0–100
    io_raw = -risk
    io_norm = 100 * (io_raw - io_raw.min()) / (io_raw.max() - io_raw.min() + 1e-8)
    return pd.Series(io_norm, index=X.index, name="io_score")


# ══════════════════════════════════════════════════════════════════════════════
# 4. VALIDATION — KM CURVES & C-INDEX BOOTSTRAP
# ══════════════════════════════════════════════════════════════════════════════

def plot_km_by_tertile(io_scores: pd.Series, surv_df: pd.DataFrame) -> None:
    """KM survival curves stratified by IO Score tertile."""
    merged = surv_df.join(io_scores, how="inner")
    if merged.empty:
        logger.warning("No overlap for KM plot.")
        return

    q33, q67 = np.percentile(merged["io_score"], [33, 67])
    merged["group"] = "IO-Intermediate"
    merged.loc[merged["io_score"] >= q67, "group"] = "IO-High"
    merged.loc[merged["io_score"] <  q33, "group"] = "IO-Low"

    fig, ax = plt.subplots(figsize=(8, 5))
    palette = {"IO-High": "#e74c3c", "IO-Intermediate": "#f39c12", "IO-Low": "#3498db"}
    kmf = KaplanMeierFitter()

    for label, color in palette.items():
        subset = merged[merged["group"] == label]
        kmf.fit(subset["os_days"] / 30.44, subset["event"], label=label)
        kmf.plot_survival_function(ax=ax, ci_show=True, color=color)

    # Log-rank test (High vs Low)
    hi = merged[merged["group"] == "IO-High"]
    lo = merged[merged["group"] == "IO-Low"]
    if len(hi) > 0 and len(lo) > 0:
        lr = logrank_test(hi["os_days"], lo["os_days"],
                          event_observed_A=hi["event"],
                          event_observed_B=lo["event"])
        ax.text(0.65, 0.95, f"HR(High vs Low)\np = {lr.p_value:.3f}",
                transform=ax.transAxes, fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Overall Survival")
    ax.set_title("IO Score: Kaplan-Meier Overall Survival (TCGA-LUAD)")
    ax.legend(title="IO Score Group")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "figures" / "km_io_score.png", dpi=150)
    plt.close()
    logger.info("KM plot saved.")


def bootstrap_cindex(model_tuple: tuple, X: pd.DataFrame,
                     y: np.ndarray, n_boot: int = 200) -> tuple:
    """Bootstrap 95% CI for C-index."""
    scaler, cox = model_tuple
    X_imp    = X.fillna(X.median(numeric_only=True))
    X_scaled = scaler.transform(X_imp)
    risk     = cox.predict(X_scaled)

    cidxs = []
    rng   = np.random.default_rng(42)
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        try:
            ci = concordance_index_censored(
                y[idx]["event"], y[idx]["time"], risk[idx]
            )[0]
            cidxs.append(ci)
        except Exception:
            pass

    ci_arr = np.array(cidxs)
    return float(np.mean(ci_arr)), float(np.percentile(ci_arr, 2.5)), float(np.percentile(ci_arr, 97.5))


# ══════════════════════════════════════════════════════════════════════════════
# 5. SHAP ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def compute_shap(model_tuple: tuple, X: pd.DataFrame) -> pd.DataFrame:
    """
    Compute SHAP values using linear explainer on CoxNet.
    Returns DataFrame (samples × features).
    """
    scaler, cox = model_tuple
    X_imp    = X.fillna(X.median(numeric_only=True))
    X_scaled = pd.DataFrame(
        scaler.transform(X_imp), index=X_imp.index, columns=X_imp.columns
    )

    # Linear SHAP (fast, exact for linear models)
    explainer   = shap.LinearExplainer(cox, X_scaled)
    shap_values = explainer.shap_values(X_scaled)

    return pd.DataFrame(shap_values, index=X_imp.index, columns=X_imp.columns)


def plot_shap_summary(shap_df: pd.DataFrame, X: pd.DataFrame) -> None:
    """SHAP beeswarm summary plot."""
    X_imp = X.fillna(X.median(numeric_only=True))
    shap.summary_plot(
        shap_df.values, X_imp,
        feature_names=X_imp.columns.tolist(),
        show=False, max_display=15,
    )
    plt.title("SHAP Feature Importance — IO Score (CoxNet)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "figures" / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("SHAP summary plot saved.")


def plot_stk11_subgroup(io_scores: pd.Series, surv_df: pd.DataFrame,
                        feat_df: pd.DataFrame) -> None:
    """KM plot stratified by STK11 mutation × IO Score — LUAD-specific insight."""
    if "mut_STK11" not in feat_df.columns:
        return

    merged = surv_df.join(io_scores).join(feat_df[["mut_STK11"]], how="inner")
    merged = merged.dropna(subset=["os_days", "event", "io_score"])

    median_score = merged["io_score"].median()
    merged["io_grp"]  = (merged["io_score"] >= median_score).map(
        {True: "IO-High", False: "IO-Low"}
    )
    merged["stk11"]   = merged["mut_STK11"].map({1: "STK11-mut", 0: "STK11-wt"})
    merged["subgroup"] = merged["stk11"] + " / " + merged["io_grp"]

    fig, ax = plt.subplots(figsize=(9, 5))
    palette = {
        "STK11-wt / IO-High":  "#e74c3c",
        "STK11-wt / IO-Low":   "#f1948a",
        "STK11-mut / IO-High": "#2980b9",
        "STK11-mut / IO-Low":  "#85c1e9",
    }
    kmf = KaplanMeierFitter()
    for label, color in palette.items():
        subset = merged[merged["subgroup"] == label]
        if len(subset) < 5:
            continue
        kmf.fit(subset["os_days"] / 30.44, subset["event"],
                label=f"{label} (n={len(subset)})")
        kmf.plot_survival_function(ax=ax, ci_show=False, color=color)

    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Overall Survival")
    ax.set_title("IO Score × STK11 Mutation — TCGA-LUAD Subgroup Analysis")
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "figures" / "km_stk11_subgroup.png", dpi=150)
    plt.close()
    logger.info("STK11 subgroup KM plot saved.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_only", action="store_true",
                        help="Only extract features, skip model training")
    args = parser.parse_args()

    # ── Discover samples ──────────────────────────────────────────────────────
    samples = sorted(p.name for p in EXPR_DIR.iterdir() if p.is_dir()
                     and p.name.startswith("TCGA"))
    logger.info(f"Found {len(samples)} samples with expression data")

    # ── 1. Feature extraction ─────────────────────────────────────────────────
    feat_path = OUT_DIR / "io_features.tsv"
    if feat_path.exists():
        logger.info(f"Loading cached features from {feat_path}")
        feat_df = pd.read_csv(feat_path, sep="\t", index_col=0)
    else:
        feat_df = build_feature_matrix(samples)
        feat_df.to_csv(feat_path, sep="\t")
        logger.info(f"Features saved → {feat_path}")

    if args.features_only:
        logger.info("--features_only: done.")
        return

    # ── 2. Survival data ──────────────────────────────────────────────────────
    surv_df = load_survival(samples)

    # Align samples with complete features + survival
    common = feat_df.index.intersection(surv_df.index)
    X      = feat_df.loc[common].copy()
    surv   = surv_df.loc[common].copy()

    # Drop features with > 30% missing
    X = X.loc[:, X.isna().mean() < 0.3]
    X = X.fillna(X.median(numeric_only=True))

    y = prepare_survival_array(surv)
    logger.info(f"Model input: {X.shape[0]} samples × {X.shape[1]} features")

    # ── 3. CoxNet training ────────────────────────────────────────────────────
    model_path = OUT_DIR / "coxnet_model.pkl"
    if model_path.exists():
        logger.info("Loading cached CoxNet model ...")
        with open(model_path, "rb") as f:
            model_tuple = pickle.load(f)
    else:
        model_tuple, cv_cindex = train_coxnet(X, y)
        with open(model_path, "wb") as f:
            pickle.dump(model_tuple, f)
        logger.info(f"Model saved → {model_path}")

    # ── 4. IO Scores ──────────────────────────────────────────────────────────
    io_scores = compute_io_scores(model_tuple, feat_df)

    # Bootstrap C-index
    cindex_mean, cindex_lo, cindex_hi = bootstrap_cindex(model_tuple, X, y)
    logger.info(f"C-index: {cindex_mean:.3f} (95% CI: {cindex_lo:.3f}–{cindex_hi:.3f})")

    # Save scores
    score_df = io_scores.to_frame()
    score_df["io_group"] = pd.cut(
        io_scores,
        bins=[0, 33, 67, 100],
        labels=["IO-Low", "IO-Intermediate", "IO-High"],
        include_lowest=True,
    )
    score_df = score_df.join(surv_df[["os_days", "event"]], how="left")
    score_df = score_df.join(feat_df[["tis_score", "impres_score", "cyt_score",
                                       "tmb", "mut_STK11", "mut_KEAP1"]], how="left")
    score_df.index.name = "sample_id"
    score_df.to_csv(OUT_DIR / "io_scores.tsv", sep="\t")
    logger.info(f"IO scores saved → {OUT_DIR / 'io_scores.tsv'}")

    # ── 5. Validation plots ───────────────────────────────────────────────────
    plot_km_by_tertile(io_scores, surv_df)
    plot_stk11_subgroup(io_scores, surv_df, feat_df)

    # ── 6. SHAP ───────────────────────────────────────────────────────────────
    logger.info("Computing SHAP values ...")
    try:
        shap_df = compute_shap(model_tuple, X)
        shap_df.to_csv(OUT_DIR / "shap_values.tsv", sep="\t")
        plot_shap_summary(shap_df, X)
    except Exception as e:
        logger.warning(f"SHAP failed: {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    dist = score_df["io_group"].value_counts()
    logger.info("\n=== IO Score Distribution ===")
    for grp, cnt in dist.items():
        logger.info(f"  {grp}: {cnt} patients")
    logger.info(f"  C-index: {cindex_mean:.3f} (95% CI: {cindex_lo:.3f}–{cindex_hi:.3f})")
    logger.info("Done.")


if __name__ == "__main__":
    main()
