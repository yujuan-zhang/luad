#!/usr/bin/env python
"""
luad_io_ml.py  —  Module 08: Immunotherapy ML
==============================================
Multi-modal IO immune activity score for LUAD.

Features (data-driven, not pre-compressed signatures)
------------------------------------------------------
  RNA   : ~270 immune genes from M03 bulk RNA-seq (log2-TPM)
           CoxNet L1 automatically selects relevant genes
  Pathology : TIL density + TIL score from M05 histopathology
  Genomics  : STK11, KEAP1, EGFR, KRAS mutation flags from M02

Model
-----
  CoxNet (Elastic Net Cox PH) — nested 5-fold CV for alpha / l1_ratio
  Endpoint: overall survival (TCGA clinical data)
  Evaluation: bootstrap C-index (95% CI)

Outputs
-------
  data/output/08_io_ml/
    io_scores.tsv          — per-patient IO score (0-100) + group
    selected_genes.tsv     — genes selected by CoxNet (coeff ≠ 0)
    coxnet_model.pkl
    figures/
      km_io_score.png
      km_stk11_subgroup.png

Upgrade path
------------
  - Replace CoxNet with DeepSurv (neural Cox PH)
  - Add MOFA multi-omics factor analysis
  - External IO cohort validation (IMvigor210, GSE135222)

Usage
-----
  python luad_io_ml.py                # full pipeline
  python luad_io_ml.py --features_only
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
PATHO_DIR    = PROJECT_DIR / "data/output/05_pathology"
VAR_DIR      = PROJECT_DIR / "data/output/02_variants"
CLINICAL_DIR = PROJECT_DIR / "data/clinical"
OUT_DIR      = PROJECT_DIR / "data/output/08_io_ml"
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "figures").mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ── Immune gene list (~270 genes, curated from ImmPort + literature) ───────────
# Categories: T cell, exhaustion, cytotoxicity, checkpoints, cytokines,
#             chemokines, NK, B cell, macrophage, DC, Treg, antigen presentation
IMMUNE_GENES = sorted(set([
    # T cell markers
    "CD3D","CD3E","CD3G","CD8A","CD8B","CD4","CD28","ICOS","CD27","CD7",
    # Exhaustion / dysfunction
    "PDCD1","CTLA4","LAG3","TIGIT","HAVCR2","TOX","TOX2","ENTPD1","EOMES",
    # Cytotoxicity
    "GZMA","GZMB","GZMH","GZMK","GZMM","PRF1","NKG7","GNLY","FASLG",
    # Checkpoint ligands
    "CD274","PDCD1LG2","CD80","CD86","VSIR","ADORA2A","LGALS9",
    # Cytokines
    "IFNG","TNF","IL2","IL10","TGFB1","TGFB2","IL6","IL12A","IL12B",
    "IL15","IL18","IL21","IL33","VEGFA","VEGFB",
    # Chemokines
    "CXCL9","CXCL10","CXCL11","CXCL13","CCL2","CCL3","CCL4","CCL5",
    "CCL19","CCL21","CX3CL1","CXCL1","CXCL2","CXCL5","CXCL8",
    # NK cell
    "KLRB1","KLRD1","KLRK1","KLRC1","KLRC2","NCR1","NCR3","FCGR3A",
    # B cell
    "CD19","MS4A1","CD79A","CD79B","IGHM","IGKC",
    # Macrophage / monocyte
    "CD68","CD163","MRC1","ARG1","NOS2","CSF1R","ITGAM","FCGR1A",
    "MARCO","MSR1","MMP9","VEGFA",
    # Dendritic cell
    "ITGAX","CLEC9A","XCR1","BATF3","IRF8","SIGLEC1","CCR7",
    # Treg
    "FOXP3","IL2RA","IKZF2","TNFRSF18","ENTPD1",
    # Antigen presentation
    "HLA-A","HLA-B","HLA-C","HLA-DRA","HLA-DRB1","HLA-DQA1","HLA-DQB1",
    "HLA-DPA1","HLA-DPB1","B2M","TAP1","TAP2","TAPBP","NLRC5",
    # Complement
    "C1QA","C1QB","C1QC","C3","C3AR1","C5AR1",
    # Interferon signaling
    "STAT1","STAT2","STAT3","IRF1","IRF3","IRF7","MX1","OAS1","OAS2",
    "IFIT1","IFIT2","IFIT3","ISG15","ISG20",
    # Immune metabolism / suppression
    "IDO1","IDO2","HAVCR2","TDO2","AREG","IL4I1","PTGES",
    # Inflammation / NF-kB
    "NFKB1","NFKB2","RELA","RELB","IL1B","IL1A","CXCL1",
    # Co-stimulation
    "TNFRSF9","TNFRSF4","CD40LG","CD40","TNFSF9","TNFSF4",
    # Myeloid / MDSC markers
    "S100A8","S100A9","S100A12","CEBPB","ARG2","IL4R",
    # TIS genes (ensure all included)
    "CD8A","CXCL9","CXCL10","IDO1","IFNG","LAG3","NKG7",
    "PDCD1","PDCD1LG2","PSMB10","STAT1","TIGIT","CD27",
    "CD276","CMKLR1","CX3CL1","HLA-DQA1","HLA-E",
    # IMPRES genes
    "CD274","CTLA4","TIGIT","CD8A","PDCD1","CD27","CD2","CXCR6",
    "KLRB1","PDCD1LG2","CD86","CD80","HAVCR2","VSIR","CD28",
    "SIGLEC7","TNFRSF9",
]))


# ── LUAD resistance / driver mutations ────────────────────────────────────────
DRIVER_GENES = ["STK11", "KEAP1", "EGFR", "KRAS", "TP53", "SMAD4"]


# ══════════════════════════════════════════════════════════════════════════════
# 1. FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_immune_expression(sample_id: str) -> dict:
    """Log2-TPM for each immune gene from M03 output."""
    path = EXPR_DIR / sample_id / f"{sample_id}_gene_expression.tsv.gz"
    if not path.exists():
        return {}
    df = pd.read_csv(path, sep="\t",
                     usecols=["SYMBOL", "TPM_LOG2_GENE"], low_memory=False)
    expr = df.groupby("SYMBOL")["TPM_LOG2_GENE"].max()
    return {f"rna_{g}": float(expr[g]) for g in IMMUNE_GENES if g in expr.index}


def extract_til(sample_id: str) -> dict:
    """TIL density and TIL score from M05 pathology."""
    path = PATHO_DIR / sample_id / f"{sample_id}_pathology_scores.tsv"
    if not path.exists():
        return {}
    row = pd.read_csv(path, sep="\t").iloc[0]
    return {
        "til_density": float(row.get("til_density", np.nan)),
        "til_score":   float(row.get("til_score",   np.nan)),
    }


def extract_mutations(sample_id: str) -> dict:
    """Binary mutation flags for LUAD driver/resistance genes."""
    path = VAR_DIR / sample_id / f"{sample_id}_variants.tsv.gz"
    if not path.exists():
        return {f"mut_{g}": 0 for g in DRIVER_GENES}
    df = pd.read_csv(path, sep="\t", compression="gzip", low_memory=False)
    sym_col = next((c for c in ["SYMBOL", "Hugo_Symbol"] if c in df.columns), None)
    if sym_col is None:
        return {f"mut_{g}": 0 for g in DRIVER_GENES}
    mutated = set(df[sym_col].dropna())
    return {f"mut_{g}": int(g in mutated) for g in DRIVER_GENES}


def build_feature_matrix(samples: list) -> pd.DataFrame:
    """Build full feature matrix: RNA immune genes + TIL + mutations."""
    records = []
    for i, sid in enumerate(samples):
        if i % 100 == 0:
            logger.info(f"  {i}/{len(samples)} ...")
        row = {"sample_id": sid}
        row.update(extract_immune_expression(sid))
        row.update(extract_til(sid))
        row.update(extract_mutations(sid))
        records.append(row)

    df = pd.DataFrame(records).set_index("sample_id")
    rna_cols   = [c for c in df.columns if c.startswith("rna_")]
    other_cols = [c for c in df.columns if not c.startswith("rna_")]
    logger.info(f"Features: {len(rna_cols)} RNA + {len(other_cols)} clinical/pathology")
    logger.info(f"Missing: RNA {df[rna_cols].isna().mean().mean():.1%}, "
                f"TIL {df['til_density'].isna().mean():.1%}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def load_survival(samples) -> pd.DataFrame:
    df = pd.read_csv(CLINICAL_DIR / "tcga_luad_survival.tsv", sep="\t")
    df = df[df["sample_id"].isin(samples)].dropna(subset=["os_days", "event"])
    df = df[df["os_days"] > 0].copy()
    df["event"] = df["event"].astype(bool)
    df = df.set_index("sample_id")
    logger.info(f"Survival: {len(df)} samples, {df['event'].sum()} events")
    return df


def train_coxnet(X: pd.DataFrame, y: np.ndarray) -> tuple:
    """Nested 5-fold CV CoxNet. Returns (scaler, model, cv_cindex)."""
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored

    alphas    = np.logspace(-3, 1, 20)
    l1_ratios = [0.1, 0.5, 0.9]
    best_ci, best_alpha, best_l1 = -1, 0.1, 0.5

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for l1 in l1_ratios:
        for alpha in alphas:
            cis = []
            for tr, te in kf.split(X):
                sc = StandardScaler()
                Xtr = sc.fit_transform(X.iloc[tr])
                Xte = sc.transform(X.iloc[te])
                try:
                    m = CoxnetSurvivalAnalysis(alphas=[alpha], l1_ratio=l1,
                                               fit_baseline_model=True, max_iter=1000)
                    m.fit(Xtr, y[tr])
                    ci = concordance_index_censored(
                        y[te]["event"], y[te]["time"], m.predict(Xte))[0]
                    cis.append(ci)
                except Exception:
                    pass
            if cis and np.mean(cis) > best_ci:
                best_ci, best_alpha, best_l1 = np.mean(cis), alpha, l1

    logger.info(f"Best: alpha={best_alpha:.4f}, l1_ratio={best_l1}, CV C-index={best_ci:.3f}")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = CoxnetSurvivalAnalysis(alphas=[best_alpha], l1_ratio=best_l1,
                                   fit_baseline_model=True, max_iter=1000)
    model.fit(Xs, y)
    return scaler, model, best_ci


def get_selected_features(model, feature_names: list) -> pd.DataFrame:
    """Return features with non-zero CoxNet coefficients."""
    coefs = model.coef_.ravel()
    df = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
    return df[df["coefficient"] != 0].sort_values("coefficient", key=abs, ascending=False)


def io_score(scaler, model, X: pd.DataFrame) -> pd.Series:
    """Normalize negated risk score to 0-100."""
    risk = model.predict(scaler.transform(X.fillna(X.median(numeric_only=True))))
    s = -risk
    return pd.Series(100 * (s - s.min()) / (s.max() - s.min() + 1e-8),
                     index=X.index, name="io_score")


def bootstrap_cindex(scaler, model, X: pd.DataFrame,
                     y: np.ndarray, n=200) -> tuple:
    from sksurv.metrics import concordance_index_censored
    Xs   = scaler.transform(X.fillna(X.median(numeric_only=True)))
    risk = model.predict(Xs)
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
# 3. VALIDATION PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_km(scores: pd.Series, surv: pd.DataFrame, path: Path) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    m = surv.join(scores, how="inner")
    q33, q67 = np.percentile(m["io_score"], [33, 67])
    m["grp"] = "IO-Intermediate"
    m.loc[m["io_score"] >= q67, "grp"] = "IO-High"
    m.loc[m["io_score"] <  q33, "grp"] = "IO-Low"

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, color in [("IO-High","#e74c3c"),("IO-Intermediate","#f39c12"),("IO-Low","#3498db")]:
        s = m[m["grp"] == label]
        KaplanMeierFitter().fit(s["os_days"]/30.44, s["event"], label=label)\
                           .plot_survival_function(ax=ax, ci_show=True, color=color)

    hi, lo = m[m["grp"]=="IO-High"], m[m["grp"]=="IO-Low"]
    if len(hi) and len(lo):
        p = logrank_test(hi["os_days"], lo["os_days"],
                         hi["event"], lo["event"]).p_value
        ax.text(0.65, 0.95, f"IO-High vs IO-Low\np = {p:.3f}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    ax.set(xlabel="Time (months)", ylabel="Overall Survival",
           title="IO Score — Kaplan-Meier (TCGA-LUAD)")
    ax.legend(title="IO Score Group")
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
                  (" / IO-High" if r["io_score"] >= med else " / IO-Low"), axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    palette = {"STK11-wt / IO-High":"#e74c3c","STK11-wt / IO-Low":"#f1948a",
               "STK11-mut / IO-High":"#2980b9","STK11-mut / IO-Low":"#85c1e9"}
    for label, color in palette.items():
        s = m[m["sub"] == label]
        if len(s) < 5: continue
        KaplanMeierFitter().fit(s["os_days"]/30.44, s["event"],
                                label=f"{label} (n={len(s)})")\
                           .plot_survival_function(ax=ax, ci_show=False, color=color)

    ax.set(xlabel="Time (months)", ylabel="Overall Survival",
           title="IO Score × STK11 — TCGA-LUAD Subgroup")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_only", action="store_true")
    parser.add_argument("--force", action="store_true", help="Ignore cached files")
    args = parser.parse_args()

    samples = sorted(p.name for p in EXPR_DIR.iterdir()
                     if p.is_dir() and p.name.startswith("TCGA"))
    logger.info(f"{len(samples)} samples found")

    # ── Features ──────────────────────────────────────────────────────────────
    feat_path = OUT_DIR / "io_features.tsv"
    if feat_path.exists() and not args.force:
        logger.info("Loading cached features ...")
        feat_df = pd.read_csv(feat_path, sep="\t", index_col=0)
    else:
        logger.info("Extracting features ...")
        feat_df = build_feature_matrix(samples)
        feat_df.to_csv(feat_path, sep="\t")

    if args.features_only:
        logger.info("Done (--features_only).")
        return

    # ── Survival ──────────────────────────────────────────────────────────────
    surv_df = load_survival(samples)
    common  = feat_df.index.intersection(surv_df.index)

    X = feat_df.loc[common].copy()
    X = X.loc[:, X.isna().mean() < 0.3]           # drop >30% missing
    X = X.fillna(X.median(numeric_only=True))

    y = np.array([(bool(e), float(t))
                  for e, t in zip(surv_df.loc[common, "event"],
                                  surv_df.loc[common, "os_days"])],
                 dtype=[("event", bool), ("time", float)])

    logger.info(f"Model input: {X.shape[0]} samples × {X.shape[1]} features")

    # ── Train ─────────────────────────────────────────────────────────────────
    model_path = OUT_DIR / "coxnet_model.pkl"
    if model_path.exists() and not args.force:
        logger.info("Loading cached model ...")
        with open(model_path, "rb") as f:
            saved = pickle.load(f)
        scaler, model, feat_cols = saved["scaler"], saved["model"], saved["feat_cols"]
        X = X[feat_cols]
    else:
        scaler, model, _ = train_coxnet(X, y)
        feat_cols = X.columns.tolist()
        with open(model_path, "wb") as f:
            pickle.dump({"scaler": scaler, "model": model, "feat_cols": feat_cols}, f)

    # ── Selected genes ────────────────────────────────────────────────────────
    sel_df = get_selected_features(model, feat_cols)
    sel_df.to_csv(OUT_DIR / "selected_genes.tsv", sep="\t", index=False)
    logger.info(f"CoxNet selected {len(sel_df)} / {len(feat_cols)} features")

    # ── Scores ────────────────────────────────────────────────────────────────
    scores = io_score(scaler, model, feat_df[feat_cols])
    ci_mean, ci_lo, ci_hi = bootstrap_cindex(scaler, model, X, y)
    logger.info(f"C-index: {ci_mean:.3f} (95% CI {ci_lo:.3f}–{ci_hi:.3f})")

    out = scores.to_frame()
    out["io_group"] = pd.cut(scores, bins=[0,33,67,100],
                              labels=["IO-Low","IO-Intermediate","IO-High"],
                              include_lowest=True)
    out = out.join(surv_df[["os_days","event"]], how="left")
    out = out.join(feat_df[["til_density","til_score",
                             "mut_STK11","mut_KEAP1","mut_EGFR","mut_KRAS"]], how="left")
    out.index.name = "sample_id"
    out.to_csv(OUT_DIR / "io_scores.tsv", sep="\t")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_km(scores, surv_df, OUT_DIR / "figures/km_io_score.png")
    plot_stk11_km(scores, surv_df, feat_df, OUT_DIR / "figures/km_stk11_subgroup.png")

    # ── Summary ───────────────────────────────────────────────────────────────
    grp = out["io_group"].value_counts()
    logger.info(f"\n{'='*40}")
    for g in ["IO-High","IO-Intermediate","IO-Low"]:
        logger.info(f"  {g}: {grp.get(g,0)}")
    logger.info(f"  C-index: {ci_mean:.3f} (95% CI {ci_lo:.3f}–{ci_hi:.3f})")
    top5 = sel_df.head(5)["feature"].tolist()
    logger.info(f"  Top features: {top5}")


if __name__ == "__main__":
    main()
