#!/usr/bin/env python
"""
luad_io_ml.py — Module 08: Immune Activity Score (IAS)
=======================================================
Biologically-driven IAS from M04 TME ssGSEA scores + M05 TIL density.

Features (7 immune features):
  Positive: CD8_T_cytotoxic, NK, Macrophage_M1, til_density
  Negative: Treg, Macrophage_M2, Fibroblast

Model:
  Simple Cox PH regression (CoxPHSurvivalAnalysis, scikit-survival)
  80 / 20 train / test split on TCGA-LUAD
  β coefficients learned from OS — weights are OS-validated, not arbitrary

IAS:
  risk = Σ βᵢ × featureᵢ   (Cox log-hazard)
  IAS  = 100 × (−risk − min) / (max − min)
  Higher IAS → lower death hazard → stronger immune activity

Validation:
  C-index + KM curves on held-out 20 % TCGA test set
  Log-rank p-value (High vs Low tertile) reported

References:
  Cox PH:       Cox, J R Stat Soc 1972
  Immunoscore:  Pagès et al., Lancet 2018  (PMID 29754777)
  TIL scoring:  Salgado et al., Ann Oncol 2015  (PMID 25995301)
  TIDE:         Jiang et al., Nat Med 2018  (PMID 30127393)

Outputs:
  data/output/08_io_ml/
    io_scores.tsv           per-patient IAS (0-100) + io_group + split
    cox_coefficients.tsv    β coefficients for 7 features
    model_metrics.json      C-index train / test, feature weights
    figures/
      km_ias_test.png       KM on 20 % held-out test set
      km_ias_full.png       KM on full TCGA cohort
      cox_coefficients.png  coefficient bar chart
"""

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_DIR  = Path(__file__).parent.parent.parent
TME_DIR      = PROJECT_DIR / "data/output/04_single_cell"
PATHO_DIR    = PROJECT_DIR / "data/output/05_pathology"
CLINICAL_DIR = PROJECT_DIR / "data/clinical"
OUT_DIR      = PROJECT_DIR / "data/output/08_io_ml"
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "figures").mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ── Feature definitions ────────────────────────────────────────────────────────
POSITIVE_TME = ["CD8_T_cytotoxic", "NK", "Macrophage_M1"]   # expected β < 0 (protective)
NEGATIVE_TME = ["Treg", "Macrophage_M2", "Fibroblast"]       # expected β > 0 (hazard)
ALL_TME      = POSITIVE_TME + NEGATIVE_TME
FEATURE_COLS = ALL_TME + ["til_density"]                      # 7 features total


# ══════════════════════════════════════════════════════════════════════════════
# 1. FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_tme_scores(sample_id: str) -> dict:
    """Load M04 ssGSEA immune cell-type scores."""
    path = TME_DIR / sample_id / f"{sample_id}_tme_scores.tsv"
    if not path.exists():
        return {}
    row = pd.read_csv(path, sep="\t").iloc[0]
    return {c: float(row.get(c, np.nan)) for c in ALL_TME}


def extract_til(sample_id: str) -> dict:
    """Load M05 TIL density score."""
    path = PATHO_DIR / sample_id / f"{sample_id}_pathology_scores.tsv"
    if not path.exists():
        return {}
    row = pd.read_csv(path, sep="\t").iloc[0]
    return {"til_density": float(row.get("til_density", np.nan))}


def build_feature_matrix(samples: list) -> pd.DataFrame:
    """Build 7-feature immune matrix from M04 + M05, z-score normalised."""
    records = []
    for i, sid in enumerate(samples):
        if i % 100 == 0:
            logger.info(f"  {i}/{len(samples)} ...")
        row = {"sample_id": sid}
        row.update(extract_tme_scores(sid))
        row.update(extract_til(sid))
        records.append(row)

    df = pd.DataFrame(records).set_index("sample_id")
    df = df.reindex(columns=FEATURE_COLS)   # enforce column order

    # Z-score normalise each feature across the cohort
    for col in df.columns:
        mu, sd = df[col].mean(), df[col].std()
        df[col] = (df[col] - mu) / (sd + 1e-8)

    n_missing = df.isna().sum().sum()
    logger.info(f"Feature matrix: {df.shape[0]} samples × {df.shape[1]} features "
                f"({n_missing} missing values)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. SURVIVAL DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_tcga_survival(samples: list) -> pd.DataFrame:
    df = pd.read_csv(CLINICAL_DIR / "tcga_luad_survival.tsv", sep="\t")
    df = df[df["sample_id"].isin(samples)].dropna(subset=["os_days", "event"])
    df = df[df["os_days"] > 0].copy()
    df["event"] = df["event"].astype(bool)
    return df.set_index("sample_id")


def make_survival_df(X: pd.DataFrame, surv: pd.DataFrame) -> pd.DataFrame:
    """Merge feature matrix with survival labels into one DataFrame for lifelines."""
    df = X.copy().fillna(0)
    df["os_days"] = surv["os_days"]
    df["event"]   = surv["event"].astype(int)
    return df.dropna(subset=["os_days", "event"])


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODEL — Simple Cox PH (lifelines)
# ══════════════════════════════════════════════════════════════════════════════

def train_cox(X_train: pd.DataFrame, surv_train: pd.DataFrame):
    """Train simple CoxPH on 7 immune features using lifelines. Returns (model, coefs)."""
    from lifelines import CoxPHFitter

    train_df = make_survival_df(X_train, surv_train)
    model = CoxPHFitter(penalizer=0.1)
    model.fit(train_df, duration_col="os_days", event_col="event")

    coefs = model.params_.rename("beta")
    logger.info("Cox β coefficients:")
    for feat, beta in coefs.items():
        direction = "protective (↑IAS)" if beta < 0 else "hazard (↓IAS)"
        logger.info(f"  {feat:30s}  β={beta:+.4f}  {direction}")
    return model, coefs


def compute_ias(model, X: pd.DataFrame) -> pd.Series:
    """IAS = invert Cox partial hazard, normalise to 0-100."""
    X_imp = X.copy().fillna(0)
    risk = model.predict_partial_hazard(X_imp).values
    s = -risk
    ias = 100.0 * (s - s.min()) / (s.max() - s.min() + 1e-8)
    return pd.Series(ias, index=X.index, name="io_score")


def compute_cindex(model, X: pd.DataFrame, surv: pd.DataFrame) -> float:
    from lifelines.utils import concordance_index
    risk = model.predict_partial_hazard(X.fillna(0)).values
    common = X.index.intersection(surv.index)
    return concordance_index(
        surv.loc[common, "os_days"],
        -risk[:len(common)],
        surv.loc[common, "event"]
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4. PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_km(scores: pd.Series, surv: pd.DataFrame,
            path: Path, title: str) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    m = surv.join(scores, how="inner").dropna(subset=["io_score", "os_days", "event"])
    if len(m) < 15:
        logger.warning(f"  Too few samples for KM plot ({len(m)}), skipping.")
        return

    q33, q67 = np.percentile(m["io_score"], [33, 67])
    m["grp"] = "Intermediate"
    m.loc[m["io_score"] >= q67, "grp"] = "High"
    m.loc[m["io_score"] <  q33, "grp"] = "Low"

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, color in [("High", "#e74c3c"), ("Intermediate", "#f39c12"), ("Low", "#3498db")]:
        s = m[m["grp"] == label]
        if len(s) < 5:
            continue
        KaplanMeierFitter().fit(s["os_days"] / 30.44, s["event"],
                                label=f"{label} (n={len(s)})")\
                           .plot_survival_function(ax=ax, ci_show=True, color=color)

    hi = m[m["grp"] == "High"]
    lo = m[m["grp"] == "Low"]
    if len(hi) >= 5 and len(lo) >= 5:
        p = logrank_test(hi["os_days"], lo["os_days"],
                         hi["event"],  lo["event"]).p_value
        ax.text(0.62, 0.95, f"High vs Low\nlog-rank p = {p:.2e}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    ax.set(xlabel="Time (months)", ylabel="Overall Survival", title=title)
    ax.legend(title="Immune Activity")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close()


def plot_coefficients(coefs: pd.Series, path: Path) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coefs_sorted = coefs.sort_values()
    colors = ["#3498db" if v < 0 else "#e74c3c" for v in coefs_sorted]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(coefs_sorted.index, coefs_sorted.values, color=colors, height=0.6)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Cox coefficient (β)")
    ax.set_title("IAS — Cox PH Coefficients\n"
                 "(blue β<0 = protective / high IAS,  red β>0 = hazard / low IAS)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Ignore cached feature matrix and recompute")
    args = parser.parse_args()

    # ── Discover samples with M04 data ────────────────────────────────────────
    all_samples = sorted(p.name for p in TME_DIR.iterdir()
                         if p.is_dir() and p.name.startswith("TCGA"))
    logger.info(f"Samples with M04 data: {len(all_samples)}")

    # ── Build / load feature matrix ───────────────────────────────────────────
    feat_path = OUT_DIR / "io_features.tsv"
    if feat_path.exists() and not args.force:
        logger.info("Loading cached feature matrix ...")
        feat = pd.read_csv(feat_path, sep="\t", index_col=0)
    else:
        feat = build_feature_matrix(all_samples)
        feat.to_csv(feat_path, sep="\t")

    # ── Load survival labels ──────────────────────────────────────────────────
    surv = load_tcga_survival(all_samples)

    # ── Align features + survival ─────────────────────────────────────────────
    common  = feat.index.intersection(surv.index)
    X_all   = feat.loc[common].copy()
    logger.info(f"Samples with features + survival: {len(common)}, "
                f"events: {int(surv.loc[common, 'event'].sum())}")

    # ── 80 / 20 train / test split (pure numpy, no sklearn) ──────────────────
    rng = np.random.default_rng(42)
    idx = np.arange(len(common))
    rng.shuffle(idx)
    n_test    = int(0.2 * len(idx))
    idx_test  = idx[:n_test]
    idx_train = idx[n_test:]

    X_train, X_test = X_all.iloc[idx_train], X_all.iloc[idx_test]
    surv_train = surv.loc[X_train.index]
    surv_test  = surv.loc[X_test.index]
    logger.info(f"Train: {len(X_train)},  Test: {len(X_test)}")

    # ── Train Cox PH ──────────────────────────────────────────────────────────
    model, coefs = train_cox(X_train, surv_train)

    # ── C-index on train and test ─────────────────────────────────────────────
    ci_train = compute_cindex(model, X_train, surv_train)
    ci_test  = compute_cindex(model, X_test,  surv_test)
    logger.info(f"C-index — Train: {ci_train:.3f},  Test (held-out): {ci_test:.3f}")

    # ── Compute IAS for all TCGA patients ─────────────────────────────────────
    ias_all = compute_ias(model, feat)

    # ── Tertile grouping ──────────────────────────────────────────────────────
    q33, q67 = np.percentile(ias_all, [33.3, 66.7])
    out = ias_all.to_frame()
    out["io_group"] = "Intermediate"
    out.loc[ias_all >= q67, "io_group"] = "High"
    out.loc[ias_all <  q33, "io_group"] = "Low"

    # Mark train / test split
    split_col = pd.Series("train", index=feat.index)
    split_col.loc[X_test.index] = "test"
    out["split"] = split_col

    out = out.join(surv[["os_days", "event"]], how="left")
    out.index.name = "sample_id"
    out.to_csv(OUT_DIR / "io_scores.tsv", sep="\t")
    logger.info(f"IAS saved for {len(out)} patients → {OUT_DIR / 'io_scores.tsv'}")

    # ── Save coefficients ─────────────────────────────────────────────────────
    coef_df = coefs.reset_index()
    coef_df.columns = ["feature", "beta"]
    coef_df["direction"] = coef_df["beta"].apply(
        lambda b: "protective" if b < 0 else "hazard")
    coef_df.to_csv(OUT_DIR / "cox_coefficients.tsv", sep="\t", index=False)

    # ── Figures ───────────────────────────────────────────────────────────────
    plot_coefficients(coefs, OUT_DIR / "figures" / "cox_coefficients.png")

    plot_km(ias_all.loc[X_test.index], surv,
            OUT_DIR / "figures" / "km_ias_test.png",
            title="IAS — Kaplan-Meier (TCGA 20 % held-out test set)")

    plot_km(ias_all, surv,
            OUT_DIR / "figures" / "km_ias_full.png",
            title="IAS — Kaplan-Meier (TCGA-LUAD full cohort)")

    # ── Save metrics JSON (read by Streamlit) ─────────────────────────────────
    metrics = {
        "n_train":    len(X_train),
        "n_test":     len(X_test),
        "n_events":   int(surv.loc[common, "event"].sum()),
        "n_features": len(coefs),
        "ci_train":   round(float(ci_train), 3),
        "ci_test":    round(float(ci_test),  3),
        "features":   {f: round(float(b), 4) for f, b in coefs.items()},
    }
    with open(OUT_DIR / "model_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info(f"Metrics: {metrics}")
    logger.info("✓  Module 08 complete.")


if __name__ == "__main__":
    main()
