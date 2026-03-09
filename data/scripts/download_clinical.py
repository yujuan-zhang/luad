#!/usr/bin/env python
"""
download_clinical.py
--------------------
Download complete TCGA-LUAD clinical data from GDC in one batch.

Output:
  data/clinical/tcga_luad_clinical.tsv   — all cases, flat clinical fields
  data/clinical/tcga_luad_survival.tsv   — survival-only subset (for KM)

Usage:
  python download_clinical.py
  python download_clinical.py --out-dir /custom/path
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import requests

# ── Config ────────────────────────────────────────────────────────────────────
GDC_CASES_URL = "https://api.gdc.cancer.gov/cases"
PROJECT_ID    = "TCGA-LUAD"
MAX_SIZE      = 2000   # well above actual cohort size (~585)

CLINICAL_FIELDS = [
    "submitter_id",
    "diagnoses.age_at_diagnosis",
    "diagnoses.ajcc_pathologic_stage",
    "diagnoses.ajcc_pathologic_t",
    "diagnoses.ajcc_pathologic_n",
    "diagnoses.ajcc_pathologic_m",
    "diagnoses.days_to_last_follow_up",
    "diagnoses.days_to_death",
    "diagnoses.primary_diagnosis",
    "diagnoses.morphology",
    "diagnoses.tissue_or_organ_of_origin",
    "demographic.gender",
    "demographic.vital_status",
    "demographic.days_to_death",
    "demographic.race",
    "demographic.ethnicity",
    "demographic.year_of_birth",
    "exposures.pack_years_smoked",
    "exposures.cigarettes_per_day",
    "exposures.smoking_history",
    "exposures.years_smoked",
]

logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
log = logging.getLogger("download-clinical")


def fetch_all_cases() -> list:
    """Fetch all TCGA-LUAD case records from GDC."""
    log.info(f"Querying GDC for all {PROJECT_ID} cases...")
    payload = {
        "filters": json.dumps({
            "op": "=",
            "content": {"field": "project.project_id", "value": PROJECT_ID},
        }),
        "fields": ",".join(CLINICAL_FIELDS),
        "format": "json",
        "size": str(MAX_SIZE),
    }
    resp = requests.get(GDC_CASES_URL, params=payload, timeout=60)
    resp.raise_for_status()
    hits = resp.json()["data"]["hits"]
    log.info(f"  Retrieved {len(hits)} cases")
    return hits


def parse_hits(hits: list) -> pd.DataFrame:
    """Flatten nested GDC JSON into a tidy DataFrame."""
    import numpy as np

    rows = []
    for h in hits:
        row = {"sample_id": h.get("submitter_id", "")}

        # ── Demographics ──────────────────────────────────────────────────────
        demo = h.get("demographic") or {}
        row["gender"]       = demo.get("gender", "unknown")
        row["vital_status"] = demo.get("vital_status", "unknown")
        row["race"]         = demo.get("race", "unknown")
        row["ethnicity"]    = demo.get("ethnicity", "unknown")
        row["year_of_birth"]= demo.get("year_of_birth")

        # ── Survival ──────────────────────────────────────────────────────────
        days_death = demo.get("days_to_death")
        diag_list  = h.get("diagnoses") or [{}]
        diag       = diag_list[0]
        days_fu    = diag.get("days_to_last_follow_up")

        def to_float(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return np.nan

        if days_death and to_float(days_death) > 0:
            row["os_days"] = to_float(days_death)
            row["event"]   = 1
        elif days_fu and to_float(days_fu) > 0:
            row["os_days"] = to_float(days_fu)
            row["event"]   = 0
        else:
            row["os_days"] = np.nan
            row["event"]   = 0

        row["os_months"] = round(row["os_days"] / 30.44, 2) if not np.isnan(row["os_days"]) else np.nan

        # ── Diagnosis ─────────────────────────────────────────────────────────
        age_days = diag.get("age_at_diagnosis")
        row["age_at_diagnosis"] = round(to_float(age_days) / 365.25, 1) if age_days else np.nan
        row["stage"]            = diag.get("ajcc_pathologic_stage", "unknown")
        row["stage_t"]          = diag.get("ajcc_pathologic_t", "unknown")
        row["stage_n"]          = diag.get("ajcc_pathologic_n", "unknown")
        row["stage_m"]          = diag.get("ajcc_pathologic_m", "unknown")
        row["primary_diagnosis"]= diag.get("primary_diagnosis", "Adenocarcinoma, NOS")
        row["morphology"]       = diag.get("morphology", "")

        # ── Exposures ─────────────────────────────────────────────────────────
        exp_list = h.get("exposures") or [{}]
        exp = exp_list[0]
        row["pack_years"]      = to_float(exp.get("pack_years_smoked"))
        row["cigarettes_per_day"] = to_float(exp.get("cigarettes_per_day"))
        row["smoking_history"] = exp.get("smoking_history", "unknown")
        row["years_smoked"]    = to_float(exp.get("years_smoked"))

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Download TCGA-LUAD clinical data")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (default: data/clinical/ relative to project root)")
    args = parser.parse_args()

    # Resolve output directory
    script_dir  = Path(__file__).parent
    project_dir = script_dir.parent
    out_dir     = Path(args.out_dir) if args.out_dir else project_dir / "clinical"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download
    hits = fetch_all_cases()
    df   = parse_hits(hits)

    # Save full clinical table
    clinical_out = out_dir / "tcga_luad_clinical.tsv"
    df.to_csv(clinical_out, sep="\t", index=False)
    log.info(f"Full clinical data saved → {clinical_out}  ({len(df)} rows)")

    # Save survival-only subset (for KM curves in M01)
    survival_cols = ["sample_id", "os_days", "os_months", "event", "stage",
                     "vital_status", "age_at_diagnosis", "gender"]
    survival_out = out_dir / "tcga_luad_survival.tsv"
    df[survival_cols].to_csv(survival_out, sep="\t", index=False)
    log.info(f"Survival subset saved     → {survival_out}")

    # Summary
    n_dead   = (df["event"] == 1).sum()
    n_alive  = (df["event"] == 0).sum()
    n_stages = df["stage"].value_counts()
    log.info(f"\n── Summary ──────────────────────────")
    log.info(f"  Total cases : {len(df)}")
    log.info(f"  Deceased    : {n_dead}")
    log.info(f"  Alive/censored: {n_alive}")
    log.info(f"  Stages:\n{n_stages.to_string()}")


if __name__ == "__main__":
    main()
