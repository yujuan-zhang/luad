"""
oncokb_mapping.py
-----------------
Query OncoKB API to annotate LUAD somatic variants with:
  - Oncogenicity (Oncogenic / Likely Oncogenic / VUS / etc.)
  - Mutation effect (Gain-of-function / Loss-of-function / etc.)
  - Therapeutic implications (Level 1–4 evidence, FDA-approved drugs)

Input:
  PCGR-annotated MAF files from modules/01_variation_annotation/data/output/
  or the raw MAF files from data/input/

Output:
  data/output/{case_id}_oncokb.tsv  — variant-level drug annotation table

Requirements:
  oncokb-annotator already cloned to ../oncokb_annotator/
  export ONCOKB_API_TOKEN="your_token_here"  # free academic token at oncokb.org

Usage:
  python oncokb_mapping.py --maf ../../data/input/TCGA-49-4507.maf
  python oncokb_mapping.py --all   # process all MAF files in data/input/
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
MAF_DIR    = SCRIPT_DIR / "../../data/input"
OUT_DIR    = SCRIPT_DIR / "../data/output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ONCOKB_TOKEN = os.environ.get("ONCOKB_API_TOKEN", "")

# OncoKB annotation levels to keep (therapeutically actionable)
ACTIONABLE_LEVELS = {"LEVEL_1", "LEVEL_2", "LEVEL_3A", "LEVEL_3B", "LEVEL_4"}


def annotate_maf_with_oncokb(maf_path: Path, token: str) -> pd.DataFrame:
    """
    Call oncokb-annotator CLI (MafAnnotator) and return annotated DataFrame.
    Requires: oncokb-annotator installed and ONCOKB_API_TOKEN set.
    """
    import subprocess
    import tempfile

    annotator = SCRIPT_DIR / "../oncokb_annotator/MafAnnotator.py"
    out_tmp = Path(tempfile.mktemp(suffix="_oncokb.maf"))
    cmd = [
        "python", str(annotator),
        "-i", str(maf_path),
        "-o", str(out_tmp),
        "-t",  # tumor-only mode
        "-b", token,
    ]
    print(f"[OncoKB] Annotating {maf_path.name} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] MafAnnotator failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(out_tmp, sep="\t", comment="#", low_memory=False)
    out_tmp.unlink(missing_ok=True)
    return df


def filter_actionable(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows with actionable therapeutic levels."""
    level_col = "HIGHEST_LEVEL"
    if level_col not in df.columns:
        return df
    return df[df[level_col].isin(ACTIONABLE_LEVELS)].copy()


def summarize(df: pd.DataFrame, case_id: str) -> pd.DataFrame:
    """Select key columns for the output summary table."""
    keep = [
        "Hugo_Symbol", "HGVSp_Short", "Variant_Classification",
        "ONCOGENIC", "MUTATION_EFFECT",
        "HIGHEST_LEVEL", "CITATIONS",
        "TREATMENTS",
    ]
    cols = [c for c in keep if c in df.columns]
    summary = df[cols].copy()
    summary.insert(0, "case_id", case_id)
    return summary


def process_maf(maf_path: Path):
    if not ONCOKB_TOKEN:
        print("[ERROR] ONCOKB_API_TOKEN not set. Export it before running.", file=sys.stderr)
        sys.exit(1)

    case_id = maf_path.stem
    df = annotate_maf_with_oncokb(maf_path, ONCOKB_TOKEN)
    df_action = filter_actionable(df)
    summary = summarize(df_action, case_id)

    out_path = OUT_DIR / f"{case_id}_oncokb.tsv"
    summary.to_csv(out_path, sep="\t", index=False)
    print(f"[Done] {len(summary)} actionable variants → {out_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="OncoKB drug annotation for LUAD MAF files")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--maf", type=Path, help="Path to a single MAF file")
    group.add_argument("--all", action="store_true", help="Process all .maf files in data/input/")
    args = parser.parse_args()

    if args.maf:
        process_maf(args.maf)
    else:
        maf_files = sorted(MAF_DIR.glob("*.maf"))
        if not maf_files:
            print(f"[ERROR] No .maf files found in {MAF_DIR}", file=sys.stderr)
            sys.exit(1)
        all_results = []
        for maf in maf_files:
            all_results.append(process_maf(maf))
        combined = pd.concat(all_results, ignore_index=True)
        combined_path = OUT_DIR / "all_samples_oncokb.tsv"
        combined.to_csv(combined_path, sep="\t", index=False)
        print(f"\n[Summary] Combined table → {combined_path}")


if __name__ == "__main__":
    main()
