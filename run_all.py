#!/usr/bin/env python
"""
run_all.py
----------
LUAD Precision Platform — Master entry script.

Orchestrates all 7 analysis modules in dependency order:

  Stage 1 (independent, no cross-module deps):
    01  patient_context   — GDC clinical data + survival curves
    02  variation_annotation — somatic variants + TMB + SBS spectrum
    03  expression        — bulk RNA-seq TPM + outlier analysis
    04  single_cell       — TME characterization (GSE131907)

  Stage 2 (depends on Stage 1 outputs):
    05  pathway           — ORA (from 02) + GSEA prerank (from 03)
    07  drug_mapping      — mutation-to-drug mapping (from 02 + 01)

  Stage 3 (GPU recommended, slow):
    06  esm2              — per-site ESM2 embeddings (from 02)
                            skipped by default; use --include_esm2 to enable

Usage:
  python run_all.py                          # all samples, all modules (skip ESM2)
  python run_all.py --sample TCGA-86-A4D0   # single sample
  python run_all.py --modules 02 05 07       # specific modules only
  python run_all.py --include_esm2           # include ESM2 inference (slow)
  python run_all.py --dry_run               # validate inputs, no computation
  python run_all.py --from_module 05        # resume from a specific module
"""

import argparse
import subprocess
import sys
import time
import textwrap
from datetime import datetime
from pathlib import Path
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
PYTHON      = sys.executable   # use the same interpreter running this script

# ── Module registry ───────────────────────────────────────────────────────────
# Each entry: (module_id, display_name, script_path, supports_sample_flag)
MODULES = [
    ("01", "Patient Context",        "modules/01_patient_context/luad_patient_context.py",    True),
    ("02", "Variation Annotation",   "modules/02_variation_annotation/luad_pcgr.py",           True),
    ("03", "Expression Analysis",    "modules/03_expression/luad_expression.py",               True),
    ("04", "Single-Cell TME",        "modules/04_single_cell/luad_singlecell.py",              False),
    ("05", "Pathway Enrichment",     "modules/05_pathway/luad_pathway.py",                     True),
    ("06", "ESM2 Site Features",     "modules/06_esm/luad_esm2.py",                            True),
    ("07", "Drug Mapping",           "modules/07_drug_mapping/luad_drug_mapping.py",           True),
]

# Modules excluded by default (require GPU / long runtime)
SLOW_MODULES = {"06"}

# ANSI colours for terminal output
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def banner():
    print(f"""
{BOLD}{CYAN}
╔══════════════════════════════════════════════════════════╗
║       LUAD Precision Oncology Platform — run_all.py      ║
║       7-module multi-omics analysis pipeline             ║
╚══════════════════════════════════════════════════════════╝
{RESET}""")


def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def run_module(
    module_id: str,
    name: str,
    script: str,
    sample: str | None,
    dry_run: bool,
    supports_sample: bool,
) -> dict:
    """
    Execute a single module as a subprocess.
    Returns a result dict with status, duration, returncode.
    """
    script_path = PROJECT_DIR / script
    if not script_path.exists():
        return {
            "id": module_id, "name": name,
            "status": "MISSING", "duration": 0,
            "rc": -1, "note": f"Script not found: {script}",
        }

    cmd = [PYTHON, str(script_path)]
    if sample and supports_sample:
        cmd += ["--sample", sample]
    if dry_run:
        cmd += ["--dry_run"]

    print(f"\n{BOLD}[{module_id}] {name}{RESET}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"  {'─'*50}")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_DIR))
    elapsed = time.time() - t0

    status = "OK" if result.returncode == 0 else "FAILED"
    color  = GREEN if status == "OK" else RED
    print(f"\n  {color}[{module_id}] {status}{RESET}  ({fmt_duration(elapsed)})")

    return {
        "id": module_id, "name": name,
        "status": status, "duration": elapsed,
        "rc": result.returncode, "note": "",
    }


def write_cohort_index():
    """
    Build data/output/cohort_index.tsv — cross-module modality index.
    One row per patient; bool flags indicate which modules have output.
    Used by Streamlit to filter the patient selector per page.
    """
    out_dir = PROJECT_DIR / "data" / "output"
    expr_dir = out_dir / "03_expression"
    if not expr_dir.exists():
        return

    samples = sorted([
        d.name for d in expr_dir.iterdir()
        if d.is_dir() and d.name.startswith("TCGA")
    ])

    rows = []
    for s in samples:
        rows.append({
            "sample_id":      s,
            "has_rnaseq":     (expr_dir / s).exists(),
            "has_wes":        (out_dir / "02_variation_annotation" / s).exists(),
            "has_pathology":  (out_dir / "08_pathology" / s).exists(),
            "has_proteomics": False,   # M04 CPTAC not yet implemented
        })

    df = pd.DataFrame(rows)
    bool_cols = ["has_rnaseq", "has_wes", "has_pathology", "has_proteomics"]
    df["modality_count"] = df[bool_cols].sum(axis=1)

    out_path = out_dir / "cohort_index.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    n_rnaseq = df["has_rnaseq"].sum()
    n_wes    = df["has_wes"].sum()
    n_both   = (df["has_rnaseq"] & df["has_wes"]).sum()
    print(f"\n  Cohort index → {out_path.relative_to(PROJECT_DIR)}")
    print(f"  {len(df)} patients | RNA-seq: {n_rnaseq} | WES: {n_wes} | both: {n_both}")


def print_summary(results: list, total_elapsed: float):
    """Print final pipeline summary table."""
    print(f"\n{BOLD}{CYAN}{'═'*58}")
    print("  Pipeline Summary")
    print(f"{'═'*58}{RESET}")

    n_ok     = sum(1 for r in results if r["status"] == "OK")
    n_fail   = sum(1 for r in results if r["status"] == "FAILED")
    n_skip   = sum(1 for r in results if r["status"] == "SKIPPED")
    n_miss   = sum(1 for r in results if r["status"] == "MISSING")

    for r in results:
        icon = {
            "OK":      f"{GREEN}✓{RESET}",
            "FAILED":  f"{RED}✗{RESET}",
            "SKIPPED": f"{YELLOW}○{RESET}",
            "MISSING": f"{YELLOW}?{RESET}",
        }.get(r["status"], " ")
        dur = fmt_duration(r["duration"]) if r["duration"] else "—"
        note = f"  ← {r['note']}" if r["note"] else ""
        print(f"  {icon}  [{r['id']}] {r['name']:<28}  {dur:>6}{note}")

    print(f"\n  {'─'*54}")
    print(f"  Modules:  {GREEN}{n_ok} OK{RESET}  "
          f"{RED}{n_fail} FAILED{RESET}  "
          f"{YELLOW}{n_skip} skipped  {n_miss} missing{RESET}")
    print(f"  Total time:  {fmt_duration(total_elapsed)}")
    print(f"  Finished:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{BOLD}{CYAN}{'═'*58}{RESET}\n")

    if n_fail:
        print(f"{RED}Some modules failed. Check logs above.{RESET}\n")
        return 1
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="LUAD Precision Platform — run all analysis modules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python run_all.py                          # full pipeline (skip ESM2)
          python run_all.py --sample TCGA-86-A4D0   # single sample
          python run_all.py --modules 02 05 07       # specific modules
          python run_all.py --include_esm2           # include ESM2 (GPU needed)
          python run_all.py --from_module 05         # resume from module 05
          python run_all.py --dry_run                # validate inputs only
        """),
    )
    parser.add_argument(
        "--sample", type=str, default=None,
        help="Process a single sample ID (default: all samples)",
    )
    parser.add_argument(
        "--modules", nargs="+", metavar="ID",
        help="Run only these module IDs (e.g. --modules 02 05 07)",
    )
    parser.add_argument(
        "--from_module", type=str, metavar="ID",
        help="Skip modules before this ID (resume from given module)",
    )
    parser.add_argument(
        "--include_esm2", action="store_true",
        help="Include module 06 (ESM2 inference — slow, GPU recommended)",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Pass --dry_run to each module (validate inputs, no computation)",
    )
    args = parser.parse_args()

    banner()

    # ── Determine which modules to run ──
    selected_ids = set(args.modules) if args.modules else {m[0] for m in MODULES}

    if not args.include_esm2:
        selected_ids -= SLOW_MODULES

    if args.from_module:
        module_order = [m[0] for m in MODULES]
        if args.from_module not in module_order:
            print(f"{RED}Unknown module ID: {args.from_module}{RESET}")
            sys.exit(1)
        skip_before = module_order.index(args.from_module)
        for mid in module_order[:skip_before]:
            selected_ids.discard(mid)

    # ── Print run plan ──
    print(f"{BOLD}Run plan:{RESET}")
    for mid, name, script, _ in MODULES:
        if mid in selected_ids:
            tag = f"{GREEN}▶ will run{RESET}"
        elif mid in SLOW_MODULES and not args.include_esm2:
            tag = f"{YELLOW}○ skipped (use --include_esm2){RESET}"
        else:
            tag = f"{YELLOW}○ skipped{RESET}"
        print(f"  [{mid}] {name:<28} {tag}")

    if args.sample:
        print(f"\n  Sample:  {CYAN}{args.sample}{RESET}")
    else:
        print(f"\n  Sample:  {CYAN}all available samples{RESET}")

    if args.dry_run:
        print(f"  Mode:    {YELLOW}DRY RUN (validate inputs only){RESET}")

    print()

    # ── Execute modules in order ──
    t_pipeline = time.time()
    results = []

    for mid, name, script, supports_sample in MODULES:
        if mid not in selected_ids:
            status = "SKIPPED"
            if mid in SLOW_MODULES and not args.include_esm2:
                note = "use --include_esm2 to enable"
            else:
                note = "not in --modules list"
            results.append({
                "id": mid, "name": name,
                "status": status, "duration": 0,
                "rc": 0, "note": note,
            })
            continue

        r = run_module(mid, name, script, args.sample, args.dry_run, supports_sample)
        results.append(r)

        # Stop pipeline on critical failure (module 02 is prerequisite for 05/07)
        if r["status"] == "FAILED" and mid in {"02"}:
            print(f"\n{RED}Module 02 failed — modules 05 and 07 depend on it.{RESET}")
            print("Remaining dependent modules will be marked as skipped.\n")
            for dep_mid in {"05", "07"}:
                if dep_mid in selected_ids:
                    results.append({
                        "id": dep_mid,
                        "name": next(m[1] for m in MODULES if m[0] == dep_mid),
                        "status": "SKIPPED",
                        "duration": 0, "rc": 0,
                        "note": "skipped: module 02 failed",
                    })
                    selected_ids.discard(dep_mid)

    total_elapsed = time.time() - t_pipeline
    rc = print_summary(results, total_elapsed)
    write_cohort_index()
    sys.exit(rc)


if __name__ == "__main__":
    main()
