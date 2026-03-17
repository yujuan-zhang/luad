"""
Parse GSE72094: extract expression matrix + clinical survival data.
Output:
  gse72094_expr_zscore.tsv.gz   — probe→gene mapped, z-scored, samples×genes
  gse72094_survival.tsv         — sample_id, os_days, event, stage, mutations
"""
import gzip, pandas as pd, numpy as np
from pathlib import Path

DIR = Path(__file__).parent

# ── 1. Clinical metadata ───────────────────────────────────────────────────
print("Parsing clinical metadata...")
path = DIR / "GSE72094_series_matrix.txt.gz"
samples = {}
gsm_order = []

with gzip.open(path, 'rt', encoding='utf-8', errors='ignore') as f:
    for line in f:
        if line.startswith("!Sample_geo_accession"):
            gsm_order = [v.strip('"') for v in line.strip().split('\t')[1:]]
        if line.startswith("!Sample_characteristics_ch1"):
            parts = line.strip().split('\t')
            for i, val in enumerate(parts[1:]):
                v = val.strip('"')
                if ':' in v:
                    k, vv = v.split(':', 1)
                    k = k.strip(); vv = vv.strip()
                    gsm = gsm_order[i] if i < len(gsm_order) else str(i)
                    if gsm not in samples: samples[gsm] = {}
                    samples[gsm][k] = vv
        if line.startswith("!series_matrix_table_begin"):
            break

meta = pd.DataFrame(samples).T
meta.index.name = "sample_id"

# Build survival table
surv = meta[["vital_status", "survival_time_in_days",
             "Stage", "kras_status", "stk11_status",
             "egfr_status", "tp53_status"]].copy()
surv = surv[surv["vital_status"].isin(["Alive", "Dead"])]
surv = surv[surv["survival_time_in_days"] != "NA"]
surv["os_days"] = pd.to_numeric(surv["survival_time_in_days"], errors="coerce")
surv["event"]   = (surv["vital_status"] == "Dead").astype(int)
surv = surv.dropna(subset=["os_days"]).query("os_days > 0")
surv.to_csv(DIR / "gse72094_survival.tsv", sep="\t")
print(f"Survival: {len(surv)} samples, {surv['event'].sum()} events")

# ── 2. Expression matrix ───────────────────────────────────────────────────
print("Parsing expression matrix (may take a minute)...")
with gzip.open(path, 'rt', encoding='utf-8', errors='ignore') as f:
    in_matrix = False
    rows = []
    cols = []
    for line in f:
        if line.startswith("!series_matrix_table_begin"):
            in_matrix = True
            cols = next(f).strip().split('\t')
            cols = [c.strip('"') for c in cols]
            continue
        if line.startswith("!series_matrix_table_end"):
            break
        if in_matrix:
            parts = line.strip().split('\t')
            rows.append(parts)

expr = pd.DataFrame(rows, columns=cols)
expr = expr.rename(columns={cols[0]: "probe_id"})
expr["probe_id"] = expr["probe_id"].str.strip('"')
expr = expr.set_index("probe_id")
expr = expr.apply(pd.to_numeric, errors="coerce")
print(f"Expression matrix: {expr.shape} (probes × samples)")

# ── 3. Probe → gene symbol mapping ────────────────────────────────────────
print("Loading GPL15048 annotation...")
gpl_path = DIR / "GPL15048.soft.gz"
probe2gene = {}
if gpl_path.exists():
    with gzip.open(gpl_path, 'rt', encoding='utf-8', errors='ignore') as f:
        in_table = False
        header = []
        for line in f:
            if line.startswith("^DATABASE") or line.startswith("^PLATFORM"):
                continue
            if line.startswith("!platform_table_begin"):
                header = next(f).strip().split('\t')
                in_table = True
                continue
            if line.startswith("!platform_table_end"):
                break
            if in_table:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    row = dict(zip(header[:len(parts)], parts))
                    probe = row.get("ID","")
                    gene  = row.get("GeneSymbol", row.get("Gene Symbol",
                            row.get("GENE_SYMBOL", row.get("gene_assignment",""))))
                    if probe and gene and gene != "---":
                        # Take first gene symbol if multiple
                        gene = gene.split("//")[0].split(" ")[0].strip()
                        if gene:
                            probe2gene[probe] = gene

    print(f"Probe→gene mappings: {len(probe2gene)}")
else:
    print("GPL file not found, using probe IDs")

# Map probes to genes (take max per gene)
expr.index = expr.index.map(lambda p: probe2gene.get(p, p))
expr = expr[~expr.index.str.startswith("AFFX")]  # remove control probes
expr = expr.groupby(level=0).max()               # max per gene
print(f"After gene mapping: {expr.shape} (genes × samples)")

# ── 4. Z-score normalization (gene-wise, within this dataset) ─────────────
print("Z-score normalizing...")
expr_z = expr.apply(lambda row: (row - row.mean()) / (row.std() + 1e-8), axis=1)

# Transpose to samples × genes
expr_z = expr_z.T
expr_z.index.name = "sample_id"

# Keep only samples with survival data
keep = expr_z.index.intersection(surv.index)
expr_z = expr_z.loc[keep]
print(f"Final: {expr_z.shape} (samples × genes), {len(keep)} with survival")

expr_z.to_csv(DIR / "gse72094_expr_zscore.tsv.gz", sep="\t", compression="gzip")
print("Done. Saved gse72094_expr_zscore.tsv.gz and gse72094_survival.tsv")
