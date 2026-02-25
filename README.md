# LUAD Precision Oncology Platform

<div align="center">

**A 7-module multi-omics analysis pipeline for lung adenocarcinoma precision medicine**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![ESM2](https://img.shields.io/badge/ESM2-650M-green)](https://github.com/facebookresearch/esm)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## Overview

The **LUAD Precision Oncology Platform** integrates somatic variant annotation, bulk RNA-seq expression, single-cell tumor microenvironment (TME) profiling, pathway enrichment, protein language model embeddings, and mutation-to-drug mapping into a single reproducible pipeline.

Given a set of TCGA lung adenocarcinoma (LUAD) patient samples, the platform:

1. Builds a **clinical context card** with survival curves and key demographics
2. Annotates **somatic variants** (VEP/PCGR), computes TMB and mutational signatures
3. Quantifies **gene expression** outliers relative to TCGA-LUAD cohort
4. Characterizes the **tumor microenvironment** from public single-cell data (GSE131907)
5. Runs **pathway enrichment** (ORA + GSEA) on mutated genes and expression profiles
6. Extracts **ESM2 protein embeddings** at missense mutation sites (GPU-accelerated)
7. Maps mutations to **targeted therapies** using NCCN/FDA guidelines and CIViC evidence

---

## Pipeline Architecture

```
Input: TCGA LUAD samples
  │
  ├─── MAF (somatic variants)
  ├─── RNA-seq TPM matrix
  └─── Clinical metadata (GDC API)
                │
  ┌─────────────┴──────────────────────────────┐
  │           STAGE 1  (independent)           │
  │                                            │
  │  01 Patient Context  ──────────────────►  Patient card PNG
  │  02 Variation Annotation  ─────────────►  Variants TSV + TMB
  │  03 Expression Analysis  ──────────────►  Outlier genes TSV
  │  04 Single-Cell TME  ──────────────────►  Cell-type fractions
  └─────────────────────────────────────────────┘
                │
  ┌─────────────┴──────────────────────────────┐
  │           STAGE 2  (depend on Stage 1)     │
  │                                            │
  │  05 Pathway Enrichment  ────────────────►  ORA / GSEA plots
  │  07 Drug Mapping  ──────────────────────►  Drug report PNG
  └─────────────────────────────────────────────┘
                │
  ┌─────────────┴──────────────────────────────┐
  │     STAGE 3  (GPU recommended, optional)   │
  │                                            │
  │  06 ESM2 Site Features  ────────────────►  1280-dim embeddings
  └─────────────────────────────────────────────┘
```

---

## Modules

| # | Module | Description | Key Output |
|---|--------|-------------|------------|
| 01 | **Patient Context** | GDC clinical data pull, OS/PFS Kaplan-Meier curves | `*_patient_card.png` |
| 02 | **Variation Annotation** | VEP/PCGR somatic annotation, TMB, SBS mutational spectrum | `*_variants.tsv`, `*_tmb.tsv` |
| 03 | **Expression Analysis** | Bulk RNA-seq TPM normalization, cohort-level outlier detection | `*_expression_outliers.tsv` |
| 04 | **Single-Cell TME** | Cell-type deconvolution using GSE131907 (Lung Cancer Atlas) | `luad_tme_overview.png` |
| 05 | **Pathway Enrichment** | ORA on mutated genes; GSEA prerank on expression fold-changes | `*_gsea.png`, `*_ora.tsv` |
| 06 | **ESM2 Site Features** | Per-site 1280-dim embeddings + masked-marginal log-odds (GPU) | `mutation_scores.tsv`, `*.npy` |
| 07 | **Drug Mapping** | NCCN/FDA + CIViC evidence-based therapy recommendation | `*_drug_report.png` |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-org/luad_workflow.git
cd luad_workflow
```

### 2. Create conda environment

```bash
conda env create -f packages/conda/env/yml/pcgr.yml
conda activate pcgr
```

### 3. Install Python dependencies

```bash
pip install streamlit pandas numpy matplotlib seaborn scipy requests biopython
# For ESM2 (module 06, optional):
pip install torch transformers fair-esm
```

### 4. Download input data

```bash
# TCGA MAF files (somatic variants)
python data/scripts/download_tcga_maf.py

# GSE131907 single-cell data (module 04)
bash modules/04_single_cell/data/scripts/download_gse131907.sh
```

---

## Usage

### Run the full pipeline

```bash
# All modules, all samples (skip ESM2)
python run_all.py

# Single sample
python run_all.py --sample TCGA-86-A4D0

# Specific modules only
python run_all.py --modules 02 05 07

# Include ESM2 inference (requires GPU, slow)
python run_all.py --include_esm2

# Resume from a specific module
python run_all.py --from_module 05

# Dry run — validate inputs without computation
python run_all.py --dry_run
```

### Run individual modules

```bash
python modules/01_patient_context/luad_patient_context.py --sample TCGA-86-A4D0
python modules/02_variation_annotation/luad_pcgr.py       --sample TCGA-86-A4D0
python modules/03_expression/luad_expression.py           --sample TCGA-86-A4D0
python modules/04_single_cell/luad_singlecell.py
python modules/05_pathway/luad_pathway.py                 --sample TCGA-86-A4D0
python modules/06_esm/luad_esm2.py                        --sample TCGA-86-A4D0  # GPU
python modules/07_drug_mapping/luad_drug_mapping.py       --sample TCGA-86-A4D0
```

### Launch the Streamlit dashboard

```bash
streamlit run app.py
```

---

## Sample Data

The platform is demonstrated on **5 TCGA-LUAD samples**:

| Sample ID | Data Available |
|-----------|---------------|
| TCGA-49-4507 | Variants, Drug report |
| TCGA-73-4666 | RNA-seq, Pathway |
| TCGA-78-7158 | Variants, Drug report |
| TCGA-86-8358 | RNA-seq, Pathway |
| TCGA-86-A4D0 | All modules |

---

## Output Structure

```
data/output/
├── 01_patient_context/
│   ├── clinical_summary.tsv
│   └── TCGA-*_patient_card.png
├── 02_variation_annotation/
│   ├── all_samples_summary.tsv
│   └── {sample}/  *_variants.tsv.gz  *_tmb.tsv  *_variant_summary.png
├── 03_expression/
│   ├── all_samples_summary.tsv
│   └── {sample}/  *_gene_expression.tsv.gz  *_expression_outliers.tsv
├── 04_single_cell/
│   ├── luad_tme_summary.tsv
│   ├── per_sample_immune_metrics.tsv
│   ├── per_sample_tme_fractions.tsv
│   └── luad_tme_overview.png
├── 05_pathway/
│   ├── all_samples_summary.tsv
│   └── {sample}/  *_ora.tsv  *_gsea.tsv  *_gsea.png
├── 06_esm/
│   └── {sample}/  site_info.csv  mutation_scores.tsv  *_esm2_summary.png
│                  wt_features.npy  mut_features.npy  delta_features.npy
└── 07_drug_mapping/
    ├── all_samples_drug_summary.tsv
    ├── drug_actionability_heatmap.png
    └── {sample}/  *_drug_report.tsv  *_drug_report.png
```

---

## Module 06 — ESM2 Protein Embeddings

Module 06 uses **ESM2-650M** (facebook/esm2_t33_650M_UR50D) to extract per-site protein embeddings for each missense mutation identified in module 02.

For each mutation site:
- **WT embedding**: 1280-dim hidden state at the wild-type residue position
- **Mut embedding**: same position after in-silico amino acid substitution
- **Delta embedding**: `mut − wt` as a structural perturbation proxy
- **Log-odds score**: `log P(mut_aa|context) − log P(wt_aa|context)` (masked-marginal, ESM-1v methodology)

These features can be used downstream to predict mutation pathogenicity or train supervised classifiers.

> **Note**: ESM2 inference is computationally intensive (~1–3 hours per sample on CPU; ~10 min on GPU). Module 06 is **skipped by default** in `run_all.py`. Enable with `--include_esm2`.

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Pipeline orchestration | Python + subprocess |
| Variant annotation | Ensembl VEP v113, PCGR v2.2.5 |
| Protein language model | ESM2-650M (Meta AI) |
| Expression analysis | pandas, scipy, seaborn |
| Single-cell analysis | scanpy, GSE131907 |
| Pathway enrichment | GSEApy (ORA + GSEA prerank) |
| Drug knowledge base | NCCN/FDA curated KB + CIViC REST API |
| Visualization | matplotlib, seaborn |
| Dashboard | Streamlit |
| Containerization | Docker |

---

## Biological Context

Lung adenocarcinoma (LUAD) is the most common subtype of non-small cell lung cancer (NSCLC). Key driver alterations include:

- **Targetable drivers**: EGFR, ALK, ROS1, RET fusions, MET exon 14 skipping, BRAF V600E, KRAS G12C
- **Tumor suppressors**: TP53, STK11/LKB1, KEAP1/NRF2 (associated with IO resistance)
- **TMB**: ≥10 mut/Mb is associated with immunotherapy eligibility
- **STK11 loss**: confers resistance to PD-1/PD-L1 checkpoint inhibitors

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Acknowledgements

- [TCGA Research Network](https://www.cancer.gov/tcga) — patient data
- [PCGR](https://github.com/sigven/pcgr) — variant annotation framework
- [Meta AI / ESMFold team](https://github.com/facebookresearch/esm) — ESM2 protein language model
- [CIViC](https://civicdb.org) — clinical variant interpretation database
- [GSE131907](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE131907) — Lung Cancer Cell Atlas
