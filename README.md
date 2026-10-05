# LUAD Precision Oncology Platform

## What it does

<div align="center">

**A 7-module multi-omics analysis pipeline for lung adenocarcinoma precision medicine**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![ESM2](https://img.shields.io/badge/ESM2-650M-green)](https://github.com/facebookresearch/esm)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

The optional ESM2 module characterizes missense mutations with **ESM2-650M**:
it extracts wild-type and mutant residue embeddings, calculates their
difference, and computes masked-marginal log-odds. This module uses the
pretrained model directly; it does not train the interaction, ubiquitination
or localization classifiers described in the separate ESM2 projects.
ESM2 scoring is optional and was not executed in the verified default runs.

## Input

### Sample Data

The platform is demonstrated on **5 TCGA-LUAD samples**:

| Sample ID | Data Available |
|-----------|---------------|
| TCGA-49-4507 | Variants, Drug report |
| TCGA-73-4666 | RNA-seq, Pathway |
| TCGA-78-7158 | Variants, Drug report |
| TCGA-86-8358 | RNA-seq, Pathway |
| TCGA-86-A4D0 | All modules |

---

## Output

### Output Structure

```
data/output/
├── 01_patients/
│   ├── clinical_summary.tsv
│   └── TCGA-*_patient_card.png
├── 02_variants/
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
├── 06_pathway/
│   ├── all_samples_summary.tsv
│   └── {sample}/  *_ora.tsv  *_gsea.tsv  *_gsea.png
├── 07_variant_impact/
│   └── {sample}/  site_info.csv  mutation_scores.tsv  *_esm2_summary.png
│                  wt_features.npy  mut_features.npy  delta_features.npy
└── 07_drug_mapping/
    ├── all_samples_drug_summary.tsv
    ├── drug_actionability_heatmap.png
    └── {sample}/  *_drug_report.tsv  *_drug_report.png
```

---

## Try it

### Installation

#### 1. Clone the repository

```bash
git clone https://github.com/yujuan-zhang/luad.git
cd luad
```

#### 2. Create a Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The previous `packages/conda/env/yml/pcgr.yml` path is not included in this
checkout. The requirements file installs the listed Python dependencies;
it does not provision every external analysis tool. Module 02 imports the external `pcgr` package, which is not installed by this
requirements file or bundled as Python source in this checkout. Install PCGR
in an environment compatible with its own dependencies before running module 02.
Check module-specific requirements before running the full pipeline.

#### 3. Optional ESM2 dependencies

```bash
python -m pip install torch transformers fair-esm
```

#### 4. Download input data

```bash
# TCGA MAF files (somatic variants)
python data/scripts/download_tcga_maf.py

# GSE131907 single-cell data (module 04)
bash modules/04_single_cell/data/scripts/download_gse131907.sh
```

---

### Usage

#### Run the full pipeline

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

# Dry run — invoke each selected module's input checks (dependencies still required)
python run_all.py --dry_run
```

Module IDs in `run_all.py` are logical workflow IDs, not directory numbers:
`05` runs `modules/06_pathway`, and `06` runs `modules/07_variant_impact`.
Use a sample that is present in your downloaded inputs. `--dry_run` still imports
each module and requires its dependencies; it is not a dependency-free check.

#### Run individual modules

```bash
python modules/01_patients/luad_patient_context.py --sample TCGA-86-A4D0
python modules/02_variants/luad_pcgr.py       --sample TCGA-86-A4D0
python modules/03_expression/luad_expression.py           --sample TCGA-86-A4D0
python modules/04_single_cell/luad_singlecell.py
python modules/06_pathway/luad_pathway.py                 --sample TCGA-86-A4D0
python modules/07_variant_impact/luad_esm2.py                        --sample TCGA-86-A4D0  # GPU
python modules/07_drug_mapping/luad_drug_mapping.py       --sample TCGA-86-A4D0
```

#### Launch the Streamlit dashboard

```bash
streamlit run streamlit_app.py
```

---

### Single-sample smoke test

On 2026-10-03, the six default runner modules were executed for the included
`TCGA-86-A4D0` sample in an isolated copy, using an existing PCGR installation,
local PCGR GRCh38 reference data, clinical/RNA-seq/MAF files, and the GSE131907
annotation. The analysis module sources matched this GitHub checkout.

```bash
python run_all.py --modules 01 02 03 --sample TCGA-86-A4D0
python run_all.py --modules 04 05 07 --sample TCGA-86-A4D0
```

Both commands completed, taking about 62 and 37 seconds respectively, excluding
environment setup and initial reference-data downloads. Non-empty clinical,
variant, expression, TME, GSEA and drug-summary tables were checked. ORA had no
significant pathways for this sample; GSEA emitted a warning about tied ranking
values. These are functional smoke-test observations, not scientific validation
or a guarantee of identical results. ESM2 inference and a clean-machine PCGR
installation were not tested. Your runtime depends on data, references and hardware.

### Overview

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

### Pipeline Architecture

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

### Modules

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

### Module 06 — ESM2 Protein Embeddings

Module 06 uses **ESM2-650M** (facebook/esm2_t33_650M_UR50D) to extract per-site protein embeddings for each missense mutation identified in module 02.

For each mutation site:
- **WT embedding**: 1280-dim hidden state at the wild-type residue position
- **Mut embedding**: same position after in-silico amino acid substitution
- **Delta embedding**: `mut − wt` as a structural perturbation proxy
- **Log-odds score**: `log P(mut_aa|context) − log P(wt_aa|context)` (masked-marginal, ESM-1v methodology)

These features can be used downstream to predict mutation pathogenicity or train supervised classifiers.

> **Note**: ESM2 inference is computationally intensive (~1–3 hours per sample on CPU; ~10 min on GPU). Module 06 is **skipped by default** in `run_all.py`. Enable with `--include_esm2`.

---

### Technology Stack

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

### Biological Context

Lung adenocarcinoma (LUAD) is the most common subtype of non-small cell lung cancer (NSCLC). Key driver alterations include:

- **Targetable drivers**: EGFR, ALK, ROS1, RET fusions, MET exon 14 skipping, BRAF V600E, KRAS G12C
- **Tumor suppressors**: TP53, STK11/LKB1, KEAP1/NRF2 (associated with IO resistance)
- **TMB**: ≥10 mut/Mb is associated with immunotherapy eligibility
- **STK11 loss**: confers resistance to PD-1/PD-L1 checkpoint inhibitors

---

### License

MIT License. See [LICENSE](LICENSE).

---

### Acknowledgements

- [TCGA Research Network](https://www.cancer.gov/tcga) — patient data
- [PCGR](https://github.com/sigven/pcgr) — variant annotation framework
- [Meta AI / ESMFold team](https://github.com/facebookresearch/esm) — ESM2 protein language model
- [CIViC](https://civicdb.org) — clinical variant interpretation database
- [GSE131907](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE131907) — Lung Cancer Cell Atlas

