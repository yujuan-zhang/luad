"""
streamlit_app.py
----------------
LUAD Precision Oncology Platform — Interactive Dashboard

Displays pre-computed analysis results from the 7-module pipeline.
Run with:  streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import io

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="LUAD Precision Oncology Platform",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ──────────────────────────────────────────────────────────────────
# All output files live under data/output/, relative to this script
OUTPUT = Path(__file__).parent / "data" / "output"

# ── Sample registry ────────────────────────────────────────────────────────
# Maps sample ID → which modules produced output for that sample
SAMPLES = [
    "TCGA-86-A4D0",   # most complete sample — all modules
    "TCGA-49-4507",
    "TCGA-73-4666",
    "TCGA-78-7158",
    "TCGA-86-8358",
]

# ── Helper: safe image display ─────────────────────────────────────────────
def show_image(path: Path, caption: str = "", width: int = None):
    """Display an image if the file exists, otherwise show a warning."""
    if path.exists():
        if width:
            st.image(str(path), caption=caption, width=width)
        else:
            st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Output not yet available: `{path.name}`\n\nRun the pipeline to generate this file.")


# ── Helper: safe table display ─────────────────────────────────────────────
def show_table(path: Path, sep: str = "\t", nrows: int = 200, caption: str = ""):
    """Display a TSV/CSV as a Streamlit dataframe, with row limit for large files."""
    if not path.exists():
        st.info(f"Output not yet available: `{path.name}`")
        return
    try:
        df = pd.read_csv(path, sep=sep, nrows=nrows)
        if caption:
            st.caption(caption)
        st.dataframe(df, use_container_width=True)
        if len(df) == nrows:
            st.caption(f"Showing first {nrows} rows. Full file: `{path}`")
    except Exception as e:
        st.error(f"Could not load `{path.name}`: {e}")


# ── Pipeline architecture figure ───────────────────────────────────────────
@st.cache_data
def make_pipeline_figure():
    """Generate the 7-module pipeline architecture diagram. Returns a PNG BytesIO."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 11)
    ax.axis("off")
    fig.patch.set_facecolor("#ffffff")

    def fbox(x, y, w, h, bg, bd, lw=1.5):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.12",
            facecolor=bg, edgecolor=bd, linewidth=lw, zorder=3,
        ))

    def txt(x, y, s, size=9, color="#1a1a2e", bold=False, ha="center", va="center"):
        ax.text(x, y, s, fontsize=size, color=color,
                fontweight="bold" if bold else "normal",
                ha=ha, va=va, zorder=4, multialignment="center")

    def arrow_dn(x, y_start, y_end):
        ax.annotate("", xy=(x, y_end), xytext=(x, y_start),
                    arrowprops=dict(arrowstyle="-|>", color="#607d8b",
                                   lw=1.8, mutation_scale=16), zorder=2)

    # Title
    txt(5.5, 10.6, "Pipeline Architecture", size=13, bold=True, color="#1a1a2e")

    # INPUT
    fbox(1.2, 9.2, 8.6, 0.85, "#f5f5f5", "#9e9e9e", lw=1.5)
    txt(5.5, 9.73, "INPUT", size=8.5, bold=True, color="#555")
    txt(5.5, 9.38,
        "MAF (somatic variants)   ·   RNA-seq TPM matrix   ·   GDC clinical metadata",
        size=7.5, color="#777")
    arrow_dn(5.5, 9.20, 8.47)

    # STAGE 1
    fbox(0.2, 6.75, 10.6, 1.70, "#e3f2fd", "#1565c0", lw=2)
    txt(0.55, 8.28, "STAGE 1  —  Independent modules (run in parallel)",
        size=8.5, bold=True, color="#1565c0", ha="left")
    bw, bh1 = 2.2, 1.05
    for i, (num, name) in enumerate([
        ("01", "Patient\nContext"), ("02", "Variation\nAnnotation"),
        ("03", "Expression\nAnalysis"), ("04", "Single-Cell\nTME"),
    ]):
        xi = 0.5 + i * 2.6
        fbox(xi, 6.90, bw, bh1, "white", "#1565c0", lw=1.5)
        txt(xi + bw / 2, 6.90 + bh1 * 0.70, f"M{num}", size=10.5, bold=True, color="#1565c0")
        txt(xi + bw / 2, 6.90 + bh1 * 0.28, name, size=7.5, color="#333")
    arrow_dn(5.5, 6.75, 6.02)

    # STAGE 2
    fbox(1.3, 4.35, 8.4, 1.65, "#e8f5e9", "#2e7d32", lw=2)
    txt(1.65, 5.83, "STAGE 2  —  Depends on Stage 1 outputs",
        size=8.5, bold=True, color="#2e7d32", ha="left")
    bh2 = 1.08
    for (num, name), xi in zip(
        [("05", "Pathway\nEnrichment"), ("07", "Drug\nMapping")], [2.7, 6.1]
    ):
        fbox(xi, 4.52, bw, bh2, "white", "#2e7d32", lw=1.5)
        txt(xi + bw / 2, 4.52 + bh2 * 0.70, f"M{num}", size=10.5, bold=True, color="#2e7d32")
        txt(xi + bw / 2, 4.52 + bh2 * 0.28, name, size=7.5, color="#333")
    arrow_dn(5.5, 4.35, 3.67)

    # STAGE 3
    fbox(2.8, 2.20, 5.4, 1.45, "#fff3e0", "#e65100", lw=2)
    txt(3.10, 3.50, "STAGE 3  —  GPU recommended (optional)",
        size=8.5, bold=True, color="#e65100", ha="left")
    fbox(4.4, 2.35, bw, 0.95, "white", "#e65100", lw=1.5)
    txt(4.4 + bw / 2, 2.35 + 0.95 * 0.70, "M06", size=10.5, bold=True, color="#e65100")
    txt(4.4 + bw / 2, 2.35 + 0.95 * 0.28, "ESM2\nEmbeddings", size=7.5, color="#333")
    arrow_dn(5.5, 2.20, 1.52)

    # DASHBOARD
    fbox(1.8, 0.55, 7.4, 0.95, "#1565c0", "#0d47a1", lw=2)
    txt(5.5, 1.08, "STREAMLIT DASHBOARD", size=10.5, bold=True, color="white")
    txt(5.5, 0.73, "Interactive multi-omics results explorer", size=7.5, color="#bbdefb")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    buf.seek(0)
    plt.close(fig)
    return buf


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🫁 LUAD Platform")
    st.markdown("**Precision Oncology · Multi-Omics**")
    st.divider()

    # Page selector
    page = st.radio(
        "Navigate",
        options=[
            "Home",
            "01 · Patient Context",
            "02 · Variation Annotation",
            "03 · Expression Analysis",
            "04 · Single-Cell TME",
            "05 · Pathway Enrichment",
            "06 · ESM2 Features",
            "07 · Drug Mapping",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    # Sample selector (not shown for pages that are sample-independent)
    sample_independent_pages = {"Home", "04 · Single-Cell TME", "07 · Drug Mapping"}
    if page not in sample_independent_pages:
        sample = st.selectbox(
            "Sample",
            options=SAMPLES,
            index=0,
            help="Select a TCGA-LUAD sample to view results for",
        )
    else:
        sample = SAMPLES[0]   # default; not used for these pages

    st.divider()
    st.caption("Data: TCGA-LUAD · GSE131907")
    st.caption("Model: ESM2-650M · PCGR v2.2.5")


# ══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════════════════════════════════════════

if page == "Home":

    # ── Global CSS ──────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    /* Hero banner */
    .hero-banner {
        background: linear-gradient(150deg, #0a1628 0%, #12263f 55%, #0f3460 100%);
        border-radius: 14px;
        padding: 52px 48px 44px 48px;
        margin-bottom: 28px;
        color: white;
        position: relative;
        overflow: hidden;
    }
    .hero-accent {
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
        background: linear-gradient(90deg, #1976d2, #42a5f5, #64b5f6, #29b6f6);
        border-radius: 14px 14px 0 0;
    }
    .hero-eyebrow {
        font-size: 0.70rem;
        letter-spacing: 2.5px;
        color: #42a5f5;
        text-transform: uppercase;
        font-weight: 700;
        margin-bottom: 14px;
    }
    .hero-title {
        font-size: 2.75rem;
        font-weight: 800;
        margin: 0 0 14px 0;
        letter-spacing: -0.8px;
        line-height: 1.15;
    }
    .hero-subtitle {
        font-size: 0.98rem;
        color: #b0c4de;
        margin: 0 0 28px 0;
        font-weight: 400;
        line-height: 1.65;
        max-width: 680px;
    }
    /* Badge pills — outlined style */
    .badge-row { margin-top: 0; display: flex; flex-wrap: wrap; gap: 8px; }
    .badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.74rem;
        font-weight: 600;
        letter-spacing: 0.4px;
        border: 1px solid;
    }
    .badge-blue   { border-color: #42a5f5; color: #90caf9; }
    .badge-green  { border-color: #66bb6a; color: #a5d6a7; }
    .badge-purple { border-color: #ba68c8; color: #ce93d8; }
    .badge-orange { border-color: #ffa726; color: #ffcc80; }
    .badge-teal   { border-color: #26a69a; color: #80cbc4; }
    /* Stat cards */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 32px;
    }
    .stat-card {
        background: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-left: 5px solid #1565c0;
        border-radius: 10px;
        padding: 20px 18px;
        text-align: center;
    }
    .stat-number { font-size: 2rem; font-weight: 800; color: #1565c0; }
    .stat-label  { font-size: 0.82rem; color: #666; margin-top: 4px; }
    /* Module cards */
    .module-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 14px;
        margin-bottom: 32px;
    }
    .module-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 16px 18px;
        background: white;
        transition: box-shadow 0.2s;
    }
    .module-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    .module-id {
        display: inline-block;
        background: #1565c0;
        color: white;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-right: 8px;
    }
    .module-name { font-weight: 700; font-size: 0.97rem; color: #1a1a2e; }
    .module-desc { font-size: 0.82rem; color: #666; margin-top: 6px; }
    .module-tag  {
        display: inline-block;
        background: #e8f0fe;
        color: #1a73e8;
        border-radius: 4px;
        padding: 1px 7px;
        font-size: 0.7rem;
        margin: 4px 3px 0 0;
    }
    /* Pipeline flow */
    .flow-step {
        display: flex;
        align-items: flex-start;
        gap: 16px;
        margin-bottom: 18px;
    }
    .flow-num {
        min-width: 36px;
        height: 36px;
        background: #1565c0;
        color: white;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 0.9rem;
        flex-shrink: 0;
    }
    .flow-text strong { font-size: 0.95rem; color: #1a1a2e; }
    .flow-text p { font-size: 0.83rem; color: #666; margin: 2px 0 0 0; }
    /* Section headings */
    .section-heading {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1a1a2e;
        border-bottom: 2px solid #1565c0;
        padding-bottom: 6px;
        margin: 28px 0 18px 0;
    }
    /* Overview cards */
    .overview-section {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 20px;
        margin-bottom: 32px;
    }
    .overview-card {
        border-radius: 12px;
        padding: 22px 20px;
    }
    .overview-card h4 {
        font-size: 0.95rem;
        font-weight: 700;
        margin: 0 0 12px 0;
    }
    .overview-card ul {
        margin: 0;
        padding-left: 18px;
    }
    .overview-card li {
        font-size: 0.83rem;
        line-height: 1.7;
        margin-bottom: 4px;
    }
    .ov-bg1 { background: #e8f0fe; border-left: 4px solid #1565c0; }
    .ov-bg1 h4 { color: #1565c0; }
    .ov-bg2 { background: #fce8e6; border-left: 4px solid #c62828; }
    .ov-bg2 h4 { color: #c62828; }
    .ov-bg3 { background: #e6f4ea; border-left: 4px solid #2e7d32; }
    .ov-bg3 h4 { color: #2e7d32; }
    </style>
    """, unsafe_allow_html=True)

    # ── Hero Banner ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-accent"></div>
        <div class="hero-eyebrow">Precision Oncology &nbsp;·&nbsp; Multi-Omics &nbsp;·&nbsp; TCGA-LUAD</div>
        <div class="hero-title">LUAD Precision Oncology Platform</div>
        <div class="hero-subtitle">
            A 7-module multi-omics pipeline integrating somatic variant annotation,
            tumor microenvironment profiling, ESM2 protein language model embeddings,
            and evidence-based drug mapping for lung adenocarcinoma precision medicine.
        </div>
        <div class="badge-row">
            <span class="badge badge-blue">Python 3.10+</span>
            <span class="badge badge-green">ESM2-650M</span>
            <span class="badge badge-purple">PCGR v2.2.5</span>
            <span class="badge badge-orange">TCGA-LUAD</span>
            <span class="badge badge-teal">CIViC · NCCN/FDA</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Overview (plain markdown, no HTML cards) ─────────────────────────────
    st.markdown("---")
    st.markdown("""
### Overview

Lung adenocarcinoma (LUAD) is the most common subtype of non-small cell lung cancer (NSCLC),
accounting for approximately 40% of all lung cancer diagnoses worldwide and remaining one of
the leading causes of cancer-related mortality globally. Unlike many other cancer types that
respond to a single standard-of-care protocol, LUAD is driven by highly heterogeneous somatic
mutations — a patient harboring an *EGFR* exon 19 deletion responds dramatically to osimertinib,
while a patient with *STK11* loss may derive little benefit from PD-1 checkpoint immunotherapy
even when tumor mutational burden appears high. Identifying the precise molecular profile of
each patient is therefore not optional: it is the foundation of effective treatment selection.

**What does this platform do?**

This platform automatically processes each patient's somatic variant calls, bulk RNA-seq
expression data, and clinical metadata through 7 integrated analysis modules — producing a
structured precision medicine report that addresses four core clinical questions: who is this
patient (survival context, stage, demographics), what mutations does their tumor carry (somatic
variants, TMB, mutational signatures), how is their tumor behaving (gene expression outliers,
immune microenvironment composition, dysregulated pathways), and which treatments have evidence
of efficacy for their specific mutation profile (targeted therapy, immunotherapy, chemotherapy).

**Why was this platform built — pain points addressed**

Current clinical bioinformatics workflows are fragmented. A typical analysis requires running
Ensembl VEP for variant annotation, a separate script to compute TMB, a dedicated pathway tool
for enrichment analysis, and a manual review of NCCN guidelines and the CIViC database for drug
matching — often across five or more disconnected tools with no shared data schema. Integrating
results across these tools requires significant manual effort, is error-prone, and is not
reproducible without careful bookkeeping. This platform replaces the entire toolchain with a
single orchestration command (`python run_all.py`), with automatic inter-module dependency
handling and a unified output structure.

Beyond workflow integration, standard clinical pipelines classify mutations using rule-based
databases (e.g., ClinVar, OncoKB) that assign a pathogenicity label but cannot capture *how
structurally damaging* a mutation is at the protein level. This platform introduces an AI layer
using **ESM2-650M** (Meta AI's protein language model), which extracts 1280-dimensional
embeddings at each missense mutation site and computes masked-marginal log-odds scores —
quantifying the degree to which the mutation is unexpected given the evolutionary context of
the protein. These features enable downstream machine learning–based variant effect prediction
that goes beyond binary rule-based classification.

**Key technical innovations**

*End-to-end automation.* A single command orchestrates all 7 modules in dependency order, with
automatic failure handling. Module 02 (variant annotation) feeds into Module 05 (pathway
enrichment) and Module 07 (drug mapping); if an upstream module fails, dependent modules are
automatically skipped with a clear error message.

*ESM2 protein language model embeddings (Module 06).* For every missense mutation identified
in Module 02, the platform retrieves the canonical protein sequence from UniProt, runs
ESM2-650M inference, and extracts the per-site 1280-dimensional hidden state for both the
wild-type and mutant amino acid. The delta embedding (mutant minus wild-type) serves as a
structural perturbation proxy. These features can be directly fed into supervised classifiers
for pathogenicity prediction or used for unsupervised clustering of mutation types.

*GPT-4o + PubMed literature integration (Module 07).* For each actionable variant identified
in drug mapping, the platform automatically queries PubMed via the NCBI Entrez API for the
most relevant clinical literature, retrieves abstracts from the top papers, and uses GPT-4o
to generate a concise structured clinical summary — covering the mutation's known oncogenic
role, associated therapies, and clinical evidence level. This connects the analysis directly
to the latest published evidence without requiring manual literature review.

*Single-cell tumor microenvironment characterization (Module 04).* The platform integrates
the 57,000-cell Lung Cancer Cell Atlas (GSE131907) to quantify the composition of the tumor
immune microenvironment — including T cell, NK cell, macrophage, dendritic cell, and cancer
cell populations. Immune infiltration patterns are a critical determinant of immunotherapy
response and are rarely assessed in standard clinical bioinformatics pipelines.

*Clinically actionable drug report (Module 07).* All mutation findings are mapped to an
embedded NCCN/FDA-curated knowledge base covering 16 LUAD driver genes, supplemented by
live CIViC REST API queries for additional evidence. Immunotherapy eligibility is assessed
based on TMB threshold (≥10 mut/Mb) and STK11/KEAP1 resistance markers, producing a
report that directly answers which therapy regimen is supported by clinical evidence for
that patient.
    """)
    st.markdown("---")

    # ── Key Stats ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="stat-grid">
        <div class="stat-card" style="border-left-color:#1565c0;">
            <div class="stat-number">7</div>
            <div class="stat-label">Analysis Modules</div>
        </div>
        <div class="stat-card" style="border-left-color:#2e7d32;">
            <div class="stat-number">5</div>
            <div class="stat-label">TCGA-LUAD Samples</div>
        </div>
        <div class="stat-card" style="border-left-color:#6a1b9a;">
            <div class="stat-number">1280</div>
            <div class="stat-label">ESM2 Embedding Dims</div>
        </div>
        <div class="stat-card" style="border-left-color:#e65100;">
            <div class="stat-number">~20K</div>
            <div class="stat-label">Genes Covered (RNA-seq)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Module Cards ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-heading">Analysis Modules</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="module-grid">
        <div class="module-card">
            <span class="module-id">01</span>
            <span class="module-name">Patient Context</span>
            <div class="module-desc">GDC clinical data retrieval · Kaplan-Meier survival curves · patient summary card</div>
            <span class="module-tag">Clinical</span>
            <span class="module-tag">Survival</span>
        </div>
        <div class="module-card">
            <span class="module-id">02</span>
            <span class="module-name">Variation Annotation</span>
            <div class="module-desc">VEP/PCGR somatic annotation · tumor mutational burden (TMB) · SBS mutational spectrum</div>
            <span class="module-tag">Genomics</span>
            <span class="module-tag">TMB</span>
            <span class="module-tag">VEP</span>
        </div>
        <div class="module-card">
            <span class="module-id">03</span>
            <span class="module-name">Expression Analysis</span>
            <div class="module-desc">Bulk RNA-seq TPM normalization · cohort-level outlier gene detection</div>
            <span class="module-tag">RNA-seq</span>
            <span class="module-tag">Outlier</span>
        </div>
        <div class="module-card">
            <span class="module-id">04</span>
            <span class="module-name">Single-Cell TME</span>
            <div class="module-desc">Cell-type deconvolution using GSE131907 Lung Cancer Cell Atlas · immune infiltration profiling</div>
            <span class="module-tag">scRNA</span>
            <span class="module-tag">Immune</span>
            <span class="module-tag">TME</span>
        </div>
        <div class="module-card">
            <span class="module-id">05</span>
            <span class="module-name">Pathway Enrichment</span>
            <div class="module-desc">Over-representation analysis (ORA) on mutated genes · GSEA prerank on expression fold-changes</div>
            <span class="module-tag">ORA</span>
            <span class="module-tag">GSEA</span>
        </div>
        <div class="module-card">
            <span class="module-id" style="background:#e65100;">06</span>
            <span class="module-name">ESM2 Site Features</span>
            <div class="module-desc">Per-site 1280-dim protein embeddings · masked-marginal log-odds · variant effect prediction <em>(GPU recommended)</em></div>
            <span class="module-tag" style="background:#fff3e0;color:#e65100;">GPU</span>
            <span class="module-tag">ESM2</span>
            <span class="module-tag">Embeddings</span>
        </div>
        <div class="module-card">
            <span class="module-id" style="background:#2e7d32;">07</span>
            <span class="module-name">Drug Mapping</span>
            <div class="module-desc">NCCN/FDA curated knowledge base · CIViC evidence · targeted therapy & immunotherapy recommendations</div>
            <span class="module-tag" style="background:#e8f5e9;color:#2e7d32;">Clinical</span>
            <span class="module-tag">NCCN</span>
            <span class="module-tag">CIViC</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Pipeline Architecture Figure ──────────────────────────────────────────
    st.markdown('<div class="section-heading">Pipeline Architecture</div>', unsafe_allow_html=True)
    _fig_buf = make_pipeline_figure()
    if _fig_buf is not None:
        _, _fig_mid, _ = st.columns([0.05, 0.90, 0.05])
        with _fig_mid:
            st.image(_fig_buf, use_container_width=True)
    else:
        st.info("Install `matplotlib` to render the pipeline diagram.")

    # ── Sample Coverage Table ────────────────────────────────────────────────
    st.markdown('<div class="section-heading">Demo Samples</div>', unsafe_allow_html=True)
    demo_data = {
        "Sample ID":  SAMPLES,
        "Variants":   ["✅", "✅", "—", "✅", "—"],
        "RNA-seq":    ["✅", "—", "✅", "—", "✅"],
        "Pathway":    ["✅", "✅", "✅", "✅", "✅"],
        "Drug Report":["✅", "✅", "—", "✅", "—"],
        "ESM2":       ["⏳ GPU", "—", "—", "—", "—"],
    }
    st.dataframe(pd.DataFrame(demo_data), use_container_width=True, hide_index=True)

    # ── Quick Start ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-heading">Quick Start</div>', unsafe_allow_html=True)
    st.code("""# Run full pipeline (all samples, skip ESM2)
python run_all.py

# Run a single sample
python run_all.py --sample TCGA-86-A4D0

# Include ESM2 inference (requires GPU)
python run_all.py --include_esm2

# Validate inputs without running analysis
python run_all.py --dry_run
""", language="bash")

    # ── Footer ───────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("""
    <div style="text-align:center; color:#666; font-size:0.85rem; padding: 8px 0 16px 0;">
        Any questions please contact <strong>Yujuan Zhang, PhD</strong>
        &nbsp;·&nbsp;
        <a href="mailto:yujuan.zhang418@gmail.com" style="color:#1565c0; text-decoration:none;">
            yujuan.zhang418@gmail.com
        </a>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 01 · PATIENT CONTEXT
# ══════════════════════════════════════════════════════════════════════════════

elif page == "01 · Patient Context":
    st.header("01 · Patient Context")
    st.caption("GDC clinical data · survival analysis · patient summary card")

    # Patient card image
    img = OUTPUT / "01_patient_context" / f"{sample}_patient_card.png"
    show_image(img, caption=f"Patient summary card — {sample}")

    st.divider()

    # Clinical summary table (all samples)
    st.subheader("Cohort Clinical Summary")
    show_table(
        OUTPUT / "01_patient_context" / "clinical_summary.tsv",
        caption="All available samples · GDC clinical metadata",
    )


# ══════════════════════════════════════════════════════════════════════════════
# 02 · VARIATION ANNOTATION
# ══════════════════════════════════════════════════════════════════════════════

elif page == "02 · Variation Annotation":
    st.header("02 · Variation Annotation")
    st.caption("Somatic variants annotated with VEP/PCGR · tumor mutational burden · SBS spectrum")

    # Variant summary image
    img = OUTPUT / "02_variation_annotation" / sample / f"{sample}_variant_summary.png"
    show_image(img, caption=f"Variant summary — {sample}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Tumor Mutational Burden")
        show_table(
            OUTPUT / "02_variation_annotation" / sample / f"{sample}_tmb.tsv",
            caption="TMB metrics for this sample",
        )

    with col2:
        st.subheader("Cohort Summary")
        show_table(
            OUTPUT / "02_variation_annotation" / "all_samples_summary.tsv",
            caption="Cross-sample variant statistics",
        )

    st.divider()

    st.subheader("Somatic Variants")
    show_table(
        OUTPUT / "02_variation_annotation" / sample / f"{sample}_variants.tsv.gz",
        nrows=100,
        caption=f"Top 100 variants shown · {sample}",
    )


# ══════════════════════════════════════════════════════════════════════════════
# 03 · EXPRESSION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "03 · Expression Analysis":
    st.header("03 · Expression Analysis")
    st.caption("Bulk RNA-seq TPM normalization · cohort-level outlier detection")

    # Expression summary image
    img = OUTPUT / "03_expression" / sample / f"{sample}_expression_summary.png"
    show_image(img, caption=f"Expression summary — {sample}")

    st.divider()

    st.subheader("Expression Outlier Genes")
    st.markdown(
        "Genes with expression significantly above or below the TCGA-LUAD cohort median. "
        "Showing first 200 rows."
    )
    show_table(
        OUTPUT / "03_expression" / sample / f"{sample}_expression_outliers.tsv",
        nrows=200,
        caption=f"Outlier genes — {sample}",
    )

    st.divider()

    st.subheader("Sample Correlation")
    show_table(
        OUTPUT / "03_expression" / sample / f"{sample}_expr_correlation.tsv",
        caption="Expression correlation with TCGA-LUAD cohort",
    )


# ══════════════════════════════════════════════════════════════════════════════
# 04 · SINGLE-CELL TME  (sample-independent)
# ══════════════════════════════════════════════════════════════════════════════

elif page == "04 · Single-Cell TME":
    st.header("04 · Single-Cell Tumor Microenvironment")
    st.caption("Cell-type deconvolution · immune infiltration · GSE131907 Lung Cancer Cell Atlas")

    st.info(
        "This module uses the GSE131907 public single-cell dataset (Lung Cancer Cell Atlas) "
        "and produces cohort-level results — not per-sample. "
        "Results represent averaged immune cell-type fractions across TCGA-LUAD."
    )

    # TME overview image
    img = OUTPUT / "04_single_cell" / "luad_tme_overview.png"
    show_image(img, caption="Tumor microenvironment overview — TCGA-LUAD cohort")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Per-Sample Immune Metrics")
        show_table(
            OUTPUT / "04_single_cell" / "per_sample_immune_metrics.tsv",
            caption="Immune score, stromal score, estimate score",
        )

    with col2:
        st.subheader("Cell-Type Fractions")
        show_table(
            OUTPUT / "04_single_cell" / "per_sample_tme_fractions.tsv",
            caption="Relative abundance of each TME cell type",
        )

    st.divider()

    st.subheader("TME Summary")
    show_table(
        OUTPUT / "04_single_cell" / "luad_tme_summary.tsv",
        caption="Cohort-level TME summary statistics",
    )


# ══════════════════════════════════════════════════════════════════════════════
# 05 · PATHWAY ENRICHMENT
# ══════════════════════════════════════════════════════════════════════════════

elif page == "05 · Pathway Enrichment":
    st.header("05 · Pathway Enrichment")
    st.caption("Over-representation analysis (ORA) on mutated genes · GSEA prerank on expression")

    # GSEA plot
    img = OUTPUT / "05_pathway" / sample / f"{sample}_gsea.png"
    show_image(img, caption=f"GSEA enrichment plot — {sample}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("GSEA Results")
        show_table(
            OUTPUT / "05_pathway" / sample / f"{sample}_gsea.tsv",
            caption=f"Top enriched gene sets (GSEA prerank) — {sample}",
        )

    with col2:
        st.subheader("ORA Results")
        show_table(
            OUTPUT / "05_pathway" / sample / f"{sample}_ora.tsv",
            caption=f"Over-represented pathways (ORA) — {sample}",
        )

    st.divider()

    st.subheader("Cross-Sample Pathway Summary")
    show_table(
        OUTPUT / "05_pathway" / "all_samples_summary.tsv",
        caption="Top enriched pathways across all samples",
    )


# ══════════════════════════════════════════════════════════════════════════════
# 06 · ESM2 SITE FEATURES
# ══════════════════════════════════════════════════════════════════════════════

elif page == "06 · ESM2 Features":
    st.header("06 · ESM2 Protein Site Features")
    st.caption("Per-site 1280-dim embeddings · masked-marginal log-odds · variant effect prediction")

    st.warning(
        "**Module 06 requires GPU acceleration and is excluded from the default pipeline run.** "
        "To generate results: `python run_all.py --include_esm2 --sample TCGA-86-A4D0`"
    )

    st.markdown("""
    **What this module does:**

    For each missense mutation identified in Module 02, ESM2-650M (Meta AI) is used to:
    1. Retrieve the canonical protein sequence from UniProt
    2. Extract the **1280-dim hidden state** at the mutation site (wild-type and mutant)
    3. Compute **masked-marginal log-odds** = log P(mut_aa | context) − log P(wt_aa | context)
    4. Compute **delta embedding** = mut_emb − wt_emb as a structural perturbation proxy

    These features can be used to predict whether a mutation is likely pathogenic.
    """)

    st.divider()

    # ESM2 summary image (if available)
    img = OUTPUT / "06_esm" / sample / f"{sample}_esm2_summary.png"
    show_image(img, caption=f"ESM2 mutation scoring summary — {sample}")

    # Mutation scores table
    scores_path = OUTPUT / "06_esm" / sample / "mutation_scores.tsv"
    if scores_path.exists():
        st.subheader("Mutation Scores")
        show_table(scores_path, caption=f"ESM2 log-odds and oncogenicity scores — {sample}")
    else:
        st.subheader("Mutation Scores")
        st.info("No ESM2 scores available yet. Run module 06 to generate.")

    # Feature file availability
    st.divider()
    st.subheader("Feature Files")
    feature_files = [
        ("wt_features.npy",    "Wild-type site embeddings (N × 1280)"),
        ("mut_features.npy",   "Mutant site embeddings (N × 1280)"),
        ("delta_features.npy", "Delta embeddings — mut minus wt (N × 1280)"),
        ("site_info.csv",      "Site metadata (gene, position, wt_aa, mut_aa)"),
    ]
    for fname, desc in feature_files:
        fpath = OUTPUT / "06_esm" / sample / fname
        status = "✅ Available" if fpath.exists() else "⏳ Not yet generated"
        st.markdown(f"- `{fname}` — {desc} &nbsp;&nbsp; **{status}**")


# ══════════════════════════════════════════════════════════════════════════════
# 07 · DRUG MAPPING  (sample-independent overview + per-sample detail)
# ══════════════════════════════════════════════════════════════════════════════

elif page == "07 · Drug Mapping":
    st.header("07 · Drug Mapping")
    st.caption("NCCN/FDA guidelines · CIViC evidence · therapy recommendations")

    # Cross-sample heatmap — square image, center at 55% width
    st.subheader("Drug Actionability — All Samples")
    img_heatmap = OUTPUT / "07_drug_mapping" / "drug_actionability_heatmap.png"
    _, mid, _ = st.columns([0.225, 0.55, 0.225])
    with mid:
        show_image(img_heatmap, caption="Drug actionability heatmap — all samples × genes")

    st.divider()

    # Per-sample selector — only show samples that have a drug report file
    drug_samples = [
        s for s in SAMPLES
        if (OUTPUT / "07_drug_mapping" / s / f"{s}_drug_report.png").exists()
    ]

    st.subheader("Per-Sample Drug Report")
    if not drug_samples:
        st.info("No drug reports found. Run `python run_all.py --modules 02 07` to generate them.")
        st.stop()

    st.caption(
        f"Drug reports available for {len(drug_samples)} of {len(SAMPLES)} samples. "
        "Samples without a drug report lack somatic variant (MAF) input data."
    )
    selected = st.selectbox(
        "Select sample",
        options=drug_samples,
        index=0,
        key="drug_sample",
    )

    img_report = OUTPUT / "07_drug_mapping" / selected / f"{selected}_drug_report.png"
    show_image(img_report, caption=f"Drug report — {selected}")

    show_table(
        OUTPUT / "07_drug_mapping" / selected / f"{selected}_drug_report.tsv",
        caption=f"Recommended therapies — {selected}",
    )

    st.divider()

    st.subheader("Cross-Sample Drug Summary")
    show_table(
        OUTPUT / "07_drug_mapping" / "all_samples_drug_summary.tsv",
        caption="Drug recommendations across all samples",
    )

    st.divider()

    # LUAD drug context
    with st.expander("LUAD Targeted Therapy Reference (NCCN/FDA)"):
        st.markdown("""
        | Gene / Alteration | Drug(s) | Evidence |
        |-------------------|---------|----------|
        | EGFR exon 19 del / L858R | Osimertinib (1st line) | FDA approved |
        | EGFR T790M (resistance) | Osimertinib | FDA approved |
        | EGFR exon 20 ins | Amivantamab + Lazertinib | FDA approved |
        | ALK fusion | Alectinib / Lorlatinib | FDA approved |
        | ROS1 fusion | Crizotinib / Entrectinib | FDA approved |
        | RET fusion | Selpercatinib | FDA approved |
        | MET exon 14 skip | Capmatinib / Tepotinib | FDA approved |
        | BRAF V600E | Dabrafenib + Trametinib | FDA approved |
        | KRAS G12C | Sotorasib / Adagrasib | FDA approved |
        | NTRK fusion | Entrectinib / Larotrectinib | FDA approved |
        | HER2 mutation | Trastuzumab deruxtecan | FDA approved |
        | TMB ≥10 mut/Mb | Pembrolizumab (IO) | FDA approved |
        | STK11 loss | IO resistance — prefer chemo | NCCN note |
        """)
