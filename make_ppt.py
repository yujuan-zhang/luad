#!/usr/bin/env python
"""
Generate LUAD Precision Oncology Platform presentation (python-pptx).
Run with:  conda run -n base python make_ppt.py
Output  :  LUAD_Platform_Presentation.pptx
"""

from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import pptx.oxml.ns as nsmap
from lxml import etree
import copy

# ── colour palette ────────────────────────────────────────────────────────────
DARK_NAVY   = RGBColor(0x0D, 0x1B, 0x2A)   # slide background / title bg
MID_BLUE    = RGBColor(0x1B, 0x4F, 0x72)   # header bars
ACCENT_TEAL = RGBColor(0x17, 0x9C, 0x8E)   # highlight
ACCENT_ORANGE = RGBColor(0xE8, 0x7D, 0x1E) # secondary accent
LIGHT_GREY  = RGBColor(0xF4, 0xF6, 0xF9)   # body background
WHITE       = RGBColor(0xFF, 0xFF, 0xFF)
TEXT_DARK   = RGBColor(0x1A, 0x1A, 0x2E)
TEXT_MED    = RGBColor(0x2C, 0x3E, 0x50)

SLIDE_W = Inches(13.33)
SLIDE_H = Inches(7.5)

BASE = Path("/Users/yujuanzhang/Desktop/2026AI/github/luad_workflow")
FIGS = {
    "background":   BASE / "data/output/home_figures/background_luad.png",
    "cohort":       BASE / "data/output/home_figures/cohort_overview.png",
    "pipeline":     BASE / "data/output/home_figures/pipeline_figure.png",
    "tme_overview": BASE / "data/output/04_single_cell/luad_tme_overview.png",
    "tme_heatmap":  BASE / "data/output/04_single_cell/tme_cohort_heatmap.png",
    "drug_heatmap": BASE / "data/output/07_drug_mapping/drug_actionability_heatmap.png",
    "km_io":        BASE / "data/output/08_io_ml/figures/km_io_score.png",
    "km_stk11":     BASE / "data/output/08_io_ml/figures/km_stk11_subgroup.png",
    "km_gse":       BASE / "data/output/08_io_ml/figures/km_gse72094.png",
    "feat_imp":     BASE / "data/output/08_io_ml/figures/feature_importance.png",
    "shap":         BASE / "data/output/08_io_ml/figures/shap_summary.png",
    "variant_sum":  BASE / "data/output/02_variants/TCGA-86-A4D0/TCGA-86-A4D0_variant_summary.png",
    "gsea":         BASE / "data/output/06_pathway/TCGA-86-A4D0/TCGA-86-A4D0_gsea.png",
    "patient_card": BASE / "data/output/01_patients/TCGA-86-A4D0_patient_card.png",
}

prs = Presentation()
prs.slide_width  = SLIDE_W
prs.slide_height = SLIDE_H

blank_layout = prs.slide_layouts[6]   # completely blank


# ── helper functions ──────────────────────────────────────────────────────────

def add_rect(slide, l, t, w, h, fill=None, line_color=None, line_width=Pt(0)):
    shape = slide.shapes.add_shape(1, l, t, w, h)  # MSO_SHAPE_TYPE.RECTANGLE=1
    if fill:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
    else:
        shape.fill.background()
    if line_color:
        shape.line.color.rgb = line_color
        shape.line.width = line_width
    else:
        shape.line.fill.background()
    return shape


def add_textbox(slide, text, l, t, w, h,
                font_size=18, bold=False, color=WHITE,
                align=PP_ALIGN.LEFT, wrap=True, italic=False):
    txBox = slide.shapes.add_textbox(l, t, w, h)
    tf = txBox.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return txBox


def add_image_safe(slide, path, l, t, w, h):
    if Path(path).exists():
        slide.shapes.add_picture(str(path), l, t, w, h)
        return True
    return False


def slide_header(slide, title, subtitle=None,
                 bg=DARK_NAVY, title_color=WHITE, bar_color=ACCENT_TEAL):
    """Full-width top header bar."""
    add_rect(slide, 0, 0, SLIDE_W, SLIDE_H, fill=LIGHT_GREY)
    add_rect(slide, 0, 0, SLIDE_W, Inches(1.15), fill=bg)
    add_rect(slide, 0, Inches(1.15), Inches(0.06), SLIDE_H - Inches(1.15), fill=bar_color)
    add_textbox(slide, title, Inches(0.25), Inches(0.1), Inches(12.8), Inches(0.7),
                font_size=32, bold=True, color=title_color, align=PP_ALIGN.LEFT)
    if subtitle:
        add_textbox(slide, subtitle, Inches(0.25), Inches(0.72), Inches(12.8), Inches(0.38),
                    font_size=16, bold=False, color=ACCENT_TEAL, align=PP_ALIGN.LEFT)


def bullet_block(slide, items, l, t, w, h,
                 font_size=15, color=TEXT_DARK, spacing=1.15, title=None, title_color=MID_BLUE):
    """Render a list of bullet strings into a text box."""
    txBox = slide.shapes.add_textbox(l, t, w, h)
    tf = txBox.text_frame
    tf.word_wrap = True

    first = True
    if title:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = PP_ALIGN.LEFT
        run = p.add_run()
        run.text = title
        run.font.size = Pt(font_size + 2)
        run.font.bold = True
        run.font.color.rgb = title_color

    for item in items:
        p = tf.add_paragraph() if not first else tf.paragraphs[0]
        first = False
        p.alignment = PP_ALIGN.LEFT
        p.space_before = Pt(3)
        indent = 0
        text = item
        if item.startswith("    "):
            indent = 1
            text = item.lstrip()
        p.level = indent
        run = p.add_run()
        run.text = ("  • " if indent == 0 else "      – ") + text
        run.font.size = Pt(font_size if indent == 0 else font_size - 1)
        run.font.bold = False
        run.font.color.rgb = color


def section_divider(title, subtitle=""):
    """Full-bleed section divider slide."""
    slide = prs.slides.add_slide(blank_layout)
    add_rect(slide, 0, 0, SLIDE_W, SLIDE_H, fill=DARK_NAVY)
    add_rect(slide, 0, Inches(3.0), SLIDE_W, Inches(0.07), fill=ACCENT_TEAL)
    add_textbox(slide, title,
                Inches(1.5), Inches(2.2), Inches(10), Inches(1.4),
                font_size=48, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    if subtitle:
        add_textbox(slide, subtitle,
                    Inches(1.5), Inches(3.6), Inches(10), Inches(0.8),
                    font_size=22, bold=False, color=ACCENT_TEAL, align=PP_ALIGN.CENTER)
    return slide


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 1 – TITLE
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(blank_layout)
add_rect(sl, 0, 0, SLIDE_W, SLIDE_H, fill=DARK_NAVY)
# decorative gradient strip
add_rect(sl, 0, Inches(4.8), SLIDE_W, Inches(0.08), fill=ACCENT_TEAL)
add_rect(sl, 0, Inches(4.88), SLIDE_W, Inches(0.04), fill=ACCENT_ORANGE)

add_textbox(sl,
    "LUAD Precision Oncology Platform",
    Inches(0.8), Inches(1.1), Inches(11.7), Inches(1.4),
    font_size=44, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

add_textbox(sl,
    "A 9-Module Multi-Omics Pipeline for Lung Adenocarcinoma Precision Medicine",
    Inches(0.8), Inches(2.5), Inches(11.7), Inches(0.8),
    font_size=22, bold=False, color=ACCENT_TEAL, align=PP_ALIGN.CENTER)

add_textbox(sl,
    "Integrating Genomics · Transcriptomics · Single-Cell TME · AI/ML · Clinical Evidence",
    Inches(0.8), Inches(3.2), Inches(11.7), Inches(0.7),
    font_size=21, bold=False, color=RGBColor(0xB0, 0xC4, 0xDE), align=PP_ALIGN.CENTER)

add_textbox(sl,
    "TCGA-LUAD Cohort  |  n ≈ 500+ patients  |  Streamlit Interactive Dashboard",
    Inches(0.8), Inches(5.1), Inches(11.7), Inches(0.65),
    font_size=20, bold=False, color=RGBColor(0x90, 0xA8, 0xC0), align=PP_ALIGN.CENTER)

add_textbox(sl,
    "May 2026",
    Inches(0.8), Inches(6.4), Inches(11.7), Inches(0.6),
    font_size=18, bold=False, color=RGBColor(0x70, 0x90, 0xA8), align=PP_ALIGN.CENTER)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 2 – OUTLINE
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(blank_layout)
slide_header(sl, "Outline")

sections = [
    ("01", "Background",       "LUAD epidemiology, clinical challenges, precision oncology rationale"),
    ("02", "Data & Methods",   "Cohort, 9-module pipeline, technology stack"),
    ("03", "Results",          "Variant landscape, TME, IO scoring, drug mapping, integration"),
    ("04", "Discussion",       "Key findings, limitations, future directions"),
]
colors_sec = [ACCENT_TEAL, MID_BLUE, ACCENT_ORANGE, RGBColor(0x8E, 0x44, 0xAD)]

for i, (num, sec, desc) in enumerate(sections):
    x = Inches(0.3 + i * 3.25)
    y = Inches(1.5)
    add_rect(sl, x, y, Inches(3.0), Inches(4.5), fill=colors_sec[i])
    add_textbox(sl, num, x + Inches(0.12), y + Inches(0.15), Inches(2.7), Inches(0.65),
                font_size=38, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_textbox(sl, sec, x + Inches(0.1), y + Inches(0.9), Inches(2.8), Inches(0.6),
                font_size=22, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_rect(sl, x + Inches(0.15), y + Inches(1.6), Inches(2.7), Inches(0.04),
             fill=RGBColor(0xFF, 0xFF, 0xFF))
    add_textbox(sl, desc, x + Inches(0.1), y + Inches(1.75), Inches(2.8), Inches(2.5),
                font_size=15, bold=False, color=WHITE, align=PP_ALIGN.CENTER)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION DIVIDER – BACKGROUND
# ══════════════════════════════════════════════════════════════════════════════
section_divider("01  Background",
                "Clinical Landscape & Motivation for Multi-Omics Precision Medicine")


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 3 – LUAD Epidemiology
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(blank_layout)
slide_header(sl, "Lung Adenocarcinoma: Clinical Landscape",
             "Most common subtype of NSCLC with complex molecular heterogeneity")

# left text block
bullet_block(sl, [
    "Lung cancer is the leading cause of cancer-related mortality worldwide",
    "NSCLC accounts for ~85% of all lung cancers; LUAD is the most prevalent subtype (~40%)",
    "5-year overall survival remains ~20% despite therapeutic advances",
    "Median OS for advanced (Stage IV) LUAD: 12–18 months with standard therapy",
], Inches(0.4), Inches(1.3), Inches(5.8), Inches(2.5),
   font_size=14, color=TEXT_DARK, title="Epidemiology", title_color=MID_BLUE)

bullet_block(sl, [
    "High genomic heterogeneity: driver mutations, copy number alterations, fusions",
    "Targetable oncogenes: EGFR (~15%), KRAS (~30%), ALK/ROS1/RET fusions (~5–10%)",
    "Immune resistance: STK11/KEAP1 loss → PD-L1-independent IO failure",
    "Tumor microenvironment shapes response to both targeted therapy and immunotherapy",
], Inches(0.4), Inches(3.95), Inches(5.8), Inches(2.6),
   font_size=14, color=TEXT_DARK, title="Molecular Complexity", title_color=MID_BLUE)

# right image
add_image_safe(sl, FIGS["background"],
               Inches(6.5), Inches(1.25), Inches(6.5), Inches(5.3))


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 4 – Precision Medicine Challenges
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(blank_layout)
slide_header(sl, "Challenges in LUAD Precision Medicine",
             "Why a single biomarker is not enough")

challenges = [
    ("Genomic Complexity",
     ["Dozens of actionable driver genes", "Co-mutations alter drug sensitivity",
      "Subclonal heterogeneity drives resistance", "TMB links genome to immunotherapy"]),
    ("Transcriptomic Gaps",
     ["Gene expression outliers signal dysregulation", "IFN-γ / CYT scores predict IO response",
      "Bulk RNA misses cell-type composition", "Single-cell data required for TME"]),
    ("Clinical Translation",
     ["Evidence tiers (OncoKB, AMP/CAP, ESCAT) vary by mutation",
      "Drug-resistance CIViC annotations often missed",
      "Clinical trial matching is manual and slow",
      "MDT reports require synthesis of multiple data layers"]),
]

cols = [ACCENT_TEAL, MID_BLUE, ACCENT_ORANGE]
for i, (title, points) in enumerate(challenges):
    x = Inches(0.3 + i * 4.3)
    y = Inches(1.3)
    add_rect(sl, x, y, Inches(4.0), Inches(5.7), fill=WHITE,
             line_color=cols[i], line_width=Pt(2))
    add_rect(sl, x, y, Inches(4.0), Inches(0.56), fill=cols[i])
    add_textbox(sl, title, x + Inches(0.1), y + Inches(0.06),
                Inches(3.8), Inches(0.44), font_size=17, bold=True,
                color=WHITE, align=PP_ALIGN.CENTER)
    for j, pt in enumerate(points):
        add_textbox(sl, "• " + pt, x + Inches(0.15), y + Inches(0.72 + j * 1.22),
                    Inches(3.7), Inches(1.15), font_size=15, color=TEXT_DARK)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 5 – Platform Motivation
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(blank_layout)
slide_header(sl, "Platform Motivation: Unifying Multi-Omics Layers",
             "From siloed analyses to an integrated, reproducible clinical decision tool")

add_rect(sl, Inches(0.3), Inches(1.35), SLIDE_W - Inches(0.6), Inches(5.7), fill=WHITE,
         line_color=LIGHT_GREY, line_width=Pt(1))

# 3-column: Gap | Solution | Impact
headers = ["Current Gap", "Our Solution", "Clinical Impact"]
hcols   = [ACCENT_ORANGE, MID_BLUE, ACCENT_TEAL]
gaps    = [
    ["Variant interpretation isolated from expression & TME",
     "No cohort-wide benchmark for outlier expression",
     "Manual, time-consuming drug–mutation matching",
     "IO biomarker assessment lacks integration"],
    ["8-module fully automated pipeline (M01–M08)",
     "TCGA cohort-level percentile benchmarking (n=517)",
     "Knowledge-base + CIViC resistance annotation",
     "Inflamed/Excluded/Desert TME phenotyping (M04+M05)"],
    ["Faster, reproducible molecular tumour boards",
     "Statistically anchored clinical decisions",
     "Evidence-graded therapy recommendations",
     "Validated IO stratification (GSE72094, n=398)"],
]
for i in range(3):
    x = Inches(0.4 + i * 4.28)
    add_rect(sl, x, Inches(1.4), Inches(4.1), Inches(0.58), fill=hcols[i])
    add_textbox(sl, headers[i], x + Inches(0.08), Inches(1.44), Inches(3.9), Inches(0.5),
                font_size=17, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    for j, txt in enumerate(gaps[i]):
        yy = Inches(2.1 + j * 1.3)
        add_rect(sl, x, yy, Inches(4.1), Inches(1.22),
                 fill=LIGHT_GREY if j % 2 == 0 else WHITE)
        add_textbox(sl, txt, x + Inches(0.1), yy + Inches(0.12), Inches(3.9), Inches(1.0),
                    font_size=15, color=TEXT_DARK)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION DIVIDER – DATA & METHODS
# ══════════════════════════════════════════════════════════════════════════════
section_divider("02  Data & Methods",
                "Cohort · 9-Module Pipeline · Technology Stack")


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 6 – Dataset Overview
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(blank_layout)
slide_header(sl, "Dataset Overview",
             "TCGA-LUAD as primary cohort with independent external validation")

# left text
datasets = [
    ("TCGA-LUAD (Primary)",
     ["n = 517 primary tumor samples (RNA-seq available)",
      "n = 272 somatic MAF files (WES-based variant calling)",
      "n = 478 H&E whole-slide pathology images (M05)",
      "n = 500+ clinical metadata records (GDC API)",
      "Data types: MAF · RNA-seq TPM · Clinical · WSI"]),
    ("GSE131907 — Single-Cell Reference",
     ["Lung Cancer Cell Atlas (Kim et al., 2020)",
      "44 LUAD / LUSC / normal tissue samples",
      "~200,000 single cells (10x Chromium)",
      "Used for TME cell-type deconvolution (M04)"]),
    ("GSE72094 — External Validation",
     ["Early-stage LUAD (n = 398, Lee et al., 2016)",
      "Used to validate Immune Activity Score (M08)",
      "Independent cohort for survival model transferability"]),
]
y_pos = Inches(1.35)
for i, (title, bullets) in enumerate(datasets):
    c = [ACCENT_TEAL, MID_BLUE, ACCENT_ORANGE][i]
    add_rect(sl, Inches(0.3), y_pos, Inches(0.07), Inches(len(bullets) * 0.36 + 0.5), fill=c)
    add_textbox(sl, title, Inches(0.5), y_pos, Inches(6.2), Inches(0.38),
                font_size=14, bold=True, color=c)
    for j, b in enumerate(bullets):
        add_textbox(sl, "  • " + b, Inches(0.55), y_pos + Inches(0.4 + j * 0.36),
                    Inches(6.0), Inches(0.34), font_size=12.5, color=TEXT_DARK)
    y_pos += Inches(len(bullets) * 0.36 + 0.75)

# right: cohort overview figure
add_image_safe(sl, FIGS["cohort"],
               Inches(7.0), Inches(1.3), Inches(6.0), Inches(5.6))


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 7 – Pipeline Architecture
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(blank_layout)
slide_header(sl, "Pipeline Architecture",
             "Three-stage modular design ensuring reproducibility and parallelism")

add_image_safe(sl, FIGS["pipeline"],
               Inches(0.3), Inches(1.3), Inches(8.2), Inches(5.7))

# stage legend on the right
stages = [
    (ACCENT_TEAL,   "Stage 1 — Independent Modules",
     "M01–M05 run independently per sample.\nOutputs: patient card, annotated variants, expression outliers, TME fractions, TIL scores."),
    (MID_BLUE,      "Stage 2 — Integrative Modules",
     "M06–M07 depend on M02/M03 outputs.\nOutputs: pathway enrichment plots, AlphaMissense/ESM2 variant scores."),
    (ACCENT_ORANGE, "Stage 3 — Integration",
     "M08: multi-omics treatment recommendation + clinical trial matching + MDT report."),
]
for i, (col, title, desc) in enumerate(stages):
    yy = Inches(1.5 + i * 2.0)
    add_rect(sl, Inches(8.7), yy, Inches(4.4), Inches(1.85),
             fill=WHITE, line_color=col, line_width=Pt(2))
    add_rect(sl, Inches(8.7), yy, Inches(4.4), Inches(0.48), fill=col)
    add_textbox(sl, title, Inches(8.8), yy + Inches(0.05), Inches(4.2), Inches(0.4),
                font_size=15, bold=True, color=WHITE)
    add_textbox(sl, desc, Inches(8.8), yy + Inches(0.56), Inches(4.2), Inches(1.2),
                font_size=14, color=TEXT_DARK)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 8 – Module Overview (M01–M05)
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(blank_layout)
slide_header(sl, "Modules M01 – M05",
             "Clinical context · Variant annotation · Expression · Single-cell TME · Pathology")

modules = [
    ("M01", "Patient Context",
     "Data: TCGA-LUAD (clinical metadata, n=585), M02 TMB output\n"
     "Methods: Kaplan-Meier OS curves · demographic summary card",
     ACCENT_TEAL),
    ("M02", "Variant Annotation",
     "Data: TCGA WES MAF (VEP pre-annotated), CIViC API, COSMIC SBS v3.3\n"
     "Methods: PCGR v2.2.5 tiering · TMB (38 Mb) · SBS signature profiling",
     MID_BLUE),
    ("M03", "Expression Analysis",
     "Data: TCGA RNA-seq TPM (n=517) · GTEx Lung normal baseline (n=287)\n"
     "Methods: Z-score vs GTEx · outlier detection (|Z|>2) · subtype correlation",
     RGBColor(0x27, 0x6B, 0xBD)),
    ("M04", "Single-Cell TME",
     "Data: GSE131907 Lung Cell Atlas (~200K cells) · TCGA bulk RNA (M03)\n"
     "Methods: ssGSEA deconvolution · 10 cell-type fractions · TME phenotyping",
     RGBColor(0x1E, 0x8B, 0x5A)),
    ("M05", "Computational Pathology",
     "Data: TCGA H&E thumbnails (n=517)\n"
     "Methods: Otsu segmentation · Macenko H&E deconvolution · TIL density scoring",
     RGBColor(0x7D, 0x3C, 0x98)),
]
for i, (num, name, desc, col) in enumerate(modules):
    x = Inches(0.25 + (i % 3) * 4.35)
    y = Inches(1.35 if i < 3 else 4.1)
    w, h = Inches(4.1), Inches(2.55)
    add_rect(sl, x, y, w, h, fill=WHITE, line_color=col, line_width=Pt(1.5))
    add_rect(sl, x, y, w, Inches(0.52), fill=col)
    add_textbox(sl, f"{num}  {name}", x + Inches(0.1), y + Inches(0.06),
                w - Inches(0.2), Inches(0.42), font_size=16, bold=True, color=WHITE)
    add_textbox(sl, desc, x + Inches(0.1), y + Inches(0.62),
                w - Inches(0.2), Inches(1.8), font_size=14, color=TEXT_DARK)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 9 – Module Overview (M06–M08)
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(blank_layout)
slide_header(sl, "Modules M06 – M08",
             "Pathway enrichment · Variant impact (AlphaMissense) · Treatment recommendation")

ML_BADGE = "  ★ ML"   # mark ML modules

modules2 = [
    ("M06", "Pathway Enrichment",
     "Data: M02 mutated genes · M03 expression · MSigDB/KEGG/Reactome gene sets\n"
     "Methods: ORA (Enrichr) on mutated genes · GSEA prerank on log2 FC",
     ACCENT_ORANGE),
    ("M07", "Variant Impact",
     "Data: M02 missense variants · AlphaMissense pre-computed table (DeepMind) · UniProt ID mapping\n"
     "Methods: Table lookup by UniProt ID + amino acid substitution · am_pathogenicity score (0–1)",
     RGBColor(0x2C, 0x3E, 0x50)),
    ("M08", "Treatment Recommendation",
     "Data: M02 variants · NCCN/FDA 24-drug KB · CIViC API · M03/M04/M05 TME · ClinicalTrials.gov (21 trials)\n"
     "Methods: OncoKB/AMP/ESCAT evidence tiers · TME phenotype (Inflamed/Excluded/Desert) · clinical trial matching · MDT report",
     ACCENT_TEAL),
]
# 2×2 grid for 4 modules
positions = [
    (Inches(0.25), Inches(1.35)),
    (Inches(6.8),  Inches(1.35)),
    (Inches(0.25), Inches(4.1)),
    (Inches(6.8),  Inches(4.1)),
]
for i, (num, name, desc, col) in enumerate(modules2):
    x, y = positions[i]
    w, h = Inches(6.2), Inches(2.55)
    add_rect(sl, x, y, w, h, fill=WHITE, line_color=col, line_width=Pt(1.5))
    add_rect(sl, x, y, w, Inches(0.52), fill=col)
    add_textbox(sl, f"{num}  {name}", x + Inches(0.1), y + Inches(0.06),
                w - Inches(0.2), Inches(0.42), font_size=16, bold=True, color=WHITE)
    add_textbox(sl, desc, x + Inches(0.1), y + Inches(0.62),
                w - Inches(0.2), Inches(1.8), font_size=14, color=TEXT_DARK)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 10 – Technology Stack
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(blank_layout)
slide_header(sl, "Technology Stack",
             "Open-source, containerised, GPU-optional pipeline")

tech_rows = [
    ("Variant Annotation",  "Ensembl VEP v113 · PCGR v2.2.5 · COSMIC SBS v3.3"),
    ("Expression Analysis", "pandas · scipy · seaborn · TCGA-LUAD TPM matrix"),
    ("Single-Cell TME",     "scanpy · GSEApy ssGSEA · GSE131907 Lung Cell Atlas"),
    ("Pathway Enrichment",  "GSEApy (ORA + GSEA prerank) · MSigDB C2/C5 gene sets"),
    ("Protein Language AI", "ESM2-650M (facebook/esm2_t33_650M_UR50D) · PyTorch · HuggingFace Transformers"),
    ("Drug Knowledge Base", "Curated NCCN/FDA KB · CIViC REST API · OncoKB evidence levels"),
    ("ML / Statistics",     "scikit-learn · lifelines CoxPH · CoxNet Elastic Net · SHAP"),
    ("Visualization",       "matplotlib · seaborn · plotly (Streamlit)"),
    ("Dashboard",           "Streamlit · Deployed on Streamlit Community Cloud"),
    ("Containerisation",    "Docker · conda (PCGR env) · Python 3.10+"),
]
for i, (layer, tools) in enumerate(tech_rows):
    row_col = LIGHT_GREY if i % 2 == 0 else WHITE
    y = Inches(1.35 + i * 0.615)
    add_rect(sl, Inches(0.3), y, Inches(12.7), Inches(0.60), fill=row_col)
    add_textbox(sl, layer, Inches(0.4), y + Inches(0.07), Inches(2.9), Inches(0.46),
                font_size=15, bold=True, color=MID_BLUE)
    add_textbox(sl, tools, Inches(3.4), y + Inches(0.07), Inches(9.4), Inches(0.46),
                font_size=15, color=TEXT_DARK)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION DIVIDER – RESULTS
# ══════════════════════════════════════════════════════════════════════════════
section_divider("03  Results",
                "Variant Landscape · TME · IO Scoring · Drug Mapping · Integration")


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 11 – Clinical Context & Patient Card
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(blank_layout)
slide_header(sl, "M01: Patient Clinical Context",
             "Automated clinical card generation with Kaplan-Meier survival curves")

add_image_safe(sl, FIGS["patient_card"],
               Inches(0.3), Inches(1.3), Inches(7.2), Inches(5.7))

bullet_block(sl, [
    "Clinical metadata from TCGA-LUAD for each sample",
    "Kaplan-Meier OS curves · cohort-level survival comparison",
    "Stage, age, sex, smoking history, histology subtype",
    "One-page PNG summary card generated per patient",
    "Enables rapid clinical contextualization before molecular review",
], Inches(7.7), Inches(1.5), Inches(5.3), Inches(4.5),
   font_size=14, color=TEXT_DARK, title="Key Features", title_color=MID_BLUE)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 12 – Somatic Variant Landscape
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(blank_layout)
slide_header(sl, "M02: Somatic Variant Annotation & TMB",
             "VEP/PCGR annotation across 272 TCGA-LUAD samples")

add_image_safe(sl, FIGS["variant_sum"],
               Inches(0.3), Inches(1.3), Inches(7.5), Inches(5.7))

bullet_block(sl, [
    "Ensembl VEP v113 functional annotation (canonical transcript)",
    "PCGR v2.2.5: oncogenicity tiers, druggability flags, SBS signatures",
    "TMB calculation: nonsynonymous SNVs per megabase (38 Mb panel)",
    "Top driver genes: KRAS (30%), TP53 (49%), STK11 (17%), EGFR (15%)",
    "SBS4 (tobacco) dominant in heavy smokers; SBS2/13 in APOBEC subgroup",
    "VAF-based clonality assessment for each targetable variant",
], Inches(8.0), Inches(1.5), Inches(5.0), Inches(5.2),
   font_size=13.5, color=TEXT_DARK, title="Variant Annotation Results", title_color=MID_BLUE)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 13 – TME Characterization
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(blank_layout)
slide_header(sl, "M04: Tumor Microenvironment Characterization",
             "Single-cell reference deconvolution across 517 TCGA-LUAD samples")

add_image_safe(sl, FIGS["tme_overview"],
               Inches(0.3), Inches(1.3), Inches(7.5), Inches(3.1))
add_image_safe(sl, FIGS["tme_heatmap"],
               Inches(0.3), Inches(4.5), Inches(7.5), Inches(2.7))

bullet_block(sl, [
    "GSE131907 (Kim et al. 2020): ~200K single cells as deconvolution reference",
    "ssGSEA-based cell-type fraction inference (10 lineages)",
    "Cell types: CD8+ T cells, NK cells, B cells, Tregs, TAMs, CAFs, endothelial",
    "TME phenotypes classified: Inflamed / Immune-Excluded / Desert",
    "~35% tumours classified as immune-inflamed (high IO potential)",
    "STK11/KEAP1 co-mutation enriched in desert/excluded phenotypes",
    "TIL density from H&E (M05) independently validated TME classification",
], Inches(8.0), Inches(1.5), Inches(5.0), Inches(5.2),
   font_size=13, color=TEXT_DARK, title="TME Deconvolution Highlights", title_color=MID_BLUE)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 14 – Pathway Enrichment
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(blank_layout)
slide_header(sl, "M06: Pathway Enrichment Analysis",
             "ORA on mutated genes + GSEA prerank on RNA-seq fold-changes")

add_image_safe(sl, FIGS["gsea"],
               Inches(0.3), Inches(1.3), Inches(7.5), Inches(5.7))

bullet_block(sl, [
    "ORA: over-representation analysis using MSigDB Hallmark + C2/C5 gene sets",
    "GSEA prerank: ranked by log2 fold-change vs. cohort median expression",
    "Per-sample HTML reports with interactive enrichment plots",
    "KRAS-mutant tumours consistently enrich KRAS signalling / EMT pathways",
    "TP53-loss samples enrich DNA damage response and cell cycle gene sets",
    "STK11-mutant enriches oxidative phosphorylation, suppresses IFN-γ signalling",
    "GSEA NES scores used as features in M08 immune activity model",
], Inches(8.0), Inches(1.5), Inches(5.0), Inches(5.2),
   font_size=13, color=TEXT_DARK, title="Pathway Analysis Results", title_color=MID_BLUE)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 15 – Drug Mapping
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(blank_layout)
slide_header(sl, "M07: Targeted Therapy Drug Mapping",
             "Evidence-graded drug recommendations across 103 TCGA-LUAD patients")

add_image_safe(sl, FIGS["drug_heatmap"],
               Inches(0.3), Inches(1.3), Inches(7.2), Inches(5.7))

bullet_block(sl, [
    "24-drug curated knowledge base (KRAS G12C, EGFR, ALK, ROS1, RET, MET, BRAF, NTRK)",
    "Evidence grading: OncoKB levels 1–4 + AMP/ASCO/CAP Tier I–IV",
    "CIViC resistance annotation: penalty applied to confidence score",
    "VAF-weighted actionability: clonal mutations score higher than subclonal",
    "Top actionable drugs: Sotorasib / Adagrasib (KRAS G12C), Osimertinib (EGFR),",
    "    Lorlatinib (ALK/ROS1), Selpercatinib (RET), Capmatinib (MET ex14)",
    "FDA-approved drug found in ≥65% of patients with driver mutations",
], Inches(7.7), Inches(1.5), Inches(5.3), Inches(5.2),
   font_size=13, color=TEXT_DARK, title="Drug Actionability Results", title_color=MID_BLUE)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 16 – Immune Activity Score
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(blank_layout)
slide_header(sl, "M08: Multi-Modal Immune Activity Score (IAS)",
             "CoxNet survival model integrating RNA-seq, genomics, and pathology")

# 2 figures top row, 1 bottom row
add_image_safe(sl, FIGS["km_io"],
               Inches(0.3), Inches(1.3), Inches(4.1), Inches(2.9))
add_image_safe(sl, FIGS["km_stk11"],
               Inches(4.55), Inches(1.3), Inches(4.1), Inches(2.9))
add_image_safe(sl, FIGS["km_gse"],
               Inches(8.8), Inches(1.3), Inches(4.2), Inches(2.9))

add_textbox(sl, "TCGA-LUAD: IAS high vs. low",
            Inches(0.3), Inches(4.2), Inches(4.1), Inches(0.3),
            font_size=10, italic=True, color=TEXT_MED, align=PP_ALIGN.CENTER)
add_textbox(sl, "STK11 subgroup analysis",
            Inches(4.55), Inches(4.2), Inches(4.1), Inches(0.3),
            font_size=10, italic=True, color=TEXT_MED, align=PP_ALIGN.CENTER)
add_textbox(sl, "External validation: GSE72094 (n=398)",
            Inches(8.8), Inches(4.2), Inches(4.2), Inches(0.3),
            font_size=10, italic=True, color=TEXT_MED, align=PP_ALIGN.CENTER)

bullet_block(sl, [
    "CoxNet (Elastic Net Cox PH) selects ~50–150 prognostic RNA features from top-5,000 variance genes",
    "Training: TCGA-LUAD n≈443 · External validation: GSE72094 n=398 · Nested 5-fold CV",
    "IAS high (top tertile) → significantly better OS (p<0.01, log-rank) in both cohorts",
    "STK11-mutant patients remain IO-low even with high TMB — captured by IAS",
    "SHAP analysis identifies IFN-γ signature genes as top predictors",
], Inches(0.3), Inches(4.6), Inches(13.0), Inches(2.6),
   font_size=13, color=TEXT_DARK, title="Model Performance", title_color=MID_BLUE)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 17 – Feature Importance / SHAP
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(blank_layout)
slide_header(sl, "M08: Feature Importance & SHAP Analysis",
             "Interpretable multi-modal predictors of immune activity")

add_image_safe(sl, FIGS["feat_imp"],
               Inches(0.3), Inches(1.3), Inches(6.3), Inches(5.7))
add_image_safe(sl, FIGS["shap"],
               Inches(6.8), Inches(1.3), Inches(6.2), Inches(5.7))

add_textbox(sl, "Top CoxNet-selected gene coefficients",
            Inches(0.3), Inches(7.0), Inches(6.3), Inches(0.3),
            font_size=10, italic=True, color=TEXT_MED, align=PP_ALIGN.CENTER)
add_textbox(sl, "SHAP beeswarm: contribution of each feature to IAS",
            Inches(6.8), Inches(7.0), Inches(6.2), Inches(0.3),
            font_size=10, italic=True, color=TEXT_MED, align=PP_ALIGN.CENTER)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 18 – M08 Treatment Recommendation (3 sub-components)
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(blank_layout)
slide_header(sl, "M08: Treatment Recommendation & Clinical Reporting",
             "Evidence-tiered therapy · Clinical trial matching · MDT report — all within M08")

info_cols = [
    (ACCENT_TEAL, "① Targeted & IO Recommendation",
     ["Evidence tiers: OncoKB · AMP/ASCO/CAP · ESCAT",
      "24-drug KB: EGFR · KRAS G12C · ALK · RET · MET…",
      "IO scoring: TMB + TME + IFN-γ + TIDE dysfunction",
      "Confidence score 0–100 per drug recommendation",
      "CIViC resistance evidence reduces confidence"]),
    (MID_BLUE, "② Clinical Trial Matching",
     ["21-trial curated LUAD database (Phase I–III)",
      "Molecular eligibility matching per variant profile",
      "KRAS G12C · EGFR · ALK · NTRK · TMB strata",
      "Real-time ClinicalTrials.gov API (24h cache)",
      "Returns eligible + potentially eligible trials"]),
    (ACCENT_ORANGE, "③ MDT Report",
     ["Structured one-page clinical summary per patient",
      "Integrates M02 variants + M03 RNA + M04 TME",
      "M08 Immune Activity Score + IO group label",
      "Ranked treatment list with evidence tier",
      "Exportable PNG / JSON for tumour board use"]),
]
for i, (col, title, points) in enumerate(info_cols):
    x = Inches(0.25 + i * 4.35)
    add_rect(sl, x, Inches(1.35), Inches(4.1), Inches(5.7),
             fill=WHITE, line_color=col, line_width=Pt(1.5))
    add_rect(sl, x, Inches(1.35), Inches(4.1), Inches(0.58), fill=col)
    add_textbox(sl, title, x + Inches(0.1), Inches(1.38), Inches(3.9), Inches(0.5),
                font_size=15, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    for j, pt in enumerate(points):
        add_textbox(sl, "• " + pt, x + Inches(0.12), Inches(2.05 + j * 0.95),
                    Inches(3.88), Inches(0.88), font_size=14, color=TEXT_DARK)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION DIVIDER – DISCUSSION
# ══════════════════════════════════════════════════════════════════════════════
section_divider("04  Discussion",
                "Key Findings · Limitations · Future Directions")


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 19 – Key Findings
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(blank_layout)
slide_header(sl, "Key Findings",
             "Biological insights emerging from the 9-module integrated analysis")

findings = [
    (ACCENT_TEAL, "01", "STK11/KEAP1 Co-Mutation as IO Resistance Marker",
     "STK11 loss robustly predicts immune-excluded TME phenotype and low IAS regardless of TMB or PD-L1 status. This dual STK11/KEAP1 co-mutation pattern was the strongest negative predictor in both TCGA-LUAD and GSE72094 validation cohorts."),
    (MID_BLUE, "02", "Multi-Modal IAS Outperforms Single-Biomarker IO Prediction",
     "Integrating RNA immune signatures with genomic and pathology features yields a more stable survival stratification than TMB or PD-L1 alone. The CoxNet model transfers to an independent cohort (GSE72094, n=398), confirming generalisation."),
    (ACCENT_ORANGE, "03", "Targetable Driver Genes Found in >65% of Profiled Patients",
     "With a 24-drug knowledge base and CIViC evidence grading, the platform identified ≥1 FDA-approved targeted therapy in the majority of variant-profiled patients. KRAS G12C (Sotorasib/Adagrasib) was the most frequent actionable alteration."),
    (RGBColor(0x7D, 0x3C, 0x98), "04", "Cohort-Level Benchmarking Enables Personalised Expression Outlier Detection",
     "Normalising per-sample RNA against the full TCGA-LUAD distribution (n=517) reveals tumour-specific outlier genes not apparent from absolute expression values — critical for identifying novel target candidates."),
]
for i, (col, num, title, desc) in enumerate(findings):
    y = Inches(1.35 + i * 1.5)
    add_rect(sl, Inches(0.3), y, Inches(0.55), Inches(1.3), fill=col)
    add_textbox(sl, num, Inches(0.3), y + Inches(0.35), Inches(0.55), Inches(0.55),
                font_size=22, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_rect(sl, Inches(0.9), y, Inches(12.1), Inches(1.3),
             fill=LIGHT_GREY if i % 2 == 0 else WHITE,
             line_color=col, line_width=Pt(1))
    add_textbox(sl, title, Inches(1.05), y + Inches(0.05), Inches(11.8), Inches(0.38),
                font_size=14, bold=True, color=col)
    add_textbox(sl, desc, Inches(1.05), y + Inches(0.48), Inches(11.8), Inches(0.72),
                font_size=12.5, color=TEXT_DARK)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 20 – Limitations
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(blank_layout)
slide_header(sl, "Limitations",
             "Scope boundaries and analytical caveats")

lims = [
    ("Data Limitations",
     ["TCGA samples are retrospective snap-shots; no longitudinal resistance tracking",
      "Bulk RNA-seq TME deconvolution is a proxy — spatial context not captured",
      "Clinical metadata completeness varies across TCGA batches (survival data ~80%)",
      "No fresh prospective validation cohort beyond GSE72094"]),
    ("Methodological Limitations",
     ["ESM2 embeddings computed without true AlphaFold3 structural context",
      "Drug sensitivity predictions rely on genomic biomarkers, not functional assays",
      "CoxNet feature selection is stochastic — minor instability across random seeds",
      "Clinical trial eligibility is rule-based, not validated by oncologist review"]),
    ("Platform Limitations",
     ["ESM2 module (M07b) requires GPU; disabled by default in cloud deployment",
      "PCGR/VEP annotation requires local Docker install; not available on Streamlit Cloud",
      "CIViC and ClinicalTrials.gov data fetched at build time — real-world lag possible",
      "Dashboard is read-only; does not support user-uploaded VCF/BAM files yet"]),
]
for i, (title, bullets) in enumerate(lims):
    x = Inches(0.3 + i * 4.35)
    add_rect(sl, x, Inches(1.35), Inches(4.1), Inches(5.7),
             fill=WHITE, line_color=ACCENT_ORANGE, line_width=Pt(1.5))
    add_rect(sl, x, Inches(1.35), Inches(4.1), Inches(0.56), fill=ACCENT_ORANGE)
    add_textbox(sl, title, x + Inches(0.1), Inches(1.38), Inches(3.9), Inches(0.48),
                font_size=16, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    for j, b in enumerate(bullets):
        add_textbox(sl, "• " + b, x + Inches(0.12), Inches(2.04 + j * 1.26),
                    Inches(3.88), Inches(1.18), font_size=15, color=TEXT_DARK)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 21 – Future Directions
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(blank_layout)
slide_header(sl, "Future Directions",
             "Expanding the platform toward real-world clinical utility")

directions = [
    (ACCENT_TEAL,   "Short-term (6–12 months)",
     "• Support user-uploaded VCF / MAF input for prospective patients\n"
     "• Integrate AlphaFold3 structural context into ESM2 variant scoring\n"
     "• Expand drug knowledge base to 50+ drugs (NTRK1/2/3, NRG1, HER2)\n"
     "• Add WGS-level copy number alteration and structural variant tracks"),
    (MID_BLUE,      "Medium-term (1–2 years)",
     "• Prospective validation in an independent clinical cohort (NLST / LCMC3)\n"
     "• Spatial transcriptomics integration for TME-topology-aware IO scoring\n"
     "• Real-time EHR integration for treatment outcome feedback loop\n"
     "• Multi-cancer extension: LUSC, SCLC, and pan-NSCLC mode"),
    (ACCENT_ORANGE, "Long-term Vision",
     "• Federated learning across hospital systems without data sharing\n"
     "• Digital pathology deep learning for direct TIL/TME quantification\n"
     "• LLM-powered natural language MDT report generation\n"
     "• Regulatory-grade software as a medical device (SaMD) pathway"),
]
for i, (col, title, text) in enumerate(directions):
    y = Inches(1.4 + i * 2.05)
    add_rect(sl, Inches(0.3), y, Inches(12.7), Inches(1.9),
             fill=WHITE, line_color=col, line_width=Pt(1.5))
    add_rect(sl, Inches(0.3), y, Inches(2.7), Inches(1.9), fill=col)
    add_textbox(sl, title, Inches(0.35), y + Inches(0.65), Inches(2.5), Inches(0.65),
                font_size=16, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_textbox(sl, text, Inches(3.1), y + Inches(0.1), Inches(9.7), Inches(1.7),
                font_size=15, color=TEXT_DARK)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 22 – Conclusion
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(blank_layout)
add_rect(sl, 0, 0, SLIDE_W, SLIDE_H, fill=DARK_NAVY)
add_rect(sl, 0, Inches(1.1), SLIDE_W, Inches(0.06), fill=ACCENT_TEAL)

add_textbox(sl, "Conclusion",
            Inches(0.5), Inches(0.15), Inches(12.3), Inches(0.85),
            font_size=36, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

conclusions = [
    ("End-to-end integration",
     "The LUAD Precision Oncology Platform integrates 9 analysis modules — somatic variants, "
     "RNA-seq, single-cell TME, pathology, protein AI (ESM2 + AlphaMissense), and clinical evidence "
     "— into a single reproducible, patient-facing Streamlit dashboard."),
    ("Validated IO stratification",
     "The multi-modal Immune Activity Score stratifies LUAD patients by immunotherapy suitability "
     "with external validation (GSE72094), capturing STK11/KEAP1-driven IO resistance "
     "not detectable by TMB or PD-L1 alone."),
    ("Actionable clinical outputs",
     "For >65% of molecularly profiled patients, the platform identifies ≥1 FDA-approved targeted "
     "therapy with evidence-graded confidence scores, clinical trial matches, and a structured "
     "MDT report — reducing time from sequencing to clinical decision."),
]
for i, (title, text) in enumerate(conclusions):
    y = Inches(1.3 + i * 1.95)
    add_rect(sl, Inches(0.4), y, Inches(12.5), Inches(1.8),
             fill=RGBColor(0x15, 0x2A, 0x40))
    add_rect(sl, Inches(0.4), y, Inches(0.08), Inches(1.8), fill=ACCENT_TEAL)
    add_textbox(sl, title, Inches(0.6), y + Inches(0.1), Inches(12.0), Inches(0.45),
                font_size=18, bold=True, color=ACCENT_TEAL)
    add_textbox(sl, text, Inches(0.6), y + Inches(0.58), Inches(12.0), Inches(1.1),
                font_size=15, color=RGBColor(0xCC, 0xDD, 0xEE))


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 23 – Acknowledgements
# ══════════════════════════════════════════════════════════════════════════════
sl = prs.slides.add_slide(blank_layout)
slide_header(sl, "Acknowledgements & References")

add_textbox(sl, "Data & Resources",
            Inches(0.4), Inches(1.28), Inches(12.5), Inches(0.46),
            font_size=18, bold=True, color=MID_BLUE)

acks = [
    "TCGA Research Network (NIH/NCI) — patient genomic and clinical data",
    "GSE131907 — Kim et al. (2020) Lung Cancer Cell Atlas, Nat Commun",
    "GSE72094  — Lee et al. (2016) Early-stage LUAD cohort, Genome Biol",
    "PCGR/VEP  — Sigven Nakken et al., Genome Med (VEP v113, PCGR v2.2.5)",
    "ESM2      — Lin et al. (2023) Meta AI, Science (ESM2-650M)",
    "CIViC     — Griffith et al. (2017) Nat Genet",
    "OncoKB    — Chakravarty et al. (2017) JCO Precis Oncol",
    "IFN-γ / TIDE — Ayers et al. (2017) JCI · Jiang et al. (2018) Nat Med",
    "CYT score / TMB — Rooney et al. (2015) Cell · Marabelle et al. (2020) JAMA Oncol",
]
for j, ack in enumerate(acks):
    add_textbox(sl, "  • " + ack,
                Inches(0.5), Inches(1.82 + j * 0.575), Inches(12.3), Inches(0.55),
                font_size=15, color=TEXT_DARK)

add_textbox(sl,
    "Platform:  https://luad-platform.streamlit.app   |   "
    "GitHub: https://github.com/yujuan-zhang/luad",
    Inches(0.4), Inches(7.0), Inches(12.5), Inches(0.42),
    font_size=14, color=ACCENT_TEAL, italic=True)


# ── SAVE ─────────────────────────────────────────────────────────────────────
out_path = BASE / "LUAD_Platform_Presentation.pptx"
prs.save(str(out_path))
print(f"✓  Saved: {out_path}")
print(f"   Slides: {len(prs.slides)}")
