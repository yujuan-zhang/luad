#!/usr/bin/env python
"""
generate_home_figures.py
------------------------
Generate three high-quality figures for the Home page:
  1. background_luad.png  — LUAD driver landscape + therapeutic challenge
  2. cohort_overview.png  — TCGA cohort statistics (real data)
  3. pipeline_figure.png  — 10-module pipeline architecture
"""

import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).parent.parent
OUT_DIR     = PROJECT_DIR / "data" / "output" / "home_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        False,
    "figure.facecolor": "white",
})


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — LUAD driver landscape
# ══════════════════════════════════════════════════════════════════════════════

def fig_background():
    """
    Two-panel background figure:
      A: LUAD driver gene alteration frequencies (TCGA 2014 + Skoulidis 2018)
      B: Therapeutic category map — what each mutation means clinically
    """
    fig = plt.figure(figsize=(16, 7))
    fig.patch.set_facecolor("white")
    gs  = gridspec.GridSpec(1, 2, figure=fig,
                            wspace=0.08, left=0.05, right=0.97,
                            top=0.88, bottom=0.10)

    # ── Panel A: mutation frequency bar chart ────────────────────────────────
    ax = fig.add_subplot(gs[0])

    # Literature frequencies (TCGA LUAD, Campbell 2016, Skoulidis 2018)
    genes  = ["TP53", "KRAS", "KEAP1", "STK11", "EGFR", "NF1",
              "BRAF", "ALK\n(fusion)", "ERBB2", "MET\n(ex14)", "RET\n(fusion)"]
    freqs  = [46, 29, 17, 17, 14, 11, 7, 5, 3, 3, 2]
    # Color coding: orange = IO resistance, blue = targeted therapy, grey = tumor suppressor
    colors = ["#78909c",   # TP53  — TSG
              "#1565c0",   # KRAS  — targeted (Sotorasib/Adagrasib)
              "#e53935",   # KEAP1 — IO resistance
              "#e53935",   # STK11 — IO resistance
              "#1565c0",   # EGFR  — targeted
              "#78909c",   # NF1   — TSG
              "#1565c0",   # BRAF  — targeted
              "#1565c0",   # ALK   — targeted
              "#1565c0",   # ERBB2 — targeted
              "#1565c0",   # MET   — targeted
              "#1565c0"]   # RET   — targeted

    y = np.arange(len(genes))
    bars = ax.barh(y, freqs, color=colors, height=0.62,
                   edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, freqs):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val}%", va="center", fontsize=9.5, color="#333", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(genes, fontsize=10.5)
    ax.set_xlabel("Alteration frequency (%)", fontsize=10)
    ax.set_xlim(0, 58)
    ax.invert_yaxis()
    ax.set_title("A   LUAD Driver Alteration Frequencies",
                 fontsize=12, fontweight="bold", loc="left", pad=10)

    legend_items = [
        mpatches.Patch(color="#1565c0", label="Targetable driver mutation"),
        mpatches.Patch(color="#e53935", label="IO resistance marker (STK11/KEAP1)"),
        mpatches.Patch(color="#78909c", label="Tumor suppressor / other"),
    ]
    ax.legend(handles=legend_items, fontsize=9, loc="lower right", framealpha=0.9)

    # ── Panel B: Therapeutic implications grid ───────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_title("B   Precision Therapy Decision Map",
                  fontsize=12, fontweight="bold", loc="left", pad=10)

    BOXES = [
        # (x, y, w, h, bg, border, title, lines)
        (0.2, 6.8, 4.5, 2.8, "#e3f2fd", "#1565c0",
         "Targeted Therapy",
         ["KRAS G12C  →  Sotorasib / Adagrasib",
          "EGFR exon19/21  →  Osimertinib",
          "ALK fusion  →  Lorlatinib / Alectinib",
          "RET fusion  →  Selpercatinib",
          "MET ex14  →  Tepotinib",
          "ERBB2  →  T-DXd (Trastuzumab deruxtecan)"]),
        (5.1, 6.8, 4.5, 2.8, "#e8f5e9", "#2e7d32",
         "Immunotherapy",
         ["TMB-high (≥10 mut/Mb)  →  Pembrolizumab",
          "Inflamed TME  →  PD-1/L1 checkpoint",
          "TIS / CYT high  →  IO suitability",
          "M08 Immune Activity Score"]),
        (0.2, 3.5, 4.5, 2.8, "#ffebee", "#c62828",
         "IO Resistance (avoid/combine)",
         ["STK11 mutation  →  reduced IO efficacy",
          "KEAP1 mutation  →  reduced IO efficacy",
          "STK11 + KRAS G12C  →  KRYSTAL-7",
          "Consider targeted + IO combination"]),
        (5.1, 3.5, 4.5, 2.8, "#f3e5f5", "#6a1b9a",
         "Clinical Trial Priority",
         ["Phase III: CodeBreaK200, KRYSTAL-12",
          "Phase III: FLAURA2, MARIPOSA, PAPILLON",
          "Phase III: CROWN, LIBRETTO-431",
          "21 trials matched by molecular profile"]),
        (0.2, 0.4, 9.4, 2.7, "#fff8e1", "#f57f17",
         "Multi-omics Integration (this platform)",
         ["M02 somatic variants  +  M03 RNA-seq  +  M04 ssGSEA TME  +  M05 H&E TIL",
          "M07 variant impact (AlphaMissense)  +  M08 Immune Activity Score",
          "M09 treatment recommendations (OncoKB · AMP/ASCO/CAP · ESCAT · CIViC)",
          "M10 clinical trial matching  +  MDT report"]),
    ]

    for bx, by, bw, bh, bg, bc, title, lines in BOXES:
        ax2.add_patch(FancyBboxPatch((bx, by), bw, bh,
                                    boxstyle="round,pad=0.18",
                                    facecolor=bg, edgecolor=bc,
                                    linewidth=1.8, zorder=1))
        ax2.text(bx + 0.18, by + bh - 0.18, title,
                 fontsize=11.5, fontweight="bold", color=bc,
                 va="top", ha="left", zorder=3)
        for li, line in enumerate(lines):
            ax2.text(bx + 0.25, by + bh - 0.52 - li * 0.44, f"• {line}",
                     fontsize=9.5, color="#333", va="top", ha="left", zorder=3)

    fig.suptitle("LUAD Precision Oncology — Background & Clinical Rationale",
                 fontsize=14, fontweight="bold", y=0.97, color="#1a1a2e")

    out = OUT_DIR / "background_luad.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {out.name}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — TCGA cohort overview (real data)
# ══════════════════════════════════════════════════════════════════════════════

def fig_cohort():
    """Four-panel cohort overview using actual project data."""
    CLIN_PATH   = PROJECT_DIR / "data/clinical/tcga_luad_survival.tsv"
    SUM_PATH    = PROJECT_DIR / "data/output/09_integration/integration_summary.tsv"
    IO_PATH     = PROJECT_DIR / "data/output/08_io_ml/io_scores.tsv"

    clin = pd.read_csv(CLIN_PATH, sep="\t") if CLIN_PATH.exists() else pd.DataFrame()
    summ = pd.read_csv(SUM_PATH,  sep="\t") if SUM_PATH.exists() else pd.DataFrame()
    io   = pd.read_csv(IO_PATH,   sep="\t", index_col=0) if IO_PATH.exists() else pd.DataFrame()

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("white")
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.48, wspace=0.38,
                            top=0.88, bottom=0.07,
                            left=0.06, right=0.97)

    BLUE, RED, GREEN, ORANGE, PURPLE = (
        "#1565c0", "#c62828", "#2e7d32", "#e65100", "#6a1b9a"
    )

    # ── Panel A: Stage distribution ──────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    if not clin.empty and "stage" in clin.columns:
        STAGE_MAP = {"Stage IA": "IA", "Stage IB": "IB", "Stage IIA": "IIA",
                     "Stage IIB": "IIB", "Stage IIIA": "IIIA", "Stage IIIB": "IIIB",
                     "Stage IV": "IV", "Stage I": "I", "Stage II": "II"}
        sc = clin["stage"].map(STAGE_MAP).dropna()
        ORDER = ["IA", "IB", "I", "IIA", "IIB", "II", "IIIA", "IIIB", "IV"]
        sc_c  = sc.value_counts().reindex([s for s in ORDER if s in sc.unique()]).dropna()
        total = sc_c.sum()
        stage_colors = {
            "IA":"#4fc3f7","IB":"#039be5","I":"#0277bd",
            "IIA":"#a5d6a7","IIB":"#43a047","II":"#2e7d32",
            "IIIA":"#ffb74d","IIIB":"#f57c00",
            "IV":"#e53935",
        }
        cols = [stage_colors.get(s, "#aaa") for s in sc_c.index]
        bars = ax0.bar(range(len(sc_c)), sc_c.values, color=cols,
                       edgecolor="white", linewidth=0.8)
        ax0.set_xticks(range(len(sc_c)))
        ax0.set_xticklabels(sc_c.index, fontsize=9, rotation=30, ha="right")
        ax0.set_ylabel("Patients", fontsize=9)
        ax0.set_title("A   Pathologic Stage Distribution", fontsize=10,
                      fontweight="bold", loc="left")
        for bar, val in zip(bars, sc_c.values):
            ax0.text(bar.get_x() + bar.get_width()/2, val + 0.8,
                     f"{val}\n({100*val/total:.0f}%)",
                     ha="center", va="bottom", fontsize=7.5)
        ax0.spines["left"].set_visible(True)

    # ── Panel B: OS distribution ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    if not clin.empty and "os_months" in clin.columns:
        os_v = pd.to_numeric(clin["os_months"], errors="coerce").dropna()
        ax1.hist(os_v, bins=30, color=BLUE, alpha=0.85, edgecolor="white", linewidth=0.6)
        ax1.axvline(os_v.median(), color=RED, linewidth=1.8, linestyle="--",
                    label=f"Median {os_v.median():.1f} mo")
        ax1.set_xlabel("Overall Survival (months)", fontsize=9)
        ax1.set_ylabel("Patients", fontsize=9)
        ax1.set_title("B   Overall Survival Distribution", fontsize=10,
                      fontweight="bold", loc="left")
        ax1.legend(fontsize=8, framealpha=0.8)

    # ── Panel C: TMB distribution ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    if not summ.empty and "tmb" in summ.columns:
        tmb = summ["tmb"].dropna()
        ax2.hist(tmb, bins=30, color=GREEN, alpha=0.85,
                 edgecolor="white", linewidth=0.6)
        ax2.axvline(10, color=RED, linewidth=1.8, linestyle="--",
                    label="TMB-high cutoff (10)")
        n_high = (tmb >= 10).sum()
        ax2.set_xlabel("TMB (mutations/Mb)", fontsize=9)
        ax2.set_ylabel("Patients", fontsize=9)
        ax2.set_title("C   Tumor Mutational Burden (TMB)", fontsize=10,
                      fontweight="bold", loc="left")
        ax2.legend(fontsize=8, framealpha=0.8)
        ax2.text(0.97, 0.95, f"TMB-high:\n{n_high} ({100*n_high/len(tmb):.0f}%)",
                 transform=ax2.transAxes, fontsize=8, ha="right", va="top",
                 color=RED, fontweight="bold")

    # ── Panel D: TME phenotype donut ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    if not summ.empty and "tme_phenotype" in summ.columns:
        tc = summ["tme_phenotype"].value_counts()
        tc = tc[tc.index != "Unknown"]
        TC_COLORS = {"Inflamed": "#d73027", "Excluded": "#fc8d59", "Desert": "#4575b4"}
        cols = [TC_COLORS.get(p, "#aaa") for p in tc.index]
        wedges, texts, autotexts = ax3.pie(
            tc.values, colors=cols, autopct="%1.0f%%",
            pctdistance=0.78, startangle=90,
            wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
        )
        for t in texts: t.set_fontsize(0)
        for at in autotexts: at.set_fontsize(9); at.set_fontweight("bold")
        legend_h = [mpatches.Patch(color=TC_COLORS.get(p, "#aaa"),
                                   label=f"{p} (n={n})")
                    for p, n in tc.items()]
        ax3.legend(handles=legend_h, fontsize=9, loc="lower center",
                   bbox_to_anchor=(0.5, -0.18), ncol=1, framealpha=0.9)
        ax3.set_title("D   TME Immune Phenotype\n(M04/M08, n=271 w/ M02)",
                      fontsize=10, fontweight="bold", loc="left")

    # ── Panel E: Immune Activity Score distribution ───────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    if not io.empty and "io_score" in io.columns:
        sc = io["io_score"].dropna()
        ax4.hist(sc, bins=30, color=PURPLE, alpha=0.85,
                 edgecolor="white", linewidth=0.6)
        ax4.axvline(sc.median(), color=RED, linewidth=1.8, linestyle="--",
                    label=f"Median {sc.median():.1f}")
        ax4.set_xlabel("Immune Activity Score (0–100)", fontsize=9)
        ax4.set_ylabel("Patients", fontsize=9)
        ax4.set_title("E   Multi-modal Immune Activity Score\n"
                      f"(M08 CoxNet, n={len(sc)}, C-index 0.786)",
                      fontsize=10, fontweight="bold", loc="left")
        ax4.legend(fontsize=8, framealpha=0.8)

    # ── Panel F: top targeted drug distribution ───────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    if not summ.empty and "top_targeted_drug" in summ.columns:
        dc = summ["top_targeted_drug"].value_counts()
        dc = dc[~dc.index.isin(["None", "nan", ""])].head(8)
        if not dc.empty:
            cmap_colors = plt.cm.Set2.colors[:len(dc)]
            ax5.barh(range(len(dc)), dc.values[::-1],
                     color=list(reversed(cmap_colors)), edgecolor="white")
            ax5.set_yticks(range(len(dc)))
            ax5.set_yticklabels(list(reversed(dc.index)), fontsize=8.5)
            ax5.set_xlabel("Patients", fontsize=9)
            ax5.set_title("F   Top Targeted Drug Eligibility\n(M09 integration)",
                          fontsize=10, fontweight="bold", loc="left")
            for i, v in enumerate(reversed(dc.values)):
                ax5.text(v + 0.2, i, str(v), va="center", fontsize=8.5)

    n_total = len(clin) if not clin.empty else "517"
    fig.suptitle(f"TCGA-LUAD Cohort Overview  ·  n = {n_total} patients",
                 fontsize=14, fontweight="bold", y=0.97, color="#1a1a2e")

    out = OUT_DIR / "cohort_overview.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {out.name}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — 10-module pipeline architecture
# ══════════════════════════════════════════════════════════════════════════════

def fig_pipeline():
    """
    Clean 10-module pipeline architecture diagram.

    Layout (data units, W=22, H=15):
      y 14.3–14.8  Title
      y 13.0–14.0  INPUT bar (full width)
      y  5.8–12.6  4 balanced columns (2 modules each → uniform box sizes)
                     Col 1: CLINICAL & VARIANTS  (M01 + M02)
                     Col 2: TRANSCRIPTOMICS       (M03 + M04)
                     Col 3: ADVANCED GENOMICS     (M06 + M07)
                     Col 4: PATHOLOGY & AI        (M05 + M08)
      y  3.9– 5.4  M09 integration bar (full width)
      y  2.1– 3.5  M10 clinical recommendation bar (full width)
      y  0.5– 1.7  Streamlit Dashboard bar
    """
    W, H = 22.0, 15.0
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ── Helpers ───────────────────────────────────────────────────────────────
    def fbox(x, y, w, h, bg, bd, lw=1.8, r=0.22):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad={r}",
            facecolor=bg, edgecolor=bd, linewidth=lw, zorder=3))

    def txt(x, y, s, sz=9, color="#1a1a2e", bold=False, ha="center", va="center"):
        ax.text(x, y, s, fontsize=sz, color=color,
                fontweight="bold" if bold else "normal",
                ha=ha, va=va, zorder=5, multialignment="center")

    def arr_v(x, y1, y2, color="#78909c"):
        ax.annotate("", xy=(x, y2), xytext=(x, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=2.2, mutation_scale=20), zorder=4)

    # ── Geometry constants ────────────────────────────────────────────────────
    MARGIN  = 0.60
    COL_W   = 4.50     # each group column width  (4×4.50 + 3×1.00 + 2×0.60 = 22.2 ≈ 22)
    COL_GAP = 1.00     # wider gap so columns breathe
    COL_X   = [MARGIN + i * (COL_W + COL_GAP) for i in range(4)]

    # Vertical zones (y_bottom, y_top)
    Z_INPUT  = (13.2, 14.1)
    Z_GROUPS = ( 6.0, 12.2)   # group frames  →  GH = 6.2
    Z_M09    = ( 4.0,  5.5)
    Z_M10    = ( 2.2,  3.5)
    Z_DASH   = ( 0.5,  1.7)

    # ── Title ─────────────────────────────────────────────────────────────────
    txt(W/2, 14.55,
        "LUAD Precision Oncology Platform — 10-Module Pipeline Architecture",
        sz=14.5, bold=True, color="#1a1a2e")

    # ── INPUT bar ─────────────────────────────────────────────────────────────
    ix, iy = MARGIN, Z_INPUT[0]
    iw = W - 2 * MARGIN
    ih = Z_INPUT[1] - Z_INPUT[0]
    fbox(ix, iy, iw, ih, "#f5f5f5", "#9e9e9e", lw=1.5, r=0.18)
    txt(ix + 0.5, iy + ih * 0.72, "INPUT DATA", sz=11.0, bold=True, color="#555", ha="left")
    txt(W / 2, iy + ih * 0.28,
        "MAF (somatic variants)  ·  Bulk RNA-seq TPM  ·  GDC Clinical metadata  ·  "
        "H&E whole-slide images  ·  scRNA-seq reference (GSE131907, 57k cells)",
        sz=10.0, color="#666")

    # Arrows INPUT → column tops
    col_centres = [cx + COL_W / 2 for cx in COL_X]
    for cx in col_centres:
        arr_v(cx, Z_INPUT[0], Z_GROUPS[1] + 0.03, color="#90a4ae")

    # ── 4 balanced modality columns (2 modules each) ──────────────────────────
    # All columns have exactly 2 modules → every module box is the same height.

    COLUMNS = [
        # (bg, border, label_color, two-line label, [(module_id, title, subtitle), ...])
        ("#fff3e0", "#e65100", "#e65100",
         "CLINICAL &\nVARIANTS",
         [("M01", "Patient Context",     "Survival · Stage · Demographics"),
          ("M02", "Variant Annotation",  "VEP · PCGR · TMB · SBS")]),

        ("#e8f5e9", "#2e7d32", "#2e7d32",
         "TRANSCRIPTOMICS",
         [("M03", "Expression Analysis", "RNA-seq TPM · GTEx z-score"),
          ("M04", "Single-Cell TME",     "ssGSEA · GSE131907 ref · 57k cells")]),

        ("#e3f2fd", "#1565c0", "#1565c0",
         "ADVANCED\nGENOMICS",
         [("M06", "Pathway Enrichment",  "ORA · GSEA · KEGG · MSigDB"),
          ("M07", "Variant Impact",      "AlphaMissense")]),

        ("#f3e5f5", "#6a1b9a", "#6a1b9a",
         "PATHOLOGY\n& AI",
         [("M05", "Digital Pathology",   "H&E WSI · TIL density · TME"),
          ("M08", "Immune Activity",     "CoxNet multi-modal · C-index 0.786")]),
    ]

    GY_TOP  = Z_GROUPS[1]    # 12.6
    GY_BOT  = Z_GROUPS[0]    # 5.8
    GH      = GY_TOP - GY_BOT   # 6.8

    LABEL_H  = 0.80   # height for group label
    PAD_SIDE = 0.28   # more side padding → narrower module boxes
    PAD_TOP  = 0.25
    PAD_BOT  = 0.25
    MOD_GAP  = 0.55   # more gap between the two module boxes

    # n=2 for all columns → compute mod_h once
    n_mods = 2
    avail  = GH - LABEL_H - PAD_TOP - PAD_BOT
    mod_h  = (avail - MOD_GAP * (n_mods - 1)) / n_mods   # ≈ 2.61

    for ci, (gbg, gbd, glc, glabel, mods) in enumerate(COLUMNS):
        gx = COL_X[ci]
        gw = COL_W

        # Outer group frame (full zone height — same for all columns)
        fbox(gx, GY_BOT, gw, GH, gbg, gbd, lw=2.2, r=0.30)

        # Group label (centred, near top; supports 2-line labels)
        txt(gx + gw / 2, GY_TOP - LABEL_H / 2 - 0.06,
            glabel, sz=11.5, bold=True, color=glc)

        # Module boxes — stacked top-down, all same height
        mx = gx + PAD_SIDE
        mw = gw - 2 * PAD_SIDE

        for mi, (mid, mline1, mline2) in enumerate(mods):
            my = GY_TOP - LABEL_H - PAD_TOP - (mi + 1) * mod_h - mi * MOD_GAP

            # White module card with coloured border
            fbox(mx, my, mw, mod_h, "white", gbd, lw=1.8, r=0.20)

            # Coloured accent strip at top of each card
            ax.add_patch(FancyBboxPatch(
                (mx, my + mod_h - 0.38), mw, 0.38,
                boxstyle="round,pad=0.05",
                facecolor=gbg, edgecolor="none", zorder=4))

            # Module ID on accent strip
            txt(mx + mw / 2, my + mod_h - 0.19,
                mid, sz=13, bold=True, color=gbd)

            # Module title
            txt(mx + mw / 2, my + mod_h * 0.46,
                mline1, sz=10.5, color="#222", bold=False)

            # Module subtitle
            txt(mx + mw / 2, my + mod_h * 0.18,
                mline2, sz=9.5, color="#555")

    # ── Arrows from group bottoms to M09 ──────────────────────────────────────
    for cx in col_centres:
        arr_v(cx, Z_GROUPS[0], Z_M09[1] + 0.03, color="#546e7a")

    # ── M09 bar ───────────────────────────────────────────────────────────────
    M09_MAR = 1.8           # M09/M10 inset from figure edge
    m09x, m09y = M09_MAR, Z_M09[0]
    m09w = W - 2 * M09_MAR
    m09h = Z_M09[1] - Z_M09[0]
    fbox(m09x, m09y, m09w, m09h, "#e8eaf6", "#283593", lw=2.5, r=0.28)
    txt(m09x + 0.7, m09y + m09h * 0.70,
        "M09", sz=15, bold=True, color="#283593", ha="left")
    txt(W / 2 + 0.8, m09y + m09h * 0.70,
        "Multi-omics Integration & Treatment Recommendation",
        sz=12.5, bold=True, color="#283593")
    txt(W / 2 + 0.8, m09y + m09h * 0.28,
        "OncoKB · AMP-ASCO-CAP · ESCAT · CIViC  ·  "
        "Targeted / Immunotherapy / Combination / Chemotherapy  ·  271 patients",
        sz=10.0, color="#444")

    arr_v(W / 2, Z_M09[0], Z_M10[1] + 0.03, color="#283593")

    # ── M10 bar ───────────────────────────────────────────────────────────────
    m10x, m10y = M09_MAR, Z_M10[0]
    m10w = W - 2 * M09_MAR
    m10h = Z_M10[1] - Z_M10[0]
    fbox(m10x, m10y, m10w, m10h, "#e8f5e9", "#1b5e20", lw=2.5, r=0.28)
    txt(m10x + 0.7, m10y + m10h * 0.70,
        "M10", sz=15, bold=True, color="#1b5e20", ha="left")
    txt(W / 2 + 0.8, m10y + m10h * 0.70,
        "Clinical Trial Matching & MDT Report",
        sz=12.5, bold=True, color="#1b5e20")
    txt(W / 2 + 0.8, m10y + m10h * 0.28,
        "21 curated LUAD trials (Phase I–III)  ·  "
        "Mutation / TMB / TME-based matching  ·  MDT one-page report (7-panel figure)",
        sz=10.0, color="#444")

    arr_v(W / 2, Z_M10[0], Z_DASH[1] + 0.03, color="#1b5e20")

    # ── Streamlit Dashboard bar ───────────────────────────────────────────────
    dx, dy = 3.8, Z_DASH[0]
    dw = W - 7.6
    dh = Z_DASH[1] - Z_DASH[0]
    fbox(dx, dy, dw, dh, "#1565c0", "#0d47a1", lw=2.8, r=0.24)
    txt(dx + dw / 2, dy + dh * 0.68,
        "STREAMLIT PRECISION ONCOLOGY DASHBOARD",
        sz=13, bold=True, color="white")
    txt(dx + dw / 2, dy + dh * 0.28,
        "Interactive per-patient explorer  ·  10 analysis modules  ·  TCGA-LUAD  ·  n = 517 patients",
        sz=10.5, color="#bbdefb")

    out = OUT_DIR / "pipeline_figure.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {out.name}")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating Home page figures ...")
    fig_background()
    fig_cohort()
    fig_pipeline()
    print("Done. All figures saved to:", OUT_DIR)
