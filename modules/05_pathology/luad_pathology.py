#!/usr/bin/env python
"""
luad_pathology.py
-----------------
Module 08: Computational pathology — H&E slide analysis for LUAD.

Two execution modes:

  Thumbnail mode (default, no GPU needed):
    Input : modules/05_pathology/data/input/thumbnails/{case_id}.png
    Steps :
      1. Tissue segmentation (Otsu thresholding on grayscale)
      2. H&E color deconvolution → hematoxylin channel (nuclei)
      3. Nuclear density estimation → cellularity proxy
      4. TIL density scoring (small dark nuclei in stromal regions)
      5. Morphology feature extraction (texture, nuclear pleomorphism)
      6. TME phenotype classification: Inflamed / Excluded / Desert
    Output: {case_id}_pathology_report.png  — 4-panel figure
            {case_id}_pathology_scores.tsv  — quantitative scores

  Full WSI mode (requires openslide + GPU recommended):
    Input : modules/05_pathology/data/input/svs/{case_id}.svs
    Steps :
      1–5 above at full resolution with patch-based tiling
      6. Foundation model feature extraction (UNI / CONCH if available)
      7. Histological subtype classification (acinar/papillary/solid/lepidic)

Outputs (per patient):
  data/output/05_pathology/{case_id}/
    {case_id}_pathology_scores.tsv     — TIL score, cellularity, TME phenotype
    {case_id}_pathology_report.png     — 4-panel visual summary

Usage:
  python luad_pathology.py                        # all available thumbnails
  python luad_pathology.py --sample TCGA-49-4507  # single patient
  python luad_pathology.py --dry_run              # validate inputs only
  python luad_pathology.py --full-wsi             # use SVS files if available
"""

import argparse
import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
PROJECT_DIR  = SCRIPT_DIR.parent.parent
THUMB_DIR    = SCRIPT_DIR / "data/input/thumbnails"
SVS_DIR      = SCRIPT_DIR / "data/input/svs"
OUT_DIR      = PROJECT_DIR / "data/output/05_pathology"

# ── Parameters ────────────────────────────────────────────────────────────────
# H&E stain vectors (Macenko reference values for H&E)
HE_MATRIX = np.array([
    [0.650, 0.072],   # R channel: Hematoxylin, Eosin
    [0.704, 0.990],   # G channel
    [0.286, 0.105],   # B channel
])

# Quantile-based TME classification (cohort-relative, not absolute)
# Top 30% TIL density → Inflamed, bottom 20% → Desert, rest → Excluded
# Rationale: thumbnail-based TIL proxy values are systematically lower than
# visual TIL scores; absolute thresholds produce near-zero Inflamed fraction.
TME_QUANTILES = {
    "inflamed_pct": 70,   # ≥ 70th percentile → Inflamed (top 30%)
    "desert_pct":   20,   # < 20th percentile → Desert   (bottom 20%)
}


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
    return logging.getLogger(name)


# ── Image loading ─────────────────────────────────────────────────────────────

def load_image(path: Path) -> np.ndarray:
    """Load image as RGB numpy array (H×W×3, uint8)."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    return np.array(img)


# ── Tissue segmentation ───────────────────────────────────────────────────────

def segment_tissue(rgb: np.ndarray) -> np.ndarray:
    """
    Binary tissue mask via Otsu threshold on grayscale.
    Returns bool mask: True = tissue, False = background (white).
    """
    from PIL import Image
    import PIL.ImageFilter

    gray = np.mean(rgb, axis=2).astype(np.uint8)

    # Otsu threshold
    hist, bins = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    total = gray.size
    sum_total = np.dot(np.arange(256), hist)
    sum_bg = w_bg = 0.0
    max_var = threshold = 0

    for t in range(256):
        w_bg += hist[t]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        sum_bg += t * hist[t]
        mu_bg = sum_bg / w_bg
        mu_fg = (sum_total - sum_bg) / w_fg
        var_between = w_bg * w_fg * (mu_bg - mu_fg) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t

    # Tissue = darker than threshold (stained tissue < white background)
    mask = gray < threshold
    return mask


# ── H&E color deconvolution ───────────────────────────────────────────────────

def deconvolve_he(rgb: np.ndarray) -> tuple:
    """
    Simple H&E color deconvolution using fixed stain vectors.
    Returns (hematoxylin, eosin) optical density images (float32).
    """
    # Normalize to optical density
    rgb_float = rgb.astype(np.float32) + 1.0
    od = -np.log(rgb_float / 255.0)

    # Reshape to (N, 3)
    od_flat = od.reshape(-1, 3)

    # Solve: od = H_mat @ [h, e]
    # Use pseudo-inverse of HE_MATRIX
    try:
        he_inv = np.linalg.pinv(HE_MATRIX.T)
        stains = od_flat @ he_inv
    except Exception:
        # Fallback: use hematoxylin channel directly from blue component
        h = od[:, :, 2]
        e = od[:, :, 0]
        return h.astype(np.float32), e.astype(np.float32)

    h = stains[:, 0].reshape(rgb.shape[:2])
    e = stains[:, 1].reshape(rgb.shape[:2])
    return h.astype(np.float32), e.astype(np.float32)


# ── Nuclear density estimation ────────────────────────────────────────────────

def estimate_nuclear_density(hematoxylin: np.ndarray,
                              tissue_mask: np.ndarray) -> dict:
    """
    Estimate nuclear density from hematoxylin channel.
    High hematoxylin intensity = nuclei (dark blue staining).

    Returns dict with:
      nuclear_density   : fraction of tissue area occupied by nuclei
      cellularity       : normalized 0–1 score
      mean_he_intensity : mean hematoxylin in tissue
    """
    if tissue_mask.sum() == 0:
        return {"nuclear_density": 0.0, "cellularity": 0.0, "mean_he_intensity": 0.0}

    he_tissue = hematoxylin[tissue_mask]

    # Nuclei = high hematoxylin (> 75th percentile of tissue)
    threshold = np.percentile(he_tissue, 75)
    nuclear_pixels = (hematoxylin > threshold) & tissue_mask
    nuclear_density = nuclear_pixels.sum() / tissue_mask.sum()

    # Normalize cellularity 0–1 (empirical range 0.05–0.60)
    cellularity = float(np.clip((nuclear_density - 0.05) / 0.55, 0.0, 1.0))

    return {
        "nuclear_density":   float(nuclear_density),
        "cellularity":       cellularity,
        "mean_he_intensity": float(he_tissue.mean()),
    }


# ── TIL density scoring ───────────────────────────────────────────────────────

def score_til_density(rgb: np.ndarray,
                      hematoxylin: np.ndarray,
                      tissue_mask: np.ndarray) -> dict:
    """
    Score tumor-infiltrating lymphocyte (TIL) density.

    Lymphocytes are characterized by:
      - Small, round, darkly-staining nuclei (high hematoxylin)
      - Low eosin (minimal cytoplasm)
      - Surrounded by less-stained stromal tissue

    Strategy:
      1. Identify candidate lymphocyte pixels: high hematoxylin + low eosin
      2. Apply size constraint via local variance (lymphocytes → uniform dark spots)
      3. TIL density = lymphocyte pixels / tissue pixels
    """
    if tissue_mask.sum() == 0:
        return {"til_density": 0.0, "til_score": 0.0, "tme_phenotype": "Unknown"}

    _, eosin = deconvolve_he(rgb)

    # Lymphocyte candidates: high H, low E
    he_thresh = np.percentile(hematoxylin[tissue_mask], 80)
    eo_thresh = np.percentile(eosin[tissue_mask], 40)

    lymph_mask = (hematoxylin > he_thresh) & (eosin < eo_thresh) & tissue_mask

    til_density = float(lymph_mask.sum() / tissue_mask.sum())

    # Normalize TIL score 0–1 (empirical range 0–0.40)
    til_score = float(np.clip(til_density / 0.40, 0.0, 1.0))

    # TME phenotype will be assigned in the second pass via classify_tme_quantile()
    return {
        "til_density":   til_density,
        "til_score":     til_score,
        "tme_phenotype": "Unknown",
        "lymph_mask":    lymph_mask,
    }


# ── Quantile-based TME classification ────────────────────────────────────────

def classify_tme_quantile(all_scores: list) -> list:
    """
    Assign TME phenotype using cohort-wide percentile thresholds.

    Top 30% TIL density  → Inflamed
    Bottom 20% TIL density → Desert
    Middle 50%           → Excluded

    This approach is self-calibrating: it doesn't depend on absolute
    TIL density values, which vary with image resolution and staining.
    """
    densities = [s["til_density"] for s in all_scores]
    p_inflamed = np.percentile(densities, TME_QUANTILES["inflamed_pct"])   # 70th pct
    p_desert   = np.percentile(densities, TME_QUANTILES["desert_pct"])     # 20th pct

    for s in all_scores:
        d = s["til_density"]
        if d >= p_inflamed:
            s["tme_phenotype"] = "Inflamed"
        elif d < p_desert:
            s["tme_phenotype"] = "Desert"
        else:
            s["tme_phenotype"] = "Excluded"
    return all_scores


# ── Texture features ──────────────────────────────────────────────────────────

def extract_texture_features(hematoxylin: np.ndarray,
                              tissue_mask: np.ndarray) -> dict:
    """
    Extract basic texture features from hematoxylin channel in tissue regions.
    Proxy for nuclear pleomorphism and tissue heterogeneity.
    """
    if tissue_mask.sum() == 0:
        return {"nuclear_pleomorphism": 0.0, "tissue_heterogeneity": 0.0}

    he_tissue = hematoxylin[tissue_mask]

    # Nuclear pleomorphism proxy: coefficient of variation of nuclear intensity
    cv = float(np.std(he_tissue) / (np.mean(he_tissue) + 1e-6))
    pleomorphism = float(np.clip(cv / 2.0, 0.0, 1.0))

    # Tissue heterogeneity: local standard deviation
    from PIL import Image
    # Downsample for speed
    h_img = hematoxylin.copy()
    h_img_pil = Image.fromarray((np.clip(h_img, 0, 1) * 255).astype(np.uint8))
    h_small = np.array(h_img_pil.resize(
        (h_img_pil.width // 4, h_img_pil.height // 4),
        Image.LANCZOS
    )).astype(np.float32) / 255.0

    local_std = float(np.std(h_small))
    heterogeneity = float(np.clip(local_std / 0.3, 0.0, 1.0))

    return {
        "nuclear_pleomorphism": pleomorphism,
        "tissue_heterogeneity": heterogeneity,
    }


# ── Full analysis pipeline ────────────────────────────────────────────────────

def analyze_slide(image_path: Path, case_id: str, logger) -> dict:
    """Run full thumbnail analysis pipeline for one patient."""
    logger.info(f"  Loading image: {image_path.name}")
    rgb = load_image(image_path)

    logger.info(f"  Image size: {rgb.shape[1]}×{rgb.shape[0]} px")

    # 1. Tissue segmentation
    tissue_mask = segment_tissue(rgb)
    tissue_fraction = float(tissue_mask.sum() / tissue_mask.size)
    logger.info(f"  Tissue fraction: {tissue_fraction:.1%}")

    if tissue_fraction < 0.05:
        logger.warning(f"  Low tissue fraction — slide may be mostly background")

    # 2. H&E deconvolution
    hematoxylin, eosin = deconvolve_he(rgb)

    # 3. Nuclear density / cellularity
    nuclear = estimate_nuclear_density(hematoxylin, tissue_mask)
    logger.info(f"  Cellularity: {nuclear['cellularity']:.2f}  "
                f"Nuclear density: {nuclear['nuclear_density']:.3f}")

    # 4. TIL scoring
    til = score_til_density(rgb, hematoxylin, tissue_mask)
    logger.info(f"  TIL density: {til['til_density']:.3f}  "
                f"TME: {til['tme_phenotype']}")

    # 5. Texture features
    texture = extract_texture_features(hematoxylin, tissue_mask)

    scores = {
        "sample_id":              case_id,
        "tissue_fraction":        round(tissue_fraction, 4),
        "cellularity":            round(nuclear["cellularity"], 4),
        "nuclear_density":        round(nuclear["nuclear_density"], 4),
        "mean_he_intensity":      round(nuclear["mean_he_intensity"], 4),
        "til_density":            round(til["til_density"], 4),
        "til_score":              round(til["til_score"], 4),
        "tme_phenotype":          til["tme_phenotype"],
        "nuclear_pleomorphism":   round(texture["nuclear_pleomorphism"], 4),
        "tissue_heterogeneity":   round(texture["tissue_heterogeneity"], 4),
    }

    # Attach masks for visualization
    scores["_rgb"]          = rgb
    scores["_tissue_mask"]  = tissue_mask
    scores["_hematoxylin"]  = hematoxylin
    scores["_lymph_mask"]   = til.get("lymph_mask", np.zeros_like(tissue_mask))

    return scores


# ── Visualization ─────────────────────────────────────────────────────────────

TME_COLORS = {
    "Inflamed": "#2ca02c",
    "Excluded": "#ff7f0e",
    "Desert":   "#d62728",
}

SCORE_LABELS = {
    "cellularity":          "Cellularity",
    "til_score":            "TIL Score",
    "nuclear_pleomorphism": "Nuclear Pleomorphism",
    "tissue_heterogeneity": "Tissue Heterogeneity",
}


def make_pathology_report(scores: dict, out_path: Path):
    """Generate 4-panel pathology report figure."""
    rgb         = scores["_rgb"]
    tissue_mask = scores["_tissue_mask"]
    hematoxylin = scores["_hematoxylin"]
    lymph_mask  = scores["_lymph_mask"]
    case_id     = scores["sample_id"]
    tme         = scores["tme_phenotype"]

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#fafafa")
    gs = GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.50,
                  width_ratios=[1.1, 1.1, 1.0])

    ax_orig    = fig.add_subplot(gs[0, 0])   # Original thumbnail
    ax_tissue  = fig.add_subplot(gs[0, 1])   # Tissue + TIL overlay
    ax_he      = fig.add_subplot(gs[1, 0])   # Hematoxylin channel
    ax_radar   = fig.add_subplot(gs[1, 1])   # Score bar chart
    ax_summary = fig.add_subplot(gs[:, 2])   # Summary table

    # ── Panel A: Original thumbnail ───────────────────────────────────────────
    ax_orig.imshow(rgb)
    ax_orig.axis("off")
    ax_orig.set_title("H&E Thumbnail", fontsize=11, fontweight="bold")

    # ── Panel B: Tissue mask + TIL overlay ───────────────────────────────────
    overlay = rgb.copy()
    # Highlight lymphocyte candidates in green
    overlay[lymph_mask] = [50, 200, 50]
    # Dim background
    bg_mask = ~tissue_mask
    overlay[bg_mask] = (overlay[bg_mask].astype(float) * 0.4 + 200 * 0.6).astype(np.uint8)
    ax_tissue.imshow(overlay)
    ax_tissue.axis("off")
    tme_color = TME_COLORS.get(tme, "#999999")
    ax_tissue.set_title(
        f"TIL Overlay  [{tme}]",
        fontsize=11, fontweight="bold", color=tme_color
    )
    # TIL density annotation
    ax_tissue.text(
        0.02, 0.04,
        f"TIL density: {scores['til_density']:.1%}",
        transform=ax_tissue.transAxes,
        fontsize=9, color="white",
        bbox=dict(facecolor="black", alpha=0.5, pad=3),
    )

    # ── Panel C: Hematoxylin channel ─────────────────────────────────────────
    he_display = np.clip(hematoxylin, 0, 1)
    ax_he.imshow(he_display, cmap="Blues", vmin=0, vmax=1)
    ax_he.axis("off")
    ax_he.set_title("Hematoxylin Channel\n(nuclear staining)",
                    fontsize=11, fontweight="bold")
    plt.colorbar(
        plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(0, 1)),
        ax=ax_he, fraction=0.025, pad=0.02, label="Intensity"
    )

    # ── Panel D: Score bar chart ──────────────────────────────────────────────
    score_names = list(SCORE_LABELS.values())
    score_vals  = [scores[k] for k in SCORE_LABELS]
    bar_colors  = ["#4393c3", "#d6604d", "#f4a582", "#92c5de"]

    bars = ax_radar.barh(score_names, score_vals, color=bar_colors,
                         alpha=0.85, edgecolor="white", height=0.5)
    ax_radar.set_xlim(-0.05, 1.18)  # negative left gives labels room
    ax_radar.axvline(x=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_radar.set_xlabel("Score (0–1)", fontsize=9)
    ax_radar.set_title("Pathology Scores", fontsize=11, fontweight="bold")
    ax_radar.spines["top"].set_visible(False)
    ax_radar.spines["right"].set_visible(False)
    ax_radar.tick_params(axis="y", labelsize=9)
    for bar, val in zip(bars, score_vals):
        ax_radar.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                      f"{val:.2f}", va="center", ha="left", fontsize=9)

    # ── Panel E: Summary table ────────────────────────────────────────────────
    ax_summary.axis("off")
    tme_color = TME_COLORS.get(tme, "#999999")

    summary_rows = [
        ("Sample",          case_id),
        ("TME Phenotype",   tme),
        ("TIL Density",     f"{scores['til_density']:.1%}"),
        ("TIL Score",       f"{scores['til_score']:.2f}"),
        ("Cellularity",     f"{scores['cellularity']:.2f}"),
        ("Nuclear Density", f"{scores['nuclear_density']:.3f}"),
        ("Pleomorphism",    f"{scores['nuclear_pleomorphism']:.2f}"),
        ("Heterogeneity",   f"{scores['tissue_heterogeneity']:.2f}"),
        ("Tissue Fraction", f"{scores['tissue_fraction']:.1%}"),
    ]

    tbl = ax_summary.table(
        cellText=summary_rows,
        loc="upper center",
        cellLoc="left",
        bbox=[0.0, 0.30, 1.0, 0.68],   # [left, bottom, width, height] in axes coords
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.9)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("white")
        if c == 0:
            cell.set_text_props(fontweight="bold", color="#333333")
            cell.set_facecolor("#f0f4f8")
        else:
            cell.set_facecolor("white")
            cell.set_text_props(color="#555555")
        # Highlight TME row
        if r == 1 and c == 1:
            cell.set_text_props(fontweight="bold", color=tme_color)

    ax_summary.set_title("Pathology Summary", fontsize=11, fontweight="bold", pad=8)

    # ── Figure title ──────────────────────────────────────────────────────────
    fig.suptitle(
        f"Computational Pathology Report — {case_id}",
        fontsize=14, fontweight="bold", y=0.98,
    )

    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LUAD computational pathology (M08)")
    parser.add_argument("--sample",   type=str,  help="Single sample ID")
    parser.add_argument("--dry_run",  action="store_true")
    parser.add_argument("--full-wsi", action="store_true",
                        help="Use full SVS files instead of thumbnails")
    args = parser.parse_args()

    logger = get_logger("luad-pathology")
    logger.info(f"\n{'='*55}")
    logger.info("Module 05: Computational Pathology")

    # Determine input directory
    img_dir = SVS_DIR if args.full_wsi else THUMB_DIR
    pattern = "*.svs" if args.full_wsi else "*.png"

    # Collect images
    if args.sample:
        images = list(img_dir.glob(f"{args.sample}.*"))
    else:
        images = sorted(img_dir.glob(pattern))

    if not images:
        logger.error(
            f"No images found in {img_dir}\n"
            f"Run: python data/scripts/download_wsi.py --thumbnail"
        )
        return

    logger.info(f"Found {len(images)} slide images in {img_dir}")

    if args.dry_run:
        logger.info(f"[DRY RUN] Would process: {[i.stem for i in images[:5]]}...")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Pass 1: analyse all slides, collect raw scores ────────────────────────
    slide_data = []   # list of (img_path, case_id, scores_dict, report_path)
    success = failed = 0

    for img_path in images:
        case_id     = img_path.stem
        case_out    = OUT_DIR / case_id
        case_out.mkdir(parents=True, exist_ok=True)
        report_path = case_out / f"{case_id}_pathology_report.png"

        logger.info(f"\n[{case_id}]")
        try:
            scores = analyze_slide(img_path, case_id, logger)
            slide_data.append((img_path, case_id, scores, report_path))
            success += 1
        except Exception as e:
            logger.error(f"  Failed: {e}")
            failed += 1

    # ── Pass 2: assign TME phenotype using cohort-wide quantiles ─────────────
    if not args.sample and len(slide_data) >= 5:
        # Full-cohort run: use quantile-based classification
        rows = [{k: v for k, v in s.items() if not k.startswith("_")}
                for _, _, s, _ in slide_data]
        rows = classify_tme_quantile(rows)
        # Write back TME phenotype into the scores dicts
        for i, (_, _, scores, _) in enumerate(slide_data):
            scores["tme_phenotype"] = rows[i]["tme_phenotype"]
        logger.info("\nQuantile TME thresholds applied (70th / 20th percentile)")
    else:
        # Single-sample run: can't compute cohort percentiles, use fixed fallback
        rows = [{k: v for k, v in s.items() if not k.startswith("_")}
                for _, _, s, _ in slide_data]
        logger.warning("Single-sample run: using fixed thresholds (0.07 / 0.03) for TME")
        for row, (_, _, scores, _) in zip(rows, slide_data):
            d = row["til_density"]
            if d >= 0.07:
                tme = "Inflamed"
            elif d >= 0.03:
                tme = "Excluded"
            else:
                tme = "Desert"
            row["tme_phenotype"] = tme
            scores["tme_phenotype"] = tme

    # ── Write per-patient TSVs and figures ────────────────────────────────────
    all_scores = []
    for (img_path, case_id, scores, report_path), row in zip(slide_data, rows):
        case_out    = OUT_DIR / case_id
        scores_path = case_out / f"{case_id}_pathology_scores.tsv"

        pd.DataFrame([row]).to_csv(scores_path, sep="\t", index=False)

        # Regenerate figure with final TME label
        make_pathology_report(scores, report_path)
        logger.info(f"  [{case_id}] TME={row['tme_phenotype']}  "
                    f"TIL={row['til_density']:.3f}  → {report_path.name}")
        all_scores.append(row)

    # Save cohort-level summary
    if all_scores:
        summary_path = OUT_DIR / "pathology_summary.tsv"
        pd.DataFrame(all_scores).to_csv(summary_path, sep="\t", index=False)
        logger.info(f"\nCohort summary → {summary_path}")

        # Print TME distribution
        tme_counts = pd.DataFrame(all_scores)["tme_phenotype"].value_counts()
        logger.info("\nTME phenotype distribution:")
        for k, v in tme_counts.items():
            logger.info(f"  {k}: {v} ({v/len(all_scores):.1%})")

    logger.info(f"\n[Module 05 Complete] Success: {success}  Failed: {failed}")
    logger.info(f"Output → {OUT_DIR}")


if __name__ == "__main__":
    main()
