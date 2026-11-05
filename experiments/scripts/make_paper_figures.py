"""Generate the manuscript figures (1-10) from the saved experiment artifacts.

Figures are written to ``figures/figure N.png`` at 300 DPI:

* Figure 1  - technical architecture & experimental-workflow block diagram
              (drawn; two-tower layout with nested dashed "× n" groups).
* Figure 2  - Human-in-the-Loop interface schematic (drawn; replace with real
              screenshots if available).
* Figure 3  - training/validation loss curves (from ``runs/seg_main/results.csv``).
* Figure 4  - normalised confusion matrix, internal validation (copied from run).
* Figure 5  - qualitative GT vs prediction, internal validation (composite).
* Figure 6  - normalised confusion matrices, three external sources (composite).
* Figure 7  - mask PR and F1 curves, three external sources (composite).
* Figure 8  - per-class external mask AP50 (from ``per_class_all.csv``).
* Figure 9  - nutrition-label extraction pipeline diagram (drawn).
* Figure 10 - qualitative label-extraction example (input + ROI + fields).

Usage:
    python experiments/scripts/make_paper_figures.py
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[2]
SEG = ROOT / "experiments" / "results" / "segmentation"
RUNS = ROOT / "experiments" / "runs" / "seg_main"
OCR_IMG = ROOT / "experiments" / "ocr_data" / "makanan_ringan"
OCR_GRID = ROOT / "experiments" / "results" / "ocr_full_grid" / "predictions.jsonl"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

GREEN = "#2e7d32"
BLUE = "#1565c0"
ORANGE = "#ef6c00"
PURPLE = "#6a1b9a"
GREY = "#455a64"
LIGHT = "#e8f5e9"
LIGHTB = "#e3f2fd"

CLASS_COLORS = {
    "Staple Food": "#f4a261",
    "Side Dishes": "#e76f51",
    "Vegetables": "#2a9d8f",
    "Fruits": "#e9c46a",
    "Others": "#8d99ae",
}


def save(fig, number: int, dpi: int = 300) -> None:
    path = OUT / f"figure {number}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


def _box(ax, x, y, w, h, text, fc, ec, fs=9.5, tc="black"):
    ax.add_patch(
        FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.008,rounding_size=0.02",
                       linewidth=1.4, edgecolor=ec, facecolor=fc)
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, linespacing=1.35)


def _arrow(ax, p1, p2):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13,
                                 lw=1.4, color="#37474f",
                                 shrinkA=1, shrinkB=1))


def _dgroup(ax, x, y, w, h, color, title=None, fs=8.4):
    """Dashed enclosure, like the repeated-layer groups in a Transformer diagram."""
    ax.add_patch(
        FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.002,rounding_size=0.012",
                       linewidth=1.4, edgecolor=color, facecolor="none",
                       linestyle=(0, (6, 3)), zorder=0)
    )
    if title:
        ax.text(x + 0.010, y + h - 0.007, title, ha="left", va="top",
                fontsize=fs, color=color, fontweight="bold", zorder=4,
                bbox=dict(fc="white", ec="none", pad=1.4))


# --------------------------------------------------------------------------- #
# Figure 1 - technical architecture & experimental-workflow diagram
# --------------------------------------------------------------------------- #
def figure1():
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Healthier system architecture and experimental workflow",
                 fontsize=15, fontweight="bold", pad=14)

    # ---- geometry ---------------------------------------------------------
    LX, LW = 0.020, 0.420
    RX, RW = 0.560, 0.420
    cxL, cxR = LX + LW / 2, RX + RW / 2
    top_in, in_h = 0.980, 0.048
    grp_top, grp_bot = 0.912, 0.280
    bw = LW - 0.048
    bxL, bxR = LX + 0.024, RX + 0.024
    BH, BG = 0.036, 0.011
    cont_top = grp_top - 0.030

    # ---- inputs -----------------------------------------------------------
    _box(ax, LX, top_in - in_h, LW, in_h,
         "Plate photo — Isi Piringku  (640×640×3)", LIGHT, GREEN, fs=9)
    _box(ax, RX, top_in - in_h, RW, in_h,
         "Nutrition-label photo  (640×640×3)", LIGHTB, BLUE, fs=9)
    _arrow(ax, (cxL, top_in - in_h), (cxL, grp_top - 0.004))
    _arrow(ax, (cxR, top_in - in_h), (cxR, grp_top - 0.004))

    # ---- dashed tower enclosures -----------------------------------------
    _dgroup(ax, LX, grp_bot, LW, grp_top - grp_bot, GREEN,
            "Segmentation network — YOLO11s-seg")
    _dgroup(ax, RX, grp_bot, RW, grp_top - grp_bot, BLUE,
            "Label extraction — PaddleOCR PP-OCRv4")

    # ---- local drawing helpers -------------------------------------------
    def put(x, y, text, fc, ec, fs=7.4):
        ax.add_patch(FancyBboxPatch(
            (x, y - BH), bw, BH,
            boxstyle="round,pad=0.003,rounding_size=0.010",
            linewidth=1.15, edgecolor=ec, facecolor=fc, zorder=2))
        ax.text(x + bw / 2, y - BH / 2, text, ha="center", va="center",
                fontsize=fs, color="#102027", zorder=3, linespacing=1.2)

    def section(x, ytop, label, color, blocks, fs=7.4):
        y = ytop - 0.019
        for text, fc, ec in blocks:
            put(x, y, text, fc, ec, fs)
            y -= BH + BG
        y += BG
        ybot = y - 0.005
        ax.add_patch(FancyBboxPatch(
            (x - 0.015, ybot), bw + 0.030, ytop - ybot,
            boxstyle="round,pad=0.002,rounding_size=0.010",
            linewidth=1.0, edgecolor=color, facecolor="none",
            linestyle=(0, (2, 2)), zorder=1))
        ax.text(x - 0.010, ytop - 0.004, label, ha="left", va="top",
                fontsize=7.6, color=color, fontweight="bold", zorder=4,
                bbox=dict(fc="white", ec="none", pad=0.9))
        return ybot

    # ---- left tower: segmentation network --------------------------------
    y = section(bxL, cont_top, "Backbone", "#7cb342", [
        ("Stem: Conv 3×3 / 2", "#f1f8e9", "#7cb342"),
        ("P2: Conv / 2 + C3k2", "#f1f8e9", "#7cb342"),
        ("P3: Conv / 2 + C3k2  →  P3", "#f1f8e9", "#7cb342"),
        ("P4: Conv / 2 + C3k2  →  P4", "#f1f8e9", "#7cb342"),
        ("P5: C3k2 + SPPF + C2PSA  →  P5", "#f1f8e9", "#7cb342"),
    ])
    # "× 3" repeated-stage marker across the P3-P5 stages
    i0, i1 = 2, 4
    mt = cont_top - 0.019 - i0 * (BH + BG) + 0.004
    mb = cont_top - 0.019 - i1 * (BH + BG) - BH - 0.004
    ax.add_patch(FancyBboxPatch(
        (bxL - 0.006, mb), bw + 0.012, mt - mb,
        boxstyle="round,pad=0.001,rounding_size=0.008",
        linewidth=0.9, edgecolor="#8bc34a", facecolor="none",
        linestyle=(0, (1, 1.8)), zorder=1.5))
    ax.text(bxL + bw + 0.002, mt - 0.003, "× 3", ha="right", va="top",
            fontsize=6.8, color="#558b2f", fontweight="bold", zorder=4)

    y -= 0.022
    y = section(bxL, y, "Neck — PAN-FPN", "#43a047", [
        ("Top-down: Upsample + Concat + C3k2", "#e8f5e9", "#43a047"),
        ("Bottom-up: Conv + Concat + C3k2", "#e8f5e9", "#43a047"),
    ])
    y -= 0.022
    y = section(bxL, y, "Segment head", "#00897b", [
        ("Detect: box (DFL) + class", "#e0f2f1", "#00897b"),
        ("Mask: Proto + coefficients", "#e0f2f1", "#00897b"),
    ])
    y -= 0.022
    left_bot = section(bxL, y, "Post-processing", "#2e7d32", [
        ("Proportion per group (Eq. 1)", "#c8e6c9", "#2e7d32"),
    ])

    # ---- right tower: label extraction -----------------------------------
    y = section(bxR, cont_top, "Preprocessing", "#42a5f5", [
        ("EXIF auto-orient", "#e3f2fd", "#64b5f6"),
        ("Deskew — Hough transform", "#e3f2fd", "#64b5f6"),
        ("Deglare — Telea inpainting", "#e3f2fd", "#64b5f6"),
        ("CLAHE contrast + ROI crop", "#e3f2fd", "#64b5f6"),
    ])
    y -= 0.022
    y = section(bxR, y, "Recognition — PP-OCRv4", "#039be5", [
        ("Text detection — DB (DBNet)", "#e1f5fe", "#039be5"),
        ("Text recognition — CRNN + CTC", "#e1f5fe", "#039be5"),
    ])
    y -= 0.022
    y = section(bxR, y, "Layout reconstruction", "#00acc1", [
        ("Semantic filter (drop ads/ingredients)", "#e0f7fa", "#00acc1"),
        ("Fuzzy field mapping (sim ≥ 0.75)", "#e0f7fa", "#00acc1"),
        ("Grid parser — rows + value columns", "#e0f7fa", "#00acc1"),
    ])
    y -= 0.022
    right_bot = section(bxR, y, "Output", "#1565c0", [
        ("Per-serving nutrient fields", "#b3e5fc", "#1565c0"),
    ])

    # ---- decision layer ---------------------------------------------------
    band_y, band_h = 0.078, 0.110
    band_top = band_y + band_h
    hx, hw = 0.285, 0.245
    rx, rw = 0.595, 0.200
    fx, fw = 0.850, 0.130
    _box(ax, hx, band_y, hw, band_h,
         "Human-in-the-Loop\nverify masks & values", "#f3e5f5", PURPLE, fs=8.2)
    _box(ax, rx, band_y, rw, band_h,
         "Clinical Rules Engine\nBMR/TDEE + limits", "#e8eaf6", "#3949ab", fs=8.2)
    _box(ax, fx, band_y, fw, band_h,
         "Feedback\nSAFE / WARNING /\nDANGER", "#ffebee", "#c62828", fs=8.2)

    _arrow(ax, (cxL, left_bot), (hx + 0.060, band_top))
    _arrow(ax, (cxR, right_bot), (hx + 0.185, band_top))
    mid = band_y + band_h / 2
    _arrow(ax, (hx + hw, mid), (rx, mid))
    _arrow(ax, (rx + rw, mid), (fx, mid))

    # ---- experiment annotations ------------------------------------------
    ax.text(0.5, 0.048,
            "Segmentation: 11,684 train / 2,017 internal-val images · YOLO11s-seg · "
            "100 epochs (best 48) · AdamW lr0 = 1×10⁻³ · batch 16 · 640×640 · "
            "internal mask mAP50 71.66%, mAP50-95 56.91%.",
            ha="center", va="center", fontsize=6.9, color=GREY)
    ax.text(0.5, 0.020,
            "External generalisation (5,013 images): mask mAP50 69.15% (UEC) · "
            "63.66% (FoodSeg103) · 17.06% (FruitSeg30).   "
            "Label pilot (n = 6): field accuracy 21.74% → 47.83%.   "
            "Rules verified 36/36 limits, 30/30 decisions.",
            ha="center", va="center", fontsize=6.9, color=GREY)
    save(fig, 1, dpi=200)


# --------------------------------------------------------------------------- #
# Figure 2 - HITL interface schematic
# --------------------------------------------------------------------------- #
def figure2():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for a in axes:
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
        a.axis("off")

    ax = axes[0]
    ax.set_title("Segmentation editor", fontsize=11, fontweight="bold")
    ax.add_patch(Rectangle((0.06, 0.16), 0.88, 0.68, fill=True,
                           facecolor="#fafafa", edgecolor="#bdbdbd"))
    rng = np.random.default_rng(42)
    centers = [(0.30, 0.58), (0.62, 0.44), (0.48, 0.72)]
    names = ["Staple Food", "Vegetables", "Side Dishes"]
    for (cx, cy), name in zip(centers, names):
        pts = np.array([
            [cx + 0.11 * np.cos(t), cy + 0.09 * np.sin(t)]
            for t in np.linspace(0, 2 * np.pi, 9)[:-1]
        ]) + rng.normal(0, 0.012, (8, 2))
        ax.add_patch(Polygon(pts, closed=True, facecolor=CLASS_COLORS[name],
                             edgecolor="white", alpha=0.75, lw=1.5))
        ax.text(cx, cy, name.split()[0], ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")
    ax.text(0.5, 0.075, "Edit predicted polygons, then confirm",
            ha="center", fontsize=8.5, color=GREY)

    ax = axes[1]
    ax.set_title("Nutrient value verification form", fontsize=11, fontweight="bold")
    fields = [("Energy (kkal)", "110"), ("Total fat (g)", "4"),
              ("Saturated fat (g)", "3"), ("Sodium (mg)", "40"),
              ("Sugar (g)", "10")]
    y0 = 0.80
    for k, (name, val) in enumerate(fields):
        yy = y0 - k * 0.135
        ax.add_patch(Rectangle((0.06, yy), 0.55, 0.10, facecolor="#f5f5f5",
                               edgecolor="#cfd8dc"))
        ax.text(0.09, yy + 0.05, name, va="center", fontsize=8.5)
        ax.add_patch(Rectangle((0.63, yy), 0.31, 0.10, facecolor="#fff8e1",
                               edgecolor="#f9a825"))
        ax.text(0.785, yy + 0.05, val, va="center", ha="center", fontsize=9)
    ax.text(0.5, 0.045, "Correct OCR values, then confirm",
            ha="center", fontsize=8.5, color=GREY)

    fig.suptitle("Human-in-the-Loop interfaces (schematic)",
                 fontsize=12, fontweight="bold", y=1.02)
    save(fig, 2)


# --------------------------------------------------------------------------- #
# Figure 3 - loss curves
# --------------------------------------------------------------------------- #
def figure3():
    rows = list(csv.DictReader((RUNS / "results.csv").open(encoding="utf-8")))
    epochs = [int(r["epoch"]) for r in rows]
    losses = [("box_loss", "Box loss"), ("seg_loss", "Segmentation loss"),
              ("cls_loss", "Classification loss"), ("dfl_loss", "DFL loss")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (key, title) in zip(axes.ravel(), losses):
        ax.plot(epochs, [float(r[f"train/{key}"]) for r in rows],
                color=BLUE, lw=1.8, label="train")
        ax.plot(epochs, [float(r[f"val/{key}"]) for r in rows],
                color=ORANGE, lw=1.8, label="val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)
    axes[0, 0].legend()
    fig.suptitle("Training and validation loss curves (YOLO11s-seg)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, 3)


# --------------------------------------------------------------------------- #
# Figure 4 - internal confusion matrix (copy)
# --------------------------------------------------------------------------- #
def figure4():
    src = RUNS / "confusion_matrix_normalized.png"
    shutil.copyfile(src, OUT / "figure 4.png")
    print("wrote", OUT / "figure 4.png", "(copy of", src.name + ")")


# --------------------------------------------------------------------------- #
# Figure 5 - qualitative GT vs prediction
# --------------------------------------------------------------------------- #
def figure5():
    gt = imread(RUNS / "val_batch0_labels.jpg")
    pr = imread(RUNS / "val_batch0_pred.jpg")
    fig, axes = plt.subplots(2, 1, figsize=(9, 9))
    for ax, img, title in ((axes[0], gt, "Ground truth"),
                           (axes[1], pr, "Prediction")):
        ax.imshow(img)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")
    fig.suptitle("Qualitative comparison on the internal validation split",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, 5, dpi=180)


# --------------------------------------------------------------------------- #
# Figure 6 - external confusion matrices
# --------------------------------------------------------------------------- #
def figure6():
    sources = [("FoodSeg103 (validation)", SEG / "foodseg_val_instance"),
               ("UECFoodPIXCOMPLETE (test)", SEG / "uec_test_instance"),
               ("FruitSeg30", SEG / "fruitseg_instance")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, folder) in zip(axes, sources):
        ax.imshow(imread(folder / "confusion_matrix_normalized.png"))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle("Normalised confusion matrices on the three external sources",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, 6, dpi=180)


# --------------------------------------------------------------------------- #
# Figure 7 - mask PR and F1 curves
# --------------------------------------------------------------------------- #
def figure7():
    sources = [("FoodSeg103 (validation)", SEG / "foodseg_val_instance"),
               ("UECFoodPIXCOMPLETE (test)", SEG / "uec_test_instance"),
               ("FruitSeg30", SEG / "fruitseg_instance")]
    fig, axes = plt.subplots(3, 2, figsize=(11, 13))
    for row, (title, folder) in enumerate(sources):
        for col, fname in enumerate(("MaskPR_curve.png", "MaskF1_curve.png")):
            ax = axes[row, col]
            ax.imshow(imread(folder / fname))
            ax.axis("off")
            ax.set_title(("Mask PR curve" if col == 0 else "Mask F1 curve"),
                         fontsize=9)
        axes[row, 0].text(-0.05, 0.5, title, rotation=90, va="center",
                          ha="right", fontsize=10, transform=axes[row, 0].transAxes)
    fig.suptitle("Mask precision-recall and F1 curves per external source",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, 7, dpi=150)


# --------------------------------------------------------------------------- #
# Figure 8 - per-class external mask AP50
# --------------------------------------------------------------------------- #
def figure8():
    rows = list(csv.DictReader((SEG / "per_class_all.csv").open(encoding="utf-8")))
    classes = ["Staple Food", "Side Dishes", "Vegetables", "Fruits", "Others"]
    sources = {"uec_test": "UEC", "foodseg_val": "FoodSeg103", "fruitseg": "FruitSeg30"}
    data = {s: {c: np.nan for c in classes} for s in sources}
    for r in rows:
        if r["dataset"] in data and r["mask_ap50"]:
            data[r["dataset"]][r["class"]] = 100 * float(r["mask_ap50"])

    x = np.arange(len(classes))
    width = 0.26
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for i, (key, label) in enumerate(sources.items()):
        vals = [data[key][c] for c in classes]
        ax.bar(x + (i - 1) * width, vals, width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(" ", "\n") for c in classes])
    ax.set_ylabel("Mask AP50 (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Test source")
    ax.set_title("Per-class external mask AP50 by source and class",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, 8)


# --------------------------------------------------------------------------- #
# Figure 9 - OCR pipeline diagram
# --------------------------------------------------------------------------- #
def figure9():
    stages = [
        ("Label\nphoto", LIGHTB, BLUE),
        ("EXIF\noorient", LIGHT, GREEN),
        ("Deskew", LIGHT, GREEN),
        ("Deglare", LIGHT, GREEN),
        ("CLAHE", LIGHT, GREEN),
        ("ROI\ncrop", LIGHT, GREEN),
        ("PP-OCRv4\ndet + rec", "#fff3e0", ORANGE),
        ("Semantic\nfilter", "#fff3e0", ORANGE),
        ("Fuzzy field\nmapping", "#fff3e0", ORANGE),
        ("Grid layout\nparser", "#fff3e0", ORANGE),
        ("Energy/fat\nfallback", "#fff3e0", ORANGE),
        ("Clinical\nvalidation", "#e8eaf6", "#3949ab"),
        ("Nutrient\nfields", "#fce4ec", "#c62828"),
    ]
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Nutrition-label extraction pipeline",
                 fontsize=13, fontweight="bold", pad=10)
    per_row = 7
    w, h = 0.118, 0.17
    gapx = 0.021
    for i, (text, fc, ec) in enumerate(stages):
        row = i // per_row
        col = i % per_row
        xx = 0.01 + col * (w + gapx)
        yy = 0.62 - row * 0.34
        _box(ax, xx, yy, w, h, text, fc, ec, fs=8.4)
        if col > 0:
            _arrow(ax, (xx - gapx, yy + h / 2), (xx, yy + h / 2))
    # wrap arrow from end of row 1, under row 2, up into the first box
    _arrow(ax, (0.99, 0.70), (0.99, 0.18))
    _arrow(ax, (0.99, 0.18), (0.069, 0.18))
    _arrow(ax, (0.069, 0.18), (0.069, 0.28))
    ax.text(0.5, 0.055,
            "Preprocessing keeps the recognition stage robust to phone-camera labels; "
            "the\nlayout parser and mapping stages recover the per-serving values.",
            ha="center", va="center", fontsize=9, style="italic", color=GREY,
            linespacing=1.5)
    save(fig, 9)


# --------------------------------------------------------------------------- #
# Figure 10 - qualitative label extraction
# --------------------------------------------------------------------------- #
STORY_IMAGE = "20241108_132045"          # grid-parser example: 6/7 fields exact
FIELD_ORDER = ["kalori", "lemak_total", "lemak_jenuh", "protein",
               "karbohidrat", "gula", "natrium"]


def _grid_fields(stem: str) -> dict:
    if not OCR_GRID.is_file():
        return {}
    for line in OCR_GRID.read_text(encoding="utf-8").splitlines():
        if stem not in line:
            continue
        obj = json.loads(line)
        out = {}
        for key, val in (obj.get("data") or {}).items():
            if isinstance(val, dict) and val.get("per_sajian") is not None:
                out[key] = val["per_sajian"]
        return out
    return {}


def figure10():
    stem = STORY_IMAGE
    img = imread(OCR_IMG / f"{stem}.jpg")
    h, w = img.shape[:2]
    roi = img[int(0.13 * h):int(0.64 * h), int(0.28 * w):int(0.64 * w)]
    extracted = _grid_fields(stem)
    gt = json.loads((OCR_IMG / f"{stem}_gt.json").read_text(encoding="utf-8"))["nutrisi"]

    fig = plt.figure(figsize=(13.5, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.05], wspace=0.06)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax0.imshow(img)
    ax0.set_title("(a) Input label photo", fontsize=10)
    ax0.axis("off")
    ax1.imshow(roi)
    ax1.set_title("(b) Nutrition-table region", fontsize=10)
    ax1.axis("off")
    ax2.axis("off")
    ax2.set_title("(c) Extracted vs ground truth", fontsize=10)
    rows = []
    for field in FIELD_ORDER:
        ext = extracted.get(field)
        ref = gt.get(field)
        rows.append([
            field.replace("_", " "),
            "-" if ext is None else f"{ext:g}",
            "-" if ref is None else f"{ref:g}",
        ])
    table = ax2.table(cellText=rows, colLabels=["Field", "Extracted", "Truth"],
                      loc="center", cellLoc="left", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.55)
    ax2.text(0.5, 0.14, "values per serving", ha="center", fontsize=8,
             style="italic", color=GREY, transform=ax2.transAxes)
    fig.suptitle("Qualitative example of label extraction",
                 fontsize=12, fontweight="bold")
    save(fig, 10, dpi=160)


def main():
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    figure6()
    figure7()
    figure8()
    figure9()
    figure10()
    print("\nAll figures written to", OUT)


if __name__ == "__main__":
    main()
