#!/usr/bin/env python
# gen_paper_figures.py
# ------------------------------------------------------------------
# Generate the two figures for the IEEE conference paper:
#   Fig. 1  model architecture diagram (matplotlib drawing)
#   Fig. 2  dataset + saliency figure in the style of the base
#           paper (van der Graaf et al.): full sagittal slice with
#           disc crops marked, Pfirrmann grade examples I-V, and
#           the four CAM overlays on a model input crop.
#
# Output: figures/fig1_architecture.png, figures/fig2_saliency.png
# ------------------------------------------------------------------
from __future__ import annotations

import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

sys.stdout.reconfigure(encoding="utf-8")

import SimpleITK as sitk

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.linewidth": 0.6,
})

# ------------------------------------------------------------------
# Fig. 1  —  architecture diagram
# ------------------------------------------------------------------
def fig1_architecture():
    fig, ax = plt.subplots(figsize=(4.6, 5.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12.4)
    ax.axis("off")

    def box(x, y, w, h, title, sub=None, fill="#ffffff", lw=1.4, fs=9.5, subfs=7.5):
        b = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.06,rounding_size=0.15",
                           facecolor=fill, edgecolor="black", linewidth=lw)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2 + (0.16 if sub else 0), title,
                ha="center", va="center", fontsize=fs, fontweight="bold")
        if sub:
            ax.text(x + w / 2, y + h / 2 - 0.34, sub,
                    ha="center", va="center", fontsize=subfs, style="italic")
        return (x, y, w, h)

    def arrow(x1, y1, x2, y2, dashed=False, color="black"):
        a = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle="-|>", mutation_scale=12,
                            linewidth=1.2, color=color,
                            linestyle=(0, (4, 2)) if dashed else "solid")
        ax.add_patch(a)

    cx = 5.0  # centre of the main column

    # Input
    box(cx - 1.8, 11.45, 3.6, 0.75, "T1-sagittal disc crop",
        "224 × 224", fill="#f2f2f2")
    arrow(cx, 11.45, cx, 10.85)

    # Encoder
    box(cx - 2.3, 9.4, 4.6, 1.45, "EfficientViT-b1 Encoder",
        "multi-scale linear attention · 7.58 M params")
    arrow(cx, 9.4, cx, 8.75)

    # CBAM
    box(cx - 1.7, 7.5, 3.4, 1.25, "CBAM",
        "channel + spatial attention")
    arrow(cx, 7.5, cx, 6.85)

    # Pool
    box(cx - 1.5, 5.95, 3.0, 0.9, "Global Average Pool", fill="#f2f2f2")
    arrow(cx, 5.95, cx, 5.3)

    # Split
    ax.plot([cx, cx], [5.3, 5.05], color="black", linewidth=1.2)
    ax.plot([cx, cx - 2.0], [5.05, 5.05], color="black", linewidth=1.2)
    ax.plot([cx, cx + 2.0], [5.05, 5.05], color="black", linewidth=1.2)
    arrow(cx - 2.0, 5.05, cx - 2.0, 4.4)
    arrow(cx + 2.0, 5.05, cx + 2.0, 4.4)

    # Heads
    box(cx - 3.9, 2.9, 3.8, 1.5, "Pathology Head",
        "7 × FC → BCE + sigmoid", fill="#e8e8e8")
    box(cx + 0.1, 2.9, 3.8, 1.5, "Pfirrmann Head",
        "FC → 5-way softmax", fill="#e8e8e8")

    # Outputs
    ax.text(cx - 2.0, 2.35, "P(modic), P(herniation), ...", ha="center",
            fontsize=8)
    ax.text(cx + 2.0, 2.35, "grade I – V", ha="center", fontsize=8)

    # ---- side annotations (uncertainty / calibration / explainability) ----
    # right side: uncertainty + calibration (post-hoc, applied at inference)
    box(7.05, 1.15, 2.9, 1.35, "Uncertainty",
        "MC-dropout · TTA · ensemble", fill="#ffffff")
    box(7.05, 2.55, 2.9, 1.05, "Calibration",
        "temperature scaling", fill="#ffffff")

    # dashed links from heads to side boxes
    arrow(cx - 1.0, 2.9, 7.3, 2.5, dashed=True, color="#444444")
    arrow(cx + 0.2, 2.9, 7.9, 2.5, dashed=True, color="#444444")
    arrow(7.0, 2.2, 6.9, 1.45, dashed=True, color="#444444")

    # CAM: from heads back to encoder
    box(0.1, 3.6, 2.6, 1.15, "Grad-CAM etc.",
        "gradients → last stage", fill="#ffffff")
    arrow(1.4, 4.75, 3.3, 8.0, dashed=True, color="#444444")
    arrow(2.7, 3.6, 4.3, 3.0, dashed=True, color="#444444")

    fig.tight_layout()
    fig.savefig(OUT / "fig1_architecture.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("Fig. 1 saved ->", OUT / "fig1_architecture.png")


# ------------------------------------------------------------------
# Fig. 2  —  data + saliency figure
# ------------------------------------------------------------------
def disc_centroid(mask_path, lab):
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))
    sel = arr == lab
    if not sel.any():
        return None
    zs, ys, xs = np.nonzero(sel)
    return (int(round(xs.mean())), int(round(ys.mean())), int(round(np.median(zs))))


def window_display(arr2d):
    """Same windowing as the crop extractor, mapped to [0,1]."""
    c = np.clip(arr2d.astype(np.float32), -1000, 4000)
    return (c + 1000.0) / 5000.0


def fig2_saliency():
    series = "16_t1"
    img_path = ROOT / "extracted" / "images" / "images" / f"{series}.mha"
    mask_path = ROOT / "extracted" / "masks" / "masks" / f"{series}.mha"
    img = sitk.GetArrayFromImage(sitk.ReadImage(str(img_path)))
    mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))
    H, W = img.shape[1], img.shape[2]

    # Centroid per disc label (labels 1..5 for this series)
    discs = []
    for lab in range(1, 8):
        cen = disc_centroid(mask_path, lab)
        if cen is not None:
            discs.append((lab, cen))
    # Slice with the most discs
    from collections import Counter
    cz_counts = Counter(c for _, (_, _, c) in discs)
    cz = max(cz_counts, key=cz_counts.get)
    discs_on_slice = [d for d in discs if d[1][2] == cz]
    discs_on_slice.sort()

    sag = window_display(img[cz])

    # CAM heatmaps + input crop for p16_16_t1_d1
    sample = "p16_16_t1_d1"
    crop_up = window_display(img[cz])  # same window for consistent look
    input_crop = np.load(ROOT / "crops" / "crops_test" / f"{sample}.npy")
    methods = [("gradcam", "Grad-CAM"), ("gradcam++", "Grad-CAM++"),
               ("eigen_cam", "Eigen-CAM"), ("score_cam", "Score-CAM")]
    cam_base = ROOT / "results" / "cam"

    fig = plt.figure(figsize=(7.4, 4.6))

    # ---- (a) full sagittal slice with crop boxes ----
    axa = fig.add_axes([0.01, 0.50, 0.21, 0.44])
    axa.imshow(sag, cmap="gray", vmin=0, vmax=1)
    for lab, (cx_, cy_, _) in discs_on_slice:
        half = 112
        axa.add_patch(Rectangle((cx_ - half, cy_ - half), 224, 224,
                                fill=False, edgecolor="#00c8ff", linewidth=1.0))
        axa.text(cx_ + half + 4, cy_ - 4, f"d{lab}", color="#00c8ff",
                 fontsize=7, va="top")
    axa.set_title("(a)", loc="left", fontsize=9, pad=2)
    axa.axis("off")

    # ---- (b) Pfirrmann grade examples I-V ----
    axb = fig.add_axes([0.24, 0.50, 0.60, 0.44])
    axb.set_title("(b)", loc="left", fontsize=9, pad=2)
    axb.axis("off")
    import pandas as pd
    labels = pd.read_csv(ROOT / "crops" / "labels.csv")
    for g in range(1, 6):
        row = labels[(labels["Pfirrmann"] == g) & (labels["split"] == "test")].iloc[0]
        npy = np.load(ROOT / "crops" / "crops_test" / f"{row['sample_id']}.npy")
        axi = axb.inset_axes([(g - 1) * 0.205 + 0.01, 0.03, 0.22, 0.78])
        axi.imshow(npy, cmap="gray", vmin=0, vmax=1)
        axi.axis("off")
        axi.set_title(f"grade {['I','II','III','IV','V'][g-1]}", fontsize=8, pad=1)

    # ---- (c) CAM overlays ----
    axc = fig.add_axes([0.01, 0.02, 0.21, 0.44])
    axc.imshow(input_crop, cmap="gray", vmin=0, vmax=1)
    axc.set_title("(c)", loc="left", fontsize=9, pad=2)
    axc.axis("off")
    axc.set_xlabel("model input", fontsize=7)

    for i, (mname, mlabel) in enumerate(methods):
        axi = fig.add_axes([0.24 + i * 0.165, 0.02, 0.165, 0.44])
        heat = np.load(cam_base / mname / "Pfirrmann" / f"{sample}.npy")
        axi.imshow(input_crop, cmap="gray", vmin=0, vmax=1)
        axi.imshow(heat, cmap="jet", alpha=0.55, vmin=0, vmax=1)
        axi.axis("off")
        axi.set_xlabel(mlabel, fontsize=7)

    fig.savefig(OUT / "fig2_saliency.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("Fig. 2 saved ->", OUT / "fig2_saliency.png")


if __name__ == "__main__":
    fig1_architecture()
    fig2_saliency()