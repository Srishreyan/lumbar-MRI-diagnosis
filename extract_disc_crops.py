"""
extract_disc_crops.py
---------------------
Convert the 3D SPIDER volumes into per-disc 2D sagittal crops so we can train
a lightweight image classifier on a CPU.

What this does, in plain English:
  * SPIDER gives us one 3D MRI volume per patient per contrast, plus a 3D mask
    in which each intervertebral disc has its own label (1..7).
  * The grading table is keyed by (Patient, IVD label) where IVD label already
    IS the mask label (1..7). For every graded disc we:
        - pick a representative series for that patient (first on disk),
        - compute the disc centre (centroid of its mask voxels),
        - take the sagittal slice through that centre and cut a square patch.
    That patch is one "image sample".
  * Each sample keeps the pathology labels + Pfirrmann grade attached, so the
    multi-task heads can be trained later.

Outputs (all under DATASET_DIR / crops /):
  crops_<split>/   cropped .npy patches (grayscale float32, 224x224)
  labels.csv       one row per sample with every label we will train on
  crops_summary.txt  quick counts for sanity checking
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

DATASET_DIR = Path(__file__).resolve().parent
EXTRACT_DIR = DATASET_DIR / "extracted"
CROP_DIR = DATASET_DIR / "crops"
CROP_SIZE = 224          # square patch size in pixels

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def disc_centroid(mask_path: str, lab: int):
    """Return (x, y, z) voxel-space centroid of one disc label in a mask.

    z = the sagittal (slice) axis, so we use the *median* z of the disc's
    voxels to pick a slice that one or two stray voxels cannot yank sideways.
    """
    arr = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))   # (z, y, x)
    sel = arr == lab
    if not sel.any():
        return None
    zs, ys, xs = np.nonzero(sel)
    return (
        int(round(xs.mean())),
        int(round(ys.mean())),
        int(round(np.median(zs))),
    )


def load_series(new_file_name: str):
    """Return (image_array_zxy, mask_array_zxy) for a series stem."""
    img = sitk.ReadImage(str(EXTRACT_DIR / "images" / "images" / f"{new_file_name}.mha"))
    mask = sitk.ReadImage(str(EXTRACT_DIR / "masks" / "masks" / f"{new_file_name}.mha"))
    return sitk.GetArrayFromImage(img), sitk.GetArrayFromImage(mask)


def crop_around(arr2d: np.ndarray, cy: int, cx: int, half: int) -> np.ndarray:
    """Extract a (2*half)x(2*half) patch centred on (cy, cx), zero-padded."""
    H, W = arr2d.shape
    y0, y1 = cy - half, cy + half
    x0, x1 = cx - half, cx + half
    ys, ye = max(y0, 0), min(y1, H)
    xs, xe = max(x0, 0), min(x1, W)
    patch = np.zeros((2 * half, 2 * half), dtype=arr2d.dtype)
    patch[ys - y0:ye - y0, xs - x0:xe - x0] = arr2d[ys:ye, xs:xe]
    return patch

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    series = pd.read_csv(DATASET_DIR / "dataset_series.csv")
    levels = pd.read_csv(DATASET_DIR / "dataset_levels.csv")

    # A representative series per patient = first (sorted) series on disk.
    img_dir = EXTRACT_DIR / "images" / "images"
    mask_dir = EXTRACT_DIR / "masks" / "masks"
    pat_series: dict[int, str] = {}
    for _, s in series.iterrows():
        pat = int(s["Patient"]); name = s["new_file_name"]
        if pat in pat_series:
            continue
        if (img_dir / f"{name}.mha").exists() and (mask_dir / f"{name}.mha").exists():
            pat_series[pat] = name

    rows: list[dict] = []
    half = CROP_SIZE // 2
    n_total = n_ok = n_skip = 0

    for _, lv in levels.iterrows():
        n_total += 1
        pat = int(lv["Patient"]); lab = int(lv["IVD label"]); split = lv["split"]
        sname = pat_series.get(pat)
        if sname is None:
            n_skip += 1
            continue

        cen = disc_centroid(str(mask_dir / f"{sname}.mha"), lab)
        if cen is None:
            n_skip += 1
            continue
        cx, cy, cz = cen

        img_zxy, _ = load_series(sname)
        if not (0 <= cz < img_zxy.shape[0]):
            n_skip += 1
            continue

        img2d = np.clip(img_zxy[cz], -1000, 4000)              # stable window
        patch = crop_around(img2d, cy, cx, half)
        patch = ((patch.astype(np.float32) + 1000.0) / 5000.0)  # -> ~[0, 1]

        out_dir = CROP_DIR / f"crops_{split}"
        out_dir.mkdir(parents=True, exist_ok=True)
        sample_id = f"p{pat}_{sname}_d{lab}"
        np.save(out_dir / f"{sample_id}.npy", patch)

        rows.append({
            "sample_id": sample_id,
            "patient": pat, "series": sname, "split": split,
            "mask_label": lab,
            "Pfirrmann": lv["Pfirrman grade"],
            # Modic is stored 0/1/2/3 (graded type); binarize to "any Modic change present".
            "Modic": int(lv["Modic"] >= 1),
            "Spondylolisthesis": lv["Spondylolisthesis"],
            "Disc_herniation": lv["Disc herniation"],
            "Disc_narrowing": lv["Disc narrowing"],
            "Disc_bulging": lv["Disc bulging"],
            "UP_endplate": lv["UP endplate"],
            "LOW_endplate": lv["LOW endplate"],
        })
        n_ok += 1

    labels_df = pd.DataFrame(rows)
    labels_df.to_csv(CROP_DIR / "labels.csv", index=False)

    with open(CROP_DIR / "crops_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Total graded rows   : {n_total}\n")
        f.write(f"Extracted crops     : {n_ok}\n")
        f.write(f"Skipped (no series / no mask label): {n_skip}\n\n")
        if n_ok:
            f.write("Per split:\n" + labels_df.groupby("split").size().to_string() + "\n\n")
            f.write("Pfirrmann distribution:\n")
            f.write(labels_df["Pfirrmann"].value_counts().sort_index().to_string() + "\n")
    print(open(CROP_DIR / "crops_summary.txt", encoding="utf-8").read())


if __name__ == "__main__":
    main()
