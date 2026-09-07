"""
prepare_dataset.py
------------------
One-time prep for the SPIDER lumbar spine MRI dataset.

Steps:
  1. Extract images.zip and masks.zip (skipped if already extracted).
  2. Load per-series metadata (overview.csv) and per-disc labels
     (radiological_gradings.csv).
  3. Split PATIENTS into train / val / test (70/15/15) so the same person
     never appears in two splits -> no data leakage.
  4. Write merged tables + a class-balance report.

Run:
    python prepare_dataset.py
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DATASET_DIR = Path(__file__).resolve().parent        # folder this script lives in
EXTRACT_DIR = DATASET_DIR / "extracted"
SPLIT = (0.70, 0.15, 0.15)                           # train, val, test
SEED = 42                                            # fixed -> reproducible splits

LABEL_COLS = [
    "Modic",
    "UP endplate",
    "LOW endplate",
    "Spondylolisthesis",
    "Disc herniation",
    "Disc narrowing",
    "Disc bulging",
]


def extract_if_needed(archive: str, out_dir: Path) -> None:
    """Unzip `archive` into `out_dir` unless it has already been extracted."""
    if out_dir.is_dir() and any(out_dir.rglob("*")):
        print(f"[skip] {archive} already extracted -> {out_dir}")
        return
    print(f"[extract] {archive} -> {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(DATASET_DIR / archive) as z:
        z.extractall(out_dir)
    print(f"         done ({len(list(out_dir.rglob('*')))} files)")


def index_scans(img_dir: Path) -> dict[str, Path]:
    """Map a series stem (e.g. "1_t1") to its scan file path on disk.

    SPIDER ships images/masks as MetaImage `.mha` files (under a nested
    `images/images/` folder); NIfTI is accepted too for robustness.
    """
    index: dict[str, Path] = {}
    for p in img_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".mha", ".nii", ".gz"):
            index[p.name.split(".")[0]] = p
    return index


def main() -> None:
    # 1. Extract archives -----------------------------------------------------
    extract_if_needed("images.zip", EXTRACT_DIR / "images")
    extract_if_needed("masks.zip", EXTRACT_DIR / "masks")

    # 2. Load metadata + labels ----------------------------------------------
    overview = pd.read_csv(DATASET_DIR / "overview.csv")
    gradings = pd.read_csv(DATASET_DIR / "radiological_gradings.csv")

    # Patient id is the leading number of the series name, e.g. "1_t1" -> 1.
    overview["Patient"] = overview["new_file_name"].str.split("_").str[0].astype(int)
    overview["contrast"] = overview["new_file_name"].str.split("_").str[1]
    gradings["Patient"] = gradings["Patient"].astype(int)

    # 3. Patient-level split ---------------------------------------------------
    patients = np.sort(overview["Patient"].unique())
    rng = np.random.RandomState(SEED)
    shuffled = patients.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * SPLIT[0])
    n_val = int(n * SPLIT[1])
    split_names = (
        ["train"] * n_train
        + ["val"] * n_val
        + ["test"] * (n - n_train - n_val)
    )
    split_df = pd.DataFrame({"Patient": shuffled, "split": split_names})
    split_df = split_df.sort_values("Patient").reset_index(drop=True)
    split_df.to_csv(DATASET_DIR / "split_assignments.csv", index=False)

    # 4. Merged tables ---------------------------------------------------------
    # Per-series table (one row per MRI series)
    series = overview.merge(split_df, on="Patient", how="left")
    images_map = index_scans(EXTRACT_DIR / "images")
    masks_map = index_scans(EXTRACT_DIR / "masks")
    series["image_path"] = series["new_file_name"].map(
        lambda name: str(images_map.get(name, "MISSING"))
    )
    series["mask_path"] = series["new_file_name"].map(
        lambda name: str(masks_map.get(name, "MISSING"))
    )
    series.to_csv(DATASET_DIR / "dataset_series.csv", index=False)

    # Per-disc-level table (the multi-task ground truth)
    levels = gradings.merge(split_df, on="Patient", how="left")
    levels.to_csv(DATASET_DIR / "dataset_levels.csv", index=False)

    # 5. Report -----------------------------------------------------------------
    report(overview, gradings, split_df, series, levels)


def report(
    overview: pd.DataFrame,
    gradings: pd.DataFrame,
    split_df: pd.DataFrame,
    series: pd.DataFrame,
    levels: pd.DataFrame,
) -> None:
    lines: list[str] = []
    add = lines.append

    add("=" * 60)
    add("SPIDER dataset summary")
    add("=" * 60)
    add(f"Patients total        : {overview['Patient'].nunique()}")
    add(f"Series total          : {len(overview)}")
    for name in ("train", "val", "test"):
        n_pat = (split_df["split"] == name).sum()
        n_ser = (series["split"] == name).sum()
        n_lvl = (levels["split"] == name).sum()
        add(f"  {name:5s} : {n_pat:3d} patients, {n_ser:3d} series, "
            f"{n_lvl:4d} disc levels")

    # Sanity checks
    add("-" * 60)
    add("Sanity checks")
    missing = series["image_path"].eq("MISSING").sum()
    add(f"  Series without image file on disk : {missing}")
    missing_masks = series["mask_path"].eq("MISSING").sum()
    add(f"  Series without mask file on disk  : {missing_masks}")
    real = series.loc[series["image_path"].ne("MISSING"), "image_path"]
    if len(real):
        add(f"  Image file format detected        : .{real.iloc[0].rsplit('.', 1)[-1]}")
    missing_labels = gradings["Patient"].isin(overview["Patient"]).sum()
    add(f"  Graded patients present in overview : {missing_labels} / {len(gradings)}")

    # Class balance per label
    add("-" * 60)
    add("Per-disc label distributions (rows = patient-IVD level)")
    for col in LABEL_COLS:
        counts = levels[col].value_counts(dropna=False).sort_index()
        add(f"  {col:20s}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))

    add("Pfirrmann grade distribution (severity task target)")
    counts = levels["Pfirrman grade"].value_counts(dropna=False).sort_index()
    add(f"  Pfirrman grade : " + ", ".join(f"{k}={v}" for k, v in counts.items()))

    # Contrast coverage
    add("-" * 60)
    add("Series contrast coverage")
    add(f"  {overview['contrast'].value_counts().to_string()}")

    # Write report + print
    text = "\n".join(lines)
    (DATASET_DIR / "dataset_summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    sys.exit(main())
