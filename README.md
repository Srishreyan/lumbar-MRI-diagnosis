# Lumbar Spine MRI Diagnosis — Trustworthy Multi-Task Framework

Lightweight, well-calibrated multi-task framework for lumbar spine MRI (SPIDER dataset) with uncertainty and explainability.

> Paper: `Trustworthy_Spine_MRI_Conference_Paper_final.docx` (IEEE conference format, 2-column)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If `requirements.txt` is missing, install core deps:

```bash
pip install torch torchvision timm SimpleITK scikit-learn matplotlib seaborn pandas scipy pillow python-docx
```

## Data

SPIDER (218 patients, 447 T1/T2 sagittal scans, 1515 discs) is not tracked in git — download separately:

- SPIDER official: https://spider.grand-challenge.org/ (images + masks)
- After download: run `python prepare_dataset.py` then `python extract_disc_crops.py` to generate `crops/` (224×224 disc crops + `labels.csv`).

Expected layout after extraction:

```
crops/
  labels.csv
  crops_train/
  crops_val/
  crops_test/
```

## Training

```bash
# Main model: EfficientViT-b1 + CBAM + CORN (Pfirrmann) — ~40 epochs
python train_efficientvit.py

# Baseline for comparison
python baseline_train_resnet50.py
```

Checkpoints (`ckpt_*.pth`) are gitignored — add your Drive/Zenodo link here after upload.

## Evaluation

```bash
python evaluate_baseline.py
python eval_tta_uncertainty.py        # TTA uncertainty
python eval_ensemble_uncertainty.py   # Deep ensemble
python calibrate.py                   # temperature scaling (ECE / reliability)
python tta_analysis.py
```

## Figures

```bash
# Regenerate all paper figures (single-file, Colab-friendly)
# Open COLAB_REGENERATE_ALL_FIGURES.ipynb in Colab and run all cells
# Or locally:
python gen_paper_figures.py
python improved_experiments.py   # CV / ablation / AUC comparisons
python gen_cam_progression.py    # 5×5 CAM grid (optional, needs checkpoint)
```

Outputs: `figures/` and `figures_improved/` (tracked).

## Paper

Source: `Trustworthy_Spine_MRI_Conference_Paper_final.docx`

Recreate DOCX programmatically (optional):

```bash
python make_ieee_final3.py
python rebuild_paper.py
python fill_paper.py
```

## Colab

- `COLAB_REGENERATE_ALL_FIGURES.ipynb` — copy-paste friendly, single `pip install` cell
- `COLAB_REGENERATE_ALL_FIGURES_CELLS.md` — same cells as markdown
- `COLAB_INSTRUCTIONS.md` — step-by-step for Drive mount flow

## Cite

If you use this work, cite SPIDER and the paper draft included in this repo.
