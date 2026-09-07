#!/usr/bin/env python
# eval_tta_uncertainty.py
# ------------------------------------------------------------------
# Test-Time Augmentation (TTA) uncertainty evaluation.
#
# Instead of MC-Dropout (random mask sampling), TTA variance measures
# whether the model's prediction changes under *semantically meaningful*
# geometric transformations. This is often more informative because:
#   - Augmentations respect image semantics (flips, rotations)
#   - Variance reflects genuine prediction instability, not dropout noise
#   - Works with ANY checkpoint (no MC-dropout training needed)
#
# For each test sample:
#   1. Generate N augmented versions (fixed augmentations, not random)
#   2. Run all N through the model (batched for speed)
#   3. Compute per-task variance across the N outputs
#   4. Report uncertainty-AUC: can variance discriminate errors?
#
# The plain ckpt_evit_b1.pth is sufficient — no MC-dropout required.
# ------------------------------------------------------------------
from __future__ import annotations

import pathlib
import json
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, Rotate,
    RandomBrightnessContrast, Affine,
)

from evaluate_baseline import EvalCropDataset, BIN_COLS, roc_auc, accuracy
from efficientvit_cbam import build_model


# ------------------------------------------------------------------
# TTA augmentation pipeline
# ------------------------------------------------------------------
def build_tta_aug(seed: int = 42):
    """Return a deterministic TTA augmentation pipeline.

    These are mild, clinically plausible transforms — the same ones
    used during training, but applied N times per sample with different
    random states to measure prediction stability.
    """
    return Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        Rotate(limit=15, p=0.5),
        RandomBrightnessContrast(p=0.5),
        Affine(translate_percent=0.05, scale=0.05, p=0.5),
    ])


def augment_sample(np_img: np.ndarray, aug, n_samples: int) -> list[np.ndarray]:
    """Generate n_samples augmented versions of a single image.

    Each call to `aug` draws new random params from the same pipeline,
    producing semantically similar but pixel-different outputs.
    """
    return [aug(image=np_img)["image"] for _ in range(n_samples)]


# ------------------------------------------------------------------
# TTA prediction: batched forward pass over all augmentations
# ------------------------------------------------------------------
def tta_predict(model, ds, device, n_samples: int = 20, batch_size: int = 32):
    """Run TTA on the entire dataset.

    For each sample, generate n_samples augmented versions, flatten
    into a big batch, run one forward pass, reshape back, and compute
    per-sample mean + variance.
    """
    aug = build_tta_aug()
    model.eval()

    all_mean, all_var, all_y_bin, all_y_pf = [], [], [], []

    for idx in range(len(ds)):
        row = ds.meta.iloc[idx]
        np_img = np.load((ds.path / f"{row['sample_id']}.npy").resolve()).astype(np.float32)

        # Generate N augmented versions
        augmented = augment_sample(np_img, aug, n_samples)

        # Stack into [N, 3, H, W]
        imgs = []
        for a in augmented:
            t = torch.from_numpy(a).unsqueeze(0).repeat(3, 1, 1).float()  # [3, H, W]
            imgs.append(t)
        batch = torch.stack(imgs, dim=0).to(device)  # [N, 3, H, W]

        # Forward pass (all N at once for speed)
        with torch.no_grad():
            logits = model(batch)  # [N, 12]

        # Per-sample statistics
        mean = logits.mean(dim=0)   # [12]
        var = logits.var(dim=0, unbiased=False)  # [12]
        all_mean.append(mean.cpu())
        all_var.append(var.cpu())
        all_y_bin.append(torch.tensor([row[c] for c in BIN_COLS], dtype=torch.float32))
        all_y_pf.append(torch.tensor(row["Pfirrmann"] - 1, dtype=torch.long))

    return {
        "mean": torch.stack(all_mean).numpy(),   # [N_samples, 12]
        "var": torch.stack(all_var).numpy(),      # [N_samples, 12]
        "y_bin": torch.stack(all_y_bin).numpy(),  # [N_samples, 7]
        "y_pf": torch.stack(all_y_pf).numpy(),    # [N_samples]
    }


# ------------------------------------------------------------------
# Selective accuracy
# ------------------------------------------------------------------
def selective_accuracy(err: np.ndarray, unc: np.ndarray, coverage: float) -> float:
    """Accuracy on the most-confident `coverage` fraction (lowest uncertainty)."""
    n_keep = int(len(err) * coverage)
    if n_keep == 0:
        return float("nan")
    keep = np.argsort(unc)[:n_keep]
    return float(1.0 - err[keep].mean())


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="TTA variance uncertainty evaluation")
    ap.add_argument("--ckpt", type=pathlib.Path, required=True,
                    help="checkpoint (plain or MC-dropout — TTA works with both)")
    ap.add_argument("--arch", type=str, default="efficientvit_b1")
    ap.add_argument("--n_samples", type=int, default=20,
                    help="number of TTA augmentations per sample")
    ap.add_argument("--data_dir", type=pathlib.Path, default="crops")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out", type=pathlib.Path, default="results")
    ap.add_argument("--tag", type=str, default="tta")
    args = ap.parse_args()

    device = torch.device("cuda") if (args.device == "auto" and torch.cuda.is_available()) else torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    from efficientvit_cbam import build_model
    model = build_model(backbone=args.arch, pretrained=False).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded {args.ckpt} ({args.arch}) on {device}")
    print(f"TTA: {args.n_samples} augmentations per sample")

    ds = EvalCropDataset(args.data_dir, args.split)
    print(f"Evaluating {len(ds)} samples ({args.split})")

    # Run TTA prediction
    res = tta_predict(model, ds, device, n_samples=args.n_samples, batch_size=args.batch_size)

    mean = res["mean"]   # [N, 12]
    var = res["var"]     # [N, 12]
    y_bin = res["y_bin"] # [N, 7]
    y_pf = res["y_pf"]   # [N]
    N = len(y_pf)

    report = {
        "checkpoint": str(args.ckpt), "arch": args.arch,
        "n_tta_samples": args.n_samples, "split": args.split, "n": N,
        "binary_tasks": [], "pfirrmann": {},
    }

    print(f"\n{'task':18s} {'acc':>6s} {'auc':>6s} {'unc-AUC':>7s} {'sel@50':>6s} {'sel@90':>6s} {'mean_var':>8s}")
    print("-" * 68)

    # --- Binary tasks ------------------------------------------------------
    for i, col in enumerate(BIN_COLS):
        z = mean[:, i]
        p = 1 / (1 + np.exp(-z))
        pred = p > 0.5
        acc = accuracy(y_bin[:, i], pred)
        auc = roc_auc(y_bin[:, i], p)
        unc = var[:, i]
        err = (pred != y_bin[:, i]).astype(int)
        unc_auc = roc_auc(err, unc)
        sel50 = selective_accuracy(err, unc, 0.50)
        sel90 = selective_accuracy(err, unc, 0.90)
        report["binary_tasks"].append({
            "task": col, "accuracy": float(acc), "auc": float(auc),
            "uncertainty_auc": float(unc_auc),
            "selective_acc@50": float(sel50), "selective_acc@90": float(sel90),
            "mean_var": float(unc.mean()),
        })
        print(f"{col:18s} {acc*100:5.1f}% {auc:6.3f} {unc_auc:7.3f} "
              f"{sel50*100:5.1f}% {sel90*100:5.1f}% {unc.mean():8.4f}")

    # --- Pfirrmann ---------------------------------------------------------
    lb = mean[:, 7:]
    p_pf = np.exp(lb - lb.max(1, keepdims=True))
    p_pf /= p_pf.sum(1, keepdims=True)
    pred_pf = lb.argmax(1)
    acc_pf = accuracy(y_pf, pred_pf)
    unc_pf = var[:, 7:].mean(axis=1)
    err_pf = (pred_pf != y_pf).astype(int)
    unc_auc_pf = roc_auc(err_pf, unc_pf)
    sel50_pf = selective_accuracy(err_pf, unc_pf, 0.50)
    sel90_pf = selective_accuracy(err_pf, unc_pf, 0.90)
    report["pfirrmann"] = {
        "accuracy": float(acc_pf), "uncertainty_auc": float(unc_auc_pf),
        "selective_acc@50": float(sel50_pf), "selective_acc@90": float(sel90_pf),
        "mean_var": float(unc_pf.mean()),
    }
    print(f"{'Pfirrmann':18s} {acc_pf*100:5.1f}% {'—':>6s} {unc_auc_pf:7.3f} "
          f"{sel50_pf*100:5.1f}% {sel90_pf*100:5.1f}% {unc_pf.mean():8.4f}")

    # --- Summary ---
    mean_unc_auc_bin = np.mean([r["uncertainty_auc"] for r in report["binary_tasks"]])
    print(f"\nMean binary uncertainty-AUC: {mean_unc_auc_bin:.3f} "
          f"({'informative' if mean_unc_auc_bin > 0.5 else 'NOT informative'})")
    print(f"Pfirrmann uncertainty-AUC:   {unc_auc_pf:.3f} "
          f"({'informative' if unc_auc_pf > 0.5 else 'NOT informative'})")

    with open(args.out / f"uncertainty_{args.tag}.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved -> {args.out / ('uncertainty_' + args.tag + '.json')}")


if __name__ == "__main__":
    main()
