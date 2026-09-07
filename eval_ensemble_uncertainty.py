#!/usr/bin/env python
# eval_ensemble_uncertainty.py
# ------------------------------------------------------------------
# Deep ensemble uncertainty evaluation for the SPIDER multi-task model.
#
# Loads N independently trained models and measures *disagreement*
# between them — the gold-standard epistemic uncertainty signal.
#
# Unlike MC-Dropout (random mask within one model) or TTA variance
# (input perturbation within one model), ensemble variance captures
# *structural* model uncertainty: do different optimization runs
# converge to different predictions for the same input?
#
# For each test sample:
#   1. Forward pass through all N ensemble members
#   2. Compute per-task mean + variance across members
#   3. Report uncertainty-AUC: can disagreement flag errors?
#
# Members should be trained with different random seeds but otherwise
# identical configs (same architecture, data, hyperparameters).
# ------------------------------------------------------------------
from __future__ import annotations

import pathlib
import json
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluate_baseline import EvalCropDataset, BIN_COLS, roc_auc, accuracy
from efficientvit_cbam import build_model


# ------------------------------------------------------------------
# Ensemble prediction
# ------------------------------------------------------------------
def ensemble_predict(models: list, ds, device, batch_size: int = 32):
    """Run all ensemble members and collect per-sample predictions.

    Returns mean + variance of logits across members.
    """
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Collect per-model predictions
    all_logits = []  # list of [N, 12] arrays, one per model
    all_y_bin, all_y_pf = None, None

    for model in models:
        model.eval()
        logits_list, y_bin_list, y_pf_list = [], [], []
        with torch.no_grad():
            for batch in loader:
                img = batch["image"].to(device)
                logits = model(img)
                logits_list.append(logits.cpu())
                y_bin_list.append(batch["y_bin"])
                y_pf_list.append(batch["y_pfirr"])

        logits_np = torch.cat(logits_list).numpy()  # [N, 12]
        all_logits.append(logits_np)
        if all_y_bin is None:
            all_y_bin = torch.cat(y_bin_list).numpy()
            all_y_pf = torch.cat(y_pf_list).numpy()

    # Stack: [M, N, 12] where M = number of models, N = number of samples
    stacked = np.stack(all_logits, axis=0)
    mean = stacked.mean(axis=0)   # [N, 12]
    var = stacked.var(axis=0)  # [N, 12]

    return mean, var, all_y_bin, all_y_pf


# ------------------------------------------------------------------
# Selective accuracy
# ------------------------------------------------------------------
def selective_accuracy(err: np.ndarray, unc: np.ndarray, coverage: float) -> float:
    n_keep = int(len(err) * coverage)
    if n_keep == 0:
        return float("nan")
    keep = np.argsort(unc)[:n_keep]
    return float(1.0 - err[keep].mean())


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Deep ensemble uncertainty evaluation")
    ap.add_argument("--ckpts", type=pathlib.Path, nargs="+", required=True,
                    help="paths to ensemble member checkpoints")
    ap.add_argument("--arch", type=str, default="efficientvit_b1")
    ap.add_argument("--data_dir", type=pathlib.Path, default="crops")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out", type=pathlib.Path, default="results")
    ap.add_argument("--tag", type=str, default="ensemble")
    args = ap.parse_args()

    device = torch.device("cuda") if (args.device == "auto" and torch.cuda.is_available()) else torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    # Load all ensemble members
    models = []
    for ckpt in args.ckpts:
        model = build_model(backbone=args.arch, pretrained=False).to(device)
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
        print(f"Loaded {ckpt}")

    print(f"Ensemble: {len(models)} members on {device}")

    ds = EvalCropDataset(args.data_dir, args.split)
    print(f"Evaluating {len(ds)} samples ({args.split})")

    # Run ensemble prediction
    mean, var, y_bin, y_pf = ensemble_predict(models, ds, device, args.batch_size)
    N = len(y_pf)

    report = {
        "checkpoints": [str(c) for c in args.ckpts],
        "arch": args.arch,
        "n_members": len(models),
        "split": args.split,
        "n": N,
        "binary_tasks": [],
        "pfirrmann": {},
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
          f"({'INFORMATIVE' if mean_unc_auc_bin > 0.5 else 'NOT informative'})")
    print(f"Pfirrmann uncertainty-AUC:   {unc_auc_pf:.3f} "
          f"({'INFORMATIVE' if unc_auc_pf > 0.5 else 'NOT informative'})")

    with open(args.out / f"uncertainty_{args.tag}.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved -> {args.out / ('uncertainty_' + args.tag + '.json')}")


if __name__ == "__main__":
    main()
