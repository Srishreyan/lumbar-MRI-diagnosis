#!/usr/bin/env python
# eval_uncertainty.py
# ------------------------------------------------------------------
# MC-Dropout uncertainty evaluation for the SPIDER multi-task model.
#
# Runs N stochastic forward passes per test sample and measures whether
# predictive variance (epistemic uncertainty) is *informative* — i.e.
# does the model know when it is wrong? Reports, per task:
#   - mean-pass accuracy / AUC  (uncertainty-smoothed predictions)
#   - uncertainty-AUC: can variance discriminate correct vs incorrect?
#   - selective accuracy: accuracy vs coverage as we drop the most
#     uncertain samples (a risk-coverage curve summary)
#
# Only meaningful when the checkpoint was trained with mc_dropout > 0.
# ------------------------------------------------------------------
from __future__ import annotations

import pathlib
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluate_baseline import EvalCropDataset, BIN_COLS, roc_auc, accuracy


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="MC-Dropout uncertainty evaluation")
    ap.add_argument("--ckpt", type=pathlib.Path, required=True)
    ap.add_argument("--arch", type=str, default="efficientvit_b1")
    ap.add_argument("--mc_dropout", type=float, default=0.2,
                    help="head dropout rate used at training time (must match ckpt)")
    ap.add_argument("--backbone_mc_dropout", type=float, default=0.0,
                    help="backbone Dropout2d rate used at training time (must match ckpt)")
    ap.add_argument("--n_samples", type=int, default=20,
                    help="number of stochastic forward passes per sample")
    ap.add_argument("--data_dir", type=pathlib.Path, default="crops")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out", type=pathlib.Path, default="results")
    ap.add_argument("--tag", type=str, default="uncert")
    args = ap.parse_args()

    device = torch.device("cuda") if (args.device == "auto" and torch.cuda.is_available()) else torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    from efficientvit_cbam import build_model
    model = build_model(backbone=args.arch, pretrained=False,
                        mc_dropout_p=args.mc_dropout,
                        backbone_mc_dropout_p=args.backbone_mc_dropout).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded {args.ckpt} | mc_dropout={args.mc_dropout} "
          f"| backbone_mc_dropout={args.backbone_mc_dropout} | n_samples={args.n_samples} on {device}")

    ds = EvalCropDataset(args.data_dir, args.split)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Evaluating {len(ds)} samples ({args.split}) x {args.n_samples} stochastic passes")

    # Collect per-sample mean + variance of logits, plus true labels.
    # Raw per-pass logits are saved too, because logit-variance is a weak
    # uncertainty proxy -- disagreement measures (flip-rate / entropy /
    # margin) are computed in tta_analysis.py from the same tensor.
    mean_all, var_all, views_all, y_bin_all, y_pf_all = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            xs = img.repeat_interleave(args.n_samples, dim=0)
            out = model(xs).view(img.size(0), args.n_samples, -1)
            views_all.append(out.cpu())
            mean_all.append(out.mean(dim=1).cpu())
            var_all.append(out.var(dim=1, unbiased=False).cpu())
            y_bin_all.append(batch["y_bin"])
            y_pf_all.append(batch["y_pfirr"])

    views = torch.cat(views_all).numpy()     # [N, n_samples, 7+5]
    mean = torch.cat(mean_all).numpy()       # [N, 7+5]
    var = torch.cat(var_all).numpy()          # [N, 7+5]
    y_bin = torch.cat(y_bin_all).numpy()      # [N, 7]
    y_pf = torch.cat(y_pf_all).numpy()        # [N]
    N = len(y_pf)

    np.savez(args.out / f"mc_views_{args.tag}.npz", views=views,
             y_bin=y_bin, y_pf=y_pf)

    report = {"checkpoint": str(args.ckpt), "arch": args.arch, "n_samples": args.n_samples,
              "split": args.split, "n": N, "binary_tasks": [], "pfirrmann": {}}

    print(f"\n{'task':18s} {'acc':>6s} {'auc':>6s} {'unc-AUC':>7s} {'sel@50':>6s} {'sel@90':>6s}")
    print("-" * 62)

    # --- Binary tasks ------------------------------------------------------
    for i, col in enumerate(BIN_COLS):
        z = mean[:, i]
        p = 1 / (1 + np.exp(-z))
        pred = p > 0.5
        acc = accuracy(y_bin[:, i], pred)
        auc = roc_auc(y_bin[:, i], p)
        unc = var[:, i]                       # predictive variance = uncertainty
        err = (pred != y_bin[:, i]).astype(int)
        unc_auc = roc_auc(err, unc)           # can variance flag errors?
        sel50, sel90 = selective_accuracy(err, unc, 0.50), selective_accuracy(err, unc, 0.90)
        report["binary_tasks"].append({
            "task": col, "accuracy": acc, "auc": auc,
            "uncertainty_auc": unc_auc, "selective_acc@50": sel50, "selective_acc@90": sel90,
            "mean_var": float(unc.mean()),
        })
        print(f"{col:18s} {acc*100:5.1f}% {auc:6.3f} {unc_auc:7.3f} {sel50*100:5.1f}% {sel90*100:5.1f}%")

    # --- Pfirrmann ---------------------------------------------------------
    lb = mean[:, 7:]
    p_pf = np.exp(lb - lb.max(1, keepdims=True)); p_pf /= p_pf.sum(1, keepdims=True)
    pred_pf = lb.argmax(1)
    acc_pf = accuracy(y_pf, pred_pf)
    unc_pf = var[:, 7:].mean(axis=1)          # mean variance across pfirrmann logits
    err_pf = (pred_pf != y_pf).astype(int)
    unc_auc_pf = roc_auc(err_pf, unc_pf)
    sel50_pf, sel90_pf = selective_accuracy(err_pf, unc_pf, 0.50), selective_accuracy(err_pf, unc_pf, 0.90)
    report["pfirrmann"] = {"accuracy": acc_pf, "uncertainty_auc": unc_auc_pf,
                           "selective_acc@50": sel50_pf, "selective_acc@90": sel90_pf,
                           "mean_var": float(unc_pf.mean())}
    print(f"{'Pfirrmann':18s} {acc_pf*100:5.1f}% {'—':>6s} {unc_auc_pf:7.3f} {sel50_pf*100:5.1f}% {sel90_pf*100:5.1f}%")

    with open(args.out / f"uncertainty_{args.tag}.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved -> {args.out / ('uncertainty_' + args.tag + '.json')}")


def selective_accuracy(err: np.ndarray, unc: np.ndarray, coverage: float) -> float:
    """Accuracy on the most-confident `coverage` fraction (lowest uncertainty)."""
    n_keep = int(len(err) * coverage)
    if n_keep == 0:
        return float("nan")
    keep = np.argsort(unc)[:n_keep]
    return float(1.0 - err[keep].mean())


if __name__ == "__main__":
    main()