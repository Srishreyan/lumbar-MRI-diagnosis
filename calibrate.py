#!/usr/bin/env python
# calibrate.py
# ------------------------------------------------------------------
# Post-hoc calibration for the SPIDER multi-task models.
#
# Fits temperature scaling per task on the VALIDATION split, then
# reports Expected Calibration Error (ECE) + Brier score on the TEST
# split, before and after calibration, for every binary pathology and
# for Pfirrmann. Works for any checkpoint (resnet50 / efficientvit_b*).
#
# Binary tasks: p = sigmoid(z / T_bin)          (fit by BCE NLL)
# Pfirrmann:    p = softmax(z / T_pfir)         (fit by CE NLL)
# ------------------------------------------------------------------
from __future__ import annotations

import pathlib
import json
import argparse

import numpy as np
import torch
import torch.nn as nn

from scipy.optimize import minimize_scalar

from evaluate_baseline import EvalCropDataset, predict, BIN_COLS


# ------------------------------------------------------------------
# Calibration metrics (all numpy, no sklearn dependency)
# ------------------------------------------------------------------
def expected_calibration_error(y_true, probs, n_bins=15):
    """ECE: bin confidence-vs-accuracy gap, weighted by bin frequency."""
    conf = np.clip(probs, 1e-12, 1 - 1e-12)
    bins = np.linspace(0, 1, n_bins + 1)
    ece, n = 0.0, len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum() == 0:
            continue
        acc = (y_true[m] == (conf[m] > 0.5)).mean()
        ece += (m.sum() / n) * abs(acc - conf[m].mean())
    return float(ece)


def brier_score(y_true, probs):
    """Binary Brier score for one task."""
    p = np.clip(probs, 0.0, 1.0)
    return float(np.mean((p - y_true) ** 2))


def pfirrmann_ece(y_true, probs_matrix, n_bins=15):
    """Multiclass ECE on the argmax class confidence (standard formulation)."""
    conf = probs_matrix.max(axis=1)
    pred = probs_matrix.argmax(axis=1)
    bins = np.linspace(0, 1, n_bins + 1)
    ece, n = 0.0, len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum() == 0:
            continue
        acc = (pred[m] == y_true[m]).mean()
        ece += (m.sum() / n) * abs(acc - conf[m].mean())
    return float(ece)


def brier_multiclass(y_true, probs_matrix):
    """Multiclass Brier = mean over rows of ||onehot - probs||^2 / 2."""
    K = probs_matrix.shape[1]
    onehot = np.eye(K)[y_true.astype(int)]
    return float(np.mean(((onehot - probs_matrix) ** 2).sum(axis=1) / 2))


# ------------------------------------------------------------------
# Temperature fitting (per task, on val)
# ------------------------------------------------------------------
def fit_temperature_binary(z: np.ndarray, y: np.ndarray, t0: float = 1.0):
    """Temperature for one binary task: minimize BCE NLL on the logits."""
    zt = torch.from_numpy(z.astype(np.float32))
    yt = torch.from_numpy(y.astype(np.float32))

    def nll(T):
        p = torch.sigmoid(zt / T)
        loss = nn.functional.binary_cross_entropy(p, yt)
        return float(loss.detach().numpy())

    res = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    return float(res.x)


def fit_temperature_pfirrmann(z: np.ndarray, y: np.ndarray):
    """Temperature for Pfirrmann: minimize CE NLL on the logits."""
    zt = torch.from_numpy(z.astype(np.float32))
    yt = torch.from_numpy(y.astype(np.int64))

    def nll(T):
        p = nn.functional.log_softmax(zt / T, dim=1)
        return float(-p.gather(1, yt[:, None]).mean().detach().numpy())

    res = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    return float(res.x)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Temperature-scale calibration + ECE/Brier report")
    ap.add_argument("--ckpt", type=pathlib.Path, required=True)
    ap.add_argument("--arch", type=str, default="resnet50",
                    help="resnet50 or efficientvit_b* (must match checkpoint)")
    ap.add_argument("--data_dir", type=pathlib.Path, default="crops")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out", type=pathlib.Path, default="results")
    ap.add_argument("--tag", type=str, default="calib")
    args = ap.parse_args()

    device = torch.device("cuda") if (args.device == "auto" and torch.cuda.is_available()) else torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    # Model (match training architecture)
    if args.arch == "resnet50":
        from evaluate_baseline import build_model as build_rn
        model = build_rn(num_bin=7, num_pfir=5).to(device)
    else:
        from efficientvit_cbam import build_model as build_ev
        model = build_ev(backbone=args.arch, pretrained=False).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded {args.ckpt} ({args.arch}) on {device}")

    # --- Val logits (for fitting T) ----------------------------------------
    val_ds = EvalCropDataset(args.data_dir, "val")
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    v = predict(model, val_loader, device)

    # --- Test logits (for the report) --------------------------------------
    test_ds = EvalCropDataset(args.data_dir, "test")
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    t = predict(model, test_loader, device)
    print(f"Val {len(v['id'])} samples (fit T), Test {len(t['id'])} samples (report)")

    # --- Fit temperatures ---------------------------------------------------
    T_bin = [fit_temperature_binary(v["logits_bin"][:, i], v["y_bin"][:, i])
             for i in range(len(BIN_COLS))]
    T_pf = fit_temperature_pfirrmann(v["logits_pf"], v["y_pf"])

    # --- Report: ECE / Brier before & after --------------------------------
    report = {"checkpoint": str(args.ckpt), "arch": args.arch, "n_val": len(v["id"]),
              "n_test": len(t["id"]), "T_binary": T_bin, "T_pfirrmann": T_pf,
              "binary_tasks": [], "pfirrmann": {}}

    print("\n--- Binary tasks: ECE (Brier) before -> after ---")
    for i, col in enumerate(BIN_COLS):
        yt = t["y_bin"][:, i]
        p_raw = 1 / (1 + np.exp(-t["logits_bin"][:, i]))
        p_cal = 1 / (1 + np.exp(-t["logits_bin"][:, i] / T_bin[i]))
        row = {
            "task": col,
            "ece_before": expected_calibration_error(yt, p_raw),
            "ece_after": expected_calibration_error(yt, p_cal),
            "brier_before": brier_score(yt, p_raw),
            "brier_after": brier_score(yt, p_cal),
            "T": T_bin[i],
        }
        report["binary_tasks"].append(row)
        print(f"{col:20s} {row['ece_before']:.4f} ({row['brier_before']:.4f}) -> "
              f"{row['ece_after']:.4f} ({row['brier_after']:.4f})  T={row['T']:.2f}")

    # Pfirrmann
    yp = t["y_pf"]
    p_raw = torch.softmax(torch.from_numpy(t["logits_pf"]), dim=1).numpy()
    p_cal = torch.softmax(torch.from_numpy(t["logits_pf"] / T_pf), dim=1).numpy()
    report["pfirrmann"] = {
        "ece_before": pfirrmann_ece(yp, p_raw),
        "ece_after": pfirrmann_ece(yp, p_cal),
        "brier_before": brier_multiclass(yp, p_raw),
        "brier_after": brier_multiclass(yp, p_cal),
        "T": T_pf,
    }
    print(f"\nPfirrmann: ECE {report['pfirrmann']['ece_before']:.4f} ({report['pfirrmann']['brier_before']:.4f}) "
          f"-> {report['pfirrmann']['ece_after']:.4f} ({report['pfirrmann']['brier_after']:.4f})  T={T_pf:.2f}")

    # Summary line
    mean_ece_b = np.mean([r["ece_before"] for r in report["binary_tasks"]] + [report["pfirrmann"]["ece_before"]])
    mean_ece_a = np.mean([r["ece_after"] for r in report["binary_tasks"]] + [report["pfirrmann"]["ece_after"]])
    print(f"\nMean ECE (all tasks): {mean_ece_b:.4f} -> {mean_ece_a:.4f}")

    with open(args.out / f"calibration_{args.tag}.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved -> {args.out / ('calibration_' + args.tag + '.json')}")


if __name__ == "__main__":
    main()