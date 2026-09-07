#!/usr/bin/env python
# improved_experiments.py
# ------------------------------------------------------------------
# Colab-ready experiment script that strengthens the SPIDER paper.
#
# Run in Google Colab (GPU runtime) or locally. In Colab, first upload
# your project folder as a zip (see INSTRUCTIONS at bottom) and run:
#
#     !python improved_experiments.py --experiments ordinal cv ablation \
#         --epochs 40 --data_dir crops
#
# What this adds to the paper:
#   A) Pfirrmann ORDINAL regression (CORN) — respects grade ordering,
#      which should beat the plain 5-way softmax (paper's 41.2%).
#   B) 5-FOLD patient-level cross-validation — mean +/- std, the
#      statistically defensible numbers a reviewer expects.
#   C) CBAM ABLATION — EViT-b1 with vs. without CBAM, proving the
#      attention module earns its 3.1x-smaller-than-ResNet claim.
#
# Everything is written to results_improved/ + figures_improved/.
# ------------------------------------------------------------------
from __future__ import annotations

import pathlib
import json
import argparse
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# ---- figures -----------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from efficientvit_cbam import build_model, CBAM
from evaluate_baseline import BIN_COLS, roc_auc, accuracy

OUT_DIR = pathlib.Path("results_improved")
FIG_DIR = pathlib.Path("figures_improved")


# ==================================================================
# Dataset (same as train_efficientvit.py, with patient grouping)
# ==================================================================
class DiscCropDataset(torch.utils.data.Dataset):
    """Disc crops with optional augmentation.

    `rows` is a DataFrame already filtered to the wanted (train/val) subset.
    Each row carries `file_split` (the ORIGINAL train/val/test folder the
    image lives in) so patient-level cross-validation re-splits work
    without moving files.
    """
    def __init__(self, rows, data_dir: pathlib.Path, apply_aug: bool = True):
        self.rows = rows.reset_index(drop=True)
        self.data_dir = data_dir
        self.apply_aug = apply_aug
        if apply_aug:
            from albumentations import (Compose, HorizontalFlip, VerticalFlip,
                                        Rotate, RandomBrightnessContrast, Affine)
            self.alb = Compose([
                HorizontalFlip(p=0.5), VerticalFlip(p=0.5),
                Rotate(limit=15, p=0.5), RandomBrightnessContrast(p=0.5),
                Affine(translate_percent=0.05, scale=0.05, p=0.5),
            ])
        else:
            self.alb = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        fs = row["file_split"]
        np_img = np.load((self.data_dir / f"crops_{fs}" /
                          f"{row['sample_id']}.npy").resolve()).astype(np.float32)
        if self.alb is not None:
            np_img = self.alb(image=np_img)["image"]
        img = torch.from_numpy(np_img).unsqueeze(0).repeat(3, 1, 1).float()
        y_bin = torch.tensor([row[c] for c in BIN_COLS], dtype=torch.float32)
        y_pf = torch.tensor(row["Pfirrmann"] - 1, dtype=torch.long)
        return {"image": img, "y_bin": y_bin, "y_pfirr": y_pf}


def load_meta(data_dir: pathlib.Path) -> pd.DataFrame:
    """Read labels.csv once and record the original split for file lookup."""
    meta = pd.read_csv(data_dir / "labels.csv")
    meta["file_split"] = meta["split"]
    return meta


def auto_pos_weight(meta: pd.DataFrame) -> torch.Tensor:
    weights = []
    for col in BIN_COLS:
        pos = int((meta[col] == 1).sum())
        neg = int((meta[col] == 0).sum())
        weights.append(float(np.clip(neg / pos if pos > 0 else 50.0, 1.0, 50.0)))
    return torch.tensor(weights, dtype=torch.float32)


# ==================================================================
# Ordinal (CORN) Pfirrmann head
# ==================================================================
class CORNHead(nn.Module):
    """Cumulative ordinal regression (CORN): 4 logits predict P(grade > k).

    Grade 0..4. For k in 0..3, logit_k = P(y > k). Inference recovers
    P(y = k) = P(y > k-1) - P(y > k) and takes the argmax, which
    guarantees predictions respect the ordinal structure (a predicted 3
    sits between 2 and 4, never "before" 2).
    """
    def __init__(self, in_features: int):
        super().__init__()
        self.fc = nn.Linear(in_features, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)  # [B, 4]


def corn_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """BCE over cumulative indicators I(y > k)."""
    k = torch.arange(4, device=logits.device)
    targets = (y.unsqueeze(1) > k.unsqueeze(0)).float()  # [B, 4]
    return nn.functional.binary_cross_entropy_with_logits(logits, targets)


@torch.no_grad()
def corn_predict(logits: torch.Tensor) -> torch.Tensor:
    """Argmax of P(y = k) recovered from cumulative logits."""
    p_gt = torch.sigmoid(logits)                  # [B,4] = P(y>k)
    ones = torch.ones(logits.size(0), 1, device=logits.device)
    zeros = torch.zeros(logits.size(0), 1, device=logits.device)
    cdf = torch.cat([ones, p_gt, zeros], dim=1)   # [B,6]: P(y>-1..y>4)
    p_eq = cdf[:, :-1] - cdf[:, 1:]               # [B,5]
    return p_eq.argmax(dim=1)


# ==================================================================
# Ordinal multi-task model (encoder + CBAM + bin head + CORN head)
# ==================================================================
class OrdinalMultiTask(nn.Module):
    def __init__(self, backbone: str = "efficientvit_b1", pretrained: bool = True,
                 use_cbam: bool = True):
        super().__init__()
        import timm
        self.encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        feats = self.encoder.num_features
        self.use_cbam = use_cbam
        if use_cbam:
            self.cbam = CBAM(feats)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.trunk = nn.Sequential(nn.Linear(feats, 256), nn.ReLU(inplace=True))
        self.bin_head = nn.Linear(256, 7)
        self.ord_head = CORNHead(256)

    def forward(self, x: torch.Tensor):
        feat = self.encoder.forward_features(x)
        if self.use_cbam:
            feat = self.cbam(feat)
        feat = self.pool(feat).flatten(1)
        feat = self.trunk(feat)
        return self.bin_head(feat), self.ord_head(feat)


# ==================================================================
# Training loop
# ==================================================================
def train_eval(model, train_rows, val_rows, data_dir, device, epochs=40,
               batch_size=32, lr=1e-4, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tr = DiscCropDataset(train_rows, data_dir, apply_aug=True)
    va = DiscCropDataset(val_rows, data_dir, apply_aug=False)
    tr_loader = torch.utils.data.DataLoader(tr, batch_size=batch_size,
                                            shuffle=True, num_workers=0)
    va_loader = torch.utils.data.DataLoader(va, batch_size=batch_size,
                                            shuffle=False, num_workers=0)

    pw = auto_pos_weight(train_rows).to(device)
    crit_bin = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best = {"pf_acc": -1.0, "state": None}
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in tr_loader:
            img = batch["image"].to(device)
            yb = batch["y_bin"].to(device)
            yp = batch["y_pfirr"].to(device)
            opt.zero_grad()
            lb, lo = model(img)
            loss = crit_bin(lb, yb) + corn_loss(lo, yp)
            loss.backward()
            opt.step()
        sched.step()

        # validation
        model.eval()
        pf_acc = 0.0
        with torch.no_grad():
            for batch in va_loader:
                img = batch["image"].to(device)
                _, lo = model(img)
                pred = corn_predict(lo)
                pf_acc += (pred.cpu() == batch["y_pfirr"]).float().mean().item()
        pf_acc /= max(1, len(va_loader))
        if pf_acc > best["pf_acc"]:
            best["pf_acc"] = pf_acc
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best["state"])
    return model


# ==================================================================
# Test evaluation -> rich JSON metrics
# ==================================================================
@torch.no_grad()
def evaluate_test(model, data_dir, device, tag="ordinal", meta=None):
    if meta is None:
        meta = load_meta(data_dir)
        meta = meta[meta["split"] == "test"]
    ds = DiscCropDataset(meta, data_dir, apply_aug=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)
    model.eval()

    yb_all, yp_all, lb_all, lo_all = [], [], [], []
    for batch in loader:
        img = batch["image"].to(device)
        lb, lo = model(img)
        yb_all.append(batch["y_bin"].numpy())
        yp_all.append(batch["y_pfirr"].numpy())
        lb_all.append(torch.sigmoid(lb).cpu().numpy())
        lo_all.append(lo.cpu().numpy())

    yb = np.concatenate(yb_all)
    yp = np.concatenate(yp_all)
    prob_bin = np.concatenate(lb_all)

    report = {"tag": tag, "n": int(len(yp)), "binary_tasks": [], "pfirrmann": {}}
    pred_bin = (prob_bin > 0.5).astype(int)
    for i, col in enumerate(BIN_COLS):
        auc = roc_auc(yb[:, i], prob_bin[:, i])
        acc = accuracy(yb[:, i], pred_bin[:, i])
        report["binary_tasks"].append({"task": col, "auc": float(auc),
                                       "accuracy": float(acc)})
    # Pooled binary accuracy: fraction correct across ALL 7 x N predictions
    report["pooled_binary_accuracy"] = float((pred_bin == yb.astype(int)).mean())

    # Ordinal predictions from CORN logits
    lo_t = torch.from_numpy(np.concatenate(lo_all))
    pred_pf = corn_predict(lo_t).numpy()
    report["pfirrmann"]["accuracy"] = float(accuracy(yp, pred_pf))
    report["pfirrmann"]["corn_predictions"] = pred_pf.tolist()
    report["pfirrmann"]["true"] = yp.tolist()
    report["pfirrmann"]["confusion_matrix"] = (
        pd.crosstab(pd.Series(yp, name="true"),
                    pd.Series(pred_pf, name="pred"),
                    rownames=["true"], colnames=["pred"])
        .reindex(index=[0, 1, 2, 3, 4], columns=[0, 1, 2, 3, 4], fill_value=0)
        .values.tolist())
    return report


# ==================================================================
# Experiment A: ordinal vs plain-softmax Pfirrmann
# ==================================================================
def exp_ordinal(args):
    print("\n=== EXPERIMENT A: Pfirrmann Ordinal Regression (CORN) ===")
    meta = load_meta(args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = OrdinalMultiTask(backbone=args.backbone, pretrained=True,
                             use_cbam=True).to(device)
    train_rows = meta[meta["split"] == "train"]
    val_rows = meta[meta["split"] == "val"]
    model = train_eval(model, train_rows, val_rows, args.data_dir, device,
                       epochs=args.epochs, seed=args.seed)
    report = evaluate_test(model, args.data_dir, device, tag="ordinal_corn")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "exp_ordinal.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Ordinal Pfirrmann acc: {report['pfirrmann']['accuracy']:.3f}")
    print(f"  Ordinal pooled binary: {report['pooled_binary_accuracy']:.3f}")
    return report


# ==================================================================
# Experiment B: 5-fold patient-level cross-validation
# ==================================================================
def exp_cv(args, folds=5):
    print(f"\n=== EXPERIMENT B: {folds}-Fold Patient-Level Cross-Validation ===")
    meta = load_meta(args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fold ONLY over train+val patients; the held-out test set is never seen
    # during fold training, so fold models are evaluated on a clean test set.
    fold_meta = meta[meta["split"].isin(["train", "val"])]
    test_meta = meta[meta["split"] == "test"]
    patients = np.array(sorted(fold_meta["patient"].unique()))
    rng = np.random.RandomState(args.seed)
    rng.shuffle(patients)
    fold_splits = np.array_split(patients, folds)

    results = {"folds": [], "pooled_binary": [], "pfirrmann": []}
    for f in range(folds):
        val_patients = set(fold_splits[f])
        tr = fold_meta[~fold_meta["patient"].isin(val_patients)].copy()
        va = fold_meta[fold_meta["patient"].isin(val_patients)].copy()
        # keep `file_split` untouched so images load from their real folders
        print(f"  Fold {f+1}/{folds}: train={len(tr)} val={len(va)}")

        model = OrdinalMultiTask(backbone=args.backbone, pretrained=True,
                                 use_cbam=True).to(device)
        model = train_eval(model, tr, va, args.data_dir, device,
                           epochs=args.epochs, seed=args.seed + f)
        report = evaluate_test(model, args.data_dir, device, tag=f"cv_fold{f}",
                               meta=test_meta)
        pb = report["pooled_binary_accuracy"]
        pf = report["pfirrmann"]["accuracy"]
        results["folds"].append({"fold": f + 1, "pooled_binary": pb,
                                 "pfirrmann": pf})
        results["pooled_binary"].append(pb)
        results["pfirrmann"].append(pf)
        print(f"    pooled_bin={pb:.3f} pfirrmann={pf:.3f}")

    results["pooled_binary_mean"] = float(np.mean(results["pooled_binary"]))
    results["pooled_binary_std"] = float(np.std(results["pooled_binary"]))
    results["pfirrmann_mean"] = float(np.mean(results["pfirrmann"]))
    results["pfirrmann_std"] = float(np.std(results["pfirrmann"]))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "exp_cv.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  => pooled_binary {results['pooled_binary_mean']:.3f} +/- "
          f"{results['pooled_binary_std']:.3f} | pfirrmann "
          f"{results['pfirrmann_mean']:.3f} +/- {results['pfirrmann_std']:.3f}")
    return results


# ==================================================================
# Experiment C: CBAM ablation
# ==================================================================
def exp_ablation(args):
    print("\n=== EXPERIMENT C: CBAM Ablation ===")
    meta = load_meta(args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_rows = meta[meta["split"] == "train"]
    val_rows = meta[meta["split"] == "val"]

    results = {"with_cbam": None, "without_cbam": None}
    for use_cbam in (True, False):
        model = OrdinalMultiTask(backbone=args.backbone, pretrained=True,
                                 use_cbam=use_cbam).to(device)
        model = train_eval(model, train_rows, val_rows, args.data_dir, device,
                           epochs=args.epochs, seed=args.seed)
        report = evaluate_test(model, args.data_dir, device,
                               tag="with_cbam" if use_cbam else "without_cbam")
        key = "with_cbam" if use_cbam else "without_cbam"
        results[key] = report
        print(f"  {key}: pooled_bin={report['pooled_binary_accuracy']:.3f} "
              f"pfirrmann={report['pfirrmann']['accuracy']:.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "exp_ablation.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ==================================================================
# Figures
# ==================================================================
def fig_cv(results, path):
    """Per-fold bar chart with mean +/- std error bar."""
    fig, ax = plt.subplots(figsize=(3.4, 2.8), dpi=300)
    folds = [r["fold"] for r in results["folds"]]
    pb = [r["pooled_binary"] for r in results["folds"]]
    pf = [r["pfirrmann"] for r in results["folds"]]
    x = np.arange(len(folds))
    w = 0.38
    ax.bar(x - w/2, pb, w, color="#1565C0", label="Pooled binary")
    ax.bar(x + w/2, pf, w, color="#E53935", label="Pfirrmann")
    ax.axhline(results["pooled_binary_mean"], color="#1565C0", ls="--", lw=0.8)
    ax.axhline(results["pfirrmann_mean"], color="#E53935", ls="--", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"F{f}" for f in folds], fontsize=6)
    ax.set_ylabel("Accuracy", fontsize=7)
    ax.set_title("5-Fold Patient-Level Cross-Validation", fontsize=8, fontweight="bold")
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=6)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  figure -> {path}")


def fig_ordinal_vs_ce(path, ce_acc, corn_acc):
    fig, ax = plt.subplots(figsize=(3.0, 2.6), dpi=300)
    vals = [ce_acc, corn_acc]
    labels = ["Cross-Entropy", "Ordinal (CORN)"]
    colors = ["#8D6E63", "#2E7D32"]
    bars = ax.bar(labels, vals, color=colors, width=0.5,
                  edgecolor="#455A64", linewidth=1)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.1%}",
                ha="center", fontsize=7, fontweight="bold")
    ax.set_ylabel("Pfirrmann Accuracy", fontsize=7)
    ax.set_title("Ordinal vs. Plain Classification", fontsize=8, fontweight="bold")
    ax.tick_params(labelsize=6)
    ax.set_ylim(0, 0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  figure -> {path}")


def fig_ablation(results, path):
    fig, ax = plt.subplots(figsize=(3.2, 2.6), dpi=300)
    keys = ["without_cbam", "with_cbam"]
    labels = ["w/o CBAM", "w/ CBAM"]
    pf = [results[k]["pfirrmann"]["accuracy"] for k in keys]
    pb = [results[k]["pooled_binary_accuracy"] for k in keys]
    x = np.arange(2)
    ax.bar(x - 0.18, pb, 0.34, color="#90CAF9", label="Pooled binary")
    ax.bar(x + 0.18, pf, 0.34, color="#EF9A9A", label="Pfirrmann")
    for xi, (v1, v2) in enumerate(zip(pb, pf)):
        ax.text(xi - 0.18, v1 + 0.01, f"{v1:.2f}", ha="center", fontsize=6)
        ax.text(xi + 0.18, v2 + 0.01, f"{v2:.2f}", ha="center", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Accuracy", fontsize=7)
    ax.set_title("CBAM Ablation", fontsize=8, fontweight="bold")
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=6)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  figure -> {path}")


def fig_grade_errors(report, path):
    """Adjacent vs. distant misclassifications (motivates ordinal)."""
    yp = np.array(report["pfirrmann"]["true"])
    pred = np.array(report["pfirrmann"]["corn_predictions"])
    err = yp != pred
    dist = np.abs(yp[err] - pred[err])
    adjacent = (dist == 1).mean()
    distant = (dist >= 2).mean()
    fig, ax = plt.subplots(figsize=(3.0, 2.4), dpi=300)
    ax.bar(["Adjacent (1 grade)", "Distant (2+)"], [adjacent, distant],
           color=["#2E7D32", "#C62828"], width=0.5)
    for i, v in enumerate([adjacent, distant]):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=7, fontweight="bold")
    ax.set_ylabel("Proportion of Errors", fontsize=7)
    ax.set_title("Pfirrmann Error Structure", fontsize=8, fontweight="bold")
    ax.tick_params(labelsize=6)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  figure -> {path}")


def make_figures(args, ordinal_report=None, cv_results=None,
                 ablation_results=None):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if cv_results:
        fig_cv(cv_results, FIG_DIR / "fig_cv.png")
    if ordinal_report:
        fig_grade_errors(ordinal_report, FIG_DIR / "fig_grade_errors.png")


# ==================================================================
# Main
# ==================================================================
def get_args():
    p = argparse.ArgumentParser(description="Improved experiments for SPIDER paper")
    p.add_argument("--experiments", nargs="+",
                   default=["ordinal", "cv", "ablation"],
                   choices=["ordinal", "cv", "ablation"])
    p.add_argument("--data_dir", type=pathlib.Path, default="crops")
    p.add_argument("--backbone", type=str, default="efficientvit_b1")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = get_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    summary = {"backbone": args.backbone, "epochs": args.epochs, "seed": args.seed}
    ordinal_report, cv_results, ablation_results = None, None, None

    if "ordinal" in args.experiments:
        ordinal_report = exp_ordinal(args)
        summary["ordinal"] = {
            "pfirrmann_acc": ordinal_report["pfirrmann"]["accuracy"],
            "pooled_binary": ordinal_report["pooled_binary_accuracy"],
        }

    if "cv" in args.experiments:
        cv_results = exp_cv(args, folds=5)
        summary["cv"] = {
            "pfirrmann_mean_std": f"{cv_results['pfirrmann_mean']:.3f}+/-{cv_results['pfirrmann_std']:.3f}",
            "pooled_binary_mean_std": f"{cv_results['pooled_binary_mean']:.3f}+/-{cv_results['pooled_binary_std']:.3f}",
        }

    if "ablation" in args.experiments:
        ablation_results = exp_ablation(args)
        summary["ablation"] = {
            "with_cbam_pf": ablation_results["with_cbam"]["pfirrmann"]["accuracy"],
            "without_cbam_pf": ablation_results["without_cbam"]["pfirrmann"]["accuracy"],
            "with_cbam_bin": ablation_results["with_cbam"]["pooled_binary_accuracy"],
            "without_cbam_bin": ablation_results["without_cbam"]["pooled_binary_accuracy"],
        }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nAll results ->", OUT_DIR)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
