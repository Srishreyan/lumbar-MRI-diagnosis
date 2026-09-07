#!/usr/bin/env python
# evaluate_baseline.py
# ------------------------------------------------------------------
# Evaluate a saved baseline ResNet-50 checkpoint on the held-out test
# split (clean, no augmentation) and dump per-task metrics to results/.
# Mirrors the exact head slicing used in baseline_train_resnet50.py so
# numbers are comparable to the checkpoint that was actually trained.
# ------------------------------------------------------------------

from __future__ import annotations

import pathlib
import json
import argparse
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms

# Same column order as the training script's y_bin list (7 binary tasks).
BIN_COLS = [
    "Modic", "UP_endplate", "LOW_endplate", "Spondylolisthesis",
    "Disc_herniation", "Disc_narrowing", "Disc_bulging",
]


# ------------------------------------------------------------------
# Dataset (evaluation only: no augmentation)
# ------------------------------------------------------------------
class EvalCropDataset(torch.utils.data.Dataset):
    """Loads a 224x224 disc crop with channel duplication, no augmentation."""
    def __init__(self, data_dir: pathlib.Path, split: str):
        self.path = data_dir / f"crops_{split}"
        self.meta = pd.read_csv(data_dir / "labels.csv")
        self.meta = self.meta[self.meta["split"] == split].reset_index(drop=True)
        self.transform = transforms.Lambda(lambda img: img.repeat(3, 1, 1))

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.meta.iloc[idx]
        np_img = np.load((self.path / f"{row['sample_id']}.npy").resolve()).astype(np.float32)
        img_tensor = torch.from_numpy(np_img).unsqueeze(0).float()
        img_tensor = self.transform(img_tensor)
        y_bin = torch.tensor([row[col] for col in BIN_COLS], dtype=torch.float32)
        pfir = torch.tensor(row["Pfirrmann"] - 1, dtype=torch.long)
        return {"image": img_tensor, "y_bin": y_bin, "y_pfirr": pfir, "id": row["sample_id"]}


# ------------------------------------------------------------------
# Model (must match training architecture exactly)
# ------------------------------------------------------------------
def build_model(num_bin: int = 7, num_pfir: int = 5) -> nn.Module:
    backbone = models.resnet50(weights=None)
    in_features = backbone.fc.in_features
    backbone.fc = nn.Linear(in_features, num_bin + num_pfir)
    return backbone


# ------------------------------------------------------------------
# Predictions
# ------------------------------------------------------------------
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ids, y_bin_all, y_pf_all, logits_bin_all, logits_pf_all = [], [], [], [], []
    for batch in loader:
        imgs = batch["image"].to(device)
        logits = model(imgs)
        y_bin = batch["y_bin"]
        y_pf = batch["y_pfirr"]
        # Exact same slicing as training: first 7 = binary, rest = Pfirrmann
        lb = logits[:, :y_bin.size(1)]
        lp = logits[:, y_bin.size(1):]
        ids.extend(batch["id"])
        y_bin_all.append(y_bin)
        y_pf_all.append(y_pf)
        logits_bin_all.append(lb.cpu())
        logits_pf_all.append(lp.cpu())
    return {
        "id": ids,
        "y_bin": torch.cat(y_bin_all).numpy(),
        "y_pf": torch.cat(y_pf_all).numpy(),
        "logits_bin": torch.cat(logits_bin_all).numpy(),
        "logits_pf": torch.cat(logits_pf_all).numpy(),
    }


# ------------------------------------------------------------------
# Metrics (numpy-only, no sklearn dependency)
# ------------------------------------------------------------------
def accuracy(y_true, y_pred):
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())


def roc_auc(y_true, y_score):
    """ROC-AUC by pairwise comparison: P(score_pos > score_neg) + 0.5*ties."""
    yt = np.asarray(y_true).astype(int)
    ys = np.asarray(y_score)
    pos_scores = ys[yt == 1]
    neg_scores = ys[yt == 0]
    n_pos = pos_scores.size
    n_neg = neg_scores.size
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    wins = 0
    ties = 0
    for p in pos_scores:
        for q in neg_scores:
            if p > q:
                wins += 1
            elif p == q:
                ties += 1
    return float((wins + 0.5 * ties) / (n_pos * n_neg))


def confusion_matrix(y_true, y_pred, n_classes):
    yt = np.asarray(y_true).astype(int)
    yp = np.asarray(y_pred).astype(int)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(yt, yp):
        cm[t, p] += 1
    return cm


def binary_metrics(y_true, logits, cols):
    """Per-task accuracy + ROC-AUC for each binary pathology."""
    rows = []
    for i, col in enumerate(cols):
        yt = y_true[:, i]
        p = 1.0 / (1.0 + np.exp(-logits[:, i]))  # sigmoid
        pred = p > 0.5
        acc = accuracy(yt, pred)
        auc = roc_auc(yt, p)
        pos = yt.sum()
        rows.append({"task": col, "n": len(yt), "positive": int(pos),
                     "pos_rate": pos / len(yt), "accuracy": acc, "auc": auc})
    return pd.DataFrame(rows)


def pfirrmann_metrics(y_true, logits):
    pred = logits.argmax(axis=1)
    acc = accuracy(y_true, pred)
    cm = confusion_matrix(y_true, pred, logits.shape[1])
    return {"accuracy": acc, "confusion_matrix": cm.tolist()}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate baseline ResNet-50 checkpoint on test split")
    ap.add_argument("--ckpt", type=pathlib.Path, default="ckpt.pth", help="path to model checkpoint")
    ap.add_argument("--data_dir", type=pathlib.Path, default="crops")
    ap.add_argument("--split", type=str, default="test",
                    help="split to evaluate: test (default), val, or train")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out", type=pathlib.Path, default="results")
    ap.add_argument("--tag", type=str, default="baseline",
                    help="label for output files e.g. 'frozen', 'finetuned', 'finetuned_auto'")
    ap.add_argument("--arch", type=str, default="resnet50",
                    help="model type: 'resnet50' or an EfficientViT timm name "
                         "(e.g. 'efficientvit_b1')")
    args = ap.parse_args()

    device = torch.device("cuda") if (args.device == "auto" and torch.cuda.is_available()) else torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    # Model + checkpoint
    if args.arch == "resnet50":
        model = build_model(num_bin=7, num_pfir=5).to(device)
    else:
        from efficientvit_cbam import build_model as ev_build
        model = ev_build(backbone=args.arch, pretrained=False).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint {args.ckpt} on {device}")

    ds = EvalCropDataset(args.data_dir, args.split)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Evaluating {len(ds)} samples ({args.split} split), no augmentation")

    res = predict(model, loader, device)

    # Binary tasks
    df_bin = binary_metrics(res["y_bin"], res["logits_bin"], BIN_COLS)
    print("\n--- Binary pathology tasks ---")
    print(df_bin.to_string(index=False))

    # Pfirrmann
    pf = pfirrmann_metrics(res["y_pf"], res["logits_pf"])
    print("\n--- Pfirrmann (5-grade) ---")
    print(f"Accuracy: {pf['accuracy']*100:.2f}%")
    print("Confusion matrix (rows=true, cols=pred):")
    print(np.array(pf["confusion_matrix"]))

    # Aggregate binary accuracy (all labels pooled)
    all_bin_acc = accuracy(res["y_bin"].ravel(),
                                 (1.0 / (1.0 + np.exp(-res["logits_bin"])) > 0.5).astype(int).ravel())
    print(f"\nPooled binary accuracy: {all_bin_acc*100:.2f}%")

    # Save outputs
    tag = f"{args.split}_{args.tag}"
    df_bin.to_csv(args.out / f"binary_{tag}.csv", index=False)
    summary = {
        "checkpoint": str(args.ckpt),
        "split": args.split,
        "n_samples": len(ds),
        "binary_tasks": df_bin.to_dict(orient="records"),
        "pooled_binary_accuracy": float(all_bin_acc),
        "pfirrmann_accuracy": float(pf["accuracy"]),
        "pfirrmann_confusion_matrix": pf["confusion_matrix"],
    }
    with open(args.out / f"summary_{tag}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved -> {args.out / ('binary_' + tag + '.csv')}")
    print(f"Saved -> {args.out / ('summary_' + tag + '.json')}")


if __name__ == "__main__":
    main()
