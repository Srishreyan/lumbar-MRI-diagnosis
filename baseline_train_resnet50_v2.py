#!/usr/bin/env python
# baseline_train_resnet50.py
# ------------------------------------------------------------
# Minimal training script for the SPIDER disc‑crop baseline.
# Loads .npy disc patches, applies augmentation, and fine‑tunes
# a ResNet‑50 head for the 8 binary pathology tasks plus the
# 5‑way Pfirrmann grading.
# ------------------------------------------------------------

from __future__ import annotations

import pathlib
import argparse
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms

# Albumentations for augmentation
from albumentations import Compose as AlbCompose
from albumentations import HorizontalFlip, VerticalFlip, Rotate
from albumentations import RandomBrightnessContrast, ShiftScaleRotate


# ------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Baseline ResNet‑50 training for SPIDER disc‑crops")
    parser.add_argument("--data_dir", type=pathlib.Path, default="crops",
                        help="directory containing crops/* and labels.csv")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size for training")
    parser.add_argument("--epochs", type=int, default=40,
                        help="number of training epochs")
    parser.add_argument("--device", type=str, default="auto",
                        help="torch device to use; auto means cuda if available, else cpu")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate")
    parser.add_argument("--scheduler", type=str, default="none",
                        help="learning‑rate scheduler: none, step, cosine")
    return parser.parse_args()


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
class DiscCropDataset(torch.utils.data.Dataset):
    """Loads a 224×224 disc crop as a torch.FloatTensor and the 8+5 target values."""
    def __init__(self, data_dir: pathlib.Path, split: str, transform=None):
        self.split = split
        self.path = data_dir / f"crops_{split}"
        self.meta = pd.read_csv(data_dir / "labels.csv")
        self.meta = self.meta[self.meta["split"] == split].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.meta.iloc[idx]
        np_img = np.load((self.path / f"{row['sample_id']}.npy").resolve())
        img_tensor = torch.from_numpy(np_img).unsqueeze(0).float()  # [1, H, W]
        if self.transform:
            img_tensor = self.transform(img_tensor)
        y_bin = torch.tensor([row[col] for col in [
            "Modic", "UP_endplate", "LOW_endplate", "Spondylolisthesis",
            "Disc_herniation", "Disc_narrowing", "Disc_bulging"
        ]], dtype=torch.float32)
        pfir = torch.tensor(row["Pfirrmann"], dtype=torch.long)
        return {"image": img_tensor, "y_bin": y_bin, "y_pfirr": pfir}


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
def build_model(num_bin: int = 8, num_pfir: int = 5, pretrained: bool = False) -> nn.Module:
    """Create ResNet‑50 backbone and attach a multi‑task head."""
    backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
    in_features = backbone.fc.in_features
    backbone.fc = nn.Linear(in_features, num_bin + num_pfir)
    return backbone


# ------------------------------------------------------------------
# Training & evaluation helpers
# ------------------------------------------------------------------
def train_one_epoch(model, loader, criterion_bin, criterion_pf, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        imgs = batch["image"].to(device)
        y_bin = batch["y_bin"].to(device)
        y_pf  = batch["y_pfirr"].to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        logits_bin = logits[:, :y_bin.size(1)]
        logits_pf  = logits[:, y_bin.size(1):]
        loss_bin = criterion_bin(logits_bin, y_bin)
        loss_pf  = criterion_pf(logits_pf, y_pf)
        loss = loss_bin + loss_pf

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion_bin, criterion_pf, device):
    model.eval()
    loss_sum = 0.0
    correct_bin = 0
    total_bin = 0
    correct_pf = 0
    total_pf = 0

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            y_bin = batch["y_bin"].to(device)
            y_pf  = batch["y_pfirr"].to(device)

            logits = model(imgs)
            logits_bin = logits[:, :y_bin.size(1)]
            logits_pf  = logits[:, y_bin.size(1):]

            loss_bin = criterion_bin(logits_bin, y_bin)
            loss_pf  = criterion_pf(logits_pf, y_pf)
            loss_sum += (loss_bin + loss_pf).item() * imgs.size(0)

            preds_bin = torch.sigmoid(logits_bin) > 0.5
            correct_bin += preds_bin.eq(y_bin.byte()).sum().item()
            total_bin += y_bin.numel()

            preds_pf = logits_pf.argmax(dim=1)
            correct_pf += preds_pf.eq(y_pf).sum().item()
            total_pf += y_pf.size(0)

    avg_loss = loss_sum / len(loader.dataset)
    return avg_loss, correct_bin / total_bin, correct_pf / total_pf


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    args = get_args()

    # Device selection
    device = torch.device("cuda") if (args.device == "auto" and torch.cuda.is_available()) else torch.device(args.device)

    # Augmentation pipeline
    transform = transforms.Compose([
        AlbCompose([
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            Rotate(limit=15, p=0.5),
            RandomBrightnessContrast(p=0.5),
            ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=0, p=0.5)
        ], p=1.0),
        transforms.Lambda(lambda img: img.repeat(3, 1, 1)),  # duplicate gray channel to 3
    ])

    # Datasets & loaders
    train_ds = DiscCropDataset(args.data_dir, "train", transform=transform)
    val_ds   = DiscCropDataset(args.data_dir, "val",   transform=transform)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                              shuffle=True, num_workers=0)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=args.batch_size,
                                              shuffle=False, num_workers=0)

    # Model, losses, optimizer
    model = build_model(num_bin=8, num_pfir=5, pretrained=False).to(device)
    criterion_bin = nn.BCEWithLogitsLoss()
    criterion_pf  = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # Scheduler (if requested)
    scheduler = None
    if args.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Training on {device}, epochs={args.epochs}")

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion_bin, criterion_pf, optimizer, device)
        val_loss, val_acc_bin, val_acc_pf = evaluate(model, val_loader, criterion_bin, criterion_pf, device)
        if scheduler:
            scheduler.step()

        print(f"Epoch {epoch:02d} | train loss={train_loss:.4f} | val loss={val_loss:.4f} | "
              f"bin acc={val_acc_bin*100:.1f}% | pf acc={val_acc_pf*100:.1f}%")

    # Smoke test – only executed if we ran a single‑epoch run
    if args.epochs == 1:
        smk_loader = torch.utils.data.DataLoader(train_ds, batch_size=5, shuffle=True, num_workers=0)
        batch = next(iter(smk_loader))
        with torch.no_grad():
            logits = model(batch["image"].to(device))
            print("\n[Smoke] Sample logits shape:", logits.shape)


if __name__ == "__main__":
    main()
