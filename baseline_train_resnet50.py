#!/usr/bin/env python
# baseline_train_resnet50.py
# ------------------------------------------------------------
# Minimal training script for the SPIDER disc‑crop baseline.
# Loads .npy disc patches, applies augmentation, and fine‑tunes
# a pretrained ResNet‑50 backbone for the 7 binary pathology tasks
# + 5‑way Pfirrmann (12‑way multi‑task head).
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
from albumentations import RandomBrightnessContrast, Affine

# Binary pathology columns in labels.csv (order must match y_bin construction).
BIN_COLS = ["Modic", "UP_endplate", "LOW_endplate", "Spondylolisthesis",
            "Disc_herniation", "Disc_narrowing", "Disc_bulging"]


def auto_pos_weight(meta: pd.DataFrame) -> torch.Tensor:
    """Per-task BCE positive weight = n_neg/n_pos from labels, clipped to [1, 50].

    Rare classes (very few positives) get a large weight so the model is forced
    to actually predict them instead of always saying "absent".
    """
    weights = []
    for col in BIN_COLS:
        pos = int((meta[col] == 1).sum())
        neg = int((meta[col] == 0).sum())
        w = neg / pos if pos > 0 else 50.0
        weights.append(float(np.clip(w, 1.0, 50.0)))
    return torch.tensor(weights, dtype=torch.float32)


# ------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Baseline ResNet‑50 training for SPIDER disc‑crops")
    parser.add_argument("--data_dir", type=pathlib.Path, default="crops",
                        help="directory containing crops/* and labels.csv")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size for training")
    parser.add_argument("--epochs", type=int, default=40, help="number of training epochs")
    parser.add_argument("--device", type=str, default="auto",
                        help="torch device to use; auto means cuda if available, else cpu")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--scheduler", type=str, default="none",
                        help="learning‑rate scheduler: none, step, cosine")
    parser.add_argument("--output", type=pathlib.Path, default=None, help="path to save model state")
    parser.add_argument("--unfreeze", action="store_true",
                        help="fine-tune the whole backbone end-to-end (default: frozen backbone, train head only)")
    parser.add_argument("--pos_weight", type=str, default="1.0",
                        help="BCE positive-class weight: 'auto' = per-task n_neg/n_pos computed from "
                             "train labels (principled fix for rare classes), or a single scalar >=1 "
                             "applied uniformly to all 7 binary tasks")
    return parser.parse_args()


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
class DiscCropDataset(torch.utils.data.Dataset):
    """Loads a 224×224 disc crop as a torch.FloatTensor and the 7+5 target values."""
    def __init__(self, data_dir: pathlib.Path, split: str, transform=None, apply_aug: bool = True):
        self.split = split
        self.path = data_dir / f"crops_{split}"
        self.meta = pd.read_csv(data_dir / "labels.csv")
        self.meta = self.meta[self.meta["split"] == split].reset_index(drop=True)
        self.transform = transform
        # Albumentations pipeline (applied on a numpy array) – kept for train only,
        # so validation/test numbers are measured on clean crops.
        self.alb = AlbCompose([
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            Rotate(limit=15, p=0.5),
            RandomBrightnessContrast(p=0.5),
            Affine(translate_percent=0.05, scale=0.05, p=0.5),
        ])
        self.apply_aug = apply_aug

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.meta.iloc[idx]
        # Load the .npy patch as np.ndarray
        np_img = np.load((self.path / f"{row['sample_id']}.npy").resolve()).astype(np.float32)
        if self.apply_aug:
            np_img = self.alb(image=np_img)["image"]
        img_tensor = torch.from_numpy(np_img).unsqueeze(0).float()  # [1,H,W]
        if self.transform:
            img_tensor = self.transform(img_tensor)
        # Binary labels
        y_bin = torch.tensor([row[col] for col in [
            "Modic", "UP_endplate", "LOW_endplate", "Spondylolisthesis",
            "Disc_herniation", "Disc_narrowing", "Disc_bulging"
        ]], dtype=torch.float32)
        # Pfirrmann is stored 1-5; CE loss expects 0-4.
        pfir = torch.tensor(row["Pfirrmann"] - 1, dtype=torch.long)
        return {"image": img_tensor, "y_bin": y_bin, "y_pfirr": pfir}


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
def build_model(num_bin: int = 7, num_pfir: int = 5, pretrained: bool = True,
                freeze: bool = True) -> nn.Module:
    """Create a pretrained ResNet‑50 backbone and attach a multi‑task head.

    `num_bin` is the true number of binary pathology columns in labels.csv (7).
    When `freeze` is True the pretrained backbone weights are frozen so only
    the (randomly initialized) head is trained — standard transfer learning.
    """
    backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
    in_features = backbone.fc.in_features
    backbone.fc = nn.Linear(in_features, num_bin + num_pfir)
    if freeze:
        for name, param in backbone.named_parameters():
            if name.startswith("fc"):
                continue  # keep the new head trainable
            param.requires_grad = False
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
        y_pf = batch["y_pfirr"].to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        logits_bin = logits[:, :y_bin.size(1)]
        logits_pf = logits[:, y_bin.size(1):]
        loss_bin = criterion_bin(logits_bin, y_bin)
        loss_pf = criterion_pf(logits_pf, y_pf)
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
            y_pf = batch["y_pfirr"].to(device)

            logits = model(imgs)
            logits_bin = logits[:, :y_bin.size(1)]
            logits_pf = logits[:, y_bin.size(1):]

            loss_bin = criterion_bin(logits_bin, y_bin)
            loss_pf = criterion_pf(logits_pf, y_pf)
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

    # Channel‑duplication (after augmentation) – <1 MB transfer
    transform = transforms.Lambda(lambda img: img.repeat(3, 1, 1))

    # Datasets & loaders (val is evaluated on clean crops, no augmentation)
    train_ds = DiscCropDataset(args.data_dir, "train", transform=transform, apply_aug=True)
    val_ds   = DiscCropDataset(args.data_dir, "val",   transform=transform, apply_aug=False)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                              shuffle=True, num_workers=0)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=args.batch_size,
                                              shuffle=False, num_workers=0)

    # Model, losses, optimizer (frozen backbone by default; --unfreeze trains end-to-end)
    model = build_model(num_bin=7, num_pfir=5, pretrained=True, freeze=not args.unfreeze).to(device)
    # BCE positive weights: "auto" -> per-task n_neg/n_pos from train labels; else a scalar
    pw_input = args.pos_weight.strip().lower()
    if pw_input == "auto":
        pos_weight = auto_pos_weight(train_ds.meta).to(device)
        pw_display = "auto (per-task n_neg/n_pos): " + ",".join(f"{w:.2f}" for w in pos_weight.cpu().tolist())
    else:
        pos_weight = torch.full((7,), float(pw_input), device=device)
        pw_display = pw_input
    criterion_bin = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_pf  = nn.CrossEntropyLoss()
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    def n_trainable():
        return sum(p.requires_grad for p in model.parameters())

    # Scheduler (if requested)
    scheduler = None
    if args.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Training on {device}, epochs={args.epochs}")
    mode = "Full fine-tune (unfrozen)" if args.unfreeze else "Frozen backbone (head only)"
    print(f"[mode] {mode} | pos_weight={pw_display} | "
          f"trainable layers={n_trainable()}/{sum(1 for _ in model.parameters())}")

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion_bin, criterion_pf, optimizer, device)
        val_loss, val_acc_bin, val_acc_pf = evaluate(model, val_loader, criterion_bin, criterion_pf, device)
        if scheduler:
            scheduler.step()

        print(f"Epoch {epoch:02d} | train loss={train_loss:.4f} | val loss={val_loss:.4f} | "
              f"bin acc={val_acc_bin*100:.1f}% | pf acc={val_acc_pf*100:.1f}%")

    # Save the model checkpoint if requested
    if args.output:
        torch.save(model.state_dict(), args.output)
        print(f"Checkpoint saved to {args.output}")

    # Optional smoke test – only if run for 1 epoch
    if args.epochs == 1:
        smk_loader = torch.utils.data.DataLoader(train_ds, batch_size=5, shuffle=True, num_workers=0)
        batch = next(iter(smk_loader))
        with torch.no_grad():
            logits = model(batch["image"].to(device))
            print("\n[Smoke] Sample logits shape:", logits.shape)


if __name__ == "__main__":
    main()
