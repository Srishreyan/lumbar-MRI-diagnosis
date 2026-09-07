#!/usr/bin/env python
# train_efficientvit.py
# ------------------------------------------------------------------
# Multi-task training for the EfficientViT + CBAM baseline on SPIDER.
# Mirrors the ResNet-50 baseline data pipeline exactly (Modic-binarized
# labels, Pfirrmann 1->0 indexing, no-aug validation) so numbers are
# directly comparable. Adds MC-Dropout uncertainty support.
# ------------------------------------------------------------------
from __future__ import annotations

import pathlib
import argparse
import random
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from albumentations import Compose as AlbCompose
from albumentations import HorizontalFlip, VerticalFlip, Rotate
from albumentations import RandomBrightnessContrast, Affine

from efficientvit_cbam import build_model

BIN_COLS = ["Modic", "UP_endplate", "LOW_endplate", "Spondylolisthesis",
            "Disc_herniation", "Disc_narrowing", "Disc_bulging"]


def auto_pos_weight(meta: pd.DataFrame) -> torch.Tensor:
    weights = []
    for col in BIN_COLS:
        pos = int((meta[col] == 1).sum())
        neg = int((meta[col] == 0).sum())
        w = neg / pos if pos > 0 else 50.0
        weights.append(float(np.clip(w, 1.0, 50.0)))
    return torch.tensor(weights, dtype=torch.float32)


def get_args():
    p = argparse.ArgumentParser(description="EfficientViT + CBAM multi-task training (SPIDER)")
    p.add_argument("--data_dir", type=pathlib.Path, default="crops")
    p.add_argument("--backbone", type=str, default="efficientvit_b1")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--scheduler", type=str, default="cosine",
                   help="none, step, cosine")
    p.add_argument("--freeze", action="store_true",
                   help="freeze backbone (train CBAM+head only)")
    p.add_argument("--pos_weight", type=str, default="auto",
                   help="'auto' per-task n_neg/n_pos, or a scalar")
    p.add_argument("--mc_dropout", type=float, default=0.0,
                   help="MC-Dropout rate in head (>0 enables uncertainty; dropouts "
                        "stay ACTIVE during eval for stochastic forward passes)")
    p.add_argument("--backbone_mc_dropout", type=float, default=0.0,
                   help="extra MC-Dropout (Dropout2d) on the backbone feature map")
    p.add_argument("--output", type=pathlib.Path, default=None)
    p.add_argument("--seed", type=int, default=42,
                   help="random seed for reproducibility")
    return p.parse_args()


class DiscCropDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: pathlib.Path, split: str, apply_aug: bool = True):
        self.split = split
        self.path = data_dir / f"crops_{split}"
        self.meta = pd.read_csv(data_dir / "labels.csv")
        self.meta = self.meta[self.meta["split"] == split].reset_index(drop=True)
        self.apply_aug = apply_aug
        self.alb = AlbCompose([
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            Rotate(limit=15, p=0.5),
            RandomBrightnessContrast(p=0.5),
            Affine(translate_percent=0.05, scale=0.05, p=0.5),
        ])

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.meta.iloc[idx]
        np_img = np.load((self.path / f"{row['sample_id']}.npy").resolve()).astype(np.float32)
        if self.apply_aug:
            np_img = self.alb(image=np_img)["image"]
        img = torch.from_numpy(np_img).unsqueeze(0).repeat(3, 1, 1).float()  # [3,H,W]
        y_bin = torch.tensor([row[col] for col in BIN_COLS], dtype=torch.float32)
        pfir = torch.tensor(row["Pfirrmann"] - 1, dtype=torch.long)
        return {"image": img, "y_bin": y_bin, "y_pfirr": pfir}


def train_one_epoch(model, loader, crit_bin, crit_pf, opt, device):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        img = batch["image"].to(device)
        y_bin = batch["y_bin"].to(device)
        y_pf = batch["y_pfirr"].to(device)
        opt.zero_grad()
        logits = model(img)
        # head outputs [B, 7+5]; split for the two losses
        lb, lp = logits[:, :y_bin.size(1)], logits[:, y_bin.size(1):]
        loss = crit_bin(lb, y_bin) + crit_pf(lp, y_pf)
        loss.backward()
        opt.step()
        total += loss.item() * img.size(0)
        n += img.size(0)
    return total / n


def evaluate(model, loader, crit_bin, crit_pf, device):
    model.eval()
    loss_sum, n = 0.0, 0
    cb, tb, cp, tp = 0, 0, 0, 0
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            y_bin = batch["y_bin"].to(device)
            y_pf = batch["y_pfirr"].to(device)
            logits = model(img)
            lb, lp = logits[:, :y_bin.size(1)], logits[:, y_bin.size(1):]
            loss_sum += (crit_bin(lb, y_bin) + crit_pf(lp, y_pf)).item() * img.size(0)
            n += img.size(0)
            pb = (torch.sigmoid(lb) > 0.5)
            cb += pb.eq(y_bin.byte()).sum().item(); tb += y_bin.numel()
            cp += (lp.argmax(1) == y_pf).sum().item(); tp += y_pf.size(0)
    return loss_sum / n, cb / tb, cp / tp


def main():
    args = get_args()
    device = torch.device("cuda") if (args.device == "auto" and torch.cuda.is_available()) else torch.device(args.device)

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Seed: {args.seed}")

    train_ds = DiscCropDataset(args.data_dir, "train", apply_aug=True)
    val_ds = DiscCropDataset(args.data_dir, "val", apply_aug=False)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(backbone=args.backbone, pretrained=True,
                        freeze=args.freeze, mc_dropout_p=args.mc_dropout,
                        backbone_mc_dropout_p=args.backbone_mc_dropout).to(device)

    pw_input = args.pos_weight.strip().lower()
    if pw_input == "auto":
        pw = auto_pos_weight(train_ds.meta).to(device)
    else:
        pw = torch.full((7,), float(pw_input), device=device)
    crit_bin = nn.BCEWithLogitsLoss(pos_weight=pw)
    crit_pf = nn.CrossEntropyLoss()
    opt = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    if args.scheduler == "cosine":
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    elif args.scheduler == "step":
        sched = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)
    else:
        sched = None

    n_tr = sum(p.requires_grad for p in model.parameters())
    n_tot = sum(1 for _ in model.parameters())
    print(f"Training on {device}, epochs={args.epochs}")
    print(f"[model] {args.backbone} + CBAM | {'frozen' if args.freeze else 'unfrozen'} | "
          f"mc_dropout={args.mc_dropout} | trainable={n_tr}/{n_tot}")

    for epoch in range(1, args.epochs + 1):
        tl = train_one_epoch(model, train_loader, crit_bin, crit_pf, opt, device)
        vl, ba, pa = evaluate(model, val_loader, crit_bin, crit_pf, device)
        if sched:
            sched.step()
        print(f"Epoch {epoch:02d} | train={tl:.4f} | val={vl:.4f} | bin acc={ba*100:.1f}% | pf acc={pa*100:.1f}%")

    if args.output:
        torch.save(model.state_dict(), args.output)
        print(f"Checkpoint saved to {args.output}")


if __name__ == "__main__":
    main()