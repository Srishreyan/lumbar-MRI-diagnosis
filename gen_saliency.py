#!/usr/bin/env python
# gen_saliency.py
# ------------------------------------------------------------------
# Generate the four CAM saliency maps (Grad-CAM, Grad-CAM++, Eigen-CAM,
# Score-CAM) on selected test samples from the SPIDER crops, save overlaid
# PNGs, and compute pairwise/mean agreement between the four methods.
#
# Target selection: for each image we explain (a) the predicted Pfirrmann
# grade (argmax over 7..11) and (b) the highest-scoring predicted binary
# pathology (max sigmoid among the 7, if > 0.5). Maps land in saliency_vis/.
# ------------------------------------------------------------------
from __future__ import annotations

import pathlib
import json
import argparse

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate_baseline import EvalCropDataset, BIN_COLS
from efficientvit_cbam import build_model
from cam_methods import compute_map, METHODS

PFIR_NAMES = ["Pfirrmann G1", "G2", "G3", "G4", "G5"]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of binarized maps at the 50th percentile of each map."""
    a_b = a >= np.median(a)
    b_b = b >= np.median(b)
    inter = np.logical_and(a_b, b_b).sum()
    union = np.logical_or(a_b, b_b).sum()
    return float(inter / union) if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=pathlib.Path, default="ckpt_evit_b1.pth")
    ap.add_argument("--arch", type=str, default="efficientvit_b1")
    ap.add_argument("--data_dir", type=pathlib.Path, default="crops")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--target_layer", type=str, default="cbam",
                    help="'cbam' or 'encoder' (last conv feature)")
    ap.add_argument("--out", type=pathlib.Path, default="saliency_vis")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--scorecam_max_channels", type=int, default=64,
                    help="subsample channels for Score-CAM to bound cost")
    args = ap.parse_args()

    device = torch.device("cuda") if (args.device == "auto" and torch.cuda.is_available()) else torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    model = build_model(backbone=args.arch, pretrained=False).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # Pick the target conv layer (CBAM output, [B,256,7,7])
    if args.target_layer == "cbam":
        target_conv = model.cbam
    else:
        target_conv = model.encoder
    print(f"Backbone={args.arch} | target_layer={args.target_layer} | device={device}")

    # Load full relevant meta once to label images & stratify
    meta = pd.read_csv(args.data_dir / "labels.csv")
    meta_test = meta[meta["split"] == args.split].reset_index(drop=True)

    # Choose samples: prefer rare-positive cases, then first N
    rare = meta_test[(meta_test["Spondylolisthesis"] == 1) |
                     (meta_test["Disc_herniation"] == 1)]
    picked = list(rare.index[: args.samples])
    for i in range(args.samples):
        if len(picked) >= args.samples:
            pick = rare.index[i]
            picked.append(int(pick))
    picked = list(dict.fromkeys(picked))[: args.samples]

    # Dataset for clean image loading
    ds = EvalCropDataset(args.data_dir, args.split)

    report = {"sample": [], "method_pair_iou": {}, "method_vs_mean_iou": {}}
    pair_keys = [f"{a}-{b}" for i, a in enumerate(METHODS) for b in list(METHODS)[i+1:]]

    for si, idx in enumerate(picked):
        item = ds[idx]
        img = item["image"].unsqueeze(0).to(device)     # [1,3,224,224]
        row = ds.meta.iloc[idx]
        sid = row["sample_id"]
        y_pf = int(ds.meta.iloc[idx]["Pfirrmann"])      # 1-based for display
        with torch.no_grad():
            logits = model(img)[0].cpu().numpy()        # [12]
        pf_logits = logits[7:]
        pred_pf = int(pf_logits.argmax()) + 1           # 1-based pred
        bin_probs = 1 / (1 + np.exp(-logits[:7]))
        pred_bin_idx = int(bin_probs.argmax())
        pred_bin_name = BIN_COLS[pred_bin_idx]
        pred_bin_score = float(bin_probs[pred_bin_idx])
        truth_list = [c for c, v in zip(BIN_COLS, row[BIN_COLS]) if v == 1]

        # Which output to explain: predicted Pfirrmann grade (the diagnosis)
        target = 7 + (pred_pf - 1)

        maps = {}
        for m in METHODS:
            maps[m] = compute_map(model, target_conv, img, target, m,
                                  spatial=224, max_channels=args.scorecam_max_channels)[0]
        maps = {m: mm.astype(np.float32) for m, mm in maps.items()}

        # ---- Save overlaid PNG for this sample/method ---------------------
        gray = img[0, 0].cpu().numpy()
        g = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
        fig, axes = plt.subplots(1, len(METHODS) + 1, figsize=(4.6 * (len(METHODS) + 1), 4.6))
        axes[0].imshow(g, cmap="gray"); axes[0].set_title("MRI", fontsize=10)
        axes[0].axis("off")
        for ax, (m, mm) in zip(axes[1:], maps.items()):
            ax.imshow(g, cmap="gray")
            ax.imshow(mm, cmap="jet", alpha=0.55)
            ax.set_title(m, fontsize=10)
            ax.axis("off")
        fig.suptitle(f"{sid}\nPfirrmann pred {pred_pf} (GT {y_pf}) | top path. "
                     f"{pred_bin_name} ({pred_bin_score:.2f}) | labeled: {truth_list or 'none'}",
                     fontsize=8)
        fig.tight_layout()
        fpath = args.out / f"saliency_{si:02d}_{sid}.png"
        fig.savefig(fpath, dpi=130)
        plt.close(fig)

        # ---- Agreement metrics --------------------------------------------
        ms = list(METHODS.keys())
        for a in ms:
            for b in ms[ms.index(a) + 1:]:
                iou_ab = iou(maps[a], maps[b])
                report["method_pair_iou"].setdefault(f"{a}-{b}", []).append(iou_ab)
        for a in ms:
            others = np.stack([maps[b] for b in ms if b != a]).mean(0)
            report["method_vs_mean_iou"].setdefault(a, []).append(iou(maps[a], others))
        report["sample"].append({
            "sample_id": sid, "idx": int(idx),
            "gt_pfirrmann": int(y_pf), "pred_pfirrmann": pred_pf,
            "gt_binary": truth_list, "top_pred_binary": pred_bin_name,
            "target_logit": int(target),
            "png": str(fpath),
        })
        print(f"[{si+1}/{len(picked)}] {sid}  GT pf={y_pf} pred pf={pred_pf} "
              f"top path={pred_bin_name}({pred_bin_score:.2f}) -> {fpath.name}")

    # Aggregate agreement
    print("\n--- Mean pairwise IoU (binarized @ median) ---")
    agg = {"samples": len(report["sample"])}
    for k, v in report["method_pair_iou"].items():
        agg[k] = round(float(np.mean(v)), 3)
        print(f"  {k:20s} {float(np.mean(v)):.3f}")
    print("--- Each method vs. mean of the other three ---")
    for k, v in report["method_vs_mean_iou"].items():
        agg[f"vs_mean_{k}"] = round(float(np.mean(v)), 3)
        print(f"  {k:20s} {float(np.mean(v)):.3f}")
    report["aggregate"] = agg
    with open(args.out / "saliency_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report -> {args.out / 'saliency_report.json'}")


if __name__ == "__main__":
    main()