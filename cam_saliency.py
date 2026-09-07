#!/usr/bin/env python
# cam_saliency.py
# ------------------------------------------------------------------
# Class Activation Mapping (CAM) saliency for the SPIDER multi-task model.
#
# Implements 4 CAM variants:
#   1. Grad-CAM (Selvaraju et al., ICCV 2017)
#   2. Grad-CAM++ (Chattopadhay et al., WACV 2018)
#   3. Eigen-CAM (Chalas et al., 2020) — first principal component of activations
#   4. Score-CAM (Wang et al., CVPR Workshop 2020) — gradient-free
#
# All methods produce a heatmap over the spatial feature map that is
# upsampled to the input resolution and overlaid on the image.
#
# For the multi-task model, we generate separate heatmaps for:
#   - Each of the 7 binary pathologies (per-class gradient w.r.t. sigmoid)
#   - Pfirrmann (gradient w.r.t. predicted class softmax)
#
# Usage:
#   python cam_saliency.py --ckpt ckpt_evit_b1.pth --arch efficientvit_b1
#       --data_dir crops --split test --n_samples 20 --out results/cam
# ------------------------------------------------------------------
from __future__ import annotations

import pathlib
import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluate_baseline import EvalCropDataset, BIN_COLS
from efficientvit_cbam import build_model


# ------------------------------------------------------------------
# Hook for capturing intermediate activations and gradients
# ------------------------------------------------------------------
class FeatureExtractor:
    """Captures activations and gradients from a specific layer.

    Usage:
        fe = FeatureExtractor(model, target_layer)
        out = model(x)
        acts = fe.activations  # [B, C, H, W]
        grads = fe.gradients   # [B, C, H, W] (only if retain_grad=True)
        fe.remove()  # cleanup hooks
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module, retain_grad: bool = False):
        self.activations = None
        self.gradients = None
        self._retain_grad = retain_grad

        def forward_hook(module, input, output):
            self.activations = output.detach()
            if retain_grad and output.requires_grad:
                output.register_hook(lambda grad: setattr(self, 'gradients', grad.detach()))

        self._handle = target_layer.register_forward_hook(forward_hook)

    def remove(self):
        self._handle.remove()


# ------------------------------------------------------------------
# CAM implementations
# ------------------------------------------------------------------
def grad_cam(fe: FeatureExtractor, target_class: int,
             activations: np.ndarray, gradients: np.ndarray) -> np.ndarray:
    """Grad-CAM: weight feature maps by average gradient.

    Args:
        fe: feature extractor (already captured activations/gradients)
        target_class: index of the class to explain
        activations: [C, H, W] feature map
        gradients: [C, H, W] gradient w.r.t. target class

    Returns:
        heatmap [H, W] (will be upsampled later)
    """
    # Average gradient per channel (global average pooling of gradients)
    weights = gradients.mean(axis=(1, 2), keepdims=True)  # [C, 1, 1]
    # Weighted combination of feature maps
    cam = (weights * activations).sum(axis=0)  # [H, W]
    # ReLU (only positive contributions)
    cam = np.maximum(cam, 0)
    return cam


def grad_cam_plus_plus(fe: FeatureExtractor, target_class: int,
                       activations: np.ndarray, gradients: np.ndarray) -> np.ndarray:
    """Grad-CAM++: improved gradient weighting for multi-label settings.

    Uses second-order gradients for better localization.
    """
    # First-order gradients
    grad_2 = gradients ** 2
    grad_3 = gradients ** 3

    # Denominator for Grad-CAM++ weighting
    denom = 2 * grad_2 + (activations * grad_3).sum(axis=(1, 2), keepdims=True)
    denom = np.where(np.abs(denom) < 1e-8, 1e-8, denom)

    # Alpha coefficients
    alpha = grad_2 / denom  # [C, 1, 1]

    # Weighted combination
    cam = (alpha * np.maximum(gradients, 0) * activations).sum(axis=0)
    cam = np.maximum(cam, 0)
    return cam


def eigen_cam(activations: np.ndarray) -> np.ndarray:
    """Eigen-CAM: first principal component of the activation tensor.

    Gradient-free — captures the most variance in the feature map.
    """
    # Reshape [C, H, W] -> [C, H*W]
    C, H, W = activations.shape
    flat = activations.reshape(C, -1)  # [C, H*W]

    # SVD (more stable than eigendecomposition)
    # We want the first right singular vector (direction of max variance)
    # Use truncated SVD for efficiency
    from numpy.linalg import svd
    U, S, Vt = svd(flat, full_matrices=False)

    # First principal component
    pc = Vt[0]  # [H*W]

    # Reshape to heatmap
    cam = pc.reshape(H, W)
    cam = np.maximum(cam, 0)
    return cam


def score_cam(model: nn.Module, feature_extractor: FeatureExtractor,
              input_tensor: torch.Tensor, target_class: int,
              activations: np.ndarray, n_samples: int = 10) -> np.ndarray:
    """Score-CAM: gradient-free spatial masking of feature maps.

    Creates spatial Gaussian masks at random locations, applies them to
    the feature map, and measures how the target class score changes.
    This is a practical approximation that runs in O(n_samples) forward passes.
    """
    model.eval()
    C, H, W = activations.shape
    cam = np.zeros((H, W), dtype=np.float32)
    device = next(model.parameters()).device

    # Random spatial locations for masks
    np.random.seed(42)
    mask_indices = np.random.choice(H * W, size=min(n_samples, H * W), replace=False)
    sigma = max(H, W) / 4.0
    y_grid, x_grid = np.mgrid[0:H, 0:W]

    for idx in mask_indices:
        h, w = divmod(idx, W)
        mask = np.exp(-((y_grid - h)**2 + (x_grid - w)**2) / (2 * sigma**2))

        # Apply mask to each channel
        masked_acts = activations * mask[np.newaxis, :, :]
        masked_tensor = torch.from_numpy(masked_acts).unsqueeze(0).float().to(device)

        # Run through CBAM + pool + head
        feat = model.cbam(masked_tensor)
        feat = model.pool(feat).flatten(1)
        logits_bin, logits_pfir = model.head(feat)

        if target_class < 7:
            score = torch.sigmoid(logits_bin[0, target_class]).item()
        else:
            score = torch.softmax(logits_pfir[0], dim=0)[target_class - 7].item()

        cam += score * mask

    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam /= cam.max()
    return cam


# ------------------------------------------------------------------
# CAM pipeline
# ------------------------------------------------------------------
def get_target_layer(model):
    """Get the target layer for CAM (encoder's last stage output).

    For EfficientViT: hook the last stage in encoder.stages, which
    produces the final [B, 256, 7, 7] feature map before CBAM.
    """
    # encoder.stages is a Sequential of EfficientVitStage modules
    stages = model.encoder.stages
    return stages[-1]


def compute_cam(model, input_tensor, target_class, method, n_score_samples=10):
    """Compute CAM heatmap for a given input and target class.

    Args:
        model: the multi-task model
        input_tensor: [1, 3, H, W] input image
        target_class: 0-6 for binary tasks, 7-11 for Pfirrmann
        method: 'gradcam', 'gradcam++', 'eigen_cam', 'score_cam'
        n_score_samples: number of random masks for Score-CAM

    Returns:
        heatmap [H, W] normalized to [0, 1]
    """
    target_layer = get_target_layer(model)
    retain_grad = method in ('gradcam', 'gradcam++')
    fe = FeatureExtractor(model, target_layer, retain_grad=retain_grad)

    # Forward pass
    model.eval()
    out = model(input_tensor)  # [1, 12]

    acts = fe.activations[0].cpu().numpy()  # [C, H, W]

    if method == 'gradcam':
        # Backward pass for target class
        model.zero_grad()
        if target_class < 7:
            score = torch.sigmoid(out[0, target_class])
        else:
            score = torch.softmax(out[0, 7:], dim=0)[target_class - 7]
        score.backward(retain_graph=True)
        grads = fe.gradients[0].cpu().numpy()  # [C, H, W]
        cam = grad_cam(fe, target_class, acts, grads)

    elif method == 'gradcam++':
        model.zero_grad()
        if target_class < 7:
            score = torch.sigmoid(out[0, target_class])
        else:
            score = torch.softmax(out[0, 7:], dim=0)[target_class - 7]
        score.backward(retain_graph=True)
        grads = fe.gradients[0].cpu().numpy()
        cam = grad_cam_plus_plus(fe, target_class, acts, grads)

    elif method == 'eigen_cam':
        cam = eigen_cam(acts)

    elif method == 'score_cam':
        cam = score_cam(model, fe, input_tensor, target_class, acts, n_score_samples)

    else:
        raise ValueError(f"Unknown method: {method}")

    fe.remove()
    return cam


def upsample_cam(cam: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Upsample CAM heatmap to input image resolution."""
    from scipy.ndimage import zoom
    H, W = cam.shape
    h, w = target_size
    cam_up = zoom(cam, (h / H, w / W), order=1)
    # Normalize to [0, 1]
    cam_min, cam_max = cam_up.min(), cam_up.max()
    if cam_max - cam_min > 1e-8:
        cam_up = (cam_up - cam_min) / (cam_max - cam_min)
    return cam_up


def overlay_cam(image: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay CAM heatmap on the original image.

    Args:
        image: [H, W] grayscale image (0-1 range)
        cam: [H, W] heatmap (0-1 range)
        alpha: overlay transparency

    Returns:
        overlay: [H, W, 3] RGB image
    """
    # Convert grayscale to RGB
    if image.ndim == 2:
        img_rgb = np.stack([image] * 3, axis=-1)
    else:
        img_rgb = image.copy()

    # Normalize image to [0, 1]
    img_rgb = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min() + 1e-8)

    # Create heatmap (red channel)
    heatmap = np.zeros_like(img_rgb)
    heatmap[:, :, 0] = cam  # red
    heatmap[:, :, 1] = np.maximum(1 - cam * 2, 0)  # green (inverse)
    heatmap[:, :, 2] = np.maximum(1 - cam * 2, 0)  # blue (inverse)

    # Overlay
    overlay = (1 - alpha) * img_rgb + alpha * heatmap
    return np.clip(overlay, 0, 1)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="CAM saliency for SPIDER multi-task model")
    ap.add_argument("--ckpt", type=pathlib.Path, required=True)
    ap.add_argument("--arch", type=str, default="efficientvit_b1")
    ap.add_argument("--data_dir", type=pathlib.Path, default="crops")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--n_samples", type=int, default=20,
                    help="number of test samples to generate CAMs for")
    ap.add_argument("--methods", type=str, nargs="+",
                    default=["gradcam", "gradcam++", "eigen_cam", "score_cam"],
                    help="CAM methods to run")
    ap.add_argument("--tasks", type=str, nargs="+", default=None,
                    help="tasks to explain (e.g., 'Spondylolisthesis' 'Pfirrmann'). "
                         "If None, explain all 7 binary + Pfirrmann.")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out", type=pathlib.Path, default="results/cam")
    args = ap.parse_args()

    device = torch.device("cuda") if (args.device == "auto" and torch.cuda.is_available()) else torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    # Load model
    model = build_model(backbone=args.arch, pretrained=False).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded {args.ckpt} ({args.arch}) on {device}")

    # Load dataset
    ds = EvalCropDataset(args.data_dir, args.split)
    print(f"Dataset: {len(ds)} samples ({args.split})")

    # Determine which tasks to explain
    all_tasks = BIN_COLS + ["Pfirrmann"]
    if args.tasks:
        tasks_to_explain = [t for t in args.tasks if t in all_tasks]
    else:
        tasks_to_explain = all_tasks

    # Map task names to class indices
    task_to_idx = {name: i for i, name in enumerate(BIN_COLS)}
    task_to_idx["Pfirrmann"] = 7  # Will use argmax instead

    # Generate CAMs for each sample
    n = min(args.n_samples, len(ds))
    print(f"Generating CAMs for {n} samples x {len(args.methods)} methods x {len(tasks_to_explain)} tasks")

    # Also evaluate prediction correctness
    results = []
    for idx in range(n):
        row = ds.meta.iloc[idx]
        sample_id = row["sample_id"]

        # Load image
        np_img = np.load((ds.path / f"{sample_id}.npy").resolve()).astype(np.float32)
        img_tensor = torch.from_numpy(np_img).unsqueeze(0).repeat(3, 1, 1).float().unsqueeze(0).to(device)

        # Get model prediction
        with torch.no_grad():
            logits = model(img_tensor)
        pred_bin = (torch.sigmoid(logits[0, :7]) > 0.5).cpu().numpy()
        pred_pf = logits[0, 7:].argmax().item()

        # True labels
        y_bin = np.array([row[c] for c in BIN_COLS])
        y_pf = int(row["Pfirrmann"] - 1)

        # Generate CAM for each task/method
        for task in tasks_to_explain:
            if task == "Pfirrmann":
                target_class = 7 + pred_pf  # explain predicted class
                correct = (pred_pf == y_pf)
            else:
                target_class = task_to_idx[task]
                correct = bool(pred_bin[target_class] == y_bin[target_class])

            for method in args.methods:
                try:
                    cam = compute_cam(model, img_tensor, target_class, method)
                    cam_up = upsample_cam(cam, (np_img.shape[0], np_img.shape[1]))
                    overlay = overlay_cam(np_img, cam_up)

                    # Save
                    out_dir = args.out / method / task
                    out_dir.mkdir(parents=True, exist_ok=True)
                    np.save(out_dir / f"{sample_id}.npy", cam_up)
                    # Save overlay as PNG for visualization
                    from PIL import Image
                    overlay_uint8 = (overlay * 255).astype(np.uint8)
                    Image.fromarray(overlay_uint8).save(out_dir / f"{sample_id}.png")

                    results.append({
                        "sample_id": sample_id, "task": task, "method": method,
                        "correct": correct, "cam_mean": float(cam_up.mean()),
                        "cam_max": float(cam_up.max()),
                    })
                    print(f"  {sample_id} | {task:20s} | {method:12s} | {'OK' if correct else 'WRONG'}")
                except Exception as e:
                    print(f"  {sample_id} | {task:20s} | {method:12s} | ERROR: {e}")

    # Summary
    import json
    summary = {
        "checkpoint": str(args.ckpt),
        "arch": args.arch,
        "n_samples": n,
        "methods": args.methods,
        "tasks": tasks_to_explain,
        "results": results,
    }
    with open(args.out / "cam_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {len(results)} CAM maps -> {args.out}")
    print(f"Summary -> {args.out / 'cam_summary.json'}")


if __name__ == "__main__":
    main()
