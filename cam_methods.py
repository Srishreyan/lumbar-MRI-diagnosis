#!/usr/bin/env python
# cam_methods.py
# ------------------------------------------------------------------
# Four class-activation saliency maps for the SPIDER multi-task model,
# implemented from scratch on top of a hooked convolutional layer:
#
#   Grad-CAM   (Selvaraju et al., ICCV 2017) [25]
#   Grad-CAM++ (Chattopadhay et al., WACV 2018) [26]
#   Score-CAM  (Wang et al., CVPR 2020) [27]
#   Eigen-CAM  (Muhammad & Yeasin, 2020) [28]
#
# All methods consume the target conv layer's activation tensor A [B,C,h,w]
# (and its backward gradients where needed) and emit a saliency map
# upsized to the input spatial resolution. `target_index` selects which of
# the 12 output logits (7 binary + 5 Pfirrmann) to explain.
# ------------------------------------------------------------------
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _upsample_spatial(map2d: torch.Tensor, size) -> torch.Tensor:
    """map2d [B,1,h,w] -> [B,1,H,W] bilinear."""
    return F.interpolate(map2d, size=size, mode="bilinear", align_corners=False)


def _weights_gradcam(grad: torch.Tensor) -> torch.Tensor:
    """Grad-CAM channel weights = GAP of gradients -> [B, C]."""
    return grad.flatten(2).mean(dim=2)          # [B, C]


def _weights_gradcam_pp(grad: torch.Tensor) -> torch.Tensor:
    """Grad-CAM++ channel weights (Chattopadhay et al., 2018)."""
    g = grad                                                       # [B,C,h,w]
    g2 = g.pow(2)
    g3 = g2 * g
    sum_g2 = g2.sum(dim=(2, 3), keepdim=True)
    sum_g3 = g3.sum(dim=(2, 3), keepdim=True)
    denom = 2 * g2 + (sum_g2 * g3) / (sum_g3 + 1e-7) + 1e-7
    alpha = (g2 / denom) * g                                       # [B,C,h,w]
    return alpha.flatten(2).sum(dim=2)                             # [B,C]


def _weights_eigencam(act: torch.Tensor) -> torch.Tensor:
    """Eigen-CAM channel weights = top eigenvector of channel covariance."""
    B, C, h, w = act.shape
    flat = act.reshape(B, C, h * w)                                # [B,C,HW]
    cov = flat @ flat.transpose(1, 2)                              # [B,C,C]
    _, eigvec = torch.linalg.eigh(cov)                             # ascending
    return eigvec[:, :, -1]                                        # [B,C] top


class CAMBackend:
    """Hooks a target conv layer and reduces it to a saliency map."""

    def __init__(self, model: nn.Module, target_conv: nn.Module):
        self.model = model
        self.act = None
        self.grad = None
        self._fh = target_conv.register_forward_hook(self._save_act)
        self._bh = target_conv.register_full_backward_hook(self._save_grad)

    def _save_act(self, mod, inp, out):
        self.act = out.detach()

    def _save_grad(self, mod, grad_in, grad_out):
        self.grad = grad_out[0].detach()

    def score(self, x: torch.Tensor, target_index: int) -> torch.Tensor:
        """Logit of the target output (one of the 12 heads), requires grad."""
        logits = self.model(x)                       # [B,12]
        return logits[:, target_index]

    def __del__(self):
        self._fh.remove()
        self._bh.remove()


def gradcam(backend: CAMBackend, x: torch.Tensor, target_index: int,
            spatial: int, max_channels: int = None) -> torch.Tensor:
    x = x.detach().clone().requires_grad_(True)
    model = backend.model
    all_p = model.parameters()
    for p in all_p:
        p.requires_grad_(False)
    s = backend.score(x, target_index)
    s.sum().backward()
    A = backend.act                                      # [B,C,h,w]
    grad = backend.grad
    w = _weights_gradcam(grad)                            # [B,C]
    cam = torch.relu((w[:, :, None, None] * A).sum(dim=1, keepdim=True))
    return _upsample_spatial(cam, (spatial, spatial))


def gradcam_pp(backend: CAMBackend, x: torch.Tensor, target_index: int,
               spatial: int, max_channels: int = None) -> torch.Tensor:
    x = x.detach().clone().requires_grad_(True)
    for p in backend.model.parameters():
        p.requires_grad_(False)
    s = backend.score(x, target_index)
    s.sum().backward()
    A = backend.act
    grad = backend.grad
    w = _weights_gradcam_pp(grad)
    cam = torch.relu((w[:, :, None, None] * A).sum(dim=1, keepdim=True))
    return _upsample_spatial(cam, (spatial, spatial))


def eigencam(backend: CAMBackend, x: torch.Tensor, target_index: int,
             spatial: int, max_channels: int = None) -> torch.Tensor:
    """No gradients needed; just a forward pass to capture activations."""
    with torch.no_grad():
        backend.model(x)                                  # populate backend.act
    A = backend.act
    w = _weights_eigencam(A)
    cam = torch.relu((w[:, :, None, None] * A).sum(dim=1, keepdim=True))
    return _upsample_spatial(cam, (spatial, spatial))


def scorecam(backend: CAMBackend, x: torch.Tensor, target_index: int,
             spatial: int, max_channels: int = None) -> torch.Tensor:
    """Channel weighting by forward-difference of the target score when each
    masked channel is shown. 256 channels => up to 256 forward passes/sample."""
    x = x.detach().clone()
    device = x.device
    with torch.no_grad():
        backend.model(x)
    A = backend.act                                          # [B,C,h,w]
    B, C, h, w = A.shape
    base = backend.model(x)[:, target_index:target_index + 1]  # [B,1]
    if max_channels is not None and C > max_channels:
        w0 = _weights_eigencam(A)
        keep = torch.topk(w0, max_channels, dim=1).indices      # [B,K]
        sel = torch.zeros(C, dtype=torch.bool, device=device)
        sel[keep.flatten().unique()] = True
        idxs = sel.nonzero(as_tuple=False).squeeze(1)
    else:
        idxs = torch.arange(C, device=device)

    weights = torch.zeros((B, C), device=device)
    sal_map = F.interpolate(A, size=(spatial, spatial), mode="bilinear", align_corners=False)
    # normalize each channel to [0,1] then mask input
    x_size = (spatial, spatial)
    for k in idxs.tolist():
        m = sal_map[:, k:k+1]
        lo = m.flatten(2).min(dim=2, keepdim=True).values[..., None]
        hi = m.flatten(2).max(dim=2, keepdim=True).values[..., None]
        m_norm = (m - lo) / (hi - lo + 1e-8)                   # [B,1,H,W]
        masked = x * m_norm                                  # channel-wise mask × input
        sc = backend.model(masked)[:, target_index]          # [B]
        weights[:, k] = sc - base[:, 0]
    cam = torch.relu((weights[:, :, None, None] * A).sum(dim=1, keepdim=True))
    return _upsample_spatial(cam, (spatial, spatial))


METHODS = {
    "gradcam": gradcam,
    "gradcampp": gradcam_pp,
    "eigencam": eigencam,
    "scorecam": scorecam,
}


def compute_map(model: nn.Module, target_conv: nn.Module, x: torch.Tensor,
                target_index: int, method: str, spatial: int = 224,
                max_channels: int = None) -> np.ndarray:
    """Returns [B,H,W] numpy saliency map (already ReLU'd, 0..1 normalized)."""
    model.eval()
    backend = CAMBackend(model, target_conv)
    fn = METHODS[method]
    m = fn(backend, x, target_index, spatial, max_channels=max_channels)
    m = m.squeeze(1)                                       # [B,H,W]
    m = torch.clamp(m, min=0.0)
    # normalize per sample
    lo = m.flatten(1).min(dim=1, keepdim=True).values
    hi = m.flatten(1).max(dim=1, keepdim=True).values
    m = (m - lo) / (hi - lo + 1e-8)
    return m.detach().cpu().numpy()