#!/usr/bin/env python
# efficientvit_cbam.py
# ------------------------------------------------------------------
# Multi-task EfficientViT + CBAM model for the SPIDER paper.
#
# Architecture:
#   EfficientViT backbone (timm, pretrained) -> intermediate feature map
#   -> CBAM (channel + spatial attention) -> global average pool
#   -> multi-task head: 7 binary pathologies (sigmoid) + 5 Pfirrmann (softmax)
#
# MC-Dropout uncertainty: when `mc_dropout_p > 0`, a dropout layer sits in the
# head; running N stochastic forward passes samples the predictive distribution
# (a pillar of the paper's "trustworthy" framing). Dropout is active in eval.
#
# The model is self-sizing: it copies the backbone feature dim at build time,
# so it works unchanged across efficientvit_b0/b1/b2/m0/m1.
# ------------------------------------------------------------------
from __future__ import annotations

import torch
import torch.nn as nn
import timm


# ------------------------------------------------------------------
# CBAM (Convolutional Block Attention Module)
#   Woo et al., ECCV 2018. Channel attention (with shared MLP) followed
#   by spatial attention (7x7 conv on channel-averaged maps).
# ------------------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.mlp(x.mean(dim=(2, 3), keepdim=True))
        mx = self.mlp(x.amax(dim=(2, 3), keepdim=True))
        return torch.sigmoid(avg + mx) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)
        nn.init.constant_(self.conv.weight, 0.1)  # gentle start

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        m = torch.cat([avg, mx], dim=1)
        return torch.sigmoid(self.conv(m)) * x


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


# ------------------------------------------------------------------
# Multi-task head on top of a pooled feature vector
# ------------------------------------------------------------------
class MultiTaskHead(nn.Module):
    """Projects a pooled feature vector to the 7 binary + 5 Pfirrmann logits.

    A single shared trunk feeds two task-specific linear layers. The shared
    trunk lets all tasks benefit from common features while each task keeps
    its own decision boundary.
    """
    def __init__(self, in_features: int, hidden: int = 256,
                 num_bin: int = 7, num_pfir: int = 5, mc_dropout_p: float = 0.0):
        super().__init__()
        self.mc_dropout_p = mc_dropout_p
        self.trunk = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(mc_dropout_p),   # MC-Dropout lives here
        )
        self.bin_head = nn.Linear(hidden, num_bin)
        self.pfir_head = nn.Linear(hidden, num_pfir)

    def forward(self, x: torch.Tensor):
        x = self.trunk(x)
        return self.bin_head(x), self.pfir_head(x)


def _set_mc_dropout(model: nn.Module, active: bool) -> None:
    """Toggle ONLY dropout layers (leave norm layers frozen in eval stats)."""
    mode = "train" if active else "eval" if not model.training else None
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            getattr(m, mode)()


# ------------------------------------------------------------------
# Full multi-task model: EfficientViT + CBAM + head
# ------------------------------------------------------------------
class EfficientViTMultiTask(nn.Module):
    def __init__(self, backbone: str = "efficientvit_b1", pretrained: bool = True,
                 freeze_backbone: bool = False, hidden: int = 256,
                 num_bin: int = 7, num_pfir: int = 5,
                 mc_dropout_p: float = 0.0, backbone_mc_dropout_p: float = 0.0):
        super().__init__()
        self.backbone = backbone
        self.mc_dropout_p = mc_dropout_p
        self.backbone_mc_dropout_p = backbone_mc_dropout_p
        self.num_bin, self.num_pfir = num_bin, num_pfir

        self.encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        in_features = self.encoder.num_features
        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Extra MC-dropout on the spatial feature map (richer epistemic signal).
        # Dropout2d drops whole channels, which suits a (C, H, W) feature tensor.
        self.backbone_dropout = nn.Dropout2d(backbone_mc_dropout_p)

        self.cbam = CBAM(in_features)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = MultiTaskHead(in_features, hidden=hidden,
                                  num_bin=num_bin, num_pfir=num_pfir,
                                  mc_dropout_p=mc_dropout_p)

    def forward(self, x: torch.Tensor):
        mc_on = (self.mc_dropout_p > 0 or self.backbone_mc_dropout_p > 0)
        if mc_on and not self.training:
            _set_mc_dropout(self, True)   # dropout active in eval for MC sampling
        feat = self.encoder.forward_features(x)        # [B, C, H/32, W/32]
        if self.backbone_mc_dropout_p > 0:
            feat = self.backbone_dropout(feat)
        feat = self.cbam(feat)
        feat = self.pool(feat).flatten(1)              # [B, C]
        logits_bin, logits_pfir = self.head(feat)
        out = torch.cat([logits_bin, logits_pfir], dim=1)
        if mc_on and not self.training:
            _set_mc_dropout(self, False)
        return out

    def stochastic_predict(self, x: torch.Tensor, n_samples: int = 10):
        """MC-Dropout: sample n forward passes -> per-sample logits + variance."""
        xs = x.repeat_interleave(n_samples, dim=0)
        with torch.no_grad():
            out = self(xs)
        out = out.view(x.size(0), n_samples, -1)
        mean = out.mean(dim=1)
        var = out.var(dim=1, unbiased=False)
        return mean, var


def build_model(backbone: str = "efficientvit_b1", pretrained: bool = True,
                freeze: bool = False, mc_dropout_p: float = 0.0,
                backbone_mc_dropout_p: float = 0.0) -> EfficientViTMultiTask:
    return EfficientViTMultiTask(backbone=backbone, pretrained=pretrained,
                                 freeze_backbone=freeze,
                                 mc_dropout_p=mc_dropout_p,
                                 backbone_mc_dropout_p=backbone_mc_dropout_p)


if __name__ == "__main__":
    m = build_model(backbone="efficientvit_b1", pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    y = m(x)
    tot = sum(p.numel() for p in m.parameters())
    train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print("output shape:", tuple(y.shape), "| total params:", round(tot/1e6, 2), "M",
          "| trainable:", round(train/1e6, 2), "M")