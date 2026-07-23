"""Stage-1 pixel losses: L1, differentiable SSIM, and VGG19 perceptual loss.

Total: ``L = lambda_l1 * L1 + lambda_perc * Perceptual + lambda_ssim * (1 - SSIM)``.
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# Index of each named ReLU inside torchvision's vgg19().features.
VGG19_LAYER_INDICES = {
    "relu1_1": 1, "relu1_2": 3,
    "relu2_1": 6, "relu2_2": 8,
    "relu3_1": 11, "relu3_2": 13, "relu3_3": 15, "relu3_4": 17,
    "relu4_1": 20, "relu4_2": 22, "relu4_3": 24, "relu4_4": 26,
    "relu5_1": 29, "relu5_2": 31, "relu5_3": 33, "relu5_4": 35,
}


class PerceptualLoss(nn.Module):
    """L1 distance between frozen VGG19 features of prediction and target.

    Inputs are RGB in [0, 1]; ImageNet normalization happens inside.
    """

    def __init__(
        self, layers: Sequence[str] = ("relu1_2", "relu2_2", "relu3_4", "relu4_4")
    ) -> None:
        super().__init__()
        self.indices = sorted(VGG19_LAYER_INDICES[name] for name in layers)
        vgg = torchvision.models.vgg19(
            weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1
        ).features[: self.indices[-1] + 1]
        vgg.eval()
        for p in vgg.parameters():
            p.requires_grad_(False)
        self.vgg = vgg
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _features(self, x: torch.Tensor) -> list:
        x = (x - self.mean) / self.std
        feats = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.indices:
                feats.append(x)
        return feats

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        feats_pred = self._features(pred)
        with torch.no_grad():
            feats_target = self._features(target)
        return sum(F.l1_loss(fp, ft) for fp, ft in zip(feats_pred, feats_target))


def _gaussian_window(size: int, sigma: float) -> torch.Tensor:
    ax = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    g = torch.exp(-(ax**2) / (2.0 * sigma**2))
    g = g / g.sum()
    return torch.outer(g, g)


class SSIM(nn.Module):
    """Differentiable SSIM (Gaussian window, default 11x11, sigma 1.5).

    Returns mean SSIM over the batch; inputs in [0, 1].
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5) -> None:
        super().__init__()
        self.window_size = window_size
        window = _gaussian_window(window_size, sigma)
        self.register_buffer("window", window[None, None])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        window = self.window.expand(c, 1, -1, -1).to(dtype=x.dtype)
        c1, c2 = 0.01**2, 0.03**2

        mu_x = F.conv2d(x, window, groups=c)
        mu_y = F.conv2d(y, window, groups=c)
        mu_x2, mu_y2, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y
        sigma_x2 = F.conv2d(x * x, window, groups=c) - mu_x2
        sigma_y2 = F.conv2d(y * y, window, groups=c) - mu_y2
        sigma_xy = F.conv2d(x * y, window, groups=c) - mu_xy

        ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
            (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
        )
        return ssim_map.mean()


class SSIMLoss(nn.Module):
    """Loss form ``1 - SSIM``."""

    def __init__(self, window_size: int = 11, sigma: float = 1.5) -> None:
        super().__init__()
        self.ssim = SSIM(window_size, sigma)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - self.ssim(pred, target)


class PixelLoss(nn.Module):
    """Weighted combination of L1, perceptual, and SSIM losses (stage 1)."""

    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_perc: float = 0.1,
        lambda_ssim: float = 0.2,
        vgg_layers: Sequence[str] = ("relu1_2", "relu2_2", "relu3_4", "relu4_4"),
    ) -> None:
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_perc = lambda_perc
        self.lambda_ssim = lambda_ssim
        self.l1 = nn.L1Loss()
        self.perceptual = PerceptualLoss(vgg_layers) if lambda_perc > 0 else None
        self.ssim_loss = SSIMLoss() if lambda_ssim > 0 else None

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Returns (total loss, detached per-component dict)."""
        l1 = self.l1(pred, target)
        total = self.lambda_l1 * l1
        comps = {"loss_l1": l1.item()}
        if self.perceptual is not None:
            perc = self.perceptual(pred, target)
            total = total + self.lambda_perc * perc
            comps["loss_perc"] = perc.item()
        if self.ssim_loss is not None:
            ssim_l = self.ssim_loss(pred, target)
            total = total + self.lambda_ssim * ssim_l
            comps["loss_ssim"] = ssim_l.item()
        comps["loss_total"] = total.item()
        return total, comps
