"""Evaluation metrics: PSNR (RGB and Y-channel) and SSIM.

The challenge protocol (spec §B.2) is PSNR on the Y channel with a 4-px
border crop at x4 — ``psnr_y(pred, target, border=scale)``.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def psnr(
    pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0, border: int = 0
) -> torch.Tensor:
    """Mean PSNR over the batch for NCHW tensors in [0, max_val].

    ``border`` crops that many pixels from each spatial edge first — SR
    benchmarks (e.g. NTIRE) conventionally crop ``scale`` pixels.
    """
    if border > 0:
        pred = pred[..., border:-border, border:-border]
        target = target[..., border:-border, border:-border]
    mse = ((pred - target) ** 2).flatten(1).mean(dim=1)
    return (10.0 * torch.log10(max_val**2 / mse.clamp_min(1e-12))).mean()


def rgb_to_y(x: torch.Tensor) -> torch.Tensor:
    """YCbCr luma (ITU-R BT.601 *studio swing*) from RGB in [0, 1].

    Y = (16 + 65.481 R + 128.553 G + 24.966 B) / 255 — the exact conversion
    BasicSR's ``bgr2ycbcr(..., y_only=True)`` applies, which is what the
    NTIRE ``calculate_psnr(..., test_y_channel=True)`` protocol uses. The
    full-swing 0.299/0.587/0.114 luma reads ~1.3 dB lower and is NOT
    comparable with challenge numbers. Keeps NCHW with C=1.
    """
    weights = torch.tensor([65.481, 128.553, 24.966], device=x.device, dtype=x.dtype)
    return ((x * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True) + 16.0) / 255.0


def psnr_y(pred: torch.Tensor, target: torch.Tensor, border: int = 0) -> torch.Tensor:
    """PSNR on the Y (luma) channel only, challenge protocol.

    Pass ``border=scale`` for the NTIRE SR protocol (the challenge's
    33.46 dB reference is Y-channel PSNR with a 4-px border crop at x4).
    """
    return psnr(rgb_to_y(pred), rgb_to_y(target), border=border)


def _gaussian_window(size: int, sigma: float) -> torch.Tensor:
    ax = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    g = torch.exp(-(ax**2) / (2.0 * sigma**2))
    g = g / g.sum()
    return torch.outer(g, g)


class SSIM(torch.nn.Module):
    """SSIM with a Gaussian window (default 11x11, sigma 1.5).

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


class SSIMMetric(torch.nn.Module):
    """SSIM metric wrapper (eval mode, no grad)."""

    def __init__(self) -> None:
        super().__init__()
        self.ssim = SSIM()
        self.eval()

    @torch.no_grad()
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ssim(pred, target)
