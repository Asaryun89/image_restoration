"""Alternating L1 / L2 / SWT losses (team17 spec §B.4).

The report says the losses "alternately iterate"; granularity is
unpublished, so the simplest faithful reading is used: a per-iteration
round-robin — ``loss = [L1, L2, SWT][iter_idx % 3]``.

SWT loss follows the spec's reading of the NTIRE-2024 wavelet-losses
reference: a level-1 stationary (undecimated) wavelet transform with the
``sym7`` wavelet on the Y channel, L1 between the sub-bands of prediction
and target, high-frequency bands (LH/HL/HH) weighted by ``hf_weight``.
The SWT is implemented as fixed depthwise convolutions on the GPU (the
spec explicitly warns that pywt CPU round-trips kill throughput). The
``sym7``/``hf_weight`` choices are the spec's declared assumptions —
SRC-B's exact SWT config is unpublished.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# pywt.Wavelet('sym7').dec_lo / dec_hi (orthonormal decomposition filters),
# hardcoded so training needs no pywt dependency. Verified against
# PyWavelets 1.x: sym7, 14 taps.
_SYM7_DEC_LO = [
    0.002681814568257878, -0.0010473848886829163, -0.01263630340325193,
    0.03051551316596357, 0.0678926935013727, -0.049552834937127255,
    0.017441255086855827, 0.5361019170917628, 0.767764317003164,
    0.2886296317515146, -0.14004724044296152, -0.10780823770381774,
    0.004010244871533663, 0.010268176708511255,
]
_SYM7_DEC_HI = [
    -0.010268176708511255, 0.004010244871533663, 0.10780823770381774,
    -0.14004724044296152, -0.2886296317515146, 0.767764317003164,
    -0.5361019170917628, 0.017441255086855827, 0.049552834937127255,
    0.0678926935013727, -0.03051551316596357, -0.01263630340325193,
    0.0010473848886829163, 0.002681814568257878,
]

_BT601_Y = (0.299, 0.587, 0.114)


class SWTLoss(nn.Module):
    """Level-1 stationary wavelet L1 loss on the Y channel (GPU, sym7)."""

    def __init__(self, hf_weight: float = 0.05) -> None:
        super().__init__()
        self.hf_weight = float(hf_weight)
        # pywt applies true convolution with dec filters scaled by 1/sqrt(2)
        # (norm=True); conv2d is correlation, so flip the filters. This bank
        # + circular padding reproduces `pywt.swt2(..., level=1, norm=True)`
        # bit-exactly (verified: max abs diff ~1e-16 vs PyWavelets 1.x).
        lo = torch.tensor(_SYM7_DEC_LO[::-1]) / math.sqrt(2.0)
        hi = torch.tensor(_SYM7_DEC_HI[::-1]) / math.sqrt(2.0)
        # Separable 2D filters as (out=4, in=1, k, k) depthwise weights:
        # LL = lo x lo, then the three high-frequency bands (pywt cV/cH/cD).
        bank = torch.stack(
            [torch.outer(a, b) for a in (lo, hi) for b in (lo, hi)]
        ).unsqueeze(1)
        self.register_buffer("bank", bank, persistent=False)
        self.register_buffer(
            "y_weights", torch.tensor(_BT601_Y).view(1, 3, 1, 1), persistent=False
        )
        k = len(_SYM7_DEC_LO)
        self.pad = (k // 2 - 1, k // 2)  # pywt swt phase alignment

    def _swt(self, y: torch.Tensor) -> torch.Tensor:
        """(B,1,H,W) -> (B,4,H,W) sub-bands LL, cV, cH, cD (undecimated).

        Circular padding matches pywt's periodization boundary; it needs
        H, W >= 14, which every training patch (LR >= 80) satisfies.
        """
        y = F.pad(y, [*self.pad, *self.pad], mode="circular")
        return F.conv2d(y, self.bank.to(y.dtype))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        y_pred = (pred * self.y_weights).sum(dim=1, keepdim=True)
        y_target = (target * self.y_weights).sum(dim=1, keepdim=True)
        bands_pred = self._swt(y_pred)
        bands_target = self._swt(y_target)
        ll = F.l1_loss(bands_pred[:, :1], bands_target[:, :1])
        hf = F.l1_loss(bands_pred[:, 1:], bands_target[:, 1:])
        return ll + self.hf_weight * hf


class AlternatingLoss(nn.Module):
    """Round-robin L1 / L2 / SWT selected by the iteration index."""

    def __init__(self, hf_weight: float = 0.05) -> None:
        super().__init__()
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        self.swt = SWTLoss(hf_weight=hf_weight)
        self._names = ("l1", "l2", "swt")

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, iter_idx: int
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Returns (loss, components dict naming the active loss)."""
        which = iter_idx % 3
        loss = (self.l1, self.l2, self.swt)[which](pred, target)
        name = self._names[which]
        return loss, {f"loss_{name}": loss.item(), "loss_total": loss.item()}
