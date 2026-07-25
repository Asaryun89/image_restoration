"""RRDBNet — the ESRGAN x4 generator (Wang et al., ECCV-W 2018).

Baseline model for comparison against SRHCnet. Layer names match the
original xinntao/ESRGAN release exactly, so `RRDB_ESRGAN_x4.pth` (and any
interpolated RRDB_PSNR/ESRGAN mix from ``net_interp.py``) loads strictly.
"""
from __future__ import annotations

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDenseBlock5C(nn.Module):
    """Five densely connected convs with 0.2-scaled residual."""

    def __init__(self, nf: int = 64, gc: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual-in-residual dense block."""

    def __init__(self, nf: int, gc: int = 32) -> None:
        super().__init__()
        self.RDB1 = ResidualDenseBlock5C(nf, gc)
        self.RDB2 = ResidualDenseBlock5C(nf, gc)
        self.RDB3 = ResidualDenseBlock5C(nf, gc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.RDB3(self.RDB2(self.RDB1(x)))
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """ESRGAN x4 generator: RRDB trunk + two nearest-neighbour upsamples.

    Args:
        in_nc / out_nc: input/output channels (RGB: 3).
        nf: trunk feature width (release weights: 64).
        nb: number of RRDB blocks (release weights: 23).
        gc: dense-block growth channels (release weights: 32).
        pretrained: optional path to a flat ESRGAN state dict, loaded
            strictly on the CPU.
    """

    def __init__(
        self,
        in_nc: int = 3,
        out_nc: int = 3,
        nf: int = 64,
        nb: int = 23,
        gc: int = 32,
        pretrained: str | None = None,
    ) -> None:
        super().__init__()
        block = functools.partial(RRDB, nf=nf, gc=gc)
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.RRDB_trunk = nn.Sequential(*(block() for _ in range(nb)))
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1)
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if pretrained:
            state = torch.load(pretrained, map_location="cpu", weights_only=True)
            self.load_state_dict(state, strict=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fea = self.conv_first(x)
        fea = fea + self.trunk_conv(self.RRDB_trunk(fea))
        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode="nearest")))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.HRconv(fea)))
