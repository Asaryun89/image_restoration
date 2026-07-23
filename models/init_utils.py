"""Shared weight-initialization helpers."""
from __future__ import annotations

import torch
import torch.nn as nn


def icnr_init(conv: nn.Conv2d, scale: int) -> None:
    """ICNR init (Aitken et al., 2017) for a conv feeding ``PixelShuffle(scale)``.

    Without it, the ``scale**2`` sub-kernels that PixelShuffle interleaves into
    each output pixel's neighborhood start independently random, producing a
    checkerboard pattern at initialization that can persist well into
    training. This copies one kaiming-initialized kernel across all
    ``scale**2`` sub-kernels per output group instead, so the upsampler starts
    as smooth nearest-neighbor interpolation.
    """
    out_channels = conv.weight.shape[0]
    sub_channels = out_channels // (scale * scale)
    sub = torch.empty(sub_channels, *conv.weight.shape[1:])
    nn.init.kaiming_normal_(sub)
    weight = sub.repeat_interleave(scale * scale, dim=0)
    with torch.no_grad():
        conv.weight.copy_(weight)
