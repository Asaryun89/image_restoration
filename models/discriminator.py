"""70x70 PatchGAN discriminator (pix2pix style) with optional spectral norm."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class PatchDiscriminator(nn.Module):
    """PatchGAN with a 70x70 receptive field for the default ``n_layers=3``.

    Conv stack 64 -> 128 -> 256 -> 512, kernel 4; the first ``n_layers``
    convs use stride 2, the last two layers stride 1. InstanceNorm on every
    layer except the first, LeakyReLU(0.2), 1-channel patch logits out.

    Args:
        in_channels: input image channels.
        base_channels: width of the first conv layer.
        n_layers: number of stride-2 conv layers (3 -> 70x70 patches).
        use_spectral_norm: wrap every conv in spectral normalization.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        n_layers: int = 3,
        use_spectral_norm: bool = True,
    ) -> None:
        super().__init__()
        sn = spectral_norm if use_spectral_norm else (lambda m: m)

        layers = [
            sn(nn.Conv2d(in_channels, base_channels, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        mult = 1
        for i in range(1, n_layers):
            prev, mult = mult, min(2**i, 8)
            layers += [
                sn(nn.Conv2d(base_channels * prev, base_channels * mult, 4, stride=2, padding=1)),
                nn.InstanceNorm2d(base_channels * mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        prev, mult = mult, min(2**n_layers, 8)
        layers += [
            sn(nn.Conv2d(base_channels * prev, base_channels * mult, 4, stride=1, padding=1)),
            nn.InstanceNorm2d(base_channels * mult),
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv2d(base_channels * mult, 1, 4, stride=1, padding=1)),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns per-patch logits of shape (N, 1, H', W')."""
        return self.model(x)
