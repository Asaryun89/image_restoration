"""Overlap-tiled inference for full-image evaluation on limited GPU memory.

Feeding a whole LR image through the generator needs activation memory for
the full output resolution (SRHMCnet's NAFNet stage runs at HR). Tiling cuts
the input into fixed-size tiles that fit in memory, runs the model per tile,
and stitches the outputs. Tiles overlap (the winner's pipeline uses 32 px) and
overlapping output regions are averaged, which suppresses seams from the
missing cross-tile context at tile borders.
"""
from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def tiled_forward(
    model: nn.Module,
    x: torch.Tensor,
    tile_size: int,
    tile_overlap: int = 32,
    scale: int = 1,
) -> torch.Tensor:
    """Run ``model`` over ``x`` in overlapping tiles and average the overlaps.

    Args:
        model: image-to-image module mapping (B, C, h, w) -> (B, C', h*scale, w*scale).
        x: input batch (B, C, H, W).
        tile_size: input-side tile edge; clamped to the image size, so a
            tile at least as large as the image degrades to one whole-image
            forward pass.
        tile_overlap: input-side overlap between neighbouring tiles.
        scale: the model's output/input resolution ratio (SR factor; 1 for
            same-resolution models like NAFNet).

    Returns:
        The stitched output, identical in layout to a whole-image forward.
        Exact for models whose output pixels depend only on same-tile input
        pixels; otherwise an approximation whose tile-border error shrinks
        as ``tile_overlap`` grows.
    """
    b, _, h, w = x.shape
    tile = min(tile_size, h, w)
    stride = max(tile - tile_overlap, 1)
    h_starts = list(range(0, h - tile, stride)) + [h - tile]
    w_starts = list(range(0, w - tile, stride)) + [w - tile]

    out: torch.Tensor | None = None
    weight: torch.Tensor | None = None
    for hs in h_starts:
        for ws in w_starts:
            patch = x[..., hs : hs + tile, ws : ws + tile]
            out_patch = model(patch)
            if out is None:
                out = x.new_zeros(b, out_patch.shape[1], h * scale, w * scale)
                weight = x.new_zeros(b, 1, h * scale, w * scale)
            out[..., hs * scale : (hs + tile) * scale, ws * scale : (ws + tile) * scale] += (
                out_patch
            )
            weight[
                ..., hs * scale : (hs + tile) * scale, ws * scale : (ws + tile) * scale
            ] += 1.0
    return out / weight
