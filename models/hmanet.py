"""HMANet (Chu et al., "HMANet: Hybrid Multi-Axis Aggregation Network for
Image Super-Resolution", NTIRE 2024) — the SR branch for :class:`SRHMCnet`.

Adapted from the official implementation (github.com/korouuuuu/HMA,
``hma/archs/hma_arch.py``) per ``SRHCnet_HMANet_swap_spec.md``:

- **Internal mean normalization is stripped** (spec §3.1): a wrapper —
  :class:`HMANetSR` for standalone stage-1 training, or ``SRHMCnet`` for
  the composite — normalizes before / de-normalizes after the core. Input
  to :class:`HMANet` is assumed already padded to a multiple of
  ``window_size`` and mean-normalized.
- Unused official machinery is dropped (``ape``, ``PatchMerging``,
  ``use_checkpoint`` — defined but never used in the official forward,
  non-'pixelshuffle' upsamplers).
- All module/attribute names match the official code one-to-one, so the
  official ``HMA_SRx4_pretrain.pth`` (``params_ema``) loads directly.

Macro structure per group (RHTB), for depth 6:
    3 × [FusedConv + FAB(shift 0)] interleaved with 3 × FAB(shift ws/2),
    then one GAB (grid attention, ``interval_size`` sparse-global), joined
    by a learnable per-channel ``scale``; Conv3x3 + group residual.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


# ------------------------------------------------------------- primitives
def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    """Stochastic depth per sample (when applied in the residual branch)."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """(B, H, W, C) -> (num_windows*B, ws, ws, C)."""
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)


def window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int) -> torch.Tensor:
    """(num_windows*B, ws, ws, C) -> (B, H, W, C)."""
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)


class PatchEmbed(nn.Module):
    """(B, C, H, W) -> (B, H*W, C), optional LayerNorm (patch_size = 1)."""

    def __init__(self, embed_dim: int, norm_layer: Optional[type] = None) -> None:
        super().__init__()
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    """(B, H*W, C) -> (B, C, H, W)."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, x_size: Tuple[int, int]) -> torch.Tensor:
        return x.transpose(1, 2).contiguous().view(x.shape[0], self.embed_dim, *x_size)


class Upsample(nn.Sequential):
    """Pixel-shuffle upsampler for power-of-2 scales and scale 3."""

    def __init__(self, scale: int, num_feat: int) -> None:
        layers = []
        if (scale & (scale - 1)) == 0:  # 1, 2, 4, 8...
            for _ in range(int(math.log2(scale))):
                layers.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                layers.append(nn.PixelShuffle(2))
        elif scale == 3:
            layers.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            layers.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f"Unsupported upscale factor: {scale}")
        super().__init__(*layers)


# ------------------------------------------------------------ grid helpers
def grid_shuffle(x: torch.Tensor, h: int, w: int, c: int, interval_size: int) -> torch.Tensor:
    """(B, H, W, C) -> (B*i*i, H/i, W/i, C): pixels ``i`` apart share a grid."""
    x = x.view(-1, h // interval_size, interval_size, w // interval_size, interval_size, c)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    return x.view(-1, h // interval_size, w // interval_size, c)


def grid_unshuffle(x: torch.Tensor, b: int, h: int, w: int, interval_size: int) -> torch.Tensor:
    """Inverse of :func:`grid_shuffle`."""
    x = x.view(b, interval_size, interval_size, h // interval_size, w // interval_size, -1)
    return x.permute(0, 3, 1, 4, 2, 5).contiguous().view(b, h, w, -1)


# ------------------------------------------------------------- attention
class DynamicPosBias(nn.Module):
    """MLP that maps relative coordinates to per-head position biases."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim), nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim), nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim), nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads),
        )

    def forward(self, biases: torch.Tensor) -> torch.Tensor:
        return self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))


class WindowAttention(nn.Module):
    """Window MSA over a pre-projected qkv tensor (HMA convention: the qkv
    Linear lives in the calling block, and ``x`` here has 3*dim channels)."""

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = (window_size, window_size)
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, x: torch.Tensor, rpi: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        b_, n, c = x.shape  # c = 3 * dim
        qkv = x.reshape(b_, n, 3, self.num_heads, c // self.num_heads // 3).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.relative_position_bias_table[rpi.view(-1)].view(n, n, -1)
        attn = attn + bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = self.attn_drop(self.softmax(attn))

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c // 3)
        return self.proj_drop(self.proj(x))


class AffineTransform(nn.Module):
    """One cross-attention step of grid attention with DynamicPosBias."""

    def __init__(
        self, dim: int, num_heads: int, attn_drop: float = 0.0, position_bias: bool = True
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.position_bias = position_bias
        if position_bias:
            self.pos = DynamicPosBias(dim // 4, num_heads)
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, h: int, w: int
    ) -> torch.Tensor:
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.position_bias:
            bias_h = torch.arange(1 - h, h, device=attn.device)
            bias_w = torch.arange(1 - w, w, device=attn.device)
            biases = torch.stack(torch.meshgrid(bias_h, bias_w, indexing="ij"))
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()

            coords = torch.stack(
                torch.meshgrid(
                    torch.arange(h, device=attn.device),
                    torch.arange(w, device=attn.device),
                    indexing="ij",
                )
            )
            coords = torch.flatten(coords, 1)
            relative = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0).contiguous()
            relative[:, :, 0] += h - 1
            relative[:, :, 1] += w - 1
            relative[:, :, 0] *= 2 * w - 1
            rpi = relative.sum(-1)

            pos = self.pos(biases)
            bias = pos[rpi.view(-1)].view(h * w, h * w, -1).permute(2, 0, 1).contiguous()
            attn = attn + bias.unsqueeze(0)

        attn = self.attn_drop(self.softmax(attn))
        return attn @ v


class GridAttention(nn.Module):
    """Two chained AffineTransforms: grid queries attend to k/v, then the
    original queries attend back through the grid."""

    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.attn_transform1 = AffineTransform(dim, num_heads, attn_drop=attn_drop)
        self.attn_transform2 = AffineTransform(dim, num_heads, attn_drop=attn_drop)

    def forward(self, qkv: torch.Tensor, grid: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b_, n, c = grid.shape
        qkv = qkv.reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        grid = grid.reshape(b_, n, self.num_heads, -1).permute(0, 2, 1, 3)
        q, k, v = qkv.unbind(0)
        x = self.attn_transform1(grid, k, v, h, w)
        x = self.attn_transform2(q, grid, x, h, w)
        return x.transpose(1, 2).reshape(b_, n, c)


# ---------------------------------------------------------------- blocks
class SEModule(nn.Module):
    """Squeeze-and-excitation channel attention (SiLU variant, as in HMA)."""

    def __init__(self, channels: int, rd_channels: int) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(channels, rd_channels, 1, bias=True)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(rd_channels, channels, 1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        se = x.mean((2, 3), keepdim=True)
        se = self.fc2(self.act(self.fc1(se)))
        return x * self.gate(se)


class FusedConv(nn.Module):
    """Standalone conv block (expand -> SE -> project) providing the conv
    inductive bias that HAT's HAB carries via its parallel CAB."""

    def __init__(self, num_feat: int, expand_size: int = 4, attn_ratio: int = 4) -> None:
        super().__init__()
        mid_feat = num_feat * expand_size
        self.pre_norm = nn.LayerNorm(num_feat)
        self.fused_conv = nn.Conv2d(num_feat, mid_feat, 3, 1, 1)
        self.norm1 = nn.LayerNorm(mid_feat)
        self.act1 = nn.GELU()
        self.se = SEModule(mid_feat, int(mid_feat / attn_ratio))
        self.conv3_1x1 = nn.Conv2d(mid_feat, num_feat, 1, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_size: Tuple[int, int],
        rpi: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        h, w = x_size
        b, _, c = x.shape
        shortcut = x
        x = x.view(b, h, w, c)
        x = self.pre_norm(x).permute(0, 3, 1, 2)
        x = self.fused_conv(x).permute(0, 2, 3, 1).contiguous()
        x = self.act1(self.norm1(x).permute(0, 3, 1, 2).contiguous())
        x = self.se(x)
        x = self.conv3_1x1(x).permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
        return x + shortcut


class FAB(nn.Module):
    """Fused Attention Block: a plain Swin block ((S)W-MSA + MLP); no
    parallel CAB — see :class:`FusedConv`."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 16,
        shift_size: int = 0,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: type = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = norm_layer(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn = WindowAttention(
            dim, window_size, num_heads, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(
        self,
        x: torch.Tensor,
        x_size: Tuple[int, int],
        rpi_sa: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        h, w = x_size
        b, _, c = x.shape

        shortcut = x
        x = self.norm1(x).view(b, h, w, c)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)
        attn_windows = self.attn(self.qkv(x_windows), rpi=rpi_sa, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)

        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x

        x = shortcut + self.drop_path(attn_x.view(b, h * w, c))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class GAB(nn.Module):
    """Grid Attention Block: the qkv channels are split between two window
    branches (plain + shifted, dim/4 each) and a grid branch (dim/2) that is
    ``grid_shuffle``d so pixels ``interval_size`` apart attend globally.

    Requires h, w divisible by both ``window_size`` and ``interval_size``
    (the wrapper's pad-to-16 covers interval 4).
    """

    def __init__(
        self,
        window_size: int,
        interval_size: int,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.interval_size = interval_size
        self.shift_size = window_size // 2

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.grid_proj = nn.Linear(dim, dim // 2)
        self.grid_attn = GridAttention(dim // 2, num_heads // 2, attn_drop=attn_drop)
        self.window_attn = WindowAttention(
            dim // 4, window_size, num_heads // 2, attn_drop=attn_drop, proj_drop=drop
        )
        self.window_attn_s = WindowAttention(
            dim // 4, window_size, num_heads // 2, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.fc = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(
        self,
        x: torch.Tensor,
        x_size: Tuple[int, int],
        rpi_sa: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        h, w = x_size
        b, _, c = x.shape
        ws = self.window_size
        shortcut = x

        qkv = self.qkv(x)
        x_window, x_qkv = torch.split(qkv, c * 3 // 2, dim=-1)

        x = x.view(b, h, w, c)
        gh, gw = h // self.interval_size, w // self.interval_size
        x_grid = self.grid_proj(grid_shuffle(x, h, w, c, self.interval_size).view(-1, gh * gw, c))
        x_qkv = grid_shuffle(
            x_qkv.view(b, h, w, c * 3 // 2), h, w, c * 3 // 2, self.interval_size
        ).view(-1, gh * gw, c * 3 // 2)

        # grid (sparse-global) branch -> dim/2
        x_grid_attn = self.grid_attn(x_qkv, x_grid, gh, gw).view(-1, gh, gw, c // 2)
        x_grid_attn = grid_unshuffle(x_grid_attn, b, h, w, self.interval_size).view(
            b, h * w, c // 2
        )

        # window branches (plain + shifted) -> dim/4 each
        x_window, x_window_s = torch.split(x_window.view(b, h, w, c * 3 // 2), c * 3 // 4, dim=-1)
        x_window = window_partition(x_window, ws).view(-1, ws * ws, c * 3 // 4)

        # Official quirk #1, kept bug-for-bug for pretrained-weight fidelity:
        # the shifted branch is NOT window-partitioned — after the spatial
        # roll it is reshaped in raster order, so each "window" the shifted
        # attention sees is a (ws*ws)-pixel raster strip, not a ws x ws tile.
        x_window_s = torch.roll(x_window_s, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        x_window_s = x_window_s.view(-1, ws * ws, c * 3 // 4)

        x_win_attn = self.window_attn(x_window, rpi=rpi_sa, mask=None).view(-1, ws, ws, c // 4)
        x_win_attn = window_reverse(x_win_attn, ws, h, w).view(b, h * w, c // 4)

        x_win_s_attn = self.window_attn_s(x_window_s, rpi=rpi_sa, mask=mask).view(
            -1, ws, ws, c // 4
        )
        x_win_s_attn = window_reverse(x_win_s_attn, ws, h, w).view(b, h * w, c // 4)
        # Official quirk #2: the reverse roll is applied after flattening to
        # (B, H*W, C), so it rolls the sequence and channel dims, not (h, w).
        x_win_s_attn = torch.roll(
            x_win_s_attn, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
        )

        x = torch.cat([x_win_attn, x_win_s_attn, x_grid_attn], dim=-1)
        x = self.norm1(self.fc(x))

        x = shortcut + self.drop_path(x)
        # NOTE: post-norm MLP (norm2 after mlp) — faithful to the official code.
        return x + self.drop_path(self.norm2(self.mlp(x)))


# ---------------------------------------------------------------- groups
class AttenBlocks(nn.Module):
    """One group: [FusedConv + FAB(0)] / FAB(ws/2) alternation, then a GAB
    joined by a learnable per-channel scale."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        interval_size: int,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: Sequence[float] = (0.0,),
        norm_layer: type = nn.LayerNorm,
    ) -> None:
        super().__init__()
        blocks = []
        for i in range(depth):
            if i % 2 == 0:
                blocks.append(FusedConv(num_feat=dim, expand_size=6, attn_ratio=2))
            blocks.append(
                FAB(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if i % 2 == 0 else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i],
                    norm_layer=norm_layer,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.gab = GAB(
            window_size=window_size,
            interval_size=interval_size,
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            drop=drop,
            drop_path=0.0,
            mlp_ratio=mlp_ratio,
        )
        self.scale = nn.Parameter(torch.empty(dim))
        trunc_normal_(self.scale, std=0.02)

    def forward(self, x: torch.Tensor, x_size: Tuple[int, int], params: dict) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, x_size, params["rpi_sa"], params["attn_mask"])
        y = self.gab(x, x_size, params["rpi_sa"], params["attn_mask"])
        return x + y * self.scale


class RHTB(nn.Module):
    """Residual Hybrid Transformer Block: AttenBlocks + conv + group residual."""

    def __init__(self, dim: int, norm_layer: type = nn.LayerNorm, **atten_kwargs) -> None:
        super().__init__()
        self.residual_group = AttenBlocks(dim, norm_layer=norm_layer, **atten_kwargs)
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)  # '1conv' resi_connection
        self.patch_embed = PatchEmbed(dim)
        self.patch_unembed = PatchUnEmbed(dim)

    def forward(self, x: torch.Tensor, x_size: Tuple[int, int], params: dict) -> torch.Tensor:
        out = self.residual_group(x, x_size, params)
        return self.patch_embed(self.conv(self.patch_unembed(out, x_size))) + x


# ----------------------------------------------------------------- model
class HMANet(nn.Module):
    """HMANet SR network, stripped of internal mean handling (spec §3.1).

    Defaults are the official SR x4 config (``options/test/HMA_SRx4.yml``):
    6 groups of depth 6. For capacity parity with the winner's HAT-L branch
    ("HMA-L", spec §5) pass ``depths=(6,)*12, num_heads=(6,)*12`` — but note
    no public pretrained weights exist at that scale.
    """

    def __init__(
        self,
        in_chans: int = 3,
        embed_dim: int = 180,
        depths: Sequence[int] = (6,) * 6,
        num_heads: Sequence[int] = (6,) * 6,
        window_size: int = 16,
        interval_size: int = 4,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: type = nn.LayerNorm,
        upscale: int = 4,
        num_feat: int = 64,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.interval_size = interval_size
        self.upscale = upscale

        self.register_buffer("relative_position_index_SA", self._calculate_rpi_sa())

        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        self.patch_embed = PatchEmbed(embed_dim, norm_layer)
        self.patch_unembed = PatchUnEmbed(embed_dim)
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = [v.item() for v in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i, depth in enumerate(depths):
            self.layers.append(
                RHTB(
                    embed_dim,
                    depth=depth,
                    num_heads=num_heads[i],
                    window_size=window_size,
                    interval_size=interval_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                    norm_layer=norm_layer,
                )
            )
        self.norm = norm_layer(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
        )
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, in_chans, 3, 1, 1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _calculate_rpi_sa(self) -> torch.Tensor:
        ws = self.window_size
        coords = torch.stack(
            torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing="ij")
        )
        coords = torch.flatten(coords, 1)
        relative = coords[:, :, None] - coords[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[:, :, 0] += ws - 1
        relative[:, :, 1] += ws - 1
        relative[:, :, 0] *= 2 * ws - 1
        return relative.sum(-1)

    def _calculate_mask(self, x_size: Tuple[int, int]) -> torch.Tensor:
        """Shifted-window attention mask for the given feature size."""
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))
        slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for hs in slices:
            for ws_ in slices:
                img_mask[:, hs, ws_, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x_size = (x.shape[2], x.shape[3])
        params = {
            "attn_mask": self._calculate_mask(x_size).to(x.device),
            "rpi_sa": self.relative_position_index_SA,
        }
        x = self.pos_drop(self.patch_embed(x))
        for layer in self.layers:
            x = layer(x, x_size, params)
        return self.patch_unembed(self.norm(x), x_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_first(x)
        x = self.conv_after_body(self.forward_features(x)) + x
        x = self.conv_before_upsample(x)
        return self.conv_last(self.upsample(x))


# ---------------------------------------------------- standalone wrapper
class HMANetSR(nn.Module):
    """Standalone HMANet SR model for stage-1 pretraining (``arch: hmanet``).

    Wraps the stripped :class:`HMANet` core with the padding and DIV2K-mean
    normalization that :class:`models.srhmcnet.SRHMCnet` would otherwise
    provide, so the bare branch can be trained on (LR, HR) pairs directly.
    Stage 2 initializes the composite's SR branch from this model's
    checkpoint (``SRHMCnet(pretrained_srhnet=...)`` strips the ``net.``
    prefix).
    """

    DIV2K_RGB_MEAN = (0.4488, 0.4371, 0.4040)

    def __init__(
        self,
        upscale: int = 4,
        window_size: int = 16,
        img_range: float = 1.0,
        **hma_kwargs,
    ) -> None:
        super().__init__()
        self.upscale = upscale
        self.window_size = window_size
        self.img_range = img_range
        self.net = HMANet(window_size=window_size, upscale=upscale, **hma_kwargs)
        self.register_buffer("mean", torch.tensor(self.DIV2K_RGB_MEAN).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        x = (x - self.mean) * self.img_range
        out = self.net(x) / self.img_range + self.mean
        return out[..., : h * self.upscale, : w * self.upscale]
