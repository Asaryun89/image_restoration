"""SRHCnet — NTIRE 2025 SR x4 winner (SamsungAICamera / SRC-B).

Two sub-networks in sequence, joined by learnable per-channel gates:

    SRHnet (HAT transformer, does the x4 upscaling)
        -> * beta, de-normalize
    SRCnet (NAFNet U-Net, same-resolution refiner with internal skip)
        -> gamma-gated refinement delta: out = y + (srcnet(y) - y) * gamma

Implementation follows ``SRHCnet_architecture_spec.md`` (derived from the
official validated code); SRHnet is a faithful HAT ("Activating More Pixels
in Image Super-Resolution Transformer", Chen et al.), SRCnet reuses this
repo's :class:`models.nafnet.NAFNet`.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from models.init_utils import icnr_init
from models.nafnet import NAFNet


def _load_state(path: str) -> dict:
    """Load a checkpoint and unwrap the common containers.

    Handles basicsr-style checkpoints (``params``/``params_ema`` — the
    team17 release stores the full composite under ``params``) and this
    repo's trainer checkpoints (``ema``/``generator``).
    """
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # This repo's trainer checkpoints carry cfg/RNG state, which the
        # weights_only unpickler rejects; they are trusted local files.
        state = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("params_ema", "params", "ema", "generator"):
        if isinstance(state, dict) and key in state:
            return state[key]
    return state


# --------------------------------------------------------------- primitives
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


class ChannelAttention(nn.Module):
    def __init__(self, num_feat: int, squeeze_factor: int = 16) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.attention(x)


class CAB(nn.Module):
    """Conv attention block run in parallel with window attention in HAB."""

    def __init__(self, num_feat: int, compress_ratio: int = 3, squeeze_factor: int = 30) -> None:
        super().__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cab(x)


# ---------------------------------------------------------------- attention
class WindowAttention(nn.Module):
    """Window MSA with relative position bias; supports shifted windows."""

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, x: torch.Tensor, rpi: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.relative_position_bias_table[rpi.view(-1)].view(n, n, -1)
        attn = attn + bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = self.attn_drop(self.softmax(attn))

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        return self.proj_drop(self.proj(x))


class HAB(nn.Module):
    """Hybrid Attention Block: (shifted-)window MSA parallel with a CAB."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 16,
        shift_size: int = 0,
        compress_ratio: int = 3,
        squeeze_factor: int = 30,
        conv_scale: float = 0.01,
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
        self.conv_scale = conv_scale
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size, num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        self.conv_block = CAB(dim, compress_ratio, squeeze_factor)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(
        self,
        x: torch.Tensor,
        x_size: tuple,
        rpi_sa: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        h, w = x_size
        b, _, c = x.shape

        shortcut = x
        x = self.norm1(x).view(b, h, w, c)

        conv_x = self.conv_block(x.permute(0, 3, 1, 2))
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)
        attn_windows = self.attn(x_windows, rpi=rpi_sa, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)

        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x
        attn_x = attn_x.view(b, h * w, c)

        x = shortcut + self.drop_path(attn_x) + conv_x * self.conv_scale
        return x + self.drop_path(self.mlp(self.norm2(x)))


class OCAB(nn.Module):
    """Overlapping cross-attention: queries from ws windows, keys/values
    from overlapping (ws + overlap_ratio*ws) windows."""

    def __init__(
        self,
        dim: int,
        window_size: int,
        overlap_ratio: float,
        num_heads: int,
        qkv_bias: bool = True,
        mlp_ratio: float = 2.0,
        norm_layer: type = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size

        self.norm1 = norm_layer(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.unfold = nn.Unfold(
            kernel_size=(self.overlap_win_size, self.overlap_win_size),
            stride=window_size,
            padding=(self.overlap_win_size - window_size) // 2,
        )
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (window_size + self.overlap_win_size - 1)
                * (window_size + self.overlap_win_size - 1),
                num_heads,
            )
        )
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, x_size: tuple, rpi: torch.Tensor) -> torch.Tensor:
        h, w = x_size
        b, _, c = x.shape
        ws, ows = self.window_size, self.overlap_win_size

        shortcut = x
        x = self.norm1(x).view(b, h, w, c)

        qkv = self.qkv(x).reshape(b, h, w, 3, c).permute(3, 0, 4, 1, 2)  # 3, b, c, h, w
        q = qkv[0].permute(0, 2, 3, 1)  # b, h, w, c
        kv = torch.cat((qkv[1], qkv[2]), dim=1)  # b, 2c, h, w

        q_windows = window_partition(q, ws).view(-1, ws * ws, c)
        kv_windows = self.unfold(kv)  # b, 2*c*ows*ows, nw
        nw = kv_windows.shape[-1]
        kv_windows = (
            kv_windows.view(b, 2, c, ows * ows, nw)
            .permute(1, 0, 4, 3, 2)
            .reshape(2, b * nw, ows * ows, c)
        )
        k_windows, v_windows = kv_windows.unbind(0)

        b_, nq, _ = q_windows.shape
        n = ows * ows
        d = c // self.num_heads
        q = q_windows.reshape(b_, nq, self.num_heads, d).permute(0, 2, 1, 3)
        k = k_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3)
        v = v_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.relative_position_bias_table[rpi.view(-1)].view(ws * ws, n, -1)
        attn = attn + bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        attn = self.softmax(attn)

        attn_windows = (attn @ v).transpose(1, 2).reshape(b_, nq, c)
        x = window_reverse(attn_windows.view(-1, ws, ws, c), ws, h, w).view(b, h * w, c)
        x = self.proj(x) + shortcut
        return x + self.mlp(self.norm2(x))


# ------------------------------------------------------------------- groups
class AttenBlocks(nn.Module):
    """A series of HABs (alternating shift 0 / ws//2) followed by one OCAB."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        compress_ratio: int,
        squeeze_factor: int,
        conv_scale: float,
        overlap_ratio: float,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: Sequence[float] = (0.0,),
        norm_layer: type = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                HAB(
                    dim,
                    num_heads,
                    window_size=window_size,
                    shift_size=0 if i % 2 == 0 else window_size // 2,
                    compress_ratio=compress_ratio,
                    squeeze_factor=squeeze_factor,
                    conv_scale=conv_scale,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.overlap_attn = OCAB(
            dim,
            window_size=window_size,
            overlap_ratio=overlap_ratio,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
        )

    def forward(self, x: torch.Tensor, x_size: tuple, params: dict) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, x_size, params["rpi_sa"], params["attn_mask"])
        return self.overlap_attn(x, x_size, params["rpi_oca"])


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

    def forward(self, x: torch.Tensor, x_size: tuple) -> torch.Tensor:
        return x.transpose(1, 2).contiguous().view(x.shape[0], self.embed_dim, *x_size)


class RHAG(nn.Module):
    """Residual Hybrid Attention Group: AttenBlocks + conv + group residual."""

    def __init__(self, dim: int, norm_layer: type = nn.LayerNorm, **atten_kwargs) -> None:
        super().__init__()
        self.residual_group = AttenBlocks(dim, norm_layer=norm_layer, **atten_kwargs)
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)  # '1conv' resi_connection
        self.patch_embed = PatchEmbed(dim)
        self.patch_unembed = PatchUnEmbed(dim)

    def forward(self, x: torch.Tensor, x_size: tuple, params: dict) -> torch.Tensor:
        out = self.residual_group(x, x_size, params)
        out = self.patch_embed(self.conv(self.patch_unembed(out, x_size)))
        return out + x


class Upsample(nn.Sequential):
    """Pixel-shuffle upsampler for power-of-2 scales and scale 3."""

    def __init__(self, scale: int, num_feat: int) -> None:
        layers = []
        if (scale & (scale - 1)) == 0:  # 1, 2, 4, 8...
            for _ in range(int(math.log2(scale))):
                conv = nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1)
                icnr_init(conv, 2)
                layers.append(conv)
                layers.append(nn.PixelShuffle(2))
        elif scale == 3:
            conv = nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1)
            icnr_init(conv, 3)
            layers.append(conv)
            layers.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f"Unsupported upscale factor: {scale}")
        super().__init__(*layers)


# ------------------------------------------------------------------- SRHnet
class SRHnet(nn.Module):
    """HAT super-resolution transformer (stage 1 of SRHCnet).

    Defaults are the spec's HAT-L scale. Input is assumed already padded to
    a multiple of ``window_size`` and mean-normalized (both handled by the
    :class:`SRHCnet` wrapper).
    """

    def __init__(
        self,
        in_chans: int = 3,
        embed_dim: int = 180,
        depths: Sequence[int] = (6,) * 12,
        num_heads: Sequence[int] = (6,) * 12,
        window_size: int = 16,
        compress_ratio: int = 3,
        squeeze_factor: int = 30,
        conv_scale: float = 0.01,
        overlap_ratio: float = 0.5,
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
        self.overlap_ratio = overlap_ratio
        self.upscale = upscale

        self.register_buffer("relative_position_index_SA", self._calculate_rpi_sa())
        self.register_buffer("relative_position_index_OCA", self._calculate_rpi_oca())

        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        self.patch_embed = PatchEmbed(embed_dim, norm_layer)
        self.patch_unembed = PatchUnEmbed(embed_dim)
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = [v.item() for v in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i, depth in enumerate(depths):
            self.layers.append(
                RHAG(
                    embed_dim,
                    depth=depth,
                    num_heads=num_heads[i],
                    window_size=window_size,
                    compress_ratio=compress_ratio,
                    squeeze_factor=squeeze_factor,
                    conv_scale=conv_scale,
                    overlap_ratio=overlap_ratio,
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

    def _calculate_rpi_oca(self) -> torch.Tensor:
        ws_ori = self.window_size
        ws_ext = ws_ori + int(self.overlap_ratio * ws_ori)
        coords_ori = torch.stack(
            torch.meshgrid(torch.arange(ws_ori), torch.arange(ws_ori), indexing="ij")
        )
        coords_ori = torch.flatten(coords_ori, 1)
        coords_ext = torch.stack(
            torch.meshgrid(torch.arange(ws_ext), torch.arange(ws_ext), indexing="ij")
        )
        coords_ext = torch.flatten(coords_ext, 1)
        relative = coords_ext[:, None, :] - coords_ori[:, :, None]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[:, :, 0] += ws_ori - ws_ext + 1
        relative[:, :, 1] += ws_ori - ws_ext + 1
        relative[:, :, 0] *= ws_ori + ws_ext - 1
        return relative.sum(-1)

    def _calculate_mask(self, x_size: tuple) -> torch.Tensor:
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
            "rpi_oca": self.relative_position_index_OCA,
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


# ------------------------------------------------------------------ wrapper
class SRHCnet(nn.Module):
    """HAT upscaler + gated NAFNet refiner (see module docstring).

    Args mirror the spec; ``srcnet`` is a kwargs dict for the NAFNet refiner
    (spec: width 64, enc [2,2,4,8], middle 12, dec [2,2,2,2]).

    Weight initialization:
        pretrained: path to a FULL composite checkpoint — the team17 release
            (``params`` container covering exactly ``srhnet.*``/``srcnet.*``/
            ``beta``/``gamma``, spec A.2) or this trainer's own checkpoints.
            Loaded strict; overrides everything below.
        pretrained_srhnet: path to a HAT checkpoint for the SR branch — the
            official HAT releases (``params_ema``, bare-core keys) or a
            checkpoint wrapping the core as ``net.*``/``srhnet.*``.
        pretrained_refiner: checkpoint containing ``srcnet.*`` weights and
            ``beta``/``gamma`` gates to copy.
        gate_init: ``"normal"`` — N(0,1) as in the winner's code; or
            ``"identity"`` — beta=1, gamma=0 (composite == SR branch at
            step 0). A gate loaded via ``pretrained_refiner`` overrides this.
        freeze_srhnet: freeze the SR branch (train only refiner + gates).
    """

    DIV2K_RGB_MEAN = (0.4488, 0.4371, 0.4040)

    def __init__(
        self,
        upscale: int = 4,
        window_size: int = 16,
        img_range: float = 1.0,
        embed_dim: int = 180,
        depths: Sequence[int] = (6,) * 12,
        num_heads: Sequence[int] = (6,) * 12,
        compress_ratio: int = 3,
        squeeze_factor: int = 30,
        conv_scale: float = 0.01,
        overlap_ratio: float = 0.5,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.1,
        num_feat: int = 64,
        srcnet: Optional[dict] = None,
        pretrained: Optional[str] = None,
        pretrained_srhnet: Optional[str] = None,
        pretrained_refiner: Optional[str] = None,
        gate_init: str = "normal",
        freeze_srhnet: bool = False,
    ) -> None:
        super().__init__()
        self.upscale = upscale
        self.window_size = window_size
        self.img_range = img_range

        self.srhnet = SRHnet(
            embed_dim=embed_dim,
            depths=tuple(depths),
            num_heads=tuple(num_heads),
            window_size=window_size,
            compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor,
            conv_scale=conv_scale,
            overlap_ratio=overlap_ratio,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_path_rate=drop_path_rate,
            upscale=upscale,
            num_feat=num_feat,
        )
        src_cfg = {
            "width": 64,
            "enc_blk_nums": (2, 2, 4, 8),
            "middle_blk_num": 12,
            "dec_blk_nums": (2, 2, 2, 2),
        }
        src_cfg.update(srcnet or {})
        self.srcnet = NAFNet(**src_cfg)

        if gate_init == "normal":
            self.beta = nn.Parameter(torch.normal(0.0, 1.0, (1, 3, 1, 1)))
            self.gamma = nn.Parameter(torch.normal(0.0, 1.0, (1, 3, 1, 1)))
        elif gate_init == "identity":
            self.beta = nn.Parameter(torch.ones(1, 3, 1, 1))
            self.gamma = nn.Parameter(torch.zeros(1, 3, 1, 1))
        else:
            raise ValueError(f"Unknown gate_init: {gate_init!r} (expected normal | identity)")
        # persistent=False: the team17 release's state dict covers exactly
        # srhnet.*/srcnet.*/beta/gamma (spec A.2) — a persistent mean buffer
        # would break `load_state_dict(..., strict=True)` against it.
        self.register_buffer(
            "mean", torch.tensor(self.DIV2K_RGB_MEAN).view(1, 3, 1, 1), persistent=False
        )

        if pretrained:
            self.load_state_dict(_load_state(pretrained), strict=True)
        if pretrained_srhnet:
            self._load_srhnet(pretrained_srhnet)
        if pretrained_refiner:
            self._load_refiner(pretrained_refiner)
        if freeze_srhnet:
            for p in self.srhnet.parameters():
                p.requires_grad_(False)

    # ------------------------------------------------------------- loading
    def _load_srhnet(self, path: str) -> None:
        state = _load_state(path)
        # Official HAT releases are bare-core (`conv_first.*`); wrapper
        # checkpoints store the core under `net.*` or `srhnet.*`.
        if not any(k.startswith("conv_first.") for k in state):
            for prefix in ("net.", "srhnet."):
                if any(k.startswith(prefix) for k in state):
                    state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
                    break
        state = dict(state)
        state.pop("mean", None)
        missing, unexpected = self.srhnet.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"pretrained_srhnet {path} does not match the HAT branch "
                f"(missing {len(missing)}: {missing[:5]}..., "
                f"unexpected {len(unexpected)}: {unexpected[:5]}...) — check "
                f"depths/embed_dim against the checkpoint's config"
            )

    def _load_refiner(self, path: str) -> None:
        state = _load_state(path)
        wanted = {k: v for k, v in state.items() if k.startswith("srcnet.") or k in ("beta", "gamma")}
        if not wanted:
            raise RuntimeError(f"pretrained_refiner {path} has no srcnet.*/beta/gamma keys")
        missing, unexpected = self.load_state_dict(wanted, strict=False)
        assert not unexpected, unexpected
        still_missing = [k for k in missing if k.startswith("srcnet.") or k in ("beta", "gamma")]
        if still_missing:
            raise RuntimeError(
                f"pretrained_refiner {path}: refiner keys not covered: {still_missing[:5]}..."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        x = (x - self.mean) * self.img_range
        srh_out = self.srhnet(x) * self.beta / self.img_range + self.mean
        # gamma gates the refinement DELTA, not the refiner's raw output:
        # NAFNet carries an internal input skip, so srcnet(y) is a full
        # image and the residual it contributes is srcnet(y) - y. Gating
        # the raw output (y + srcnet(y)*gamma) nearly doubles the signal
        # and collapses PSNR to ~14 dB; this form reproduces the released
        # checkpoint's ~34 dB on DIV2K val.
        src_out = srh_out + (self.srcnet(srh_out) - srh_out) * self.gamma
        return src_out[..., : h * self.upscale, : w * self.upscale]
