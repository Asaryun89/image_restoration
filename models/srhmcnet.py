"""SRHMCnet — the NTIRE-winner pipeline with an HMANet SR branch.

Per ``SRHCnet_HMANet_swap_spec.md``: the winning SRHCnet pipeline is kept —

    input -> SR branch -> beta gate -> NAFNet refiner -> gamma gate -> output

with :class:`models.hmanet.HMANet` as the SR branch (stripped of its
internal mean handling so the wrapper's normalization is the only one).
This is the stage-2 model: its SR branch is initialized from a stage-1
:class:`models.hmanet.HMANetSR` checkpoint via ``pretrained_srhnet``.

Capacity note (spec §5): the default ``depths=(6,)*6`` is the official HMA
x4 config — roughly HAT(base) capacity, *below* the winner's HAT-L (12
groups). For parity pass ``depths=(6,)*12, num_heads=(6,)*12`` ("HMA-L"),
for which no public pretrained weights exist.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hmanet import HMANet
from models.nafnet import NAFNet


def _load_state(path: str) -> dict:
    """Load a checkpoint and unwrap the common containers.

    Handles basicsr-style checkpoints (``params_ema``/``params`` — the
    official HMA release stores EMA weights under ``params_ema``) and this
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


class SRHMCnet(nn.Module):
    """HMANet upscaler + gated NAFNet refiner (see module docstring).

    ``interval_size`` is HMA's grid stride (window 16 is a multiple of
    interval 4, so the wrapper's pad-to-16 satisfies both constraints —
    spec §1).

    Weight initialization (spec §3.3):
        pretrained_srhnet: path to a stage-1 :class:`HMANetSR` checkpoint
            (``net.*`` prefix stripped automatically) or the official
            ``HMA_SRx4_pretrain.pth``; loaded into the SR branch (EMA
            weights preferred).
        pretrained_refiner: path to a checkpoint containing ``srcnet.*``
            weights and ``beta``/``gamma`` gates to copy (the refiner
            consumes an RGB HR image regardless of which network produced
            it).
        gate_init: ``"normal"`` — N(0,1) as in the winner's code; or
            ``"identity"`` — beta=1, gamma=0 (spec §6.4's recovery init,
            pass-through of the SR branch at step 0). A gate loaded via
            ``pretrained_refiner`` overrides this.
        freeze_srhnet: freeze the SR branch (spec §4 step 1 warm-up:
            train only refiner + gates).
    """

    DIV2K_RGB_MEAN = (0.4488, 0.4371, 0.4040)

    def __init__(
        self,
        upscale: int = 4,
        window_size: int = 16,
        img_range: float = 1.0,
        embed_dim: int = 180,
        depths: Sequence[int] = (6,) * 6,
        num_heads: Sequence[int] = (6,) * 6,
        interval_size: int = 4,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.1,
        num_feat: int = 64,
        srcnet: Optional[dict] = None,
        pretrained_srhnet: Optional[str] = None,
        pretrained_refiner: Optional[str] = None,
        gate_init: str = "normal",
        freeze_srhnet: bool = False,
    ) -> None:
        super().__init__()
        self.upscale = upscale
        self.window_size = window_size
        self.img_range = img_range

        self.srhnet = HMANet(
            embed_dim=embed_dim,
            depths=tuple(depths),
            num_heads=tuple(num_heads),
            window_size=window_size,
            interval_size=interval_size,
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
        self.register_buffer("mean", torch.tensor(self.DIV2K_RGB_MEAN).view(1, 3, 1, 1))

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
        # Stage-1 checkpoints wrap the core: HMANetSR stores it as `net.*`
        # (plus a `mean` buffer); an SRHMCnet checkpoint as `srhnet.*`.
        # Official HMA releases are bare-core already (`conv_first.*`).
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
                f"pretrained_srhnet {path} does not match the HMANet branch "
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

    # ------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # identical to SRHCnet
        h, w = x.shape[-2:]
        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        x = (x - self.mean) * self.img_range
        srh_out = self.srhnet(x) * self.beta / self.img_range + self.mean
        src_out = srh_out + self.srcnet(srh_out) * self.gamma
        return src_out[..., : h * self.upscale, : w * self.upscale]
