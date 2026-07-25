"""SRHCnet x4 super-resolution inference — webapp-ready.

Loads the NTIRE-2025 winner composite (HAT-L + gated NAFNet refiner) and
upscales RGB images x4. Works with the released checkpoint
(`17_SRHCnet_39000_final.pth`, `params` container) and this repo's trainer
checkpoints (EMA weights are preferred automatically).

CLI:
    python infer.py --checkpoint 17_SRHCnet_39000_final.pth \
        --input in.png --output out.png [--tile 256] [--device cuda]

Webapp: import and reuse one loaded model across requests —
    from infer import SRUpscaler
    sr = SRUpscaler("17_SRHCnet_39000_final.pth", device="cuda")
    out_pil = sr.upscale(in_pil)          # PIL.Image -> PIL.Image (4x)

Memory: the NAFNet stage runs at output resolution, so whole-image
forwards on large inputs are expensive; `tile` (default 256, LR side)
caps peak VRAM at roughly 6 GB and stitches overlapping tiles seamlessly.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image

from models.rrdbnet import RRDBNet
from models.srhcnet import SRHCnet
from models.tiling import tiled_forward


def _build(arch: str, checkpoint: str) -> torch.nn.Module:
    if arch == "srhcnet":
        return SRHCnet(pretrained=checkpoint, gate_init="identity")
    if arch == "esrgan":
        return RRDBNet(pretrained=checkpoint)
    raise ValueError(f"Unknown arch: {arch!r} (expected srhcnet | esrgan)")


class SRUpscaler:
    """Load once, upscale many. Thread-unsafe; serialize calls per GPU."""

    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda",
        tile: int = 256,
        arch: str = "srhcnet",
    ) -> None:
        self.device = torch.device(device)
        self.tile = tile
        self.arch = arch
        self.model = _build(arch, checkpoint).eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def upscale(self, img: Image.Image) -> Image.Image:
        arr = np.asarray(img.convert("RGB"), np.float32) / 255.0
        x = torch.from_numpy(arr.transpose(2, 0, 1))[None].to(self.device)
        if self.tile:
            y = tiled_forward(self.model, x, self.tile, tile_overlap=32, scale=4)
        else:
            y = self.model(x)
        out = y.clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0)
        return Image.fromarray((out * 255.0).round().astype(np.uint8))


def main() -> None:
    ap = argparse.ArgumentParser(description="SRHCnet x4 upscaling")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tile", type=int, default=256,
                    help="LR-side tile size; 0 = whole-image forward")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--arch", default="srhcnet", choices=["srhcnet", "esrgan"])
    args = ap.parse_args()

    sr = SRUpscaler(args.checkpoint, device=args.device, tile=args.tile, arch=args.arch)
    sr.upscale(Image.open(args.input)).save(args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
