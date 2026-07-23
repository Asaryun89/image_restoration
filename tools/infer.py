"""Batch inference: restore a folder of degraded images with a checkpoint.

Saves one restored PNG per input (tiled 256/32 by default, the validation
protocol), plus an optional side-by-side panel — bicubic-upsampled input |
restored | GT (GT column only when --gt is given).

    python tools/infer.py --checkpoint checkpoints/oldphoto_stageB/best_psnr.pth \
        --input data/oldphoto_val/lr --gt data/oldphoto_val/hr \
        --out results/stageB_val --max-images 12

Real photographs (spec §7): pre-resize the scan so 4x its size is the
desired output (~2000-4000 px long side) before feeding it here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import list_images  # noqa: E402
from evaluate import load_generator  # noqa: E402
from models.tiling import tiled_forward  # noqa: E402


def _load(path: Path, device: torch.device) -> torch.Tensor:
    img = np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)


def _save(t: torch.Tensor, path: Path) -> None:
    arr = (t.squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).round()
    Image.fromarray(arr.astype(np.uint8)).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch restoration inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None,
                        help="config for the model (release checkpoints only)")
    parser.add_argument("--input", required=True, help="folder of degraded (LR) images")
    parser.add_argument("--gt", default=None,
                        help="optional folder of matching GT images for the panel")
    parser.add_argument("--out", required=True, help="output folder")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--tile", type=int, default=256, help="LR-side tile (0 = whole image)")
    parser.add_argument("--tile-overlap", type=int, default=32)
    parser.add_argument("--no-panel", action="store_true", help="save restored images only")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_generator(args.checkpoint, device, args.config)
    scale = int(cfg["model"].get("upscale", 4))

    paths = list_images(args.input)
    if args.max_images is not None:
        paths = paths[: args.max_images]
    gt_paths = {p.stem: p for p in list_images(args.gt)} if args.gt else {}

    out_dir = Path(args.out)
    restored_dir = out_dir / "restored"
    restored_dir.mkdir(parents=True, exist_ok=True)
    panel_dir = out_dir / "panels"
    if not args.no_panel:
        panel_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for path in tqdm(paths, desc="infer"):
            lq = _load(path, device)
            if args.tile:
                sr = tiled_forward(model, lq, args.tile, args.tile_overlap, scale)
            else:
                sr = model(lq)
            sr = sr.clamp(0, 1)
            _save(sr, restored_dir / f"{path.stem}.png")

            if not args.no_panel:
                cols = [
                    F.interpolate(lq, size=sr.shape[-2:], mode="bicubic",
                                  align_corners=False).clamp(0, 1),
                    sr,
                ]
                gt_path = gt_paths.get(path.stem)
                if gt_path is not None:
                    gt = _load(gt_path, device)[..., : sr.shape[-2], : sr.shape[-1]]
                    if gt.shape == sr.shape:
                        cols.append(gt)
                _save(torch.cat(cols, dim=-1), panel_dir / f"{path.stem}.png")

    print(f"{len(paths)} images -> {restored_dir}"
          + ("" if args.no_panel else f" (panels in {panel_dir})"))


if __name__ == "__main__":
    main()
