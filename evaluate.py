"""Evaluate a checkpoint on paired LR/HR folders (spec §B.7 protocol).

Reports Y-channel PSNR with a ``scale``-px border crop (the challenge
metric), plus RGB PSNR and SSIM. Works with both the team17 release
(``params``) and this trainer's checkpoints (EMA weights preferred).

Baseline (acceptance check 2 — before any training):
    python evaluate.py --checkpoint checkpoints/17_SRHCnet_39000_final.pth \
        --config configs/team17_div2k_ft.yaml

After fine-tuning (acceptance check 3):
    python evaluate.py --checkpoint checkpoints/team17_ft/best_psnr.pth

Full 2K images need a lot of activation memory (the NAFNet stage runs at
HR); pass ``--tile`` (LR-side tile edge, e.g. 256) if a whole-image
forward OOMs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
import yaml
from tqdm import tqdm

from data.dataset import PairedImageDataset, build_dataloader
from engine.checkpoint import CheckpointManager
from eval.metrics import SSIMMetric, psnr, psnr_y
from models import build_generator
from models.tiling import tiled_forward


def load_generator(checkpoint: str, device: torch.device, config: Optional[str]) -> tuple:
    """Build the generator and load weights (EMA > generator > params)."""
    state = CheckpointManager.load(checkpoint, map_location=str(device))
    if config is not None:
        with open(config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    elif isinstance(state, dict) and "cfg" in state:
        cfg = state["cfg"]
    else:
        raise SystemExit(
            f"{checkpoint} carries no config (release checkpoint?) — pass --config"
        )
    model_cfg = dict(cfg["model"])
    model_cfg.pop("pretrained", None)  # weights come from --checkpoint below
    model = build_generator(model_cfg).to(device)
    weights = state
    for key in ("ema", "generator", "params_ema", "params"):
        if isinstance(state, dict) and key in state:
            weights = state[key]
            break
    model.load_state_dict(weights)
    model.eval()
    return model, cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on paired LR/HR folders")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None,
                        help="config for model/data (required for release checkpoints)")
    parser.add_argument("--gt", default=None, help="override val GT root")
    parser.add_argument("--lq", default=None, help="override val LQ root")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--out", default=None, help="results JSON path")
    parser.add_argument(
        "--tile", type=int, default=None,
        help="LR-side tile size for overlap-tiled inference (default: whole image)",
    )
    parser.add_argument("--tile-overlap", type=int, default=32, help="LR-side tile overlap")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_generator(args.checkpoint, device, args.config)
    scale = int(cfg["model"].get("upscale", 4))

    data_cfg = cfg.get("data", {})
    gt_root = args.gt or data_cfg["val_gt"]
    lq_root = args.lq or data_cfg["val_lq"]
    dataset = PairedImageDataset(gt_root, lq_root, split="val", gt_size=None, scale=scale)
    if args.max_images is not None:
        dataset.gt_paths = dataset.gt_paths[: args.max_images]
        dataset.lq_paths = dataset.lq_paths[: args.max_images]
    loader = build_dataloader(dataset, batch_size=1, num_workers=2)

    ssim_metric = SSIMMetric().to(device)
    per_image = []
    with torch.no_grad():
        for i, (lq, gt) in enumerate(tqdm(loader, desc="eval")):
            lq = lq.to(device)
            gt = gt.to(device)
            if args.tile:
                restored = tiled_forward(
                    model, lq, args.tile, args.tile_overlap, scale
                ).clamp(0, 1)
            else:
                restored = model(lq).clamp(0, 1)
            per_image.append(
                {
                    "image": dataset.gt_paths[i].name,
                    # Challenge metric: Y-channel PSNR, scale-px border crop.
                    "psnr_y": psnr_y(restored, gt, border=scale).item(),
                    "psnr": psnr(restored, gt).item(),
                    "ssim": ssim_metric(restored, gt).item(),
                }
            )

    keys = ["psnr_y", "psnr", "ssim"]
    means = {k: sum(e[k] for e in per_image) / len(per_image) for k in keys}

    header = f"{'metric':<10}" + "".join(f"{k:>12}" for k in keys)
    print(header)
    print("-" * len(header))
    print(f"{'mean':<10}" + "".join(f"{means[k]:>12.4f}" for k in keys))
    print(f"({len(per_image)} images, checkpoint: {args.checkpoint})")

    out_path = Path(args.out) if args.out else Path(args.checkpoint).parent / "results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"checkpoint": args.checkpoint, "mean": means, "per_image": per_image},
            f,
            indent=2,
        )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
