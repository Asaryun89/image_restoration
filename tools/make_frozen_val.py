"""Build the frozen synthetic validation set (old-photo spec §4, track 1).

Degrades held-out HR images once with fixed seeds and writes the LR/HR
pairs to disk, so PSNR-Y/SSIM on them is comparable across epochs. The
trainer then consumes them as an ordinary paired val set
(``data.val_gt: <out>/hr``, ``data.val_lq: <out>/lr``).

Each image is seeded as ``seed + index``, so the set is stable regardless
of ``--max`` and can be extended without changing existing pairs.
Regenerate only when ``degradation_version`` bumps — and re-baseline.

Usage:
    python tools/make_frozen_val.py --config configs/srhc_oldphoto_small.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import list_images  # noqa: E402
from data.degradation import OldPhotoDegradation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML with a degradation: section")
    parser.add_argument("--hr-root", default="data/DIV2K/DIV2K_valid_HR")
    parser.add_argument("--out", default="data/oldphoto_val")
    parser.add_argument("--max", type=int, default=100, help="number of val images")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    deg_cfg = cfg.get("degradation") or {}
    scale = int(cfg.get("model", {}).get("upscale", 4))

    device = torch.device(args.device)
    degrader = OldPhotoDegradation(deg_cfg, scale=scale).to(device)

    out = Path(args.out)
    (out / "hr").mkdir(parents=True, exist_ok=True)
    (out / "lr").mkdir(parents=True, exist_ok=True)

    paths = list_images(args.hr_root)[: args.max]
    for idx, path in enumerate(paths):
        torch.manual_seed(args.seed + idx)
        img = np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0
        h, w = (img.shape[0] // scale) * scale, (img.shape[1] // scale) * scale
        hr = (
            torch.from_numpy(np.ascontiguousarray(img[:h, :w].transpose(2, 0, 1)))
            .unsqueeze(0)
            .to(device)
        )
        lr = degrader(hr)

        for name, tensor in (("hr", hr), ("lr", lr)):
            arr = (
                (tensor[0].clamp(0, 1) * 255.0)
                .round()
                .byte()
                .cpu()
                .numpy()
                .transpose(1, 2, 0)
            )
            Image.fromarray(arr).save(out / name / f"{path.stem}.png")
        print(f"[{idx + 1}/{len(paths)}] {path.name}: HR {h}x{w} -> LR {h//scale}x{w//scale}")

    manifest = {
        "degradation_version": degrader.version,
        "seed": args.seed,
        "count": len(paths),
        "hr_root": str(args.hr_root),
        "scale": scale,
        "degradation_cfg": deg_cfg,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(paths)} frozen pairs + manifest to {out}")


if __name__ == "__main__":
    main()
