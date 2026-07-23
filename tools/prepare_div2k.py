"""Download official DIV2K (HR + bicubic x4 LR) from Hugging Face and
pre-crop training sub-images (spec §B.2).

- Train: DIV2K 800 HR + official bicubic x4 LR, cropped into 480x480 HR /
  120x120 LR sub-images (stride 240 / 60 — the BasicSR convention) for IO
  throughput.
- Val: DIV2K val 100 HR + bicubic x4 LR, kept as full images.

Source: the ``bezdarnost/DF2K-bicubic`` dataset repo. Its 4-digit-named
files (``hr/0001.png``-``hr/0900.png``, ``x4/0001x4.png``-``x4/0900x4.png``)
are the complete official DIV2K — train 0001-0800 plus val 0801-0900 under
their original ETH numbering, HR and the official bicubic LR alike
(verified pixel-identical against the ETH originals on 2026-07-18; the
repo's 6-digit files are Flickr2K and are not downloaded).

Usage:
    python tools/prepare_div2k.py --root data/DIV2K
    python tools/prepare_div2k.py --root data/DIV2K --skip-download  # re-crop only
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image
from tqdm import tqdm

HF_REPO = "bezdarnost/DF2K-bicubic"

# Official DIV2K split boundaries (by original image number).
TRAIN_RANGE = range(1, 801)
VAL_RANGE = range(801, 901)

# Final layout mirrors the official zips, so configs/tools stay unchanged.
DIRS = {
    "train_hr": "DIV2K_train_HR",
    "train_lr": "DIV2K_train_LR_bicubic/X4",
    "val_hr": "DIV2K_valid_HR",
    "val_lr": "DIV2K_valid_LR_bicubic/X4",
}


def download_and_arrange(root: Path) -> None:
    """Fetch the official DIV2K files from HF and lay them out officially."""
    targets = {k: root / d for k, d in DIRS.items()}
    if all(t.is_dir() and any(t.glob("*.png")) for t in targets.values()):
        print("official layout already present, skipping download")
        return

    staging = root / "_hf_div2k"
    # ????.png / ??????.png match only the 4-digit (DIV2K) names — the
    # 6-digit Flickr2K files stay excluded.
    snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        allow_patterns=["hr/????.png", "x4/??????.png"],
        local_dir=staging,
    )

    n_hr = sum(1 for _ in (staging / "hr").glob("*.png"))
    n_lr = sum(1 for _ in (staging / "x4").glob("*.png"))
    if n_hr != 900 or n_lr != 900:
        sys.exit(f"ERROR: expected 900 HR / 900 LR files from {HF_REPO}, got {n_hr}/{n_lr}")

    for t in targets.values():
        t.mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(1, 901), desc="arranging"):
        split = "train" if i in TRAIN_RANGE else "val"
        (staging / "hr" / f"{i:04d}.png").rename(targets[f"{split}_hr"] / f"{i:04d}.png")
        (staging / "x4" / f"{i:04d}x4.png").rename(targets[f"{split}_lr"] / f"{i:04d}x4.png")
    shutil.rmtree(staging)
    print(f"train: {len(TRAIN_RANGE)} pairs, val: {len(VAL_RANGE)} pairs")


def make_sub_images(
    src: Path, dst: Path, crop: int, stride: int, min_keep: int
) -> None:
    """Crop every image into overlapping ``crop`` tiles at ``stride``.

    Edge tiles are shifted inward so every pixel is covered and all tiles
    are full-size (BasicSR extract_subimages behaviour). ``min_keep``
    short-circuits when the folder already holds that many files.
    """
    dst.mkdir(parents=True, exist_ok=True)
    if sum(1 for _ in dst.glob("*.png")) >= min_keep:
        print(f"{dst.name}: already prepared, skipping")
        return
    paths = sorted(src.glob("*.png"))
    for path in tqdm(paths, desc=f"sub-images {src.name}"):
        img = np.asarray(Image.open(path))
        h, w = img.shape[:2]
        ys = sorted({min(y, h - crop) for y in range(0, h - crop + stride, stride)})
        xs = sorted({min(x, w - crop) for x in range(0, w - crop + stride, stride)})
        idx = 0
        for y in ys:
            for x in xs:
                idx += 1
                tile = img[y : y + crop, x : x + crop]
                Image.fromarray(tile).save(dst / f"{path.stem}_s{idx:03d}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare DIV2K for the team17 reproduction")
    parser.add_argument("--root", default="data/DIV2K", help="dataset root directory")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--hr-crop", type=int, default=480)
    parser.add_argument("--hr-stride", type=int, default=240)
    args = parser.parse_args()

    root = Path(args.root)
    if not args.skip_download:
        download_and_arrange(root)

    scale = 4
    if args.hr_crop % scale or args.hr_stride % scale:
        sys.exit("--hr-crop/--hr-stride must be divisible by 4")
    make_sub_images(
        root / "DIV2K_train_HR", root / "DIV2K_train_HR_sub",
        crop=args.hr_crop, stride=args.hr_stride, min_keep=800,
    )
    make_sub_images(
        root / "DIV2K_train_LR_bicubic/X4", root / "DIV2K_train_LR_bicubic_X4_sub",
        crop=args.hr_crop // scale, stride=args.hr_stride // scale, min_keep=800,
    )
    n_hr = sum(1 for _ in (root / "DIV2K_train_HR_sub").glob("*.png"))
    n_lr = sum(1 for _ in (root / "DIV2K_train_LR_bicubic_X4_sub").glob("*.png"))
    print(f"train sub-images: {n_hr} HR / {n_lr} LR")
    if n_hr != n_lr:
        sys.exit(f"ERROR: HR/LR sub-image counts differ ({n_hr} vs {n_lr})")
    print("done")


if __name__ == "__main__":
    main()
