"""Paired LR/HR dataset for the team17 DIV2K reproduction (spec §B.2).

Reads pre-degraded pairs from two folders — the official DIV2K bicubic x4
LR images against their HR sources (or the 480/120 sub-images produced by
``tools/prepare_div2k.py``). No synthetic degradation: the challenge's
degradation IS the official bicubic LR set.

Pairing is positional over the two sorted file lists (DIV2K naming keeps
them aligned: ``0001.png`` <-> ``0001x4.png``, sub-images likewise); a
count mismatch raises immediately.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(root: str) -> List[Path]:
    """Recursively collect image files under one root, sorted."""
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Data root does not exist: {root_path}")
    paths = sorted(p for p in root_path.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not paths:
        raise RuntimeError(f"No images found under {root_path}")
    return paths


def _load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float()


class PairedImageDataset(Dataset):
    """Returns ``(lq, gt)`` float32 CHW tensors in [0, 1].

    Train split: aligned random crop (``gt_size`` on the HR side, ``gt_size
    / scale`` on the LR side) plus random horizontal/vertical flips and
    90-degree rotations (spec §B.5's standard geometric augmentations —
    channel shuffle / mixing are batch-level and live in the trainer).
    Val split: full images, HR trimmed to ``LR * scale`` alignment.
    """

    def __init__(
        self,
        gt_root: str,
        lq_root: str,
        split: str = "train",
        gt_size: Optional[int] = 320,
        scale: int = 4,
        use_hflip: bool = True,
        use_rot: bool = True,
    ) -> None:
        if split not in ("train", "val"):
            raise ValueError(f"Unknown split: {split}")
        self.gt_paths = list_images(gt_root)
        self.lq_paths = list_images(lq_root)
        if len(self.gt_paths) != len(self.lq_paths):
            raise RuntimeError(
                f"Unpaired data: {len(self.gt_paths)} GT images under {gt_root} vs "
                f"{len(self.lq_paths)} LQ images under {lq_root}"
            )
        self.paths = self.gt_paths  # sample naming (evaluate.py, sample grids)
        self.split = split
        self.scale = int(scale)
        self.gt_size = gt_size
        self.use_hflip = bool(use_hflip)
        self.use_rot = bool(use_rot)
        if gt_size is not None and gt_size % self.scale:
            raise ValueError(f"gt_size {gt_size} not divisible by scale {self.scale}")

    def __len__(self) -> int:
        return len(self.gt_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        gt = _load(self.gt_paths[idx])
        lq = _load(self.lq_paths[idx])
        s = self.scale

        # Guard against off-by-scale size mismatches before cropping.
        if gt.shape[0] < lq.shape[0] * s or gt.shape[1] < lq.shape[1] * s:
            raise RuntimeError(
                f"GT {self.gt_paths[idx].name} {gt.shape[:2]} smaller than "
                f"LQ {self.lq_paths[idx].name} {lq.shape[:2]} x{s}"
            )

        if self.split == "train" and self.gt_size is not None:
            lq_size = self.gt_size // s
            if lq.shape[0] < lq_size or lq.shape[1] < lq_size:
                raise RuntimeError(
                    f"LQ {self.lq_paths[idx].name} {lq.shape[:2]} smaller than "
                    f"crop {lq_size} — use a larger source or smaller gt_size"
                )
            # Per-worker torch RNG (seeded by the DataLoader) drives all
            # randomness, so workers never repeat each other's samples.
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
            rng = np.random.default_rng(seed)
            y = int(rng.integers(0, lq.shape[0] - lq_size + 1))
            x = int(rng.integers(0, lq.shape[1] - lq_size + 1))
            lq = lq[y : y + lq_size, x : x + lq_size]
            gt = gt[y * s : (y + lq_size) * s, x * s : (x + lq_size) * s]

            if self.use_hflip and rng.random() < 0.5:
                lq, gt = lq[:, ::-1], gt[:, ::-1]
            if self.use_rot:
                if rng.random() < 0.5:
                    lq, gt = lq[::-1, :], gt[::-1, :]
                k = int(rng.integers(0, 4))
                if k:
                    lq, gt = np.rot90(lq, k), np.rot90(gt, k)
        else:
            h, w = lq.shape[:2]
            gt = gt[: h * s, : w * s]

        return _to_tensor(lq), _to_tensor(gt)


class HROnlyDataset(Dataset):
    """Returns single HR float32 CHW tensors in [0, 1] (old-photo spec §2/§3).

    The prepared sub-image tiles serve as HR-only sources — the LR side is
    synthesized on GPU per batch by ``data.degradation.OldPhotoDegradation``.
    Geometric augs (h/v flips, 90° rotations) are applied to HR *before*
    synthesis so the degradation automatically matches (§3.4); channel
    shuffle and mixup are deliberately absent — they are not PSNR-safe
    under sepia/JPEG degradation.

    ``subset_stride``/``max_images`` shrink the dataset (reduced-size runs):
    every ``subset_stride``-th tile is kept, then capped at ``max_images``.
    """

    def __init__(
        self,
        gt_root: str,
        gt_size: Optional[int] = 320,
        scale: int = 4,
        use_hflip: bool = True,
        use_rot: bool = True,
        subset_stride: int = 1,
        max_images: Optional[int] = None,
    ) -> None:
        self.gt_paths = list_images(gt_root)[:: max(int(subset_stride), 1)]
        if max_images is not None:
            self.gt_paths = self.gt_paths[: int(max_images)]
        self.paths = self.gt_paths
        self.scale = int(scale)
        self.gt_size = gt_size
        self.use_hflip = bool(use_hflip)
        self.use_rot = bool(use_rot)
        if gt_size is not None and gt_size % self.scale:
            raise ValueError(f"gt_size {gt_size} not divisible by scale {self.scale}")

    def __len__(self) -> int:
        return len(self.gt_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        gt = _load(self.gt_paths[idx])

        if self.gt_size is not None:
            size = self.gt_size
            if gt.shape[0] < size or gt.shape[1] < size:
                raise RuntimeError(
                    f"HR {self.gt_paths[idx].name} {gt.shape[:2]} smaller than "
                    f"crop {size} — use a larger source or smaller gt_size"
                )
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
            rng = np.random.default_rng(seed)
            y = int(rng.integers(0, gt.shape[0] - size + 1))
            x = int(rng.integers(0, gt.shape[1] - size + 1))
            gt = gt[y : y + size, x : x + size]

            if self.use_hflip and rng.random() < 0.5:
                gt = gt[:, ::-1]
            if self.use_rot:
                if rng.random() < 0.5:
                    gt = gt[::-1, :]
                k = int(rng.integers(0, 4))
                if k:
                    gt = np.rot90(gt, k)
        else:
            # Full image, trimmed so H and W divide by the model scale.
            s = self.scale
            gt = gt[: gt.shape[0] // s * s, : gt.shape[1] // s * s]

        return _to_tensor(gt)


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = False,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        drop_last=drop_last,
    )
