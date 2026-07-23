"""On-the-fly degradation engine.

Every op works on a float32 numpy image in HWC layout with values in [0, 1]
and draws all randomness from an explicit ``numpy.random.Generator`` so a
pipeline call is fully deterministic given a seed (used for validation pairs).
"""
from __future__ import annotations

import io
import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


def _conv2d_reflect(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve an HWC image with a 2D kernel (per channel, reflect padding)."""
    k = kernel.shape[0]
    t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))[None].float()
    w = torch.from_numpy(kernel).float()[None, None].repeat(t.shape[1], 1, 1, 1)
    t = F.pad(t, [k // 2] * 4, mode="reflect")
    out = F.conv2d(t, w, groups=t.shape[1])
    return out[0].numpy().transpose(1, 2, 0)


def _odd(v: int) -> int:
    return v if v % 2 == 1 else v + 1


class DegradationOp:
    """Base class: one composable degradation gated by ``prob``."""

    def __init__(self, prob: float = 1.0) -> None:
        self.prob = float(prob)

    def __call__(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError


class GaussianNoise(DegradationOp):
    """Additive white Gaussian noise, sigma given in 8-bit units."""

    def __init__(self, prob: float = 0.7, sigma: Sequence[float] = (1, 30)) -> None:
        super().__init__(prob)
        self.sigma = tuple(sigma)

    def __call__(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        sigma = rng.uniform(*self.sigma) / 255.0
        return img + rng.normal(0.0, sigma, img.shape).astype(np.float32)


class PoissonNoise(DegradationOp):
    """Signal-dependent Poisson (shot) noise; peak = scale * 255."""

    def __init__(self, prob: float = 0.3, scale: Sequence[float] = (0.5, 3.0)) -> None:
        super().__init__(prob)
        self.scale = tuple(scale)

    def __call__(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        peak = rng.uniform(*self.scale) * 255.0
        return (rng.poisson(np.clip(img, 0, 1) * peak) / peak).astype(np.float32)


class JPEGCompression(DegradationOp):
    """JPEG compression artifacts via an in-memory encode/decode round trip."""

    def __init__(self, prob: float = 0.5, quality: Sequence[int] = (30, 95)) -> None:
        super().__init__(prob)
        self.quality = tuple(quality)

    def __call__(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        q = int(rng.integers(self.quality[0], self.quality[1] + 1))
        pil = Image.fromarray((np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8))
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return np.asarray(Image.open(buf).convert("RGB"), np.float32) / 255.0


class GaussianBlur(DegradationOp):
    """Isotropic Gaussian blur with random odd kernel size and sigma."""

    def __init__(
        self,
        prob: float = 0.5,
        kernel: Sequence[int] = (3, 21),
        sigma: Sequence[float] = (0.2, 3.0),
    ) -> None:
        super().__init__(prob)
        self.kernel = tuple(kernel)
        self.sigma = tuple(sigma)

    def __call__(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        k = _odd(int(rng.integers(self.kernel[0], self.kernel[1] + 1)))
        sigma = rng.uniform(*self.sigma)
        ax = np.arange(k, dtype=np.float32) - (k - 1) / 2.0
        g = np.exp(-(ax**2) / (2.0 * sigma**2))
        kern = np.outer(g, g)
        kern /= kern.sum()
        return _conv2d_reflect(img, kern.astype(np.float32))


class MotionBlur(DegradationOp):
    """Mild linear motion blur along a random direction."""

    def __init__(self, prob: float = 0.2, kernel: Sequence[int] = (3, 15)) -> None:
        super().__init__(prob)
        self.kernel = tuple(kernel)

    def __call__(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        k = _odd(int(rng.integers(self.kernel[0], self.kernel[1] + 1)))
        angle = rng.uniform(0.0, math.pi)
        kern = np.zeros((k, k), np.float32)
        c = (k - 1) / 2.0
        dx, dy = math.cos(angle), math.sin(angle)
        for t in np.linspace(-c, c, 4 * k):
            x, y = int(round(c + t * dx)), int(round(c + t * dy))
            if 0 <= x < k and 0 <= y < k:
                kern[y, x] = 1.0
        kern /= kern.sum()
        return _conv2d_reflect(img, kern)


class Scratches(DegradationOp):
    """Thin random curves (quadratic Beziers), light and dark, variable opacity."""

    def __init__(
        self,
        prob: float = 0.4,
        num: Sequence[int] = (1, 6),
        width: Sequence[int] = (1, 3),
        opacity: Sequence[float] = (0.3, 1.0),
    ) -> None:
        super().__init__(prob)
        self.num = tuple(num)
        self.width = tuple(width)
        self.opacity = tuple(opacity)

    def __call__(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        h, w = img.shape[:2]
        out = img.copy()
        n = int(rng.integers(self.num[0], self.num[1] + 1))
        for _ in range(n):
            mask_img = Image.new("L", (w, h), 0)
            draw = ImageDraw.Draw(mask_img)
            pts = rng.uniform(0, 1, (3, 2)) * np.array([w, h], np.float32)
            ts = np.linspace(0.0, 1.0, 24)[:, None]
            curve = ((1 - ts) ** 2) * pts[0] + 2 * (1 - ts) * ts * pts[1] + (ts**2) * pts[2]
            line_w = int(rng.integers(self.width[0], self.width[1] + 1))
            draw.line([tuple(p) for p in curve], fill=255, width=line_w)
            mask = np.asarray(mask_img, np.float32)[..., None] / 255.0
            alpha = mask * rng.uniform(*self.opacity)
            value = rng.uniform(0.8, 1.0) if rng.random() < 0.5 else rng.uniform(0.0, 0.2)
            out = out * (1.0 - alpha) + value * alpha
        return out


class Fade(DegradationOp):
    """Aged-photo look: reduced contrast/saturation plus a slight color cast."""

    def __init__(
        self,
        prob: float = 0.4,
        contrast: Sequence[float] = (0.5, 0.9),
        saturation: Sequence[float] = (0.4, 0.9),
        cast: float = 0.05,
    ) -> None:
        super().__init__(prob)
        self.contrast = tuple(contrast)
        self.saturation = tuple(saturation)
        self.cast = float(cast)

    def __call__(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        c = rng.uniform(*self.contrast)
        out = (img - 0.5) * c + 0.5
        s = rng.uniform(*self.saturation)
        gray = out @ np.array([0.299, 0.587, 0.114], np.float32)
        out = gray[..., None] + (out - gray[..., None]) * s
        shift = rng.uniform(-self.cast, self.cast, 3).astype(np.float32)
        return out + shift


_OP_REGISTRY = {
    "gaussian_noise": GaussianNoise,
    "poisson_noise": PoissonNoise,
    "jpeg": JPEGCompression,
    "gaussian_blur": GaussianBlur,
    "motion_blur": MotionBlur,
    "scratches": Scratches,
    "fade": Fade,
}

# Application order: photometric fade first, then optics (blur), physical
# damage (scratches), sensor noise, and compression last.
_ORDER = (
    "fade",
    "gaussian_blur",
    "motion_blur",
    "scratches",
    "gaussian_noise",
    "poisson_noise",
    "jpeg",
)


class DegradationPipeline:
    """Composable degradation stack; a random subset of ops fires per call.

    Args:
        cfg: mapping of op name -> kwargs (must include ``prob``); ops
            missing from the config are skipped.
    """

    def __init__(self, cfg: Dict[str, dict]) -> None:
        self.ops = []
        for name in _ORDER:
            if name in cfg:
                self.ops.append(_OP_REGISTRY[name](**cfg[name]))

    def __call__(self, img: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """Degrade a clean image. Deterministic when ``seed`` is given."""
        rng = np.random.default_rng(seed)
        out = img.astype(np.float32, copy=True)
        for op in self.ops:
            # Always consume the gate draw so the sequence is seed-stable.
            if rng.random() < op.prob:
                out = op(out, rng)
        return np.clip(out, 0.0, 1.0).astype(np.float32)
