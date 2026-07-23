"""Checkpoint manager: latest + best-PSNR (+ optional periodic), full resume."""
from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def collect_rng_states() -> Dict[str, Any]:
    """Snapshot Python / NumPy / torch (CPU + CUDA) RNG states."""
    states: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()
    return states


def restore_rng_states(states: Optional[Dict[str, Any]]) -> None:
    """Restore RNG states saved by :func:`collect_rng_states` (best effort)."""
    if not states:
        return
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    # set_rng_state APIs demand CPU ByteTensors, but a checkpoint loaded
    # with map_location='cuda' arrives with these tensors on the GPU.
    torch.set_rng_state(states["torch"].cpu().to(torch.uint8))
    if "cuda" in states and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in states["cuda"]])
        except RuntimeError:
            pass  # device count changed since the checkpoint was written


class CheckpointManager:
    """Writes ``latest.pth`` every epoch, ``best_psnr.pth`` on improvement,
    and optionally ``epoch_XXXX.pth`` every ``periodic_every`` epochs."""

    def __init__(self, ckpt_dir: str, periodic_every: int = 0) -> None:
        self.dir = Path(ckpt_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.periodic_every = int(periodic_every)

    @property
    def latest_path(self) -> Path:
        return self.dir / "latest.pth"

    @property
    def best_path(self) -> Path:
        return self.dir / "best_psnr.pth"

    def save(self, state: Dict[str, Any], epoch: int, is_best: bool) -> None:
        """Atomically write latest; copy to best/periodic files as needed."""
        tmp = self.latest_path.with_suffix(".tmp")
        torch.save(state, tmp)
        tmp.replace(self.latest_path)
        if is_best:
            shutil.copyfile(self.latest_path, self.best_path)
        if self.periodic_every > 0 and (epoch + 1) % self.periodic_every == 0:
            shutil.copyfile(self.latest_path, self.dir / f"epoch_{epoch + 1:04d}.pth")

    @staticmethod
    def load(path: str, map_location: str = "cpu") -> Dict[str, Any]:
        """Load a full training checkpoint (trusted file; contains RNG state)."""
        return torch.load(path, map_location=map_location, weights_only=False)
