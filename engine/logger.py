"""Local run logger: text log, append-only CSV, sample grids, optional TensorBoard.

Everything is written to disk under ``runs/<run_name>_<YYYYmmdd-HHMMSS>/`` —
no external services or network calls.
"""
from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import yaml
from torchvision.utils import make_grid, save_image

CSV_COLUMNS = [
    "step", "epoch", "split",
    "loss_total", "loss_l1", "loss_l2", "loss_swt",
    "lr", "grad_norm", "psnr", "psnr_y", "ssim",
    "beta_drift", "gamma_drift",
]


class RunLogger:
    """Creates a timestamped run directory and handles all local logging."""

    def __init__(self, cfg: dict) -> None:
        log_cfg = cfg.get("log", {})
        runs_dir = Path(log_cfg.get("runs_dir", "runs"))
        run_name = cfg.get("run_name", "run")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = runs_dir / f"{run_name}_{stamp}"
        self.samples_dir = self.dir / "samples"
        self.samples_dir.mkdir(parents=True, exist_ok=True)

        # Exact copy of the resolved config for reproducibility.
        with open(self.dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        self._logger = logging.getLogger(f"restoration.{stamp}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        for handler in (
            logging.StreamHandler(),
            logging.FileHandler(self.dir / "train.log", encoding="utf-8"),
        ):
            handler.setFormatter(fmt)
            self._logger.addHandler(handler)

        self._csv_file = open(self.dir / "metrics.csv", "a", newline="", encoding="utf-8")
        self._csv = csv.DictWriter(self._csv_file, fieldnames=CSV_COLUMNS, restval="")
        if self._csv_file.tell() == 0:
            self._csv.writeheader()
            self._csv_file.flush()

        self.tb = None
        if log_cfg.get("tensorboard", True):
            try:
                from torch.utils.tensorboard import SummaryWriter  # noqa: PLC0415

                self.tb = SummaryWriter(log_dir=str(self.dir / "tb"))
            except ImportError:
                self._logger.warning("tensorboard not installed; TB logging disabled")

    def info(self, msg: str) -> None:
        self._logger.info(msg)

    def log_scalars(self, step: int, epoch: int, split: str, values: Dict[str, float]) -> None:
        """Append one CSV row (flushed immediately) and mirror to TensorBoard."""
        row = {"step": step, "epoch": epoch, "split": split}
        for key, val in values.items():
            if key in CSV_COLUMNS and val is not None:
                row[key] = f"{val:.6g}"
        self._csv.writerow(row)
        self._csv_file.flush()
        if self.tb is not None:
            for key, val in values.items():
                if val is not None:
                    self.tb.add_scalar(f"{split}/{key}", val, step)

    def save_samples(
        self,
        epoch: int,
        degraded: torch.Tensor,
        restored: torch.Tensor,
        clean: torch.Tensor,
    ) -> None:
        """Save a ``degraded | restored | clean`` grid for fixed val images."""
        rows = []
        for d, r, c in zip(degraded, restored, clean):
            rows.extend([d.detach().cpu(), r.detach().cpu(), c.detach().cpu()])
        grid = torch.stack(rows).clamp(0, 1)
        path = self.samples_dir / f"epoch_{epoch + 1:04d}.png"
        save_image(grid, path, nrow=3)
        if self.tb is not None:
            self.tb.add_image("val/samples", make_grid(grid, nrow=3), epoch + 1)

    def close(self) -> None:
        self._csv_file.close()
        if self.tb is not None:
            self.tb.close()
        for handler in list(self._logger.handlers):
            handler.close()
            self._logger.removeHandler(handler)
