"""Team17 DIV2K fine-tune training loop (spec Part B).

Trains the SRHCnet composite on official DIV2K bicubic x4 pairs with:

- alternating L1 / L2 / SWT losses, per-iteration round-robin (spec B.4);
- progressive HR patch sizes via ``data.progressive`` phases (spec B.4);
- channel-shuffle and mixup-style pair-mixing batch augmentations, both
  applied identically to LR and HR (spec B.5, definitions are the spec's
  declared interpretations);
- validation with the challenge metric — Y-channel PSNR with a
  ``scale``-pixel border crop — plus RGB PSNR and SSIM (spec B.2);
- gate-drift logging against the loaded checkpoint (acceptance check 4).

Also handles AMP (bfloat16 — fp16's range overflows on unclamped SR
outputs), EMA, warmup + cosine LR, gradient clipping, checkpointing with
full resume, optional partial-freeze, a consecutive-NaN abort, and a
graceful Ctrl-C that saves ``latest.pth``.
"""
from __future__ import annotations

import math
import random
from copy import deepcopy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from data.dataset import HROnlyDataset, PairedImageDataset, build_dataloader
from data.degradation import OldPhotoDegradation
from engine.checkpoint import CheckpointManager, collect_rng_states, restore_rng_states
from engine.early_stopping import EarlyStopping
from engine.logger import RunLogger
from eval.metrics import SSIMMetric, psnr, psnr_y
from losses.alternating import AlternatingLoss
from models import build_generator
from models.tiling import tiled_forward


class ModelEMA:
    """Exponential moving average of a model's parameters and buffers."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.module = deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for key, ema_v in self.module.state_dict().items():
            model_v = msd[key].detach()
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(self.decay).add_(model_v, alpha=1.0 - self.decay)
            else:
                ema_v.copy_(model_v)

    def state_dict(self) -> dict:
        return self.module.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.module.load_state_dict(state)


def build_warmup_cosine(
    optimizer: torch.optim.Optimizer,
    total_iters: int,
    warmup_iters: int,
    lr: float,
    lr_min: float,
) -> LambdaLR:
    """Linear warmup for ``warmup_iters`` then cosine annealing to ``lr_min``."""
    min_ratio = lr_min / lr
    warmup = min(warmup_iters, max(total_iters - 1, 1))

    def factor(it: int) -> float:
        if it < warmup:
            return (it + 1) / warmup
        t = (it - warmup) / max(total_iters - warmup, 1)
        return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * min(t, 1.0)))

    return LambdaLR(optimizer, factor)


class Trainer:
    """DIV2K fine-tune trainer for the SRHCnet composite (spec Part B)."""

    def __init__(self, cfg: dict, resume: Optional[str] = None, debug: bool = False) -> None:
        self.cfg = cfg
        self.debug = debug
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        data_cfg = dict(cfg["data"])
        train_cfg = cfg["train"]
        optim_cfg = cfg["optim"]
        self.epochs = int(train_cfg["epochs"])
        # Virtual epoch length in iterations, independent of dataset size
        # (the sub-image train set is ~30k tiles; a full pass per "epoch"
        # would make validation/checkpoints far too infrequent).
        self.iters_per_epoch_cap: Optional[int] = train_cfg.get("iters_per_epoch")
        val_max = data_cfg.get("val_max_images")
        if debug:
            self.epochs = min(self.epochs, 2)
            self.iters_per_epoch_cap = 2
            val_max = min(val_max or 4, 4)
            data_cfg["num_workers"] = 0

        self.scale = int(cfg["model"].get("upscale", 4))
        # Old-photo mode (old-photo spec §3): a `degradation:` section makes
        # the train set HR-only and synthesizes LR on GPU per batch.
        # Validation still reads pairs from disk — point val_gt/val_lq at the
        # frozen synthetic set from tools/make_frozen_val.py (§4).
        deg_cfg = cfg.get("degradation")
        if deg_cfg is not None:
            self.train_ds = HROnlyDataset(
                gt_root=data_cfg["train_gt"],
                gt_size=data_cfg.get("crop_size", 320),
                scale=self.scale,
                subset_stride=data_cfg.get("train_subset_stride", 1),
                max_images=data_cfg.get("train_max_images"),
            )
        else:
            self.train_ds = PairedImageDataset(
                gt_root=data_cfg["train_gt"],
                lq_root=data_cfg["train_lq"],
                split="train",
                gt_size=data_cfg.get("crop_size", 320),
                scale=self.scale,
            )
        self.val_ds = PairedImageDataset(
            gt_root=data_cfg["val_gt"],
            lq_root=data_cfg["val_lq"],
            split="val",
            gt_size=None,
            scale=self.scale,
        )
        if val_max is not None:
            self.val_ds.gt_paths = self.val_ds.gt_paths[:val_max]
            self.val_ds.lq_paths = self.val_ds.lq_paths[:val_max]

        # Progressive patch enlargement (spec B.4): each phase starts at its
        # (0-based) `epoch` with the given HR crop/batch size.
        self.phases = [
            {
                "epoch": int(p["epoch"]),
                "crop_size": int(p["crop_size"]),
                "batch_size": int(p.get("batch_size", data_cfg["batch_size"])),
            }
            for p in data_cfg.get("progressive") or []
        ]
        self.phases.sort(key=lambda p: p["epoch"])
        for phase in self.phases:
            if phase["crop_size"] % self.scale:
                raise ValueError(
                    f"progressive crop_size {phase['crop_size']} not divisible "
                    f"by model upscale {self.scale}"
                )
        if not self.phases or self.phases[0]["epoch"] > 0:
            self.phases.insert(
                0,
                {
                    "epoch": 0,
                    "crop_size": int(data_cfg.get("crop_size", 320)),
                    "batch_size": int(data_cfg["batch_size"]),
                },
            )
        self.data_cfg = data_cfg
        self.current_phase: Optional[dict] = None
        self.train_loader: Optional[torch.utils.data.DataLoader] = None
        self._apply_phase(self.phases[0])
        self.val_loader = build_dataloader(
            self.val_ds,
            batch_size=1,  # full-size val images vary in shape
            num_workers=data_cfg.get("num_workers", 0),
            shuffle=False,
            pin_memory=data_cfg.get("pin_memory", True),
            persistent_workers=data_cfg.get("persistent_workers", True),
        )

        # Full 2K val images OOM even large GPUs (the NAFNet stage runs at
        # HR in fp32), so validation tiles the LR input with overlap-average
        # stitching — same path evaluate.py uses.
        self.val_tile = int(data_cfg.get("val_tile", 256))
        self.val_tile_overlap = int(data_cfg.get("val_tile_overlap", 32))

        # Batch augmentations (spec B.5; prob/alpha are assumptions).
        aug_cfg = cfg.get("augment", {})
        self.channel_shuffle_prob = float(aug_cfg.get("channel_shuffle_prob", 0.5))
        self.mixup_prob = float(aug_cfg.get("mixup_prob", 0.2))
        self.mixup_alpha = float(aug_cfg.get("mixup_alpha", 1.2))

        # On-the-fly degradation synthesis (old-photo spec §3). Channel
        # shuffle / mixup are skipped in this mode regardless of `augment:`
        # — they are not PSNR-safe under sepia/JPEG degradation (§3.4).
        self.degrader: Optional[OldPhotoDegradation] = None
        if deg_cfg is not None:
            self.degrader = OldPhotoDegradation(deg_cfg, scale=self.scale).to(self.device)

        # ---- model --------------------------------------------------------
        self.generator = build_generator(cfg["model"]).to(self.device)
        self.ema = ModelEMA(self.generator, decay=optim_cfg.get("ema_decay", 0.999))
        # Acceptance check 4: record the loaded gates to report drift.
        self._init_gates = {
            name: getattr(self.generator, name).detach().clone()
            for name in ("beta", "gamma")
            if hasattr(self.generator, name)
        }

        self.amp = bool(optim_cfg.get("amp", True)) and self.device.type == "cuda"
        # bfloat16, not float16: fp16's ~65504 max overflows on unclamped SR
        # outputs (e.g. squared inside metrics/losses); bf16 shares fp32's
        # exponent range.
        self.amp_dtype = torch.bfloat16 if self.amp else torch.float32
        self.grad_clip = float(optim_cfg.get("grad_clip", 1.0))
        self.scaler_g = torch.amp.GradScaler(self.device.type, enabled=self.amp)
        self.nan_abort_streak = int(optim_cfg.get("nan_abort_streak", 20))
        self._nan_streak = 0

        # Optional partial-freeze: train only parameters whose names start
        # with one of these prefixes.
        train_only = optim_cfg.get("train_only_prefixes")
        if train_only:
            n_train = n_frozen = 0
            for name, p in self.generator.named_parameters():
                keep = any(name.startswith(pref) for pref in train_only)
                p.requires_grad_(keep)
                n_train += p.numel() * keep
                n_frozen += p.numel() * (not keep)
            if n_train == 0:
                raise ValueError(
                    f"optim.train_only_prefixes {train_only} matched no parameters"
                )
            self._freeze_msg = (
                f"training only {train_only}: {n_train / 1e6:.2f}M trainable, "
                f"{n_frozen / 1e6:.2f}M frozen"
            )
        else:
            self._freeze_msg = None

        # Spec B.3: Adam(0.9, 0.99); weight_decay defaults to 0 (BasicSR
        # convention) — AdamW with wd=0 is plain Adam.
        self.opt_g = torch.optim.AdamW(
            [p for p in self.generator.parameters() if p.requires_grad],
            lr=optim_cfg["lr"],
            betas=tuple(optim_cfg.get("betas", (0.9, 0.99))),
            weight_decay=optim_cfg.get("weight_decay", 0.0),
        )
        total_iters = sum(self._iters_in_epoch(e) for e in range(self.epochs))
        self.sched_g = build_warmup_cosine(
            self.opt_g,
            total_iters,
            optim_cfg.get("warmup_iters", 0),
            optim_cfg["lr"],
            optim_cfg.get("lr_min", 1e-6),
        )

        # ---- loss (spec B.4) ---------------------------------------------
        loss_cfg = cfg.get("loss", {})
        self.criterion = AlternatingLoss(
            hf_weight=loss_cfg.get("swt_hf_weight", 0.05)
        ).to(self.device)

        # ---- metrics / logging / checkpointing ----------------------------
        self.ssim_metric = SSIMMetric().to(self.device)
        self.logger = RunLogger(cfg)
        ckpt_cfg = cfg.get("checkpoint", {})
        self.ckpt = CheckpointManager(
            ckpt_cfg.get("dir", "checkpoints/team17_ft"),
            periodic_every=ckpt_cfg.get("periodic_every", 0),
        )
        # Challenge metric drives model selection (spec B.2 / B.7).
        self.best_metric = cfg.get("val_monitor", "psnr_y")

        self.early_stopping: Optional[EarlyStopping] = None
        es_cfg = cfg.get("early_stopping")
        if es_cfg is not None:
            self.es_monitor = es_cfg.get("monitor", self.best_metric)
            self.early_stopping = EarlyStopping(
                patience=es_cfg.get("patience", 5),
                min_delta=es_cfg.get("min_delta", 1e-4),
                mode=es_cfg.get("mode", "max"),
            )

        self.start_epoch = 0
        self.global_step = 0
        self.best_val = -math.inf

        # Optional same-arch warm start of the full generator (the primary
        # init path is the model-level `pretrained` key; this covers
        # resuming weights from this trainer's own runs).
        gen_ckpt = cfg.get("resume_generator")
        if gen_ckpt and not resume:
            self._load_generator_weights(gen_ckpt)
        if resume:
            self._resume(resume)

        # Fixed validation crops for the sample grids (center 128px LR
        # windows so full-size val images stack into one uniform grid).
        n_samples = min(4, len(self.val_ds))
        lq_win = 128
        sample_lq, sample_gt = [], []
        for i in range(n_samples):
            lq, gt = self.val_ds[i]
            win = min(lq_win, lq.shape[-2], lq.shape[-1])
            y = (lq.shape[-2] - win) // 2
            x = (lq.shape[-1] - win) // 2
            sample_lq.append(lq[..., y : y + win, x : x + win])
            s = self.scale
            sample_gt.append(gt[..., y * s : (y + win) * s, x * s : (x + win) * s])
        self.sample_lq = torch.stack(sample_lq)
        self.sample_gt = torch.stack(sample_gt)

        self.logger.info(
            f"device {self.device} | amp {self.amp} ({self.amp_dtype}) | "
            f"arch {cfg['model'].get('arch', 'srhcnet')} x{self.scale} | "
            f"train {len(self.train_ds)} tiles, val {len(self.val_ds)} imgs | "
            f"{total_iters} iters over {self.epochs} epochs | "
            f"selecting best by {self.best_metric}"
        )
        if self._freeze_msg:
            self.logger.info(self._freeze_msg)
        if self.degrader is not None:
            self.logger.info(
                f"old-photo mode: on-the-fly LR synthesis "
                f"(degradation_version {self.degrader.version}); "
                "channel shuffle/mixup disabled"
            )
        if len(self.phases) > 1:
            schedule = " -> ".join(
                f"ep{p['epoch'] + 1}+: {p['crop_size']}px @ bs{p['batch_size']}"
                for p in self.phases
            )
            self.logger.info(f"progressive patch schedule: {schedule}")

    # ------------------------------------------------- progressive patches
    def _phase_for_epoch(self, epoch: int) -> dict:
        active = self.phases[0]
        for phase in self.phases:
            if epoch >= phase["epoch"]:
                active = phase
        return active

    def _iters_in_epoch(self, epoch: int) -> int:
        n = len(self.train_ds) // self._phase_for_epoch(epoch)["batch_size"]
        if self.iters_per_epoch_cap is not None:
            n = min(n, self.iters_per_epoch_cap)
        return n

    def _apply_phase(self, phase: dict) -> None:
        """(Re)build the train loader for a phase's crop/batch size.

        Rebuilding is required even for a pure crop change: persistent
        workers keep their own dataset copy, so mutating ``gt_size`` on the
        main-process dataset never reaches already-running workers.
        """
        if phase is self.current_phase:
            return
        self.train_ds.gt_size = phase["crop_size"]
        self.train_loader = build_dataloader(
            self.train_ds,
            batch_size=phase["batch_size"],
            num_workers=self.data_cfg.get("num_workers", 0),
            shuffle=True,
            pin_memory=self.data_cfg.get("pin_memory", True),
            persistent_workers=self.data_cfg.get("persistent_workers", True),
            drop_last=True,
        )
        self.current_phase = phase
        if len(self.phases) > 1 and hasattr(self, "logger"):
            self.logger.info(
                f"progressive phase: crop {phase['crop_size']}px, "
                f"batch {phase['batch_size']}"
            )

    # ------------------------------------------------------------------ io
    def _load_generator_weights(self, path: str) -> None:
        state = CheckpointManager.load(path, map_location=str(self.device))
        for key in ("ema", "generator", "params_ema", "params"):
            if isinstance(state, dict) and key in state:
                state = state[key]
                break
        self.generator.load_state_dict(state)
        self.ema.load_state_dict(state)
        self.logger.info(f"initialised generator from {path}")

    def _state(self, epoch: int) -> dict:
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "best_val": self.best_val,
            "cfg": self.cfg,
            "generator": self.generator.state_dict(),
            "ema": self.ema.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "sched_g": self.sched_g.state_dict(),
            "scaler_g": self.scaler_g.state_dict(),
            "rng": collect_rng_states(),
        }
        if self.early_stopping is not None:
            state["early_stopping"] = self.early_stopping.state_dict()
        return state

    def _resume(self, path: str) -> None:
        state = CheckpointManager.load(path, map_location=str(self.device))
        self.generator.load_state_dict(state["generator"])
        self.ema.load_state_dict(state["ema"])
        self.opt_g.load_state_dict(state["opt_g"])
        self.sched_g.load_state_dict(state["sched_g"])
        self.scaler_g.load_state_dict(state["scaler_g"])
        if self.early_stopping is not None and "early_stopping" in state:
            self.early_stopping.load_state_dict(state["early_stopping"])
        self.start_epoch = state["epoch"] + 1
        self.global_step = state["global_step"]
        self.best_val = state.get("best_val", state.get("best_psnr", -math.inf))
        restore_rng_states(state.get("rng"))
        self.logger.info(f"resumed from {path} at epoch {self.start_epoch}")

    # ------------------------------------------------------- augmentations
    def _augment_batch(
        self, lq: torch.Tensor, gt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Spec B.5 batch augmentations, identical on LR and HR.

        Channel shuffle: one random RGB permutation for the batch. Mixing:
        mixup-style blend with a shared Beta-distributed coefficient
        against a shuffled pairing of the same batch. Both commute with
        bicubic degradation, so the pairs stay consistent.
        """
        if self.channel_shuffle_prob > 0 and random.random() < self.channel_shuffle_prob:
            perm = torch.randperm(3, device=lq.device)
            lq, gt = lq[:, perm], gt[:, perm]
        if self.mixup_prob > 0 and lq.shape[0] > 1 and random.random() < self.mixup_prob:
            lam = float(torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample())
            idx = torch.randperm(lq.shape[0], device=lq.device)
            lq = lam * lq + (1.0 - lam) * lq[idx]
            gt = lam * gt + (1.0 - lam) * gt[idx]
        return lq, gt

    # --------------------------------------------------------------- steps
    def _clip_and_step(
        self,
        loss: torch.Tensor,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
    ) -> float:
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
        # Guard explicitly so a bad batch is provably a no-op regardless of
        # AMP dtype/config.
        if math.isfinite(grad_norm):
            scaler.step(optimizer)
        scaler.update()
        return float(grad_norm)

    def _step(
        self, lq: torch.Tensor, gt: torch.Tensor
    ) -> Tuple[Dict[str, float], float]:
        with torch.amp.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp):
            restored = self.generator(lq)
            total, comps = self.criterion(restored, gt, self.global_step)
        grad_norm = self._clip_and_step(total, self.generator, self.opt_g, self.scaler_g)
        return comps, grad_norm

    # --------------------------------------------------------------- loops
    def _train_epoch(self, epoch: int) -> None:
        self._apply_phase(self._phase_for_epoch(epoch))
        self.generator.train()
        log_interval = self.cfg.get("log", {}).get("log_interval", 100)
        pbar = tqdm(
            self.train_loader,
            total=self._iters_in_epoch(epoch),
            desc=f"epoch {epoch + 1}/{self.epochs}",
            leave=False,
        )
        for i, batch in enumerate(pbar):
            if self.iters_per_epoch_cap is not None and i >= self.iters_per_epoch_cap:
                break
            if self.degrader is not None:
                gt = batch.to(self.device, non_blocking=True)
                lq = self.degrader(gt)
            else:
                lq, gt = batch
                lq = lq.to(self.device, non_blocking=True)
                gt = gt.to(self.device, non_blocking=True)
                lq, gt = self._augment_batch(lq, gt)

            comps, grad_norm = self._step(lq, gt)

            if math.isfinite(grad_norm):
                self._nan_streak = 0
            else:
                self._nan_streak += 1
                if self._nan_streak >= self.nan_abort_streak:
                    self.logger.info(
                        f"aborting: {self._nan_streak} consecutive non-finite grad "
                        f"norms (most recently at step {self.global_step + 1})"
                    )
                    raise RuntimeError(
                        f"training diverged: {self._nan_streak} consecutive non-finite "
                        f"grad norms, most recently at step {self.global_step + 1} — "
                        "aborting instead of burning GPU time on a NaN'd model "
                        "(consider lowering optim.lr or increasing optim.warmup_iters)"
                    )

            self.ema.update(self.generator)
            self.sched_g.step()
            self.global_step += 1

            lr = self.opt_g.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{comps['loss_total']:.4f}", lr=f"{lr:.2e}")
            if self.global_step % log_interval == 0 or self.debug:
                values = dict(comps)
                values["lr"] = lr
                values["grad_norm"] = grad_norm
                self.logger.log_scalars(self.global_step, epoch, "train", values)
                parts = " ".join(f"{k}={v:.4f}" for k, v in comps.items())
                self.logger.info(
                    f"step {self.global_step} | {parts} | lr={lr:.2e} | gn={grad_norm:.3f}"
                )

    def _gate_drift(self) -> Dict[str, float]:
        """Relative gate movement vs the loaded checkpoint (acceptance 4)."""
        drift = {}
        for name, init in self._init_gates.items():
            cur = getattr(self.generator, name).detach()
            denom = init.abs().mean().clamp_min(1e-8)
            drift[f"{name}_drift"] = float((cur - init.to(cur.device)).abs().mean() / denom)
        return drift

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate with EMA weights on full val images.

        Reports the challenge metric (Y-channel PSNR, ``scale``-px border
        crop) alongside RGB PSNR and SSIM.
        """
        model = self.ema.module
        model.eval()
        sums = {"psnr": 0.0, "psnr_y": 0.0, "ssim": 0.0}
        n = 0
        for lq, gt in tqdm(self.val_loader, desc="val", leave=False):
            lq = lq.to(self.device, non_blocking=True)
            gt = gt.to(self.device, non_blocking=True)
            restored = tiled_forward(
                model, lq, self.val_tile, self.val_tile_overlap, self.scale
            ).clamp(0, 1)
            sums["psnr"] += psnr(restored, gt).item()
            sums["psnr_y"] += psnr_y(restored, gt, border=self.scale).item()
            sums["ssim"] += self.ssim_metric(restored, gt).item()
            n += 1

        metrics = {k: v / n for k, v in sums.items()}
        drift = self._gate_drift()
        self.logger.log_scalars(self.global_step, epoch, "val", {**metrics, **drift})
        self.logger.info(
            f"epoch {epoch + 1} val | "
            + " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            + " | "
            + " ".join(f"{k}={v:.2%}" for k, v in drift.items())
        )

        sample_interval = self.cfg.get("log", {}).get("sample_interval", 1)
        if (epoch + 1) % max(sample_interval, 1) == 0:
            restored = model(self.sample_lq.to(self.device)).clamp(0, 1)
            # Nearest-upsample the LR inputs so the grid rows align.
            lq_vis = F.interpolate(self.sample_lq, size=restored.shape[-2:], mode="nearest")
            self.logger.save_samples(epoch, lq_vis, restored, self.sample_gt)
        return metrics

    def fit(self) -> None:
        """Run the fine-tune; Ctrl-C saves ``latest.pth`` before exiting."""
        epoch = self.start_epoch
        try:
            for epoch in range(self.start_epoch, self.epochs):
                self._train_epoch(epoch)
                metrics = self.validate(epoch)
                monitored = metrics[self.best_metric]
                is_best = monitored > self.best_val
                if is_best:
                    self.best_val = monitored
                    self.logger.info(f"new best {self.best_metric} {self.best_val:.4f}")
                self.ckpt.save(self._state(epoch), epoch, is_best)

                if self.early_stopping is not None:
                    if self.early_stopping.step(metrics.get(self.es_monitor, monitored)):
                        self.logger.info(f"early stopping at epoch {epoch + 1}")
                        break
            self.logger.info(
                f"training finished; best {self.best_metric} {self.best_val:.4f}"
            )
        except KeyboardInterrupt:
            self.logger.info("interrupted — saving latest.pth before exit")
            self.ckpt.save(self._state(epoch), epoch, is_best=False)
        finally:
            self.logger.close()
