"""Stage-2 adversarial and combined generator GAN losses.

Generator total: ``L_G = lambda_adv * L_adv + lambda_l1 * L1 + lambda_perc * Perceptual``.
Default adversarial formulation is LSGAN; relativistic average GAN (RaGAN,
BCE-with-logits form as in ESRGAN) is available via ``mode="ragan"``.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.pixel_losses import PerceptualLoss


class GANLoss(nn.Module):
    """Adversarial loss for both discriminator and generator sides.

    Args:
        mode: ``"lsgan"`` (least-squares) or ``"ragan"`` (relativistic average).
    """

    def __init__(self, mode: str = "lsgan") -> None:
        super().__init__()
        if mode not in ("lsgan", "ragan"):
            raise ValueError(f"Unknown GAN mode: {mode}")
        self.mode = mode

    def d_loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        """Discriminator loss on real (clean) vs fake (restored) logits.

        ``fake_logits`` must come from a detached generator output.
        """
        if self.mode == "lsgan":
            return 0.5 * (
                F.mse_loss(real_logits, torch.ones_like(real_logits))
                + F.mse_loss(fake_logits, torch.zeros_like(fake_logits))
            )
        rel_real = real_logits - fake_logits.mean()
        rel_fake = fake_logits - real_logits.mean()
        return 0.5 * (
            F.binary_cross_entropy_with_logits(rel_real, torch.ones_like(rel_real))
            + F.binary_cross_entropy_with_logits(rel_fake, torch.zeros_like(rel_fake))
        )

    def g_loss(
        self, fake_logits: torch.Tensor, real_logits: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generator adversarial loss.

        For RaGAN, ``real_logits`` (detached from D's graph on real images)
        is required; LSGAN ignores it.
        """
        if self.mode == "lsgan":
            return F.mse_loss(fake_logits, torch.ones_like(fake_logits))
        if real_logits is None:
            raise ValueError("RaGAN generator loss needs real_logits")
        rel_real = real_logits - fake_logits.mean()
        rel_fake = fake_logits - real_logits.mean()
        return 0.5 * (
            F.binary_cross_entropy_with_logits(rel_real, torch.zeros_like(rel_real))
            + F.binary_cross_entropy_with_logits(rel_fake, torch.ones_like(rel_fake))
        )


class GeneratorGANLoss(nn.Module):
    """Combined stage-2 generator loss: adversarial + L1 + perceptual."""

    def __init__(
        self,
        mode: str = "lsgan",
        lambda_adv: float = 0.005,
        lambda_l1: float = 1.0,
        lambda_perc: float = 0.1,
        vgg_layers: Sequence[str] = ("relu1_2", "relu2_2", "relu3_4", "relu4_4"),
    ) -> None:
        super().__init__()
        self.gan = GANLoss(mode)
        self.lambda_adv = lambda_adv
        self.lambda_l1 = lambda_l1
        self.lambda_perc = lambda_perc
        self.l1 = nn.L1Loss()
        self.perceptual = PerceptualLoss(vgg_layers) if lambda_perc > 0 else None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        fake_logits: torch.Tensor,
        real_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Returns (total generator loss, detached per-component dict)."""
        adv = self.gan.g_loss(fake_logits, real_logits)
        l1 = self.l1(pred, target)
        total = self.lambda_adv * adv + self.lambda_l1 * l1
        comps = {"loss_adv": adv.item(), "loss_l1": l1.item()}
        if self.perceptual is not None:
            perc = self.perceptual(pred, target)
            total = total + self.lambda_perc * perc
            comps["loss_perc"] = perc.item()
        comps["loss_total"] = total.item()
        return total, comps
