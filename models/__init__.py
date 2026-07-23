"""Generator factory: builds the model named by ``model.arch`` in the config."""
from __future__ import annotations

import torch.nn as nn

from models.nafnet import NAFNet
from models.srhcnet import SRHCnet


def build_generator(model_cfg: dict) -> nn.Module:
    """Build a generator from a config's ``model`` section.

    ``arch`` selects the architecture; every other key is passed through as
    a constructor kwarg.

    - ``srhcnet``: HAT-L + gated NAFNet refiner (team17 / SRC-B composite).
    - ``nafnet``: plain NAFNet restoration generator.
    """
    cfg = dict(model_cfg)
    arch = str(cfg.pop("arch", "srhcnet")).lower()
    if arch == "srhcnet":
        return SRHCnet(**cfg)
    if arch == "nafnet":
        return NAFNet(**cfg)
    raise ValueError(f"Unknown generator arch: {arch!r} (expected srhcnet | nafnet)")
