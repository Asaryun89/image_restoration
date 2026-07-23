"""Training entry point — team17 DIV2K fine-tune reproduction.

Usage:
    python tools/prepare_div2k.py --root data/DIV2K            # once
    python train.py --config configs/team17_div2k_ft.yaml
    python train.py --config configs/team17_div2k_ft.yaml --resume checkpoints/team17_ft/latest.pth
    python train.py --config configs/team17_div2k_ft.yaml --debug   # 2-iteration smoke test
"""
from __future__ import annotations

import argparse
import random
from typing import Optional

import numpy as np
import torch
import yaml

from engine.trainer import Trainer


def load_config(path: str) -> dict:
    """Load a YAML config into a plain dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch (CPU + CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Image restoration training")
    parser.add_argument("--config", required=True, help="path to a YAML config")
    parser.add_argument("--resume", default=None, help="checkpoint path for full resume")
    parser.add_argument("--seed", type=int, default=None, help="override config seed")
    parser.add_argument(
        "--debug", action="store_true", help="smoke test: 2 iters/epoch, tiny val set"
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    cfg["seed"] = seed
    set_seed(seed)

    trainer = Trainer(cfg, resume=args.resume, debug=args.debug)
    trainer.fit()


if __name__ == "__main__":
    main()
