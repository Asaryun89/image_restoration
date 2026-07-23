"""Early stopping on a monitored validation metric."""
from __future__ import annotations

import math
from typing import Any, Dict


class EarlyStopping:
    """Stops training when the monitored metric stops improving.

    Args:
        patience: epochs without improvement before stopping.
        min_delta: minimum change that counts as an improvement.
        mode: ``"min"`` (e.g. LPIPS) or ``"max"`` (e.g. PSNR).
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4, mode: str = "min") -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"Unknown mode: {mode}")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = math.inf if mode == "min" else -math.inf
        self.counter = 0

    def step(self, value: float) -> bool:
        """Record a new metric value; returns True when training should stop."""
        improved = (
            value < self.best - self.min_delta
            if self.mode == "min"
            else value > self.best + self.min_delta
        )
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

    def state_dict(self) -> Dict[str, Any]:
        return {"best": self.best, "counter": self.counter}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.best = state["best"]
        self.counter = state["counter"]
