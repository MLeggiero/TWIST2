"""Velocity limiting for streamed Wuji Hand 2 position commands."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np


class JointVelocityLimiter:
    """Limit per-joint commanded-position slope in hardware joint order.

    Elapsed time is capped at one nominal control period so a delayed loop
    produces slower motion instead of a large catch-up command.
    """

    def __init__(self, max_velocity, control_period: float):
        limits = np.asarray(max_velocity, dtype=np.float64)
        if limits.shape != (20,) or not np.all(np.isfinite(limits)):
            raise ValueError("max_velocity must be finite shape (20,)")
        if np.any(limits <= 0):
            raise ValueError("all max_velocity values must be positive")
        if not np.isfinite(control_period) or control_period <= 0:
            raise ValueError("control_period must be finite and positive")
        self.max_velocity = limits
        self.control_period = float(control_period)
        self._position: Optional[np.ndarray] = None
        self._last_update: Optional[float] = None

    @property
    def position(self) -> Optional[np.ndarray]:
        return None if self._position is None else self._position.copy()

    def reset(self, position, now: Optional[float] = None) -> np.ndarray:
        value = self._validate_position(position, "initial position")
        timestamp = time.monotonic() if now is None else float(now)
        if not np.isfinite(timestamp):
            raise ValueError("reset time must be finite")
        self._position = value.copy()
        self._last_update = timestamp
        return self._position.copy()

    def step(self, target, now: Optional[float] = None) -> np.ndarray:
        target_value = self._validate_position(target, "target")
        current_time = time.monotonic() if now is None else float(now)
        if not np.isfinite(current_time):
            raise ValueError("update time must be finite")
        if self._position is None or self._last_update is None:
            return self.reset(target_value, current_time)

        elapsed = max(0.0, current_time - self._last_update)
        effective_dt = min(elapsed, self.control_period)
        max_delta = self.max_velocity * effective_dt
        delta = np.clip(target_value - self._position, -max_delta, max_delta)
        self._position = self._position + delta
        self._last_update = current_time
        return self._position.copy()

    def at_target(self, target, atol: float = 1e-7) -> bool:
        if self._position is None:
            return False
        target_value = self._validate_position(target, "target")
        return bool(np.all(np.abs(target_value - self._position) <= atol))

    @staticmethod
    def _validate_position(value, description: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (20,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{description} must be finite shape (20,)")
        return array
