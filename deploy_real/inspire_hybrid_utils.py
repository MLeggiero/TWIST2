"""Pure validation and command filtering for one-sided Inspire control."""

from __future__ import annotations

import json
import time
from typing import Optional

import numpy as np


INSPIRE_COMMAND_DIM = 6
INSPIRE_COMMAND_MIN = 0.0
INSPIRE_COMMAND_MAX = 1000.0


def decode_inspire_target(raw) -> Optional[np.ndarray]:
    """Decode and clamp one finite six-axis Inspire target."""

    try:
        value = np.asarray(json.loads(raw), dtype=np.float64)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if value.shape != (INSPIRE_COMMAND_DIM,) or not np.all(np.isfinite(value)):
        return None
    return np.clip(value, INSPIRE_COMMAND_MIN, INSPIRE_COMMAND_MAX)


def timestamp_age(raw, now: Optional[float] = None) -> Optional[float]:
    """Return age in seconds for a Unix timestamp, or ``None`` if invalid."""

    try:
        timestamp = float(raw)
    except (TypeError, ValueError):
        return None
    current = time.time() if now is None else float(now)
    if not np.isfinite(timestamp) or not np.isfinite(current):
        return None
    return current - timestamp


class InspireCommandFilter:
    """Measured-state startup interpolation plus a 6-D slew-rate ceiling.

    The rate is an application-level limit in Inspire command counts/second,
    not a claim about a native device acceleration or velocity limit. Elapsed
    time is capped at one nominal control period, preventing a delayed loop
    from issuing a catch-up jump.
    """

    def __init__(self, max_rate: float, control_period: float,
                 startup_duration: float):
        if not np.isfinite(max_rate) or max_rate <= 0:
            raise ValueError("max_rate must be finite and positive")
        if not np.isfinite(control_period) or control_period <= 0:
            raise ValueError("control_period must be finite and positive")
        if not np.isfinite(startup_duration) or startup_duration < 0:
            raise ValueError("startup_duration must be finite and non-negative")
        self.max_rate = float(max_rate)
        self.control_period = float(control_period)
        self.startup_duration = float(startup_duration)
        self.position: Optional[np.ndarray] = None
        self._last_update: Optional[float] = None
        self._startup_position: Optional[np.ndarray] = None
        self._startup_time: Optional[float] = None

    @staticmethod
    def _position(value, description: str) -> np.ndarray:
        result = np.asarray(value, dtype=np.float64)
        if result.shape != (INSPIRE_COMMAND_DIM,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{description} must be finite shape (6,)")
        return np.clip(result, INSPIRE_COMMAND_MIN, INSPIRE_COMMAND_MAX)

    def reset(self, measured, now: Optional[float] = None) -> np.ndarray:
        current = self._position(measured, "measured position")
        timestamp = time.monotonic() if now is None else float(now)
        if not np.isfinite(timestamp):
            raise ValueError("reset time must be finite")
        self.position = current.copy()
        self._last_update = timestamp
        self._startup_position = None
        self._startup_time = None
        return current.copy()

    def hold(self) -> Optional[np.ndarray]:
        return None if self.position is None else self.position.copy()

    def step(self, requested, now: Optional[float] = None) -> np.ndarray:
        target = self._position(requested, "requested position")
        timestamp = time.monotonic() if now is None else float(now)
        if not np.isfinite(timestamp):
            raise ValueError("update time must be finite")
        if self.position is None or self._last_update is None:
            return self.reset(target, timestamp)
        if self._startup_time is None:
            self._startup_time = timestamp
            self._startup_position = self.position.copy()

        elapsed_startup = max(0.0, timestamp - self._startup_time)
        alpha = (1.0 if self.startup_duration <= 0 else
                 min(elapsed_startup / self.startup_duration, 1.0))
        desired = self._startup_position + (target - self._startup_position) * alpha

        elapsed = max(0.0, timestamp - self._last_update)
        effective_dt = min(elapsed, self.control_period)
        max_delta = self.max_rate * effective_dt
        self.position = self.position + np.clip(
            desired - self.position, -max_delta, max_delta
        )
        self._last_update = timestamp
        return self.position.copy()
