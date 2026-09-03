"""Pure XRoboToolkit/PICO hand-landmark conversion helpers."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


PICO_TO_MEDIAPIPE = [
    "Wrist",
    "ThumbMetacarpal", "ThumbProximal", "ThumbDistal", "ThumbTip",
    "IndexProximal", "IndexIntermediate", "IndexDistal", "IndexTip",
    "MiddleProximal", "MiddleIntermediate", "MiddleDistal", "MiddleTip",
    "RingProximal", "RingIntermediate", "RingDistal", "RingTip",
    "LittleProximal", "LittleIntermediate", "LittleDistal", "LittleTip",
]


def pico_hand_to_mediapipe(
    hand_tuple: Any, side: str
) -> Optional[np.ndarray]:
    """Convert an XRobotStreamer hand tuple to wrist-relative MediaPipe order.

    Returns ``None`` for inactive, incomplete, malformed, or non-finite input.
    Input and output positions remain in the streamer's metre units.
    """
    if side not in ("Left", "Right"):
        raise ValueError("side must be 'Left' or 'Right'")
    if not isinstance(hand_tuple, (tuple, list)) or len(hand_tuple) != 2:
        return None

    active, hand_data = hand_tuple
    if not active or not isinstance(hand_data, dict):
        return None

    points = []
    prefix = f"{side}Hand"
    for joint_name in PICO_TO_MEDIAPIPE:
        value = hand_data.get(prefix + joint_name)
        if not isinstance(value, (tuple, list)) or len(value) < 1:
            return None
        try:
            position = np.asarray(value[0], dtype=np.float32)
        except (TypeError, ValueError):
            return None
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            return None
        points.append(position)

    keypoints = np.asarray(points, dtype=np.float32)
    if keypoints.shape != (21, 3):
        return None
    keypoints = keypoints - keypoints[0]
    return keypoints.astype(np.float32, copy=False)
