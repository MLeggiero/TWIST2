"""PICO hand tracking -> Inspire RH56DFTP command via dex_retargeting IK.

Mirrors the retargeting path used by XRoboToolkit-Teleop-Sample-Python
(scripts/simulation/teleop_inspire_hand_placo.py). The PICO SDK streams a
27x7 per-hand joint array; we reindex to the 21-joint MediaPipe convention,
estimate a wrist frame, transform into the MANO frame, and run
dex_retargeting's vector optimizer to get URDF qpos. The 6 target joints
are then scaled from radians to Inspire's [0, 1000] command range
(1000 = open, 0 = closed per the RH56DFTP manual).
"""

from pathlib import Path
from typing import Optional

import numpy as np
from dex_retargeting.constants import (
    OPERATOR2MANO,
    HandType,
    RetargetingType,
    RobotName,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig


PICO_TO_MEDIAPIPE = {
    1: 0,   # Wrist
    2: 1, 3: 2, 4: 3, 5: 4,        # Thumb: metacarpal, proximal, distal, tip
    7: 5, 8: 6, 9: 7, 10: 8,       # Index: proximal, intermediate, distal, tip
    12: 9, 13: 10, 14: 11, 15: 12, # Middle
    17: 13, 18: 14, 19: 15, 20: 16, # Ring
    22: 17, 23: 18, 24: 19, 25: 20, # Little
}

# Dict-key order used by XRobotStreamer (GMR.xrobot_utils.hand_joint_names).
# NOTE: GMR iterates its name list while indexing into the raw (27,7) PICO
# array, so the dict labels "LeftHandWrist" / "LeftHandPalm" are actually
# swapped relative to PICO's physical layout (PICO row 0 = Palm, row 1 =
# Wrist, but the dict stores PICO row 0 under "...Wrist" and row 1 under
# "...Palm"). We preserve that ordering here so arr[1] ends up holding the
# Wrist position — which is what PICO_TO_MEDIAPIPE expects.
PICO_JOINT_NAME_SUFFIXES = (
    "Wrist", "Palm",
    "ThumbMetacarpal", "ThumbProximal", "ThumbDistal", "ThumbTip",
    "IndexMetacarpal", "IndexProximal", "IndexIntermediate", "IndexDistal", "IndexTip",
    "MiddleMetacarpal", "MiddleProximal", "MiddleIntermediate", "MiddleDistal", "MiddleTip",
    "RingMetacarpal", "RingProximal", "RingIntermediate", "RingDistal", "RingTip",
    "LittleMetacarpal", "LittleProximal", "LittleIntermediate", "LittleDistal", "LittleTip",
)


def xrobot_hand_dict_to_array(hand_dict: dict, hand_side: str) -> Optional[np.ndarray]:
    """Convert an XRobotStreamer hand dict into the (27,7) array the tracker expects.

    XRobotStreamer returns hand tracking as a dict keyed by OpenXR joint names
    (e.g. ``LeftHandThumbMetacarpal``) with values ``[pos, rot]`` where
    ``pos=[x,y,z]`` and ``rot=[qw,qx,qy,qz]`` (scalar-first). The downstream
    retargeting code only reads xyz, so the quaternion conversion is mostly a
    convenience for callers that inspect the array.
    """
    prefix = f"{hand_side.title()}Hand"
    arr = np.zeros((26, 7), dtype=np.float32)
    found = 0
    for i, suffix in enumerate(PICO_JOINT_NAME_SUFFIXES):
        entry = hand_dict.get(prefix + suffix)
        if entry is None:
            continue
        pos, rot = entry
        arr[i, 0:3] = pos
        # PICO quat is scalar-first; store scalar-last to match common conventions.
        arr[i, 3:7] = (rot[1], rot[2], rot[3], rot[0])
        found += 1
    if found == 0:
        return None
    return arr

# TWIST2 Inspire command order: [pinky, ring, middle, index, thumb_bend, thumb_rotation]
TWIST2_TARGET_JOINTS = [
    "pinky_proximal_joint",
    "ring_proximal_joint",
    "middle_proximal_joint",
    "index_proximal_joint",
    "thumb_proximal_pitch_joint",   # bend
    "thumb_proximal_yaw_joint",     # rotation
]

# Per-joint upper limits from assets/inspire_hand/inspire_hand_{left,right}.urdf.
# URDF qpos=0 is the fully-open pose; qpos=upper is the fully-closed pose.
JOINT_UPPER_LIMITS = np.array(
    [1.47, 1.47, 1.47, 1.47, 0.6, 1.308], dtype=np.float32
)

# Default asset dir: <repo>/assets, which contains inspire_hand/inspire_hand_{left,right}.urdf.
_DEFAULT_ASSET_DIR = Path(__file__).resolve().parents[2] / "assets"


def pico_hand_state_to_mediapipe(hand_state: np.ndarray) -> Optional[np.ndarray]:
    """Reindex a 27x7 PICO hand state to a 21x3 MediaPipe layout, centered at wrist."""
    arr = np.asarray(hand_state, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return None
    rows = arr.shape[0]
    if rows not in (26, 27):
        return None
    mediapipe = np.zeros((21, 3), dtype=np.float64)
    for pico_idx, mp_idx in PICO_TO_MEDIAPIPE.items():
        if pico_idx < rows:
            mediapipe[mp_idx] = arr[pico_idx, :3]
    return mediapipe - mediapipe[0:1, :]


def estimate_frame_from_hand_points(keypoints: np.ndarray) -> np.ndarray:
    """Estimate a 3x3 wrist frame in MANO convention from 21 MediaPipe points."""
    pts = keypoints[[0, 5, 9], :]
    x_vector = pts[0] - pts[2]
    centered = pts - pts.mean(axis=0, keepdims=True)
    _, _, v = np.linalg.svd(centered)
    normal = v[2, :]
    x = x_vector - np.dot(x_vector, normal) * normal
    x = x / (np.linalg.norm(x) + 1e-6)
    z = np.cross(x, normal)
    if np.dot(z, pts[1] - pts[2]) < 0:
        normal = -normal
        z = -z
    return np.stack([x, normal, z], axis=1)


class DexFingerTracker:
    """Dex_retargeting-based PICO -> Inspire command converter for one hand.

    Args:
        hand_side: 'left' or 'right'. Selects the matching URDF + config.
        asset_dir: Directory containing the inspire_hand/ subdir. Defaults to
            <repo>/assets.
        low_pass_alpha: forwarded to RetargetingConfig (smaller = smoother,
            more latency). None keeps the config default (0.2).
    """

    def __init__(
        self,
        hand_side: str,
        asset_dir: Optional[Path] = None,
        low_pass_alpha: Optional[float] = None,
    ):
        if hand_side not in ("left", "right"):
            raise ValueError(f"hand_side must be 'left' or 'right', got {hand_side}")
        self.hand_side = hand_side
        self.hand_type_enum = HandType.left if hand_side == "left" else HandType.right
        self.asset_dir = Path(asset_dir) if asset_dir is not None else _DEFAULT_ASSET_DIR
        self._operator2mano = OPERATOR2MANO[self.hand_type_enum]

        RetargetingConfig.set_default_urdf_dir(str(self.asset_dir))
        config_path = get_default_config_path(
            RobotName.inspire, RetargetingType.vector, self.hand_type_enum
        )
        cfg = RetargetingConfig.load_from_file(config_path)
        if low_pass_alpha is not None:
            cfg.low_pass_alpha = float(low_pass_alpha)

        # Bump from the stock 1.15 so the human wrist->fingertip vectors
        # reach the robot's fully-open pose (otherwise the pinky in
        # particular stops a few degrees short of qpos=0).
        cfg.scaling_factor = 1.2

        # Stock config only constrains the five fingertips. That leaves the
        # thumb's two joints (pitch + yaw) under-constrained and the optimizer
        # reaches thumb_tip targets via yaw alone, skipping the fold. Add a
        # second thumb vector (wrist -> thumb_distal, MediaPipe idx 3) so
        # pitch has to engage to match the intermediate joint position.
        cfg.target_origin_link_names = list(cfg.target_origin_link_names) + ["base"]
        cfg.target_task_link_names = list(cfg.target_task_link_names) + ["thumb_distal"]
        cfg.target_link_human_indices = np.concatenate(
            [cfg.target_link_human_indices, np.array([[0], [3]], dtype=np.int64)],
            axis=1,
        )

        self.retargeting = cfg.build()

        # Build index map: TWIST2 command slot -> position in retargeting.joint_names.
        joint_names = list(self.retargeting.joint_names)
        self._target_indices = np.array(
            [joint_names.index(n) for n in TWIST2_TARGET_JOINTS], dtype=np.int64
        )

    def pico_to_inspire_angles(
        self, hand_data, _side: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """Return a 6-element float32 array of Inspire commands in [0, 1000].

        Order matches TWIST2 Inspire convention:
            [pinky, ring, middle, index, thumb_bend, thumb_rotation].

        Args:
            hand_data: raw PICO hand tracking, either a numpy array of shape
                (27, 7) / (26, 7) or the dict XRobotStreamer emits (keyed by
                OpenXR joint names like "LeftHandThumbMetacarpal").
            _side: accepted for API compatibility with the legacy
                legacy pico_to_inspire_angles(hand_data, side) signature;
                the tracker's hand side is fixed at construction time.

        Returns None if the input is missing, malformed, or all-zero.
        """
        if hand_data is None:
            return None
        # XRobotStreamer sometimes hands us (is_active, dict); accept that too.
        if isinstance(hand_data, tuple) and len(hand_data) == 2:
            is_active, hand_data = hand_data
            if not is_active or hand_data is None:
                return None
        if isinstance(hand_data, dict):
            arr = xrobot_hand_dict_to_array(hand_data, self.hand_side)
            if arr is None:
                return None
        else:
            arr = np.asarray(hand_data)
            if arr.dtype == object:
                return None
        if arr.size == 0 or np.all(np.abs(arr) < 1e-8):
            return None

        mediapipe = pico_hand_state_to_mediapipe(arr)
        if mediapipe is None:
            return None

        try:
            wrist_rot = estimate_frame_from_hand_points(mediapipe)
            transformed = mediapipe @ wrist_rot @ self._operator2mano

            indices = self.retargeting.optimizer.target_link_human_indices
            origin = indices[0, :]
            task = indices[1, :]
            ref_value = transformed[task, :] - transformed[origin, :]

            qpos = self.retargeting.retarget(ref_value)
        except (RuntimeWarning, RuntimeError, ValueError) as e:
            print(f"[DexFingerTracker:{self.hand_side}] retarget failed: {e}")
            return None

        target_rad = np.asarray(qpos, dtype=np.float32)[self._target_indices]
        # URDF 0 = open = Inspire cmd 1000; URDF upper = closed = cmd 0.
        normalized = 1.0 - np.clip(target_rad / JOINT_UPPER_LIMITS, 0.0, 1.0)
        return (normalized * 1000.0).astype(np.float32)

    def reset(self):
        """Reset the retargeting low-pass filter; call on state transitions."""
        if hasattr(self.retargeting, "reset"):
            self.retargeting.reset()


class DualDexFingerTracker:
    """Convenience wrapper holding one DexFingerTracker per hand."""

    def __init__(
        self,
        asset_dir: Optional[Path] = None,
        low_pass_alpha: Optional[float] = None,
    ):
        self.left = DexFingerTracker("left", asset_dir, low_pass_alpha)
        self.right = DexFingerTracker("right", asset_dir, low_pass_alpha)
        self._last_left: Optional[np.ndarray] = None
        self._last_right: Optional[np.ndarray] = None

    def pico_to_inspire_angles(
        self, hand_data: np.ndarray, hand_side: str
    ) -> Optional[np.ndarray]:
        tracker = self.left if hand_side == "left" else self.right
        result = tracker.pico_to_inspire_angles(hand_data)
        if result is None:
            return None
        if hand_side == "left":
            self._last_left = result
        else:
            self._last_right = result
        return result

    def reset(self):
        self.left.reset()
        self.right.reset()
        self._last_left = None
        self._last_right = None


if __name__ == "__main__":
    tracker = DualDexFingerTracker()
    # Straight-hand sanity test: all joints in a line, fingers extended.
    hand = np.zeros((27, 7), dtype=np.float32)
    hand[1, :3] = [0, 0, 0]
    for i, idx in enumerate([6, 7, 8, 9, 10]):
        hand[idx, :3] = [0.04, (i + 1) * 0.02, 0]
    for i, idx in enumerate([11, 12, 13, 14, 15]):
        hand[idx, :3] = [0.02, (i + 1) * 0.02, 0]
    for i, idx in enumerate([16, 17, 18, 19, 20]):
        hand[idx, :3] = [0.00, (i + 1) * 0.02, 0]
    for i, idx in enumerate([21, 22, 23, 24, 25]):
        hand[idx, :3] = [-0.02, (i + 1) * 0.02, 0]
    for i, idx in enumerate([2, 3, 4, 5]):
        hand[idx, :3] = [0.06 + i * 0.015, 0.01, 0]
    out = tracker.pico_to_inspire_angles(hand, "right")
    print("straight-hand right angles [0,1000]:", out)
