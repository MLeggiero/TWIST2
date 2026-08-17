"""
G1 dataset schema (GR00T / LeRobot compatible).

This module is the single source of truth for how recorded TWIST2 episodes
label their state and action vectors. It does two jobs:

1. **Reorder** the raw Redis-wire vectors that the low-level controller and
   teleop server publish so that, in the *recorded* dataset, the 29 joint
   positions come first and are grouped by limb. The Redis wire format is
   left untouched on purpose -- the trained ONNX policy expects the original
   layout, and reordering it would break inference. The reorder happens once,
   at recording time, inside ``episode_writer.EpisodeWriter``.

2. Emit a ``meta/modality.json`` file (LeRobot / Isaac-GR00T convention) and
   a top-level ``schema`` block embedded in each episode's ``data.json`` so
   the dataset is self-describing and downstream consumers (GR00T training,
   visualisation tools) can slice the flat vectors back into named groups
   without consulting external code.

Layout summary
--------------

state_body (34 dims, recorded order)::

    [0:29]   dof_pos  -- joint positions, see JOINT_NAMES
    [29:32]  imu_ang_vel  -- base angular velocity (rad/s, body frame)
    [32:34]  imu_rp       -- base roll, pitch (rad)

action_body (35 dims, recorded order)::

    [0:29]   dof_pos_target  -- commanded joint positions, see JOINT_NAMES
    [29:31]  root_lin_vel_xy -- commanded base xy velocity (m/s, local frame)
    [31:32]  root_height     -- commanded base height (m)
    [32:34]  root_rp         -- commanded base roll, pitch (rad)
    [34:35]  root_yaw_rate   -- commanded base yaw rate (rad/s)

effort_body (29 dims)::

    [0:29]   dof_tau  -- per-joint estimated torque (N*m), see JOINT_NAMES

``effort_body`` is recorded as its own top-level field rather than being
appended to ``state_body``, so the 34-dim layout that existing episodes and
``modality.json`` indices assume stays valid. It needs no reordering: the
low-level controller publishes it already in ``JOINT_NAMES`` order, which is
the order ``state_body[0:29]`` ends up in after ``reorder_state_body``.

The ``[0:29]`` slice of ``state_body`` and ``action_body`` reference the
**same** joint indices in the **same** order. ``action_body[0:29]`` is the
controller's target for ``state_body[0:29]`` at the next control step, so on
that slice action *is* a one-step delayed observation. The remaining tail
dimensions encode different physical quantities (measured IMU vs commanded
root motion) and are intentionally quarantined into separate sub-modalities
so they are never confused.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Joint names (Unitree G1 29-DoF, matches robot_control/configs/g1.yaml)
# ---------------------------------------------------------------------------

JOINT_NAMES: List[str] = [
    # left leg (0-5)
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # right leg (6-11)
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # waist (12-14)
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # left arm (15-21)
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # right arm (22-28)
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

assert len(JOINT_NAMES) == 29

# Half-open [start, end) slices into the 29-dim joint vector, grouped by limb.
LIMB_SLICES: Dict[str, Tuple[int, int]] = {
    "left_leg":  (0,   6),
    "right_leg": (6,  12),
    "waist":     (12, 15),
    "left_arm":  (15, 22),
    "right_arm": (22, 29),
}

NUM_JOINTS = 29
STATE_BODY_DIM = 34   # 29 joints + 3 ang_vel + 2 rp
ACTION_BODY_DIM = 35  # 29 joints + 2 vel_xy + 1 z + 2 rp + 1 yaw_rate
EFFORT_BODY_DIM = 29  # one estimated torque per joint, JOINT_NAMES order


# ---------------------------------------------------------------------------
# Reorder helpers (raw Redis order -> recorded labeled order)
# ---------------------------------------------------------------------------
#
# Raw state_body publishing order (server_low_level_g1_real.py):
#     [ang_vel(3), rp(2), dof_pos(29)]
#
# Raw action_body publishing order (xrobot_teleop_to_robot_w_hand.py):
#     [base_vel_xy(2), root_z(1), rp(2), yaw_rate(1), dof_pos(29)]


def reorder_state_body(raw):
    """Reorder a raw 34-dim state_body vector to put joint positions first.

    Returns a plain ``list`` so the result is JSON-serialisable directly.
    Accepts any sequence (list, tuple, ``np.ndarray``).
    """
    arr = np.asarray(raw, dtype=np.float64)
    if arr.shape != (STATE_BODY_DIM,):
        raise ValueError(
            f"reorder_state_body expects shape ({STATE_BODY_DIM},), got {arr.shape}"
        )
    ang_vel = arr[0:3]
    rp = arr[3:5]
    dof_pos = arr[5:34]
    return np.concatenate([dof_pos, ang_vel, rp]).tolist()


def reorder_action_body(raw):
    """Reorder a raw 35-dim action_body vector to put joint targets first."""
    arr = np.asarray(raw, dtype=np.float64)
    if arr.shape != (ACTION_BODY_DIM,):
        raise ValueError(
            f"reorder_action_body expects shape ({ACTION_BODY_DIM},), got {arr.shape}"
        )
    base_vel_xy = arr[0:2]
    root_z = arr[2:3]
    rp = arr[3:5]
    yaw_rate = arr[5:6]
    dof_pos_target = arr[6:35]
    return np.concatenate(
        [dof_pos_target, base_vel_xy, root_z, rp, yaw_rate]
    ).tolist()


# ---------------------------------------------------------------------------
# Tactile region layout (mirrors inspire_hand_wrapper.TACTILE_LAYOUT)
# ---------------------------------------------------------------------------
#
# Each entry is (region_name, flat_start_index, n_points, (rows, cols)).
# flat_start_index is the offset into the 1062-element flat tactile buffer
# that ``InspireHandController`` publishes. Kept here (not imported) so the
# dataset schema does not pull in the pymodbus dependency.

TACTILE_REGIONS: List[Tuple[str, int, int, Tuple[int, int]]] = [
    ("little_tip",    0,    9, (3,  3)),
    ("little_nail",   9,   96, (12, 8)),
    ("little_pad",    105,  80, (10, 8)),
    ("ring_tip",      185,  9, (3,  3)),
    ("ring_nail",     194, 96, (12, 8)),
    ("ring_pad",      290, 80, (10, 8)),
    ("middle_tip",    370,  9, (3,  3)),
    ("middle_nail",   379, 96, (12, 8)),
    ("middle_pad",    475, 80, (10, 8)),
    ("index_tip",     555,  9, (3,  3)),
    ("index_nail",    564, 96, (12, 8)),
    ("index_pad",     660, 80, (10, 8)),
    ("thumb_tip",     740,  9, (3,  3)),
    ("thumb_nail",    749, 96, (12, 8)),
    ("thumb_middle",  845,  9, (3,  3)),
    ("thumb_pad",     854, 96, (12, 8)),
    ("palm",          950, 112, (8, 14)),
]

TACTILE_FLAT_LEN = sum(n for _, _, n, _ in TACTILE_REGIONS)
assert TACTILE_FLAT_LEN == 1062


# ---------------------------------------------------------------------------
# modality.json (LeRobot / Isaac-GR00T convention)
# ---------------------------------------------------------------------------

def build_modality_json() -> dict:
    """Return the dict that should be written to ``<task>/meta/modality.json``.

    Uses the LeRobot ``state.<group>: {start, end}`` slicing convention used
    by Isaac-GR00T (see ``demo_data/gr1.PickNPlace/meta/modality.json`` in
    the GR00T repo). ``state``/``action`` cover the joint-first labelled
    body vector; ``state_hand_*``/``action_hand_*`` and ``tactile_hand_*``
    are separate top-level modality groups so consumers don't have to
    juggle one giant flat tensor.
    """

    # state.<group> indices into the 34-dim labelled state_body vector
    state_groups: Dict[str, Dict[str, int]] = {}
    for limb, (s, e) in LIMB_SLICES.items():
        state_groups[limb] = {"start": s, "end": e}
    state_groups["imu_ang_vel"] = {"start": 29, "end": 32}
    state_groups["imu_rp"]      = {"start": 32, "end": 34}

    # action.<group> indices into the 35-dim labelled action_body vector
    action_groups: Dict[str, Dict[str, int]] = {}
    for limb, (s, e) in LIMB_SLICES.items():
        action_groups[limb] = {"start": s, "end": e}
    action_groups["root_lin_vel_xy"] = {"start": 29, "end": 31}
    action_groups["root_height"]     = {"start": 31, "end": 32}
    action_groups["root_rp"]         = {"start": 32, "end": 34}
    action_groups["root_yaw_rate"]   = {"start": 34, "end": 35}

    # effort.<limb> indices into the 29-dim effort_body vector. Shares the
    # joint indexing of state/action, so the same limb slices apply.
    effort_groups: Dict[str, Dict[str, int]] = {}
    for limb, (s, e) in LIMB_SLICES.items():
        effort_groups[limb] = {"start": s, "end": e}

    # tactile sub-region slices into the flat 1062-element buffer
    tactile_groups: Dict[str, Dict] = {}
    for name, start, n, shape in TACTILE_REGIONS:
        tactile_groups[name] = {
            "start": start,
            "end": start + n,
            "shape": list(shape),
        }

    return {
        "state": state_groups,
        "action": action_groups,
        "effort_body": effort_groups,
        "state_hand_left":  {"all": {"start": 0, "end": 6}},
        "state_hand_right": {"all": {"start": 0, "end": 6}},
        "action_hand_left":  {"all": {"start": 0, "end": 6}},
        "action_hand_right": {"all": {"start": 0, "end": 6}},
        "tactile_hand_left":  tactile_groups,
        "tactile_hand_right": tactile_groups,
        "video": {
            "rgb": {"original_key": "observation.images.rgb"},
            "rgb_wrist_left": {"original_key": "observation.images.rgb_wrist_left"},
            "rgb_wrist_right": {"original_key": "observation.images.rgb_wrist_right"},
        },
        "annotation": {
            "human.task_description": {"original_key": "text.goal"},
        },
    }


def build_schema_block() -> dict:
    """Return a self-describing schema dict to embed inline in ``data.json``.

    Unlike ``modality.json`` (which follows the GR00T convention exactly),
    this block is verbose on purpose: it spells out joint names, units, and
    the relationship between state and action so a human reading the
    episode file can understand the layout without referring to any other
    file.
    """
    return {
        "version": "1.0.0",
        "robot": "unitree_g1_29dof",
        "joint_names": JOINT_NAMES,
        "limb_slices": {k: list(v) for k, v in LIMB_SLICES.items()},
        "state_body": {
            "dim": STATE_BODY_DIM,
            "layout": [
                {"name": "dof_pos",     "start": 0,  "end": 29,
                 "unit": "rad", "labels": JOINT_NAMES},
                {"name": "imu_ang_vel", "start": 29, "end": 32,
                 "unit": "rad/s",
                 "labels": ["wx", "wy", "wz"]},
                {"name": "imu_rp",      "start": 32, "end": 34,
                 "unit": "rad",
                 "labels": ["roll", "pitch"]},
            ],
        },
        "action_body": {
            "dim": ACTION_BODY_DIM,
            "layout": [
                {"name": "dof_pos_target", "start": 0,  "end": 29,
                 "unit": "rad", "labels": JOINT_NAMES},
                {"name": "root_lin_vel_xy", "start": 29, "end": 31,
                 "unit": "m/s",
                 "labels": ["vx_local", "vy_local"]},
                {"name": "root_height",     "start": 31, "end": 32,
                 "unit": "m",
                 "labels": ["z"]},
                {"name": "root_rp",         "start": 32, "end": 34,
                 "unit": "rad",
                 "labels": ["roll", "pitch"]},
                {"name": "root_yaw_rate",   "start": 34, "end": 35,
                 "unit": "rad/s",
                 "labels": ["yaw_rate"]},
            ],
        },
        "effort_body": {
            "dim": EFFORT_BODY_DIM,
            "layout": [
                {"name": "dof_tau", "start": 0, "end": 29,
                 "unit": "N*m", "labels": JOINT_NAMES},
            ],
            "note": (
                "Per-joint estimated torque (tau_est) read from the motor "
                "state, indexed identically to state_body[0:29] in the "
                "recorded joint-first layout. This is the motor's internal "
                "estimate, not an external torque sensor; treat it as "
                "proportional to joint load rather than a calibrated value."
            ),
        },
        "state_action_relationship": (
            "state_body[0:29] and action_body[0:29] reference the same 29 "
            "joints in the same order; action_body[0:29] is the commanded "
            "target for state_body[0:29] at the next control step. The "
            "remaining tail dimensions are NOT a delayed-observation pair: "
            "state_body[29:34] is measured IMU (angular velocity + roll/"
            "pitch) while action_body[29:35] is the commanded root motion "
            "(planar velocity, height, roll/pitch, yaw rate). They are "
            "kept in separate sub-modalities so they are never conflated."
        ),
        "hand": {
            "dim": 6,
            "note": "Inspire RH56DFTP joint angles, raw 0-1000 register units.",
        },
        "tactile": {
            "flat_dim": TACTILE_FLAT_LEN,
            "dtype": "uint16",
            "value_range": [0, 4095],
            "regions": [
                {"name": name, "start": start, "end": start + n,
                 "shape": list(shape)}
                for name, start, n, shape in TACTILE_REGIONS
            ],
            "note": (
                "Per-hand tactile is published as a flat 1062-element list. "
                "Reshape any region with the (rows, cols) shape above, or "
                "use deploy_real.robot_control.inspire_hand_wrapper.slice_tactile."
            ),
        },
    }
