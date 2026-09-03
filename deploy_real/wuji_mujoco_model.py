#!/usr/bin/env python3
"""Compose the G1 body and official Wuji Hand 2 MJCF models with MuJoCo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import mujoco
import numpy as np
import yaml


SIDES = ("left", "right")

# The G1 fixed-hand visual starts at this location in g1_sim2sim_29dof.xml.
# Wuji's local fingers extend along -Z, so -90 degrees about parent Y aligns
# them with the G1 wrist's +X direction. These are visualization/test mounts,
# not a dimensional specification for a physical adapter.
DEFAULT_MOUNT_POSITIONS = {
    "left": (0.0415, 0.003, 0.0),
    "right": (0.0415, -0.003, 0.0),
}
DEFAULT_MOUNT_QUATERNION = (
    float(np.sqrt(0.5)),
    0.0,
    -float(np.sqrt(0.5)),
    0.0,
)


@dataclass(frozen=True)
class CombinedWujiModel:
    """A compiled model plus hand actuator names in device command order."""

    model: mujoco.MjModel
    hand_actuator_names: Mapping[str, Tuple[str, ...]]
    hand_mjcf_paths: Mapping[str, Path]


def resolve_mjcf_path(config_path: Path | str) -> Path:
    """Resolve ``optimizer.mjcf_path`` relative to a retargeter YAML file."""

    config = Path(config_path).expanduser().resolve()
    if not config.is_file():
        raise FileNotFoundError(f"Wuji config not found: {config}")
    with config.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    relative = (payload.get("optimizer") or {}).get("mjcf_path")
    if not relative:
        raise ValueError(f"optimizer.mjcf_path is missing from {config}")
    result = (config.parent / relative).resolve()
    if not result.is_file():
        raise FileNotFoundError(f"Wuji MJCF referenced by {config} not found: {result}")
    return result


def _finite_vector(value: Sequence[float], size: int, description: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{description} must contain {size} finite values")
    return result


def _remove_fixed_hand_placeholder(spec: mujoco.MjSpec, side: str) -> None:
    placeholder = spec.body(f"{side}_rubber_hand")
    if placeholder is not None:
        spec.delete(placeholder)


def _actuator_names(spec: mujoco.MjSpec, side: str) -> Tuple[str, ...]:
    names = tuple(actuator.name for actuator in spec.actuators)
    if len(names) != 20 or any(not name for name in names):
        raise ValueError(
            f"{side} Wuji Hand 2 MJCF must define exactly 20 named actuators; "
            f"found {len(names)}"
        )
    if len(set(names)) != len(names):
        raise ValueError(f"{side} Wuji Hand 2 MJCF contains duplicate actuator names")
    return names


def _inherit_parent_physics_options(
    parent: mujoco.MjSpec, child: mujoco.MjSpec
) -> None:
    """Use one explicit set of global options for the composed simulation."""

    # A composed MuJoCo world has one global option block. Keep the existing
    # G1 simulator settings; hand-local joints and actuators remain unchanged.
    for name in ("timestep", "tolerance", "integrator", "jacobian", "solver"):
        setattr(child.option, name, getattr(parent.option, name))


def _require_names(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    names: Iterable[str],
    description: str,
) -> None:
    missing = [
        name
        for name in names
        if mujoco.mj_name2id(model, object_type, name) < 0
    ]
    if missing:
        raise ValueError(f"compiled model is missing {description}: {missing}")


def compose_g1_wuji_model(
    base_xml: Path | str,
    hand_configs: Mapping[str, Path | str],
    mount_positions: Optional[Mapping[str, Sequence[float]]] = None,
    mount_quaternions: Optional[Mapping[str, Sequence[float]]] = None,
    save_compiled_model: Optional[Path | str] = None,
) -> CombinedWujiModel:
    """Attach exact Wuji MJCFs to named G1 wrist bodies and compile the model.

    Hand actuator order is captured before attachment and retained verbatim. It
    is the same device/MJCF order published by ``wuji_hand_bridge.py``.
    """

    base_path = Path(base_xml).expanduser().resolve()
    if not base_path.is_file():
        raise FileNotFoundError(f"G1 base MJCF not found: {base_path}")
    unknown = set(hand_configs) - set(SIDES)
    if unknown:
        raise ValueError(f"unknown hand sides: {sorted(unknown)}")
    if not hand_configs:
        raise ValueError("at least one Wuji hand config is required")

    positions: Dict[str, Sequence[float]] = dict(DEFAULT_MOUNT_POSITIONS)
    if mount_positions:
        positions.update(mount_positions)
    quaternions: Dict[str, Sequence[float]] = {
        side: DEFAULT_MOUNT_QUATERNION for side in SIDES
    }
    if mount_quaternions:
        quaternions.update(mount_quaternions)

    parent = mujoco.MjSpec.from_file(str(base_path))
    parent.copy_during_attach = True
    hand_actuator_names: Dict[str, Tuple[str, ...]] = {}
    hand_mjcf_paths: Dict[str, Path] = {}

    for side in SIDES:
        if side not in hand_configs:
            continue
        wrist = parent.body(f"{side}_wrist_yaw_link")
        if wrist is None:
            raise ValueError(f"G1 model has no {side}_wrist_yaw_link body")

        mjcf_path = resolve_mjcf_path(hand_configs[side])
        child = mujoco.MjSpec.from_file(str(mjcf_path))
        hand_actuator_names[side] = _actuator_names(child, side)
        hand_mjcf_paths[side] = mjcf_path

        _inherit_parent_physics_options(parent, child)

        _remove_fixed_hand_placeholder(parent, side)
        frame = wrist.add_frame()
        frame.name = f"wuji_{side}_mount"
        frame.pos = _finite_vector(positions[side], 3, f"{side} mount position")
        quaternion = _finite_vector(
            quaternions[side], 4, f"{side} mount quaternion"
        )
        norm = float(np.linalg.norm(quaternion))
        if norm <= 1e-12:
            raise ValueError(f"{side} mount quaternion cannot be zero")
        frame.quat = quaternion / norm

        # An explicit empty prefix preserves the official l_*/r_* names. Passing
        # None makes MuJoCo introduce a '/' namespace prefix.
        parent.attach(child, prefix="", frame=frame)

    model = parent.compile()
    if save_compiled_model:
        output = Path(save_compiled_model).expanduser().resolve()
        if output.suffix.lower() != ".mjb":
            raise ValueError("compiled MuJoCo model output must use a .mjb suffix")
        output.parent.mkdir(parents=True, exist_ok=True)
        mujoco.mj_saveModel(model, str(output), None)

    _require_names(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        (name for names in hand_actuator_names.values() for name in names),
        "Wuji actuators",
    )
    expected_actuators = 29 + 20 * len(hand_actuator_names)
    if model.nu != expected_actuators:
        raise ValueError(
            f"expected {expected_actuators} total actuators "
            f"(29 body + 20 per hand), found {model.nu}"
        )

    return CombinedWujiModel(
        model=model,
        hand_actuator_names=hand_actuator_names,
        hand_mjcf_paths=hand_mjcf_paths,
    )
