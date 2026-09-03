#!/usr/bin/env python3
"""Run the unchanged 29-DOF TWIST2 policy with Wuji Hand 2 in MuJoCo.

The body policy owns only the 29 G1 body actuators. Wuji position targets are
read independently from ``wuji_action_hand_left/right`` in the exact actuator
order of the official Wuji MJCF selected by each retargeter configuration.
This module never imports ``wuji_sdk`` and cannot connect to physical hands.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import mujoco
import numpy as np
import redis

from data_utils.rot_utils import quatToEuler
from wuji_mujoco_model import (
    DEFAULT_MOUNT_POSITIONS,
    DEFAULT_MOUNT_QUATERNION,
    CombinedWujiModel,
    compose_g1_wuji_model,
)


BODY_JOINT_NAMES: Tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

DEFAULT_BODY_POSITION = np.asarray(
    [
        -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
        -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.4, 0.0, 1.2, 0.0, 0.0, 0.0,
        0.0, -0.4, 0.0, 1.2, 0.0, 0.0, 0.0,
    ],
    dtype=np.float64,
)

STIFFNESS = np.asarray(
    [
        100, 100, 100, 150, 40, 40,
        100, 100, 100, 150, 40, 40,
        150, 150, 150,
        40, 40, 40, 40, 20, 20, 20,
        40, 40, 40, 40, 20, 20, 20,
    ],
    dtype=np.float64,
)
DAMPING = np.asarray(
    [
        2, 2, 2, 4, 2, 2,
        2, 2, 2, 4, 2, 2,
        4, 4, 4,
        5, 5, 5, 5, 1, 1, 1,
        5, 5, 5, 5, 1, 1, 1,
    ],
    dtype=np.float64,
)
TORQUE_LIMITS = np.asarray(
    [
        100, 100, 100, 150, 40, 40,
        100, 100, 100, 150, 40, 40,
        150, 150, 150,
        40, 40, 40, 40, 20, 20, 20,
        40, 40, 40, 40, 20, 20, 20,
    ],
    dtype=np.float64,
)
ACTION_SCALE = np.full(29, 0.5, dtype=np.float64)
ANKLE_INDICES = np.asarray([4, 5, 10, 11], dtype=np.int32)
DEFAULT_BODY_MIMIC = np.concatenate(
    [np.zeros(2), np.asarray([0.8]), np.zeros(3), DEFAULT_BODY_POSITION]
)


def decode_vector(raw, size: int) -> Optional[np.ndarray]:
    """Decode a finite JSON vector of an exact size, or return ``None``."""

    try:
        value = np.asarray(json.loads(raw), dtype=np.float64)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if value.shape != (size,) or not np.all(np.isfinite(value)):
        return None
    return value


def named_ids(model: mujoco.MjModel, object_type, names: Sequence[str]) -> np.ndarray:
    ids = np.asarray(
        [mujoco.mj_name2id(model, object_type, name) for name in names],
        dtype=np.int32,
    )
    missing = [name for name, identifier in zip(names, ids) if identifier < 0]
    if missing:
        raise ValueError(f"MuJoCo model is missing names: {missing}")
    return ids


def joint_addresses(
    model: mujoco.MjModel, joint_names: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray]:
    joint_ids = named_ids(model, mujoco.mjtObj.mjOBJ_JOINT, joint_names)
    scalar_types = {
        int(mujoco.mjtJoint.mjJNT_HINGE),
        int(mujoco.mjtJoint.mjJNT_SLIDE),
    }
    invalid = [
        name
        for name, joint_id in zip(joint_names, joint_ids)
        if int(model.jnt_type[joint_id]) not in scalar_types
    ]
    if invalid:
        raise ValueError(f"expected scalar hinge/slide joints: {invalid}")
    return model.jnt_qposadr[joint_ids].copy(), model.jnt_dofadr[joint_ids].copy()


def actuator_joint_qpos_addresses(
    model: mujoco.MjModel, actuator_ids: Sequence[int]
) -> np.ndarray:
    joint_ids = model.actuator_trnid[np.asarray(actuator_ids), 0]
    if np.any(joint_ids < 0):
        raise ValueError("Wuji position actuators must use direct joint transmissions")
    return model.jnt_qposadr[joint_ids].copy()


class OnnxPolicy:
    def __init__(self, path: Path | str, device: str):
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise ImportError(
                "onnxruntime is required for simulation; install onnxruntime "
                "or onnxruntime-gpu in the MuJoCo environment"
            ) from error

        available = ort.get_available_providers()
        providers = []
        if device.startswith("cuda") and "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        elif device.startswith("cuda"):
            print("CUDAExecutionProvider unavailable; using CPUExecutionProvider")
        providers.append("CPUExecutionProvider")
        self.session = ort.InferenceSession(str(path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"ONNX providers: {self.session.get_providers()}")

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        outputs = self.session.run(
            None,
            {self.input_name: np.asarray(observation, dtype=np.float32)[None, :]},
        )
        action = np.asarray(outputs[0], dtype=np.float64).squeeze()
        if action.shape != (29,) or not np.all(np.isfinite(action)):
            raise ValueError(f"TWIST2 policy returned invalid action shape {action.shape}")
        return action


class G1WujiMujocoController:
    NUM_ACTIONS = 29
    NUM_MIMIC_OBS = 35
    NUM_PROPRIO = 3 + 2 + 3 * 29
    NUM_OBS_SINGLE = NUM_MIMIC_OBS + NUM_PROPRIO
    HISTORY_LENGTH = 10
    TOTAL_OBS = NUM_OBS_SINGLE * (HISTORY_LENGTH + 1) + NUM_MIMIC_OBS

    def __init__(
        self,
        bundle: CombinedWujiModel,
        policy_path: Path | str,
        redis_ip: str,
        redis_port: int,
        device: str,
        policy_frequency: float,
        body_command_timeout: float,
        hand_command_timeout: float,
        realtime: bool,
        headless: bool,
        max_steps: int,
        print_every: int,
        require_fresh_pico_body: bool = False,
    ):
        self.model = bundle.model
        self.data = mujoco.MjData(self.model)
        self.hand_actuator_names = bundle.hand_actuator_names
        self.policy = OnnxPolicy(policy_path, device)
        self.redis = redis.Redis(host=redis_ip, port=redis_port, db=0)
        self.redis.ping()

        self.realtime = realtime
        self.headless = headless
        self.max_steps = max_steps
        self.print_every = print_every
        self.body_command_timeout = body_command_timeout
        self.hand_command_timeout = hand_command_timeout
        self.require_fresh_pico_body = require_fresh_pico_body
        self.sim_dt = float(self.model.opt.timestep)
        self.decimation = max(1, int(round(1.0 / (policy_frequency * self.sim_dt))))

        self.body_qpos, self.body_dof = joint_addresses(self.model, BODY_JOINT_NAMES)
        self.body_actuators = named_ids(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, BODY_JOINT_NAMES
        )
        pelvis_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis"
        )
        if pelvis_id < 0 or self.model.jnt_type[pelvis_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("combined model must contain the G1 pelvis free joint")
        self.pelvis_qpos = int(self.model.jnt_qposadr[pelvis_id])
        self.pelvis_dof = int(self.model.jnt_dofadr[pelvis_id])

        self.hand_actuators: Dict[str, np.ndarray] = {}
        self.hand_qpos: Dict[str, np.ndarray] = {}
        self.hand_targets: Dict[str, np.ndarray] = {}
        self.hand_warning_times = {side: 0.0 for side in self.hand_actuator_names}
        for side, names in self.hand_actuator_names.items():
            actuator_ids = named_ids(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, names
            )
            if len(actuator_ids) != 20:
                raise ValueError(f"{side} must have 20 Wuji actuators")
            self.hand_actuators[side] = actuator_ids
            self.hand_qpos[side] = actuator_joint_qpos_addresses(
                self.model, actuator_ids
            )
            self.hand_targets[side] = self._clip_hand_target(
                side, np.zeros(20, dtype=np.float64)
            )

        self.last_action = np.zeros(29, dtype=np.float64)
        self.last_pd_target = DEFAULT_BODY_POSITION.copy()
        self.last_body_mimic = DEFAULT_BODY_MIMIC.copy()
        self.history = deque(maxlen=self.HISTORY_LENGTH)
        for _ in range(self.HISTORY_LENGTH):
            self.history.append(np.zeros(self.NUM_OBS_SINGLE, dtype=np.float64))
        self.last_body_warning = 0.0

        print(
            f"Combined model: nq={self.model.nq}, nv={self.model.nv}, "
            f"nu={self.model.nu}; body=29, "
            + ", ".join(f"{side}=20" for side in self.hand_actuators)
        )
        print(
            f"MuJoCo dt={self.sim_dt:.4f}s, policy decimation={self.decimation} "
            f"({1.0 / (self.sim_dt * self.decimation):.1f} Hz)"
        )
        if self.require_fresh_pico_body:
            print("PICO body source freshness guard: enabled")

    def _clip_hand_target(self, side: str, target: np.ndarray) -> np.ndarray:
        result = np.asarray(target, dtype=np.float64).copy()
        ids = self.hand_actuators[side]
        limited = self.model.actuator_ctrllimited[ids].astype(bool)
        ranges = self.model.actuator_ctrlrange[ids]
        result[limited] = np.clip(
            result[limited], ranges[limited, 0], ranges[limited, 1]
        )
        return result

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.pelvis_qpos : self.pelvis_qpos + 7] = (
            0.0, 0.0, 0.793, 1.0, 0.0, 0.0, 0.0
        )
        self.data.qpos[self.body_qpos] = DEFAULT_BODY_POSITION
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        for side in self.hand_actuators:
            self.data.qpos[self.hand_qpos[side]] = self.hand_targets[side]
            self.data.ctrl[self.hand_actuators[side]] = self.hand_targets[side]
        mujoco.mj_forward(self.model, self.data)

    @staticmethod
    def _age(raw_timestamp, divisor: float, now: float) -> Optional[float]:
        try:
            timestamp = float(raw_timestamp) / divisor
        except (TypeError, ValueError):
            return None
        return now - timestamp

    def _warn_body(self, message: str, now: float) -> None:
        if now - self.last_body_warning >= 2.0:
            print(f"WARNING body: {message}; holding last target")
            self.last_body_warning = now

    def _read_body_mimic(self, now: float) -> Optional[np.ndarray]:
        keys = ["action_body_unitree_g1_with_hands", "t_action"]
        if self.require_fresh_pico_body:
            keys.append("pico_body_timestamp")
        values = self.redis.mget(*keys)
        raw_action, raw_timestamp = values[:2]

        age = self._age(raw_timestamp, 1000.0, now)
        if age is None:
            self._warn_body("missing t_action", now)
            return None
        if age < -1.0 or age > self.body_command_timeout:
            self._warn_body(f"stale timestamp (age={age:.3f}s)", now)
            return None

        if self.require_fresh_pico_body:
            source_age = self._age(values[2], 1.0, now)
            if source_age is None:
                self._warn_body(
                    "missing pico_body_timestamp (XR source not yet proven live)", now
                )
                return None
            if source_age < -1.0 or source_age > self.body_command_timeout:
                self._warn_body(
                    f"stale PICO body source (age={source_age:.3f}s)", now
                )
                return None

        action = decode_vector(raw_action, self.NUM_MIMIC_OBS)
        if action is None:
            self._warn_body("invalid 35-D action", now)
        return action

    def _read_hand_target(self, side: str, now: float) -> Optional[np.ndarray]:
        raw_action, raw_timestamp = self.redis.mget(
            f"wuji_action_hand_{side}", f"wuji_action_timestamp_{side}"
        )
        age = self._age(raw_timestamp, 1.0, now)
        problem = None
        if age is None:
            problem = "missing action timestamp"
        elif age < -1.0 or age > self.hand_command_timeout:
            problem = f"stale action (age={age:.3f}s)"
        action = decode_vector(raw_action, 20) if problem is None else None
        if action is None and problem is None:
            problem = "invalid 20-D action"
        if problem:
            if now - self.hand_warning_times[side] >= 2.0:
                print(f"WARNING {side}: {problem}; holding last target")
                self.hand_warning_times[side] = now
            return None
        return self._clip_hand_target(side, action)

    def _body_state(self):
        dof_pos = self.data.qpos[self.body_qpos].copy()
        dof_vel = self.data.qvel[self.body_dof].copy()
        quaternion = self.data.qpos[self.pelvis_qpos + 3 : self.pelvis_qpos + 7]
        angular_velocity = self.data.qvel[
            self.pelvis_dof + 3 : self.pelvis_dof + 6
        ].copy()
        return dof_pos, dof_vel, quaternion, angular_velocity

    def _publish_state(
        self,
        dof_pos: np.ndarray,
        quaternion: np.ndarray,
        angular_velocity: np.ndarray,
    ) -> None:
        rpy = quatToEuler(quaternion)
        body = np.concatenate([angular_velocity, rpy[:2], dof_pos])
        pipeline = self.redis.pipeline()
        pipeline.set(
            "state_body_unitree_g1_with_hands", json.dumps(body.tolist())
        )
        pipeline.set(
            "action_low_level_unitree_g1_with_hands",
            json.dumps(self.last_pd_target.tolist()),
        )
        pipeline.set("state_neck_unitree_g1_with_hands", json.dumps([0.0, 0.0]))
        pipeline.set("t_state", int(time.time() * 1000))
        state_time = time.time()
        for side, addresses in self.hand_qpos.items():
            pipeline.set(
                f"wuji_state_hand_{side}",
                json.dumps(self.data.qpos[addresses].tolist()),
            )
            pipeline.set(f"wuji_state_timestamp_{side}", str(state_time))
        pipeline.execute()

    def _policy_update(
        self,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        quaternion: np.ndarray,
        angular_velocity: np.ndarray,
        now: float,
    ) -> None:
        received_mimic = self._read_body_mimic(now)
        if received_mimic is not None:
            self.last_body_mimic = received_mimic
        mimic = self.last_body_mimic

        # High-level tracking loss must not turn off the closed-loop balance
        # policy. Continue inference against the last valid mimic target; only
        # the high-level target is held. At startup that target is the normal
        # upright DEFAULT_BODY_MIMIC.
        rpy = quatToEuler(quaternion)
        observed_velocity = dof_vel.copy()
        observed_velocity[ANKLE_INDICES] = 0.0
        proprio = np.concatenate(
            [
                angular_velocity * 0.25,
                rpy[:2],
                dof_pos - DEFAULT_BODY_POSITION,
                observed_velocity * 0.05,
                self.last_action,
            ]
        )
        current = np.concatenate([mimic, proprio])
        history = np.asarray(self.history).reshape(-1)
        observation = np.concatenate([current, history, mimic])
        if observation.shape != (self.TOTAL_OBS,):
            raise ValueError(
                f"expected {self.TOTAL_OBS} policy inputs, got {observation.shape}"
            )
        self.history.append(current)
        raw_action = self.policy(observation)
        self.last_action = raw_action.copy()
        self.last_pd_target = (
            DEFAULT_BODY_POSITION
            + np.clip(raw_action, -10.0, 10.0) * ACTION_SCALE
        )

        for side in self.hand_targets:
            target = self._read_hand_target(side, now)
            if target is not None:
                self.hand_targets[side] = target

        self._publish_state(dof_pos, quaternion, angular_velocity)

    def run(self) -> None:
        self.reset()
        viewer = None
        if not self.headless:
            from mujoco import viewer as mj_viewer

            viewer = mj_viewer.launch_passive(
                self.model, self.data, show_left_ui=False, show_right_ui=False
            )
            viewer.cam.distance = 2.0

        step = 0
        started = time.monotonic()
        print("Starting software-only G1 + Wuji MuJoCo simulation")
        try:
            while (
                (self.max_steps <= 0 or step < self.max_steps)
                and (viewer is None or viewer.is_running())
            ):
                loop_start = time.monotonic()
                dof_pos, dof_vel, quaternion, angular_velocity = self._body_state()
                if step % self.decimation == 0:
                    self._policy_update(
                        dof_pos,
                        dof_vel,
                        quaternion,
                        angular_velocity,
                        time.time(),
                    )
                    if viewer is not None:
                        viewer.cam.lookat[:] = self.data.xpos[
                            self.model.body("pelvis").id
                        ]
                        viewer.sync()
                    policy_step = step // self.decimation
                    if self.print_every and policy_step % self.print_every == 0:
                        states = ", ".join(
                            f"{side}_q=[{self.data.qpos[addresses].min():.2f},"
                            f" {self.data.qpos[addresses].max():.2f}]"
                            for side, addresses in self.hand_qpos.items()
                        )
                        print(f"step={step} {states}")

                torque = (
                    (self.last_pd_target - dof_pos) * STIFFNESS
                    - dof_vel * DAMPING
                )
                self.data.ctrl[self.body_actuators] = np.clip(
                    torque, -TORQUE_LIMITS, TORQUE_LIMITS
                )
                for side, actuator_ids in self.hand_actuators.items():
                    self.data.ctrl[actuator_ids] = self.hand_targets[side]
                mujoco.mj_step(self.model, self.data)
                step += 1

                if self.realtime:
                    remaining = self.sim_dt - (time.monotonic() - loop_start)
                    if remaining > 0:
                        time.sleep(remaining)
        except KeyboardInterrupt:
            print("Stopping simulation")
        finally:
            if viewer is not None:
                viewer.close()
            elapsed = time.monotonic() - started
            print(f"Simulation finished after {step} steps ({elapsed:.2f}s wall time)")


def parse_args(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xml", default=str(root / "assets/g1/g1_sim2sim_29dof.xml")
    )
    parser.add_argument("--policy")
    parser.add_argument("--left-config")
    parser.add_argument("--right-config")
    parser.add_argument("--redis-ip", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--policy-frequency", type=float, default=100.0)
    parser.add_argument("--body-command-timeout", type=float, default=0.5)
    parser.add_argument("--hand-command-timeout", type=float, default=0.5)
    parser.add_argument(
        "--require-fresh-pico-body",
        action="store_true",
        help=("Accept new 35-D body targets only while pico_body_timestamp is "
              "fresh; otherwise keep balancing on the last valid target."),
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--model-only", action="store_true")
    parser.add_argument("--save-composed-model", metavar="PATH.mjb")
    parser.add_argument(
        "--left-mount-pos",
        type=float,
        nargs=3,
        default=DEFAULT_MOUNT_POSITIONS["left"],
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--right-mount-pos",
        type=float,
        nargs=3,
        default=DEFAULT_MOUNT_POSITIONS["right"],
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--left-mount-quat",
        type=float,
        nargs=4,
        default=DEFAULT_MOUNT_QUATERNION,
        metavar=("W", "X", "Y", "Z"),
    )
    parser.add_argument(
        "--right-mount-quat",
        type=float,
        nargs=4,
        default=DEFAULT_MOUNT_QUATERNION,
        metavar=("W", "X", "Y", "Z"),
    )
    args = parser.parse_args(argv)
    if not args.left_config and not args.right_config:
        parser.error("at least one of --left-config/--right-config is required")
    if not args.model_only and not args.policy:
        parser.error("--policy is required unless --model-only is used")
    if args.policy_frequency <= 0:
        parser.error("--policy-frequency must be positive")
    if args.body_command_timeout <= 0 or args.hand_command_timeout <= 0:
        parser.error("command timeouts must be positive")
    if args.max_steps < 0 or args.print_every < 0:
        parser.error("--max-steps and --print-every cannot be negative")
    return args


def build_model_from_args(args) -> CombinedWujiModel:
    configs = {
        side: getattr(args, f"{side}_config")
        for side in ("left", "right")
        if getattr(args, f"{side}_config")
    }
    positions = {
        side: getattr(args, f"{side}_mount_pos") for side in configs
    }
    quaternions = {
        side: getattr(args, f"{side}_mount_quat") for side in configs
    }
    return compose_g1_wuji_model(
        args.xml,
        configs,
        mount_positions=positions,
        mount_quaternions=quaternions,
        save_compiled_model=args.save_composed_model,
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    bundle = build_model_from_args(args)
    print(
        f"Validated combined model: nq={bundle.model.nq}, nv={bundle.model.nv}, "
        f"nu={bundle.model.nu}"
    )
    for side, names in bundle.hand_actuator_names.items():
        print(f"{side} MJCF: {bundle.hand_mjcf_paths[side]}")
        print(f"{side} actuator/device order: {list(names)}")
    if args.model_only:
        print("MODEL ONLY: no Redis connection, policy inference, or hardware access")
        return 0

    policy = Path(args.policy).expanduser().resolve()
    if not policy.is_file():
        raise FileNotFoundError(f"TWIST2 policy not found: {policy}")
    controller = G1WujiMujocoController(
        bundle=bundle,
        policy_path=policy,
        redis_ip=args.redis_ip,
        redis_port=args.redis_port,
        device=args.device,
        policy_frequency=args.policy_frequency,
        body_command_timeout=args.body_command_timeout,
        hand_command_timeout=args.hand_command_timeout,
        realtime=not args.no_realtime,
        headless=args.headless,
        max_steps=args.max_steps,
        print_every=args.print_every,
        require_fresh_pico_body=args.require_fresh_pico_body,
    )
    controller.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
