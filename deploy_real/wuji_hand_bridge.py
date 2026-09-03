#!/usr/bin/env python3
"""Redis bridge from PICO MediaPipe landmarks to Wuji Hand 2.

Dry-run is intentionally hardware-free: wuji_sdk is imported only while creating
a physical session. Commands and measured state use Wuji Hand 2 finger-major
device order (finger1..5, joint1..4), matching the MJCF actuator order.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import redis

from wuji_velocity_limiter import JointVelocityLimiter


SIDES = ("left", "right")


def resolve_mjcf_path(config_path: Path) -> Optional[str]:
    import yaml

    config_path = config_path.resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    relative = (config.get("optimizer") or {}).get("mjcf_path")
    return str((config_path.parent / relative).resolve()) if relative else None


def mjcf_joint_order(mjcf_path: Optional[str]) -> Optional[list]:
    if not mjcf_path:
        return None
    import mujoco

    model = mujoco.MjModel.from_xml_path(mjcf_path)
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(model.njnt)
    ]


def qpos_reorder_perm(src_names, dst_names) -> Optional[np.ndarray]:
    if not dst_names:
        return None
    lookup = {name: index for index, name in enumerate(src_names)}
    try:
        permutation = np.asarray([lookup[name] for name in dst_names], dtype=int)
    except KeyError:
        return None
    return permutation if len(permutation) == len(src_names) else None


def validate_keypoints(raw) -> Optional[np.ndarray]:
    try:
        value = np.asarray(json.loads(raw), dtype=np.float32)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if value.shape != (21, 3) or not np.all(np.isfinite(value)):
        return None
    return value


def device_order_state(frame) -> Optional[np.ndarray]:
    """Map a complete SDK state frame by explicit nid, never packet order."""
    if frame is None or not getattr(frame, "joints", None):
        return None
    entries = {int(entry.nid): float(entry.position) for entry in frame.joints}
    if set(entries) == set(range(20)):
        ids = range(20)
    elif set(entries) == set(range(1, 21)):
        ids = range(1, 21)
    else:
        return None
    state = np.asarray([entries[nid] for nid in ids], dtype=np.float32)
    return state if np.all(np.isfinite(state)) else None


def _set_with_retry(description, callback, attempts=3, backoff=0.6):
    for attempt in range(attempts):
        try:
            return callback()
        except (AttributeError, TypeError):
            raise
        except Exception:
            if attempt == attempts - 1:
                raise
            print(f"{description} timed out; retrying in {backoff:.1f}s")
            time.sleep(backoff)


class WujiHand2Session:
    """One physical hand session; process-level code owns manager cleanup."""

    ENABLE_TIMEOUT = 5.0

    def __init__(self, manager, sdk, side, address, kp, kd, current_limit):
        self.side = side
        self.hand = manager.connect(address=address, device_name="wuji_hand_2")
        self.publisher = None
        self.state_subscription = None
        self.enabled = False
        try:
            reported = str(self.hand.handedness().get()).lower()
            if reported != side:
                raise RuntimeError(
                    f"{address} reports {reported!r}, expected {side!r}"
                )
            online = int(self.hand.online_joints_count().get())
            if online != 20:
                raise RuntimeError(f"{side} hand has {online}/20 joints online")

            _set_with_retry(
                f"{side} effort_limit",
                lambda: self.hand.effort_limit().set(current_limit),
            )
            _set_with_retry(
                f"{side} mit_params",
                lambda: self.hand.mit_params().set((kp, kd)),
            )
            measured = np.asarray(self.hand.read_joint_state().position, dtype=np.float32)
            if measured.shape != (20,) or not np.all(np.isfinite(measured)):
                raise RuntimeError(f"invalid initial {side} measured state")
            self.initial_state = measured
            self.state_subscription = self.hand.joint_states().subscribe()

            self.hand.enable()
            diagnostics = self.hand.joint_diagnostics().subscribe()
            try:
                deadline = time.monotonic() + self.ENABLE_TIMEOUT
                while time.monotonic() < deadline:
                    time.sleep(0.2)
                    frame = diagnostics.recv()
                    if frame is None:
                        continue
                    live = [entry for entry in frame.joints if entry.vbus_v_fb > 0.5]
                    if live and all(entry.status_word.ext_state == 2 for entry in live):
                        break
                else:
                    self.hand.disable()
                    raise RuntimeError(f"{side} hand enable timeout")
            finally:
                diagnostics.close()

            self.enabled = True
            self.publisher = self.hand.joint_command().publish()
            self.JointCommand = sdk.JointCommand
        except BaseException:
            self.close()
            raise

    def send(self, qpos: np.ndarray) -> None:
        positions = np.asarray(qpos, dtype=np.float64)
        if positions.shape != (20,) or not np.all(np.isfinite(positions)):
            raise ValueError("Wuji command must be finite shape (20,)")
        self.publisher.send(
            [self.JointCommand(float(position), 0.0, 0.0) for position in positions]
        )

    def receive_state(self) -> Optional[np.ndarray]:
        return device_order_state(self.state_subscription.recv())

    def disable(self) -> None:
        if self.enabled:
            self.hand.disable()
            self.enabled = False

    def close(self) -> None:
        if self.publisher is not None:
            self.publisher.close()
            self.publisher = None
        if self.state_subscription is not None:
            self.state_subscription.close()
            self.state_subscription = None
        if getattr(self, "hand", None) is not None:
            try:
                self.disable()
            finally:
                self.hand.disconnect()


@dataclass
class SideRuntime:
    side: str
    config: Path
    address: str
    retargeter: object
    permutation: Optional[np.ndarray]
    max_velocity: np.ndarray
    last_input_timestamp: Optional[float] = None
    last_valid_wall_time: Optional[float] = None
    last_warning_time: float = 0.0
    session: Optional[WujiHand2Session] = None
    target: Optional[np.ndarray] = None
    velocity_limiter: Optional[JointVelocityLimiter] = None
    disabled_for_timeout: bool = False

    def to_device_order(self, keypoints: np.ndarray) -> np.ndarray:
        q_urdf = np.asarray(self.retargeter.retarget(keypoints), dtype=np.float32)
        if q_urdf.shape != (20,) or not np.all(np.isfinite(q_urdf)):
            raise ValueError(f"{self.side} retargeter returned invalid shape/value")
        return q_urdf if self.permutation is None else q_urdf[self.permutation]


def build_runtime(side: str, config_value: str, address: str) -> SideRuntime:
    from wuji_retargeting import Retargeter

    config = Path(config_value).expanduser().resolve()
    if not config.is_file():
        raise FileNotFoundError(f"{side} config not found: {config}")
    retargeter = Retargeter.from_yaml(str(config), side)
    mjcf_path = resolve_mjcf_path(config)
    permutation = qpos_reorder_perm(
        retargeter.optimizer.robot.dof_joint_names,
        mjcf_joint_order(mjcf_path),
    )
    if mjcf_path is not None and permutation is None:
        raise ValueError(
            f"{side} URDF and MJCF joint names do not align; refusing unsafe identity order"
        )
    if permutation is not None:
        print(f"{side}: URDF -> device permutation {permutation.tolist()}")
    max_velocity = np.asarray(
        retargeter.optimizer.robot.model.velocityLimit, dtype=np.float64
    )
    if max_velocity.shape != (20,) or not np.all(np.isfinite(max_velocity)):
        raise ValueError(f"{side} URDF must provide 20 finite velocity limits")
    if np.any(max_velocity <= 0):
        raise ValueError(f"{side} URDF velocity limits must all be positive")
    if permutation is not None:
        max_velocity = max_velocity[permutation]
    return SideRuntime(
        side=side,
        config=config,
        address=address,
        retargeter=retargeter,
        permutation=permutation,
        max_velocity=max_velocity,
    )


def read_input(client, side: str, timeout: float, now: float):
    pipeline = client.pipeline()
    pipeline.get(f"pico_hand_{side}_mediapipe")
    pipeline.get(f"pico_hand_{side}_timestamp")
    raw_keypoints, raw_timestamp = pipeline.execute()
    try:
        timestamp = float(raw_timestamp)
    except (TypeError, ValueError):
        return None, None, "missing timestamp"
    age = now - timestamp
    if age < -1.0:
        return None, timestamp, "timestamp is in the future"
    if age > timeout:
        return None, timestamp, f"tracking stale ({age:.3f}s)"
    keypoints = validate_keypoints(raw_keypoints)
    if keypoints is None:
        return None, timestamp, "invalid 21x3 landmarks"
    return keypoints, timestamp, None


def publish_vector(client, key: str, timestamp_key: str, value: np.ndarray) -> None:
    pipeline = client.pipeline()
    pipeline.set(key, json.dumps(np.asarray(value).tolist()))
    pipeline.set(timestamp_key, str(time.time()))
    pipeline.execute()


def interpolate_startup(
    session, target, duration, rate_hz, velocity_limiter, publish_callback
):
    start = session.initial_state
    started = time.monotonic()
    while True:
        now = time.monotonic()
        alpha = 1.0 if duration <= 0 else min((now - started) / duration, 1.0)
        requested = start + (target - start) * alpha
        command = velocity_limiter.step(requested, now)
        session.send(command)
        publish_callback(command)
        if alpha >= 1.0 and velocity_limiter.at_target(target):
            break
        time.sleep(1.0 / rate_hz)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redis-ip", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--left-config")
    parser.add_argument("--right-config")
    parser.add_argument("--left-ip", default="")
    parser.add_argument("--right-ip", default="")
    parser.add_argument("--kp", type=float, default=3.0)
    parser.add_argument("--kd", type=float, default=0.1)
    parser.add_argument("--current-limit", type=float, default=1.5)
    parser.add_argument("--tracking-timeout", type=float, default=0.25)
    parser.add_argument("--startup-interpolation-duration", type=float, default=1.75)
    parser.add_argument("--rate-hz", type=float, default=100.0)
    parser.add_argument(
        "--velocity-limit-scale",
        type=float,
        default=1.0,
        help=(
            "Scale the per-joint URDF velocity ceilings in (0, 1]; "
            "values below 1 provide a slower operational limit"
        ),
    )
    parser.add_argument("--disable-after-timeout", type=float, default=0.0,
                        help="Seconds before disabling motors; 0 keeps this disabled")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Exit after N new frames (primarily for dry-run tests)")
    args = parser.parse_args(argv)
    if not args.left_config and not args.right_config:
        parser.error("at least one of --left-config/--right-config is required")
    if args.tracking_timeout <= 0 or args.rate_hz <= 0:
        parser.error("--tracking-timeout and --rate-hz must be positive")
    if args.disable_after_timeout < 0 or args.startup_interpolation_duration < 0:
        parser.error("timeout/interpolation durations cannot be negative")
    if not 0 < args.velocity_limit_scale <= 1:
        parser.error("--velocity-limit-scale must be in (0, 1]")
    if not args.dry_run and args.left_config and args.right_config:
        if not args.left_ip or not args.right_ip:
            parser.error("dual-hand hardware mode requires both explicit hand addresses")
    return args


def _single_discovered_address(manager) -> str:
    hands = [item for item in manager.scan() if str(item.sn).upper().startswith("WH")]
    if len(hands) != 1:
        raise RuntimeError(
            f"expected exactly one discoverable Wuji Hand 2, found {len(hands)}; pass --left-ip/--right-ip"
        )
    return str(hands[0].address)


def main(argv=None) -> int:
    args = parse_args(argv)
    client = redis.Redis(host=args.redis_ip, port=args.redis_port, db=0)
    client.ping()
    runtimes: Dict[str, SideRuntime] = {}
    for side in SIDES:
        config = getattr(args, f"{side}_config")
        if config:
            runtime = build_runtime(side, config, getattr(args, f"{side}_ip"))
            runtime.max_velocity = runtime.max_velocity * args.velocity_limit_scale
            runtime.velocity_limiter = JointVelocityLimiter(
                runtime.max_velocity, 1.0 / args.rate_hz
            )
            print(
                f"{side}: commanded velocity limits "
                f"range=[{runtime.max_velocity.min():.3f}, "
                f"{runtime.max_velocity.max():.3f}] rad/s "
                f"(URDF scale={args.velocity_limit_scale:.3f})"
            )
            runtimes[side] = runtime

    manager = sdk = None
    if args.dry_run:
        print("DRY RUN: no Wuji SDK connection, enable, or motor commands will occur")
    else:
        import wuji_sdk
        from wuji_sdk import SdkManager

        sdk = wuji_sdk
        manager = SdkManager.instance()
        for runtime in runtimes.values():
            if not runtime.address:
                runtime.address = _single_discovered_address(manager)

    processed = 0
    try:
        while True:
            loop_start = time.monotonic()
            now = time.time()
            for side, runtime in runtimes.items():
                keypoints, timestamp, problem = read_input(
                    client, side, args.tracking_timeout, now
                )
                if problem:
                    if now - runtime.last_warning_time >= 2.0:
                        print(f"WARNING {side}: {problem}; holding last target")
                        runtime.last_warning_time = now
                    if (runtime.session is not None and args.disable_after_timeout > 0
                            and runtime.last_valid_wall_time is not None
                            and now - runtime.last_valid_wall_time > args.disable_after_timeout
                            and not runtime.disabled_for_timeout):
                        runtime.session.disable()
                        runtime.disabled_for_timeout = True
                        print(f"{side}: motors disabled after long timeout; restart bridge to resume")
                    continue
                if runtime.disabled_for_timeout:
                    continue

                new_input = timestamp != runtime.last_input_timestamp
                if new_input:
                    runtime.target = runtime.to_device_order(keypoints)
                    runtime.last_input_timestamp = timestamp
                    runtime.last_valid_wall_time = now
                    processed += 1
                if runtime.target is None:
                    continue

                action_key = f"wuji_action_hand_{side}"
                action_time_key = f"wuji_action_timestamp_{side}"
                publish = lambda value, ak=action_key, tk=action_time_key: publish_vector(
                    client, ak, tk, value
                )

                if args.dry_run:
                    if runtime.velocity_limiter.position is None:
                        runtime.velocity_limiter.reset(runtime.target, loop_start)
                    command = runtime.velocity_limiter.step(runtime.target, loop_start)
                    publish(command)
                else:
                    if runtime.session is None:
                        runtime.session = WujiHand2Session(
                            manager, sdk, side, runtime.address,
                            args.kp, args.kd, args.current_limit,
                        )
                        runtime.velocity_limiter.reset(
                            runtime.session.initial_state, time.monotonic()
                        )
                        interpolate_startup(
                            runtime.session,
                            runtime.target,
                            args.startup_interpolation_duration,
                            args.rate_hz,
                            runtime.velocity_limiter,
                            publish,
                        )
                    else:
                        command = runtime.velocity_limiter.step(
                            runtime.target, time.monotonic()
                        )
                        runtime.session.send(command)
                        publish(command)

                if new_input and (processed == 1 or processed % 100 == 0):
                    print(
                        f"{side}: input=(21, 3) output={runtime.target.shape} "
                        f"target_range=[{runtime.target.min():.3f}, "
                        f"{runtime.target.max():.3f}]"
                    )

            for side, runtime in runtimes.items():
                if runtime.session is None:
                    continue
                state = runtime.session.receive_state()
                if state is not None:
                    publish_vector(
                        client,
                        f"wuji_state_hand_{side}",
                        f"wuji_state_timestamp_{side}",
                        state,
                    )

            if args.max_frames and processed >= args.max_frames:
                return 0
            elapsed = time.monotonic() - loop_start
            time.sleep(max(0.0, 1.0 / args.rate_hz - elapsed))
    except KeyboardInterrupt:
        print("Stopping Wuji bridge")
        return 0
    finally:
        for runtime in runtimes.values():
            if runtime.session is not None:
                try:
                    runtime.session.close()
                except Exception as error:
                    print(f"{runtime.side} cleanup warning: {error}")
        if manager is not None:
            manager.disconnect_all()


if __name__ == "__main__":
    raise SystemExit(main())
