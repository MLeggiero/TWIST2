#!/usr/bin/env python3
"""Safely inspect or bench-test the hybrid right-Inspire command stream.

The default is Redis-only: physical hardware is not imported or connected.
Pass ``--hardware`` and an explicit ``--right-ip`` to opt into motor commands.
Never run hardware mode concurrently with the real hybrid G1 controller.
"""

import argparse
import time

import numpy as np
import redis

from inspire_hybrid_utils import (
    InspireCommandFilter,
    decode_inspire_target,
    timestamp_age,
)


ACTION_KEY = "inspire_action_hand_right"
TIMESTAMP_KEY = "inspire_action_timestamp_right"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dry-run or explicitly drive the hybrid right Inspire hand"
    )
    parser.add_argument("--redis-ip", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--right-ip",
                        help="Required explicit right-hand address in hardware mode")
    parser.add_argument(
        "--hardware", action="store_true",
        help="OPT IN to connecting and sending commands to the physical hand",
    )
    parser.add_argument("--tracking-timeout", type=float, default=0.5)
    parser.add_argument("--command-rate-limit", type=float, default=250.0,
                        help="Application slew ceiling in Inspire counts/second")
    parser.add_argument("--startup-interpolation-duration", type=float,
                        default=1.75)
    parser.add_argument("--rate-hz", type=float, default=50.0)
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="Exit after this many loop iterations; zero runs until Ctrl+C",
    )
    args = parser.parse_args()
    if args.hardware and not args.right_ip:
        parser.error("--hardware requires an explicit --right-ip")
    if args.tracking_timeout <= 0:
        parser.error("--tracking-timeout must be positive")
    if args.command_rate_limit <= 0:
        parser.error("--command-rate-limit must be positive")
    if args.startup_interpolation_duration < 0:
        parser.error("--startup-interpolation-duration cannot be negative")
    if args.rate_hz <= 0:
        parser.error("--rate-hz must be positive")
    if args.max_frames < 0:
        parser.error("--max-frames cannot be negative")
    return args


def main():
    args = parse_args()
    redis_client = redis.Redis(
        host=args.redis_ip, port=args.redis_port, db=0
    )
    redis_client.ping()

    period = 1.0 / args.rate_hz
    command_filter = InspireCommandFilter(
        max_rate=args.command_rate_limit,
        control_period=period,
        startup_duration=args.startup_interpolation_duration,
    )
    hand_ctrl = None
    if args.hardware:
        # Keep the hardware dependency and connection out of the default path.
        from robot_control.inspire_hand_wrapper import InspireHandController

        hand_ctrl = InspireHandController(
            right_ip=args.right_ip,
            re_init=False,
            enable_left=False,
            enable_right=True,
            strict_io=True,
        )
        _, measured_right = hand_ctrl.get_hand_state()
        command_filter.reset(measured_right)
        print(f"HARDWARE mode: right Inspire at {args.right_ip}")
    else:
        print("DRY-RUN mode: Redis only; no hardware module or device connection")

    print(f"Reading {ACTION_KEY} and {TIMESTAMP_KEY}; Ctrl+C exits")
    last_warning = 0.0
    last_debug = 0.0
    iterations = 0
    try:
        while args.max_frames == 0 or iterations < args.max_frames:
            loop_start = time.monotonic()
            raw_action, raw_timestamp = redis_client.mget(
                ACTION_KEY, TIMESTAMP_KEY
            )
            age = timestamp_age(raw_timestamp)
            target = decode_inspire_target(raw_action)
            valid = (
                target is not None
                and age is not None
                and -1.0 <= age <= args.tracking_timeout
            )

            if valid:
                if command_filter.hold() is None:
                    # A dry run has no measured state; initialize at the first
                    # valid target. Hardware starts from measured feedback.
                    command_filter.reset(target)
                command = command_filter.step(target)
                if hand_ctrl is not None:
                    hand_ctrl.ctrl_dual_hand(np.zeros(6), command)
                now = time.monotonic()
                if now - last_debug >= 1.0:
                    mode = "sent" if hand_ctrl is not None else "computed"
                    print(
                        f"{mode}: target={np.rint(target).astype(int).tolist()} "
                        f"command={np.rint(command).astype(int).tolist()} "
                        f"age={age:.3f}s"
                    )
                    last_debug = now
            else:
                now = time.monotonic()
                if now - last_warning >= 2.0:
                    reason = ("invalid/missing 6-D target" if target is None
                              else "invalid/missing timestamp" if age is None
                              else f"stale timestamp (age={age:.3f}s)")
                    print(f"WARNING: {reason}; holding without a new write")
                    last_warning = now

            iterations += 1
            elapsed = time.monotonic() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        if hand_ctrl is not None:
            # Never surprise-open or zero the hand during a bench test exit.
            hand_ctrl.close(move_to_default=False)
            print("Disconnected without sending an exit pose")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
