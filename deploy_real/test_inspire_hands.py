#!/usr/bin/env python3
"""
Standalone Inspire hand test — no robot, no Unitree SDK needed.

Reads PICO controller trigger/grip from Redis (published by xrobot_teleop_to_robot_w_hand.py)
and sends commands directly to Inspire hands via Modbus TCP.

Usage:
  # Standalone sweep — no teleop/Redis needed, just send a test command to a hand:
  python test_inspire_hands.py --hand left --sweep
  python test_inspire_hands.py --hand right --sweep

  # Or drive from teleop via Redis:
  Terminal 1 (gmr env): python xrobot_teleop_to_robot_w_hand.py --robot unitree_g1 --hand_type inspire
  Terminal 2 (twist2 env): python test_inspire_hands.py --hand left

Use --hand {left,right,both} to pick which hand(s) to connect to and command.
Use --no-hardware to just monitor Redis values without connecting to hands.
"""
import argparse
import json
import time
import sys

import numpy as np
import redis

sys.path.insert(0, '.')


def run_sweep(hand_ctrl, enable_left, enable_right):
    """Send a standalone open->close->open sweep to the enabled hand(s).

    Does not touch Redis — useful for bench-testing a single hand. Disabled
    hands are ignored by ctrl_dual_hand, so passing both targets is safe.
    """
    print("\nRunning standalone sweep (0=open -> 1000=closed -> open)...")
    sequence = list(range(0, 1001, 100)) + list(range(1000, -1, -100))
    for angle in sequence:
        hand_ctrl.ctrl_dual_hand([angle] * 6, [angle] * 6)
        time.sleep(0.25)
        l_state, r_state = hand_ctrl.get_hand_state()
        parts = []
        if enable_left:
            parts.append(f"L={[f'{v:.0f}' for v in l_state]}")
        if enable_right:
            parts.append(f"R={[f'{v:.0f}' for v in r_state]}")
        print(f"  target={angle:4d}  " + " | ".join(parts))
    # Leave hands open
    hand_ctrl.ctrl_dual_hand([0] * 6, [0] * 6)
    time.sleep(0.5)
    print("Sweep done.")


def main():
    parser = argparse.ArgumentParser(description="Test Inspire hands from Redis or via standalone sweep")
    parser.add_argument("--left_ip", default="192.168.124.210", help="Left hand IP")
    parser.add_argument("--right_ip", default="192.168.123.211", help="Right hand IP")
    parser.add_argument("--hand", choices=["left", "right", "both"], default="both",
                        help="Which hand(s) to connect to and command")
    parser.add_argument("--sweep", action="store_true",
                        help="Run a standalone open/close sweep (no Redis/teleop needed) and exit")
    parser.add_argument("--no-hardware", action="store_true",
                        help="Monitor Redis only, don't connect to hands")
    parser.add_argument("--rate", type=float, default=50, help="Control loop Hz")
    args = parser.parse_args()

    enable_left = args.hand in ("left", "both")
    enable_right = args.hand in ("right", "both")
    print(f"Hand selection: {args.hand} (left={enable_left}, right={enable_right})")

    if args.sweep and args.no_hardware:
        parser.error("--sweep requires hardware; do not combine with --no-hardware")

    # Connect to Inspire hands (optional)
    hand_ctrl = None
    if not args.no_hardware:
        try:
            from robot_control.inspire_hand_wrapper import InspireHandController
            hand_ctrl = InspireHandController(
                left_ip=args.left_ip,
                right_ip=args.right_ip,
                re_init=False,
                enable_left=enable_left,
                enable_right=enable_right,
            )
            print(f"Inspire hands connected: "
                  f"{'L=' + args.left_ip + ' ' if enable_left else ''}"
                  f"{'R=' + args.right_ip if enable_right else ''}")
        except Exception as e:
            print(f"Failed to connect to hands: {e}")
            if args.sweep:
                return
            print("Running in monitor-only mode")
    else:
        print("Monitor-only mode (no hardware)")

    # Standalone sweep mode: drive the hand directly, no Redis needed.
    if args.sweep:
        try:
            run_sweep(hand_ctrl, enable_left, enable_right)
        except KeyboardInterrupt:
            print("\nInterrupted.")
        finally:
            print("Opening hands and closing...")
            hand_ctrl.ctrl_dual_hand(np.zeros(6), np.zeros(6))
            time.sleep(0.5)
            hand_ctrl.close()
        return

    # Connect to Redis for teleop-driven mode
    r = redis.Redis(host='localhost', port=6379, db=0)
    r.ping()
    print("Redis connected")

    dt = 1.0 / args.rate
    count = 0

    print("\nWaiting for hand data on Redis...")
    print("  Keys: action_hand_left_unitree_g1_with_hands")
    print("        action_hand_right_unitree_g1_with_hands")
    print("Press Ctrl+C to exit\n")

    try:
        while True:
            t0 = time.time()

            # Read hand actions from Redis
            left_raw = r.get("action_hand_left_unitree_g1_with_hands")
            right_raw = r.get("action_hand_right_unitree_g1_with_hands")

            if left_raw is None and right_raw is None:
                if count % 50 == 0:
                    print("No hand data in Redis yet...", end="\r")
                count += 1
                time.sleep(dt)
                continue

            left_cmd = np.array(json.loads(left_raw), dtype=np.float32) if left_raw else np.zeros(6)
            right_cmd = np.array(json.loads(right_raw), dtype=np.float32) if right_raw else np.zeros(6)

            # Send to hardware
            if hand_ctrl is not None:
                hand_ctrl.ctrl_dual_hand(left_cmd, right_cmd)

            # Print diagnostics
            count += 1
            if count % 25 == 0:
                l_fingers = left_cmd[:4]
                l_thumb = left_cmd[4:]
                r_fingers = right_cmd[:4]
                r_thumb = right_cmd[4:]
                print(f"[{count:6d}] "
                      f"L fingers={l_fingers[0]:4.0f} thumb={l_thumb[0]:4.0f} | "
                      f"R fingers={r_fingers[0]:4.0f} thumb={r_thumb[0]:4.0f}")

                # Also read state feedback if hardware connected
                if hand_ctrl is not None:
                    try:
                        l_state, r_state = hand_ctrl.get_hand_state()
                        print(f"         L state={[f'{v:.0f}' for v in l_state]} | "
                              f"R state={[f'{v:.0f}' for v in r_state]}")
                    except Exception:
                        pass

            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\n\nShutting down...")
        if hand_ctrl is not None:
            # Open hands on exit
            print("Opening hands...")
            hand_ctrl.ctrl_dual_hand(np.zeros(6), np.zeros(6))
            time.sleep(0.5)
        print("Done.")


if __name__ == "__main__":
    main()
