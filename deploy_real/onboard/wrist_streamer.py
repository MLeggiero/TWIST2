#!/usr/bin/env python3
"""
Arducam USB2.0 fisheye wrist camera streamer for the G1 Orin.

Run once per wrist camera (separate process per --device/--port) via
start_wrist_left.sh / start_wrist_right.sh. Captures RGB over V4L2 and
publishes frames over ZMQ PUB with the same 16-byte header used by
realsense_streamer.py, with depth_data_len always 0 (no depth payload) so
data_utils/vision_client.py::VisionClient needs no changes to consume it.

Wire format (per message):
  [4B: width][4B: height][4B: rgb_jpeg_len][4B: depth_data_len=0]
  [rgb_jpeg_bytes]

Use --test to publish synthetic frames without a camera attached.
"""

import argparse
import signal
import struct
import time

import cv2
import numpy as np
import zmq


def main():
    parser = argparse.ArgumentParser(description="Arducam wrist camera ZMQ streamer")
    parser.add_argument("--device", type=str, default=None,
                        help="V4L2 device path, e.g. /dev/video_wrist_left (required unless --test)")
    parser.add_argument("--port", type=int, default=5556, help="ZMQ PUB port")
    parser.add_argument("--width", type=int, default=640, help="Stream width")
    parser.add_argument("--height", type=int, default=480, help="Stream height")
    parser.add_argument("--fps", type=int, default=30, help="Camera FPS")
    parser.add_argument("--jpeg-quality", type=int, default=80, help="JPEG encode quality (0-100)")
    parser.add_argument("--test", action="store_true", help="Publish synthetic frames (no hardware needed)")
    args = parser.parse_args()

    if not args.test and args.device is None:
        parser.error("--device is required unless --test is set")

    # ---- ZMQ setup ----
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://*:{args.port}")
    print(f"[Wrist] ZMQ PUB bound on port {args.port}")

    # ---- Camera setup ----
    cap = None
    if args.test:
        print("[Wrist] Running in TEST mode with synthetic frames")
    else:
        cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera device {args.device}")
        print(f"[Wrist] Opened {args.device}: {args.width}x{args.height}@{args.fps}fps")

    # JPEG encode params
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]

    # Graceful shutdown
    running = True

    def _sigint_handler(_sig, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _sigint_handler)

    frame_count = 0
    t_start = time.time()
    target_dt = 1.0 / args.fps
    test_angle = 0.0

    try:
        while running:
            t_frame_start = time.time()

            if args.test:
                # Synthetic moving-gradient pattern
                test_angle += 0.05
                ramp = np.linspace(0, 255, args.width, dtype=np.uint8)
                color_img = np.tile(ramp, (args.height, 3, 1)).transpose(0, 2, 1).astype(np.uint8)
                color_img = np.roll(color_img, int(test_angle * 10) % args.width, axis=1)
            else:
                ret, color_img = cap.read()
                if not ret:
                    continue

            ok, jpeg_buf = cv2.imencode(".jpg", color_img, encode_params)
            if not ok:
                continue
            jpeg_bytes = jpeg_buf.tobytes()

            # Build message: 16-byte header + payload (depth_data_len always 0)
            header = struct.pack("iiii", args.width, args.height, len(jpeg_bytes), 0)
            message = header + jpeg_bytes

            sock.send(message, zmq.NOBLOCK)

            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - t_start
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"[Wrist] Frames: {frame_count}, FPS: {fps:.1f}, JPEG: {len(jpeg_bytes)}B")

            if args.test:
                elapsed = time.time() - t_frame_start
                if elapsed < target_dt:
                    time.sleep(target_dt - elapsed)

    finally:
        if cap is not None:
            cap.release()
        sock.close()
        ctx.term()
        print(f"[Wrist] Shutdown complete. Total frames: {frame_count}")


if __name__ == "__main__":
    main()
