#!/usr/bin/env python3
"""
RealSense D435i streamer for the G1 Orin.

Captures RGB + aligned depth from the Intel RealSense D435i and publishes
frames over ZMQ PUB with a 16-byte header.

Wire format (per message):
  [4B: width][4B: height][4B: rgb_jpeg_len][4B: depth_data_len]
  [rgb_jpeg_bytes][depth_bytes (uint16 raw)]

When --no-depth is set, depth_data_len = 0 and no depth bytes are appended.
"""

import argparse
import signal
import struct
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import zmq


def main():
    parser = argparse.ArgumentParser(description="RealSense D435i ZMQ streamer")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ PUB port")
    parser.add_argument("--width", type=int, default=424, help="Stream width")
    parser.add_argument("--height", type=int, default=240, help="Stream height")
    parser.add_argument("--fps", type=int, default=60, help="Camera FPS")
    parser.add_argument("--jpeg-quality", type=int, default=80, help="JPEG encode quality (0-100)")
    parser.add_argument("--no-depth", action="store_true", help="Disable depth streaming")
    args = parser.parse_args()

    # ---- ZMQ setup ----
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://*:{args.port}")
    print(f"[RealSense] ZMQ PUB bound on port {args.port}")

    # ---- RealSense setup ----
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    if not args.no_depth:
        config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    pipeline.start(config)
    print(f"[RealSense] Pipeline started: {args.width}x{args.height}@{args.fps}fps, depth={'off' if args.no_depth else 'on'}")

    # Align depth to color
    align = rs.align(rs.stream.color) if not args.no_depth else None

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

    try:
        while running:
            frames = pipeline.wait_for_frames()

            if align is not None:
                frames = align.process(frames)

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_img = np.asanyarray(color_frame.get_data())  # BGR uint8

            # JPEG encode RGB
            ok, jpeg_buf = cv2.imencode(".jpg", color_img, encode_params)
            if not ok:
                continue
            jpeg_bytes = jpeg_buf.tobytes()

            # Depth
            if not args.no_depth:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    depth_img = np.asanyarray(depth_frame.get_data())  # uint16
                    depth_bytes = depth_img.tobytes()
                else:
                    depth_bytes = b""
            else:
                depth_bytes = b""

            # Build message: 16-byte header + payloads
            header = struct.pack("iiii", args.width, args.height, len(jpeg_bytes), len(depth_bytes))
            message = header + jpeg_bytes + depth_bytes

            sock.send(message, zmq.NOBLOCK)

            frame_count += 1
            if frame_count % 60 == 0:
                elapsed = time.time() - t_start
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"[RealSense] Frames: {frame_count}, FPS: {fps:.1f}, JPEG: {len(jpeg_bytes)}B, Depth: {len(depth_bytes)}B")

    finally:
        pipeline.stop()
        sock.close()
        ctx.term()
        print(f"[RealSense] Shutdown complete. Total frames: {frame_count}")


if __name__ == "__main__":
    main()
