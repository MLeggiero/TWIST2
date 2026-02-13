#!/usr/bin/env python3
"""
Livox MID-360 LiDAR streamer for the G1 Orin.

Captures point cloud data from the Livox MID-360 and publishes
frames over ZMQ PUB with an 8-byte header.

Wire format (per message):
  [4B: num_points][4B: point_data_len]
  [point_data_bytes (N x 4 float32: x, y, z, intensity)]

Requires the Livox SDK2 shared library (liblivox_lidar_sdk_shared.so).
Falls back to synthetic data in --test mode for pipeline validation.

Install Livox SDK2:
  git clone https://github.com/Livox-SDK/Livox-SDK2.git
  cd Livox-SDK2 && mkdir build && cd build
  cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local && make -j$(nproc) && sudo make install
"""

import argparse
import ctypes
import json
import os
import signal
import struct
import threading
import time

import numpy as np
import zmq


class LivoxMID360:
    """Interface to the Livox MID-360 via the Livox SDK2 shared library."""

    def __init__(self, lidar_ip="192.168.1.100", host_ip="192.168.1.50",
                 data_port=56000, cmd_port=56100,
                 sdk_path=None, max_points=30000):
        self.lidar_ip = lidar_ip
        self.host_ip = host_ip
        self.data_port = data_port
        self.cmd_port = cmd_port
        self.max_points = max_points
        self._lock = threading.Lock()
        self._latest_cloud = np.zeros((0, 4), dtype=np.float32)
        self._accumulating = np.zeros((0, 4), dtype=np.float32)
        self._frame_ready = threading.Event()

        # Try to load the SDK2 shared library
        if sdk_path is None:
            candidates = [
                "/usr/local/lib/liblivox_lidar_sdk_shared.so",
                "/usr/lib/liblivox_lidar_sdk_shared.so",
                "liblivox_lidar_sdk_shared.so",
            ]
            for c in candidates:
                if os.path.isfile(c):
                    sdk_path = c
                    break

        if sdk_path is None or not os.path.isfile(sdk_path):
            raise FileNotFoundError(
                "Livox SDK2 shared library not found. "
                "Install it from https://github.com/Livox-SDK/Livox-SDK2 "
                "or pass --sdk-path. Use --test for synthetic data."
            )

        self._lib = ctypes.CDLL(sdk_path)
        self._setup_sdk()

    def _setup_sdk(self):
        """Initialize the Livox SDK2 and start data streaming."""
        # Create a config JSON for the SDK2 init
        config = {
            "MID360": {
                "lidar_net_info": {
                    "cmd_data_port": self.cmd_port,
                    "push_msg_port": 0,
                    "point_data_port": self.data_port,
                    "imu_data_port": 0,
                    "log_data_port": 0,
                },
                "host_net_info": [
                    {
                        "lidar_ip": self.lidar_ip,
                        "host_ip": self.host_ip,
                    }
                ],
            }
        }
        config_str = json.dumps(config)
        config_path = "/tmp/livox_mid360_config.json"
        with open(config_path, "w") as f:
            f.write(config_str)

        # Define the point cloud callback type
        # void (*)(uint32_t handle, const uint8_t dev_type,
        #          LivoxLidarEthernetPacket *data, void *client_data)
        self._POINT_CLOUD_CB = ctypes.CFUNCTYPE(
            None, ctypes.c_uint32, ctypes.c_uint8,
            ctypes.c_void_p, ctypes.c_void_p
        )
        self._cb = self._POINT_CLOUD_CB(self._point_cloud_callback)

        # Init SDK
        ret = self._lib.LivoxLidarSdkInit(config_path.encode("utf-8"))
        if ret != 0:
            raise RuntimeError(f"LivoxLidarSdkInit failed with code {ret}")

        # Set point cloud callback
        self._lib.SetLivoxLidarPointCloudCallBack(self._cb, None)

        # Start SDK
        ret = self._lib.LivoxLidarSdkStart()
        if ret != 0:
            raise RuntimeError(f"LivoxLidarSdkStart failed with code {ret}")

        print(f"[MID360] SDK initialized, listening on {self.host_ip}:{self.data_port}")

    def _point_cloud_callback(self, handle, dev_type, data_ptr, client_data):
        """SDK2 point cloud callback — accumulates points into a frame."""
        if data_ptr is None:
            return

        try:
            # Parse the Livox SDK2 packet header to get point count and data
            # LivoxLidarEthernetPacket layout (simplified):
            #   uint8_t version, uint8_t slot, ...
            #   uint16_t data_num (number of points)
            #   ... followed by point data
            # Each point (Cartesian): int32 x, int32 y, int32 z, uint8 reflectivity, uint8 tag
            # x/y/z are in mm

            raw = ctypes.cast(data_ptr, ctypes.POINTER(ctypes.c_uint8))

            # Offset to data_num (bytes 8-9 in typical SDK2 packet)
            data_num = int.from_bytes(bytes([raw[8], raw[9]]), byteorder='little')

            if data_num == 0:
                return

            # Point data starts at offset ~18 for Cartesian mode
            point_offset = 18
            point_size = 14  # 4+4+4+1+1 = 14 bytes per point (Cartesian)

            points = np.zeros((data_num, 4), dtype=np.float32)
            for i in range(data_num):
                base = point_offset + i * point_size
                x = int.from_bytes(bytes([raw[base], raw[base+1], raw[base+2], raw[base+3]]),
                                   byteorder='little', signed=True)
                y = int.from_bytes(bytes([raw[base+4], raw[base+5], raw[base+6], raw[base+7]]),
                                   byteorder='little', signed=True)
                z = int.from_bytes(bytes([raw[base+8], raw[base+9], raw[base+10], raw[base+11]]),
                                   byteorder='little', signed=True)
                reflectivity = raw[base + 12]

                points[i] = [x / 1000.0, y / 1000.0, z / 1000.0, reflectivity / 255.0]

            with self._lock:
                self._accumulating = np.vstack([self._accumulating, points])

                # MID-360 rotates at ~10 Hz — accumulate roughly 100ms of data per frame
                if len(self._accumulating) >= self.max_points:
                    self._latest_cloud = self._accumulating[:self.max_points].copy()
                    self._accumulating = np.zeros((0, 4), dtype=np.float32)
                    self._frame_ready.set()

        except Exception:
            pass  # Silently skip malformed packets

    def get_cloud(self, timeout=1.0):
        """Block until a frame is ready, return (N, 4) float32 array."""
        if self._frame_ready.wait(timeout=timeout):
            self._frame_ready.clear()
            with self._lock:
                return self._latest_cloud.copy()
        return None

    def close(self):
        """Shutdown the SDK."""
        try:
            self._lib.LivoxLidarSdkUninit()
        except Exception:
            pass


class SyntheticMID360:
    """Generate synthetic point cloud data for testing the pipeline."""

    def __init__(self, max_points=10000, fps=10):
        self.max_points = max_points
        self.fps = fps
        self._frame_interval = 1.0 / fps
        self._angle = 0.0

    def get_cloud(self, timeout=1.0):
        """Return a synthetic point cloud frame."""
        time.sleep(self._frame_interval)

        N = self.max_points
        # Simulate a rotating scan pattern
        self._angle += 0.1
        theta = np.linspace(0, 2 * np.pi, N) + self._angle
        r = np.random.uniform(1.0, 15.0, N)

        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.random.uniform(-0.5, 2.5, N)
        intensity = np.random.uniform(0.0, 1.0, N)

        cloud = np.stack([x, y, z, intensity], axis=1).astype(np.float32)
        return cloud

    def close(self):
        pass


def main():
    parser = argparse.ArgumentParser(description="Livox MID-360 ZMQ streamer")
    parser.add_argument("--port", type=int, default=5557, help="ZMQ PUB port")
    parser.add_argument("--max-points", type=int, default=10000,
                        help="Max points per frame")
    parser.add_argument("--fps", type=int, default=10,
                        help="Target publishing rate (Hz)")
    parser.add_argument("--lidar-ip", type=str, default="192.168.1.100",
                        help="MID-360 IP address")
    parser.add_argument("--host-ip", type=str, default="192.168.1.50",
                        help="Host IP on the LiDAR network")
    parser.add_argument("--sdk-path", type=str, default=None,
                        help="Path to liblivox_lidar_sdk_shared.so")
    parser.add_argument("--test", action="store_true",
                        help="Use synthetic data (no hardware needed)")
    args = parser.parse_args()

    # ---- ZMQ setup ----
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://*:{args.port}")
    print(f"[MID360] ZMQ PUB bound on port {args.port}")

    # ---- LiDAR setup ----
    if args.test:
        print("[MID360] Running in TEST mode with synthetic data")
        lidar = SyntheticMID360(max_points=args.max_points, fps=args.fps)
    else:
        lidar = LivoxMID360(
            lidar_ip=args.lidar_ip,
            host_ip=args.host_ip,
            sdk_path=args.sdk_path,
            max_points=args.max_points,
        )

    # Graceful shutdown
    running = True

    def _sigint_handler(_sig, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _sigint_handler)

    frame_count = 0
    t_start = time.time()
    target_dt = 1.0 / args.fps

    try:
        while running:
            t_frame_start = time.time()

            cloud = lidar.get_cloud(timeout=1.0)
            if cloud is None:
                continue

            num_points = cloud.shape[0]
            point_bytes = cloud.tobytes()

            # Build message: 8-byte header + payload
            header = struct.pack("ii", num_points, len(point_bytes))
            message = header + point_bytes

            sock.send(message, zmq.NOBLOCK)

            frame_count += 1
            if frame_count % 10 == 0:
                elapsed = time.time() - t_start
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"[MID360] Frames: {frame_count}, FPS: {fps:.1f}, "
                      f"Points: {num_points}, Size: {len(point_bytes)}B")

            # Rate limiting
            elapsed = time.time() - t_frame_start
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

    finally:
        lidar.close()
        sock.close()
        ctx.term()
        print(f"[MID360] Shutdown complete. Total frames: {frame_count}")


if __name__ == "__main__":
    main()
