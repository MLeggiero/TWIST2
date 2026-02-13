#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZeroMQ subscriber client for receiving point cloud data from the MID-360
LiDAR streamer and rendering a bird's-eye view (BEV) image.

Mirrors the VisionClient pattern used for the RealSense D435i.
"""

import struct
import threading
import time
from collections import deque
from multiprocessing import shared_memory

import cv2
import numpy as np
import zmq
from rich import print


class LidarClient:
    """
    ZeroMQ subscriber client for the MID-360 LiDAR streamer.

    Receives point cloud frames, renders a BEV image, and optionally
    writes the point cloud and BEV image into shared memory buffers.
    """

    def __init__(
        self,
        server_address="127.0.0.1",
        port=5557,
        # Shared memory for point cloud (N, 4) float32
        cloud_max_points=10000,
        cloud_shm_name=None,
        # Shared memory for BEV image
        bev_shape=None,
        bev_shm_name=None,
        # BEV rendering parameters
        bev_size=400,
        bev_range=15.0,
        # Display
        bev_show=False,
        unit_test=False,
    ):
        self.server_address = server_address
        self.port = port
        self.running = True
        self.bev_show = bev_show

        # BEV rendering config
        self.bev_size = bev_size
        self.bev_range = bev_range  # meters from center in each direction

        # Shared memory for point cloud
        self.cloud_max_points = cloud_max_points
        self.cloud_shm_name = cloud_shm_name
        self.cloud_shm_enabled = False
        self.cloud_num_points = 0  # actual number of points in current frame
        if cloud_shm_name is not None:
            cloud_shape = (cloud_max_points, 4)
            self.cloud_shm = shared_memory.SharedMemory(name=cloud_shm_name)
            self.cloud_array = np.ndarray(cloud_shape, dtype=np.float32,
                                          buffer=self.cloud_shm.buf)
            self.cloud_shm_enabled = True

        # Shared memory for BEV image
        self.bev_shape = bev_shape
        self.bev_shm_name = bev_shm_name
        self.bev_shm_enabled = False
        if bev_shape is not None and bev_shm_name is not None:
            self.bev_shm = shared_memory.SharedMemory(name=bev_shm_name)
            self.bev_array = np.ndarray(bev_shape, dtype=np.uint8,
                                        buffer=self.bev_shm.buf)
            self.bev_shm_enabled = True

        # Performance metrics
        self.unit_test = unit_test
        if self.unit_test:
            self._init_performance_metrics()

    def _init_performance_metrics(self):
        self.frame_count = 0
        self.start_time = time.time()
        self.time_window = 1.0
        self.frame_times = deque()

    def _update_performance_metrics(self, print_info, verbose=False):
        if not self.unit_test:
            return
        now = time.time()
        self.frame_times.append(now)
        while self.frame_times and self.frame_times[0] < now - self.time_window:
            self.frame_times.popleft()
        self.frame_count += 1
        if self.frame_count % 10 == 0:
            real_time_fps = len(self.frame_times) / self.time_window
            if verbose:
                print(f"[LidarClient] FPS: {real_time_fps:.2f}, {print_info}")

    def render_bev(self, cloud):
        """
        Render a bird's-eye view (BEV) image from a point cloud.

        Args:
            cloud: (N, 4) float32 array [x, y, z, intensity]

        Returns:
            BEV image as (H, W, 3) uint8 BGR array.
        """
        bev = np.zeros((self.bev_size, self.bev_size, 3), dtype=np.uint8)

        if cloud is None or len(cloud) == 0:
            return bev

        x = cloud[:, 0]
        y = cloud[:, 1]
        z = cloud[:, 2]
        intensity = cloud[:, 3]

        # Filter to BEV range
        mask = (np.abs(x) < self.bev_range) & (np.abs(y) < self.bev_range)
        x = x[mask]
        y = y[mask]
        z = z[mask]
        intensity = intensity[mask]

        if len(x) == 0:
            return bev

        # Map world coordinates to pixel coordinates
        # x -> right (columns), y -> forward (rows, inverted)
        scale = self.bev_size / (2 * self.bev_range)
        px = ((x + self.bev_range) * scale).astype(np.int32)
        py = ((self.bev_range - y) * scale).astype(np.int32)  # flip y for display

        # Clip to image bounds
        valid = (px >= 0) & (px < self.bev_size) & (py >= 0) & (py < self.bev_size)
        px = px[valid]
        py = py[valid]
        z = z[valid]
        intensity = intensity[valid]

        if len(px) == 0:
            return bev

        # Color by height (z): blue (low) -> green -> red (high)
        z_min, z_max = -0.5, 2.5
        z_norm = np.clip((z - z_min) / (z_max - z_min + 1e-6), 0.0, 1.0)
        z_uint8 = (z_norm * 255).astype(np.uint8)

        # Apply JET colormap
        color_lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(1, -1),
                                       cv2.COLORMAP_JET)[0]  # (256, 3)
        colors = color_lut[z_uint8]  # (N, 3)

        # Draw points
        bev[py, px] = colors

        # Draw center crosshair
        center = self.bev_size // 2
        cv2.drawMarker(bev, (center, center), (255, 255, 255),
                        cv2.MARKER_CROSS, 10, 1)

        # Draw range rings every 5m
        for r_m in range(5, int(self.bev_range) + 1, 5):
            r_px = int(r_m * scale)
            cv2.circle(bev, (center, center), r_px, (60, 60, 60), 1)
            cv2.putText(bev, f"{r_m}m", (center + r_px + 2, center),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)

        return bev

    def handle_cloud(self, cloud):
        """Handle a received point cloud frame."""
        if cloud is None:
            return

        # Copy to shared memory (pad/truncate to max_points)
        if self.cloud_shm_enabled:
            n = min(cloud.shape[0], self.cloud_max_points)
            self.cloud_array[:n] = cloud[:n]
            if n < self.cloud_max_points:
                self.cloud_array[n:] = 0
            self.cloud_num_points = n

        # Render and display BEV
        bev = self.render_bev(cloud)

        if self.bev_shm_enabled and self.bev_shape is not None:
            if bev.shape == self.bev_shape:
                np.copyto(self.bev_array, bev)

        if self.bev_show:
            cv2.imshow("LidarClient - BEV", bev)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False

    def _close(self):
        self.socket.close()
        self.context.term()
        if self.bev_show:
            cv2.destroyAllWindows()
        print("[LidarClient] Closed.")

    def receive_process(self):
        """Main loop: subscribe to ZMQ, decode point clouds, render BEV."""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://{self.server_address}:{self.port}")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")

        print(f"[LidarClient] Subscribed to tcp://{self.server_address}:{self.port}. Waiting for data...")

        try:
            while self.running:
                try:
                    message = self.socket.recv(zmq.NOBLOCK)

                    # 8-byte header: [num_points][data_len]
                    if len(message) < 8:
                        continue

                    num_points = struct.unpack('i', message[0:4])[0]
                    data_len = struct.unpack('i', message[4:8])[0]

                    actual_payload = len(message) - 8
                    if actual_payload != data_len:
                        continue

                    if num_points <= 0 or data_len <= 0:
                        continue

                    # Decode point cloud
                    cloud = np.frombuffer(
                        message[8:], dtype=np.float32
                    ).reshape(num_points, 4)

                    self.handle_cloud(cloud)

                    print_info = f"points: {num_points}, size: {data_len}B"
                    self._update_performance_metrics(print_info)

                except zmq.Again:
                    time.sleep(0.001)
                    continue

        except KeyboardInterrupt:
            print("[LidarClient] Interrupted by user.")
        except Exception as e:
            print(f"[LidarClient] Error: {e}")
        finally:
            self._close()


if __name__ == "__main__":
    # Standalone test — connects to MID360 streamer and displays BEV
    bev_shape = (400, 400, 3)
    bev_shm = shared_memory.SharedMemory(
        create=True, size=int(np.prod(bev_shape) * np.uint8().itemsize)
    )

    cloud_max_points = 10000
    cloud_shape = (cloud_max_points, 4)
    cloud_shm = shared_memory.SharedMemory(
        create=True, size=int(np.prod(cloud_shape) * np.float32().itemsize)
    )

    client = LidarClient(
        server_address="192.168.123.164",
        port=5557,
        cloud_max_points=cloud_max_points,
        cloud_shm_name=cloud_shm.name,
        bev_shape=bev_shape,
        bev_shm_name=bev_shm.name,
        bev_show=True,
        unit_test=True,
    )

    lidar_thread = threading.Thread(target=client.receive_process, daemon=True)
    lidar_thread.start()

    try:
        while True:
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("[LidarClient] Interrupted by user.")
    finally:
        bev_shm.unlink()
        bev_shm.close()
        cloud_shm.unlink()
        cloud_shm.close()
