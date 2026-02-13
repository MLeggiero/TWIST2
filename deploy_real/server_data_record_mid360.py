#!/usr/bin/env python3

"""
Data collection script for the Livox MID-360 LiDAR.

Collects point cloud data from the MID-360 (via ZMQ), plus body/hand
state and action from Redis, and writes episodes to disk.

Mirrors server_data_record_d435.py but for LiDAR point clouds.
The BEV (bird's-eye view) image is displayed for monitoring.
"""

import argparse
import json
import os
import time

import cv2
import numpy as np
import redis
from datetime import datetime
from multiprocessing import shared_memory
import threading

from data_utils.episode_writer import EpisodeWriter
from data_utils.lidar_client import LidarClient
from rich import print
from robot_control.speaker import Speaker


def main(args):
    # ---- Redis connection ----
    try:
        redis_pool = redis.ConnectionPool(
            host="localhost",
            port=6379,
            db=0,
            max_connections=10,
            retry_on_timeout=True,
            socket_timeout=0.1,
            socket_connect_timeout=0.1,
        )
        redis_client = redis.Redis(connection_pool=redis_pool)
        redis_pipeline = redis_client.pipeline()
        redis_client.ping()
        print(f"Connected to Redis at localhost:6379, DB=0 with connection pool")
    except Exception as e:
        print(f"Error connecting to Redis: {e}")
        return

    # ---- Shared memory for point cloud (max_points x 4, float32) ----
    cloud_max_points = args.max_points
    cloud_shape = (cloud_max_points, 4)
    cloud_shared_memory = shared_memory.SharedMemory(
        create=True, size=int(np.prod(cloud_shape) * np.float32().itemsize)
    )
    cloud_array = np.ndarray(cloud_shape, dtype=np.float32,
                             buffer=cloud_shared_memory.buf)

    # ---- Shared memory for BEV image ----
    bev_shape = (args.bev_size, args.bev_size, 3)
    bev_shared_memory = shared_memory.SharedMemory(
        create=True, size=int(np.prod(bev_shape) * np.uint8().itemsize)
    )
    bev_array = np.ndarray(bev_shape, dtype=np.uint8,
                           buffer=bev_shared_memory.buf)

    # ---- LiDAR client ----
    lidar_client = LidarClient(
        server_address=args.robot_ip,
        port=args.lidar_port,
        cloud_max_points=cloud_max_points,
        cloud_shm_name=cloud_shared_memory.name,
        bev_shape=bev_shape,
        bev_shm_name=bev_shared_memory.name,
        bev_show=False,  # We handle display in main loop
        unit_test=True,
    )
    lidar_thread = threading.Thread(target=lidar_client.receive_process, daemon=True)
    lidar_thread.start()

    # ---- Episode writer ----
    recording = False
    save_data_keys = ["pointcloud"]
    task_dir = os.path.join(args.data_folder, args.task_name)
    # Use a dummy image_shape since EpisodeWriter expects it
    recorder = EpisodeWriter(
        task_dir=task_dir,
        frequency=args.frequency,
        image_shape=(args.bev_size, args.bev_size, 3),
        data_keys=save_data_keys,
    )
    recorder.text_desc(goal=args.goal, desc=args.desc, steps=args.steps)

    control_dt = 1.0 / args.frequency
    step_count = 0
    frame_counter = 0
    running = True

    # LiDAR typically at 10 Hz, recording at 10 Hz → subsample=1
    subsample_interval = max(1, round(10.0 / args.frequency))

    print(f"Recorded control frequency: {args.frequency} Hz (subsample every {subsample_interval} frames)")

    speaker = Speaker()
    prev_button_pressed = False
    prev_right_axis_click_pressed = False

    try:
        while running:
            start_time = time.time()

            # ---- Controller input ----
            controller_data = json.loads(redis_client.get("controller_data"))
            button_pressed = controller_data["LeftController"]["key_two"]

            quit_key = controller_data["LeftController"]["axis_click"]
            if quit_key:
                running = False
                speaker.speak("Recording stopped.")
                print("\nQuitting...")
                break

            right_axis_click = controller_data["RightController"]["axis_click"]

            # Rising-edge toggle
            if button_pressed and not prev_button_pressed:
                print("button pressed")
                recording = not recording
                if recording:
                    speaker.speak("lidar episode recording started.")
                    if not recorder.create_episode():
                        recording = False
                    step_count = 0
                    frame_counter = 0
                    print("lidar episode recording started...")
                else:
                    recorder.save_episode(label="successful")
                    speaker.speak("lidar episode saved as successful.")

            # Right axis_click: save as unsuccessful
            if right_axis_click and not prev_right_axis_click_pressed and recording:
                recorder.save_episode(label="unsuccessful")
                recording = False
                speaker.speak("lidar episode saved as unsuccessful.")

            prev_button_pressed = button_pressed
            prev_right_axis_click_pressed = right_axis_click

            if recording:
                frame_counter += 1

                if frame_counter % subsample_interval != 0:
                    elapsed = time.time() - start_time
                    if elapsed < control_dt:
                        time.sleep(control_dt - elapsed)
                    continue

                data_dict = {"idx": step_count}

                # ---- LiDAR data ----
                # Read point cloud from shared memory
                n_pts = lidar_client.cloud_num_points
                if n_pts > 0:
                    data_dict["pointcloud"] = cloud_array[:n_pts].copy()
                else:
                    data_dict["pointcloud"] = None
                data_dict["t_lidar"] = int(time.time() * 1000)

                # ---- Redis state & action (same as D435 recorder) ----
                redis_keys = [
                    "state_body_unitree_g1_with_hands",
                    "state_hand_left_unitree_g1_with_hands",
                    "state_hand_right_unitree_g1_with_hands",
                    "state_neck_unitree_g1_with_hands",
                    "t_state",
                    "action_body_unitree_g1_with_hands",
                    "action_hand_left_unitree_g1_with_hands",
                    "action_hand_right_unitree_g1_with_hands",
                    "action_neck_unitree_g1_with_hands",
                    "t_action",
                    "action_low_level_unitree_g1_with_hands",
                ]

                data_dict_keys = [
                    "state_body",
                    "state_hand_left",
                    "state_hand_right",
                    "state_neck",
                    "t_state",
                    "action_body",
                    "action_hand_left",
                    "action_hand_right",
                    "action_neck",
                    "t_action",
                    "action_low_level",
                ]

                try:
                    for key in redis_keys:
                        redis_pipeline.get(key)
                    redis_results = redis_pipeline.execute()

                    for i, (result, dict_key) in enumerate(zip(redis_results, data_dict_keys)):
                        if result is not None:
                            try:
                                data_dict[dict_key] = json.loads(result)
                            except json.JSONDecodeError:
                                print(f"Warning: Failed to decode JSON for key {redis_keys[i]}")
                                data_dict[dict_key] = None
                        else:
                            print(f"Warning: No data found for key {redis_keys[i]}")
                            data_dict[dict_key] = None
                except Exception as e:
                    print(f"Error in Redis pipeline operation: {e}")
                    continue

                recorder.add_item(data_dict)

                # Display BEV
                if bev_array is not None and bev_array.size > 0:
                    window_name = "MID360 BEV - Press controller button to start/stop recording"
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, args.bev_size, args.bev_size)
                    cv2.moveWindow(window_name, 550, 50)
                    cv2.imshow(window_name, bev_array)
                    cv2.waitKey(1)

                step_count += 1
                elapsed = time.time() - start_time
                if elapsed < control_dt:
                    time.sleep(control_dt - elapsed)
            else:
                # Display BEV even when not recording
                if bev_array is not None and bev_array.size > 0:
                    window_name = "MID360 BEV - Press controller button to start/stop recording"
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, args.bev_size, args.bev_size)
                    cv2.moveWindow(window_name, 550, 50)
                    cv2.imshow(window_name, bev_array)
                    cv2.waitKey(1)
                else:
                    time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nReceived Ctrl+C, exiting...")
        running = False
    finally:
        print(f"\nDone! Recorded {recorder.episode_id + 1} episodes to {task_dir}")

        cloud_shared_memory.unlink()
        cloud_shared_memory.close()
        bev_shared_memory.unlink()
        bev_shared_memory.close()
        recorder.close()
        cv2.destroyAllWindows()

        print("Exiting the recording...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MID-360 LiDAR data recording.")
    cur_time = datetime.now().strftime("%Y%m%d_%H%M")

    parser.add_argument("--data_folder", default="twist2_demonstration", help="Data folder")
    parser.add_argument("--task_name", default=f"{cur_time}", help="Task name")
    parser.add_argument("--frequency", default=10, type=int, help="Recording frequency in Hz")
    parser.add_argument("--robot", default="unitree_g1", choices=["unitree_g1"], help="Robot name")
    parser.add_argument("--robot_ip", default="192.168.123.164", help="Robot / Orin IP")
    parser.add_argument("--lidar_port", default=5557, type=int, help="ZMQ port for MID360 streamer")
    parser.add_argument("--max_points", default=10000, type=int, help="Max points per frame")
    parser.add_argument("--bev_size", default=400, type=int, help="BEV image size in pixels")

    # Task description metadata
    parser.add_argument("--goal", default="pick up the red cup", help="Task goal description")
    parser.add_argument("--desc", default="A humanoid robot picks up a red cup from the table.", help="Task description")
    parser.add_argument("--steps", default="step1: approach table. step2: grasp cup. step3: lift cup.", help="Task steps")

    args = parser.parse_args()
    main(args)
