#!/usr/bin/env python3

"""
Data collection script for the internal RealSense D435i camera.

Collects RGB + aligned depth from the D435i (via ZMQ), plus body/hand
state and action from Redis, and writes episodes to disk.

Differences from server_data_record.py (ZED):
  - Single 424x240 RGB image (not stereo 1280x360)
  - Aligned uint16 depth at 424x240
  - Default 60 Hz recording frequency (configurable via --frequency)
  - Task description via CLI args (--goal, --desc, --steps)
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
from data_utils.vision_client import VisionClient
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

    # ---- Shared memory for RGB (424x240x3) ----
    image_shape = (args.height, args.width, 3)
    image_shared_memory = shared_memory.SharedMemory(
        create=True, size=int(np.prod(image_shape) * np.uint8().itemsize)
    )
    image_array = np.ndarray(image_shape, dtype=np.uint8, buffer=image_shared_memory.buf)

    # ---- Shared memory for depth (424x240, float32) ----
    depth_shape = (args.height, args.width)
    depth_shared_memory = shared_memory.SharedMemory(
        create=True, size=int(np.prod(depth_shape) * np.float32().itemsize)
    )
    depth_array = np.ndarray(depth_shape, dtype=np.float32, buffer=depth_shared_memory.buf)

    # ---- Vision client ----
    image_show = True
    vision_client = VisionClient(
        server_address=args.robot_ip,
        port=args.camera_port,
        img_shape=image_shape,
        img_shm_name=image_shared_memory.name,
        depth_shape=depth_shape,
        depth_shm_name=depth_shared_memory.name,
        image_show=False,
        depth_show=False,
        unit_test=True,
    )
    vision_thread = threading.Thread(target=vision_client.receive_process, daemon=True)
    vision_thread.start()

    # ---- Episode writer ----
    recording = False
    save_data_keys = ["rgb", "depth"]
    task_dir = os.path.join(args.data_folder, args.task_name)
    recorder = EpisodeWriter(
        task_dir=task_dir,
        frequency=args.frequency,
        image_shape=image_shape,
        data_keys=save_data_keys,
    )
    recorder.text_desc(goal=args.goal, desc=args.desc, steps=args.steps)

    control_dt = 1.0 / args.frequency
    step_count = 0
    frame_counter = 0  # for sub-sampling
    running = True

    # Sub-sampling: how many camera frames to skip between recorded frames.
    # e.g. camera at 60fps, --frequency 30 → record every 2nd frame.
    subsample_interval = max(1, round(60.0 / args.frequency))

    print(f"Recorded control frequency: {args.frequency} Hz (subsample every {subsample_interval} frames)")

    speaker = Speaker()
    prev_button_pressed = False

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

            # Rising-edge toggle
            if button_pressed and not prev_button_pressed:
                print("button pressed")
                recording = not recording
                if recording:
                    speaker.speak("episode recording started.")
                    if not recorder.create_episode():
                        recording = False
                    step_count = 0
                    frame_counter = 0
                    print("episode recording started...")
                else:
                    recorder.save_episode()
                    speaker.speak("episode saved.")

            prev_button_pressed = button_pressed

            if recording:
                frame_counter += 1

                # Sub-sample: only record on the correct cadence
                if frame_counter % subsample_interval != 0:
                    elapsed = time.time() - start_time
                    if elapsed < control_dt:
                        time.sleep(control_dt - elapsed)
                    continue

                data_dict = {"idx": step_count}

                # ---- Vision data ----
                data_dict["rgb"] = image_array.copy()
                data_dict["t_img"] = int(time.time() * 1000)

                # Depth: read shared memory and convert float32 → uint16 (mm)
                raw_depth = depth_array.copy()
                if np.any(raw_depth > 0):
                    data_dict["depth"] = raw_depth.astype(np.uint16)
                else:
                    data_dict["depth"] = None

                # ---- Redis state & action ----
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

                if image_show:
                    if image_array is not None and image_array.size > 0:
                        window_name = "D435i - Press controller button to start/stop recording"
                        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(window_name, image_array.shape[1], image_array.shape[0])
                        cv2.moveWindow(window_name, 50, 50)
                        cv2.imshow(window_name, image_array)
                        cv2.waitKey(1)

                step_count += 1
                elapsed = time.time() - start_time
                if elapsed < control_dt:
                    time.sleep(control_dt - elapsed)
            else:
                if image_show:
                    if image_array is not None and image_array.size > 0:
                        window_name = "D435i - Press controller button to start/stop recording"
                        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(window_name, image_array.shape[1], image_array.shape[0])
                        cv2.moveWindow(window_name, 50, 50)
                        cv2.imshow(window_name, image_array)
                        cv2.waitKey(1)
                else:
                    time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nReceived Ctrl+C, exiting...")
        running = False
    finally:
        print(f"\nDone! Recorded {recorder.episode_id + 1} episodes to {task_dir}")

        image_shared_memory.unlink()
        image_shared_memory.close()
        depth_shared_memory.unlink()
        depth_shared_memory.close()
        recorder.close()
        cv2.destroyAllWindows()

        print("Exiting the recording...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="D435i data recording with RGB + depth.")
    cur_time = datetime.now().strftime("%Y%m%d_%H%M")

    parser.add_argument("--data_folder", default="twist2_demonstration", help="Data folder")
    parser.add_argument("--task_name", default=f"{cur_time}", help="Task name")
    parser.add_argument("--frequency", default=60, type=int, help="Recording frequency in Hz")
    parser.add_argument("--robot", default="unitree_g1", choices=["unitree_g1"], help="Robot name")
    parser.add_argument("--robot_ip", default="192.168.123.164", help="Robot / Orin IP")
    parser.add_argument("--camera_port", default=5555, type=int, help="ZMQ port for D435i streamer")
    parser.add_argument("--width", default=424, type=int, help="Image width")
    parser.add_argument("--height", default=240, type=int, help="Image height")

    # Task description metadata
    parser.add_argument("--goal", default="pick up the red cup", help="Task goal description")
    parser.add_argument("--desc", default="A humanoid robot picks up a red cup from the table.", help="Task description")
    parser.add_argument("--steps", default="step1: approach table. step2: grasp cup. step3: lift cup.", help="Task steps")

    args = parser.parse_args()
    main(args)
