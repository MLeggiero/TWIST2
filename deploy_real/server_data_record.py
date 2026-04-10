#!/usr/bin/env python3

"""
Data collection script.
Collect data from Redis and save to a file.
The data includes:
- vision data
- body and hand state
- body and hand action


"""

import argparse
import os
import json
import time
import redis
import cv2
import numpy as np
from multiprocessing import shared_memory, Array, Lock
import threading
from data_utils.episode_writer import EpisodeWriter
from data_utils.vision_client import VisionClient
from rich import print
from robot_control.speaker import Speaker
from datetime import datetime

def main(args):

    # Connect to Redis with connection pool for better performance
    try:
        redis_pool = redis.ConnectionPool(
            host="localhost", 
            port=6379, 
            db=0,
            max_connections=10,
            retry_on_timeout=True,
            socket_timeout=0.1,
            socket_connect_timeout=0.1
        )
        redis_client = redis.Redis(connection_pool=redis_pool)
        redis_pipeline = redis_client.pipeline()
        # Test connection
        redis_client.ping()
        print(f"Connected to Redis at localhost:6379, DB=0 with connection pool")
    except Exception as e:
        print(f"Error connecting to Redis: {e}")
        return

    # Initialize OpenCV window
    # Create shared memory for single camera - 640x480 image
    num_cameras = 2
    image_shape = (360, 640*num_cameras, 3)  # Height, Width, Channels for OpenCV format
    image_shared_memory = shared_memory.SharedMemory(create=True, size=int(np.prod(image_shape) * np.uint8().itemsize * num_cameras))
    image_array = np.ndarray(image_shape, dtype=np.uint8, buffer=image_shared_memory.buf)


    # Display settings for single camera
    image_show = True

    vision_client = VisionClient(
        server_address=args.robot_ip,  # robot IP
        port=5555,
        img_shape=image_shape,
        img_shm_name=image_shared_memory.name,
        image_show=False,
        depth_show=False,
        unit_test=True
    )
    vision_thread = threading.Thread(target=vision_client.receive_process, daemon=True)
    vision_thread.daemon = True
    vision_thread.start()
    
    # create recorder
    recording = False
    save_data_keys = ['rgb']
    task_dir = os.path.join(args.data_folder, args.task_name)
    recorder = EpisodeWriter(task_dir = task_dir, frequency = args.frequency,
                             image_shape=image_shape,
                             data_keys=save_data_keys)
    recorder.text_desc(goal="walk ahead and pick a box.",
                       desc="a humanoid robot walk head and pick a box from the table.",
                       steps="step1: walk ahead 1 meter. step2: pick a box from the table.")
    
    control_dt = 1 / args.frequency
    step_count = 0
    running = True
    
    print("Recorded control frequency: ", args.frequency)
    
   
    speaker = Speaker()
    
    # Initialize button state tracking
    prev_button_pressed = False
    
    try:
        while running:

            start_time = time.time()
            
            # handle controller input
            controller_data = json.loads(redis_client.get(f"controller_data"))
            button_pressed = controller_data['LeftController']['key_two']
            # print(f"==> button_pressed: {button_pressed}", end="\r")
            
            quit_key = controller_data['LeftController']['axis_click']
            if quit_key:
                running = False
                speaker.speak("Recording stopped.")
                print("\nQuitting...")
                break
            
            # Detect button press (rising edge detection)
            if button_pressed and not prev_button_pressed:
                print("button pressed")
                recording = not recording
                if recording:
                    speaker.speak("episode recording started.")
                    if not recorder.create_episode():
                        recording = False
                    step_count = 0
                    print("episode recording started...")
                else:
                    recorder.save_episode()
                    speaker.speak("episode saved.")
            
            # Update previous button state
            prev_button_pressed = button_pressed
                
           
            if recording:
                # Attempt to retrieve "action_mimic" from Redis
                data_dict = {'idx': step_count}
                # receive vision data
                data_dict["rgb"] = image_array.copy()  # type: ignore
                data_dict["t_img"] = int(time.time() * 1000) # current timestamp in ms

                # Pipeline Redis operations for better performance
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

                    "force_hand_left_unitree_g1_with_hands",
                    "force_hand_right_unitree_g1_with_hands",

                    "tactile_hand_left_unitree_g1_with_hands",
                    "tactile_hand_right_unitree_g1_with_hands",
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

                    "force_hand_left",
                    "force_hand_right",

                    "tactile_hand_left",
                    "tactile_hand_right",
                ]
                
                try:
                    # Use Redis pipeline to batch all GET operations (1 network round-trip instead of 10)
                    for key in redis_keys:
                        redis_pipeline.get(key)
                    redis_results = redis_pipeline.execute()
                    
                    # Process results with error handling
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
                    # Fallback: skip this recording cycle
                    continue
                
                # write data to recorder
                recorder.add_item(data_dict)
                
                if image_show:
                    # Check if image array has valid data
                    if image_array is not None and image_array.size > 0:
                        # resize image for display (stereo image is already concatenated)
                        # image_display = cv2.resize(image_array, (image_array.shape[1]//2, image_array.shape[0]//2))
                        image_display = image_array
                        # Create window with size matching image
                        window_name = "Press controller button to start/stop recording"
                        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(window_name, image_display.shape[1], image_display.shape[0])
                        cv2.moveWindow(window_name, 50, 50)  # Position window on left side
                        cv2.imshow(window_name, image_display)
                        cv2.waitKey(1)
                
                step_count += 1
                elapsed = time.time() - start_time
                if elapsed < control_dt:
                    time.sleep(control_dt - elapsed)
            else:
                if image_show:
                    # Check if image array has valid data
                    if image_array is not None and image_array.size > 0:
                        # resize stereo image for display
                        # image_display = cv2.resize(image_array, (image_array.shape[1]//2, image_array.shape[0]//2))
                        image_display = image_array
                        # Create window with size matching image
                        window_name = "Press controller button to start/stop recording"
                        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(window_name, image_display.shape[1], image_display.shape[0])
                        cv2.moveWindow(window_name, 50, 50)  # Position window on left side
                        cv2.imshow(window_name, image_display)
                        cv2.waitKey(1)
                else:
                    # For keyboard mode, just sleep to avoid busy waiting
                    time.sleep(0.1)
                    
    except KeyboardInterrupt:
        print("\nReceived Ctrl+C, exiting...")
        running = False
    finally:
        print(f"\nDone! Recorded {recorder.episode_id + 1} episodes to {task_dir}")

        # unlink and release shared memory
        image_shared_memory.unlink()
        image_shared_memory.close()
        recorder.close()
        
        cv2.destroyAllWindows()  # Close OpenCV window
        
        print("Exiting the recording...")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Record 'mimic_obs' from Redis.")
    cur_time = datetime.now().strftime("%Y%m%d_%H%M")
    parser.add_argument("--data_folder", default=f"/home/ANT.AMAZON.COM/yanjieze/projects/TWIST2/TWIST2-clean/deploy_real/twist2_demonstration", help="data folder")
    parser.add_argument("--task_name", default=f"{cur_time}", help="task name")
    parser.add_argument("--frequency", default=30, type=int)
    parser.add_argument("--robot", default="unitree_g1", choices=["unitree_g1"], help="robot name")
    parser.add_argument("--robot_ip", default="192.168.123.164", help="robot ip")
    
    args = parser.parse_args()

    main(args)
