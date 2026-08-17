#!/bin/bash
# Start the left-wrist Arducam fisheye streamer on the G1 Orin.
# Usage: bash start_wrist_left.sh
#
# For testing without hardware:
#   python3 ~/g1-onboard/wrist_streamer.py --port 5556 --test

python3 ~/g1-onboard/wrist_streamer.py --device /dev/video_wrist_left --port 5556 --width 640 --height 480 --fps 30
