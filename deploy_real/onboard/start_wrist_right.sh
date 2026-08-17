#!/bin/bash
# Start the right-wrist Arducam fisheye streamer on the G1 Orin.
# Usage: bash start_wrist_right.sh
#
# For testing without hardware:
#   python3 ~/g1-onboard/wrist_streamer.py --port 5558 --test

python3 ~/g1-onboard/wrist_streamer.py --device /dev/video_wrist_right --port 5558 --width 640 --height 480 --fps 30
