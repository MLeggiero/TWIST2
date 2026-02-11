#!/bin/bash
# Start the RealSense D435i streamer on the G1 Orin.
# Usage: bash start_realsense.sh

python3 ~/g1-onboard/realsense_streamer.py --port 5555 --fps 60 --width 424 --height 240
