#!/bin/bash
# Start the Livox MID-360 LiDAR streamer on the G1 Orin.
# Usage: bash start_mid360.sh
#
# For testing without hardware:
#   python3 ~/g1-onboard/mid360_streamer.py --port 5557 --test

python3 ~/g1-onboard/mid360_streamer.py --port 5557 --fps 10 --max-points 10000
