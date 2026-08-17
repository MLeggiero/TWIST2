#!/bin/bash
# Data recording with the internal RealSense D435i (RGB + aligned depth).
# Optionally also merges the two Arducam wrist cameras into the same episode
# — uncomment the --wrist_left_port/--wrist_right_port lines below once
# both wrist streamers are running (start_wrist_left.sh/start_wrist_right.sh
# on the Orin).
# Usage: bash data_record_d435.sh

source ~/miniconda3/bin/activate twist2

cd deploy_real

robot_ip="192.168.123.164"
data_frequency=60

python server_data_record_d435.py \
    --frequency ${data_frequency} \
    --robot_ip ${robot_ip} \
    --goal "pick up the red cup" \
    --desc "A humanoid robot picks up a red cup from the table." \
    --steps "step1: approach table. step2: grasp cup. step3: lift cup."
    # --wrist_left_port 5556 \
    # --wrist_right_port 5558
