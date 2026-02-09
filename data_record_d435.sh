#!/bin/bash
# Data recording with the internal RealSense D435i (RGB + aligned depth).
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
