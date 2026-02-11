#!/bin/bash
source ~/miniconda3/bin/activate diffplan
cd deploy_real
python convert_twist2_to_lerobot.py \
    --data_dir twist2_demonstration/20260210_1017 \
    --output_dir /path/to/output_lowlevel \
    --repo_id "username/twist2_demos_lowlevel" \
    --fps 60 --action_mode low_level --use_videos
