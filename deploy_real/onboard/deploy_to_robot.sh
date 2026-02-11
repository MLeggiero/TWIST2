#!/bin/bash
# Deploy RealSense streamer files to the G1 Orin.
# Usage: bash deploy_real/onboard/deploy_to_robot.sh

set -e

ROBOT_USER="unitree"
ROBOT_IP="192.168.123.164"
REMOTE_DIR="~/g1-onboard"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Creating remote directory and copying files to ${ROBOT_USER}@${ROBOT_IP}:${REMOTE_DIR}/..."
ssh "${ROBOT_USER}@${ROBOT_IP}" "mkdir -p ${REMOTE_DIR}"
scp "${SCRIPT_DIR}/realsense_streamer.py" \
    "${SCRIPT_DIR}/start_realsense.sh" \
    "${SCRIPT_DIR}/requirements.txt" \
    "${ROBOT_USER}@${ROBOT_IP}:${REMOTE_DIR}/"

echo "==> Installing Python dependencies on robot..."
ssh "${ROBOT_USER}@${ROBOT_IP}" "pip install -r ${REMOTE_DIR}/requirements.txt"

echo "==> Deploy complete."
