#!/bin/bash
# Deploy RealSense + MID-360 streamer files to the G1 Orin.
# Usage: bash deploy_real/onboard/deploy_to_robot.sh
#
# The Jetson has no internet access on the robot's internal network,
# so we download wheels on the workstation and transfer them over.
# pyrealsense2 must be installed via apt (no aarch64 PyPI wheel exists).

set -e

ROBOT_USER="unitree"
ROBOT_IP="192.168.123.164"
REMOTE_DIR="~/g1-onboard"
PLATFORM="manylinux2014_aarch64"
PYTHON_VER="3.10"  # JetPack 6.2 / Ubuntu 22.04 ships Python 3.10

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHEEL_DIR="${SCRIPT_DIR}/.wheels"

# --- Step 1: Download wheels on workstation (has internet) ---
echo "==> Downloading wheels for ${PLATFORM} / ${PYTHON_VER}..."
mkdir -p "${WHEEL_DIR}"
pip download \
    --dest "${WHEEL_DIR}" \
    --platform "${PLATFORM}" \
    --python-version "${PYTHON_VER}" \
    --only-binary=:all: \
    -r "${SCRIPT_DIR}/requirements.txt" 2>&1 | tail -5

# --- Step 2: Copy files + wheels to robot ---
echo "==> Creating remote directory and copying files to ${ROBOT_USER}@${ROBOT_IP}:${REMOTE_DIR}/..."
ssh "${ROBOT_USER}@${ROBOT_IP}" "mkdir -p ${REMOTE_DIR}/wheels"
scp "${SCRIPT_DIR}/realsense_streamer.py" \
    "${SCRIPT_DIR}/start_realsense.sh" \
    "${SCRIPT_DIR}/mid360_streamer.py" \
    "${SCRIPT_DIR}/start_mid360.sh" \
    "${SCRIPT_DIR}/requirements.txt" \
    "${SCRIPT_DIR}/build_pyrealsense2.sh" \
    "${ROBOT_USER}@${ROBOT_IP}:${REMOTE_DIR}/"
if ls "${WHEEL_DIR}"/*.whl 1>/dev/null 2>&1; then
    scp "${WHEEL_DIR}"/*.whl "${ROBOT_USER}@${ROBOT_IP}:${REMOTE_DIR}/wheels/"
else
    echo "    WARNING: No wheels found in ${WHEEL_DIR}. Downloading failed or was skipped."
fi

# --- Step 3: Install from local wheels (no internet needed) ---
echo "==> Installing Python dependencies on robot from local wheels..."
ssh "${ROBOT_USER}@${ROBOT_IP}" "pip install --no-index --find-links ${REMOTE_DIR}/wheels -r ${REMOTE_DIR}/requirements.txt"

# --- Step 4: Install librealsense2 .deb packages (offline) ---
DEB_DIR="${SCRIPT_DIR}/.realsense_debs"
if [ -d "${DEB_DIR}" ] && ls "${DEB_DIR}"/*.deb 1>/dev/null 2>&1; then
    echo "==> Transferring librealsense2 .deb packages to robot..."
    ssh "${ROBOT_USER}@${ROBOT_IP}" "mkdir -p ${REMOTE_DIR}/debs"
    scp "${DEB_DIR}"/*.deb "${ROBOT_USER}@${ROBOT_IP}:${REMOTE_DIR}/debs/"
    echo "==> Installing librealsense2 packages on robot..."
    ssh "${ROBOT_USER}@${ROBOT_IP}" "sudo dpkg -i ${REMOTE_DIR}/debs/*.deb 2>&1 || sudo apt-get install -f -y 2>&1"
else
    echo "==> No librealsense2 .deb packages found in ${DEB_DIR}."
    echo "    Run fetch_realsense_debs.sh first to download them, or install manually."
fi

# --- Step 5: Verify pyrealsense2 ---
echo "==> Checking pyrealsense2 on robot..."
ssh "${ROBOT_USER}@${ROBOT_IP}" "python3 -c 'import pyrealsense2; print(\"pyrealsense2 version:\", pyrealsense2.__version__)'" 2>/dev/null && {
    echo "    pyrealsense2 OK."
} || {
    echo "    WARNING: pyrealsense2 is NOT importable."
    echo "    Connect the Jetson to WiFi, then run on the Jetson:"
    echo "      bash ~/g1-onboard/build_pyrealsense2.sh"
}

echo "==> Deploy complete."
