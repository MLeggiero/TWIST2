#!/bin/bash
# Build and install pyrealsense2 Python bindings from source on the Jetson.
# Run this ON the Jetson (e.g., via SSH to unitree@192.168.123.164).
# Requires internet access (for apt and git clone).
#
# Usage: bash build_pyrealsense2.sh

set -e

RS_VERSION="v2.56.3"  # Known stable release for JetPack 6.x
BUILD_DIR="/tmp/librealsense-build"
NUM_JOBS=$(nproc)

echo "==> Installing build dependencies..."
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    git cmake build-essential pkg-config \
    libssl-dev libusb-1.0-0-dev libgtk-3-dev libglfw3-dev \
    python3-dev python3-pip

echo "==> Cloning librealsense ${RS_VERSION}..."
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
git clone --depth 1 --branch "${RS_VERSION}" https://github.com/IntelRealSense/librealsense.git
cd librealsense

echo "==> Building librealsense with Python bindings (${NUM_JOBS} jobs)..."
mkdir build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DPYTHON_EXECUTABLE=$(which python3) \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_GRAPHICAL_EXAMPLES=OFF \
    -DBUILD_WITH_CUDA=ON \
    -DFORCE_RSUSB_BACKEND=ON
make -j${NUM_JOBS}

echo "==> Installing..."
sudo make install
sudo ldconfig

# Install the Python binding to the user's site-packages
PYRS_SO=$(find . -name "pyrealsense2*.so" | head -1)
if [ -n "${PYRS_SO}" ]; then
    SITE_PACKAGES=$(python3 -c "import site; print(site.getusersitepackages())")
    mkdir -p "${SITE_PACKAGES}"
    cp -v "${PYRS_SO}" "${SITE_PACKAGES}/"
    # Also copy the pybackend if present
    find . -name "pybackend2*.so" -exec cp -v {} "${SITE_PACKAGES}/" \;
fi

echo "==> Verifying..."
python3 -c "import pyrealsense2 as rs; print('pyrealsense2 version:', rs.__version__)" && {
    echo "==> SUCCESS: pyrealsense2 installed."
} || {
    echo "==> FAILED: pyrealsense2 not importable. Check build output above."
    exit 1
}

echo "==> Cleaning up build directory..."
rm -rf "${BUILD_DIR}"
echo "==> Done."
