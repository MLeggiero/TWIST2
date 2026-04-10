#!/bin/bash
# Download librealsense2 .deb packages for
# JetPack 6.2 / Ubuntu 22.04 (jammy) arm64, for offline install on the Jetson.
#
# NOTE: python3-pyrealsense2 is NOT available for arm64 in the Intel repo.
# Use build_pyrealsense2.sh on the Jetson to build it from source.
#
# Run this on a machine with internet access.
# The deploy_to_robot.sh script will transfer and install them.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEB_DIR="${SCRIPT_DIR}/.realsense_debs"
DISTRO="jammy"
ARCH="arm64"
REPO_URL="https://librealsense.intel.com/Debian/apt-repo"

mkdir -p "${DEB_DIR}"
cd "${DEB_DIR}"

# Create a temporary apt config to download without polluting the host system
TMPDIR=$(mktemp -d)
trap "rm -rf ${TMPDIR}" EXIT

# Get the Intel signing keys from keyserver (the direct URL returns 403)
echo "==> Fetching Intel RealSense GPG keys from keyserver..."
KEYRING="${TMPDIR}/librealsense.gpg"
GPG_KEYS=(
    "F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE"
    "FB0B24895113F120"
)
for key in "${GPG_KEYS[@]}"; do
    gpg --keyserver keyserver.ubuntu.com --recv-key "${key}" 2>/dev/null
done
gpg --export "${GPG_KEYS[@]}" | gpg --dearmor -o "${KEYRING}" 2>/dev/null
echo "    Keys imported."

echo "==> Downloading .deb packages for ${DISTRO} / ${ARCH}..."

# Only packages that exist for arm64 (python3-pyrealsense2 is NOT available)
PACKAGES=(
    "librealsense2"
    "librealsense2-utils"
    "librealsense2-udev-rules"
    "librealsense2-gl"
    "librealsense2-dev"
)

# Set up isolated apt config pointing at the Intel repo
APT_CONF="${TMPDIR}/apt.conf"
SOURCES_DIR="${TMPDIR}/sources.list.d"
APT_CACHE="${TMPDIR}/cache"
APT_STATE="${TMPDIR}/state"
mkdir -p "${SOURCES_DIR}" "${APT_CACHE}/archives/partial" "${APT_STATE}/lists/partial"

cat > "${APT_CONF}" <<APTEOF
Dir::Etc::sourcelist "${SOURCES_DIR}/realsense.list";
Dir::Etc::sourceparts "${SOURCES_DIR}";
Dir::Cache "${APT_CACHE}";
Dir::State "${APT_STATE}";
Dir::Etc::trusted "${KEYRING}";
APT::Architecture "${ARCH}";
APTEOF

echo "deb [arch=${ARCH}] ${REPO_URL} ${DISTRO} main" > "${SOURCES_DIR}/realsense.list"

echo "    Updating package list..."
apt-get -c "${APT_CONF}" update -qq 2>&1 | grep -v "Key is stored in legacy"

echo "    Downloading packages..."
for pkg in "${PACKAGES[@]}"; do
    echo "    -> ${pkg}"
    apt-get -c "${APT_CONF}" download "${pkg}:${ARCH}" -q 2>&1 || \
        echo "       WARNING: Could not download ${pkg}"
done

echo ""
echo "==> Downloaded packages:"
ls -lh "${DEB_DIR}"/*.deb 2>/dev/null || echo "    No .deb files found - check errors above."
echo ""
echo "NOTE: python3-pyrealsense2 is not available for arm64."
echo "After deploying, run build_pyrealsense2.sh on the Jetson (requires internet)."
echo ""
echo "==> Run deploy_to_robot.sh to transfer and install these on the Jetson."
