#!/usr/bin/env bash
# setup_isaacteleop.sh — Build and install IsaacTeleop's so101_leader plugin.
# This plugin is required for SO-101 Leader joint-space teleop in Isaac Lab.

set -euo pipefail

# Where to clone/build IsaacTeleop
ISAAC_TELEOP_DIR="${HOME}/IsaacTeleop"
INSTALL_DIR="${ISAAC_TELEOP_DIR}/install"

echo "═══════════════════════════════════════════════════════════════"
echo "  IsaacTeleop SO-101 Leader Plugin Build"
echo "  Source: ${ISAAC_TELEOP_DIR}"
echo "  Install: ${INSTALL_DIR}"
echo "═══════════════════════════════════════════════════════════════"

# 1. Install build prerequisites
echo "[1/5] Installing build prerequisites..."
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake libx11-dev clang-format-14 ccache patchelf \
    pkg-config glslang-tools libyaml-cpp-dev

# 2. Clone IsaacTeleop
echo "[2/5] Cloning IsaacTeleop..."
if [ -d "${ISAAC_TELEOP_DIR}" ]; then
    echo "  Directory exists, pulling latest..."
    cd "${ISAAC_TELEOP_DIR}"
    git pull
else
    git clone https://github.com/NVIDIA/IsaacTeleop.git "${ISAAC_TELEOP_DIR}"
    cd "${ISAAC_TELEOP_DIR}"
fi

# 3. Configure with CMake
echo "[3/5] Configuring with CMake..."
cmake -B build \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    -DENABLE_CLANG_FORMAT_CHECK=ON

# 4. Build
echo "[4/5] Building (parallel)..."
cmake --build build --parallel

# 5. Install
echo "[5/5] Installing to ${INSTALL_DIR}..."
cmake --install build

# Verify
PLUGIN="${INSTALL_DIR}/plugins/so101_leader/so101_leader_plugin"
if [ -f "${PLUGIN}" ]; then
    echo ""
    echo "✅ Build successful!"
    echo "   Plugin: ${PLUGIN}"
    echo ""
    echo "Test the plugin (synthetic mode, no hardware):"
    echo "  ${PLUGIN}"
    echo ""
    echo "To use with real SO-101 Leader on /dev/ttyACM1:"
    echo "  source ~/.cloudxr/run/cloudxr.env"
    echo "  ${PLUGIN} /dev/ttyACM1 so101_leader so101_leader.calib"
    echo ""
    echo "Calibrate the leader first:"
    echo "  ${PLUGIN} calibrate /dev/ttyACM1 so101_leader.calib"
else
    echo "❌ Build failed - plugin not found at ${PLUGIN}"
    exit 1
fi