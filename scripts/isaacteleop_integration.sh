#!/usr/bin/env bash
# isaacteleop_integration.sh — One-stop integration for IsaacTeleop + Isaac Lab SO-101

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

usage() {
    cat <<'EOF'
IsaacTeleop + Isaac Lab SO-101 Integration
===========================================

This script wraps the full IsaacTeleop workflow for SO-101:
  1. Build so101_leader plugin (required for joint-space teleop)
  2. Start CloudXR runtime
  3. Launch teleop in Isaac Lab (sim)
  4. Record demonstrations to HDF5

Usage:
  ./scripts/isaacteleop_integration.sh <command> [options]

Commands:
  build-plugin          Build so101_leader plugin from source
  start-cloudxr         Start CloudXR runtime (accepts EULA)
  teleop                Run teleop in Isaac Lab (sim)
  record                Record demonstrations to HDF5
  real-teleop           Run teleop on REAL robot (via LeRobot)
  real-record           Record demonstrations on REAL robot

Options:
  --task TASK           Isaac Lab task (default: SO101-CylReach-Single-v0)
  --device DEVICE       Teleop device: so101_leader | xr_controller (default: so101_leader)
  --xr                  Enable XR/CloudXR (for headset)
  --num-demos N         Number of demos to record (default: 10)
  --dataset-file PATH   Output HDF5 file (default: ./datasets/so101_demos.hdf5)
  --leader-port PORT    Leader serial port (default: /dev/ttyACM1)
  --follower-port PORT  Follower serial port (default: /dev/ttyACM0)
  --calib-file FILE     Leader calibration file (default: so101_leader.calib)

Examples:
  # Build plugin (one-time)
  ./scripts/isaacteleop_integration.sh build-plugin

  # Start CloudXR (one-time per session)
  ./scripts/isaacteleop_integration.sh start-cloudxr

  # Teleop in sim (SO-101 Leader, no headset)
  ./scripts/isaacteleop_integration.sh teleop --task SO101-CylReach-Single-v0

  # Teleop in sim (XR controller with headset)
  ./scripts/isaacteleop_integration.sh teleop --task SO101-CylReach-Single-v0 --xr --device xr_controller

  # Record demos in sim
  ./scripts/isaacteleop_integration.sh record --task SO101-CylReach-Single-v0 --num-demos 20

  # Real robot teleop (requires LeRobot checkout)
  ./scripts/isaacteleop_integration.sh real-teleop --leader-port /dev/ttyACM1 --follower-port /dev/ttyACM0

  # Real robot recording
  ./scripts/isaacteleop_integration.sh real-record --leader-port /dev/ttyACM1 --follower-port /dev/ttyACM0
EOF
}

# Defaults
TASK="SO101-CylReach-Single-v0"
DEVICE="so101_leader"
XR=false
NUM_DEMOS=10
DATASET_FILE="./datasets/so101_demos.hdf5"
LEADER_PORT="/dev/ttyACM1"
FOLLOWER_PORT="/dev/ttyACM0"
CALIB_FILE="so101_leader.calib"
ISAAC_TELEOP_INSTALL="${HOME}/IsaacTeleop/install"

if [ $# -eq 0 ]; then
    usage
    exit 1
fi

CMD="$1"
shift

while [ $# -gt 0 ]; do
    case "$1" in
        --task) TASK="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --xr) XR=true; shift ;;
        --num-demos) NUM_DEMOS="$2"; shift 2 ;;
        --dataset-file) DATASET_FILE="$2"; shift 2 ;;
        --leader-port) LEADER_PORT="$2"; shift 2 ;;
        --follower-port) FOLLOWER_PORT="$2"; shift 2 ;;
        --calib-file) CALIB_FILE="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

# Build XR args
XR_ARGS=()
if [ "$XR" = true ]; then
    XR_ARGS=(--xr)
fi

# Build viz args
VIZ_ARGS=(--viz kit)

case "$CMD" in
    build-plugin)
        echo "Building so101_leader plugin..."
        bash "${REPO_ROOT}/scripts/setup_isaacteleop.sh"
        ;;

    start-cloudxr)
        echo "Starting CloudXR runtime..."
        python -m isaacteleop.cloudxr.service start --accept-eula
        echo "CloudXR started. Source env with: source ~/.cloudxr/run/cloudxr.env"
        ;;

    teleop)
        echo "Launching teleop in Isaac Lab: ${TASK}"
        echo "Device: ${DEVICE} | XR: ${XR}"
        ./isaaclab.sh -p scripts/teleop_se3_agent.py \
            --task "${TASK}" \
            --teleop-device "${DEVICE}" \
            "${XR_ARGS[@]}" \
            "${VIZ_ARGS[@]}"
        ;;

    record)
        echo "Recording demonstrations: ${TASK}"
        echo "Target demos: ${NUM_DEMOS} | Output: ${DATASET_FILE}"
        ./isaaclab.sh -p scripts/record_demos.py \
            --task "${TASK}" \
            --dataset_file "${DATASET_FILE}" \
            --num_demos "${NUM_DEMOS}" \
            --step_hz 30 \
            --teleop-device "${DEVICE}" \
            "${XR_ARGS[@]}" \
            "${VIZ_ARGS[@]}"
        ;;

    real-teleop)
        echo "Real robot teleop (via LeRobot + IsaacTeleop)"
        echo "Leader: ${LEADER_PORT} | Follower: ${FOLLOWER_PORT}"

        # Check LeRobot checkout
        if [ ! -d "${HOME}/lerobot/examples/isaac_teleop_to_so101" ]; then
            echo "ERROR: LeRobot checkout not found at ~/lerobot"
            echo "Clone with: git clone https://github.com/huggingface/lerobot.git ~/lerobot"
            exit 1
        fi

        # Check plugin
        PLUGIN="${ISAAC_TELEOP_INSTALL}/plugins/so101_leader/so101_leader_plugin"
        if [ ! -f "${PLUGIN}" ]; then
            echo "ERROR: so101_leader plugin not found at ${PLUGIN}"
            echo "Run: ./scripts/isaacteleop_integration.sh build-plugin"
            exit 1
        fi

        # Source CloudXR env
        if [ -f ~/.cloudxr/run/cloudxr.env ]; then
            source ~/.cloudxr/run/cloudxr.env
        else
            echo "WARNING: CloudXR env not found. Run 'start-cloudxr' first."
        fi

        cd ~/lerobot
        python -m examples.isaac_teleop_to_so101.teleoperate \
            --robot.type=so101_follower \
            --robot.port="${FOLLOWER_PORT}" \
            --robot.id=mnri_arms_1_follower \
            --teleop.type="${DEVICE}" \
            --teleop.port="${LEADER_PORT}" \
            --teleop.id=mnri_arms_1_leader \
            --launch_plugin="${PLUGIN}"
        ;;

    real-record)
        echo "Real robot recording (via LeRobot + IsaacTeleop)"
        echo "Leader: ${LEADER_PORT} | Follower: ${FOLLOWER_PORT}"

        if [ ! -d "${HOME}/lerobot/examples/isaac_teleop_to_so101" ]; then
            echo "ERROR: LeRobot checkout not found at ~/lerobot"
            exit 1
        fi

        PLUGIN="${ISAAC_TELEOP_INSTALL}/plugins/so101_leader/so101_leader_plugin"
        if [ ! -f "${PLUGIN}" ]; then
            echo "ERROR: so101_leader plugin not found. Run build-plugin first."
            exit 1
        fi

        if [ -f ~/.cloudxr/run/cloudxr.env ]; then
            source ~/.cloudxr/run/cloudxr.env
        fi

        cd ~/lerobot
        python -m examples.isaac_teleop_to_so101.record \
            --robot.type=so101_follower \
            --robot.port="${FOLLOWER_PORT}" \
            --robot.id=mnri_arms_1_follower \
            --teleop.type="${DEVICE}" \
            --teleop.port="${LEADER_PORT}" \
            --teleop.id=mnri_arms_1_leader \
            --launch_plugin="${PLUGIN}" \
            --robot.cameras="{wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, overhead: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \
            --dataset.repo_id=thakk100/so101_duck_push \
            --dataset.single_task="push the duck into the square tape" \
            --dataset.num_episodes=50 \
            --dataset.episode_time_s=20 \
            --dataset.reset_time_s=5
        ;;

    *)
        echo "Unknown command: $CMD"
        usage
        exit 1
        ;;
esac