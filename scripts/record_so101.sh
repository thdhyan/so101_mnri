#!/usr/bin/env bash
# Record single-arm SO-101 teleop demos to a LeRobotDataset, streaming the
# live view into the Rerun debug server's Web UI.
#
#   terminal 1:  bash scripts/so101_debug_server.py            # the Web UI
#   terminal 2:  bash scripts/record_so101.sh                 # record
#   browser:     http://localhost:9090
#
# Ports and camera indices are env-overridable, because the USB enumeration
# order is not stable across reboots. Set them in scripts/so101_env.sh (or
# export them yourself) rather than editing this file.
#
#   LEADER_PORT   serial device of the arm you hold
#   FOLLOWER_PORT serial device of the arm on the table
#   GLOBAL_CAM    /dev/videoN of the overhead/workspace camera
#   WRIST_CAM     /dev/videoN of the camera on the follower's wrist
#
# Everything else: RECORD_TASK, REPO_ID, NUM_EPISODES, EPISODE_TIME_S, FPS,
# WIDTH, HEIGHT, GRPC_PORT, DEBUG_PORT.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
[[ -f scripts/so101_env.sh ]] && source scripts/so101_env.sh

LEADER_SERIAL="${LEADER_SERIAL:-5AAF263743}"
FOLLOWER_SERIAL="${FOLLOWER_SERIAL:-5AAF263600}"
LEADER_ID="${LEADER_ID:-mnri_arms_1_leader}"
FOLLOWER_ID="${FOLLOWER_ID:-mnri_arms_1_follower}"

# Resolve the roles to whatever /dev/ttyACM* they occupy RIGHT NOW. Do not
# hardcode these: enumeration order reverses when you replug or change USB
# ports, and a stale path silently drives the wrong arm -- including writing
# that arm's calibration into the other one's servos.
if ! eval "$(bash "$REPO_ROOT/scripts/resolve_arm_ports.sh" \
              --leader "$LEADER_SERIAL" --follower "$FOLLOWER_SERIAL" --emit)"; then
  echo "ERROR: could not resolve arm serials to serial ports. See above." >&2
  echo "       Check:  ls /dev/ttyACM*   and   bash scripts/resolve_arm_ports.sh" >&2
  exit 1
fi

GLOBAL_CAM="${GLOBAL_CAM:-4}"
WRIST_CAM="${WRIST_CAM:-2}"
WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-720}"
FPS="${FPS:-30}"
FOURCC="${FOURCC:-MJPG}"
WARMUP_S="${WARMUP_S:-5}"

REPO_ID="${REPO_ID:-thakk100/so101_single_arm_6dof}"
RECORD_TASK="${RECORD_TASK:-push the duck into the square tape}"
NUM_EPISODES="${NUM_EPISODES:-100}"
EPISODE_TIME_S="${EPISODE_TIME_S:-30}"
DRY_RUN=""
[[ "${1:-}" == "--dry-run" ]] && { DRY_RUN="--dry-run"; shift; }

GRPC_PORT="${GRPC_PORT:-9876}"

# --dataset.root is the dataset directory ITSELF, not a parent to put the
# repo_id folder under, and LeRobotDatasetMetadata.create() requires that
# directory NOT to exist yet. So build the full path here rather than passing
# the bare parent (passing "data" makes LeRobot try to mkdir data/, which fails
# with FileExistsError on the second run).
DATA_PARENT="${DATA_PARENT:-data}"
DATASET_ROOT="${DATA_PARENT}/$(printf '%s' "$REPO_ID" | tr '/' '_')"

CALIB_DIR="$REPO_ROOT/robots/so101/calibration"
PY="$REPO_ROOT/.venv/bin/python"

# --- preflight -------------------------------------------------------------
if [[ ! -x "$PY" ]]; then
  echo "ERROR: venv python not found at $PY" >&2
  exit 1
fi
for p in "$LEADER_PORT" "$FOLLOWER_PORT"; do
  [[ -e "$p" ]] || { echo "ERROR: $p does not exist" >&2; exit 1; }
done
if [[ "$LEADER_PORT" == "$FOLLOWER_PORT" ]]; then
  echo "ERROR: LEADER_PORT and FOLLOWER_PORT are both $LEADER_PORT" >&2
  exit 1
fi
if ! ss -ltn 2>/dev/null | grep -q ":$GRPC_PORT\b"; then
  echo "WARNING: nothing is listening on $GRPC_PORT, so the debug server is not running." >&2
  echo "         The recording will still work, but you will not see the live view." >&2
  echo "         Start it with:  bash scripts/so101_debug_server.py" >&2
  echo >&2
fi

# A fresh dataset needs a directory that does not exist yet; an existing one has
# to be resumed, because LeRobotDataset.create() does mkdir(exist_ok=False) and
# would fail with FileExistsError. Pick the mode from what is on disk.
RESUME=false
if [[ -d "$DATASET_ROOT" ]]; then
  if [[ -f "$DATASET_ROOT/meta/info.json" ]]; then
    RESUME=true
  else
    echo "ERROR: $DATASET_ROOT exists but has no meta/info.json." >&2
    echo "       It is not a LeRobot dataset. Move it aside, or point DATA_PARENT/REPO_ID elsewhere." >&2
    exit 1
  fi
fi

# The wrist cam is a plain /dev/videoN node. Odd-numbered nodes are the UVC
# metadata nodes, not capture devices, so only even indices really work.
# warmup_s is per-camera; see the note in so101_env.sh.
CAMERAS=$(printf '{"global":{"type":"opencv","index_or_path":%s,"fps":%s,"width":%s,"height":%s,"fourcc":"%s","warmup_s":%s},"wrist":{"type":"opencv","index_or_path":%s,"fps":%s,"width":%s,"height":%s,"fourcc":"%s","warmup_s":%s}}' \
  "$GLOBAL_CAM" "$FPS" "$WIDTH" "$HEIGHT" "$FOURCC" "$WARMUP_S" \
  "$WRIST_CAM"  "$FPS" "$WIDTH" "$HEIGHT" "$FOURCC" "$WARMUP_S")

cat <<EOF
============================================================
 SO-101 single-arm recording
============================================================
 leader    $LEADER_PORT  (serial $LEADER_SERIAL, id=$LEADER_ID)
 follower  $FOLLOWER_PORT  (serial $FOLLOWER_SERIAL, id=$FOLLOWER_ID)
 cameras   global=/dev/video$GLOBAL_CAM  wrist=/dev/video$WRIST_CAM
           ${WIDTH}x${HEIGHT} @ ${FPS}fps $FOURCC
 dataset   $REPO_ID  -> $DATASET_ROOT
 task      "$RECORD_TASK"
 episodes  $NUM_EPISODES x ${EPISODE_TIME_S}s
 mode      $([[ $RESUME == true ]] && echo "RESUME (dataset already exists)" || echo "NEW dataset")
 debug UI  http://localhost:${DEBUG_PORT:-9090}/?url=rerun%2Bhttp%3A%2F%2Flocalhost%3A${GRPC_PORT}%2Fproxy
============================================================

EOF

# Guarded wrapper: installs scripts/so101_torque_guard.py, then calls
# lerobot-record's own main() with this exact argv. The guard syncs
# Goal_Position := Present_Position before enabling torque (the follower's
# EEPROM-backed goals read 0, which would otherwise command a slam into the
# limits) and retries the per-motor enable with a settle gap between motors.
exec "$PY" "$REPO_ROOT/scripts/record_so101.py" \
  --guard-retries="${GUARD_RETRIES:-4}" \
  --guard-motor-delay="${GUARD_MOTOR_DELAY:-0.25}" \
  --guard-retry-delay="${GUARD_RETRY_DELAY:-0.4}" \
  ${DRY_RUN} \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id="$FOLLOWER_ID" \
  --robot.calibration_dir="$CALIB_DIR" \
  --robot.cameras="$CAMERAS" \
  --teleop.type=so101_leader \
  --teleop.port="$LEADER_PORT" \
  --teleop.id="$LEADER_ID" \
  --teleop.calibration_dir="$CALIB_DIR" \
  --dataset.repo_id="$REPO_ID" \
  --dataset.single_task="$RECORD_TASK" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.fps="$FPS" \
  --dataset.num_episodes="$NUM_EPISODES" \
  --dataset.episode_time_s="$EPISODE_TIME_S" \
  --resume="$RESUME" \
  --display_data=true \
  --display_mode=rerun \
  --display_ip=127.0.0.1 \
  --display_port="$GRPC_PORT" \
  --play_sounds=false
