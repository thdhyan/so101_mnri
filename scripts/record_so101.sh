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
# Test the exit status of the RESOLVER, not of eval. `eval "$(cmd)"` swallows
# the failure: when cmd writes nothing to stdout (its errors go to stderr) the
# command substitution is empty, `eval ""` succeeds, the guard never fires, and
# set -u then dies later with a bare "LEADER_PORT: unbound variable" instead of
# the useful message. Capture first, check the status, then eval.
if ! resolved_ports="$(bash "$REPO_ROOT/scripts/resolve_arm_ports.sh" \
                        --leader "$LEADER_SERIAL" --follower "$FOLLOWER_SERIAL" --emit)"; then
  echo >&2
  echo "ERROR: could not resolve arm serials to serial ports. See above." >&2
  echo "       Arms are identified by USB serial, never by /dev/ttyACM* order." >&2
  echo "       Check:  ls /dev/ttyACM*   and   bash scripts/resolve_arm_ports.sh" >&2
  echo "       Both arms must be plugged in and powered." >&2
  exit 1
fi
eval "$resolved_ports"

GLOBAL_CAM="${GLOBAL_CAM:-4}"
WRIST_CAM="${WRIST_CAM:-2}"
WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-480}"
FPS="${FPS:-15}"
CAMERA_FPS="${CAMERA_FPS:-30}"
FOURCC="${FOURCC:-MJPG}"
WARMUP_S="${WARMUP_S:-5}"
STREAMING_ENCODING="${STREAMING_ENCODING:-true}"
ENCODER_THREADS="${ENCODER_THREADS:-2}"

# --- which cameras are mounted upside down ----------------------------------
# VFLIP_CAMS holds camera NAMES (global, wrist, both, none); the flip itself
# is applied by scripts/so101_camera_flip.py, which the Python wrapper installs.
# Resolve the names to /dev/videoN indices here, because that is the only
# handle the camera object carries.
VFLIP_CAMS="${VFLIP_CAMS:-none}"
# A typo in this variable would silently record an upside-down dataset, so an
# unrecognised token is a hard error rather than "flip nothing".
for tok in ${VFLIP_CAMS//,/ }; do
  case "$tok" in
    none|global|wrist|all|both) ;;
    *)
      echo "ERROR: VFLIP_CAMS='$VFLIP_CAMS' contains unknown camera '$tok'." >&2
      echo "       Valid: none, global, wrist, both, all (comma-separated)." >&2
      exit 2
      ;;
  esac
done
VFLIP_IDX=""
for cam_name in global wrist; do
  case ",${VFLIP_CAMS}," in
    *",${cam_name},"*|*,all,*|*,both,*)
      case "$cam_name" in
        global)  cam_idx="$GLOBAL_CAM" ;;
        wrist)   cam_idx="$WRIST_CAM" ;;
      esac
      VFLIP_IDX="${VFLIP_IDX:+$VFLIP_IDX,}$cam_idx"
      ;;
  esac
done
VFLIP_FLAG="--vflip=$VFLIP_IDX"

# "2,4" -> "/dev/video2 /dev/video4"; "" -> "no camera flipped".
flip_summary() {
  local idx="$1"
  if [[ -z "$idx" ]]; then
    echo "no camera flipped (frames recorded as the cameras deliver them)"
  else
    local out=""
    local i
    for i in ${idx//,/ }; do out+="/dev/video$i "; done
    echo "${out% } (recorded upright)"
  fi
}

REPO_ID="${REPO_ID:-thakk100/so101_single_arm_6dof}"
RECORD_TASK="${RECORD_TASK:-push the duck into the square tape}"
NUM_EPISODES="${NUM_EPISODES:-100}"
EPISODE_TIME_S="${EPISODE_TIME_S:-30}"
DRY_RUN=""
PROFILE=""
TURN_FLAG=""
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN="--dry-run"; shift ;;
    --profile) PROFILE="--profile=30"; shift ;;
    --profile=*) PROFILE="$arg"; shift ;;
    --no-turn)
      # Escape hatch: fall back to LeRobot's timer-driven reset, where 'n' cuts
      # the reset short and an episode can start before the duck is placed.
      echo "WARNING: --no-turn requested. Episodes will start on a timer, not when" >&2
      echo "         you say the scene is ready. First frames may catch the reset." >&2
      TURN_FLAG="--no-turn"; shift ;;
    --watch)
      echo "ERROR: --watch is a check_pose_sync.py flag, not a record flag." >&2
      echo "       To watch the leader/follower alignment live, run:" >&2
      echo "         python scripts/check_pose_sync.py --watch" >&2
      exit 2 ;;
    "") ;;
    *)
      echo "ERROR: unknown argument '$arg'." >&2
      echo "       record_so101.sh takes no positional arguments." >&2
      echo "       Flags: --dry-run, --profile[=SECONDS], --no-turn" >&2
      exit 2 ;;
  esac
done

GRPC_PORT="${GRPC_PORT:-9876}"

# --- live view -------------------------------------------------------------
# Default: Rerun's own native viewer. `rerun-record --display_data` with no
# display_ip/display_port calls rr.spawn(), which opens the desktop viewer
# window. No proxy, no second process, no ?url= parameter.
#
# RERUN_WEB=1 opts back into the browser view, which needs the gRPC sink
# running:  python scripts/so101_debug_server.py
DISPLAY_DATA="${DISPLAY_DATA:-true}"
DISPLAY_TARGET=""
# Treat 1/true/yes as on, so both RERUN_WEB=1 and RERUN_WEB=true work.
case "$(printf '%s' "${RERUN_WEB:-0}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on) USE_WEB=true ;;
  *)             USE_WEB=false ;;
esac
if [[ "$USE_WEB" == "true" ]]; then
  DISPLAY_TARGET="--display_ip=127.0.0.1 --display_port=$GRPC_PORT"
elif [[ "$DISPLAY_DATA" == "true" ]]; then
  # LeRobot passes this straight to rr.spawn(memory_limit=...).
  export RERUN_MEMORY_LIMIT="${RERUN_MEMORY_LIMIT:-10%}"
fi

# --dataset.root is the dataset directory ITSELF, not a parent to put the
# repo_id folder under, and LeRobotDatasetMetadata.create() requires that
# directory NOT to exist yet. So build the full path here rather than passing
# the bare parent (passing "data" makes LeRobot try to mkdir data/, which fails
# with FileExistsError on the second run).
DATA_PARENT="${DATA_PARENT:-data}"
# --profile ends mid-episode and LeRobot still saves that partial take, so keep
# it out of the real dataset.
[[ -n "$PROFILE" ]] && REPO_ID="${REPO_ID}_profile"
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
    # fps and the camera shape are baked into meta/info.json and drive
    # delta_timestamps at train time. Resuming with different values silently
    # produces a dataset whose episodes disagree about their own timing.
    if [[ "${ALLOW_MIXED_CAPTURE_SETTINGS:-false}" != "true" ]]; then
      have_fps="$("$PY" -c "import json;print(json.load(open('$DATASET_ROOT/meta/info.json')).get('fps'))" 2>/dev/null || echo "?")"
      have_shape="$("$PY" -c "
import json
f=json.load(open('$DATASET_ROOT/meta/info.json'))['features']
k=[x for x in f if x.startswith('observation.images')]
print(sorted(f[x]['shape'][1] for x in k)[0] if k else '?')" 2>/dev/null || echo "?")"
      if [[ "$have_fps" != "$FPS" || "$have_shape" != "$WIDTH" ]]; then
        echo "ERROR: $DATASET_ROOT was created at ${have_fps} fps / ${have_shape}px wide," >&2
        echo "       but this run is configured for ${FPS} fps / ${WIDTH}px wide." >&2
        echo "       Mixing them corrupts the dataset's timing." >&2
        echo "       Either restore the original settings, start a new REPO_ID, or set" >&2
        echo "       ALLOW_MIXED_CAPTURE_SETTINGS=true if you really mean it." >&2
        exit 1
      fi
    fi
  else
    echo "ERROR: $DATASET_ROOT exists but has no meta/info.json." >&2
    echo "       It is not a LeRobot dataset. Move it aside, or point DATA_PARENT/REPO_ID elsewhere." >&2
    exit 1
  fi
fi

# The wrist cam is a plain /dev/videoN node. Odd-numbered nodes are the UVC
# metadata nodes, not capture devices, so only even indices really work.
# Camera fps is CAMERA_FPS (the hardware's only supported rate), NOT the
# dataset FPS -- see the note in so101_env.sh. warmup_s is per-camera.
CAMERAS=$(printf '{"global":{"type":"opencv","index_or_path":%s,"fps":%s,"width":%s,"height":%s,"fourcc":"%s","warmup_s":%s},"wrist":{"type":"opencv","index_or_path":%s,"fps":%s,"width":%s,"height":%s,"fourcc":"%s","warmup_s":%s}}' \
  "$GLOBAL_CAM" "$CAMERA_FPS" "$WIDTH" "$HEIGHT" "$FOURCC" "$WARMUP_S" \
  "$WRIST_CAM"  "$CAMERA_FPS" "$WIDTH" "$HEIGHT" "$FOURCC" "$WARMUP_S")

cat <<EOF
============================================================
 SO-101 single-arm recording
============================================================
 leader    $LEADER_PORT  (serial $LEADER_SERIAL, id=$LEADER_ID)
 follower  $FOLLOWER_PORT  (serial $FOLLOWER_SERIAL, id=$FOLLOWER_ID)
 cameras   global=/dev/video$GLOBAL_CAM  wrist=/dev/video$WRIST_CAM
           ${WIDTH}x${HEIGHT} @ ${CAMERA_FPS}fps capture, recorded at ${FPS}fps
 flip      VFLIP_CAMS="$VFLIP_CAMS" -> $(flip_summary "$VFLIP_IDX")
 dataset   $REPO_ID  -> $DATASET_ROOT
 task      "$RECORD_TASK"
 episodes  $NUM_EPISODES x ${EPISODE_TIME_S}s
 mode      $([[ $RESUME == true ]] && echo "RESUME (dataset already exists)" || echo "NEW dataset")
 live view $(if [[ "$DISPLAY_DATA" != "true" ]]; then echo "OFF (DISPLAY_DATA=$DISPLAY_DATA)";
             elif [[ "$USE_WEB" == "true" ]]; then echo "browser  http://localhost:${DEBUG_PORT:-9090}/?url=rerun%2Bhttp%3A%2F%2Flocalhost%3A${GRPC_PORT}%2Fproxy";
             else echo "native Rerun window (spawned automatically)"; fi)
 labels    s=success+end  f=failure+end  d=discard+re-record  n=end+keep  q=quit
 turn      SPACE = "duck is placed, start recording".  Waits here first, and
           after every episode, so nothing records while you reset.
           (--no-turn goes back to LeRobot's timer reset.)
EOF

# The prompt list and the zone keys, printed outside the heredoc so the long
# strings are not mangled by its expansion rules. Sourced from so101_env.sh,
# which is the single source of truth for both.
echo " prompts   (press the number BEFORE the take; applies to the next episode)"
i=0
while IFS= read -r p; do
  i=$((i + 1))
  printf '             %d  %s\n' "$i" "$p"
done < <(printf '%s\n' "${TASK_PROMPTS:-}" | tr '|' '\n')
zn=0
zlist=""
for k in ${ZONE_KEYS:-}; do
  zn=$((zn + 1))
  zlist+="${k}=${zn} "
done
cat <<EOF
 zones      $zlist(press while placing the duck; z clears)
           coverage:  .venv/bin/python scripts/episode_outcomes.py --root $DATASET_ROOT
============================================================

EOF

# Guarded wrapper: installs scripts/so101_torque_guard.py, then calls
# lerobot-record's own main() with this exact argv. The guard syncs
# Goal_Position := Present_Position before enabling torque (the follower's
# EEPROM-backed goals read 0, which would otherwise command a slam into the
# limits) and retries the per-motor enable with a settle gap between motors.

# --- pose gate -------------------------------------------------------------
# record_loop calls send_action() on its FIRST tick with whatever the leader
# reads. If the two arms are far apart, that commands a large move: six servos
# stall against gravity, the supply rail sags, and the Feetech bus drops
# packets. That is exactly what produced the 5.3 Hz warning followed by
# "no status packet" on every motor and then a failed torque-disable.
#
# This is READ-ONLY; it only refuses to start while the mismatch is dangerous.
if [[ "${SKIP_POSE_GATE:-false}" != "true" && -z "$DRY_RUN" ]]; then
  if ! "$PY" "$REPO_ROOT/scripts/check_pose_sync.py" --max-delta="${MAX_POSE_DELTA:-10}"; then
    echo >&2
    echo "ERROR: leader and follower are not aligned. Move the arm in your HAND" >&2
    echo "       to match the pose of the arm on the table, then re-run." >&2
    echo "       Live view:  python scripts/check_pose_sync.py --watch" >&2
    echo "       To override: SKIP_POSE_GATE=true (not recommended)." >&2
    exit 2
  fi
  echo
fi

exec "$PY" "$REPO_ROOT/scripts/record_so101.py" \
  --guard-retries="${GUARD_RETRIES:-4}" \
  --guard-motor-delay="${GUARD_MOTOR_DELAY:-0.25}" \
  --guard-retry-delay="${GUARD_RETRY_DELAY:-0.4}" \
  "$VFLIP_FLAG" \
  ${TURN_FLAG} \
  ${DRY_RUN} \
  ${PROFILE} \
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
  --dataset.streaming_encoding="$STREAMING_ENCODING" \
  --dataset.encoder_threads="$ENCODER_THREADS" \
  --dataset.push_to_hub=false \
  --dataset.num_episodes="$NUM_EPISODES" \
  --dataset.episode_time_s="$EPISODE_TIME_S" \
  --resume="$RESUME" \
  --display_data="$DISPLAY_DATA" \
  --display_mode=rerun \
  ${DISPLAY_TARGET} \
  --play_sounds=false
