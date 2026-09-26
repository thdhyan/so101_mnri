# Hardware assignment for SO-101 single-arm data collection.
# Sourced by scripts/record_so101.sh -- edit here, not in the script.
#
# Arms are identified by USB SERIAL, never by /dev/ttyACM* path. Enumeration
# order changes on every replug, and on 2026-09-25 it silently reversed: the
# leader moved from ttyACM1 to ttyACM0. A path-based config then wrote the
# FOLLOWER's calibration into the LEADER's servos.
#
# record_so101.sh resolves these serials to whatever the ports are right now,
# via scripts/resolve_arm_ports.sh, and aborts if either is missing.
#
# To see the current mapping:
#   bash scripts/resolve_arm_ports.sh
#
# To re-identify an arm from scratch (move ONE arm, then read the report):
#   tmux new -d -s so101id \
#     '.venv/bin/python scripts/probe_arm_ports.py --port /dev/ttyACM0 --port /dev/ttyACM1 --out data/port_id/probe.jsonl'
#   .venv/bin/python scripts/port_id_report.py

# --- arms ------------------------------------------------------------------
# Leader = the arm you hold in your hand. Follower = the arm on the table.
LEADER_SERIAL=5AAF263743
FOLLOWER_SERIAL=5AAF263600

# LeRobot calibration ids; files live in robots/so101/calibration/.
LEADER_ID=mnri_arms_1_leader
FOLLOWER_ID=mnri_arms_1_follower

# --- cameras ---------------------------------------------------------------
# Only EVEN /dev/videoN indices are real capture devices; the odd ones are the
# matching UVC metadata nodes. Open with fourcc=MJPG -- the default open picks
# YUYV and comes back blown out to flat white.
GLOBAL_CAM=4    # 'USB Camera' 1920x1080, aimed at the workspace
WRIST_CAM=2     # 'Innomaker-U20CAM-720P', on the follower's wrist

WIDTH=1280
HEIGHT=720
FPS=30
FOURCC=MJPG
# LeRobot's connect() loop waits this long for the first frame, and requires a
# FRESH frame each ~100ms (the event is cleared after every read). The wrist
# camera measured 0.85s for its first frame at 1280x720 MJPG, so warmup_s=1
# leaves only ~150ms of margin and intermittently raises
# "TimeoutError: Timed out waiting for frame from camera OpenCVCamera(2)".
# connect() always burns the full window, so this costs one-time startup time.
WARMUP_S=5

# --- dataset ---------------------------------------------------------------
# NOTE: data/thakk100_so101_duck_push/ already holds 22 episodes (8.7 GB) from
# the OLD hand-rolled scripts/teleop_record_real.py path. Those are custom .npz
# files, 5-DOF (no wrist_roll) at 640x480 -- not a LeRobotDataset, and not
# compatible with the 6-DOF arms. Point at a fresh repo_id so the two formats
# never land in the same directory.
REPO_ID=thakk100/so101_single_arm_6dof
RECORD_TASK="push the duck into the square tape"
# Parent directory only. The record script appends REPO_ID with "/" -> "_", so
# the dataset lands in data/thakk100_so101_single_arm_6dof. LeRobot's
# --dataset.root is the dataset directory itself and must not exist yet, which
# is why passing the bare parent here would fail.
DATA_PARENT=data
NUM_EPISODES=100
EPISODE_TIME_S=30

# --- debug server ----------------------------------------------------------
GRPC_PORT=9876     # producers (lerobot-record) send log data here
DEBUG_PORT=9090   # the Web UI you look at
