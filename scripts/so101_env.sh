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

# 640x480, not 1280x720. Measured on this laptop: the camera read costs ~32 ms
# at ANY resolution (it is bound by the 30 fps frame interval, not by decode),
# but every pixel after that costs real time -- dataset video encoding, the
# rerun JPEG per frame, and the image-writer queues. 720p made the record loop
# run at 5 Hz, which LeRobot correctly warns drops frames and destabilises
# control. 640x480 cuts that pixel load ~4x.
WIDTH=640
HEIGHT=480
FOURCC=MJPG

# 15 Hz dataset/control rate, not 30. This is a laptop; two cameras + video
# encoding + the live debug view cannot hold 30 Hz. FPS is written into
# meta/info.json and drives delta_timestamps at train time, so pick a rate the
# loop genuinely sustains rather than one it drops frames trying to reach.
FPS=15

# The CAMERA capture rate is a separate thing and must be what the hardware
# actually does. Both cameras expose exactly one frame interval at 640x480 --
# `Interval: Discrete 0.033s (30.000 fps)` -- and OpenCVCamera._validate_fps
# uses math.isclose(..., rel_tol=1e-3), so asking for 15 raises
#   RuntimeError: OpenCVCamera(4) failed to set fps=15 (actual_fps=30.0).
# The camera streams at 30 and the 15 Hz loop takes the latest frame from it via
# read_latest(), so you get 15 evenly spaced distinct frames per second.
CAMERA_FPS=30

# Video encoding is the single biggest cost in the loop. Streaming encoding
# moves it off the critical path instead of doing it all at save_episode.
STREAMING_ENCODING=true
ENCODER_THREADS=2
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
# Fresh dataset as of 2026-09-25 (the earlier so101_single_arm_6dof takes were
# deleted before the real collection).
REPO_ID=thakk100/pink_duck_in_a_box
RECORD_TASK="push the duck into the square tape"
# Parent directory only. The record script appends REPO_ID with "/" -> "_", so
# the dataset lands in data/thakk100_pink_duck_in_a_box. LeRobot's
# --dataset.root is the dataset directory itself and must not exist yet, which
# is why passing the bare parent here would fail.
DATA_PARENT=data
NUM_EPISODES=100
EPISODE_TIME_S=20
# Dataset is currently empty (0 episodes). Re-resolving this and running
# record_so101.sh again would mix two different capture settings into one
# dataset, so the launcher refuses unless this is flipped deliberately.
ALLOW_MIXED_CAPTURE_SETTINGS=false

# --- debug server ----------------------------------------------------------
GRPC_PORT=9876     # producers (lerobot-record) send log data here
DEBUG_PORT=9090   # the Web UI you look at
