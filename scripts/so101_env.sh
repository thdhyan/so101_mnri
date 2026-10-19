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
#
# The two cameras were re-plugged on 2026-09-26 and their /dev/videoN indices
# SWAPPED. The roles did not change -- what each one is was confirmed from its
# own frame on that date:
#   video2 "USB Camera"               -> the whole table: duck, tape square, arm
#   video4 "Innomaker-U20CAM-720P"    -> the gripper, close up
# So the global camera is video2 and the wrist camera is video4, the reverse of
# what this file said before. Confirm with a frame before trusting it again:
#   .venv/bin/python scripts/so101_camera_flip.py --snap 2,4 --snap-out /tmp/flip
# Indices are not stable across replugs, so re-check if the views look swapped.
GLOBAL_CAM=2    # 'USB Camera' 1920x1080, aimed at the workspace
WRIST_CAM=4     # 'Innomaker-U20CAM-720P', on the follower's wrist

# 640x480, not 1280x720. Measured on this laptop: the camera read costs ~32 ms
# at ANY resolution (it is bound by the 30 fps frame interval, not by decode),
# but every pixel after that costs real time -- dataset video encoding, the
# rerun JPEG per frame, and the image-writer queues. 720p made the record loop
# run at 5 Hz, which LeRobot correctly warns drops frames and destabilises
# control. 640x480 cuts that pixel load ~4x.
WIDTH=640
HEIGHT=480
FOURCC=MJPG

# --- orientation ------------------------------------------------------------
# The global camera is mounted INVERTED, so its frames arrive upside down.
#
# Neither camera has a V4L2 flip control (v4l2-ctl -L lists only
# brightness/contrast/exposure), and LeRobot's `rotation` can only do
# 0/90/180/-90 -- 180 also mirrors left-right, which is not the same fix. So
# the flip is applied in software by scripts/so101_camera_flip.py, in the
# camera's own read thread, before the dataset or the live view ever see it.
#
# List camera NAMES, not indices: global, wrist, both, or none.
#   VFLIP_CAMS=global   # the inverted one
#   VFLIP_CAMS=both
#   VFLIP_CAMS=none
# The names are resolved to indices at launch, so this follows the camera even
# when the indices swap again.
#
# NOTE: this could not be confirmed from a still frame. The workspace view is
# top-down and its features (duck, tape square) lie along the horizontal axis,
# so they look the same way up in both orientations -- only a live view settles
# it. If the Rerun window shows the global view upside down, the fix is
# VFLIP_CAMS=wrist below; nothing else changes.
#
# This is part of the dataset's meaning, like fps and resolution: flipping one
# camera mid-dataset leaves half the episodes the other way up. Set it before
# the first episode, or start a new REPO_ID.
#
# To eyeball it without touching the arms (writes the frame AS RECORDED):
#   .venv/bin/python scripts/so101_camera_flip.py --snap 2,4 --snap-out /tmp/flip
VFLIP_CAMS=global

# --- start zones ------------------------------------------------------------
# Keys pressed while placing the duck, in zone order: the 1st key is zone 1, the
# 2nd is zone 2, and so on. Letters rather than digits because the digits are
# now the prompt selector. The rig is taped as a 2x2 grid, so four zones.
#   far from the square
#     i j        j o        o k        k l
#     i o
#     k l
# The goal (taped square) stays fixed; these only mark where the duck STARTS.
ZONE_KEYS="i o k l"

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
# Fallback prompt, used when no prompt key is pressed before an episode. Kept
# identical to the string the first 12 episodes were recorded with, so resuming
# does not orphan them as a separate task.
RECORD_TASK="push the duck into the square tape"

# --- the five prompts in this dataset ---------------------------------------
# Press 1-5 to choose the prompt for the next episode. Each frame records the
# prompt as its own `task` string, so this is a stock multi-task LeRobotDataset:
# `meta/tasks.parquet` accumulates the distinct prompts and stock LeRobot trains
# on it unchanged. Order matters -- position N here is what key N selects.
#
#   1  pick only: lift the duck clear of the table
#   2  push: gripper stays open, planar motion into the taped square
#   3  pick-and-place into the taped square
#   4  pick-and-place into the red cup
#   5  pick-and-place into the brown cup
#
# Note 2 is the only planar push; 3-5 need the gripper to close, lift and
# release. That is fine in one dataset -- it is what the instruction is for --
# but it means `s` means "the duck is in the target named by the active prompt",
# so check you pressed the right prompt digit before you label.
TASK_PROMPTS="pick the duck|push the duck into the square tape|pick the duck and put it in the square|pick the duck and put it in the red cup|pick the duck and put it in the brown cup"
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
