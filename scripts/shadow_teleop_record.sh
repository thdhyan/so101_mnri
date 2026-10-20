#!/usr/bin/env bash
# shadow_teleop_record.sh — Shadow teleop + camera recording in parallel.
# Leader (ACM1) -> MuJoCo sim + real follower (ACM0), while recording cameras to dataset.

set -euo pipefail

# ─── Config ──────────────────────────────────────────────────────────────────
LEADER_PORT="/dev/ttyACM1"
LEADER_ID="mnri_arms_1_leader"
FOLLOWER_PORT="/dev/ttyACM0"
FOLLOWER_ID="mnri_arms_1_follower_left"
WRIST_CAM=0          # laptop webcam (or follower-mounted cam)
GLOBAL_CAM=5         # USB Camera (overhead)
DATASET="thakk100/so101_duck_push"
TASK="push the duck into the square tape"
EPISODES=400
HZ=30
DATA_ROOT="data"
# ──────────────────────────────────────────────────────────────────────────────

echo "═══════════════════════════════════════════════════════════════"
echo "  Shadow Teleop + Camera Recording"
echo "  Leader:   $LEADER_PORT ($LEADER_ID)"
echo "  Follower: $FOLLOWER_PORT ($FOLLOWER_ID)"
echo "  Wrist cam:  $WRIST_CAM"
echo "  Global cam: $GLOBAL_CAM"
echo "  Dataset:    $DATASET"
echo "═══════════════════════════════════════════════════════════════"

# Run shadow teleop in background
python scripts/teleop_leader_lerobot.py \
  --env single \
  --leader-port "$LEADER_PORT" \
  --leader-id "$LEADER_ID" \
  --mirror-real \
  --follower-port "$FOLLOWER_PORT" \
  --follower-id "$FOLLOWER_ID" \
  --hz 60 &
TELEOP_PID=$!

# Give teleop a moment to connect
sleep 2

# Run camera recording (reads from leader, but cameras are independent)
python -m vla.record_demos \
  --leader-port "$LEADER_PORT" \
  --leader-id "$LEADER_ID" \
  --wrist-cam "$WRIST_CAM" \
  --global-cam "$GLOBAL_CAM" \
  --dataset "$DATASET" \
  --task "$TASK" \
  --episodes "$EPISODES" \
  --hz "$HZ" \
  --data-root "$DATA_ROOT"

# When recording ends (or Ctrl+C), kill teleop
kill $TELEOP_PID 2>/dev/null || true
wait $TELEOP_PID 2>/dev/null || true

echo "Done."