# ENVIRONMENT TESTING GUIDE — all envs, how to control

Everything assumes:
```bash
cd /Storage/Projects/so101_mnri && source .venv/bin/activate
```

---

## Available environments

### MuJoCo (9 envs, CPU-capable)

| Env | Type | Task | Obs dims | Success criterion |
|-----|------|------|----------|-------------------|
| `single` | single arm | reach joint positions | 29 | joints match target within 0.2 rad |
| `dual` | dual arm | reach joint positions | 33 | both arms match targets |
| `pick_lift` | single arm | lift box 10 cm | 29 | box lifted > 10 cm |
| `pick_place` | single arm | pick → place on platform | 29 | box on target platform |
| `cyl_grasp` | dual arm | both arms grasp cylinder | 33 | both grippers closed on cylinder |
| `cyl_reach` | dual arm | both arms reach cylinder | 33 | EE within 5 cm of cylinder |
| `push_t` | single arm | push T-block onto target outline | 27 | center < 2.5 cm + yaw < 15° |
| `cube_push_ramp` | single arm | push cube up 15° ramp | 24 (full) / 114 (belief) | cube reaches goal height |
| `cube_push_bridge` | single arm | push cube across narrow bridge | 24 (full) / 114 (belief) | cube reaches goal position |

### Isaac Sim (5 tasks, GPU required)

| Task ID | Arms | Description |
|---------|------|-------------|
| `SO101-PickLift-Single-v0` | 1 | Lift box |
| `SO101-PickPlace-Single-v0` | 1 | Pick and place |
| `SO101-CylReach-Single-v0` | 1 | Single-arm cylinder reach |
| `SO101-CylGrasp-Dual-v0` | 2 | Dual-arm cylinder grasp |
| `SO101-CylReach-Dual-v0` | 2 | Dual-arm cylinder reach |

---

## Testing each environment

### Step 1: Validation (no hardware, no viewer)

```bash
# MuJoCo — all 9 envs, headless, ~30s total
MUJOCO_GL=egl python scripts/validate_actions.py          # zero + random actions
MUJOCO_GL=egl python scripts/verify_manual.py --env single --headless-check
MUJOCO_GL=egl python scripts/verify_manual.py --env dual --headless-check
MUJOCO_GL=egl python scripts/verify_manual.py --env pick_lift --headless-check
MUJOCO_GL=egl python scripts/verify_manual.py --env pick_place --headless-check
MUJOCO_GL=egl python scripts/verify_manual.py --env cyl_grasp --headless-check
MUJOCO_GL=egl python scripts/verify_manual.py --env cyl_reach --headless-check

# Isaac — all 5 tasks, GPU, headless
python -m envs.isaac.scripts.validate_actions   # zero + random, all tasks
```

### Step 2: Gamepad dry-run (no hardware, verifies DS4 mapping)

```bash
# MuJoCo — single + dual
MUJOCO_GL=egl python scripts/teleop_gamepad_ik.py --env single --dry-run
MUJOCO_GL=egl python scripts/teleop_gamepad_ik.py --env dual --dry-run

# Isaac — single + dual
python scripts/isaac_teleop_gamepad.py --task SO101-CylReach-Single-v0 --null-device --headless
python scripts/isaac_teleop_gamepad.py --task SO101-CylGrasp-Dual-v0 --null-device --arm both
```

### Step 3: Camera renders (captures images for docs)

```bash
python -m envs.isaac.scripts.render_cameras   # all Isaac tasks
python scripts/capture_env_images.py           # MuJoCo envs
```

---

## Live gamepad teleop

### Prerequisites

1. DS4 connected (USB or Bluetooth)
2. `python scripts/check_gamepad.py` — verify axes/buttons move sanely
3. `groups | grep input` — must show `input` group

### MuJoCo — single arm

```bash
# Desktop session (viewer opens):
python scripts/teleop_gamepad_ik.py --env single

# Headless (SSH, no display):
MUJOCO_GL=egl python scripts/teleop_gamepad_ik.py --env single --no-viewer
```

**Controls** (single arm):
- Left stick = move gripper XY
- Right stick (up/down) = move gripper Z
- Circle (hold) = gripper close · Cross (hold) = gripper open
- L1 or R1 (press) = safe-home (ramp to HOME_POSE, re-center)
- Options = re-center target at current gripper position
- PS = e-stop (freeze; Options resumes)
- Square = safe quit
- Red sphere = IK target (tracks through e-stop/homing)

### MuJoCo — dual arm

```bash
python scripts/teleop_gamepad_ik.py --env dual --arm left    # Share toggles arms
```

Both arms simulate simultaneously; only the "active" arm responds to
sticks. Press **Share** to switch which arm you're driving.

### Isaac — single arm

```bash
python scripts/isaac_teleop_gamepad.py --task SO101-PickLift-Single-v0
python scripts/isaac_teleop_gamepad.py --task SO101-CylReach-Single-v0
```

Same mapping as MuJoCo single. Opens Isaac window with red sphere at target.

### Isaac — dual arm

```bash
python scripts/isaac_teleop_gamepad.py --task SO101-CylGrasp-Dual-v0 --arm both
```

Both arms respond simultaneously (after the dual-arm remap is done per
DUAL_ARM_GAMEPAD_HANDOFF.md). Until then, use `--arm left` + Share toggle.

---

## VR teleop (Quest 3)

**The VR control is already what you described**: joysticks = target position,
trigger = continuous analog gripper.

### Quick start

**Terminal 1** — CloudXR runtime + web client:
```bash
python -m isaacteleop.cloudxr --host-client
# Ready when: "CloudXR runtime: running" + "CloudXR WSS proxy: running"
```

**Terminal 2** — Isaac with XR session:
```bash
python scripts/isaac_teleop_vr.py --headless --cloudxr external
# Wait for: "[vr] waiting for the Quest ..."
```

**Quest 3**:
1. Put on headset → open **Browser**
2. Go to `https://<laptop-ip>:48322/` → accept cert → see "Certificate Accepted"
3. Open: `https://<laptop-ip>:48322/client/?serverIP=<laptop-ip>&port=48322`
4. Click **Enter VR** → click **Play / Start Teleop**
5. Move right controller → SO-101 follows the delta

### VR control mapping

| Input | Effect |
|---|---|
| Right controller pose | EE target (clutched delta around origin) |
| Right trigger (analog, continuous) | Gripper closedness (0=open, 1=closed) |
| Right grip (squeeze) | Re-center clutch (arm stays, new origin) |
| A button | Toggle XR anchor rotation |

The trigger is a **continuous analog** value — squeeze halfway = half closed.
This is exactly the "grippers mounted to the continuous triggers" you asked for.

### Validate VR without headset

```bash
OMNI_KIT_ACCEPT_EULA=YES python scripts/isaac_teleop_vr.py --null-device --steps 100
# Runs the real retargeting + IK pipeline headless with a scripted trajectory
# Prints PASS if all values are finite and gripper ramp is correct
```

### One-time setup (do tonight, not demo morning)

```bash
# 1. Accept CloudXR EULA
CXR_INSTALL_DIR=~/.cloudxr python -c \
  "from isaacteleop.cloudxr.runtime import check_eula; check_eula(accept_eula=True); print('EULA OK')"

# 2. Pre-fetch web client (makes demo day LAN-independent)
ORIGIN=$(python -c "from isaacteleop.cloudxr.oob_teleop_env import default_web_client_origin; print(default_web_client_origin())")
mkdir -p ~/.cloudxr/static-client
curl -fsSL -o ~/.cloudxr/static-client/index.html "$ORIGIN/index.html"
curl -fsSL -o ~/.cloudxr/static-client/bundle.js   "$ORIGIN/bundle.js"

# 3. Free firewall ports
sudo ufw allow 48322/tcp
sudo ufw allow 49100/tcp && sudo ufw allow 49100/udp

# 4. Smoke-test runtime
timeout 40 python -m isaacteleop.cloudxr
# Should show readiness sentinel within ~15s
```

---

## Leader arm → sim (real2sim)

```bash
# Sanity check first
python scripts/real_to_sim_check.py --port /dev/ttyACM0 --leader-id leader
python scripts/real_to_sim_check.py --port /dev/ttyACM0 --leader-id leader --mirror-sim

# Full teleop
python scripts/teleop_leader_lerobot.py --env single \
    --leader-port /dev/ttyACM0 --leader-id leader
```

---

## Sim → real follower (sim2real)

```bash
# Dry-run first (always)
python scripts/sim_to_real.py --source gamepad --dry-run

# Live with DS4
python scripts/sim_to_real.py --source gamepad \
    --follower-port /dev/ttyACM1 --follower-id follower --torque-limit 40
```

---

## Quick reference — all dry-run commands

```bash
# MuJoCo validation
MUJOCO_GL=egl python scripts/validate_actions.py
MUJOCO_GL=egl python scripts/verify_manual.py --env single --headless-check

# MuJoCo gamepad dry-run
MUJOCO_GL=egl python scripts/teleop_gamepad_ik.py --env single --dry-run
MUJOCO_GL=egl python scripts/teleop_gamepad_ik.py --env dual --dry-run

# Isaac validation
python -m envs.isaac.scripts.validate_actions

# Isaac gamepad dry-run
python scripts/isaac_teleop_gamepad.py --task SO101-CylReach-Single-v0 --null-device --headless
python scripts/isaac_teleop_gamepad.py --task SO101-CylGrasp-Dual-v0 --null-device --arm both

# VR dry-run
OMNI_KIT_ACCEPT_EULA=YES python scripts/isaac_teleop_vr.py --null-device

# Real2sim / sim2real dry-runs
python scripts/real_to_sim_check.py --dry-run
python scripts/sim_to_real.py --source gamepad --dry-run
python scripts/sim_to_real.py --source script --dry-run
python scripts/sim_to_real.py --source checkpoint --task pick_lift --checkpoint <path> --dry-run
```

All dry-runs should exit 0. If any fail, check `test.md` for known issues.
