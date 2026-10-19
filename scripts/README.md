# SO-101 teleop / viewer scripts

All scripts use `/home/thakk100/miniconda3/envs/so101/bin/python`. For headless
rendering (no X server / GUI), set `MUJOCO_GL=egl`.

Cameras, joints, and actuators are discovered dynamically from the compiled
model at runtime (via `model.ncam`, `model.actuator(name)`, `model.joint(name)`,
`model.site(name)`) rather than hardcoded, since `envs/mujoco/*/assets/scene.xml`
may gain/rename cameras and keyframes independently of these scripts. The
stable contract these scripts rely on is: joints/actuators named
`{shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll,gripper}`
(optionally `left_`/`right_`-prefixed in the dual-arm scene), and gripper
sites named `gripperframe` / `left_gripperframe` / `right_gripperframe`.

## 1. `teleop_gamepad_ik.py` — gamepad Cartesian teleop via IK

Driver: **DualShock 4** ("Wireless Controller"), USB or Bluetooth, read via
pygame 2.x SDL (`DS4Gamepad`: SDL2 game-controller mapping preferred, raw
DS4-on-Linux indices as fallback).

### DS4 setup

1. Pair: hold **SHARE + PS** until the light bar double-flashes, connect
   "Wireless Controller" in your Bluetooth manager — or just plug USB.
2. Your user must read `/dev/input`: check `groups`; fix with
   `sudo usermod -aG input $USER` and log out/in.
3. If no device appears: `sudo modprobe joydev`, then replug/re-pair.
4. Verify pairing + all axes/buttons live in 5 seconds:
   `python scripts/check_gamepad.py`

Control mapping:

| Input                  | Effect                                        |
|-------------------------|-----------------------------------------------|
| Left stick (x/y)        | EE x/y velocity, 0.15 m/s at full deflection  |
| Right stick vertical    | EE z velocity                                 |
| Right stick horizontal  | wrist_roll rate                               |
| L1 / R1                 | wrist_flex (pitch) rate, opposite directions  |
| L2 / R2 (analog)        | gripper close / open rate                     |
| **Options**             | **re-center IK target to current EE pose** (also resumes from e-stop) |
| Share                   | toggle active arm (dual env; both arms stay simulated) |
| **PS button**           | **e-stop latch**: all arms freeze at current pose, neutral gripper; Options resumes |
| **Square**              | safe quit: freeze at current pose, neutral gripper, settle ~0.5 s, exit |

Deadzone 0.15 on sticks (0.08 on triggers after normalization). Control loop
runs at ~60 Hz; physics is stepped at the model's native timestep (multiple
substeps per control tick).

IK is damped-least-squares on the gripper site Jacobian (`mujoco.mj_jacSite`),
damping ~1e-2, with a nullspace bias pulling the arm toward the home pose
`[0, -0.5, 0.8, 0.4, 0, 0]`. Solved joint targets are clamped to joint range
and actuator ctrlrange, then written to `data.ctrl` for the position
actuators. wrist_roll/wrist_flex are driven as direct joint-rate offsets
outside the IK nullspace (2-DoF position IK is underdetermined for a 4-DoF
arm chain, so roll/pitch are teleoperated directly rather than solved).

A small red sphere marker (drawn via `viewer.user_scn`) shows the current EE
target in the passive viewer window.

**Integration point for AR-based teleop:** the `IKTeleop` class exposes
`set_ee_target(pos, quat=None)` and `set_gripper(openness)`, fully decoupled
from the gamepad-reading loop. A driver based on
[mujoco-ar-viewer](https://github.com/Improbable-AI/mujoco-ar-viewer) can
construct an `IKTeleop` the same way `build_ik()` does in this file and call
those two methods each frame instead of reading `pygame` joystick axes — no
changes to the IK/physics code are needed.

`--dry-run` runs 100 headless IK steps tracking a scripted circular target
(no window, no joystick), self-tests the DS4 deadzone/trigger math, smoke-
tests the re-center safety path, probes for a gamepad (printing setup
guidance when none is found — absence is not an error), and prints the final
EE-target tracking error — useful for CI verification.

### Dual-arm (`--env dual --arm left|right`)

Both arms are simulated and solved every tick; **Share** toggles which arm the
sticks drive. `--arm both` is deliberately NOT offered: one DS4 has only 4
stick DoF while driving two arms' XY+Z simultaneously needs 6 — that's a
control-surface limitation, not an IK limitation (the `ik_by_arm` dict already
runs two independent `IKTeleop` instances trivially).

## 2. `teleop_leader_arm.py` — real SO-101 leader arm drives the sim follower

```
python scripts/teleop_leader_arm.py --env single --port /dev/ttyACM0
python scripts/teleop_leader_arm.py --env dual --left-port /dev/ttyACM0 --right-port /dev/ttyACM1
python scripts/teleop_leader_arm.py --calibrate --port /dev/ttyACM0
MUJOCO_GL=egl python scripts/teleop_leader_arm.py --dry-run --env single
```

Talks to Feetech STS3215 servos (IDs 1-6, 1 Mbaud) via `scservo_sdk`. On
connect, torque is disabled on all leader servos (address 40 = 0) so the
physical arm moves freely by hand. Each ~60 Hz tick: `GroupSyncRead` present
position (address 56, 2 bytes) on all 6 servos, convert ticks -> radians
using the calibration, clamp to actuator ctrlrange, write to `data.ctrl`,
step physics, sync the passive viewer.

`--calibrate` prompts you to hold the physical arm at the home pose
`[0, -0.5, 0.8, 0.4, 0, 0]` (radians), then records per-servo tick offsets to
`scripts/leader_calib.json` (default path, override with `--calib`). Each
entry is `{id, offset_ticks, sign, ticks_per_rad}` (`ticks_per_rad` defaults
to `4096 / 2π`). If no calibration file exists, an identity calibration
(center = 2048 ticks, sign = +1) is used with a warning.

Missing serial ports produce a clear error and exit before touching MuJoCo.
`--dry-run` fakes the leader with a sinusoidal trajectory and runs 100
headless steps — no hardware required, for CI verification.

## 3. `view_cameras.py` — live tiled camera view

```
MUJOCO_GL=egl python scripts/view_cameras.py --env single --save /tmp/single_cams.png
MUJOCO_GL=egl python scripts/view_cameras.py --env dual --save /tmp/dual_cams.png
python scripts/view_cameras.py --env single --width 480 --height 360 --fps 30   # live GUI, ESC to quit
```

Compiles `scene.xml` directly, enumerates every named camera on the model
(`model.ncam`), renders each with a single `mujoco.Renderer`, tiles the
frames into a grid with OpenCV (camera name overlaid per tile), and either:
- `--save PATH`: renders one grid frame headless to a PNG and exits (no GUI
  needed — this is the CI-friendly path), or
- no `--save`: steps physics and shows a live `cv2.imshow` loop at `--fps`,
  ESC to quit.

Uses the `home` keyframe if the scene defines one, otherwise default qpos.

## 4. `isaac_teleop_vr.py` — Meta Quest 3 VR teleop of the Isaac Lab tasks

Streams the Isaac sim to a Quest 3 over CloudXR (WebXR browser client — no
APK sideload) and drives `SO101-CylReach-Single-v0` with the right controller
(clutched EE pose, analog-trigger gripper, grip-squeeze re-center) through the
`isaacteleop` SO-101 retargeters + damped-least-squares IK into absolute joint
actions. `--null-device` runs a scripted-trajectory dry run headless on GPU
(no headset / XR runtime needed). **Full demo-day setup, connection steps and
troubleshooting: `VR_TELEOP_SETUP.md` at the repo root.**

```bash
OMNI_KIT_ACCEPT_EULA=YES python scripts/isaac_teleop_vr.py --null-device   # no-hardware validation
python scripts/isaac_teleop_vr.py --headless --cloudxr external            # VR demo (2 terminals)
```

## 5. `sim_to_real.py` / `real_to_sim_check.py` — real-arm ↔ sim bridge

Sim→real (stream sim joint commands to the REAL follower arm; sources:
scripted trajectory, DS4 gamepad IK, skrl checkpoint) and a 30-second
leader→sim sanity check. Safety: per-tick jump clamp (`--max-jump-rad`),
torque-limit flag, Ctrl-C ramp-to-home + torque-off, red REAL HARDWARE banner,
full `--dry-run` coverage.

**See `TELEOP_GUIDE.md` at the repo root** — it is the master doc for all four
teleop modes (gamepad→MuJoCo, gamepad→Isaac status, leader→sim, sim→real),
hardware setup, and troubleshooting.

## 5. `isaac_teleop_gamepad.py` — DS4 gamepad Cartesian teleop for the Isaac Lab tasks

```
python scripts/isaac_teleop_gamepad.py --task SO101-CylReach-Single-v0 --null-device --headless  # CI dry-run, no pygame
python scripts/isaac_teleop_gamepad.py --task SO101-PickLift-Single-v0        # live demo (windowed)
python scripts/isaac_teleop_gamepad.py --task SO101-CylGrasp-Dual-v0 --arm both
python scripts/isaac_teleop_gamepad.py --task SO101-CylReach-Dual-v0 --arm left   # Share toggles arms
```

Same DS4 mapping as `teleop_gamepad_ik.py` (left stick = EE x/y, right stick
Y = EE z, right stick X = wrist roll, L1/R1 = pitch, L2/R2 = gripper,
Share = toggle arm, Options = re-center, PS = quit). Architecture mirrors
Isaac Lab's official teleop stack: device -> delta pose ->
`isaaclab.controllers.DifferentialIKController` (position command, relative
mode, damped least squares) -> joint-position targets clipped to soft limits
-> absolute joint actions through `env.step()` (6 per arm, preserve_order).
EE feedback pose and Jacobian come from articulation data exactly like the
built-in `DifferentialInverseKinematicsAction` term; wrist_flex/wrist_roll are
direct joint offsets on top of the IK solution and the gripper is an openness
scalar mapped onto its joint range (position-only IK — same rationale as the
MuJoCo script). Camera sensors are stripped from the scene cfg at load time
(state-based teleop never renders; avoids test.md's headless-RTX flakiness).

`--null-device` replaces the gamepad with a scripted circular EE trajectory
(radius `--circle-radius`, default 3 cm) plus a sinusoidal gripper sweep and
prints mean EE-vs-target tracking error every 10 steps — exit 0 requires all
finite rewards, decreasing error, and final error < 3 cm. Validated on
SO101-CylReach-Single-v0 (~4 mm steady-state lag) and SO101-CylGrasp-Dual-v0
with `--arm left` / `--arm both`. In `--arm both` mode one stick drives both
arms with mirrored Y so they converge/diverge symmetrically on the cylinder.
