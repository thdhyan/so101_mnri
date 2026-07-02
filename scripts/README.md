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

```
MUJOCO_GL=egl python scripts/teleop_gamepad_ik.py --env single
python scripts/teleop_gamepad_ik.py --env dual --arm left --device 0
python scripts/teleop_gamepad_ik.py --dry-run --env single   # headless, no joystick needed
```

Control mapping:

| Input                  | Effect                                  |
|-------------------------|------------------------------------------|
| Left stick (x/y)        | EE x/y velocity, 0.15 m/s at full deflection |
| Right stick vertical    | EE z velocity |
| Right stick horizontal  | wrist_roll rate |
| LB / RB (shoulder btns) | wrist_flex (pitch) rate, opposite directions |
| Triggers (axes 4/5)     | gripper close / open rate |
| Back/Select button      | toggle active arm (dual env only) |

Deadzone 0.15 on all axes. Control loop runs at ~60 Hz; physics is stepped at
the model's native timestep (multiple substeps per control tick).

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
(no window, no joystick) and prints the final EE-target tracking error —
useful for CI verification.

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
