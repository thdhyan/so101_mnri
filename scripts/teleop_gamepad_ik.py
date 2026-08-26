#!/usr/bin/env python
"""Gamepad Cartesian teleop for SO-101 (single or dual arm) via damped-least-squares IK.

Driver: DualShock 4 ("Wireless Controller"), USB or Bluetooth, read through
pygame 2.x's SDL layer. `DS4Gamepad` below prefers SDL2's standardized
game-controller mapping and falls back to the raw DS4-on-Linux layout
(LX=a0 LY=a1 RX=a2 L2=a3 R2=a4 RY=a5, triggers resting at -1).

The IK / target-tracking logic lives in the `IKTeleop` class, decoupled from the
gamepad-reading loop. `IKTeleop.set_ee_target(pos, quat=None)` and
`IKTeleop.set_gripper(openness)` are the two entry points a driver needs to call
each control tick -- the gamepad loop below is one such driver. A future driver
based on mujoco-ar-viewer (https://github.com/Improbable-AI/mujoco-ar-viewer) can
call the exact same two methods instead of reading a joystick.

DS4 control mapping (also printed at startup):
    Left stick X/Y       EE x/y velocity            (LINEAR_SCALE m/s at full)
    Right stick Y        EE z velocity              (LINEAR_SCALE m/s at full)
    Right stick X        DEAD (wrist-roll channel removed)
    Circle (held)        gripper CLOSE rate         (GRIPPER_SCALE rad/s)
    Cross (held)         gripper OPEN rate          (GRIPPER_SCALE rad/s)
    L2 / R2 (analog)     DEAD (trigger-gripper channel removed)
    L1 / R1 (press)      SAFE-HOME: rate-limited ~1.5 s ramp of ALL joints to
                         HOME_POSE, then the IK target re-centers at the home
                         EE pose. Interruptible: e-stop cancels; any stick
                         deflection beyond deadzone cancels mid-ramp.
    Viewer overlay       semi-transparent red sphere (r = 1.2 cm) drawn at the
                         IK EE target every frame; tracks through e-stop and
                         homing so you always see where the arm is headed
    Options              RE-CENTER: snap IK target to current EE pose
                         (also resumes from e-stop)  <-- safety: use liberally
    Share                toggle active arm (dual env only; both arms stay simulated)
    PS button            E-STOP latch: all arms freeze at current pose, neutral
                         gripper; press Options to re-center and resume
                         (also cancels an in-progress safe-home ramp)
    Square               safe quit: freeze at current pose, neutral gripper,
                         settle 0.5 s, exit cleanly

DS4 setup (demo day):
    1. Pair: hold SHARE + PS until the light bar double-flashes, then connect
       "Wireless Controller" in your Bluetooth manager -- or just plug USB.
    2. User must read /dev/input: `sudo usermod -aG input $USER`, log out/in,
       verify with `groups`.
    3. Verify pairing + axes/buttons live in 5 seconds:
           python scripts/check_gamepad.py
    4. No device listed? `sudo modprobe joydev` and replug/re-pair.

Usage:
    MUJOCO_GL=egl python scripts/teleop_gamepad_ik.py --env single
    MUJOCO_GL=egl python scripts/teleop_gamepad_ik.py --env dual --arm left
    MUJOCO_GL=egl python scripts/teleop_gamepad_ik.py --env single --dry-run
        (dry-run also replays a scripted mock-button sequence -- Circle/Cross
        gripper rates, right-stick-X dead, L1 homing ramp + interrupts)

Limitation -- `--arm both` intentionally NOT offered: the IK layer trivially
supports two independent IKTeleop instances (dual mode already solves both
arms every tick), but a single DS4 has only 4 stick DoF while driving two
arms' XY+Z simultaneously needs 6; the supported dual workflow is Share-
toggle between arms. Adding 'both' is a control-mapping problem, not an IK
architecture problem.
"""
import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import mujoco

# Joystick input needs no display (see check_gamepad.py); the MuJoCo viewer
# uses GLFW directly and is unaffected. Keeps --dry-run/mock tests headless.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from _env_utils import scene_path, joint_names, gripper_site_name, HOME_POSE, resolve_actuator_ids, resolve_joint_ids

# Canonical SO-101 joint order (matches lerobot motor names) — index-aligned
# with joint_ids/actuator_ids in IKTeleop.
JOINT_SUFFIXES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

DEADZONE = 0.15
TRIGGER_DEADZONE = 0.08  # applied after [-1,1] -> [0,1] normalization
LINEAR_SCALE = 0.15   # m/s at full stick deflection
GRIPPER_SCALE = 1.5    # rad/s while Circle (close) / Cross (open) is held
HOMING_DURATION = 1.5  # s for the L1/R1 safe-home ramp across the full travel
CONTROL_HZ = 60.0
NEUTRAL_GRIPPER = 0.5  # openness used by e-stop / safe shutdown


def apply_deadzone(x, dz=DEADZONE):
    # clamp first: some DS4 units report wildly-out-of-range axes through
    # pygame's sdl2-controller layer (observed +-1000s); commands must stay bounded
    x = float(np.clip(x, -1.0, 1.0))
    if abs(x) < dz:
        return 0.0
    # rescale so output is continuous from 0 at the deadzone edge
    sign = 1.0 if x > 0 else -1.0
    return sign * (abs(x) - dz) / (1.0 - dz)


# ---------------------------------------------------------------------------
# DualShock 4 reader
#
# Raw SDL-joystick layout for a DS4 on Linux (USB or Bluetooth, hid-play /
# hid-sony kernels, matches SDL's bundled gamecontrollerdb entry for Sony
# pads -- note the classic quirk: triggers occupy axes 3/4 and right-stick-Y
# is axis 5):
#     axes    0=LX  1=LY  2=RX  3=L2  4=R2  5=RY     (L2/R2 rest at -1)
#     buttons 0=square 1=cross 2=circle 3=triangle 4=L1 5=R1
#             8=share 9=options 10=L3 11=R3 17=PS(guide); dpad -> hat 0
# When SDL recognizes the pad as a game controller we use the standardized
# pygame._sdl2.controller API instead (identical logical mapping, immune to
# firmware/kernel axis-order differences).
# ---------------------------------------------------------------------------
RAW_DS4_AXES = {"lx": 0, "ly": 1, "rx": 2, "l2": 3, "r2": 4, "ry": 5}
RAW_DS4_BUTTONS = {
    "square": 0, "cross": 1, "circle": 2, "triangle": 3,
    "l1": 4, "r1": 5, "share": 8, "options": 9, "l3": 10, "r3": 11,
}
ALL_BUTTON_NAMES = ["square", "cross", "circle", "triangle",
                    "l1", "r1", "share", "options", "ps", "l3", "r3"]

NO_GAMEPAD_HELP = """No gamepad detected. Demo-day checklist:
  1. Plug the DS4 in over USB, or pair Bluetooth: hold SHARE + PS until the
     light bar double-flashes, connect "Wireless Controller".
  2. Permission: you must be in the 'input' group to read /dev/input --
     `groups` to check, `sudo usermod -aG input $USER` + re-login to fix.
  3. Kernel driver: `sudo modprobe joydev`, then replug / re-pair.
  4. Diagnose live with: python scripts/check_gamepad.py"""


@dataclass
class PadState:
    left_stick: np.ndarray   # (2,) deadzoned, [-1, 1]
    right_stick: np.ndarray  # (2,) deadzoned, [-1, 1]
    l2: float                # [0, 1]
    r2: float                # [0, 1]
    buttons: dict = field(default_factory=dict)  # name -> bool (incl. "ps")
    dpad: tuple = (0, 0)     # hat value (-1..1 per component)


def ds4_trigger_norm(raw):
    """Raw DS4 joystick trigger rests at -1 and pulls to +1 -> [0, 1]."""
    return float(np.clip((raw + 1.0) / 2.0, 0.0, 1.0))


def _trigger_dz(v):
    """Small deadzone on a normalized [0, 1] trigger value."""
    return 0.0 if v < TRIGGER_DEADZONE else float(v)


class DS4Gamepad:
    """Reads one DualShock 4 ('Wireless Controller') via pygame 2.x SDL.

    Primary path: SDL2 game-controller mapping (pygame._sdl2.controller) --
    consistent across USB/Bluetooth regardless of kernel quirks.
    Fallback path: raw DS4 indices (RAW_DS4_AXES / RAW_DS4_BUTTONS above).
    """

    def __init__(self, device_index=0):
        import pygame

        self._pygame = pygame
        n = pygame.joystick.get_count()
        if n == 0:
            raise RuntimeError(NO_GAMEPAD_HELP)
        idx = min(max(device_index, 0), n - 1)
        self.joy = pygame.joystick.Joystick(idx)
        self.joy.init()
        self.name = self.joy.get_name()
        self._mode = "sdl2-controller"
        self._ctl = None
        try:
            from pygame._sdl2 import controller as sdl2_controller
            sdl2_controller.init()
            try:
                self._ctl = sdl2_controller.Controller.from_joystick(self.joy)
            except Exception:
                self._ctl = None
                raise
        except Exception:
            self._ctl = None
            self._mode = "raw-ds4"
            if self.joy.get_numaxes() != len(RAW_DS4_AXES):
                print(f"[gamepad] WARNING: '{self.name}' exposes "
                      f"{self.joy.get_numaxes()} axes (expected 6 for a DS4); "
                      f"trigger/right-stick indices may be wrong.")
            if "4c05" not in (self.joy.get_guid() or ""):
                print(f"[gamepad] WARNING: GUID does not look like a Sony pad; "
                      f"raw button indices may differ.")

    @property
    def mode(self):
        return self._mode

    @classmethod
    def detect(cls, device_index=0):
        """Return a connected DS4Gamepad or None (never raises)."""
        import pygame
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            return None
        try:
            return cls(device_index)
        except Exception:
            return None

    @staticmethod
    def _btn_const(pygame, *names):
        for n in names:
            c = getattr(pygame, n, None)
            if c is not None:
                return c
        return None

    def poll(self):
        """Pump SDL events and return a deadzoned PadState snapshot."""
        pygame = self._pygame
        pygame.event.pump()

        if self._ctl is not None:
            A = pygame.CONTROLLER_AXIS_LEFTX, pygame.CONTROLLER_AXIS_LEFTY, \
                pygame.CONTROLLER_AXIS_RIGHTX, pygame.CONTROLLER_AXIS_RIGHTY, \
                pygame.CONTROLLER_AXIS_TRIGGERLEFT, pygame.CONTROLLER_AXIS_TRIGGERRIGHT
            lx, ly, rx, ry, lt, rt = (self._ctl.get_axis(a) for a in A)
            # SDL game-controller trigger axes already span [0, 1] (rest = 0)
            l2, r2 = _trigger_dz(min(max(lt, 0.0), 1.0)), _trigger_dz(min(max(rt, 0.0), 1.0))

            def held(*names):
                c = self._btn_const(pygame, *names)
                return bool(self._ctl.get_button(c)) if c is not None else False

            # NOTE: pygame exposes the classic SDL button names (A/B/X/Y,
            # BACK/START/GUIDE); newer pygame may add SHARE/MISC1, so each
            # lookup tries modern names first and falls back to the SDL
            # standard mapping for a DualShock 4:
            #   cross->A, circle->B, square->X, triangle->Y,
            #   share->BACK, options->START, PS->GUIDE (or MISC1).
            buttons = {
                "square": held("CONTROLLER_BUTTON_SQUARE", "CONTROLLER_BUTTON_X"),
                "cross": held("CONTROLLER_BUTTON_CROSS", "CONTROLLER_BUTTON_A"),
                "circle": held("CONTROLLER_BUTTON_CIRCLE", "CONTROLLER_BUTTON_B"),
                "triangle": held("CONTROLLER_BUTTON_TRIANGLE", "CONTROLLER_BUTTON_Y"),
                "l1": held("CONTROLLER_BUTTON_LEFTSHOULDER"),
                "r1": held("CONTROLLER_BUTTON_RIGHTSHOULDER"),
                "share": held("CONTROLLER_BUTTON_SHARE", "CONTROLLER_BUTTON_BACK"),
                "options": held("CONTROLLER_BUTTON_START"),
                "ps": held("CONTROLLER_BUTTON_MISC1", "CONTROLLER_BUTTON_GUIDE"),
                "l3": held("CONTROLLER_BUTTON_LEFTSTICK"),
                "r3": held("CONTROLLER_BUTTON_RIGHTSTICK"),
            }
        else:
            ax = self.joy.get_numaxes()
            lx = self.joy.get_axis(RAW_DS4_AXES["lx"])
            ly = self.joy.get_axis(RAW_DS4_AXES["ly"])
            rx = self.joy.get_axis(RAW_DS4_AXES["rx"]) if ax > 2 else 0.0
            ry = self.joy.get_axis(RAW_DS4_AXES["ry"]) if ax > 5 else (
                self.joy.get_axis(3) if ax > 3 else 0.0)
            l2_raw = self.joy.get_axis(RAW_DS4_AXES["l2"]) if ax > 3 else -1.0
            r2_raw = self.joy.get_axis(RAW_DS4_AXES["r2"]) if ax > 4 else -1.0
            l2, r2 = _trigger_dz(ds4_trigger_norm(l2_raw)), _trigger_dz(ds4_trigger_norm(r2_raw))

            nb = self.joy.get_numbuttons()

            def raw(i):
                return bool(self.joy.get_button(i)) if i < nb else False

            # dpad-as-buttons firmwares repurpose 12-16, so only trust 13 for
            # PS when a hat exists (i.e. 12-16 are genuinely unused).
            ps_idx_candidates = (17, 13) if self.joy.get_numhats() > 0 else (17,)
            buttons = {name: raw(i) for name, i in RAW_DS4_BUTTONS.items()}
            buttons["ps"] = any(raw(i) for i in ps_idx_candidates)

        return PadState(
            left_stick=np.array([apply_deadzone(lx), apply_deadzone(ly)]),
            right_stick=np.array([apply_deadzone(rx), apply_deadzone(ry)]),
            l2=l2,
            r2=r2,
            buttons=buttons,
            dpad=self.joy.get_hat(0) if self.joy.get_numhats() > 0 else (0, 0),
        )


def print_mapping_table():
    print("""DS4 mapping:
  left stick X/Y .... EE x/y vel      right stick Y ..... EE z vel
  right stick X ..... DEAD           L2 / R2 ........... DEAD
  CIRCLE (hold) ..... gripper CLOSE  CROSS (hold) ...... gripper OPEN  (1.5 rad/s)
  L1 / R1 (press) ... SAFE-HOME: ~1.5 s ramp of all joints to HOME_POSE,
                      then IK target re-centers at home EE pose
                      (e-stop or stick deflection cancels mid-ramp)
  OPTIONS ........... re-center IK target to current EE pose (safety)
  SHARE ............. toggle active arm (dual env)
  PS ................ E-STOP latch (freeze + neutral gripper; Options resumes;
                      cancels homing)
  SQUARE ............ safe quit (freeze + neutral gripper + settle)
  viewer ............ red sphere @ IK target (tracks through e-stop/homing)""")


class IKTeleop:
    """Damped-least-squares Cartesian IK controller for one SO-101 arm.

    Owns a mujoco model/data pair (or a view into a shared dual-arm model),
    tracks a Cartesian EE target + gripper openness, and solves for joint
    position-actuator commands each `step()` call.
    """

    def __init__(self, model, data, joint_ids, actuator_ids, site_id,
                 home_pose=HOME_POSE, damping=1e-2, use_orientation=False):
        self.model = model
        self.data = data
        self.joint_ids = joint_ids          # qpos/qvel address indices for this arm's joints (assumes 1-dof hinge joints)
        self.actuator_ids = actuator_ids
        self.site_id = site_id
        self.home_pose = np.asarray(home_pose, dtype=float)
        self.damping = damping
        self.use_orientation = use_orientation
        self.n = len(joint_ids)

        self.qpos_adr = np.array([model.jnt_qposadr[j] for j in joint_ids])
        self.dof_adr = np.array([model.jnt_dofadr[j] for j in joint_ids])
        self.jnt_range = np.array([model.jnt_range[j] for j in joint_ids])
        self.ctrl_range = np.array([model.actuator_ctrlrange[a] for a in actuator_ids])

        # gripper is last joint in the arm's chain by contract
        self._gripper_local_idx = self.n - 1

        init_pos = self.data.xpos[model.site_bodyid[site_id]] if False else None
        self.ee_target = self._current_ee_pos()
        self.ee_target_quat = None
        self.gripper_openness = 0.0  # 0=closed .. 1=open

        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    # ---- public interface (this is what an external driver, e.g. AR viewer, calls) ----
    def set_ee_target(self, pos, quat=None):
        self.ee_target = np.asarray(pos, dtype=float).copy()
        if quat is not None:
            self.ee_target_quat = np.asarray(quat, dtype=float).copy()

    def set_gripper(self, openness):
        self.gripper_openness = float(np.clip(openness, 0.0, 1.0))

    def nudge_ee_target(self, delta_pos):
        self.ee_target = self.ee_target + np.asarray(delta_pos, dtype=float)

    def recenter_target(self):
        """Safety: snap EE target to current EE pose and sync gripper openness
        to the gripper's actual position (used by Options re-center and e-stop).
        Zeroes commanded motion without moving the arm."""
        self.ee_target = self._current_ee_pos()
        self.ee_target_quat = None
        lo, hi = self.jnt_range[self._gripper_local_idx]
        gq = self.data.qpos[self.qpos_adr[self._gripper_local_idx]]
        self.gripper_openness = float(np.clip((gq - lo) / max(hi - lo, 1e-9), 0.0, 1.0))

    # ---- internals ----
    def _current_ee_pos(self):
        return self.data.site_xpos[self.site_id].copy()

    def _current_qpos(self):
        return self.data.qpos[self.qpos_adr].copy()

    def solve_step(self):
        """Run one damped-least-squares IK step and write to data.ctrl."""
        mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.site_id)
        # columns of J for just this arm's dofs
        Jp = self._jacp[:, self.dof_adr]  # (3, n)

        cur_pos = self._current_ee_pos()
        pos_err = self.ee_target - cur_pos

        if self.use_orientation and self.ee_target_quat is not None:
            Jr = self._jacr[:, self.dof_adr]
            cur_quat = np.zeros(4)
            mujoco.mju_mat2Quat(cur_quat, self.data.site_xmat[self.site_id])
            quat_err = np.zeros(3)
            neg_cur = np.zeros(4)
            mujoco.mju_negQuat(neg_cur, cur_quat)
            diff_quat = np.zeros(4)
            mujoco.mju_mulQuat(diff_quat, self.ee_target_quat, neg_cur)
            mujoco.mju_quat2Vel(quat_err, diff_quat, 1.0)
            J = np.vstack([Jp, Jr])
            err = np.concatenate([pos_err, quat_err])
        else:
            J = Jp
            err = pos_err

        # damped least squares: dq = J^T (J J^T + lambda^2 I)^-1 err
        m = J.shape[0]
        lam2 = self.damping ** 2
        JJt = J @ J.T + lam2 * np.eye(m)
        dq = J.T @ np.linalg.solve(JJt, err)

        # nullspace bias toward home pose (excluding gripper dof)
        cur_q = self._current_qpos()
        q_bias = np.zeros(self.n)
        q_bias[:-1] = 0.3 * (self.home_pose[:-1] - cur_q[:-1])
        Jpinv = J.T @ np.linalg.solve(JJt, np.eye(m))
        N = np.eye(self.n) - Jpinv @ J
        dq = dq + N @ q_bias

        target_q = cur_q + dq
        # gripper is driven directly by openness, not by IK
        lo, hi = self.jnt_range[self._gripper_local_idx]
        target_q[self._gripper_local_idx] = lo + self.gripper_openness * (hi - lo)

        target_q = np.clip(target_q, self.jnt_range[:, 0], self.jnt_range[:, 1])
        ctrl = np.clip(target_q, self.ctrl_range[:, 0], self.ctrl_range[:, 1])

        self.data.ctrl[self.actuator_ids] = ctrl
        return cur_pos

    def ee_error(self):
        return float(np.linalg.norm(self.ee_target - self._current_ee_pos()))


def build_ik(env, arm):
    path = scene_path(env)
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)

    jnames = joint_names(env, arm)
    site_name = gripper_site_name(env, arm)
    joint_ids = resolve_joint_ids(model, jnames)
    actuator_ids = resolve_actuator_ids(model, jnames)
    site_id = model.site(site_name).id

    # initialize qpos at home pose for this arm so IK starts near a sane config
    qpos_adr = np.array([model.jnt_qposadr[j] for j in joint_ids])
    data.qpos[qpos_adr] = HOME_POSE
    data.ctrl[actuator_ids] = HOME_POSE
    mujoco.mj_forward(model, data)

    ik = IKTeleop(model, data, joint_ids, actuator_ids, site_id)
    return model, data, ik


def self_test_pad_math():
    """Offline sanity of deadzone/trigger math -- no hardware needed."""
    assert apply_deadzone(0.0) == 0.0
    assert abs(apply_deadzone(DEADZONE)) < 1e-12
    assert abs(apply_deadzone(-DEADZONE)) < 1e-12
    for x in np.linspace(-1, 1, 201):
        y = apply_deadzone(x)
        assert -1.0 <= y <= 1.0
        if abs(x) >= DEADZONE:
            assert (y > 0) == (x > 0) or y == 0.0
    grid = [apply_deadzone(x) for x in np.linspace(DEADZONE, 1.0, 50)]
    assert all(b > a for a, b in zip(grid, grid[1:])), "deadzone must be monotonic"
    assert ds4_trigger_norm(-1.0) == 0.0 and ds4_trigger_norm(1.0) == 1.0
    assert ds4_trigger_norm(0.0) == 0.5
    print("[dry-run] DS4 mapping math self-test OK")


def probe_gamepad():
    """Exercise DS4Gamepad detection; prints guidance when no device is present.

    Never fails the dry-run -- hardware absence is an expected condition here.
    """
    import pygame
    pygame.init()
    pygame.joystick.init()
    pad = DS4Gamepad.detect()
    if pad is None:
        print("[dry-run] no gamepad detected (fine for a hardware-free dry-run)")
        for line in NO_GAMEPAD_HELP.splitlines():
            print(f"[dry-run]   {line}")
        return
    try:
        st = pad.poll()
        print(f"[dry-run] gamepad detected: {pad.name} [{pad.mode}]")
        print(f"[dry-run]   sticks=({st.left_stick[0]:+.2f},{st.left_stick[1]:+.2f}) "
              f"({st.right_stick[0]:+.2f},{st.right_stick[1]:+.2f}) "
              f"L2={st.l2:.2f} R2={st.r2:.2f} dpad={st.dpad}")
        held = [n for n, v in st.buttons.items() if v] or ["none"]
        print(f"[dry-run]   buttons held: {', '.join(held)}")
    except Exception as e:
        print(f"[dry-run] gamepad present but unreadable ({e}); see checklist above")


class _MockPad:
    """Scripted DS4Gamepad stand-in: replays a fixed PadState frame list."""

    def __init__(self, frames):
        self._frames = list(frames)
        self.name = "MockDS4 (scripted)"
        self.mode = "dry-run-script"

    def poll(self):
        if len(self._frames) > 1:
            return self._frames.pop(0)
        return self._frames[0]


def _mock_frame(left=(0.0, 0.0), right=(0.0, 0.0), buttons=None):
    buttons = buttons or {}
    return PadState(
        left_stick=np.array([apply_deadzone(v) for v in left]),
        right_stick=np.array([apply_deadzone(v) for v in right]),
        l2=0.0, r2=0.0,
        buttons={n: bool(buttons.get(n, False)) for n in ALL_BUTTON_NAMES},
        dpad=(0, 0),
    )


def mock_button_test(env, arm="left", steps=100):
    """Hardware-free unit test of the new mapping: drives the REAL gamepad
    loop (run_gamepad) with a scripted pad and asserts gripper rates, the
    dead right-stick-X channel, homing completion, and both interrupts."""
    frames = []
    marks = {}

    def add(n, label=None, **kw):
        if label:
            marks[label] = len(frames)
        frames.extend(_mock_frame(**kw) for _ in range(n))
        return marks[label]

    add(10, "settle")
    add(30, "open", buttons={"cross": True})       # Cross -> openness rises
    add(5, "post_open")
    add(30, "close", buttons={"circle": True})     # Circle -> openness falls
    add(5, "post_close")
    add(30, "rx", right=(1.0, 0.0))                # right stick X must be dead
    add(10, "pre_home")
    add(1, "home1_edge", buttons={"l1": True})     # L1 press -> safe-home
    add(160, "home1_ramp")                         # ~2.7 s of runway > 1.5 s ramp
    add(30, "away", left=(1.0, 0.0))               # drag EE off home
    add(1, "home2_edge", buttons={"l1": True})
    add(20, "home2_mid")                           # mid-ramp...
    add(1, "estop_edge", buttons={"ps": True})     # ...e-stop cancels homing
    add(8, "estop_hold")
    add(1, "resume_edge", buttons={"options": True})
    add(4, "post_resume")
    add(1, "home3_edge", buttons={"l1": True})
    add(15, "home3_mid")                           # mid-ramp...
    add(1, "stick_cancel", left=(1.0, 0.0))        # ...stick deflection cancels
    add(20, "after_cancel")
    add(1, "quit_edge", buttons={"square": True})

    history = []
    run_gamepad(env, arm, device_index=0, no_viewer=True,
                pad=_MockPad(frames), tick_hook=history.append)

    def avg_grip(a, b):
        return float(np.mean([history[i]["grip"] for i in range(a, b + 1)]))

    h = history
    assert len(h) >= marks["quit_edge"], f"loop ended early ({len(h)} ticks)"

    g0 = avg_grip(marks["settle"] + 7, marks["settle"] + 9)
    g1 = avg_grip(marks["open"] + 27, marks["open"] + 29)
    assert g1 > g0 + 0.03, f"Cross must open gripper ({g0:.3f} -> {g1:.3f})"
    print(f"[mock] Cross held 30 ticks: openness {g0:.3f} -> {g1:.3f} (increased)")

    g2 = avg_grip(marks["close"] + 27, marks["close"] + 29)
    gp = avg_grip(marks["post_open"], marks["post_open"] + 2)
    assert g2 < gp - 0.03, f"Circle must close gripper ({gp:.3f} -> {g2:.3f})"
    print(f"[mock] Circle held 30 ticks: openness {gp:.3f} -> {g2:.3f} (decreased)")

    i_rx = marks["rx"]
    ee_a, ee_b = h[i_rx]["ee"], h[i_rx + 29]["ee"]
    assert np.allclose(ee_a[:2], ee_b[:2], atol=1e-12), \
        f"right stick X moved XY target: {ee_a} -> {ee_b}"
    roll_a = h[i_rx]["qpos"][4]
    roll_b = h[i_rx + 29]["qpos"][4]
    assert abs(roll_b - roll_a) < 1e-3, f"right stick X rolled wrist ({roll_a}->{roll_b})"
    assert np.allclose(h[i_rx]["ee"][2], h[i_rx + 29]["ee"][2], atol=1e-12), \
        "right stick X leaked into Z"
    print(f"[mock] right-stick X full deflection 30 ticks: "
          f"XY d={np.linalg.norm(ee_b[:2]-ee_a[:2]):.1e} m, "
          f"roll d={abs(roll_b-roll_a):.1e} rad (dead)")

    i0 = next(i for i in range(marks["home1_edge"], len(h)) if h[i]["homing"])
    done = [i for i in range(i0, min(i0 + 200, len(h)))
            if not h[i]["homing"]
            and np.allclose(h[i]["qpos"], HOME_POSE, atol=0.05)
            and np.allclose(h[i]["ctrl"], HOME_POSE, atol=0.05)]
    assert done, "L1 homing ramp never reached HOME_POSE"
    j = done[0]
    assert h[j]["ee_err"] < 5e-3, \
        f"IK target not re-centered after homing (ee_err={h[j]['ee_err']:.2e} m)"
    print(f"[mock] L1 -> homing reached HOME_POSE at tick {j - i0} "
          f"({(j - i0) / CONTROL_HZ:.2f} s) and cleared; ctrl == HOME_POSE, "
          f"target re-centered (ee_err={h[j]['ee_err']:.1e} m)")

    q_away = h[marks["away"] + 29]["qpos"]
    assert np.max(np.abs(q_away - HOME_POSE)) > 0.02, "arm failed to leave home"

    k = marks["estop_edge"]
    assert not h[k]["homing"] and h[k]["estop"], "e-stop did not cancel homing"
    assert not np.allclose(h[k]["qpos"], HOME_POSE, atol=0.02), \
        "homing should have been cancelled mid-ramp"
    print(f"[mock] PS mid-ramp: homing cleared, estop latched, "
          f"max|q-HOME|={np.max(np.abs(h[k]['qpos'] - HOME_POSE)):.3f} rad (cancelled)")

    r = marks["resume_edge"]
    assert not h[r]["estop"], "Options did not clear e-stop"

    s = marks["stick_cancel"]
    assert not h[s]["homing"], "stick deflection did not cancel homing"
    drift = np.max(np.abs(h[s + 19]["qpos"] - h[s + 2]["qpos"]))
    assert drift < 0.05, f"ramp continued after cancel (drift {drift:.3f} rad)"
    print(f"[mock] stick deflection mid-ramp: homing cleared; "
          f"17-tick joint drift after cancel = {drift:.1e} rad (held)")

    print("[mock] mock-button mapping test OK")


def run_dry_run(env, arm, steps=100):
    self_test_pad_math()
    model, data, ik = build_ik(env, arm)
    center = ik._current_ee_pos().copy()
    radius = 0.05
    dt_ctrl = 1.0 / CONTROL_HZ
    substeps = max(1, int(round(dt_ctrl / model.opt.timestep)))

    for i in range(steps):
        theta = 2 * np.pi * i / steps
        target = center + np.array([radius * np.cos(theta), radius * np.sin(theta), 0.0])
        ik.set_ee_target(target)
        ik.set_gripper(0.5 * (1 + np.sin(theta)))
        ik.solve_step()
        for _ in range(substeps):
            mujoco.mj_step(model, data)

    err = ik.ee_error()
    print(f"[dry-run] env={env} arm={arm} steps={steps} final EE-target error = {err*100:.3f} cm")
    if err > 0.05:
        print("[dry-run] WARNING: error exceeds 5 cm threshold")
        sys.exit(1)

    # safety-feature smoke test: recenter must zero EE target motion without moving
    pre_q = ik._current_qpos().copy()
    ik.recenter_target()
    assert np.allclose(ik.ee_target, ik._current_ee_pos())
    assert pre_q.shape == ik._current_qpos().shape
    probe_gamepad()

    # scripted mock-button test of the new DS4 mapping (no hardware needed)
    mock_button_test(env, arm=arm if arm is not None else "left")

    print("[dry-run] OK")


def run_gamepad(env, arm, device_index, no_viewer=False, mirror=None,
                pad=None, tick_hook=None):
    import pygame

    pygame.init()
    model, data, ik = build_ik(env, arm)

    active_arm = arm  # only meaningful for dual; toggled via Share
    ik_by_arm = {arm: ik}
    if env == "dual":
        other = "right" if arm == "left" else "left"
        jnames = joint_names(env, other)
        site_name = gripper_site_name(env, other)
        joint_ids = resolve_joint_ids(model, jnames)
        actuator_ids = resolve_actuator_ids(model, jnames)
        site_id = model.site(site_name).id
        qpos_adr = np.array([model.jnt_qposadr[j] for j in joint_ids])
        data.qpos[qpos_adr] = HOME_POSE
        data.ctrl[actuator_ids] = HOME_POSE
        ik_by_arm[other] = IKTeleop(model, data, joint_ids, actuator_ids, site_id)
        mujoco.mj_forward(model, data)

    if pad is None:
        pad = DS4Gamepad.detect(device_index)
    if pad is None:
        print(NO_GAMEPAD_HELP)
        sys.exit(1)
    print(f"Using gamepad: {pad.name} [{pad.mode}]")
    print_mapping_table()

    dt_ctrl = 1.0 / CONTROL_HZ
    substeps = max(1, int(round(dt_ctrl / model.opt.timestep)))

    def freeze_and_settle(label):
        """Zero velocity at the current pose, neutral gripper, brief settle."""
        for a in ik_by_arm.values():
            a.set_ee_target(a._current_ee_pos())
            a.set_gripper(NEUTRAL_GRIPPER)
        for a in ik_by_arm.values():
            a.solve_step()
        for _ in range(substeps * 30):  # ~0.5 s
            mujoco.mj_step(model, data)
        print(f"[{label}] arms frozen at current pose, gripper neutral")

    viewer = None
    if not no_viewer:
        # NOTE: must use the "as" form -- plain `import mujoco.viewer` would
        # bind `mujoco` as a function-local name and break earlier uses.
        import mujoco.viewer as mujoco_viewer
        viewer = mujoco_viewer.launch_passive(model, data)

    prev_buttons = {}
    estop = False
    homing = False          # safe-home ramp in progress (active arm)
    homing_t0 = 0.0
    homing_start_q = None
    running = True
    next_status = time.time()

    try:
        while running:
            t0 = time.time()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            try:
                st = pad.poll()
            except pygame.error as e:
                print(f"\n[gamepad] controller lost ({e})")
                running = False
                break

            # rising edges
            edges = {}
            for name, now in st.buttons.items():
                was = prev_buttons.get(name, False)
                edges[name] = now and not was
                prev_buttons[name] = now

            if edges.get("square"):
                running = False
                break

            if edges.get("ps"):
                estop = not estop
                if estop and homing:
                    homing = False  # e-stop interrupts the safe-home ramp
                print(f"[ESTOP] {'LATCHED' if estop else 'released'} "
                      f"(freeze + neutral gripper; OPTIONS re-centers and resumes)")

            if edges.get("options"):
                for a in ik_by_arm.values():
                    a.recenter_target()
                estop = False
                print("[recenter] IK target snapped to current EE pose")

            if env == "dual" and edges.get("share"):
                active_arm = "right" if active_arm == "left" else "left"
                homing = False  # ramp state belongs to the previous arm
                print(f"Active arm -> {active_arm}")

            cur_ik = ik_by_arm[active_arm]

            # safe-home trigger: rising edge on L1 OR R1
            if (edges.get("l1") or edges.get("r1")):
                if estop:
                    print("[safe-home] ignored while e-stopped (OPTIONS resumes first)")
                elif not homing:
                    homing = True
                    homing_t0 = time.time()
                    homing_start_q = cur_ik._current_qpos().copy()
                    print(f"[safe-home] ramping all joints to HOME_POSE "
                          f"(~{HOMING_DURATION:.1f} s); stick deflection or PS cancels")

            if homing and not estop:
                # interruptible: any stick deflection beyond deadzone aborts
                if st.left_stick.any() or abs(st.right_stick[1]) > 0.0:
                    homing = False
                    cur_ik.recenter_target()
                    print("[safe-home] cancelled by stick input; target re-centered")
                else:
                    alpha = min((time.time() - homing_t0) / HOMING_DURATION, 1.0)
                    target_q = homing_start_q + (HOME_POSE - homing_start_q) * alpha
                    target_q = np.clip(target_q, cur_ik.jnt_range[:, 0], cur_ik.jnt_range[:, 1])
                    data.ctrl[cur_ik.actuator_ids] = np.clip(
                        target_q, cur_ik.ctrl_range[:, 0], cur_ik.ctrl_range[:, 1])
                    # keep the IK target glued to the EE so the debug sphere
                    # tracks the arm throughout the ramp
                    cur_ik.set_ee_target(cur_ik._current_ee_pos())
                    if alpha >= 1.0:
                        homing = False
                        cur_ik.recenter_target()
                        print("[safe-home] reached HOME_POSE; IK target re-centered")
            elif estop:
                for a in ik_by_arm.values():
                    a.set_ee_target(a._current_ee_pos())
                    a.set_gripper(NEUTRAL_GRIPPER)
            else:
                lx, ly = st.left_stick
                _, ry = st.right_stick  # right-stick X is DEAD (wrist roll removed)

                vx = -ly * LINEAR_SCALE
                vy = -lx * LINEAR_SCALE
                vz = -ry * LINEAR_SCALE

                new_target = cur_ik.ee_target + np.array([vx, vy, vz]) * dt_ctrl
                cur_ik.set_ee_target(new_target)

                # Cross held opens, Circle held closes, at GRIPPER_SCALE rad/s
                # (converted to the normalized openness the IK layer uses)
                lo, hi = cur_ik.jnt_range[cur_ik._gripper_local_idx]
                grip_rate = (float(st.buttons.get("cross", False))
                             - float(st.buttons.get("circle", False))) * GRIPPER_SCALE
                d_openness = grip_rate * dt_ctrl / max(hi - lo, 1e-9)
                cur_ik.set_gripper(cur_ik.gripper_openness + d_openness)

            for a in ik_by_arm.values():
                if homing and a is cur_ik:
                    continue  # ctrl is driven directly by the ramp this tick
                a.solve_step()
            # wrist_flex / wrist_roll now have no live input channel (right-X
            # and L1/R1 pitch removed); pin their ctrl to the current qpos so
            # IK nullspace/home bias cannot drift them (also while e-stopped).
            # Skipped during homing, which writes all six ctrls itself.
            if not homing:
                qpos_adr = cur_ik.qpos_adr
                data.ctrl[cur_ik.actuator_ids[4]] = data.qpos[qpos_adr[4]]
                data.ctrl[cur_ik.actuator_ids[3]] = data.qpos[qpos_adr[3]]

            # mirror the active arm's joint targets to a real follower (sim + real together)
            if mirror is not None:
                mirror.send({
                    f"{name}.pos": math.degrees(float(data.ctrl[aid]))
                    for name, aid in zip(JOINT_SUFFIXES, cur_ik.actuator_ids)
                })

            for _ in range(substeps):
                mujoco.mj_step(model, data)

            if tick_hook is not None:
                tick_hook({
                    "active": active_arm, "estop": estop, "homing": homing,
                    "grip": cur_ik.gripper_openness,
                    "ee": cur_ik.ee_target.copy(),
                    "qpos": data.qpos[cur_ik.qpos_adr].copy(),
                    "ctrl": data.ctrl[cur_ik.actuator_ids].copy(),
                    "ee_err": cur_ik.ee_error(),
                })

            if viewer is not None:
                _draw_target_marker(viewer, cur_ik.ee_target)
                viewer.sync()
                if not viewer.is_running():
                    running = False

            if time.time() >= next_status:
                next_status += 1.0
                held = [n for n in ("ps", "l1", "r1") if st.buttons.get(n)]
                print(f"[{time.strftime('%H:%M:%S')}] arm={active_arm} "
                      f"ee_err={cur_ik.ee_error()*100:.2f}cm "
                      f"grip={cur_ik.gripper_openness:.2f} "
                      f"{'ESTOP ' if estop else ''}{'HOMING ' if homing else ''}"
                      f"{','.join(held)}")

            elapsed = time.time() - t0
            time.sleep(max(0.0, dt_ctrl - elapsed))
    finally:
        freeze_and_settle("shutdown")
        if mirror is not None:
            try:
                mirror.robot.disconnect()
            except Exception:
                pass
        if viewer is not None:
            viewer.close()


def _draw_target_marker(viewer, pos):
    """Semi-transparent red debug sphere at the IK EE target (r = 1.2 cm),
    drawn through the passive viewer's user_scn each frame."""
    scn = viewer.user_scn
    scn.ngeom = 0
    mujoco.mjv_initGeom(
        scn.geoms[0],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.012, 0, 0],
        pos=np.asarray(pos, dtype=float),
        mat=np.eye(3).flatten(),
        rgba=np.array([1.0, 0.15, 0.15, 0.6], dtype=float),
    )
    scn.ngeom = 1


def main():
    parser = argparse.ArgumentParser(description="Gamepad Cartesian IK teleop for SO-101")
    parser.add_argument("--env", choices=["single", "dual"], default="single")
    parser.add_argument("--arm", choices=["left", "right"], default="left",
                         help="Initial active arm (dual env only). NOTE: 'both' is "
                              "deliberately unsupported -- one DS4 has 4 stick DoF, "
                              "two simultaneous XY+Z streams need 6; use Share to toggle.")
    parser.add_argument("--device", type=int, default=0, help="pygame joystick index")
    parser.add_argument("--dry-run", action="store_true",
                         help="Run headless IK loop with a scripted circular target, 100 steps")
    parser.add_argument("--no-viewer", action="store_true", help="Disable passive viewer window")
    parser.add_argument("--real-follower-port", default=None,
                         help="Mirror IK joint targets to a real SO-101 follower on this port "
                              "(sim + real move together; needs lerobot calibration for the follower)")
    parser.add_argument("--real-follower-id", default=None, help="LeRobot calibration id for the follower")
    args = parser.parse_args()

    arm = args.arm if args.env == "dual" else None

    if args.dry_run:
        run_dry_run(args.env, arm)
        return

    mirror = None
    if args.real_follower_port:
        from lerobot.robots.so_follower import SO101Follower
        from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig

        cfg = SO101FollowerConfig(port=args.real_follower_port, id=args.real_follower_id)
        mirror = SO101Follower(cfg)
        mirror.connect()
        print(f"Mirroring IK targets to real follower on {args.real_follower_port}")

    run_gamepad(args.env, arm if arm is not None else "left", args.device,
                no_viewer=args.no_viewer, mirror=mirror)


if __name__ == "__main__":
    main()
