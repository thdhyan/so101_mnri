#!/usr/bin/env python
"""Stream SIM joint commands to a REAL SO-101 follower arm (Feetech STS3215, via lerobot).

Three command sources (--source):

    script      Home<->reach joint-space trajectory. Pure numpy fallback, needs
                nothing but mujoco model bounds. Demo-safe.
    gamepad     DS4-driven Cartesian IK in the MuJoCo sim (reuses IKTeleop +
                build_ik from teleop_gamepad_ik.py); streams the solved joint
                targets to the real arm while showing the passive viewer.
    checkpoint  Load an skrl PPO checkpoint (same construction as rl/play.py
                mujoco path), roll the policy in the MuJoCo task env, stream
                the resulting joint-position targets.

Sink: lerobot SO101Follower (class SOFollower, config SO101FollowerConfig,
'<joint>.pos' keys in DEGREES). A calibration file for --follower-id must
already exist (lerobot-calibrate --robot.type=so101_follower ...).

SAFETY
    --max-jump-rad   per-tick clamp between successive streamed targets (default 0.15)
    --torque-limit   Max_Torque_Limit register % written to ALL servos after connect
    Ctrl-C           smooth ramp back to HOME_POSE, then torque off, then disconnect
    --dry-run        prints commands, NEVER touches hardware (also skips pygame/lerobot)

NOTE on frames: sim targets are radians in the MJCF joint convention; the real
arm interprets degrees through its lerobot calibration (homing offsets). The
two agree only if the follower was calibrated consistently with the sim zero
pose — verify with a low --max-jump-rad before trusting large motions.

Usage:
    MUJOCO_GL=egl python scripts/sim_to_real.py --source script     --dry-run
    python scripts/sim_to_real.py --source script --follower-port /dev/ttyACM1 --follower-id myfollower
    python scripts/sim_to_real.py --source gamepad --follower-port /dev/ttyACM1 --follower-id myfollower
    MUJOCO_GL=egl python scripts/sim_to_real.py --source checkpoint \
        --task pick_lift --checkpoint rl/runs/mujoco/pick_lift_skrl/<stamp>/agent.pt --dry-run
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (_HERE, str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from _env_utils import HOME_POSE, JOINT_SUFFIXES

JOINT_ORDER = list(JOINT_SUFFIXES)

# Demo-safe reach pose (inside all MJCF ctrlranges, see robots/so101/so101_follower.xml).
REACH_POSE = np.array([0.6, -0.95, 1.25, 0.15, 0.0, 0.0])

DRY_RUN_TICKS = 100


def joint_limits():
    """Sim actuator ctrlrange for the six joints (single-arm scene = canonical)."""
    from _env_utils import joint_names, resolve_actuator_ids

    import mujoco

    model = mujoco.MjModel.from_xml_path(str(_ROOT / "envs/mujoco/so101_single_arm/assets/scene.xml"))
    aids = resolve_actuator_ids(model, joint_names("single", None))
    crange = model.actuator_ctrlrange[aids]
    return crange[:, 0].copy(), crange[:, 1].copy()


def print_banner(args):
    if args.dry_run:
        print("[dry-run] DRY RUN — no serial port will be opened, no hardware touched.")
        return
    r = "\033[1;91m"
    w = "\033[1;97m"
    z = "\033[0m"
    torque_txt = f"{args.torque_limit:.0f}%" if args.torque_limit is not None else "lerobot defaults"
    print(f"""{r}
+=====================================================+
|{w}         ***   R E A L   H A R D W A R E   ***       {r}|
|                                                     |
|  Follower arm port : {args.follower_port:<32}|
|  Torque limit      : {torque_txt:<32}|
|  Per-tick clamp    : {args.max_jump_rad:.2f} rad{' ' * 25}|
|  Control rate      : {args.hz:.0f} Hz{' ' * 28}|
|  Ctrl-C            : ramp to HOME, torque off       |
+====================================================={z}""", flush=True)


# ---------------------------------------------------------------------------
# Sink: sim joint targets (rad) -> real follower arm
# ---------------------------------------------------------------------------

class FollowerSink:
    """Clamps per-tick jumps and forwards '<joint>.pos' degree actions."""

    def __init__(self, args):
        self.args = args
        self.dry_run = args.dry_run
        self.max_jump = args.max_jump_rad
        self.hz = args.hz
        self.last = None
        self.tick = 0
        self.robot = None

    def connect(self):
        if self.dry_run:
            return self
        from lerobot.robots.so_follower import SO101Follower
        from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig

        cfg = SO101FollowerConfig(
            port=self.args.follower_port,
            id=self.args.follower_id,
            # second safety layer inside lerobot itself (degrees per send)
            max_relative_target=math.degrees(self.max_jump),
        )
        self.robot = SO101Follower(cfg)
        self.robot.connect()
        if self.args.torque_limit is not None:
            raw = int(round(self.args.torque_limit * 10))  # STS3215 register: 1000 == 100%
            for motor in self.robot.bus.motors:
                self.robot.bus.write("Max_Torque_Limit", motor, raw)
            print(f"[sink] Max_Torque_Limit = {self.args.torque_limit:.0f}% on all 6 servos")
        return self

    def send(self, targets_rad):
        t = np.asarray(targets_rad, dtype=float).reshape(-1)[:6]
        if self.last is not None:
            t = self.last + np.clip(t - self.last, -self.max_jump, self.max_jump)
        self.last = t.copy()
        self.tick += 1
        if self.dry_run:
            print(f"[dry-run][tick {self.tick:3d}] "
                  + " ".join(f"{j}={v:+.3f}" for j, v in zip(JOINT_ORDER, t)))
            return
        action = {f"{j}.pos": math.degrees(float(t[i])) for i, j in enumerate(JOINT_ORDER)}
        self.robot.send_action(action)

    def close(self):
        """Ramp smoothly to HOME_POSE, disable torque, disconnect."""
        if self.robot is None:
            return
        try:
            start = self.last if self.last is not None else np.asarray(HOME_POSE, dtype=float)
            n_steps = int(math.ceil(np.max(np.abs(np.asarray(HOME_POSE, dtype=float) - start)) / self.max_jump))
            print(f"[sink] returning to home in {n_steps} steps, then disabling torque...")
            for k in range(1, n_steps + 1):
                t = start + (np.asarray(HOME_POSE, dtype=float) - start) * k / n_steps
                action = {f"{j}.pos": math.degrees(float(t[i])) for i, j in enumerate(JOINT_ORDER)}
                self.robot.send_action(action)
                time.sleep(1.0 / self.hz)
        except Exception as e:
            print(f"[sink] WARNING during home ramp: {e}")
        finally:
            try:
                self.robot.bus.disable_torque()
                print("[sink] torque disabled")
            except Exception as e:
                print(f"[sink] WARNING: could not disable torque explicitly: {e}")
            try:
                self.robot.disconnect()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Source: scripted home <-> reach trajectory (pure numpy)
# ---------------------------------------------------------------------------

def run_script_source(sink, lo, hi):
    period = sink.args.period
    tick = 0
    while True:
        t_loop = time.perf_counter()
        t = tick / sink.hz
        phase = 0.5 * (1.0 - math.cos(2.0 * math.pi * t / period))
        q = np.asarray(HOME_POSE, dtype=float) + (REACH_POSE - np.asarray(HOME_POSE)) * phase
        q[5] = 0.75 + 0.45 * math.sin(2.0 * math.pi * t / (period / 2.0))
        sink.send(np.clip(q, lo, hi))
        tick += 1
        if sink.dry_run and tick >= DRY_RUN_TICKS:
            break
        elapsed = time.perf_counter() - t_loop
        time.sleep(max(0.0, 1.0 / sink.hz - elapsed))


# ---------------------------------------------------------------------------
# Source: DS4 gamepad -> IKTeleop (imported from teleop_gamepad_ik)
# ---------------------------------------------------------------------------

def run_gamepad_source(sink, args, lo, hi):
    import pygame

    import mujoco

    from _env_utils import gripper_site_name, joint_names, resolve_actuator_ids, resolve_joint_ids
    from teleop_gamepad_ik import (
        GRIPPER_SCALE,
        LINEAR_SCALE,
        PITCH_SCALE,
        ROLL_SCALE,
        IKTeleop,
        build_ik,
    )

    arm = args.arm
    model, data, ik = build_ik(args.env, arm)
    ik_by_arm = {arm: ik}
    if args.env == "dual":
        other = "right" if arm == "left" else "left"
        jnames = joint_names(args.env, other)
        joint_ids = resolve_joint_ids(model, jnames)
        actuator_ids = resolve_actuator_ids(model, jnames)
        qpos_adr = np.array([model.jnt_qposadr[j] for j in joint_ids])
        data.qpos[qpos_adr] = HOME_POSE
        data.ctrl[actuator_ids] = HOME_POSE
        ik_by_arm[other] = IKTeleop(model, data, joint_ids, actuator_ids,
                                    model.site(gripper_site_name(args.env, other)).id)
        mujoco.mj_forward(model, data)

    pad = None
    if not sink.dry_run:
        pygame.init()
        from teleop_gamepad_ik import DS4Gamepad
        pad = DS4Gamepad.detect(args.joy_index)
        if pad is None:
            print("No joystick found. Connect the DS4 (see TELEOP_GUIDE.md) or add --dry-run.")
            sys.exit(1)
        print(f"Using joystick: {pad.name} [{pad.mode}]")

    viewer = None
    if not sink.dry_run and not args.no_viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(model, data)

    dt_ctrl = 1.0 / args.hz
    substeps = max(1, int(round(dt_ctrl / model.opt.timestep)))
    share_prev = options_prev = False
    estop = False
    active_arm = arm

    try:
        tick = 0
        while True:
            t_loop = time.perf_counter()
            if pad is not None:
                state = pad.poll()
                lx, ly = state.left_stick
                rx, ry = state.right_stick
                trig_open, trig_close = state.r2, state.l2
                pitch_rate = (state.buttons["r1"] - state.buttons["l1"]) * PITCH_SCALE
                if state.buttons["ps"]:
                    estop = True
                options_now = bool(state.buttons["options"])
                if options_now and not options_prev:
                    if estop:  # resume from e-stop ...
                        estop = False
                    cur_ik = ik_by_arm[active_arm]
                    cur_ik.set_ee_target(cur_ik._current_ee_pos())  # ... and re-center target
                    print("[gamepad] re-centered IK target" + (" (e-stop released)" if not estop else ""))
                options_prev = options_now
                if args.env == "dual":
                    share_now = bool(state.buttons["share"])
                    if share_now and not share_prev:
                        active_arm = "right" if active_arm == "left" else "left"
                        print(f"Active arm -> {active_arm}")
                    share_prev = share_now
            else:
                # dry-run driver: scripted circular EE target + oscillating gripper
                lx = math.cos(2 * math.pi * tick / DRY_RUN_TICKS)
                ly = math.sin(2 * math.pi * tick / DRY_RUN_TICKS)
                rx = ry = 0.0
                trig_open = 0.5 + 0.5 * math.sin(2 * math.pi * tick / DRY_RUN_TICKS)
                trig_close = pitch_rate = 0.0

            cur_ik = ik_by_arm[active_arm]
            if not estop:
                vx, vy, vz = -ly * LINEAR_SCALE, -lx * LINEAR_SCALE, -ry * LINEAR_SCALE
                cur_ik.set_ee_target(cur_ik.ee_target + np.array([vx, vy, vz]) * dt_ctrl)
                cur_ik.set_gripper(cur_ik.gripper_openness
                                   + (trig_open - trig_close) * GRIPPER_SCALE * dt_ctrl)

                roll_idx, pitch_idx = 4, 3  # wrist_roll / wrist_flex driven outside IK nullspace
                qadr = cur_ik.qpos_adr
                data.qpos[qadr[roll_idx]] = np.clip(
                    data.qpos[qadr[roll_idx]] + rx * ROLL_SCALE * dt_ctrl, *cur_ik.jnt_range[roll_idx])
                data.qpos[qadr[pitch_idx]] = np.clip(
                    data.qpos[qadr[pitch_idx]] + pitch_rate * dt_ctrl, *cur_ik.jnt_range[pitch_idx])

            for a in ik_by_arm.values():
                a.solve_step()
            qadr = cur_ik.qpos_adr
            data.ctrl[cur_ik.actuator_ids[4]] = data.qpos[qadr[4]]
            data.ctrl[cur_ik.actuator_ids[3]] = data.qpos[qadr[3]]

            sink.send(np.clip(data.ctrl[cur_ik.actuator_ids], lo, hi))

            for _ in range(substeps):
                mujoco.mj_step(model, data)

            if viewer is not None:
                viewer.sync()
                if not viewer.is_running():
                    return

            tick += 1
            if sink.dry_run and tick >= DRY_RUN_TICKS:
                err = cur_ik.ee_error()
                print(f"[dry-run] {tick} ticks; final EE-target error = {err * 100:.3f} cm")
                break
            elapsed = time.perf_counter() - t_loop
            time.sleep(max(0.0, dt_ctrl - elapsed))
    finally:
        if viewer is not None:
            viewer.close()


# ---------------------------------------------------------------------------
# Source: skrl checkpoint rolled in the MuJoCo env (rl/play.py pattern)
# ---------------------------------------------------------------------------

def run_checkpoint_source(sink, args, lo, hi):
    import importlib

    import torch

    from rl.train import MUJOCO_TASKS, DropImagesObs, SkrlPolicy, SkrlValue

    module_path, func_name = MUJOCO_TASKS[args.task].split(":")
    vec_env = getattr(importlib.import_module(module_path), func_name)(n_envs=1)
    env = vec_env.envs[0]
    if tuple(env.action_space.shape) != (6,):
        sys.exit(f"ERROR: task '{args.task}' is not single-arm (action space {env.action_space.shape}); "
                 "need a 6-DoF task for one follower arm")
    # State-obs policy (DropImagesObs discards images anyway) — skip camera
    # rendering entirely so the control loop isn't taxed by EGL renders.
    env._render_camera = lambda name, w, h: np.zeros((h, w, 3), dtype=np.uint8)

    from skrl.agents.torch.ppo import PPO, PPO_CFG
    from skrl.envs.wrappers.torch import wrap_env
    from skrl.memories.torch import RandomMemory

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    wrapped = wrap_env(DropImagesObs(env), wrapper="gymnasium")
    models = {
        "policy": SkrlPolicy(wrapped.observation_space, wrapped.action_space, device).to(device),
        "value": SkrlValue(wrapped.observation_space, wrapped.action_space, device).to(device),
    }
    agent = PPO(
        models=models,
        memory=RandomMemory(memory_size=1, num_envs=1, device=device),
        cfg=PPO_CFG(),
        observation_space=wrapped.observation_space,
        action_space=wrapped.action_space,
        device=device,
    )
    if args.checkpoint:
        agent.load(args.checkpoint)
        print(f"[checkpoint] loaded {args.checkpoint} on {device}")
    else:
        print("[dry-run] NO --checkpoint given: rolling an UNTRAINED policy (random weights)")

    episodes = 0
    state, _ = wrapped.reset()
    tick = 0
    while True:
        t_loop = time.perf_counter()
        with torch.no_grad():
            # skrl fork API: act(observations, states, *, timestep, timesteps)
            actions = agent.act(state, None, timestep=tick, timesteps=10 ** 9)[0]
        state, reward, terminated, truncated, info = wrapped.step(actions)
        targets = actions.squeeze(0).detach().cpu().numpy().astype(float)
        sink.send(np.clip(targets, lo, hi))
        tick += 1
        if bool((terminated | truncated).any()):
            episodes += 1
            print(f"[checkpoint] episode {episodes} ended after {tick} ticks "
                  f"(reward={float(reward):.2f}); auto-reset")
            state, _ = wrapped.reset()
        if sink.dry_run and tick >= DRY_RUN_TICKS:
            break
        elapsed = time.perf_counter() - t_loop
        time.sleep(max(0.0, 1.0 / sink.hz - elapsed))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["script", "gamepad", "checkpoint"], default="script",
                   help="where the joint commands come from")
    p.add_argument("--follower-port", default="/dev/ttyACM1",
                   help="serial port of the REAL follower arm")
    p.add_argument("--follower-id", default=None, help="LeRobot calibration id of the follower")
    p.add_argument("--max-jump-rad", type=float, default=0.15,
                   help="per-tick clamp on target change (safety; 0.15 rad ~= 8.6 deg)")
    p.add_argument("--torque-limit", type=float, default=None,
                   help="%% of max torque written to all servos at connect (e.g. 40)")
    p.add_argument("--hz", type=float, default=30.0, help="control-loop rate (keep <=60 on the Feetech bus)")
    p.add_argument("--dry-run", action="store_true",
                   help=f"print commands instead of touching hardware; runs {DRY_RUN_TICKS} ticks and exits 0")
    p.add_argument("--period", type=float, default=6.0,
                   help="script source: seconds per home->reach->home cycle")
    p.add_argument("--task", default="pick_lift",
                   help="checkpoint source: MuJoCo task name (key of rl.train.MUJOCO_TASKS)")
    p.add_argument("--checkpoint", default=None,
                   help="skrl agent checkpoint (.pt); required for live checkpoint runs")
    p.add_argument("--device", default=None, help="torch device for the policy (default cuda:0 if available)")
    p.add_argument("--env", choices=["single", "dual"], default="single",
                   help="gamepad source: sim scene (one real follower mirrors the active arm)")
    p.add_argument("--arm", choices=["left", "right"], default="left",
                   help="gamepad source: initial active arm in the dual env")
    p.add_argument("--joy-index", type=int, default=0, help="pygame joystick index for the DS4")
    p.add_argument("--no-viewer", action="store_true", help="disable the MuJoCo viewer window")
    args = p.parse_args(argv)
    return args


def main(argv=None):
    args = parse_args(argv)

    if args.dry_run or args.no_viewer:
        # Headless-safe GL before any GL context is created (live+viewer needs a display instead).
        os.environ.setdefault("MUJOCO_GL", "egl")

    lo, hi = joint_limits()

    if not args.dry_run:
        if not os.path.exists(args.follower_port):
            print(f"ERROR: follower port not found: {args.follower_port} "
                  "(check `ls /dev/ttyACM*`; see TELEOP_GUIDE.md troubleshooting)")
            sys.exit(1)
        if args.source == "checkpoint" and not args.checkpoint:
            print("ERROR: --source checkpoint requires --checkpoint <path.pt> for live runs")
            sys.exit(1)

    print_banner(args)

    sink = FollowerSink(args)
    try:
        sink.connect()
        if args.source == "script":
            run_script_source(sink, lo, hi)
        elif args.source == "gamepad":
            run_gamepad_source(sink, args, lo, hi)
        else:
            run_checkpoint_source(sink, args, lo, hi)
    except KeyboardInterrupt:
        print("\n[sim_to_real] interrupted — shutting down safely")
    finally:
        sink.close()
    print(f"[sim_to_real] done ({'dry-run' if args.dry_run else 'live'}), {sink.tick} ticks streamed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
