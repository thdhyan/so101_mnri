#!/usr/bin/env python
"""Drive an SO-101 MuJoCo follower sim from a real SO-101 "leader" arm read via
Feetech STS3215 servos (scservo_sdk), IDs 1..6, 1 Mbaud.

Usage:
    python scripts/teleop_leader_arm.py --env single --port /dev/ttyACM0
    python scripts/teleop_leader_arm.py --env dual --left-port /dev/ttyACM0 --right-port /dev/ttyACM1
    python scripts/teleop_leader_arm.py --calibrate --port /dev/ttyACM0
    MUJOCO_GL=egl python scripts/teleop_leader_arm.py --dry-run --env single
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import mujoco

from _env_utils import scene_path, joint_names, HOME_POSE, resolve_actuator_ids, resolve_joint_ids

BAUDRATE = 1_000_000
PROTOCOL_END = 0  # Feetech STS protocol
SERVO_IDS = [1, 2, 3, 4, 5, 6]  # shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper

ADDR_TORQUE_ENABLE = 40
ADDR_PRESENT_POSITION = 56
LEN_PRESENT_POSITION = 2

TICKS_PER_RAD_DEFAULT = 4096 / (2 * math.pi)
CONTROL_HZ = 60.0

DEFAULT_CALIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "leader_calib.json")


def ticks_to_rad(ticks, offset_ticks, sign, ticks_per_rad):
    return sign * (ticks - offset_ticks) / ticks_per_rad


class LeaderArm:
    """Wraps a scservo_sdk connection to one 6-servo SO-101 leader arm."""

    def __init__(self, port, calib):
        import scservo_sdk as sc
        self.sc = sc
        self.calib = calib  # list of dicts: {id, offset_ticks, sign, ticks_per_rad}

        self.port_handler = sc.PortHandler(port)
        self.packet_handler = sc.PacketHandler(PROTOCOL_END)
        if not self.port_handler.openPort():
            raise RuntimeError(f"Failed to open leader arm port: {port}")
        if not self.port_handler.setBaudRate(BAUDRATE):
            raise RuntimeError(f"Failed to set baudrate {BAUDRATE} on {port}")

        self.group_read = sc.GroupSyncRead(
            self.port_handler, self.packet_handler, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
        )
        for c in self.calib:
            if not self.group_read.addParam(c["id"]):
                raise RuntimeError(f"Failed to add servo id {c['id']} to sync read group")

        self._disable_torque()

    def _disable_torque(self):
        for c in self.calib:
            self.packet_handler.write1ByteTxRx(self.port_handler, c["id"], ADDR_TORQUE_ENABLE, 0)

    def read_positions_rad(self):
        result = self.group_read.txRxPacket()
        if result != self.sc.COMM_SUCCESS:
            raise RuntimeError(f"Leader arm sync read failed: {self.packet_handler.getTxRxResult(result)}")
        rad = np.zeros(len(self.calib))
        for i, c in enumerate(self.calib):
            ticks = self.group_read.getData(c["id"], ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
            rad[i] = ticks_to_rad(ticks, c["offset_ticks"], c["sign"], c["ticks_per_rad"])
        return rad

    def read_raw_ticks(self):
        result = self.group_read.txRxPacket()
        if result != self.sc.COMM_SUCCESS:
            raise RuntimeError(f"Leader arm sync read failed: {self.packet_handler.getTxRxResult(result)}")
        return {c["id"]: self.group_read.getData(c["id"], ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
                for c in self.calib}

    def close(self):
        self.port_handler.closePort()


def default_calib():
    return [{"id": sid, "offset_ticks": 2048, "sign": 1, "ticks_per_rad": TICKS_PER_RAD_DEFAULT}
            for sid in SERVO_IDS]


def load_calib(path):
    if not os.path.exists(path):
        print(f"No calibration file at {path}; using identity calibration "
              f"(center=2048 ticks, sign=+1). Run --calibrate for accuracy.")
        return default_calib()
    with open(path) as f:
        return json.load(f)


def run_calibrate(port, calib_path, home_pose=HOME_POSE):
    import scservo_sdk as sc
    port_handler = sc.PortHandler(port)
    packet_handler = sc.PacketHandler(PROTOCOL_END)
    if not port_handler.openPort():
        print(f"ERROR: failed to open port {port}")
        sys.exit(1)
    if not port_handler.setBaudRate(BAUDRATE):
        print(f"ERROR: failed to set baudrate on {port}")
        sys.exit(1)

    for sid in SERVO_IDS:
        packet_handler.write1ByteTxRx(port_handler, sid, ADDR_TORQUE_ENABLE, 0)

    input(f"Hold the leader arm at the home pose {home_pose.tolist()} "
          f"(radians, joint order shoulder_pan/lift/elbow_flex/wrist_flex/wrist_roll/gripper), "
          f"then press Enter to record...")

    group_read = sc.GroupSyncRead(port_handler, packet_handler, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
    for sid in SERVO_IDS:
        group_read.addParam(sid)
    result = group_read.txRxPacket()
    if result != sc.COMM_SUCCESS:
        print(f"ERROR: sync read failed: {packet_handler.getTxRxResult(result)}")
        sys.exit(1)

    calib = []
    for sid, home_rad in zip(SERVO_IDS, home_pose):
        ticks = group_read.getData(sid, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
        # offset such that ticks_to_rad(ticks, offset, +1, tpr) == home_rad
        offset_ticks = ticks - home_rad * TICKS_PER_RAD_DEFAULT
        calib.append({"id": sid, "offset_ticks": offset_ticks, "sign": 1,
                       "ticks_per_rad": TICKS_PER_RAD_DEFAULT})

    with open(calib_path, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"Saved calibration to {calib_path}")
    port_handler.closePort()


def run_dry_run(env, steps=100):
    path = scene_path(env)
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)

    arms = [None] if env == "single" else ["left", "right"]
    actuator_ids_by_arm = {}
    ctrl_range_by_arm = {}
    for arm in arms:
        jnames = joint_names(env, arm)
        aids = resolve_actuator_ids(model, jnames)
        actuator_ids_by_arm[arm] = aids
        ctrl_range_by_arm[arm] = model.actuator_ctrlrange[aids]
        data.ctrl[aids] = HOME_POSE
    mujoco.mj_forward(model, data)

    dt_ctrl = 1.0 / CONTROL_HZ
    substeps = max(1, int(round(dt_ctrl / model.opt.timestep)))

    for i in range(steps):
        t = i * dt_ctrl
        fake_leader = HOME_POSE + 0.15 * np.sin(2 * np.pi * 0.5 * t) * np.ones(6)
        for arm in arms:
            aids = actuator_ids_by_arm[arm]
            crange = ctrl_range_by_arm[arm]
            ctrl = np.clip(fake_leader, crange[:, 0], crange[:, 1])
            data.ctrl[aids] = ctrl
        for _ in range(substeps):
            mujoco.mj_step(model, data)

    print(f"[dry-run] env={env} steps={steps} completed. Final qpos sample: {data.qpos[:6]}")
    print("[dry-run] OK")


def run_teleop(env, ports, calib_path):
    """ports: dict mapping arm-name ('single' arm uses key None) -> serial port string."""
    path = scene_path(env)
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)

    calib = load_calib(calib_path)

    leaders = {}
    actuator_ids_by_arm = {}
    ctrl_range_by_arm = {}
    for arm, port in ports.items():
        jnames = joint_names(env, arm)
        aids = resolve_actuator_ids(model, jnames)
        actuator_ids_by_arm[arm] = aids
        ctrl_range_by_arm[arm] = model.actuator_ctrlrange[aids]
        data.ctrl[aids] = HOME_POSE
        try:
            leaders[arm] = LeaderArm(port, calib)
        except RuntimeError as e:
            print(f"ERROR connecting to leader arm on {port}: {e}")
            sys.exit(1)
    mujoco.mj_forward(model, data)

    dt_ctrl = 1.0 / CONTROL_HZ
    substeps = max(1, int(round(dt_ctrl / model.opt.timestep)))

    import mujoco.viewer
    viewer = mujoco.viewer.launch_passive(model, data)
    try:
        running = True
        while running and viewer.is_running():
            t0 = time.time()
            for arm, leader in leaders.items():
                rad = leader.read_positions_rad()
                aids = actuator_ids_by_arm[arm]
                crange = ctrl_range_by_arm[arm]
                ctrl = np.clip(rad, crange[:, 0], crange[:, 1])
                data.ctrl[aids] = ctrl

            for _ in range(substeps):
                mujoco.mj_step(model, data)

            viewer.sync()
            elapsed = time.time() - t0
            time.sleep(max(0.0, dt_ctrl - elapsed))
    finally:
        viewer.close()
        for leader in leaders.values():
            leader.close()


def main():
    parser = argparse.ArgumentParser(description="Real SO-101 leader arm -> MuJoCo follower sim")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Leader arm serial port (single env)")
    parser.add_argument("--env", choices=["single", "dual"], default="single")
    parser.add_argument("--left-port", default="/dev/ttyACM0", help="Left leader arm port (dual env)")
    parser.add_argument("--right-port", default="/dev/ttyACM1", help="Right leader arm port (dual env)")
    parser.add_argument("--calib", default=DEFAULT_CALIB_PATH, help="Path to calibration JSON")
    parser.add_argument("--calibrate", action="store_true", help="Run interactive calibration and exit")
    parser.add_argument("--dry-run", action="store_true",
                         help="Fake the leader with a sinusoid trajectory, headless, 100 steps")
    args = parser.parse_args()

    if args.calibrate:
        run_calibrate(args.port, args.calib)
        return

    if args.dry_run:
        run_dry_run(args.env)
        return

    if args.env == "single":
        if not os.path.exists(args.port):
            print(f"ERROR: leader arm port not found: {args.port}")
            sys.exit(1)
        run_teleop(args.env, {None: args.port}, args.calib)
    else:
        for p in (args.left_port, args.right_port):
            if not os.path.exists(p):
                print(f"ERROR: leader arm port not found: {p}")
                sys.exit(1)
        run_teleop(args.env, {"left": args.left_port, "right": args.right_port}, args.calib)


if __name__ == "__main__":
    main()
