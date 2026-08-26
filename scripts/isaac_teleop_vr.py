#!/usr/bin/env python
"""VR (Meta Quest 3) teleop for the SO-101 Isaac Lab tasks via isaacteleop + CloudXR.

Wired path (see VR_TELEOP_SETUP.md for the full setup):

    Quest 3 browser (WebXR CloudXR client)
      | WebRTC stream + controller poses
      v
    isaacteleop CloudXR runtime (Monado, bundled in the wheel) + WSS proxy :48322
      v
    Isaac Sim Kit OpenXR session (--xr; auto-starts headless)
      v  isaacsim.kit.xr.teleop.bridge handles
    TeleopSession -> retargeting pipeline (everything shipped in isaacteleop):
      ControllersSource(right) -> ControllerTransform(world_T_anchor = base_T_world)
        -> SO101ClutchRetargeter  : absolute EE pose [xyz quat_xyzw], robot base frame
        -> SO101GripperRetargeter : analog jaw closedness c in [0, 1]
      -> TensorReorderer       : flat 8D action [x y z qx qy qz qw c]
      v
    DifferentialIKController (pose, absolute, dls) -> 5 arm joint targets
    closedness affine -> gripper joint target   =>  6D absolute JointPositionAction

Right-controller controls:
    pose            EE target (clutched delta around the engage origin)
    trigger         gripper closedness (analog)
    grip (squeeze)  re-center the clutch origin (arm stays put)
    A button        toggle XR anchor rotation (built into IsaacTeleopDevice)

IK / 3D coordinate tuning:
    The VR controller pose IS the 3D EE target in robot-base frame.
    --anchor-pos sets where the VR stage origin maps to in the world: tune this
    to align controller movement with the arm's workspace.
    --ik-lambda controls DLS damping (higher = smoother but less accurate).
    Squeeze grip to re-clutch (reset controller origin without moving arm).

Dual-arm VR:
    Use task SO101-CylReach-Dual-v0 (or SO101-CylGrasp-Dual-v0).
    Right controller -> right arm (env arm index 0).
    Left  controller -> left  arm (env arm index 1).
    Both arms get the same SO101ClutchRetargeter pipeline, each in their own
    robot base frame.  Pass --task SO101-CylReach-Dual-v0 to enable.

Real arm (USB follower):
    Pass --follower-port /dev/ttyACM0 (and optionally --follower-id NAME).
    Requires: lerobot installed, arm calibrated (lerobot-calibrate --robot.type=so101_follower).
    Sim joints (rad) are forwarded to the real arm every step via FollowerSink,
    with a per-tick clamp (--max-jump-rad, default 0.15 rad) for safety.
    Ctrl-C ramps the real arm back to home and disables torque before exiting.

Usage:
    # dry run: scripted controller trajectory through the real retargeting+IK
    # pipeline; headless GPU; no XR runtime / CloudXR / headset required
    OMNI_KIT_ACCEPT_EULA=YES python scripts/isaac_teleop_vr.py --null-device

    # VR sim-only (see VR_TELEOP_SETUP.md)
    python -m isaacteleop.cloudxr --cloudxr-env-config ~/.cloudxr/quest3-lo.env --host-client
    python scripts/isaac_teleop_vr.py --headless --cloudxr external

    # VR + mirror to real arm
    python scripts/isaac_teleop_vr.py --headless --cloudxr external \\
        --follower-port /dev/ttyACM0 --follower-id my_arm

    # Dual-arm VR (sim only)
    python scripts/isaac_teleop_vr.py --headless --cloudxr external \\
        --task SO101-CylReach-Dual-v0
"""

import argparse
import math
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TASK = "SO101-CylReach-Single-v0"
ANCHOR_LEAF = "world_T_anchor"
BASE_PRIM_SUFFIX = "/Robot/Geometry/base_link"
EE_BODY_SUFFIX = "gripper_frame_link"
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_OPEN_RAD = -0.174533  # URDF lower limit
GRIPPER_CLOSE_RAD = 1.74533   # URDF upper limit


def _fail(msg: str) -> None:
    print(f"\n[isaac_teleop_vr] ERROR: {msg}\n", file=sys.stderr)
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", type=str, default=DEFAULT_TASK,
                        help=f"registered task id (default {DEFAULT_TASK}; single-arm only)")
    parser.add_argument("--null-device", action="store_true",
                        help="dry run: scripted controller trajectory through the real "
                        "retargeting+IK pipeline, headless, no XR runtime needed")
    parser.add_argument("--steps", type=int, default=100,
                        help="control steps for --null-device (default 100)")
    parser.add_argument("--headless", action="store_true",
                        help="no local Isaac window (the Quest still gets its own stream)")
    parser.add_argument("--cloudxr", choices=["auto", "external", "off"], default="auto",
                        help="auto: launch runtime+WSS from this script (default); external: "
                        "runtime started separately (python -m isaacteleop.cloudxr ...); "
                        "off: no CloudXR (drive Kit's XR panel manually)")
    parser.add_argument("--anchor-pos", type=float, nargs=3, default=(0.9, 0.3, 1.0),
                        metavar=("X", "Y", "Z"),
                        help="world point shown at the headset tracking origin "
                        "(default 0.9 0.3 1.0 = ~90cm in front of arm, 30cm to side, 1m up; "
                        "tune X to step closer/further, Z for height, Y for side offset)")
    parser.add_argument("--gripper-open", type=float, default=GRIPPER_OPEN_RAD)
    parser.add_argument("--gripper-close", type=float, default=GRIPPER_CLOSE_RAD)
    parser.add_argument("--ik-lambda", type=float, default=0.05,
                        help="damped-least-squares damping (default 0.05; higher=smoother/less accurate)")
    # Real arm (USB follower) — optional; requires lerobot + calibrated arm
    parser.add_argument("--follower-port", type=str, default=None,
                        metavar="PORT",
                        help="serial port for real SO-101 follower arm (e.g. /dev/ttyACM0); "
                        "omit to run sim-only")
    parser.add_argument("--follower-id", type=str, default="so101_follower",
                        metavar="ID",
                        help="lerobot calibration ID for the follower arm (default: so101_follower)")
    parser.add_argument("--max-jump-rad", type=float, default=0.15,
                        help="per-tick joint clamp for real arm safety (default 0.15 rad ≈ 8.6°)")
    parser.add_argument("--follower-torque-limit", type=float, default=None,
                        metavar="PCT",
                        help="Max_Torque_Limit %% written to all servos on connect (e.g. 60)")
    parser.add_argument("--dry-run-follower", action="store_true",
                        help="print follower commands but do not open the serial port")
    return parser.parse_args()


##
# Real arm sink — mirrors sim joint targets to a USB SO-101 follower arm via lerobot
##

_FOLLOWER_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


class FollowerSink:
    """Forward sim joint targets (rad) to a real SO-101 arm via lerobot.

    Clamps per-tick jumps for safety. On close(), ramps smoothly back to home
    and disables torque. Pass dry_run=True to print commands without touching HW.
    """

    def __init__(self, port: str, follower_id: str, max_jump_rad: float = 0.15,
                 torque_limit_pct: float | None = None, dry_run: bool = False):
        self._port = port
        self._id = follower_id
        self._max_jump = max_jump_rad
        self._torque_limit = torque_limit_pct
        self._dry_run = dry_run
        self._last = None
        self._tick = 0
        self._robot = None

    def connect(self) -> "FollowerSink":
        if self._dry_run:
            print(f"[follower][dry-run] would connect to {self._port} id={self._id}")
            return self
        from lerobot.robots.so_follower import SO101Follower
        from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
        cfg = SO101FollowerConfig(
            port=self._port,
            id=self._id,
            max_relative_target=math.degrees(self._max_jump),
        )
        self._robot = SO101Follower(cfg)
        self._robot.connect()
        if self._torque_limit is not None:
            raw = int(round(self._torque_limit * 10))
            for motor in self._robot.bus.motors:
                self._robot.bus.write("Max_Torque_Limit", motor, raw)
            print(f"[follower] Max_Torque_Limit = {self._torque_limit:.0f}% on all servos")
        print(f"[follower] connected: {self._port} id={self._id}  clamp={self._max_jump:.3f} rad")
        return self

    def send(self, targets_rad) -> None:
        import numpy as np
        t = np.asarray(targets_rad, dtype=float).reshape(-1)[:6]
        if self._last is not None:
            t = self._last + np.clip(t - self._last, -self._max_jump, self._max_jump)
        self._last = t.copy()
        self._tick += 1
        if self._dry_run:
            print(f"[follower][dry-run][{self._tick:4d}] "
                  + " ".join(f"{j}={v:+.3f}" for j, v in zip(_FOLLOWER_JOINTS, t)))
            return
        action = {f"{j}.pos": math.degrees(float(t[i])) for i, j in enumerate(_FOLLOWER_JOINTS)}
        self._robot.send_action(action)

    def close(self) -> None:
        if self._robot is None:
            return
        import numpy as np
        from scripts._env_utils import HOME_POSE  # noqa: local import
        home = np.asarray(HOME_POSE, dtype=float)
        start = self._last if self._last is not None else home
        n = max(1, int(np.ceil(np.max(np.abs(home - start)) / self._max_jump)))
        print(f"[follower] ramping to home in {n} steps …")
        for k in range(1, n + 1):
            self.send(start + (home - start) * k / n)
        try:
            self._robot.bus.disable_torque()
            print("[follower] torque disabled")
        except Exception as exc:
            print(f"[follower] WARNING disable torque: {exc}")
        try:
            self._robot.disconnect()
        except Exception:
            pass


##
# isaacteleop lazy imports (kept out of module scope so --help works without them)
##


def _import_isaacteleop_bits() -> dict:
    try:
        import numpy as np
        from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource
        from isaacteleop.retargeting_engine.interface import (
            ComputeContext,
            ExecutionEvents,
            ExecutionState,
            OutputCombiner,
            ValueInput,
        )
        from isaacteleop.retargeting_engine.interface.tensor_group import TensorGroup
        from isaacteleop.retargeting_engine.interface.tensor_group_type import OptionalType
        from isaacteleop.retargeting_engine.tensor_types import (
            ControllerInput,
            ControllerInputIndex,
            TransformMatrix,
        )
        from isaacteleop.retargeting_engine.utilities.controller_transform import ControllerTransform
        from isaacteleop.retargeters import (
            SO101ClutchRetargeter,
            SO101GripperRetargeter,
            TensorReorderer,
        )
    except ModuleNotFoundError as e:
        _fail(
            f"isaacteleop component missing ({e}). Setup steps:\n"
            "  1. source .venv/bin/activate\n"
            "  2. pip install 'isaacteleop[retargeters-lite]' "
            "--extra-index-url https://pypi.nvidia.com\n"
            "     (already installed in this repo's venv — check you are using it)\n"
            "  3. if SO101ClutchRetargeter is missing, upgrade: pip install -U isaacteleop"
        )
    return dict(
        np=np,
        ControllersSource=ControllersSource,
        ComputeContext=ComputeContext,
        ExecutionEvents=ExecutionEvents,
        ExecutionState=ExecutionState,
        OutputCombiner=OutputCombiner,
        ValueInput=ValueInput,
        TensorGroup=TensorGroup,
        OptionalType=OptionalType,
        ControllerInput=ControllerInput,
        ControllerInputIndex=ControllerInputIndex,
        TransformMatrix=TransformMatrix,
        ControllerTransform=ControllerTransform,
        SO101ClutchRetargeter=SO101ClutchRetargeter,
        SO101GripperRetargeter=SO101GripperRetargeter,
        TensorReorderer=TensorReorderer,
    )


@dataclass
class PipelineBundle:
    combiner: object          # OutputCombiner with an "action" output (flat 8D)
    clutch: object            # SO101ClutchRetargeter instance (re-center handle)
    ctrl_leaf_name: str       # script-mode controller leaf name (None in xr mode)
    source_mode: str          # "xr" | "script"


def build_so101_pipeline(bits: dict, home_base_T_ee, source_mode: str) -> PipelineBundle:
    """Build the SO-101 retargeting graph shared by the VR and null-device paths.

    source_mode="xr" uses a real ControllersSource (TeleopSession polls the
    DeviceIO tracker);     source_mode="script" swaps in a ValueInput leaf fed by
    --null-device. Downstream nodes are identical.
    """
    if source_mode == "xr":
        controllers = bits["ControllersSource"](name="controllers")
        right_src = controllers.output("controller_right")
        ctrl_leaf = None
    elif source_mode == "script":
        ctrl_leaf = "scripted_controllers"
        leaf = bits["ValueInput"](ctrl_leaf, bits["OptionalType"](bits["ControllerInput"]()))
        right_src = leaf.output("value")
    else:
        raise ValueError(f"unknown source_mode {source_mode!r}")

    anchor_in = bits["ValueInput"](ANCHOR_LEAF, bits["TransformMatrix"]())
    xform = bits["ControllerTransform"]("controller_xform")
    transformed = xform.connect({
        "controller_right": right_src,
        "transform": anchor_in.output("value"),
    })

    clutch = bits["SO101ClutchRetargeter"](
        name="so101_ee",
        input_device="controller_right",
        home_base_T_ee=home_base_T_ee,
    )
    ee_graph = clutch.connect({"controller_right": transformed.output("controller_right")})

    gripper = bits["SO101GripperRetargeter"](name="so101_gripper", input_device="controller_right")
    grip_graph = gripper.connect({"controller_right": transformed.output("controller_right")})

    reorderer = bits["TensorReorderer"](
        input_config={
            "ee_pose": ["ee_x", "ee_y", "ee_z", "ee_qx", "ee_qy", "ee_qz", "ee_qw"],
            "gripper_command": ["gripper_closedness"],
        },
        output_order=["ee_x", "ee_y", "ee_z", "ee_qx", "ee_qy", "ee_qz", "ee_qw", "gripper_closedness"],
        name="so101_action_flat",
        input_types={"ee_pose": "array", "gripper_command": "scalar"},
    )
    flat = reorderer.connect({
        "ee_pose": ee_graph.output("ee_pose"),
        "gripper_command": grip_graph.output("gripper_command"),
    })
    combiner = bits["OutputCombiner"]({"action": flat.output("output")})
    return PipelineBundle(combiner=combiner, clutch=clutch, ctrl_leaf_name=ctrl_leaf, source_mode=source_mode)


def make_controller_tensorgroup(bits: dict, pos, quat_xyzw, trigger: float, valid: bool = True):
    """One fully-populated ControllerInput TensorGroup (null-device feeding)."""
    np = bits["np"]
    tg = bits["TensorGroup"](bits["ControllerInput"]())
    zero3, zero4 = np.zeros(3, dtype=np.float32), np.zeros(4, dtype=np.float32)
    tg[int(bits["ControllerInputIndex"].GRIP_POSITION)] = np.asarray(pos, dtype=np.float32)
    tg[int(bits["ControllerInputIndex"].GRIP_ORIENTATION)] = np.asarray(quat_xyzw, dtype=np.float32)
    tg[int(bits["ControllerInputIndex"].GRIP_IS_VALID)] = bool(valid)
    tg[int(bits["ControllerInputIndex"].AIM_POSITION)] = zero3
    tg[int(bits["ControllerInputIndex"].AIM_ORIENTATION)] = zero4
    tg[int(bits["ControllerInputIndex"].AIM_IS_VALID)] = False
    for field in ("PRIMARY_CLICK", "SECONDARY_CLICK", "THUMBSTICK_X", "THUMBSTICK_Y",
                  "THUMBSTICK_CLICK", "MENU_CLICK", "SQUEEZE_VALUE"):
        tg[int(getattr(bits["ControllerInputIndex"], field))] = 0.0
    tg[int(bits["ControllerInputIndex"].TRIGGER_VALUE)] = float(trigger)
    return tg


def reclutch(bundle: PipelineBundle) -> None:
    """Re-center the clutch at the current controller pose (arm does not move).

    Clearing the latched origin makes the next RUNNING frame re-capture p0 while
    keeping the running home at the last commanded EE pose.
    """
    if getattr(bundle.clutch, "_origin", None) is not None:
        bundle.clutch._origin = None


def step_pipeline_script(bits: dict, bundle: PipelineBundle, base_T_world, ctrl_tg):
    """Execute the graph one step directly; returns the flat 8D action (numpy)."""
    np = bits["np"]
    xform_tg = bits["TensorGroup"](bits["TransformMatrix"]())
    xform_tg[0] = np.asarray(base_T_world, dtype=np.float32)
    ctx = bits["ComputeContext"](
        execution_events=bits["ExecutionEvents"](
            reset=False, execution_state=bits["ExecutionState"].RUNNING
        )
    )
    outputs = bundle.combiner.execute_pipeline(
        {bundle.ctrl_leaf_name: {"value": ctrl_tg}, ANCHOR_LEAF: {"value": xform_tg}},
        context=ctx,
    )
    return np.from_dlpack(outputs["action"][0]).astype(np.float32)


##
# Environment / IK driver
##


def make_env(task_id: str, num_envs: int = 1):
    import gymnasium as gym

    import envs.isaac  # noqa: F401  (task registration side effects)
    from isaaclab.envs import ManagerBasedRLEnv

    entry = gym.spec(task_id).kwargs["env_cfg_entry_point"]
    mod_name, _, cls_name = entry.partition(":")
    import importlib

    cfg = getattr(importlib.import_module(mod_name), cls_name)()
    cfg.scene.num_envs = num_envs
    # XR rendering conflicts with extra camera sensors (isaaclab_teleop docs)
    for cam in ("wrist_cam", "wrist_cam_left", "wrist_cam_right", "overhead_cam", "front_cam"):
        if hasattr(cfg.scene, cam):
            setattr(cfg.scene, cam, None)
    return ManagerBasedRLEnv(cfg=cfg)


class So101IKDriver:
    """8D retargeter output -> DifferentialIK -> 6D absolute joint-position action (single-arm)
    or 12D for dual-arm (right arm: indices 0-5, left arm: indices 6-11)."""

    def __init__(self, env, gripper_open: float, gripper_close: float, ik_lambda: float):
        import torch
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
        from isaaclab.utils.math import matrix_from_quat, quat_inv, subtract_frame_transforms

        self.torch = torch
        self.subtract_frame_transforms = subtract_frame_transforms
        self.matrix_from_quat = matrix_from_quat
        self.quat_inv = quat_inv
        self.gripper_open = gripper_open
        self.gripper_close = gripper_close
        self.num_envs = env.num_envs
        self.device = env.device

        scene = env.unwrapped.scene
        # Detect single-arm vs dual-arm setup
        self.is_dual_arm = "robot_left" in scene and "robot_right" in scene
        if self.is_dual_arm:
            self._init_dual_arm(scene, ik_lambda)
        else:
            self._init_single_arm(scene, ik_lambda)

    def _init_single_arm(self, scene, ik_lambda):
        """Initialize single-arm driver."""
        robot = scene["robot"]
        body_names = list(robot.data.body_names)
        joint_names = list(robot.data.joint_names)

        def find(names, suffix, what):
            hits = [i for i, n in enumerate(names) if n.endswith(suffix)]
            if len(hits) != 1:
                raise KeyError(f"expected exactly one {what} ending '{suffix}', got {hits} in {names}")
            return hits[0]

        self.robot = robot
        self.ee_body_idx = find(body_names, EE_BODY_SUFFIX, "body")
        self.base_body_idx = find(body_names, "base_link", "body")
        self.ee_jacobi_idx = self.ee_body_idx - 1
        self.arm_joint_ids = [find(joint_names, j, "joint") for j in ARM_JOINTS]

        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls",
            ik_params={"lambda_val": ik_lambda},
        )
        self.ik = DifferentialIKController(ik_cfg, num_envs=self.num_envs, device=self.device)

    def _init_dual_arm(self, scene, ik_lambda):
        """Initialize dual-arm driver with separate IK per arm."""
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

        def setup_arm(robot, suffix):
            body_names = list(robot.data.body_names)
            joint_names = list(robot.data.joint_names)

            def find(names, suf, what):
                hits = [i for i, n in enumerate(names) if n.endswith(suf)]
                if len(hits) != 1:
                    raise KeyError(f"expected exactly one {what} ending '{suf}', got {hits} in {names}")
                return hits[0]

            ee_body_idx = find(body_names, EE_BODY_SUFFIX, "body")
            base_body_idx = find(body_names, "base_link", "body")
            ee_jacobi_idx = ee_body_idx - 1
            arm_joint_ids = [find(joint_names, j, "joint") for j in ARM_JOINTS]

            ik_cfg = DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls",
                ik_params={"lambda_val": ik_lambda},
            )
            ik = DifferentialIKController(ik_cfg, num_envs=self.num_envs, device=self.device)
            return {
                "robot": robot,
                "ee_body_idx": ee_body_idx,
                "base_body_idx": base_body_idx,
                "ee_jacobi_idx": ee_jacobi_idx,
                "arm_joint_ids": arm_joint_ids,
                "ik": ik,
            }

        self.robot_right = setup_arm(scene["robot_right"], "right")
        self.robot_left = setup_arm(scene["robot_left"], "left")
        self.ik = [self.robot_right["ik"], self.robot_left["ik"]]  # list for reset()

    def reset(self):
        if self.is_dual_arm:
            for ik in self.ik:
                ik.reset()
        else:
            self.ik.reset()

    def base_T_world(self):
        """(4, 4) base_link_T_world float32 — fed to the pipeline as world_T_anchor."""
        if self.is_dual_arm:
            robot = self.robot_right["robot"]
            base_body_idx = self.robot_right["base_body_idx"]
        else:
            robot = self.robot
            base_body_idx = self.base_body_idx

        pose = robot.data.body_pose_w.torch[:, base_body_idx]
        t = self.torch.eye(4, device=robot.device)
        t[:3, :3] = self.matrix_from_quat(pose[:, 3:7])[0]
        t[:3, 3] = pose[0, :3]
        return t.detach().cpu().numpy().astype("float32")

    def home_base_T_ee(self):
        """(4, 4) base_link_T_ee float64 at the current (reset) pose — clutch seed."""
        if self.is_dual_arm:
            robot = self.robot_right["robot"]
            ee_body_idx = self.robot_right["ee_body_idx"]
            base_body_idx = self.robot_right["base_body_idx"]
        else:
            robot = self.robot
            ee_body_idx = self.ee_body_idx
            base_body_idx = self.base_body_idx

        pose = robot.data.body_pose_w.torch
        ee_w, base_w = pose[:, ee_body_idx], pose[:, base_body_idx]
        pos_b, quat_b = self.subtract_frame_transforms(
            base_w[:, 0:3], base_w[:, 3:7], ee_w[:, 0:3], ee_w[:, 3:7]
        )
        t = self.torch.eye(4, device=robot.device, dtype=self.torch.float64)
        t[:3, :3] = self.matrix_from_quat(quat_b)[0].to(self.torch.float64)
        t[:3, 3] = pos_b[0].to(self.torch.float64)
        return t.detach().cpu().numpy()

    def _step_arm(self, robot_info, action8):
        """Compute IK for one arm."""
        robot = robot_info["robot"]
        ee_body_idx = robot_info["ee_body_idx"]
        base_body_idx = robot_info["base_body_idx"]
        ee_jacobi_idx = robot_info["ee_jacobi_idx"]
        arm_joint_ids = robot_info["arm_joint_ids"]
        ik = robot_info["ik"]

        torch = self.torch
        cmd = torch.as_tensor(action8, device=robot.device, dtype=torch.float32).unsqueeze(0)

        pose = robot.data.body_pose_w.torch
        ee_w, base_w = pose[:, ee_body_idx], pose[:, base_body_idx]
        ee_pos_b, ee_quat_b = self.subtract_frame_transforms(
            base_w[:, 0:3], base_w[:, 3:7], ee_w[:, 0:3], ee_w[:, 3:7]
        )

        jacobian = robot.data.body_link_jacobian_w.torch[:, ee_jacobi_idx, :, arm_joint_ids]
        rot_to_base = self.matrix_from_quat(self.quat_inv(base_w[:, 3:7]))
        jacobian[:, :3, :] = torch.bmm(rot_to_base, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(rot_to_base, jacobian[:, 3:, :])

        ik.set_command(cmd[:, 0:7])
        joint_pos_arm = robot.data.joint_pos.torch[:, arm_joint_ids]
        arm_targets = ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos_arm)
        limits = robot.data.joint_pos_limits.torch[:, arm_joint_ids]
        arm_targets = arm_targets.clamp(limits[..., 0], limits[..., 1])

        closedness = float(cmd[0, 7])
        gripper_target = self.gripper_open + closedness * (self.gripper_close - self.gripper_open)
        action = joint_pos_arm.new_zeros(robot.num_instances, len(arm_joint_ids) + 1)
        action[:, :-1] = arm_targets
        action[:, -1] = gripper_target
        return action

    def step(self, action8):
        """action8: (8,) [x y z qx qy qz qw closedness] expressed in base frame.
        Returns: (num_envs, 6) for single-arm or (num_envs, 12) for dual-arm."""
        if self.is_dual_arm:
            right_action = self._step_arm(self.robot_right, action8)
            left_action = self._step_arm(self.robot_left, action8)
            return self.torch.cat([right_action, left_action], dim=1)
        else:
            return self._step_arm(
                {
                    "robot": self.robot,
                    "ee_body_idx": self.ee_body_idx,
                    "base_body_idx": self.base_body_idx,
                    "ee_jacobi_idx": self.ee_jacobi_idx,
                    "arm_joint_ids": self.arm_joint_ids,
                    "ik": self.ik,
                },
                action8,
            )


def build_home_and_bundle(bits: dict, driver: So101IKDriver, source_mode: str) -> PipelineBundle:
    home = driver.home_base_T_ee()
    print(f"[setup] clutch home EE (base frame): {[round(v, 4) for v in home[:3, 3]]}")
    return build_so101_pipeline(bits, home_base_T_ee=home, source_mode=source_mode)


##
# --null-device dry run (headless GPU, no XR anything)
##


def run_null_device(args) -> int:
    import math

    bits = _import_isaacteleop_bits()  # fail fast, before Kit boots
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True)
    simulation_app = launcher.app
    rc = 1
    try:
        import numpy as np
        import torch

        print(f"[null-device] creating {args.task} (num_envs=1, cameras stripped) ...")
        env = make_env(args.task)
        env.reset()
        driver = So101IKDriver(env, args.gripper_open, args.gripper_close, args.ik_lambda)
        bundle = build_home_and_bundle(bits, driver, source_mode="script")
        cx, cy, cz = driver.home_base_T_ee()[:3, 3].tolist()
        r = 0.06
        ident = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        finite = True
        reclutch_step = max(args.steps // 2, 2)
        last = None
        for t in range(args.steps):
            frac = t / max(args.steps - 1, 1)
            phase = 2.0 * math.pi * 2.0 * frac
            pos = [
                cx + r * math.sin(phase),
                cy + r * (math.cos(phase) - 1.0),
                cz + 0.04 * math.sin(2.0 * phase),
            ]
            trigger = abs(math.sin(math.pi * frac))
            ctrl = make_controller_tensorgroup(bits, pos, ident, trigger, valid=True)
            action8 = step_pipeline_script(bits, bundle, driver.base_T_world(), ctrl)
            finite &= bool(np.isfinite(action8).all())
            joint_action = driver.step(torch.as_tensor(action8))
            finite &= bool(torch.isfinite(joint_action).all())
            env.step(joint_action)
            if t == reclutch_step:
                reclutch(bundle)
                print(f"[null-device] step {t}: squeeze -> clutch origin re-latched (arm held)")
            last = (action8, joint_action)

        a8, ja = last
        print(f"[null-device] final ee target  : {[round(float(v), 4) for v in a8[:7]]}")
        print(f"[null-device] final closedness : {float(a8[7]):.4f}")
        print(f"[null-device] final joint tgt  : {ja[0].detach().cpu().numpy().round(4).tolist()}")
        ok = finite and abs(float(a8[7])) < 1e-5
        print("[null-device] PASS" if ok else "[null-device] FAIL (non-finite values or bad gripper ramp)")
        env.close()
        rc = 0 if ok else 1
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        _fail(str(e))
    finally:
        simulation_app.close()
    return rc


##
# Live VR loop
##


def run_vr(args) -> int:
    if args.cloudxr != "off":
        # Pick up the resolved CloudXR env (XR_RUNTIME_JSON etc.) written by the
        # runtime launcher BEFORE Kit boots, so Kit's OpenXR loader finds the
        # bundled Monado/CloudXR runtime immediately.
        env_file = Path.home() / ".cloudxr" / "run" / "cloudxr.env"
        if env_file.is_file():
            import os

            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("export ") and "=" in line:
                    k, _, v = line[len("export "):].partition("=")
                    val = shlex.split(v)
                    os.environ[k.strip()] = val[0] if val else ""

    bits = _import_isaacteleop_bits()  # fail fast, before Kit boots
    from isaaclab.app import AppLauncher

    # Note: do NOT set OMNI_KIT_HEADLESS=1 — that suppresses the XR compositor,
    # which breaks CloudXR swapchain creation. AppLauncher(headless=True) is
    # sufficient to suppress the local window while keeping XR streaming alive.
    launcher = AppLauncher(headless=args.headless, xr=True)
    simulation_app = launcher.app

    # Keep XR display pipeline alive so CloudXR can create its video encoder
    # (nvstServer) and stream frames to the Quest. Previously this block disabled
    # renderer/enabled and xr/profile/display/enabled to save VRAM, but that
    # prevents the swapchain from being created → nvstServer[2] never initializes
    # → NVST_R_INVALID_STATE spam. RTX 4060 has enough VRAM (~5-6 GB free) for
    # the 4096×3584 per-eye unwarped textures CloudXR expects.
    # AppLauncher(headless=True) already suppresses the local window; no extra
    # renderer disable needed.

    # isaaclab_teleop depends on 'carb' (Kit SDK) which is only available
    # AFTER AppLauncher boots Kit — must import here, not before.
    try:
        from isaaclab_teleop import IsaacTeleopCfg
        from isaaclab_teleop.isaac_teleop_cfg import CLOUDXR_JS_ENV
        from isaaclab_teleop.isaac_teleop_device import create_isaac_teleop_device
        from isaaclab_teleop.xr_cfg import XrCfg
    except ModuleNotFoundError as e:
        _fail(
            f"isaaclab_teleop unavailable ({e}). Setup steps:\n"
            "  pip install 'isaaclab-teleop' --extra-index-url https://pypi.nvidia.com\n"
            "  (it ships inside the isaaclab 3.0.0b2 wheel already present in this venv)"
        )
    rc = 1
    follower: "FollowerSink | None" = None
    try:
        import torch

        print(f"[vr] creating {args.task} ...")
        env = make_env(args.task)
        env.reset()
        driver = So101IKDriver(env, args.gripper_open, args.gripper_close, args.ik_lambda)
        bundle = build_home_and_bundle(bits, driver, source_mode="xr")

        base_prim = "/World/envs/env_0" + BASE_PRIM_SUFFIX
        teleop_cfg = IsaacTeleopCfg(
            xr_cfg=XrCfg(anchor_pos=tuple(args.anchor_pos)),
            pipeline_builder=lambda: bundle.combiner,
            target_frame_prim_path=base_prim,
            app_name="SO101VRTeleop",
        )

        def on_reset():
            env.reset()
            driver.reset()

        if args.cloudxr == "off":
            cloudxr_env_file, auto_launch = None, False
        else:
            cloudxr_env_file, auto_launch = CLOUDXR_JS_ENV, args.cloudxr == "auto"
            if args.cloudxr == "external":
                import os

                os.environ.setdefault("ISAACLAB_CXR_SKIP_AUTOLAUNCH", "1")

        device = create_isaac_teleop_device(
            teleop_cfg,
            sim_device=str(env.device),
            callbacks={"RESET": on_reset},
            cloudxr_env_file=cloudxr_env_file,
            auto_launch_cloudxr=auto_launch,
        )
        CII = bits["ControllerInputIndex"]

        print(
            "\n[vr] waiting for the Quest ...\n"
            "  1. headset browser -> WebXR client (see VR_TELEOP_SETUP.md for the URL)\n"
            f"  2. server IP = this laptop's LAN IP, port 48322\n"
            "  3. accept the self-signed cert, Enter VR, then press Play/Start\n"
            "  (advance()==None means still waiting; tensors flow once connected)\n"
        )

        # Real arm sink — connected only if --follower-port is given
        if args.follower_port or args.dry_run_follower:
            follower = FollowerSink(
                port=args.follower_port or "/dev/ttyACM0",
                follower_id=args.follower_id,
                max_jump_rad=args.max_jump_rad,
                torque_limit_pct=args.follower_torque_limit,
                dry_run=args.dry_run_follower,
            ).connect()

        with device:
            squeeze_prev = False
            while simulation_app.is_running():
                action8 = device.advance()
                if action8 is None:
                    # Still waiting for the OpenXR session / client: keep the Kit
                    # event loop ticking so the XR session can come up.
                    simulation_app.update()
                    continue
                right = device._session_lifecycle.last_right_controller
                squeeze_now = bool(
                    right is not None
                    and not right.is_none
                    and float(right[int(CII.SQUEEZE_VALUE)]) > 0.5
                )
                if squeeze_now and not squeeze_prev:
                    reclutch(bundle)
                    print("[vr] grip press -> clutch re-centered")
                squeeze_prev = squeeze_now
                joint_action = driver.step(action8)
                env.step(joint_action)
                # Mirror joint targets to real arm (6 values: 5 arm + gripper, rad)
                if follower is not None:
                    follower.send(joint_action[0].cpu().numpy())
        env.close()
        if follower is not None:
            follower.close()
        rc = 0
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        if follower is not None:
            follower.close()
        _fail(str(e))
    finally:
        simulation_app.close()
    return rc


def main():
    args = parse_args()
    return run_null_device(args) if args.null_device else run_vr(args)


if __name__ == "__main__":
    raise SystemExit(main())
