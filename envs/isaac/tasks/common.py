"""Shared scene helpers for the SO-101 Isaac Lab tasks.

Camera conventions match the MuJoCo envs: euler XYZ intrinsic in degrees,
camera looks along its local -Z with +Y up (Isaac Lab "opengl" offset
convention), so tuned MuJoCo euler values transfer directly. All camera
poses are plain config fields — customize pos/euler/fov per task via the
env cfg's ``__post_init__`` or from the training CLI config.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils.math import convert_quat, quat_from_euler_xyz

from envs.isaac.so101 import TABLE_TOP_POS, TABLE_TOP_SIZE

# Camera defaults (mirror envs/mujoco/*/assets/scene.xml + env.py CameraConfig).
OVERHEAD_CAM_POS = (0.40, 0.0, 1.45)
OVERHEAD_CAM_EULER = (0.0, 0.0, 90.0)  # straight down
FRONT_CAM_POS = (0.40, -1.0, 1.05)
FRONT_CAM_EULER = (75.0, 0.0, 0.0)  # front view of the whole workspace
WRIST_CAM_POS = (0.0, 0.0, 0.02)  # in gripper_frame_link frame
WRIST_CAM_EULER = (0.0, 0.0, 0.0)
# Wrist-camera orientation as explicit local quats (wxyz), calibrated offline
# against the TCP frame at home pose so each camera aims at the task object
# (the OffsetCfg euler path mis-aims — see git history). Recompute with
# `lookat` math if the home pose or object placement changes.
WRIST_CAM_ROT_SINGLE = (-0.3731, 0.6212, 0.5913, -0.3540)    # single arm -> cube below-front
WRIST_CAM_ROT_LEFT = (-0.3763, 0.1987, 0.7351, 0.5278)       # left arm: Y-mirror conjugate of the verified right quat
WRIST_CAM_ROT_RIGHT = (-0.3763, -0.1987, 0.7351, -0.5278)    # right arm -> cylinder between arms (verified live)

CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FOV = 75.0  # vertical FOV, degrees
CAM_APERTURE = 20.955  # USD default horizontal aperture


def euler_deg_to_quat(roll_deg: float, pitch_deg: float, yaw_deg: float) -> tuple[float, float, float, float]:
    """XYZ-intrinsic euler (degrees) -> quaternion for CameraCfg offset rot.

    NOTE: `CameraCfg.OffsetCfg.rot` is documented as (x, y, z, w), but the
    value flows into the USD spawner which expects **(w, x, y, z)** — verified
    empirically (an identity quat authored as xyzw came out as a 180 deg
    Z-flip on the spawned camera prim). We therefore return wxyz here.
    """
    quat_wxyz = quat_from_euler_xyz(
        torch.tensor([math.radians(roll_deg)]),
        torch.tensor([math.radians(pitch_deg)]),
        torch.tensor([math.radians(yaw_deg)]),
    )[0]
    w, x, y, z = (float(v) for v in quat_wxyz)
    return (w, x, y, z)


def vfov_to_focal_length(vfov_deg: float, width: int, height: int) -> float:
    """Vertical FOV (deg) -> USD pinhole focal length (same units as the
    horizontal aperture), matching MuJoCo's fovy semantics."""
    hfov_rad = 2.0 * math.atan(math.tan(math.radians(vfov_deg) / 2.0) * (width / height))
    return (CAM_APERTURE / 2.0) / math.tan(hfov_rad / 2.0)


def camera_cfg(
    prim_path: str,
    pos: tuple[float, float, float],
    euler: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rot_wxyz: tuple[float, float, float, float] | None = None,
    fov: float = CAM_FOV,
    width: int = CAM_WIDTH,
    height: int = CAM_HEIGHT,
) -> CameraCfg:
    """A customizable RGB camera (vectorized rendering built in).

    ``prim_path`` may be a world prim (global cameras) or a robot body prim
    (wrist cameras, e.g. ``{ENV_REGEX_NS}/Robot/gripper_frame_link``).
    Orientation: pass ``euler`` (XYZ-intrinsic degrees, MuJoCo convention —
    works for world-fixed cams) or ``rot_wxyz`` (explicit local quaternion,
    wxyz — recommended for body-attached cams; see WRIST_CAM_ROT_* notes).
    """
    quat = rot_wxyz if rot_wxyz is not None else euler_deg_to_quat(*euler)
    # Body-attached cams must track their parent prim (else the pose/render
    # stays at the pre-reset spawn pose — the arm's zero pose, not home).
    track = "gripper_frame_link" in prim_path
    return CameraCfg(
        prim_path=prim_path,
        update_period=0,
        update_latest_camera_pose=track,
        height=height,
        width=width,
        data_types=["rgb"],
        offset=CameraCfg.OffsetCfg(
            pos=pos,
            rot=quat,
            convention="opengl",  # -Z forward, +Y up — same as MuJoCo cameras
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=vfov_to_focal_length(fov, width, height),
            clipping_range=(0.01, 20.0),
        ),
    )


def table_cfg() -> AssetBaseCfg:
    """Static tabletop matching the MuJoCo scenes (top surface at z = 0.82)."""
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=TABLE_TOP_POS),
        spawn=sim_utils.CuboidCfg(
            size=TABLE_TOP_SIZE,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.75, 0.6)),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )


def ground_and_light() -> list[AssetBaseCfg]:
    return [
        AssetBaseCfg(prim_path="/World/GroundPlane", spawn=GroundPlaneCfg()),
        AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        ),
    ]
