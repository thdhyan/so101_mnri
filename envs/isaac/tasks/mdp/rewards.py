"""Custom reward terms for the SO-101 Isaac Lab tasks.

All distance-based terms use tanh kernels (1 - tanh(d / std)) like the
Isaac Lab manipulation tasks; binary grasp terms mirror the MuJoCo envs'
distance-threshold grasp checks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_rotate, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import FrameTransformer


def _ee_pos_w(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_pos_w.torch[..., 0, :]


def _object_pos_w(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    return obj.data.root_pos_w.torch[:, :3]


def _object_quat_w(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    return obj.data.root_quat_w.torch[:, :4]


def ee_object_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Tanh-kernel reward for the end-effector approaching the object."""
    d = torch.linalg.norm(_object_pos_w(env, object_cfg) - _ee_pos_w(env, ee_frame_cfg), dim=1)
    return 1.0 - torch.tanh(d / std)


def grasp_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Binary 1/0 reward while the end-effector is within ``threshold`` of the object."""
    d = torch.linalg.norm(_object_pos_w(env, object_cfg) - _ee_pos_w(env, ee_frame_cfg), dim=1)
    return (d < threshold).float()


def lift_progress(
    env: ManagerBasedRLEnv,
    lift_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Object height gain scaled to [0, 3], gated on the object being grasped."""
    pos_w = _object_pos_w(env, object_cfg)
    obj: RigidObject = env.scene[object_cfg.name]
    init_z = obj.data.default_root_state.torch[:, 2]
    gain = (pos_w[:, 2] - init_z) / lift_threshold
    grasped = grasp_bonus(env, 0.035, object_cfg, ee_frame_cfg)
    return grasped * torch.clamp(gain, 0.0, 3.0)


def place_progress(
    env: ManagerBasedRLEnv,
    std: float,
    target_pos: tuple[float, float, float],
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Tanh-kernel reward on the object's XY distance to the target, gated on grasping."""
    target = torch.tensor(target_pos, device=env.device).repeat(env.num_envs, 1)
    d_xy = torch.linalg.norm(_object_pos_w(env, object_cfg)[:, :2] - target[:, :2], dim=1)
    grasped = grasp_bonus(env, 0.035, object_cfg, ee_frame_cfg)
    return grasped * (1.0 - torch.tanh(d_xy / std))


def cyl_end_positions_w(
    env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg, half_length: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """World positions of the cylinder's two end sites (local ±Y axis)."""
    pos = _object_pos_w(env, object_cfg)
    quat = _object_quat_w(env, object_cfg)
    end_a = pos + quat_rotate(quat, torch.tensor([0.0, -half_length, 0.0], device=env.device).repeat(env.num_envs, 1))
    end_b = pos + quat_rotate(quat, torch.tensor([0.0, half_length, 0.0], device=env.device).repeat(env.num_envs, 1))
    return end_a, end_b


def cyl_end_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    end: str,
    half_length: float = 0.075,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Tanh-kernel reward for one end-effector approaching one cylinder end."""
    end_a, end_b = cyl_end_positions_w(env, object_cfg, half_length)
    end_pos = end_a if end == "a" else end_b
    d = torch.linalg.norm(end_pos - _ee_pos_w(env, ee_frame_cfg), dim=1)
    return 1.0 - torch.tanh(d / std)


def cyl_grasp_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    end: str,
    half_length: float = 0.075,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Binary 1/0 reward while one end-effector grasps one cylinder end."""
    end_a, end_b = cyl_end_positions_w(env, object_cfg, half_length)
    end_pos = end_a if end == "a" else end_b
    d = torch.linalg.norm(end_pos - _ee_pos_w(env, ee_frame_cfg), dim=1)
    return (d < threshold).float()


def cyl_lift_progress(
    env: ManagerBasedRLEnv,
    lift_threshold: float,
    half_length: float = 0.075,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_left_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame_left"),
    ee_frame_right_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame_right"),
) -> torch.Tensor:
    """Cylinder height gain scaled to [0, 3], gated on BOTH ends being grasped."""
    pos_w = _object_pos_w(env, object_cfg)
    obj: RigidObject = env.scene[object_cfg.name]
    init_z = obj.data.default_root_state.torch[:, 2]
    gain = (pos_w[:, 2] - init_z) / lift_threshold
    grasped_a = cyl_grasp_bonus(env, 0.035, "a", half_length, object_cfg, ee_frame_left_cfg)
    grasped_b = cyl_grasp_bonus(env, 0.035, "b", half_length, object_cfg, ee_frame_right_cfg)
    return grasped_a * grasped_b * torch.clamp(gain, 0.0, 3.0)


def cyl_reach_target_distance(
    env: ManagerBasedRLEnv,
    std: float,
    end: str,
    z_offset: float,
    half_length: float = 0.075,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Tanh-kernel reward for reaching a point ``z_offset`` above one cylinder end."""
    end_a, end_b = cyl_end_positions_w(env, object_cfg, half_length)
    end_pos = end_a if end == "a" else end_b
    target = end_pos + torch.tensor([0.0, 0.0, z_offset], device=env.device)
    d = torch.linalg.norm(target - _ee_pos_w(env, ee_frame_cfg), dim=1)
    return 1.0 - torch.tanh(d / std)


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object position expressed in the robot's root frame."""
    robot = env.scene[robot_cfg.name]
    pos_w = _object_pos_w(env, object_cfg)
    pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, pos_w
    )
    return pos_b
