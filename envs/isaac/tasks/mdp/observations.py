"""Custom observation terms for the SO-101 Isaac Lab tasks.

State-based policy observations (images stay available through the camera
sensors but are not concatenated into the policy vector, matching the
MuJoCo envs where `images` is a separate obs dict entry).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_rotate, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _root_frame(env, robot_cfg, pos_w):
    robot = env.scene[robot_cfg.name]
    pos_b, quat_b = subtract_frame_transforms(
        robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, pos_w
    )
    return pos_b, quat_b


def object_pose_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object pose (pos 3 + quat 4, xyzw) in the robot's root frame."""
    obj = env.scene[object_cfg.name]
    pos_b, quat_b = _root_frame(env, robot_cfg, obj.data.root_pos_w.torch[:, :3])
    return torch.cat([pos_b, quat_b], dim=1)


def cylinder_ends_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    half_length: float = 0.075,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Both cylinder end-site positions (6,) in the robot's root frame."""
    obj = env.scene[object_cfg.name]
    pos_w = obj.data.root_pos_w.torch[:, :3]
    quat_w = obj.data.root_quat_w.torch[:, :4]
    end_a_w = pos_w + quat_rotate(quat_w, torch.tensor([0.0, -half_length, 0.0], device=env.device).repeat(env.num_envs, 1))
    end_b_w = pos_w + quat_rotate(quat_w, torch.tensor([0.0, half_length, 0.0], device=env.device).repeat(env.num_envs, 1))
    end_a_b, _ = _root_frame(env, robot_cfg, end_a_w)
    end_b_b, _ = _root_frame(env, robot_cfg, end_b_w)
    return torch.cat([end_a_b, end_b_b], dim=1)
