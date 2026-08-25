from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _gripper_pos(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  return robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)


def object_pos_b(
  env: ManagerBasedRlEnv,
  object_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Object position relative to robot base origin, world-axis-aligned."""
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  return obj.data.root_link_pos_w - robot.data.root_link_pos_w


def ee_to_object(
  env: ManagerBasedRlEnv,
  object_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """tcp_pos - obj_pos, world-axis-aligned (matches env.py's tcp_to_obj)."""
  obj: Entity = env.scene[object_name]
  ee_pos_w = _gripper_pos(env, asset_cfg)
  return ee_pos_w - obj.data.root_link_pos_w


def is_grasped(
  env: ManagerBasedRlEnv,
  object_name: str,
  grasp_dist_threshold: float,
  gripper_joint_threshold: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  gripper_joint_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """0/1 float: gripper within grasp_dist_threshold of object AND gripper closed."""
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  ee_pos_w = _gripper_pos(env, asset_cfg)
  dist = torch.norm(ee_pos_w - obj.data.root_link_pos_w, dim=-1)
  gripper_qpos = robot.data.joint_pos[:, gripper_joint_cfg.joint_ids].squeeze(-1)
  grasped = (dist < grasp_dist_threshold) & (gripper_qpos > gripper_joint_threshold)
  return grasped.float().unsqueeze(-1)
