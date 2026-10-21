from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .commands import LiftCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def success(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  grasp_dist_threshold: float,
  gripper_joint_threshold: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  gripper_joint_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """terminated=True when grasped AND height_gain > lift_threshold."""
  command = cast(LiftCommand, env.command_manager.get_term(command_name))
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]

  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  dist = torch.norm(ee_pos_w - obj.data.root_link_pos_w, dim=-1)
  gripper_qpos = robot.data.joint_pos[:, gripper_joint_cfg.joint_ids].squeeze(-1)
  grasped = (dist < grasp_dist_threshold) & (gripper_qpos > gripper_joint_threshold)

  height_gain = obj.data.root_link_pos_w[:, 2] - command.init_height
  return grasped & (height_gain > command.cfg.lift_threshold)
