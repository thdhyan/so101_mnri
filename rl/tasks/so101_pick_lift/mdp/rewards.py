from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .commands import LiftCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _grasp_mask(
  env: ManagerBasedRlEnv,
  object_name: str,
  grasp_dist_threshold: float,
  gripper_joint_threshold: float,
  asset_cfg: SceneEntityCfg,
  gripper_joint_cfg: SceneEntityCfg,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  dist = torch.norm(ee_pos_w - obj.data.root_link_pos_w, dim=-1)
  gripper_qpos = robot.data.joint_pos[:, gripper_joint_cfg.joint_ids].squeeze(-1)
  return (dist < grasp_dist_threshold) & (gripper_qpos > gripper_joint_threshold)


def reach_ee_object(
  env: ManagerBasedRlEnv,
  object_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Negative distance from EE to object — always active."""
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  return -torch.norm(ee_pos_w - obj.data.root_link_pos_w, dim=-1)


def grasp_bonus(
  env: ManagerBasedRlEnv,
  object_name: str,
  grasp_dist_threshold: float,
  gripper_joint_threshold: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  gripper_joint_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  grasped = _grasp_mask(
    env,
    object_name,
    grasp_dist_threshold,
    gripper_joint_threshold,
    asset_cfg,
    gripper_joint_cfg,
  )
  return grasped.float()


def lift_progress(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  grasp_dist_threshold: float,
  gripper_joint_threshold: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  gripper_joint_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Height-gain progress toward lift_threshold, gated on grasp."""
  command = cast(LiftCommand, env.command_manager.get_term(command_name))
  obj: Entity = env.scene[object_name]
  height_gain = obj.data.root_link_pos_w[:, 2] - command.init_height
  progress = torch.clamp(height_gain / command.cfg.lift_threshold, min=0.0, max=1.0)
  grasped = _grasp_mask(
    env,
    object_name,
    grasp_dist_threshold,
    gripper_joint_threshold,
    asset_cfg,
    gripper_joint_cfg,
  )
  return progress * grasped.float()


def alive_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Constant per-step cost. Weighted larger than action_rate_l2 so the
  policy can't profitably stall near the goal to farm dense reward instead
  of committing to success and terminating."""
  return torch.ones(env.num_envs, device=env.device)


def success_bonus(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  grasp_dist_threshold: float,
  gripper_joint_threshold: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  gripper_joint_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  command = cast(LiftCommand, env.command_manager.get_term(command_name))
  obj: Entity = env.scene[object_name]
  height_gain = obj.data.root_link_pos_w[:, 2] - command.init_height
  grasped = _grasp_mask(
    env,
    object_name,
    grasp_dist_threshold,
    gripper_joint_threshold,
    asset_cfg,
    gripper_joint_cfg,
  )
  success = grasped & (height_gain > command.cfg.lift_threshold)
  return success.float()
