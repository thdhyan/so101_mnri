from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class LiftCommand(CommandTerm):
  """Tracks per-env cube spawn (with xy DR) and the height it must be lifted above.

  Not a spatial target like mjlab's LiftingCommand — pick-lift's success
  condition is height-gain relative to the object's own reset height, so
  this term's job is bookkeeping (`init_height`) rather than sampling a 3D
  goal position.
  """

  cfg: LiftCommandCfg

  def __init__(self, cfg: LiftCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.object: Entity = env.scene[cfg.entity_name]
    self.init_height = torch.zeros(self.num_envs, device=self.device)
    self.metrics["height_gain"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.init_height.unsqueeze(-1)

  def _update_metrics(self) -> None:
    height = self.object.data.root_link_pos_w[:, 2]
    self.metrics["height_gain"] = height - self.init_height

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    r = self.cfg.object_pose_range
    nominal = torch.tensor(self.cfg.nominal_pos, device=self.device)
    lower = nominal + torch.tensor([r.x[0], r.y[0], r.z[0]], device=self.device)
    upper = nominal + torch.tensor([r.x[1], r.y[1], r.z[1]], device=self.device)
    pos = sample_uniform(lower, upper, (n, 3), device=self.device)
    pos = pos + self._env.scene.env_origins[env_ids]

    quat = torch.zeros(n, 4, device=self.device)
    quat[:, 0] = 1.0  # identity orientation, matches env.py reference
    pose = torch.cat([pos, quat], dim=-1)
    velocity = torch.zeros(n, 6, device=self.device)

    self.object.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    self.object.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)

    self.init_height[env_ids] = pos[:, 2]

  def _update_command(self) -> None:
    pass

  def _debug_vis_impl(self, visualizer) -> None:  # noqa: ANN001
    pass


@dataclass(kw_only=True)
class LiftCommandCfg(CommandTermCfg):
  entity_name: str
  lift_threshold: float = 0.05
  nominal_pos: tuple[float, float, float] = (0.45, 0.0, 0.83)

  @dataclass
  class ObjectPoseRangeCfg:
    x: tuple[float, float] = (-0.06, 0.06)
    y: tuple[float, float] = (-0.06, 0.06)
    z: tuple[float, float] = (0.0, 0.0)

  object_pose_range: ObjectPoseRangeCfg

  def build(self, env: ManagerBasedRlEnv) -> LiftCommand:
    return LiftCommand(self, env)
