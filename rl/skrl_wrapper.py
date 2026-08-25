"""skrl VecEnv wrapper for mjlab's ManagerBasedRlEnv.

Modeled on mjlab's own RslRlVecEnvWrapper
(third_party/mjlab/src/mjlab/rl/vecenv_wrapper.py), adapted for skrl's
Wrapper interface: single flat torch.Tensor observations (not TensorDict),
and real gymnasium.Space objects (mjlab's own Space/Box classes are not
gymnasium-compatible, so they're converted here).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from skrl.envs.wrappers.torch.base import Wrapper

from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.spaces import Box as MjlabBox
from mjlab.utils.spaces import Space as MjlabSpace


def _to_gym_space(space: MjlabSpace) -> gym.Space:
  if isinstance(space, MjlabBox):
    low = np.broadcast_to(np.array(space.low, dtype=np.float32), space.shape).copy()
    high = np.broadcast_to(np.array(space.high, dtype=np.float32), space.shape).copy()
    return gym.spaces.Box(low=low, high=high, shape=space.shape, dtype=np.float32)
  return gym.spaces.Box(
    low=-np.inf, high=np.inf, shape=space.shape, dtype=np.float32
  )


class SkrlVecEnvWrapper(Wrapper):
  """Wraps a mjlab ManagerBasedRlEnv for skrl's torch agents.

  Uses the "actor" observation group as the flat policy observation (matches
  env_cfg.py's actor/critic group split, where both currently share the same
  terms — see rl/tasks/<task>/env_cfg.py for the observation term list).
  """

  def __init__(self, env: ManagerBasedRlEnv, obs_group: str = "actor"):
    super().__init__(env)
    self._obs_group = obs_group

    unbatched_obs_space = env.single_observation_space.spaces[obs_group]
    unbatched_action_space = env.single_action_space
    self._observation_space = _to_gym_space(unbatched_obs_space)
    self._action_space = _to_gym_space(unbatched_action_space)

  @property
  def observation_space(self) -> gym.Space:
    return self._observation_space

  @property
  def action_space(self) -> gym.Space:
    return self._action_space

  def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
    obs_dict, extras = self._env.reset()
    return obs_dict[self._obs_group], extras

  def step(
    self, actions: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    obs_dict, reward, terminated, truncated, extras = self._env.step(actions)
    # skrl's memory slots are (num_envs, 1) for scalars — match the shape
    # convention of skrl's own isaaclab wrapper (1-D tensors would broadcast
    # to (N, N) in PPO's time-limit bootstrapping).
    return (
      obs_dict[self._obs_group],
      reward.view(-1, 1),
      terminated.view(-1, 1),
      truncated.view(-1, 1),
      extras,
    )

  def state(self) -> torch.Tensor | None:
    return None

  def render(self, *args, **kwargs) -> Any:
    return self._env.render()

  def close(self) -> None:
    self._env.close()
