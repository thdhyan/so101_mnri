"""Success termination terms for the SO-101 Isaac Lab tasks.

Each mirrors its MuJoCo env's success criterion with the tolerances fixed in
MJLAB_INTEGRATION.md: 1 cm position tolerance, 10 deg orientation tolerance.
Success fires ``terminated=True``; ``mdp.time_out`` fires ``truncated``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import quat_rotate

from .rewards import (
    _ee_pos_w,
    _object_pos_w,
    _object_quat_w,
    cyl_end_positions_w,
    cyl_grasp_bonus,
    grasp_bonus,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def pick_lift_success(
    env: ManagerBasedRLEnv,
    lift_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Grasped AND lifted above the height-gain threshold."""
    lifted = lift_progress_term(env, lift_threshold, object_cfg, ee_frame_cfg)
    grasped = grasp_bonus(env, 0.035, object_cfg, ee_frame_cfg)
    return lifted & grasped.bool()


def lift_progress_term(
    env: ManagerBasedRLEnv,
    lift_threshold: float,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
) -> torch.Tensor:
    pos_w = _object_pos_w(env, object_cfg)
    obj = env.scene[object_cfg.name]
    init_z = obj.data.default_root_state.torch[:, 2]
    return (pos_w[:, 2] - init_z) > lift_threshold


def pick_place_success(
    env: ManagerBasedRLEnv,
    target_pos: tuple[float, float, float],
    pos_tol: float = 0.01,
    rot_tol_deg: float = 10.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Object within 1 cm XY of target, tilted < 10 deg, released and near-static."""
    pos_w = _object_pos_w(env, object_cfg)
    target = torch.tensor(target_pos, device=env.device).repeat(env.num_envs, 1)
    pos_ok = torch.linalg.norm(pos_w[:, :2] - target[:, :2], dim=1) < pos_tol

    # Tilt of the object's local +Z axis away from world +Z.
    quat_w = _object_quat_w(env, object_cfg)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1)
    z_world = quat_rotate(quat_w, z_axis)
    tilt_cos = torch.clamp(z_world[:, 2], -1.0, 1.0)
    rot_ok = torch.rad2deg(torch.acos(tilt_cos)) < rot_tol_deg

    obj = env.scene[object_cfg.name]
    released = torch.linalg.norm(pos_w - _ee_pos_w(env, ee_frame_cfg), dim=1) > 0.05
    static = torch.linalg.norm(obj.data.root_lin_vel_w.torch, dim=1) < 0.05
    return pos_ok & rot_ok & released & static


def cyl_grasp_success(
    env: ManagerBasedRLEnv,
    lift_threshold: float,
    half_length: float = 0.075,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_left_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame_left"),
    ee_frame_right_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame_right"),
) -> torch.Tensor:
    """Both ends grasped AND cylinder lifted above the height-gain threshold."""
    pos_w = _object_pos_w(env, object_cfg)
    obj = env.scene[object_cfg.name]
    init_z = obj.data.default_root_state.torch[:, 2]
    lifted = (pos_w[:, 2] - init_z) > lift_threshold
    grasped_a = cyl_grasp_bonus(env, 0.035, "a", half_length, object_cfg, ee_frame_left_cfg).bool()
    grasped_b = cyl_grasp_bonus(env, 0.035, "b", half_length, object_cfg, ee_frame_right_cfg).bool()
    return lifted & grasped_a & grasped_b


class cyl_reach_success(ManagerTermBase):
    """Both grippers within 1 cm of their targets, held for ``hold_steps``.

    Targets are ``z_offset`` above the (static) cylinder's end sites, so they
    move with any cylinder pose randomization automatically.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._count = None

    def reset(self, env_ids: torch.Tensor | None = None):
        if self._count is not None and env_ids is not None:
            self._count[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        z_offset: float,
        hold_steps: int = 10,
        pos_tol: float = 0.01,
        half_length: float = 0.075,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        ee_frame_left_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame_left"),
        ee_frame_right_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame_right"),
    ) -> torch.Tensor:
        if self._count is None or self._count.shape[0] != env.num_envs:
            self._count = torch.zeros(env.num_envs, device=env.device)
        end_a, end_b = cyl_end_positions_w(env, object_cfg, half_length)
        off = torch.tensor([0.0, 0.0, z_offset], device=env.device)
        d_l = torch.linalg.norm(end_a + off - _ee_pos_w(env, ee_frame_left_cfg), dim=1)
        d_r = torch.linalg.norm(end_b + off - _ee_pos_w(env, ee_frame_right_cfg), dim=1)
        in_band = (d_l < pos_tol) & (d_r < pos_tol)
        self._count = torch.where(in_band, self._count + 1, torch.zeros_like(self._count))
        return self._count >= hold_steps
