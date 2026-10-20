"""
Random-action and zero-action validation for the SO-101 Isaac Lab tasks.

    python -m envs.isaac.scripts.validate_actions
"""

import argparse
import importlib
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]

TASKS = {
    "SO101-PickLift-Single-v0": ("pick_lift", "PickLiftEnvCfg", 6),
    "SO101-PickPlace-Single-v0": ("pick_place", "PickPlaceEnvCfg", 6),
    "SO101-CylReach-Single-v0": ("cylinder_reach_single", "CylReachSingleEnvCfg", 6),
    "SO101-CylGrasp-Dual-v0": ("cylinder_grasp", "CylGraspEnvCfg", 12),
    "SO101-CylReach-Dual-v0": ("cylinder_reach", "CylReachEnvCfg", 12),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    from isaaclab.app import AppLauncher

    # cameras must be enabled: the task scenes spawn camera sensors, and
    # Kit crashes at sim start when they exist without --enable_cameras
    launcher = AppLauncher(headless=True, enable_cameras=True)
    simulation_app = launcher.app

    all_ok = True
    try:
        import torch

        import envs.isaac  # noqa: F401
        from isaaclab.envs import ManagerBasedRLEnv

        print("Isaac Lab random/zero-action validation:")
        for task_id, (short, cfg_cls, act_dim) in TASKS.items():
            module = importlib.import_module(f"envs.isaac.tasks.{short}.env_cfg")
            for mode in ("zero", "random"):
                cfg = getattr(module, cfg_cls)()
                cfg.scene.num_envs = 2
                env = ManagerBasedRLEnv(cfg=cfg)
                obs, _ = env.reset()
                gen = torch.Generator(device="cpu").manual_seed(0)
                total = 0.0
                finite = True
                for i in range(args.steps):
                    if mode == "zero":
                        act = torch.zeros(env.num_envs, act_dim, device=env.device)
                    else:
                        lo = env.action_manager.action_term_dim
                        act = torch.rand(env.num_envs, act_dim, generator=gen) * 2 - 1
                        act = act.to(env.device)
                    obs, rew, term, trunc, extras = env.step(act)
                    total += float(rew.sum())
                    flat = obs["policy"]
                    finite &= bool(torch.isfinite(flat).all()) and bool(torch.isfinite(rew).all())
                    if term.any() or trunc.any():
                        obs, _ = env.reset()
                env.close()
                print(f"  [{task_id}] {mode:6s}: return={total:.2f} finite={finite}")
                all_ok &= finite
    finally:
        simulation_app.close()
    print("ALL OK" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
