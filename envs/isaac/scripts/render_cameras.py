"""
Headless camera-render for the SO-101 Isaac Lab tasks.

Creates each registered task with --enable_cameras, resets, settles a few
steps, then saves every camera's RGB output as a PNG plus a tiled grid:

    envs/isaac/tasks/<task>/images/<camera>.png
    envs/isaac/tasks/<task>/images/cameras.png

Run from the repo root:

    python -m envs.isaac.scripts.render_cameras                # all tasks
    python -m envs.isaac.scripts.render_cameras --tasks SO101-PickLift-Single-v0
    python -m envs.isaac.scripts.render_cameras --width 320 --height 240
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

TASKS = {
    "SO101-PickLift-Single-v0": "pick_lift",
    "SO101-PickPlace-Single-v0": "pick_place",
    "SO101-CylReach-Single-v0": "cylinder_reach_single",
    "SO101-CylGrasp-Dual-v0": "cylinder_grasp",
    "SO101-CylReach-Dual-v0": "cylinder_reach",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="*", default=list(TASKS))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--settle-steps", type=int, default=10)
    args = parser.parse_args()

    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True, enable_cameras=True)
    simulation_app = launcher.app

    try:
        import cv2
        import numpy as np
        import torch
        from isaaclab.envs import ManagerBasedRLEnv

        import envs.isaac  # noqa: F401

        for task_id in args.tasks:
            short = TASKS[task_id]
            module = importlib.import_module(f"envs.isaac.tasks.{short}.env_cfg")
            cfg_cls = {
                "pick_lift": "PickLiftEnvCfg",
                "pick_place": "PickPlaceEnvCfg",
                "cylinder_reach_single": "CylReachSingleEnvCfg",
                "cylinder_grasp": "CylGraspEnvCfg",
                "cylinder_reach": "CylReachEnvCfg",
            }[short]
            cfg = getattr(module, cfg_cls)()
            cfg.scene.num_envs = 1
            for attr in [a for a in dir(cfg.scene) if a.endswith("_cam")]:
                cam = getattr(cfg.scene, attr)
                if cam is not None:
                    cam.width, cam.height = args.width, args.height

            env = ManagerBasedRLEnv(cfg=cfg)
            env.reset()  # home pose; no stepping — screenshots of the reset state

            out_dir = REPO_ROOT / "envs" / "isaac" / "tasks" / short.replace("SO101-", "").lower() / "images"
            out_dir.mkdir(parents=True, exist_ok=True)

            names, tiles = [], []
            for name, sensor in env.scene.sensors.items():
                if "cam" not in name:
                    continue
                sensor.update(dt=0.0)
                rgb = sensor.data.output["rgb"][0, ..., :3].cpu().numpy()
                if rgb.max() == 0:
                    print(f"[warn] {task_id}/{name}: blank render")
                tile = rgb.copy()
                cv2.putText(tile, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                names.append(name)
                tiles.append(tile)
                cv2.imwrite(str(out_dir / f"{name}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            ncols = 2
            nrows = (len(tiles) + ncols - 1) // ncols
            while len(tiles) < nrows * ncols:
                tiles.append(np.zeros_like(tiles[0]))
            grid = np.vstack([np.hstack(tiles[i * ncols:(i + 1) * ncols]) for i in range(nrows)])
            cv2.imwrite(str(out_dir / "cameras.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
            print(f"[ok] {task_id}: {names} -> {out_dir}/cameras.png")
            env.close()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
