"""Capture overview + camera-grid PNGs for a MuJoCo env into images/.

    MUJOCO_GL=egl python scripts/capture_env_images.py --env push_t
    MUJOCO_GL=egl python scripts/capture_env_images.py --env cube_push_ramp --no-overview

Writes images/<env>_overview.png (free-camera view) and
images/<env>_cameras.png (tiled grid of the env's cameras).
"""

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ENVS = {
    "single": ("envs.mujoco.so101_single_arm.env", "SO101SingleArmEnv"),
    "dual": ("envs.mujoco.so101_dual_arm.env", "SO101DualArmEnv"),
    "pick_lift": ("envs.mujoco.so101_single_arm_pick_lift.env", "SO101SingleArmPickLiftEnv"),
    "pick_place": ("envs.mujoco.so101_single_arm_pick_place.env", "SO101SingleArmPickAndPlaceEnv"),
    "cyl_grasp": ("envs.mujoco.so101_dual_arm_cylinder_grasp.env", "SO101DualArmCylinderGraspEnv"),
    "cyl_reach": ("envs.mujoco.so101_dual_arm_cylinder_reach.env", "SO101DualArmCylinderReachEnv"),
    "push_t": ("envs.mujoco.so101_single_arm_push_t.env", "SO101SingleArmPushTEnv"),
    "cube_push_ramp": ("envs.mujoco.so101_single_arm_cube_push_ramp.env", "SO101SingleArmCubePushRampEnv"),
    "cube_push_bridge": ("envs.mujoco.so101_single_arm_cube_push_bridge.env", "SO101SingleArmCubePushBridgeEnv"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=list(ENVS), required=True)
    parser.add_argument("--out", default="images")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--skip-overview", action="store_true")
    args = parser.parse_args()

    import mujoco
    from PIL import Image

    module, cls = ENVS[args.env]
    env = getattr(importlib.import_module(module), cls)()
    env.reset()
    mujoco.mj_forward(env.model, env.data)
    renderer = mujoco.Renderer(env.model, args.height, args.width)

    out = Path(args.out)
    out.mkdir(exist_ok=True)

    if not args.skip_overview:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0.45, 0.0, 0.85]
        cam.distance = 1.2
        cam.azimuth = 135.0
        cam.elevation = -25.0
        renderer.update_scene(env.data, camera=cam)
        Image.fromarray(renderer.render()).save(out / f"{args.env}_overview.png")
        print(f"saved {out}/{args.env}_overview.png")

    cam_names = [
        mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        for i in range(env.model.ncam)
    ]
    tiles = []
    for name in cam_names:
        renderer.update_scene(env.data, camera=name)
        img = renderer.render().copy()
        tiles.append(img)
        print(f"  {name}: std={img.std():.1f}")

    ncols = 2
    nrows = (len(tiles) + ncols - 1) // ncols
    h, w = tiles[0].shape[:2]
    grid = np.zeros((nrows * h, ncols * w, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, ncols)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = tile
    Image.fromarray(grid).save(out / f"{args.env}_cameras.png")
    print(f"saved {out}/{args.env}_cameras.png ({cam_names})")
    env.close()


if __name__ == "__main__":
    main()
