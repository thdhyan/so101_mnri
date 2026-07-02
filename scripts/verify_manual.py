"""
Manual verification: opens the interactive 3D viewer for an SO-101 env plus
a live tiled window of every camera, so a human can eyeball robot geometry,
camera framing, and physics stability before trusting the env in training.

Usage:
    conda activate so101
    python scripts/verify_manual.py --env single
    python scripts/verify_manual.py --env dual
    python scripts/verify_manual.py --env pick_lift
    python scripts/verify_manual.py --env pick_place
    python scripts/verify_manual.py --env cyl_grasp
    python scripts/verify_manual.py --env cyl_reach
    python scripts/verify_manual.py --env single --headless-check   # no GUI, CI-safe

--headless-check runs the same automated checks as verify_envs.py (compile,
keyframe load, render every camera, reset/step) and exits with a nonzero
code on failure — use this in CI; use the default interactive mode by hand.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import mujoco
import mujoco.viewer
import numpy as np


def load_env(env_name: str, task: str = "push"):
    if env_name == "single":
        from envs.mujoco.so101_single_arm.env import SO101SingleArmEnv, SingleArmEnvConfig
        return SO101SingleArmEnv(SingleArmEnvConfig(task=task))
    if env_name == "dual":
        from envs.mujoco.so101_dual_arm.env import SO101DualArmEnv, DualArmEnvConfig
        return SO101DualArmEnv(DualArmEnvConfig(task=task))
    if env_name == "pick_lift":
        from envs.mujoco.so101_single_arm_pick_lift.env import SO101SingleArmPickLiftEnv
        return SO101SingleArmPickLiftEnv()
    if env_name == "pick_place":
        from envs.mujoco.so101_single_arm_pick_place.env import SO101SingleArmPickAndPlaceEnv
        return SO101SingleArmPickAndPlaceEnv()
    if env_name == "cyl_grasp":
        from envs.mujoco.so101_dual_arm_cylinder_grasp.env import SO101DualArmCylinderGraspEnv
        return SO101DualArmCylinderGraspEnv()
    from envs.mujoco.so101_dual_arm_cylinder_reach.env import SO101DualArmCylinderReachEnv
    return SO101DualArmCylinderReachEnv()


def discover_cameras(model) -> list[str]:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        for i in range(model.ncam)
    ]


def headless_check(env_name: str) -> bool:
    print(f"[{env_name}] compiling + resetting...")
    env = load_env(env_name)
    obs, _ = env.reset()
    cams = discover_cameras(env.model)
    ok = True
    print(f"[{env_name}] cameras: {cams}")
    for cam in cams:
        img = obs["images"][cam]
        std = float(img.std())
        status = "OK" if std > 1.0 else "FAIL (near-blank image)"
        if std <= 1.0:
            ok = False
        print(f"  {cam}: shape={img.shape} std={std:.2f}  {status}")

    action = np.zeros(env.action_space.shape, dtype=np.float32)
    obs, reward, term, trunc, info = env.step(action)
    print(f"[{env_name}] step OK, reward={reward:.4f}")
    env.close()
    print(f"[{env_name}] {'PASSED' if ok else 'FAILED'}")
    return ok


def interactive(env_name: str, cols: int = 3, tile_w: int = 480, tile_h: int = 360):
    env = load_env(env_name)
    env.reset()
    model, data = env.model, env.data
    cam_names = discover_cameras(model)
    print(f"[{env_name}] cameras: {cam_names}")
    renderer = env._renderer

    win = f"SO-101 {env_name} — cameras (ESC to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def tile_grid():
        imgs = []
        for cam in cam_names:
            renderer.update_scene(data, camera=cam)
            img = cv2.resize(renderer.render(), (tile_w, tile_h))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.putText(img, cam, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            imgs.append(img)
        ncols = min(cols, len(imgs))
        nrows = (len(imgs) + ncols - 1) // ncols
        pad = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        rows = []
        for r in range(nrows):
            row = imgs[r * ncols:(r + 1) * ncols]
            row += [pad] * (ncols - len(row))
            rows.append(np.hstack(row))
        return np.vstack(rows)

    print("3D viewer + camera grid open. Close either window, or press ESC in the")
    print("camera grid, to exit. Robot holds its 'home' keyframe pose; physics runs.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
        viewer.opt.sitegroup[3] = True
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            cv2.imshow(win, tile_grid())
            if cv2.waitKey(1) & 0xFF == 27:
                break
    cv2.destroyAllWindows()
    env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=["single", "dual", "pick_lift", "pick_place", "cyl_grasp", "cyl_reach"], default="single")
    ap.add_argument("--headless-check", action="store_true",
                     help="Run automated checks only, no GUI (CI-safe).")
    args = ap.parse_args()

    if args.headless_check:
        ok = headless_check(args.env)
        sys.exit(0 if ok else 1)
    interactive(args.env)


if __name__ == "__main__":
    main()
