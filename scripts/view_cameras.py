#!/usr/bin/env python
"""Tiled live view of all cameras defined in an SO-101 scene.

Enumerates every named camera in the compiled model (no hardcoded camera
list), renders each with a single mujoco.Renderer, tiles them into a grid
with OpenCV, and either shows a live window (stepping physics each frame)
or saves a single grid frame to PNG headless.

Usage:
    MUJOCO_GL=egl python scripts/view_cameras.py --env single --save /tmp/single_cams.png
    MUJOCO_GL=egl python scripts/view_cameras.py --env dual --save /tmp/dual_cams.png
    python scripts/view_cameras.py --env single   # live GUI window, ESC to quit
"""
import argparse
import math
import time

import cv2
import numpy as np
import mujoco

from _env_utils import scene_path


def load_model(env):
    model = mujoco.MjModel.from_xml_path(scene_path(env))
    data = mujoco.MjData(model)

    # Use the 'home' keyframe if the scene defines one, else default qpos.
    home_id = -1
    try:
        home_id = model.key("home").id
    except KeyError:
        home_id = -1
    if home_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)
    return model, data


def enumerate_cameras(model):
    names = []
    for i in range(model.ncam):
        name = model.camera(i).name
        names.append(name if name else f"cam_{i}")
    return names


def render_grid(model, data, renderer, cam_names, tile_w, tile_h):
    tiles = []
    for name in cam_names:
        renderer.update_scene(data, camera=name)
        img = renderer.render()  # RGB, (h, w, 3)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_bgr = cv2.resize(img_bgr, (tile_w, tile_h))
        cv2.putText(img_bgr, name, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0), 1, cv2.LINE_AA)
        tiles.append(img_bgr)

    n = len(tiles)
    cols = max(1, math.ceil(math.sqrt(n)))
    rows = max(1, math.ceil(n / cols))
    grid = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        r, c = divmod(idx, cols)
        grid[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = tile
    return grid


def main():
    parser = argparse.ArgumentParser(description="Tiled view of all cameras in an SO-101 scene")
    parser.add_argument("--env", choices=["single", "dual", "pick_lift", "pick_place", "cyl_grasp", "cyl_reach"], default="single")
    parser.add_argument("--width", type=int, default=480, help="Per-tile render width")
    parser.add_argument("--height", type=int, default=360, help="Per-tile render height")
    parser.add_argument("--save", type=str, default=None,
                         help="Render one grid frame to this PNG path headless and exit")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    model, data = load_model(args.env)
    cam_names = enumerate_cameras(model)
    if not cam_names:
        raise RuntimeError(f"No cameras defined in scene for env={args.env}")
    print(f"Found {len(cam_names)} cameras: {cam_names}")

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    if args.save:
        # Step physics briefly to settle before capturing.
        for _ in range(10):
            mujoco.mj_step(model, data)
        grid = render_grid(model, data, renderer, cam_names, args.width, args.height)
        cv2.imwrite(args.save, grid)
        print(f"Saved camera grid to {args.save} (shape={grid.shape}, "
              f"variance={float(grid.astype(np.float32).var()):.3f})")
        return

    dt = 1.0 / args.fps
    win_name = f"SO-101 cameras ({args.env})"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    try:
        while True:
            t0 = time.time()
            for _ in range(max(1, int(round(dt / model.opt.timestep)))):
                mujoco.mj_step(model, data)
            grid = render_grid(model, data, renderer, cam_names, args.width, args.height)
            cv2.imshow(win_name, grid)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elapsed = time.time() - t0
            time.sleep(max(0.0, dt - elapsed))
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
