#!/usr/bin/env python
"""Render a consistent free-camera "overview" screenshot for an SO-101 env.

Not one of the env's own named cameras — a fixed 3rd-person MjvCamera
looking at the table from the same relative angle every time, so overview
shots stay visually consistent as new envs are added (see ENVS.md).

Usage:
    MUJOCO_GL=egl python scripts/capture_overview.py --env pick_lift --save images/pick_lift_overview.png
    MUJOCO_GL=egl python scripts/capture_overview.py --env cyl_grasp --save images/cyl_grasp_overview.png
"""
import argparse

import cv2
import mujoco
import numpy as np

from _env_utils import scene_path

# Same lookat/elevation/azimuth for every env — only distance varies (dual-arm
# scenes are wider, so they need to pull back further to fit both arms).
LOOKAT = (0.35, 0.0, 0.5)
ELEVATION = -20
AZIMUTH = 120
DIST_SINGLE = 1.4
DIST_DUAL = 1.7

DUAL_ENVS = {"dual", "cyl_grasp", "cyl_reach"}


def load_model(env: str):
    model = mujoco.MjModel.from_xml_path(scene_path(env))
    data = mujoco.MjData(model)
    try:
        home_id = model.key("home").id
        mujoco.mj_resetDataKeyframe(model, data, home_id)
    except KeyError:
        pass
    mujoco.mj_forward(model, data)
    return model, data


def render_overview(model, data, width=640, height=480, dist=None) -> np.ndarray:
    cam = mujoco.MjvCamera()
    cam.lookat[:] = LOOKAT
    cam.elevation = ELEVATION
    cam.azimuth = AZIMUTH
    cam.distance = dist

    renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.update_scene(data, camera=cam)
    img = renderer.render()
    renderer.close()
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", required=True,
                     choices=["single", "dual", "pick_lift", "pick_place", "cyl_grasp", "cyl_reach"])
    ap.add_argument("--save", required=True, help="Output PNG path")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--dist", type=float, default=None,
                     help="Override camera distance (default: 1.4 single-arm envs, 1.7 dual-arm envs)")
    args = ap.parse_args()

    dist = args.dist if args.dist is not None else (DIST_DUAL if args.env in DUAL_ENVS else DIST_SINGLE)

    model, data = load_model(args.env)
    img = render_overview(model, data, args.width, args.height, dist)
    cv2.imwrite(args.save, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"Saved {args.save} (shape={img.shape}, std={float(img.std()):.2f})")


if __name__ == "__main__":
    main()
