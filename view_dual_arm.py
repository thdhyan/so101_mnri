"""
Dual arm: interactive 3D viewer + tiled camera feed.

Layout (2×2):
  [ left_wrist ] [ right_wrist ]
  [ overhead_left ] [ overhead_right ]

Usage:
    conda activate isaac6
    cd /home/thakk100/Projects/so101_mnri
    python view_dual_arm.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import mujoco
import mujoco.viewer
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from so101_dual_arm_env.env import SO101DualArmEnv, DualArmEnvConfig, CameraConfig

# ── Edit camera positions here ───────────────────────────────────────────────
cfg = DualArmEnvConfig(
    task="push",
    left_wrist_camera=CameraConfig(
        "left_wrist",
        pos=(0.000, -0.043, -0.042),
        euler=(29.0, -6.0, 0.5),
        fov=75.0,
    ),
    right_wrist_camera=CameraConfig(
        "right_wrist",
        pos=(0.000, -0.043, -0.042),
        euler=(29.0, -6.0, 6.0),
        fov=75.0,
    ),
    overhead_cameras=[
        CameraConfig("overhead_left",  pos=(0.637,  0.027, 1.248), euler=(-26.0,  0.5, -90.5), fov=80.0),
        CameraConfig("overhead_right", pos=(0.530, -0.491, 1.061), euler=(-67.0, -5.0,-159.5), fov=80.0),
    ],
)
# ─────────────────────────────────────────────────────────────────────────────

CAM_GRID = [
    ["left_wrist",    "right_wrist"],
    ["overhead_left", "overhead_right"],
]
CAM_W, CAM_H = 640, 480

env   = SO101DualArmEnv(cfg)
model = env.model
data  = env.data
env.reset()

renderer = env._renderer

# ── Matplotlib tiled window (2×2) ────────────────────────────────────────────
nrows, ncols = len(CAM_GRID), len(CAM_GRID[0])
fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
fig.suptitle("SO-101 Dual Arm — Camera Feeds", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])

im_handles = {}
for r, row in enumerate(CAM_GRID):
    for c, cam in enumerate(row):
        ax = axes[r][c]
        renderer.update_scene(data, camera=cam)
        img = renderer.render().copy()
        h = ax.imshow(img)
        ax.set_title(cam, fontsize=10)
        ax.axis("off")
        im_handles[cam] = h

plt.ion()
plt.show()

# ── Passive 3D viewer ────────────────────────────────────────────────────────
print("3D viewer open. Tiles update live.")
print("Close either window to exit.")

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth   = 160
    viewer.cam.elevation = -20
    viewer.cam.distance  = 2.2
    viewer.cam.lookat[:] = [0.35, 0.0, 0.95]

    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
    viewer.opt.sitegroup[3] = True

    while viewer.is_running() and plt.fignum_exists(fig.number):
        mujoco.mj_step(model, data)
        viewer.sync()

        for cam, h in im_handles.items():
            renderer.update_scene(data, camera=cam)
            h.set_data(renderer.render().copy())

        fig.canvas.draw_idle()
        fig.canvas.flush_events()

env.close()
plt.close("all")
print("Done.")
