"""
Single arm: interactive 3D viewer + tiled camera feed.

Layout:
  [  wrist  ] [ outside_left ] [ outside_right ]

Usage:
    conda activate isaac6
    cd /home/thakk100/Projects/so101_mnri
    python view_single_arm.py
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

from envs.mujoco.so101_single_arm.env import SO101SingleArmEnv, SingleArmEnvConfig, CameraConfig

# ── Edit camera positions here ───────────────────────────────────────────────
# wrist: pos/euler in gripper body frame. euler=(0,0,0) looks toward jaw (-Z gripper).
#   Tweak euler roll/pitch/yaw (degrees) and pos (metres) to aim the camera.
# outside cams: pos in world frame, lookat = world point to face.
cfg = SingleArmEnvConfig(
    task="push",
    wrist_camera=CameraConfig(
        "wrist",
        pos=(0.000, -0.043, -0.042),
        euler=(29.0, -6.0, 0.5),
        fov=75.0,
    ),
    outside_cameras=[
        CameraConfig("outside_left",  pos=(0.462, -0.110, 1.273), euler=(0.5,  3.5,  91.0), fov=65.0),
        CameraConfig("outside_right", pos=(0.382, -0.304, 1.066), euler=(65.0, 1.0,  -3.0), fov=65.0),
    ],
)
# ─────────────────────────────────────────────────────────────────────────────

CAM_NAMES = ["wrist", "outside_left", "outside_right"]
CAM_W, CAM_H = 640, 480

env   = SO101SingleArmEnv(cfg)
model = env.model
data  = env.data
env.reset()

renderer = env._renderer

# ── Matplotlib tiled window ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(CAM_NAMES), figsize=(6 * len(CAM_NAMES), 5))
fig.suptitle("SO-101 Single Arm — Camera Feeds", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])

im_handles = []
for ax, cam in zip(axes, CAM_NAMES):
    renderer.update_scene(data, camera=cam)
    img = renderer.render().copy()
    h = ax.imshow(img)
    ax.set_title(cam, fontsize=10)
    ax.axis("off")
    im_handles.append(h)

plt.ion()
plt.show()

# ── Passive 3D viewer ────────────────────────────────────────────────────────
print("3D viewer open. Move cameras in the scene, tiles update live.")
print("Close the matplotlib window OR the 3D viewer to exit.")

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth   = 160
    viewer.cam.elevation = -20
    viewer.cam.distance  = 1.8
    viewer.cam.lookat[:] = [0.35, 0.0, 0.95]

    # Show body frames (RGB axes = X/Y/Z) on every body incl. gripper
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
    viewer.opt.sitegroup[3] = True

    while viewer.is_running() and plt.fignum_exists(fig.number):
        mujoco.mj_step(model, data)
        viewer.sync()

        for h, cam in zip(im_handles, CAM_NAMES):
            renderer.update_scene(data, camera=cam)
            h.set_data(renderer.render().copy())

        fig.canvas.draw_idle()
        fig.canvas.flush_events()

env.close()
plt.close("all")
print("Done.")
