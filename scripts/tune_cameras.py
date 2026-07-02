#!/usr/bin/env python
"""Interactive camera tuner for any SO-101 env.

Sliders for pos (x,y,z) and euler roll/pitch/yaw on every camera defined in
the scene (discovered dynamically via model.ncam — no hardcoded camera
list, so this works unmodified as new envs/cameras are added). Camera tiles
update live. "Print" button dumps CameraConfig-ready values for pasting
into an env's CameraConfig / scene.xml.

Usage:
    conda activate so101
    python scripts/tune_cameras.py --env single
    python scripts/tune_cameras.py --env pick_lift
    python scripts/tune_cameras.py --env cyl_grasp
"""
import argparse
import math

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import mujoco
import mujoco.viewer
import numpy as np

from _env_utils import scene_path


def _euler_deg_to_quat(roll, pitch, yaw):
    r, p, y = math.radians(roll), math.radians(pitch), math.radians(yaw)
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


class CamTuner:
    """Holds current pos/euler state for one camera and applies it to the model."""

    def __init__(self, model, cam_name):
        self.model = model
        self.name = cam_name
        self.cam_id = model.camera(cam_name).id
        self.pos = list(model.cam_pos[self.cam_id])
        self.euler = list(_quat_to_euler_deg(model.cam_quat[self.cam_id]))

    def set_pos(self, axis, val):
        self.pos[axis] = val
        self.model.cam_pos[self.cam_id] = np.array(self.pos)

    def set_euler(self, axis, val):
        self.euler[axis] = val
        self.model.cam_quat[self.cam_id] = _euler_deg_to_quat(*self.euler)

    def config_str(self):
        return (
            f'CameraConfig("{self.name}",\n'
            f'    pos=({self.pos[0]:.3f}, {self.pos[1]:.3f}, {self.pos[2]:.3f}),\n'
            f'    euler=({self.euler[0]:.1f}, {self.euler[1]:.1f}, {self.euler[2]:.1f}),\n'
            f'    fov={self.model.cam_fovy[self.cam_id]:.1f},\n)'
        )


def _quat_to_euler_deg(quat) -> np.ndarray:
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=np.float64))
    mat = mat.reshape(3, 3)
    # XYZ intrinsic euler matching _euler_deg_to_quat's roll/pitch/yaw convention.
    sy = -mat[2, 0]
    sy = max(-1.0, min(1.0, sy))
    pitch = math.asin(sy)
    if abs(sy) < 0.9999999:
        roll = math.atan2(mat[2, 1], mat[2, 2])
        yaw = math.atan2(mat[1, 0], mat[0, 0])
    else:
        roll = math.atan2(-mat[1, 2], mat[1, 1])
        yaw = 0.0
    return np.degrees([roll, pitch, yaw])


def discover_cameras(model) -> list[str]:
    return [model.camera(i).name for i in range(model.ncam)]


def run_tuner(env_name: str):
    model = mujoco.MjModel.from_xml_path(scene_path(env_name))
    data = mujoco.MjData(model)
    try:
        home_id = model.key("home").id
        mujoco.mj_resetDataKeyframe(model, data, home_id)
    except KeyError:
        pass
    mujoco.mj_forward(model, data)

    cam_names = discover_cameras(model)
    if not cam_names:
        raise RuntimeError(f"No cameras defined in scene for env={env_name}")
    tuners = [CamTuner(model, name) for name in cam_names]
    n_cams = len(cam_names)

    renderer = mujoco.Renderer(model, 480, 640)

    SLIDER_H = 0.028
    SLIDER_PAD = 0.005
    N_SLIDERS = 6
    bottom_space = N_SLIDERS * (SLIDER_H + SLIDER_PAD) + 0.06

    fig_h = 5.5 + bottom_space
    fig = plt.figure(figsize=(max(14, 5 * n_cams), fig_h))
    fig.suptitle(f"Camera Tuner — {env_name}   |   [Print] to copy config", fontsize=12)

    tile_top = 1.0 - (5.5 / fig_h)
    slider_area = 1.0 - tile_top - 0.04

    tile_axes, im_handles = [], []
    for i, cam in enumerate(cam_names):
        ax = fig.add_axes([i / n_cams + 0.01, tile_top, 0.98 / n_cams - 0.02, tile_top * 0.92])
        renderer.update_scene(data, camera=cam)
        img = renderer.render().copy()
        h = ax.imshow(img)
        ax.set_title(cam, fontsize=9)
        ax.axis("off")
        tile_axes.append(ax)
        im_handles.append(h)

    slider_objs = []

    def make_sliders(tuner, cam_idx):
        col_w = 1.0 / n_cams
        col_x = cam_idx * col_w + 0.08 * col_w
        ax_w = col_w * 0.82

        labels = ["pos-x", "pos-y", "pos-z", "roll°", "pitch°", "yaw°"]
        inits = tuner.pos[:] + tuner.euler[:]
        ranges = [(-3.0, 3.0), (-3.0, 3.0), (-0.5, 3.0),
                  (-180, 180), (-180, 180), (-180, 180)]

        for si, (lbl, init, (lo, hi)) in enumerate(zip(labels, inits, ranges)):
            row_y = slider_area * 0.92 - si * (SLIDER_H + SLIDER_PAD) - 0.01
            ax_s = fig.add_axes([col_x, row_y, ax_w, SLIDER_H])
            sl = Slider(ax_s, f"{tuner.name}\n{lbl}", lo, hi,
                        valinit=init, valstep=0.001 if "°" not in lbl else 0.5)
            sl.label.set_fontsize(7)
            sl.valtext.set_fontsize(7)

            def on_change(val, _si=si, _t=tuner):
                if _si < 3:
                    _t.set_pos(_si, val)
                else:
                    _t.set_euler(_si - 3, val)
                refresh_tiles()

            sl.on_changed(on_change)
            slider_objs.append(sl)

    for i, tuner in enumerate(tuners):
        make_sliders(tuner, i)

    btn_ax = fig.add_axes([0.40, 0.005, 0.20, 0.03])
    btn = Button(btn_ax, "Print CameraConfig values")

    def on_print(_):
        print("\n" + "=" * 60)
        print(f"Paste these into {env_name}'s env.py CameraConfig list:")
        print("=" * 60)
        for t in tuners:
            print(t.config_str())
        print("=" * 60 + "\n")

    btn.on_clicked(on_print)
    slider_objs.append(btn)

    def refresh_tiles():
        mujoco.mj_forward(model, data)
        for h, cam in zip(im_handles, cam_names):
            renderer.update_scene(data, camera=cam)
            h.set_data(renderer.render().copy())
        fig.canvas.draw_idle()

    refresh_tiles()
    plt.ion()
    plt.show()

    print("Sliders in matplotlib window | 3D viewer for reference")
    print("Press [Print CameraConfig values] to get code to paste.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 160
        viewer.cam.elevation = -20
        viewer.cam.distance = 1.8 if model.stat.extent < 1.3 else 2.2
        viewer.cam.lookat[:] = [0.35, 0.0, 0.95]
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
        viewer.opt.sitegroup[3] = True

        while viewer.is_running() and plt.fignum_exists(fig.number):
            mujoco.mj_step(model, data)
            viewer.sync()
            fig.canvas.flush_events()

    plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True,
                         choices=["single", "dual", "pick_lift", "pick_place", "cyl_grasp", "cyl_reach"])
    args = parser.parse_args()
    run_tuner(args.env)
