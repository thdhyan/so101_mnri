"""
Interactive camera tuner for SO-101 environments.
Sliders for pos (x,y,z) and euler (roll,pitch,yaw) on every camera.
Camera tile updates live. Print button outputs final CameraConfig values.

Usage:
    conda activate isaac6
    cd /home/thakk100/Projects/so101_mnri
    python tune_cameras.py --env single
    python tune_cameras.py --env dual
"""

import argparse
import sys
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import math

sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# Shared helpers (duplicated from env.py to avoid import side-effects)
# ---------------------------------------------------------------------------

def _euler_deg_to_quat(roll, pitch, yaw):
    r, p, y = math.radians(roll), math.radians(pitch), math.radians(yaw)
    cr, sr = math.cos(r/2), math.sin(r/2)
    cp, sp = math.cos(p/2), math.sin(p/2)
    cy, sy = math.cos(y/2), math.sin(y/2)
    return np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ])


# ---------------------------------------------------------------------------
# Camera descriptor — one per camera to tune
# ---------------------------------------------------------------------------

class CamTuner:
    """Holds current state for one camera: pos xyz + euler rpy (degrees)."""
    def __init__(self, model, cam_name, init_pos, init_euler):
        self.model  = model
        self.name   = cam_name
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        self.pos    = list(init_pos)
        self.euler  = list(init_euler)
        self._apply()

    def _apply(self):
        if self.cam_id < 0:
            return
        self.model.cam_pos[self.cam_id]  = np.array(self.pos)
        self.model.cam_quat[self.cam_id] = _euler_deg_to_quat(*self.euler)

    def set_pos(self, axis, val):
        self.pos[axis] = val
        self._apply()

    def set_euler(self, axis, val):
        self.euler[axis] = val
        self._apply()

    def config_str(self):
        return (
            f'CameraConfig("{self.name}",\n'
            f'    pos=({self.pos[0]:.3f}, {self.pos[1]:.3f}, {self.pos[2]:.3f}),\n'
            f'    euler=({self.euler[0]:.1f}, {self.euler[1]:.1f}, {self.euler[2]:.1f}),\n'
            f'    fov={self.model.cam_fovy[self.cam_id]:.1f},\n)'
        )


# ---------------------------------------------------------------------------
# Main tuner UI
# ---------------------------------------------------------------------------

def run_tuner(env_name: str):
    if env_name == "single":
        from envs.mujoco.so101_single_arm.env import SO101SingleArmEnv, SingleArmEnvConfig
        env = SO101SingleArmEnv(SingleArmEnvConfig())
        cam_names  = ["wrist", "outside_left", "outside_right"]
        tuners = [
            CamTuner(env.model, "wrist",
                     init_pos=(0.000, -0.043, -0.042), init_euler=(29.0, -6.0, 0.5)),
            CamTuner(env.model, "outside_left",
                     init_pos=(-0.30, -0.55, 1.05), init_euler=(-74.7, 48.7, 11.6)),
            CamTuner(env.model, "outside_right",
                     init_pos=(0.90, -0.55, 1.05), init_euler=(-74.7, -44.0, -10.7)),
        ]
    else:
        from envs.mujoco.so101_dual_arm.env import SO101DualArmEnv, DualArmEnvConfig
        env = SO101DualArmEnv(DualArmEnvConfig())
        cam_names = ["left_wrist", "right_wrist", "overhead_left", "overhead_right"]
        tuners = [
            CamTuner(env.model, "left_wrist",
                     init_pos=(0.000, -0.043, -0.042), init_euler=(29.0, -6.0, 0.5)),
            CamTuner(env.model, "right_wrist",
                     init_pos=(0.000, -0.043, -0.042), init_euler=(29.0, -6.0, 0.5)),
            CamTuner(env.model, "overhead_left",
                     init_pos=(-0.10, 0.70, 1.30), init_euler=(41.2, 23.7, 155.3)),
            CamTuner(env.model, "overhead_right",
                     init_pos=(-0.10, -0.70, 1.30), init_euler=(-41.2, 23.7, 24.7)),
        ]

    env.reset()
    model, data = env.model, env.data
    renderer = mujoco.Renderer(model, 480, 640)

    n_cams = len(cam_names)

    # ── Figure layout ────────────────────────────────────────────────────────
    # Top half: camera tiles
    # Bottom half: sliders (6 per camera, stacked)
    SLIDER_H   = 0.028
    SLIDER_PAD = 0.005
    N_SLIDERS  = 6
    bottom_space = n_cams * (N_SLIDERS * (SLIDER_H + SLIDER_PAD) + 0.04) + 0.06

    fig_h = 5.5 + bottom_space
    fig = plt.figure(figsize=(max(14, 5 * n_cams), fig_h))
    fig.suptitle(f"Camera Tuner — {env_name}   |   [Print] to copy config", fontsize=12)

    tile_top    = 1.0 - (5.5 / fig_h)
    slider_area = 1.0 - tile_top - 0.04

    # Camera tiles
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

    # Slider panel per camera
    all_sliders = []
    slider_objs = []   # keep references so GC doesn't kill them

    def make_sliders(tuner, cam_idx, n_cams):
        col_w    = 1.0 / n_cams
        col_x    = cam_idx * col_w + 0.08 * col_w
        ax_w     = col_w * 0.82
        block_h  = N_SLIDERS * (SLIDER_H + SLIDER_PAD) + 0.035
        block_y  = 0.01 + (n_cams - 1 - cam_idx) % 1 * 0  # stack rows if needed

        # Vertical position: divide slider_area equally per camera column
        base_y   = 0.005 + cam_idx * 0  # all in one row, per column

        sliders = []
        labels  = ["pos-x", "pos-y", "pos-z", "roll°", "pitch°", "yaw°"]
        inits   = tuner.pos[:] + tuner.euler[:]
        ranges  = [(-3.0, 3.0), (-3.0, 3.0), (-0.5, 3.0),
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
            sliders.append(sl)
            slider_objs.append(sl)

        return sliders

    for i, tuner in enumerate(tuners):
        make_sliders(tuner, i, n_cams)

    # Print button
    btn_ax = fig.add_axes([0.40, 0.005, 0.20, 0.03])
    btn = Button(btn_ax, "Print CameraConfig values")

    def on_print(_):
        print("\n" + "=" * 60)
        print("Paste these into view_single_arm.py / env.py:")
        print("=" * 60)
        for t in tuners:
            print(t.config_str())
        print("=" * 60 + "\n")

    btn.on_clicked(on_print)
    slider_objs.append(btn)

    # ── Refresh tiles ────────────────────────────────────────────────────────
    def refresh_tiles():
        mujoco.mj_forward(model, data)
        for h, cam in zip(im_handles, cam_names):
            renderer.update_scene(data, camera=cam)
            h.set_data(renderer.render().copy())
        fig.canvas.draw_idle()

    refresh_tiles()
    plt.ion()
    plt.show()

    # ── Passive 3D viewer ────────────────────────────────────────────────────
    print("Sliders in matplotlib window | 3D viewer for reference")
    print("Press [Print CameraConfig values] to get code to paste.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth   = 160
        viewer.cam.elevation = -20
        viewer.cam.distance  = 1.8 if env_name == "single" else 2.2
        viewer.cam.lookat[:] = [0.35, 0.0, 0.95]
        viewer.opt.frame     = mujoco.mjtFrame.mjFRAME_BODY
        viewer.opt.sitegroup[3] = True

        while viewer.is_running() and plt.fignum_exists(fig.number):
            mujoco.mj_step(model, data)
            viewer.sync()
            fig.canvas.flush_events()

    env.close()
    plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["single", "dual"], default="single")
    args = parser.parse_args()
    run_tuner(args.env)
