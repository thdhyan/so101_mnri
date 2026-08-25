"""
One-shot URDF → USD conversion for the SO-101 arm.

Run from the repo root with the project venv:

    python -m envs.isaac.convert_urdf

Converts `robots/so101/so101.urdf` (the canonical robot definition) into
`envs/isaac/assets/so101.usd`. Requires a working Isaac Sim pip install and
accepts the Kit EULA (OMNI_KIT_ACCEPT_EULA=YES — exported by .envrc).

Notes:
    - fix_base=True: the arm's base is world-fixed (table-mounted, as in MuJoCo).
    - merge_fixed_joints=False: keeps `gripper_frame_link` (the TCP link used
      for EE frames and wrist-camera attachment) as its own rigid body.
    - joint_drive: position targets with the servo-like PD gains; final gains
      are overridden at runtime by the ImplicitActuatorCfg in so101.py.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
URDF_PATH = REPO_ROOT / "robots" / "so101" / "so101.urdf"
USD_PATH = REPO_ROOT / "envs" / "isaac" / "assets" / "so101.usd"


def main():
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})

    import omni.log

    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    cfg = UrdfConverterCfg(
        asset_path=str(URDF_PATH),
        usd_dir=str(USD_PATH.parent),
        usd_file_name=USD_PATH.stem,
        fix_base=True,
        merge_fixed_joints=False,
        self_collision=False,
        force_usd_conversion=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=100.0,
                damping=2.5,
            ),
        ),
    )
    converter = UrdfConverter(cfg)
    print(f"USD written to: {converter.usd_path}")

    app.close()


if __name__ == "__main__":
    main()
