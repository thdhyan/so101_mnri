"""Procedural scene pieces for the pick-lift task: cube entity, table geoms.

Table/floor are not authored as a separate entity — mjlab's SceneCfg.spec_fn
hook lets us inject static geoms directly into the compiled scene spec,
matching the fixed table_top/legs in envs/mujoco/so101_single_arm_pick_lift's
scene.xml.
"""

from pathlib import Path

import mujoco

ROBOT_XML = str(Path(__file__).parent / "assets" / "so101_follower.xml")

CUBE_SIZE = (0.025, 0.025, 0.025)
CUBE_MASS_NOMINAL = 0.08
CUBE_RGBA = (0.85, 0.2, 0.2, 1.0)

TABLE_TOP_POS = (0.35, 0.0, 0.80)
TABLE_TOP_SIZE = (0.40, 0.35, 0.02)
TABLE_SURFACE_RGBA = (0.85, 0.75, 0.6, 1.0)


def get_cube_spec(
  size: tuple[float, float, float] = CUBE_SIZE,
  mass: float = CUBE_MASS_NOMINAL,
  rgba: tuple[float, float, float, float] = CUBE_RGBA,
) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="cube")
  body.add_freejoint(name="cube_joint")
  body.add_geom(
    name="cube_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=size,
    mass=mass,
    rgba=rgba,
    condim=4,
    friction=(1.0, 0.005, 0.0001),
  )
  return spec


def add_table(spec: mujoco.MjSpec) -> None:
  """SceneCfg.spec_fn callback: adds a static table (top + 4 legs) to the
  compiled scene, matching envs/mujoco/so101_single_arm_pick_lift's scene.xml.
  """
  top = spec.worldbody.add_body(name="table_top_body", pos=TABLE_TOP_POS)
  top.add_geom(
    name="table_top",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=TABLE_TOP_SIZE,
    rgba=TABLE_SURFACE_RGBA,
    condim=3,
  )
  leg_positions = {
    "table_leg_fl": (0.70, 0.30, 0.40),
    "table_leg_fr": (0.70, -0.30, 0.40),
    "table_leg_bl": (0.02, 0.30, 0.40),
    "table_leg_br": (0.02, -0.30, 0.40),
  }
  for name, pos in leg_positions.items():
    leg = spec.worldbody.add_body(name=f"{name}_body", pos=pos)
    leg.add_geom(
      name=name,
      type=mujoco.mjtGeom.mjGEOM_CYLINDER,
      size=(0.02, 0.40, 0.0),
    )
