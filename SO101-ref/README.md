# SO101-ref: SO-101 base mount (L-mount)

A 3D-printed L-shaped mount for the SO-101 follower arm base. The top plate is the SO-ARM100
`arm_base` (overhead-cam tabs kept), with pockets that the base's underside ribs drop into. A
10 mm back panel (180 wide, 183.8 long) gives the clamps a surface. The base bolts down with
#8-32 x 3/4" button heads into captive hex nuts from below.

**Agents: read `HANDOFF.md` §9–§10 first.** The screw length vs. nut issue is open (§10).

| File | What |
|---|---|
| `SO101-mount_L_interlock.stl` | current mount to print (panel face on the bed) |
| `build_mount_L.py` | builds the mount; all dimensions are parameters at the top |
| `render_L.py` | top / iso / bottom / bolt-section renders (`render_L_*.png`) |
| `render_screw_section.py` | front-face sections through both bolt rows with screws and nuts (`render_screw_section.png`) |
| `Base_SO101_official.stl` | official SO-101 base (TheRobotStudio), watertight |
| `Base_SO101_spotface.{stl,step}` | base with a flat head seat at y=15.2 (lets the 3/4" button heads pass the nut; the stock base leaves them 0.97 short) |
| `SO101-arm_reference.stl`, `arm_base.stl` | SO-ARM100 arm_base in plate frame / original frame |
| `plate_*.stl`, `SO101-mount_plate_engraved.stl`, `render_{faceon,stackup,...}.png` | superseded flat plate (history) |
| `SO101-mount_features.json`, `*_resp.json` | Onshape feature-tree snapshots |

Rebuild: `python3 build_mount_L.py && python3 render_L.py && python3 render_screw_section.py`
(needs trimesh, manifold3d, shapely, opencv-python, scipy, pillow).
