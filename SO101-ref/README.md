# SO101-ref: SO-101 base mount (L-mount)

A 3D-printed L-shaped mount for the SO-101 follower arm base. The top plate is the SO-ARM100
`arm_base` (overhead-cam tabs kept), with pockets that the base's underside ribs drop into. A
10 mm back panel (180 wide, 183.8 long) gives the clamps a surface. The base bolts down with
M5 x 25 screws into captive M5 hex nuts from below: button heads in the rear row (by the back
panel), countersunk heads in the front row (low profile where the arm swings). Under the bolts
the plate is 12 mm thick; the cam-mount tabs stay 7.2. Use the official base.

**Agents: read `HANDOFF.md` §9, §11 and §12 first.**

| File | What |
|---|---|
| `SO101-mount_L_interlock.stl` | current mount to print (panel face on the bed) |
| `build_mount_L.py` | builds the mount; all dimensions are parameters at the top |
| `render_L.py` | top / iso / bottom / bolt-section renders (`render_L_*.png`) |
| `render_screw_section.py` | front-face sections through both bolt rows with screws and nuts (`render_screw_section.png`) |
| `render_assembly_M5.py` | assembly with M5 screws + nuts: `render_asm_{iso,top,front,right,views}.png`, section-sweep videos `slice_{front,rear}_pair.mp4` (+ `_frame.png`) |
| `Base_SO101_official.stl` | official SO-101 base (TheRobotStudio), watertight; the one to print |
| `Base_SO101_spotface.{stl,step}` | old #8-32 fix (flat seat); don't print with M5 countersunk heads. Still used as the underside heightmap source |
| `SO101-arm_reference.stl`, `arm_base.stl` | SO-ARM100 arm_base in plate frame / original frame |
| `plate_*.stl`, `SO101-mount_plate_engraved.stl`, `render_{faceon,stackup,...}.png` | superseded flat plate (history) |
| `SO101-mount_features.json`, `*_resp.json` | Onshape feature-tree snapshots |

Rebuild: `python3 build_mount_L.py && python3 render_L.py && python3 render_screw_section.py && python3 render_assembly_M5.py`
(needs trimesh, manifold3d, shapely, opencv-python, scipy, pillow).
