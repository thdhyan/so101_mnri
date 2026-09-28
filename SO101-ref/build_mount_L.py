#!/usr/bin/env python3
"""SO-101 L-mount: arm_base top plate (interlocking with the SO-101 base underside)
+ back panel for clamping (matches Onshape Part Studio 1: 10 thick, x +-90,
z -65..-55, down to y=-183.8), bridged to the arm_base rear edge.

Plate frame (same as the old Onshape mount / SO101-arm_reference.stl):
  x lateral, y up (plate top y=0, bottom y=-7.2), z fore-aft, rear = -z.
  Base sits at z_base = z_plate + 29.8, base underside flat (y_b=2.4) on plate top.
Print orientation: back panel outer face on the bed, +z up (holes teardrop apex +z,
hex nut pockets vertex +z).
"""
import os, numpy as np, trimesh, cv2, manifold3d as mf
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

R = os.path.dirname(os.path.abspath(__file__)) + '/'
BASE_STL = R + 'Base_SO101_spotface.stl'
ARM_STL = R + 'SO101-arm_reference.stl'          # = SO-ARM100 arm_base.stl, plate frame
OUT = R + 'SO101-mount_L_interlock.stl'

# ---- parameters (mm) ----
ARM_SHIFT = (-0.18, 0.0, -0.33)   # arm_base holes -> exact base hole positions
BASE_DZ = -29.8                   # z_plate = z_base - 29.8
BASE_FLAT = 2.4                   # base underside flat (base frame); ribs/bosses below it
CLEAR_XZ = 0.2                    # interlock side clearance
POCKET_DEPTH = BASE_FLAT + 0.2    # rib pocket depth below plate top
T_PLATE = 7.2                     # arm_base thickness
PANEL_Z = (-65.0, -55.0)          # back panel z range (Onshape plate: 10 thick)
PANEL_X = 90.0                    # back panel half width (Onshape plate: 180 wide)
PANEL_BOTTOM = -183.8             # back panel reaches y=-183.8 (Onshape plate)
HOLE_D = 5.0
HOLES = [(31.75, -37.3), (-31.75, -37.3), (27.776, 32.475), (-27.776, 32.475)]
NUT_AF = 7.94 + 0.30              # #8-32 hex nut 5/16" AF + clearance
NUT_CEIL = -3.5                   # nut pocket ceiling (y); pocket open to the bottom
SEAT_Y_BASE = 15.20               # spotfaced head seat (base frame)
SCREW_L = 19.05                   # #8-32 x 3/4" button head
NUT_T = 2.24


# plate (x,y,z) <-> work frame W (X=x, Y=-z, Z=y): extrusions run along plate y
def to_w(v):  return np.c_[v[:, 0], -v[:, 2], v[:, 1]]
def from_w(v): return np.c_[v[:, 0], v[:, 2], -v[:, 1]]

def load(p):
    m = trimesh.load(p, process=False)
    if m.bounds[1].max() < 1: m.apply_scale(1000.0)
    return m

def to_manifold(m):
    return mf.Manifold(mf.Mesh(vert_properties=np.asarray(m.vertices, np.float32),
                               tri_verts=np.asarray(m.faces, np.uint32)))

def to_trimesh(man):
    g = man.to_mesh()
    return trimesh.Trimesh(g.vert_properties[:, :3], g.tri_verts, process=True)

def xz_polys_to_cs(geom):
    """shapely polygons in plate (x,z) -> CrossSection in W (X=x, Y=-z)."""
    geoms = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    rings = []
    for p in geoms:
        for ring in [p.exterior, *p.interiors]:
            c = np.asarray(ring.coords)[:-1]
            rings.append(np.c_[c[:, 0], -c[:, 1]])
    return mf.CrossSection(rings, mf.FillRule.EvenOdd)

def slab(cs, y0, y1):
    return mf.Manifold.extrude(cs, y1 - y0).translate((0, 0, y0))

def box(x0, x1, y0, y1, z0, z1):   # plate-frame box
    return mf.Manifold.cube((x1 - x0, z1 - z0, y1 - y0)).translate((x0, -z1, y0))


def base_underside_mask(res=0.05):
    """Region where the base protrudes below its flat (ribs, hole bosses), plate frame."""
    b = load(BASE_STL)
    x0, x1, z0, z1 = -60, 60, -50, 45
    xs = np.arange(x0, x1, res); zs = np.arange(z0, z1, res)
    X, Z = np.meshgrid(xs, zs)
    O = np.c_[X.ravel(), np.full(X.size, -10.0), Z.ravel() - BASE_DZ]
    D = np.tile([0, 1.0, 0], (X.size, 1))
    loc, idx, _ = b.ray.intersects_location(O, D, multiple_hits=False)
    H = np.full(X.size, np.nan); H[idx] = loc[:, 1]
    H = H.reshape(X.shape)
    mask = (~np.isnan(H) & (H < BASE_FLAT - 0.1)).astype(np.uint8)
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for i, c in enumerate(cnts):
        if hier[0][i][3] != -1 or len(c) < 3: continue   # outer contours only
        pts = c[:, 0, :].astype(float)
        poly = Polygon(np.c_[x0 + pts[:, 0] * res, z0 + pts[:, 1] * res])
        holes = [np.c_[x0 + cc[:, 0, 0] * res, z0 + cc[:, 0, 1] * res]
                 for j, cc in enumerate(cnts) if hier[0][j][3] == i and len(cc) >= 3]
        polys.append(Polygon(poly.exterior.coords, holes).buffer(0))
    return unary_union(polys), (xs, zs, H)


def hole_2d(x, z):
    r = HOLE_D / 2
    circ = Polygon([(x + r * np.cos(t), z + r * np.sin(t))
                    for t in np.linspace(0, 2 * np.pi, 64, endpoint=False)])
    s = r / np.sqrt(2)          # 45 deg teardrop, apex +z (print up)
    tri = Polygon([(x - s, z + s), (x + s, z + s), (x, z + r * np.sqrt(2))])
    return unary_union([circ, tri.buffer(0.001)])

def hex_2d(x, z):
    rc = NUT_AF / np.sqrt(3)    # corner radius; vertex pointing +z
    return Polygon([(x + rc * np.cos(np.pi / 2 + k * np.pi / 3),
                     z + rc * np.sin(np.pi / 2 + k * np.pi / 3)) for k in range(6)])


def build():
    arm = load(ARM_STL); arm.merge_vertices(); arm.apply_translation(ARM_SHIFT)
    A = to_manifold(trimesh.Trimesh(to_w(arm.vertices), arm.faces, process=False))
    (ax0, _, az0), (ax1, _, _) = arm.bounds

    notch_fill = box(-12.5, 12.5, -T_PLATE, 0, az0, -3.0)   # rear notch -> solid under base
    panel = box(-PANEL_X, PANEL_X, PANEL_BOTTOM, 0, *PANEL_Z)
    bridge = box(-PANEL_X, PANEL_X, -T_PLATE, 0, PANEL_Z[1] - 0.5, az0 + 0.5)  # plate -> panel
    part = A + notch_fill + panel + bridge

    mask_geom, hm = base_underside_mask()
    cutter = mask_geom.buffer(CLEAR_XZ, join_style=1).simplify(0.01)
    part = part - slab(xz_polys_to_cs(cutter), -POCKET_DEPTH, 1.0)

    holes = unary_union([hole_2d(x, z) for x, z in HOLES])
    nuts = unary_union([hex_2d(x, z) for x, z in HOLES])
    part = part - slab(xz_polys_to_cs(holes), -T_PLATE - 1, 1.0)
    part = part - slab(xz_polys_to_cs(nuts), -T_PLATE - 1, NUT_CEIL)

    g = part.simplify(0.001).to_mesh()   # drop coplanar-union slivers
    out = trimesh.Trimesh(from_w(g.vert_properties[:, :3].astype(float)), g.tri_verts, process=False)
    return out, mask_geom, cutter, hm


if __name__ == '__main__':
    m, mask_geom, cutter, hm = build()
    m.export(OUT)
    tip = SEAT_Y_BASE - BASE_FLAT - SCREW_L
    print('saved', OUT)
    print('bounds', m.bounds.round(2).tolist(), 'vol', round(m.volume, 1),
          'watertight', m.is_watertight, 'bodies', len(m.split()))
    print(f'screw tip y={tip:.2f}  nut zone {NUT_CEIL - NUT_T:.2f}..{NUT_CEIL:.2f}  '
          f'tip past nut {NUT_CEIL - NUT_T - tip:.2f}  web over nut to pocket floor '
          f'{-POCKET_DEPTH - NUT_CEIL:.2f}')
