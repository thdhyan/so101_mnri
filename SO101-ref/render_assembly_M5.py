#!/usr/bin/env python3
"""Assembly renders with M5 x 25 screws + nuts: official Base_SO101 on the L-mount.
Stills: render_asm_{iso,top,front,right}.png + render_asm_views.png (2x2).
Videos: slice_front_pair.mp4 / slice_rear_pair.mp4 - section plane (normal z) swept
through each bolt row, with a top-view inset marking the cut."""
import os, subprocess, numpy as np, trimesh
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon, box as sbox
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'render.py')).read()
     .split("plate = load(")[0])                      # render()
import build_mount_L as B
import render_screw_section as S                      # seat_y(), section_polys(), draw_geom(), base, part

R = S.R
GREY, BLUE, RED, ORANGE = (170, 170, 175), (80, 135, 205), (205, 55, 55), (235, 165, 45)
TO_Y = trimesh.transformations.rotation_matrix(-np.pi / 2, (1, 0, 0))   # revolve axis z -> plate y


def revolved(profile, x, z):
    m = trimesh.creation.revolve(np.asarray(profile, float), sections=48)
    m.apply_transform(TO_Y); m.apply_translation((x, 0, z)); return m


def screw(x, z, scr):
    sy = S.seat_y(x, z, scr); tip = sy - B.SCREW_L; r = S.MAJOR_D / 2; Rh = scr['d'] / 2
    if scr['kind'] == 'button':     # flat underside, dome approximated by an arc
        dome = [(Rh * np.cos(t), sy + 0.6 + (scr['h'] - 0.6) * np.sin(t))
                for t in np.linspace(0.0, np.pi / 2, 10)]
        prof = [(0, tip), (r, tip), (r, sy), (Rh, sy)] + dome
    else:                           # 90 deg countersunk; sy = head top
        prof = [(0, tip), (r, tip), (r, sy - (Rh - r) - 0.3), (Rh, sy - 0.3), (Rh, sy), (0, sy)]
    return revolved(prof, x, z)


def nut(x, z):
    rc = 8.0 / np.sqrt(3)
    hexa = Polygon([(rc * np.cos(np.pi / 2 + k * np.pi / 3), rc * np.sin(np.pi / 2 + k * np.pi / 3))
                    for k in range(6)])
    hole = Polygon([(2.5 * np.cos(t), 2.5 * np.sin(t)) for t in np.linspace(0, 2 * np.pi, 40, endpoint=False)])
    m = trimesh.creation.extrude_polygon(hexa - hole, B.NUT_T)
    m.apply_transform(TO_Y)                           # polygon y -> -z (hex symmetric), height -> +y
    m.apply_translation((x, B.NUT_CEIL - B.NUT_T, z)); return m


ROWS = {'rear': (-37.3, S.BUTTON, 'M5x25 button head'), 'front': (32.475, S.CSK, 'M5x25 countersunk')}
screws, nuts = [], []
for x, z in B.HOLES:
    scr = ROWS['rear' if z < 0 else 'front'][1]
    screws.append(screw(x, z, scr)); nuts.append(nut(x, z))
SCREWS = trimesh.util.concatenate(screws); NUTS = trimesh.util.concatenate(nuts)


def sect(mesh, z):
    """section_polys, empty when the plane misses the mesh."""
    if mesh.section(plane_origin=(0, 0, z), plane_normal=(0, 0, 1)) is None: return Polygon()
    return S.section_polys(mesh, z)


def label(img, text, font):
    d = ImageDraw.Draw(img); d.text((20, 14), text, fill=(0, 0, 0), font=font); return img


def stills(font):
    items = [(S.part, BLUE), (S.base, GREY), (SCREWS, RED), (NUTS, ORANGE)]
    views = [('iso', (0.55, 0.6, 0.6), (0, 1, 0), 'Iso: base (grey) on L-mount (blue), M5 screws (red)'),
             ('top', (0, 1, 0), (0, 0, -1), 'Top (back panel up, front down): rear row button heads, front row countersunk'),
             ('front', (0, 0, 1), (0, 1, 0), 'Front (from +z)'),
             ('right', (1, 0, 0), (0, 1, 0), 'Right (from +x; front at left, back panel at right)')]
    out = {}
    for name, cam, up, title in views:
        im = label(render(items, cam, up, W=1400, H=1100), title, font)
        im.save(R + f'render_asm_{name}.png'); out[name] = im; print('saved', f'render_asm_{name}.png')
    grid = Image.new('RGB', (2800, 2200), 'white')
    for i, n in enumerate(('iso', 'top', 'front', 'right')):
        grid.paste(out[n], ((i % 2) * 1400, (i // 2) * 1100))
    grid.save(R + 'render_asm_views.png'); print('saved render_asm_views.png')


def inset_map(size=420):
    """Top-view outline of mount + base + holes with a z->pixel map (x right, front down)."""
    x_lo, x_hi, z_lo, z_hi = -95, 95, -70, 50
    s = size / (x_hi - x_lo); H = int((z_hi - z_lo) * s)
    img = Image.new('RGB', (size, H), (245, 245, 245)); d = ImageDraw.Draw(img)
    tf = lambda x, z: ((x - x_lo) * s, (z - z_lo) * s)            # rear (-z) at top
    for mesh, y, fill in ((S.part, -1.0, (150, 185, 230)), (S.base, 3.0, (185, 185, 190))):
        sec = mesh.section(plane_origin=(0, y, 0), plane_normal=(0, 1, 0))
        g = Polygon()
        for p in sec.discrete:
            if len(p) > 3: g = g.symmetric_difference(Polygon(np.asarray(p)[:, [0, 2]]).buffer(0))
        for p in getattr(g, 'geoms', [g]):
            if p.geom_type == 'Polygon' and not p.is_empty:
                d.polygon([tf(*c) for c in p.exterior.coords], fill=fill, outline=(90, 90, 90))
    for x, z in B.HOLES:
        cx, cz = tf(x, z); d.ellipse([cx - 4, cz - 4, cx + 4, cz + 4], fill=RED)
    return img, tf


def sweep(row, font, small, fps=20, frames=100):
    z0, scr, sname = ROWS[row]
    holes = [x for x, z in B.HOLES if z == z0]
    inset, tf = inset_map()
    x_lo, x_hi, y_lo, y_hi = -50, 50, -15, 42
    Wp = 1600; s = (Wp - 40) / (x_hi - x_lo); Hs = int((y_hi - y_lo) * s); W = Wp + inset.width + 20
    H = Hs + 130; H += H % 2
    fdir = R + f'_frames_{row}'; os.makedirs(fdir, exist_ok=True)
    zs = z0 + 7.0 * np.sin(np.linspace(-np.pi / 2, 1.5 * np.pi, frames))   # -7 -> +7 -> -7 mm
    clip = sbox(x_lo, y_lo, x_hi, y_hi)
    for i, z in enumerate(zs):
        img = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(img)
        t = lambda x, y: (20 + (x - x_lo) * s, 110 + (y_hi - y) * s)
        for mesh, fill, ol in ((S.base, GREY, (90, 90, 95)), (S.part, BLUE, (30, 70, 130)),
                               (NUTS, ORANGE, (150, 90, 0)), (SCREWS, RED, (120, 20, 20))):
            S.draw_geom(d, sect(mesh, z).intersection(clip), t, fill, ol)
        for yl, lab in ((0, 'plate top y=0'), (-B.T_PAD, 'pad bottom y=-12')):
            d.line([t(x_lo, yl), t(x_hi, yl)], fill=(0, 130, 0), width=1)
            d.text(t(x_lo + 0.3, yl + 1.2), lab, fill=(0, 110, 0), font=small)
        d.text((20, 12), f'{row.upper()} PAIR - {sname} + M5 nut (orange) | official base (grey), '
               f'L-mount (blue)', fill=(0, 0, 0), font=font)
        d.text((20, 44), f'section plane z = {z:+.2f} mm   (bolt axis z = {z0}, offset {z - z0:+.2f})   '
               f'bolts at x = {holes[0]:+.2f}, {holes[1]:+.2f}', fill=(0, 0, 0), font=small)
        d.text((20, 68), 'view from front (+z), x right, y up. Plane sweeps -7 -> +7 -> -7 mm through the '
               'screws, nuts and pockets.', fill=(80, 80, 80), font=small)
        ins = inset.copy(); di = ImageDraw.Draw(ins)
        di.line([tf(-95, z), tf(95, z)], fill=(220, 0, 0), width=3)
        di.text((6, 4), 'top view: cut line', fill=(0, 0, 0), font=small)
        img.paste(ins, (Wp + 10, H - ins.height - 20))
        img.save(f'{fdir}/{i:04d}.png')
    out = R + f'slice_{row}_pair.mp4'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', f'{fdir}/%04d.png',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', out], check=True)
    Image.open(f'{fdir}/{frames // 4:04d}.png').save(R + f'slice_{row}_pair_frame.png')
    for f in os.listdir(fdir): os.remove(f'{fdir}/{f}')
    os.rmdir(fdir); print('saved', out)


if __name__ == '__main__':
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 24); small = ImageFont.truetype('DejaVuSans.ttf', 17)
    except OSError:
        font = small = ImageFont.load_default()
    for m in (SCREWS, NUTS): print('watertight', m.is_watertight, m.bounds.round(2).tolist())
    sweep('front', font, small); sweep('rear', font, small)
    stills(font)
