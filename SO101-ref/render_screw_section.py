#!/usr/bin/env python3
"""Front-face (plane normal z) sections through both bolt rows: official Base_SO101 +
L-mount + #8-32 x 3/4" button head screws + hex nuts. Writes render_screw_section.png."""
import os, numpy as np, trimesh
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon, box as sbox
from shapely.ops import unary_union
import build_mount_L as B

R = os.path.dirname(os.path.abspath(__file__)) + '/'
BASE_OFFICIAL = R + 'Base_SO101_official.stl'   # TheRobotStudio SO-ARM100 STL/SO101/Individual

# button head socket cap #8-32 (ASME B18.3): head 0.312" x 0.087" (flat underside, domed top), major 0.164"
HEAD_D, HEAD_H, MAJOR_D = 7.92, 2.21, 4.17
NUT_T = B.NUT_T

part = trimesh.load(B.OUT)
base = trimesh.load(BASE_OFFICIAL)
base.apply_translation([0, -B.BASE_FLAT, B.BASE_DZ])   # seated: base flat on plate top y=0


def seat_y(x, z):
    """Head underside height: highest base surface within the head radius (plate frame)."""
    hs = []
    for r in np.linspace(0, HEAD_D / 2, 8):
        for a in np.linspace(0, 2 * np.pi, 24, endpoint=False):
            l, _, _ = base.ray.intersects_location(
                [[x + r * np.cos(a), 200, z + r * np.sin(a)]], [[0, -1, 0]])
            if len(l): hs.append(l[:, 1].max())
    return max(hs)


def section_polys(mesh, z):
    sec = mesh.section(plane_origin=(0, 0, z), plane_normal=(0, 0, 1))
    loops = [Polygon(np.asarray(d)[:, :2]).buffer(0) for d in sec.discrete if len(d) > 3]
    out = Polygon()
    for p in loops: out = out.symmetric_difference(p)          # even-odd
    return out


def draw_geom(dr, g, tf, fill, outline):
    for p in getattr(g, 'geoms', [g]):
        if p.is_empty or p.geom_type != 'Polygon': continue
        dr.polygon([tf(*c) for c in p.exterior.coords], fill=fill, outline=outline)
        for h in p.interiors:
            dr.polygon([tf(*c) for c in h.coords], fill=(255, 255, 255), outline=outline)


def panel(z_row, holes, x_lo, x_hi, y_lo, y_hi, W, title, font, small):
    s = (W - 40) / (x_hi - x_lo)
    H = int((y_hi - y_lo) * s) + 90
    img = Image.new('RGB', (W, H), (255, 255, 255)); dr = ImageDraw.Draw(img)
    tf = lambda x, y: (20 + (x - x_lo) * s, 70 + (y_hi - y) * s)
    clip = sbox(x_lo, y_lo, x_hi, y_hi)
    draw_geom(dr, section_polys(base, z_row).intersection(clip), tf, (190, 190, 195), (90, 90, 95))
    draw_geom(dr, section_polys(part, z_row).intersection(clip), tf, (120, 165, 220), (30, 70, 130))
    notes = []
    for x in holes:
        sy = seat_y(x, z_row); tip = sy - B.SCREW_L
        nut_top, nut_bot = B.NUT_CEIL, B.NUT_CEIL - NUT_T
        head = sbox(x - HEAD_D / 2, sy, x + HEAD_D / 2, sy + HEAD_H)
        shank = sbox(x - MAJOR_D / 2, tip, x + MAJOR_D / 2, sy)
        nut = sbox(x - B.NUT_AF / 2 + 0.15, nut_bot, x + B.NUT_AF / 2 - 0.15, nut_top) - \
            sbox(x - MAJOR_D / 2, nut_bot, x + MAJOR_D / 2, nut_top)
        draw_geom(dr, nut.intersection(clip), tf, (240, 170, 50), (150, 90, 0))
        draw_geom(dr, unary_union([head, shank]).intersection(clip), tf, (215, 60, 60), (120, 20, 20))
        engage = min(nut_top, sy) - max(nut_bot, tip)
        ok = tip <= nut_bot
        notes.append((x, sy, tip, engage, ok))
    for yl, c, lab in ((0, (0, 140, 0), 'plate top y=0'), (-B.T_PLATE, (0, 90, 0), 'plate bottom')):
        dr.line([tf(x_lo, yl), tf(x_hi, yl)], fill=c, width=1)
        dr.text(tf(x_lo, yl + 0.4), lab, fill=c, font=small)
    dr.text((20, 8), title, fill=(0, 0, 0), font=font)
    for i, (x, sy, tip, eng, ok) in enumerate(notes):
        dr.text((20 + i * (W // 2), 34),
                f'x={x:+.2f}: head seat y={sy:.2f}  tip y={tip:.2f}  nut {B.NUT_CEIL - NUT_T:.2f}..{B.NUT_CEIL:.2f}  '
                f'thread in nut {eng:.2f}/{NUT_T:.2f} mm  {"THROUGH" if ok else "SHORT by %.2f" % (tip - (B.NUT_CEIL - NUT_T))}',
                fill=(0, 110, 0) if ok else (190, 0, 0), font=small)
    return img, notes


if __name__ == '__main__':
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 20); small = ImageFont.truetype('DejaVuSans.ttf', 15)
    except OSError:
        font = small = ImageFont.load_default()
    rows = [(-37.3, (31.75, -31.75), 'rear bolt row'), (32.475, (27.776, -27.776), 'front bolt row')]
    imgs = []
    for z, holes, name in rows:
        im, notes = panel(z, holes, -92, 92, -12, 40, 1800,
                          f'Section z={z} ({name}) - official Base_SO101 (grey), L-mount (blue), '
                          f'#8-32x3/4" button head (red), hex nut (orange)', font, small)
        imgs.append(im)
        for x in holes[:1]:   # zoom on one bolt
            zim, _ = panel(z, (x,), x - 9, x + 9, -9, 21, 900,
                           f'zoom x={x}, z={z}', font, small)
            imgs.append(zim)
        for n in notes: print(name, 'x=%.2f seat=%.2f tip=%.2f engage=%.2f through=%s' % n)
    W = 1800; Hs = [imgs[0].height, max(imgs[1].height, imgs[3].height), imgs[2].height]
    out = Image.new('RGB', (W, imgs[0].height + imgs[2].height + max(imgs[1].height, imgs[3].height)), 'white')
    out.paste(imgs[0], (0, 0)); out.paste(imgs[2], (0, imgs[0].height))
    y2 = imgs[0].height + imgs[2].height
    out.paste(imgs[1], (0, y2)); out.paste(imgs[3], (900, y2))
    out.save(R + 'render_screw_section.png'); print('saved render_screw_section.png')
