#!/usr/bin/env python3
"""Front-face (plane normal z) sections through both bolt rows: official Base_SO101 +
L-mount + M5 x 25 screws (rear row button head, front row countersunk) + M5 hex nuts.
Writes render_screw_section.png."""
import os, numpy as np, trimesh
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon, box as sbox
from shapely.ops import unary_union
import build_mount_L as B

R = os.path.dirname(os.path.abspath(__file__)) + '/'
BASE_OFFICIAL = R + 'Base_SO101_official.stl'   # TheRobotStudio SO-ARM100 STL/SO101/Individual

# M5 button head ISO 7380: head 9.5 x 2.75, flat underside, length under head.
# M5 countersunk socket ISO 10642 / DIN 7991: 90 deg head Ø10 x 2.8, length overall.
MAJOR_D = 5.0
BUTTON = dict(kind='button', d=B.BUTTON_D, h=2.75)
CSK = dict(kind='csk', d=B.CSK_D, h=2.8)
NUT_T = B.NUT_T

part = trimesh.load(B.OUT)
base = trimesh.load(BASE_OFFICIAL)
base.apply_translation([0, -B.BASE_FLAT, B.BASE_DZ])   # seated: base flat on plate top y=0


def seat_y(x, z, scr):
    """Button: head underside = highest base surface within the head radius.
    Countersunk: head top, lowest position where the 90 deg head cone clears the base."""
    R = scr['d'] / 2; hs = []
    for r in np.linspace(0, R, 16):
        for a in np.linspace(0, 2 * np.pi, 24, endpoint=False):
            l, _, _ = base.ray.intersects_location(
                [[x + r * np.cos(a), 200, z + r * np.sin(a)]], [[0, -1, 0]])
            if len(l): hs.append(l[:, 1].max() + (R - r if scr['kind'] == 'csk' else 0))
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


def panel(z_row, holes, scr, x_lo, x_hi, y_lo, y_hi, W, title, font, small):
    s = (W - 40) / (x_hi - x_lo)
    H = int((y_hi - y_lo) * s) + 90
    img = Image.new('RGB', (W, H), (255, 255, 255)); dr = ImageDraw.Draw(img)
    tf = lambda x, y: (20 + (x - x_lo) * s, 70 + (y_hi - y) * s)
    clip = sbox(x_lo, y_lo, x_hi, y_hi)
    draw_geom(dr, section_polys(base, z_row).intersection(clip), tf, (190, 190, 195), (90, 90, 95))
    draw_geom(dr, section_polys(part, z_row).intersection(clip), tf, (120, 165, 220), (30, 70, 130))
    notes = []
    for x in holes:
        sy = seat_y(x, z_row, scr); tip = sy - B.SCREW_L    # sy: button underside / csk head top
        nut_top, nut_bot = B.NUT_CEIL, B.NUT_CEIL - NUT_T
        R = scr['d'] / 2
        if scr['kind'] == 'button':
            head = sbox(x - R, sy, x + R, sy + scr['h'])
        else:   # 90 deg cone down to the shank, small cylindrical edge on top
            ch = R - MAJOR_D / 2
            head = Polygon([(x - R, sy), (x + R, sy), (x + R, sy - (scr['h'] - ch)),
                            (x + MAJOR_D / 2, sy - scr['h']), (x - MAJOR_D / 2, sy - scr['h']),
                            (x - R, sy - (scr['h'] - ch))])
        shank = sbox(x - MAJOR_D / 2, tip, x + MAJOR_D / 2, sy)
        nut = sbox(x - B.NUT_AF / 2 + 0.15, nut_bot, x + B.NUT_AF / 2 - 0.15, nut_top) - \
            sbox(x - MAJOR_D / 2, nut_bot, x + MAJOR_D / 2, nut_top)
        draw_geom(dr, nut.intersection(clip), tf, (240, 170, 50), (150, 90, 0))
        draw_geom(dr, unary_union([head, shank]).intersection(clip), tf, (215, 60, 60), (120, 20, 20))
        engage = min(nut_top, sy) - max(nut_bot, tip)
        ok = tip <= nut_bot
        notes.append((x, sy, tip, engage, ok))
    for yl, c, lab in ((0, (0, 140, 0), 'plate top y=0'), (-B.T_PAD, (0, 90, 0), 'pad bottom')):
        dr.line([tf(x_lo, yl), tf(x_hi, yl)], fill=c, width=1)
        dr.text(tf(x_lo, yl + 0.4), lab, fill=c, font=small)
    dr.text((20, 8), title, fill=(0, 0, 0), font=font)
    for i, (x, sy, tip, eng, ok) in enumerate(notes):
        dr.text((20 + i * (W // 2), 34),
                f'x={x:+.2f}: {"head seat" if scr["kind"] == "button" else "head top"} y={sy:.2f}  tip y={tip:.2f}  nut {B.NUT_CEIL - NUT_T:.2f}..{B.NUT_CEIL:.2f}  '
                f'thread in nut {eng:.2f}/{NUT_T:.2f} mm  {"THROUGH" if ok else "SHORT by %.2f" % (tip - (B.NUT_CEIL - NUT_T))}',
                fill=(0, 110, 0) if ok else (190, 0, 0), font=small)
    return img, notes


if __name__ == '__main__':
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 20); small = ImageFont.truetype('DejaVuSans.ttf', 15)
    except OSError:
        font = small = ImageFont.load_default()
    rows = [(-37.3, (31.75, -31.75), BUTTON, 'rear bolt row', 'M5x25 button head'),
            (32.475, (27.776, -27.776), CSK, 'front bolt row', 'M5x25 countersunk')]
    imgs = []
    for z, holes, scr, name, sname in rows:
        im, notes = panel(z, holes, scr, -92, 92, -16, 40, 1800,
                          f'Section z={z} ({name}) - official Base_SO101 (grey), L-mount (blue), '
                          f'{sname} (red), M5 hex nut (orange)', font, small)
        imgs.append(im)
        for x in holes[:1]:   # zoom on one bolt
            zim, _ = panel(z, (x,), scr, x - 10, x + 10, -14, 21, 900,
                           f'zoom x={x}, z={z}', font, small)
            imgs.append(zim)
        for n in notes: print(name, 'x=%.2f seat=%.2f tip=%.2f engage=%.2f through=%s' % n)
    W = 1800; Hs = [imgs[0].height, max(imgs[1].height, imgs[3].height), imgs[2].height]
    out = Image.new('RGB', (W, imgs[0].height + imgs[2].height + max(imgs[1].height, imgs[3].height)), 'white')
    out.paste(imgs[0], (0, 0)); out.paste(imgs[2], (0, imgs[0].height))
    y2 = imgs[0].height + imgs[2].height
    out.paste(imgs[1], (0, y2)); out.paste(imgs[3], (900, y2))
    out.save(R + 'render_screw_section.png'); print('saved render_screw_section.png')
