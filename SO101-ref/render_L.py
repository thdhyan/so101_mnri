#!/usr/bin/env python3
"""Renders for SO101-mount_L_interlock.stl (reuses render() from render.py)."""
import numpy as np, trimesh
from PIL import Image, ImageDraw
exec(open('render.py').read().split("plate = load(")[0])   # load(), render()
import build_mount_L as B

part = load(B.OUT)
base = load(B.BASE_STL); base.apply_translation([0, -B.BASE_FLAT, B.BASE_DZ])
BLUE = (70, 130, 200); GREY = (165, 165, 170)

render([(part, BLUE)], cam_dir=(0, 1, 0), up_hint=(0, 0, 1), W=1600, H=1100,
       depth_shade=(-3.0, 0.0)).save('render_L_top.png')
render([(part, BLUE), (base, GREY)], cam_dir=(0.55, 0.6, 0.6), up_hint=(0, 1, 0),
       W=1600, H=1300).save('render_L_iso.png')
render([(part, BLUE)], cam_dir=(0.35, -0.8, 0.5), up_hint=(0, 0, 1),
       W=1600, H=1300).save('render_L_bottom.png')

# section through bottom-right bolt (x=31.75): plate + base, screw & nut drawn
x0 = 31.75
Wc, Hc, s = 1400, 900, 12.0
img = Image.new('RGB', (Wc, Hc), (252, 252, 252)); dr = ImageDraw.Draw(img)
zc, yc = -30.0, 5.0
def tf(z, y): return ((z - zc) * s + Wc / 2, Hc / 2 - (y - yc) * s)
for mesh, col in ((base, GREY), (part, BLUE)):
    sec = mesh.section(plane_origin=(x0, 0, 0), plane_normal=(1, 0, 0))
    for ent in sec.entities:
        pts = sec.vertices[ent.points]
        dr.line([tf(p[2], p[1]) for p in pts], fill=tuple(int(c * 0.6) for c in col), width=3)
zb, _ = B.HOLES[0][1], None
tip = B.SEAT_Y_BASE - B.BASE_FLAT - B.SCREW_L
seat = B.SEAT_Y_BASE - B.BASE_FLAT
dr.rectangle([tf(zb - 2.08, seat), tf(zb + 2.08, tip)], outline=(200, 40, 40), width=3)       # shank
dr.rectangle([tf(zb - 3.43, seat + 2.87), tf(zb + 3.43, seat)], outline=(200, 40, 40), width=3)  # button head
dr.rectangle([tf(zb - 3.97, B.NUT_CEIL), tf(zb + 3.97, B.NUT_CEIL - B.NUT_T)], fill=(230, 170, 60))
dr.line([tf(zc - 55, 0), tf(zc + 55, 0)], fill=(0, 150, 0), width=1)
dr.text((20, 20), f'section x={x0}: #8-32x3/4 button head, seat y={seat:.2f}, tip y={tip:.2f}, '
        f'nut {B.NUT_CEIL - B.NUT_T:.2f}..{B.NUT_CEIL:.2f} (orange), plate top y=0 (green)', fill=(0, 0, 0))
img.save('render_L_section.png')
print('saved render_L_{top,iso,bottom,section}.png')
