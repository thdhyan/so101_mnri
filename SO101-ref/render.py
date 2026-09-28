#!/usr/bin/env python3
"""Software z-buffer renderer for the SO101 mount STLs (no matplotlib/OpenGL needed)."""
import numpy as np, trimesh
from PIL import Image, ImageDraw

def load(path):
    m = trimesh.load(path, process=False)
    if m.bounds[1][0] < 1: m.apply_scale(1000.0)
    return m

def render(items, cam_dir, up_hint=(0,0,1), W=1600, H=1100, bg=(252,252,252),
           depth_shade=None):
    """items: list of (mesh, rgb_color). cam_dir = scene->camera direction.
    depth_shade=(z_lo, z_hi): brightness 0.5..1.0 for camera-depth z_lo..z_hi
    (clamped) — makes same-normal pocket floors visible in plan views."""
    f = np.array(cam_dir, float); f /= np.linalg.norm(f)
    r = np.cross(np.array(up_hint,float), f)
    if np.linalg.norm(r) < 1e-9: r = np.cross(np.array((0,1,0),float), f)
    r /= np.linalg.norm(r)
    u = np.cross(f, r)
    basis = np.stack([r,u,f])
    allp = [m.vertices @ basis for m,_ in items]
    P = np.vstack(allp)
    x0,x1 = P[:,0].min(), P[:,0].max(); y0,y1 = P[:,1].min(), P[:,1].max()
    s = min((W-40)/(x1-x0), (H-40)/(y1-y0))
    ox = (W - s*(x1-x0))/2 - s*x0; oy = (H - s*(y1-y0))/2 + s*y1
    zbuf = np.full((H,W), -np.inf)
    img = np.full((H,W,3), bg, np.float32)
    lights = [np.array(v,float)/np.linalg.norm(v) for v in
              ((0.35,0.75,0.55), (-0.5,0.6,0.3), (0.1,0.2,1.0))]
    lamk  = (0.55, 0.22, 0.16)
    for mesh, col in items:
        V = mesh.vertices @ basis
        sx = V[:,0]*s + ox; sy = oy - V[:,1]*s; sz = V[:,2]
        F = mesh.faces
        tri = V[F]                                   # (n,3,3) camera space
        n = np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0])
        ln = np.linalg.norm(n,axis=1); ln[ln==0]=1; n = n/ln[:,None]
        shade = np.full(len(F), 0.30)                # ambient
        for L,k in zip(lights, lamk):
            shade += k*np.abs(n @ L)
        if depth_shade is not None:
            z_lo, z_hi = depth_shade
            t = np.clip((sz[F].mean(axis=1) - z_lo) / (z_hi - z_lo), 0, 1)
            shade = shade * (0.42 + 0.58*t)
        col = np.array(col, float)
        order = np.argsort(sz[F].mean(axis=1))       # far first (helps z-fight)
        for i in order:
            x = sx[F[i]]; y = sy[F[i]]; z = sz[F[i]]
            minx=max(int(np.floor(x.min())),0); maxx=min(int(np.ceil(x.max())),W-1)
            miny=max(int(np.floor(y.min())),0); maxy=min(int(np.ceil(y.max())),H-1)
            if maxx<minx or maxy<miny: continue
            xs=np.arange(minx,maxx+1)+0.5; ys=np.arange(miny,maxy+1)+0.5
            gx,gy=np.meshgrid(xs,ys)
            d=( (y[1]-y[2])*(x[0]-x[2]) + (x[2]-x[1])*(y[0]-y[2]) )
            if abs(d)<1e-12: continue
            w0=((y[1]-y[2])*(gx-x[2])+(x[2]-x[1])*(gy-y[2]))/d
            w1=((y[2]-y[0])*(gx-x[2])+(x[0]-x[2])*(gy-y[2]))/d
            w2=1-w0-w1
            m=(w0>=0)&(w1>=0)&(w2>=0)
            if not m.any(): continue
            zz=w0*z[0]+w1*z[1]+w2*z[2]
            sub=zbuf[miny:maxy+1,minx:maxx+1]
            hit=m&(zz>sub)
            if not hit.any(): continue
            sub[hit]=zz[hit]
            c=np.clip(col*shade[i],0,255)
            img[miny:maxy+1,minx:maxx+1][hit]=c
    return Image.fromarray(img.astype(np.uint8))

plate = load('plate_grooved.stl')
arm   = load('Part_armsrc.stl') if False else None
base  = load('Base_SO101_spotface.stl')
import zipfile, io
# arm body from last export zip isn't stored standalone; reuse arm reference stl (same geometry)
arm   = load('SO101-arm_reference.stl')

BLUE=(70,130,200); GREY=(165,165,170); TAN=(215,175,110)

# 1) plan view of top face: grooves, engraving, holes (depth-shaded floors)
render([(plate, BLUE)], cam_dir=(0,1,0), up_hint=(0,0,-1), W=1700, H=1150,
       depth_shade=(-3.2, 0.0)).save('render_faceon.png')

# 2) iso stackup: plate + base + arm
render([(plate, BLUE), (base, GREY), (arm, TAN)], cam_dir=(0.55,0.5,0.67),
       up_hint=(0,0,1), W=1700, H=1150).save('render_stackup.png')

# 3) iso of plate alone, angled to see groove depth at +x wing
render([(plate, BLUE)], cam_dir=(0.62,0.55,0.56), up_hint=(0,0,1),
       W=1700, H=1150).save('render_iso_groove.png')

# 4) section at x=70 (crosses both right-wing grooves): profile view
sec = plate.section(plane_origin=(70,0,0), plane_normal=(1,0,0))
p2,_ = sec.to_planar()
Wc,Hc = 1100,1400
img = Image.new('RGB',(Wc,Hc),(252,252,252)); dr=ImageDraw.Draw(img)
xs=[]; ys=[]; polys=[]
for ent in p2.entities:
    pts = p2.vertices[ent.points]
    polys.append(pts); xs.append(pts[:,0]); ys.append(pts[:,1])
xs=np.concatenate(xs); ys=np.concatenate(ys)
s=min((Wc-80)/(xs.max()-xs.min()), (Hc-80)/(ys.max()-ys.min()))
def tf(pts):
    px=(pts[:,0]-xs.min())*s+40
    py=(Hc-40)-(pts[:,1]-ys.min())*s
    return list(zip(px,py))
for pts in polys:
    dr.polygon(tf(pts), fill=(70,130,200), outline=(30,60,100))
img.save('render_assembly_section.png')
print('saved render_faceon.png render_stackup.png render_iso_groove.png render_assembly_section.png')
