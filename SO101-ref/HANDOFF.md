# SO101-mount — Agent Handoff (state, API recipes, pitfalls, next task)

**Home:** `so101_mnri` repo → `SO101-ref/` (on `main`; run scripts from this folder).
`~/Downloads/SO101-ref/` is the old scratch copy — don't edit it.
**Updated:** 2026-09-29 · first written 2026-09-24 (ses_f35dce26dffegiKQiJUGOJxNzR)

**Current deliverable:** `SO101-mount_L_interlock.stl` (L-mount, §9, M5 update §11). **Read §9,
§11 and §12 first**; §10 is the superseded #8-32 check; §3–§8 describe the superseded flat plate
in Onshape (kept for API recipes / history).

**Next task:** none open. User to print and test fit. Unconfirmed: M5 nut thickness (4.7 assumed),
and whether the M5 shank passes the base's Ø5.0 printed hole (see §11 risks).

## 1. Connect to Onshape

**IDs**

| Thing | ID |
|---|---|
| Document | `c1022e9d09b4c0bb203b269c` |
| Workspace | `626890f8b924980aa69188a0` |
| Mount Part Studio (plate + arm) | `a906b86589d095b6a9637257` |
| `Base_SO101` Part Studio (imported STEP) | `d92d3410f335d349b7897280` |
| `Base_SO101_spotface` Part Studio | `7958d40c518c626758fb274d` |
| Blob `Base_SO101.step` | `67b30e4b34ed0dc9607c307f` |
| Blob `Base_SO101_spotface.step` | `2207b2ce4be183aa41ed60bb` |

**Auth** — `~/Projects/onshape-mcp/.env`:
```
ONSHAPE_ACCESS_KEY / ONSHAPE_SECRET_KEY  →  Authorization: Basic base64(access:secret)
```

**REST base:** `https://cad.onshape.com/api/v17`

Python bootstrap (verified):
```python
import json, urllib.request, base64
env = dict(l.strip().split('=',1) for l in open('/home/thakk100/Projects/onshape-mcp/.env')
           if l.strip() and not l.startswith('#'))
AUTH = 'Basic '+base64.b64encode(f"{env['ONSHAPE_ACCESS_KEY']}:{env['ONSHAPE_SECRET_KEY']}".encode()).decode()
def get(url):   # ALWAYS wrap: urlopen(url, headers=…) is a TypeError
    return json.loads(urllib.request.urlopen(
        urllib.request.Request(url, headers={'Authorization':AUTH}), timeout=30).read())
```

**MCP:** the onshape MCP server has **read-only** tools that work (`get_document`,
`get_features`, `get_parts`, …). All MCP **write** calls failed (create → 400, PATCH → 405).
**Do all writes via REST.**

**OpenAPI spec** was at `/tmp/opencode/openapi.json` (wiped) — re-fetch from Onshape's
OpenAPI endpoint before relying on request shapes you haven't used before.

---

## 2. Verified REST recipes

### 2.1 Read features
```
GET /partstudios/d/{did}/w/{wid}/e/{eid}/features
→ {"features":[…23…], "rollbackIndex":23, …}   # key = "features"
```
Current snapshot: **`SO101-ref/SO101-mount_features.json`** (476 KB, re-fetched 2026-09-24).

### 2.2 Update an existing feature (VERIFIED)
```
POST /partstudios/d/{did}/w/{wid}/e/{eid}/features/featureid/{fid}
Content-Type: application/json;charset=UTF-8
{"btType":"BTFeatureDefinitionCall-1406",
 "feature":{ …full feature object exactly as GET returned, with edits… },
 "libraryVersion": <from GET response>}
```
Used to change extrude depth (`expression` param), then re-GET to confirm `featureStates` all OK.

### 2.3 Export STL (VERIFIED)
```
POST /partstudios/d/{did}/w/{wid}/e/{eid}/translations        # JSON! (see pitfall P1)
{"formatName":"STL","unit":"millimeter","grouping":false,"storeInDocument":false}
→ {"id": jid}
GET /translations/{jid}  until requestState=="DONE"   (~10–30 s, poll 2 s)
→ resultExternalDataIds[0] = fid
GET /documents/d/{did}/externaldata/{fid}  → ZIP
```
⚠️ The ZIP **always contains BOTH parts** (`…Part 1.stl` = ARM vol ≈59 253,
`…Part 1 (1).stl` = PLATE vol ≈288 459). **Classify by volume, never take `[0]`.**
⚠️ Output may be **metres** → if `max|bounds| < 1.0`, `mesh.apply_scale(1000)`.

### 2.4 Import STEP (VERIFIED)
```
POST /blobelements/d/{did}/w/{wid}
multipart: file=@part.step;type=application/step
→ translationId; a new Part Studio named after the file appears in ~12 s
   (GET document elements to confirm)
```

### 2.5 Create a NEW feature (NOT YET VERIFIED — likely next step)
Candidate endpoints (from OpenAPI):
```
POST /partstudios/d/{did}/{wvm}/{wvmid}/e/{eid}/features        # create (same JSON wrapper)
POST /partstudios/d/{did}/w/{wid}/e/{eid}/features/rollback     # insert position
GET  /partstudios/d/{did}/w/{wid}/e/{eid}/featurespecs          # validate shapes first
POST /partstudios/d/{did}/w/{wid}/e/{eid}/featurescript         # run FeatureScript ⭐ recommended
```
**Recommended path for grooves:** author **FeatureScript** (one POST, creates sketch +
extrude-cut parametrically, no hand-authored entity JSON), then verify with GET features.
`DELETE …/features/featureid/{fid}` exists if you must remove a bad feature.

---

## 3. Feature tree (23, rollbackIndex 23, all featureStates OK)

| # | featureId | name | notes |
|---|---|---|---|
| 0 | FrdYJznNlLGRbpo_0 | Sketch 1 | plate outline (x ±60, z −65..55) — **edit here to widen** |
| 1 | Fh7z8CGCjmIeNJF_0 | Extrude 1 | plate, 6 mm (y 0 → −6) |
| 2 | FWIvF6DjYzhykX9_1 | Derived 1 | |
| 3 | Fw72VW8m9cN31Fh_2 | Transform 1 | |
| 4–9 | FypeSzobtXP9E3B_2, FMkwseI0aP2Jpcq_2, F9bobFJY4pZrDkB_2, F6DFH2ahrhTKCoS_2, FVpRO4e5LWym7Mn_1, F8JFBtXyklmyTas_1 | Sketch/Extrude 2–4 | holes, teardrops, nut pockets, counterbores |
| 10 | FKIv7xSEcXGhZzr_1 | **Derived arm_base** | = SO-ARM100 `arm_base.stl` (same mesh) |
| 11 | FBgNkfWQWjA5ggv_2 | Position arm_base | |
| 12–13 | FD6u82p9KRolHTI_2, F8IvZ6O5sg70EcI_2 | Transform 2/3 | |
| 14 | FKLQv2z8SnqV5Xh_2 | Pattern slab (sketch) | engraving slab footprint — **extend if wings should be engraved** |
| 15 | FJOXNmt4AcRVkvU_2 | Pattern slab (extrude) | depth **2 mm** (was 1, changed) |
| 16 | FWOzc1nBOFjFicD_3 | Pattern slab copy | 2 mm |
| 17 | FdDaE8xp8VN1PgO_4 | split pattern | |
| 18 | FYzVSZDVrZKAdg9_4 | pattern skin | |
| 19 | FlzLsYUg7eGT7KF_3 | Bolt seat protect (sketch) | r6 rings ×4 |
| 20 | Fx0OMDWtj8sf7sV_3 | Bolt seat protect (extrude) | |
| 21 | F6WxcK2bTgBdPet_7 | keep bolt seats | SUBTRACTION |
| 22 | FIpevh98QIIfce0_2 | **arm_base subtract** | the engraving boolean (A−(A−B), see P4) |

Parts: **JHD = plate**, **JaD = arm** (2 bodies).

---

## 4. Geometry reference (verified by mesh/STEP probes)

**Plate:** front face **y=0** (mates base), back **y=−6**; outline x ±60, z −65..55.
Engraving floor **y=−2.0** (2 mm deep, 4 mm retained). Retained-volume check after
depth change: 288 459.3 mm³ watertight, 2 426 faces.

**Bolt holes (plate frame):** bottom (±31.75, −37.3) Ø5.0 teardrop (apex +z, void extent
3.6 at r); top (±27.776, +32.475). Pitch **63.5 / 55.552**, row separation **69.775**.
Counterbore/oblong **Ø8.31 × 9.40 × 5 deep**, pocket floor **y=−0.95**, web 0.95 mm.
Nut: #8-32 hex, AF 7.94, corners 9.17, thick 2.24 — captive (corner-bound, can't spin).

**Base:** `Base_SO101` y=0 mates plate y=0; `z_plate = z_base − 29.8`. Mating lips
**2051 mm², 198/198 centroids supported at y=−0.5 → sits flush.** Base top at holes is a
46° cone 15.25→16.45 (no flat seat) → **spotface fix chosen**: flat seat cut at
**y=15.20** (`Base_SO101_spotface`, −199.7 mm³) → 3/4" screw tip lands −3.85 →
**full 2.24 mm nut engagement +0.66 spare.**

**Stack-up:** head seat 16.45 as-modeled / **15.20 spotfaced**; screw 8-32×3/4" = 19.05;
tip = seat − 19.05; nut zone −0.95…−3.19. Alternative: 7/8" screw, seat 16.45, tip −5.775.

---

## 5. Overhead-cam-mount attachment analysis (source: SO-ARM100 repo)

Files: `SO101-ref/{arm_base.stl, cam_mount_bottom.stl, Overhead_Cam_Mount_README.md}`
(raw: `github.com/TheRobotStudio/SO-ARM100/Optional/Overhead_Cam_Mount_32x32_UVC_Module`).

**`arm_base.stl`** — **identical mesh to the "arm" body already in the doc**
(vol 59 252.7 = 59 253, 1 736 faces, extents 159.04 × 7.2 × 89.6). So the plate's
engraved lattice *is* arm_base's pocket pattern, and feature #10 derives this file.
- Bounds (STL frame): x[−93.92, −4.32], y[−5, 2.2], z[−132.95, 26.09].
- Construction: 5 mm plate (y −5…0) + two 2.2 mm boss pads (y 0…2.2) in two zones
  (z −132.95…−69.21 and −37.65…26.09) with a waist between (z −69.21…−37.65).
- **4× Ø5.0 through-holes**, centers (x, z):
  **(−86.80, −85.18), (−86.80, −21.68), (−17.04, −81.21), (−17.04, −25.65)**
  → column pitch Δz = **63.50** and **55.56**, column spacing Δx = **69.76**
  → **exactly the SO-101 base bolt pattern (63.5 / 55.552 / 69.775), rotated 90°.**
  ⇒ arm_base is a drop-in bolt-on of the base footprint; **repeat this pattern on the
  plate wings** and any arm_base / base-footprint part bolts on directly.
- Hole edge margin ≈ 4.6 mm material to outline.

**`cam_mount_bottom.stl`** — bounds x[0, 0, −10] → [37.4, 230.95, 83.02]
(37.4 thick × 230.95 tall × 93.02 deep). World-frame planar faces (analysis method below):
- x = 0.00 / 37.40 (outer walls): area 525.8 each, span y[0, 7.2] × z[0, 73.02]
- x = 6.00 / 31.40 (main column walls, 25.4 wide): area 7209, y[8.70, 223.10] × z[19.70, 53.33]
- x = 6.85 / 30.55 (recess lips): area 110.4, y[0.75, 6.45], z[−10, 83.02]
- x = 11.15 / 26.25 (top bosses): area 97.2, y[223.10, 230.20], z[18.20, 54.83]
- z = 0.00 / 73.02: area 156.4, x[0, 37.4] × y[0, 7.2]; z = −10.00 / 83.02: area 112.9, x[6.85, 30.55] × y[0, 7.2]
- z = 18.20 / 54.83 (column front/back): area 4920.8, x[7.50, 29.90] × y[8.70, 230.95]
- README Step 4: *"Push the Arm Base into the joint lines on the side of the Mount Bottom"*
  — the side grooves accept arm_base's stepped 5.0 plate + 2.2 boss edge.
- Small Ø1.57 circles at (x≈3.6/227, …) are M2 pilot holes (M2×16 = camera stack screws).

**Groove design targets for the plate wings** (from arm_base edge):
- channel width **5.0 + 0.3 clearance** (arm_base plate), boss relief **2.2 + 0.2**,
  engagement length ≥ ~30 mm, entry chamfer for insertion.

---

## 6. Common pitfalls (API)

1. **P1 Wrong translations endpoint** — doc-level `POST /translations` is a **multipart
   IMPORT** → JSON there = **415**. Export = `POST /partstudios/…/translations` with JSON.
2. **P2 No GET on featureid** — `GET …/features/featureid/{fid}` → **405**. Update = POST only.
3. **P3 MCP writes dead** — create → 400, PATCH → 405. REST only.
4. **P4 Booleans broken in this doc** — INTERSECTION/UNION fail. Engraving done as
   **SUBTRACTION of `A − (A − B)`** (i.e. intersection via two subtractions).
5. **P5 Update body shape** — must be `{"btType":"BTFeatureDefinitionCall-1406","feature":{…}}`
   with the full GET feature object; missing wrapper → error.
6. **P6 Export zip has both parts** — classify PLATE by vol >150 000 (ARM ≈59 253).
   (Once overwrote the plate STL with the arm by taking `namelist()[0]`.)
7. **P7 STL units** — export can be metres → rescale ×1000 if bounds < 1.
8. **P8 urllib** — `urlopen(url, headers=…)` TypeError → `Request(url, headers=…)`.
9. **P9 trimesh BytesIO** — `trimesh.load(io.BytesIO(b))` needs `file_type='stl'`.
10. **P10 contains() ray artifacts** — points exactly on imprint/protector boundary walls give
    flaky solid/void votes. Verify with **81-sample majority vote** nudging x,z by
    ±0.0137…±0.0621; a lone "violation" at a wall is an artifact, not geometry.
11. **P11 Section parity** — a vertex lying exactly on the section plane misaligns crossing
    parity → nudge plane or majority-vote instead.
12. **P12 `to_planar()` bounds are in plane coords, not world** — use `path.discrete` +
    world axes for measurements.
13. **P13 Engraving depth ≤ 2.0 mm** — arm pockets are only 2.2 deep; deeper cuts carve the
    solid backing → internal slot. Slab `FJOXNmt4AcRVkvU_2` expression = 2 mm.
14. **P14 Don't thin the plate** for nut engagement — nut rides pocket floor at −0.95,
    fixed by the front face.
15. **P15 Local tooling** — matplotlib broken (use PIL + trimesh); `cq.TopoDS` absent (use
    `OCP.TopoDS`); node names may be localized (find nodes by type, not name).
16. **P16 /tmp wiped on restart** — persist everything in the repo's `SO101-ref/`.

---

## 7. Next task brief — modular side extensions

**Requirements (user):**
1. Extend the plate outline sideways (wings) — Sketch 1 `FrdYJznNlLGRbpo_0`.
2. Modular **single → double arm** with **variable distance** (wings must accept attachments
   at adjustable x positions → **slots/obround holes**, not just round holes).
3. **Center truss** between two arms will be 3D-printed later — wings must expose a bolt
   pattern + grooves it can clamp into.
4. **Overhead cam mount designed later** — but the wing side must stay compatible with its
   attachment methodology: **repeat arm_base's 4×Ø5 pattern (63.5/55.552/69.775)** and/or
   provide **grooves that accept arm_base's stepped edge (5.0 plate + 2.2 boss)**.
5. Preserve everything currently verified.

**How to add the grooves (recommended procedure):**
- **Design:** side-wing edge grooves (sliding joint, mirrors cam-mount "joint lines"):
  5.3 wide × ~2.9 total depth (5.0 channel + 2.4 relief) OR simple **Ø5 obround slots**
  along x (20–30 travel) for variable spacing. Keep ≥4 mm material if groove crosses
  engraved zones; keep clear of r6 bolt-seat protectors, nut pockets, teardrops, and the
  2051 mm² base mating lips (front y=0 — lips sample at y=−0.5, so back-side grooves must
  stop ≥0.5+ margin short of the front, i.e. depth ≤ ~3 from back face y=−6 if crossing
  under lips — simplest: put attachment features only where base lips aren't, i.e. wings).
- **Implement (REST):** preferred = `POST …/featurescript` with FeatureScript that sketches
  the profiles on the back face (or a wing sketch plane) and `extrudeRemove`s them;
  alternative = hand-author sketch + extrude features via `POST …/features` (validate shapes
  against `GET …/featurespecs` first). Insert after engraving (index > 22) unless grooves
  must interact with it. Re-GET features → all `featureStates` OK.
- **Widen outline:** edit Sketch 1 rectangle (POST featureid update on its entities),
  confirm downstream features regenerate (protectors/slab may need their sketches extended
  too if wings should be engraved — otherwise wings stay flat, which is fine for mating).
- **Verify (mesh suite, after STL export):** engraved floor −1.95 void / −2.05 solid /
  −5.95 solid where engraved; protector rings 144/144 solid; web @−0.5 solid; counterbore
  void @−3.0; holes open; teardrop apex +z 3.6; **base lips 198/198 supported**; new: groove
  width/depth probes + wing hole pattern position check (±0.1 vs 63.5/55.552/69.775);
  `is_watertight` + 2 bodies.
- **Deliver:** updated `SO101-mount_plate_engraved.stl`, regenerated renders
  (`render_faceon/stackup/assembly_section.png` recipes in session history — PIL section
  raster, z=−37.3).

**Effort estimate:** small-to-moderate — outline edit + 1–2 new features; the expensive part
is re-running the verification suite, all of which is documented above.

---

## 8. Deliverables of the old flat plate (superseded by §9)

| File | Status |
|---|---|
| `SO101-mount_plate_engraved.stl` | ✅ 2 mm engraving, watertight, 288 459.3 mm³ |
| `SO101-arm_reference.stl` | ✅ |
| `Base_SO101_spotface.step/.stl` | ✅ chosen screw fix (flat seat y=15.20) |
| `render_faceon.png` | ✅ depth-colored (gray lands / blue 2 mm lattice / red holes) |
| `render_stackup.png` | ✅ A fail-red / B OK +0.66 spare / C OK |
| `render_assembly_section.png` | ✅ |
| `SO101-mount_features.json` | ✅ live feature tree snapshot |
| `SO101-ref/HANDOFF.md` | this file |

Onshape doc holds: mount Part Studio (plate+arm), `Base_SO101`, `Base_SO101_spotface`.

---

## 9. 2026-09-28 — REDESIGN: L-mount from arm_base (supersedes plate §3–§8 for printing)

**Why:** the old plate's engraving copied arm_base inverted → base ribs sat on plate ribs
("hills on hills"), no interlock. Stock arm_base grooves *do* line up with the SO-101 base
ribs (8–12 ribs, 2.9 wide, 2.4 tall below base flat y_b=2.4).

**New part:** `SO101-mount_L_interlock.stl` (built by `build_mount_L.py`, renders `render_L.py`)
- Top plate = `SO101-arm_reference.stl` (= SO-ARM100 arm_base, cam-mount side tabs kept),
  shifted (−0.18, 0, −0.33) so its holes hit base holes exactly; 7.2 thick; rear notch filled.
- Interlock pocket = base underside region below its flat (ribs + top-hole bosses), from a
  ray-cast heightmap of `Base_SO101_spotface.stl` (STL not watertight → no mesh boolean),
  +0.2 side clearance, 2.6 deep. Base flat bears on plate top y=0.
- Back panel matches Onshape Part Studio 1 plate: x ±90 (180 wide), 10 thick (z −65…−55),
  y 0 → −183.8; 7.2-thick bridge (x ±90) joins arm_base rear edge (z −44.43) to the panel.
- Ø5 holes, teardrop apex +z; hex nut pockets AF 8.24, vertex +z, from bottom to y=−3.5.
- Stack-up **with the spotfaced base only** (#8-32×3/4" button head; stock base → see §10): seat y=12.80, tip −6.25, nut −5.74…−3.50
  → tip 0.51 past nut; web nut↔rib pocket floor 0.90.
- Verified: watertight after STL reload, 1 body, vol 407 351 mm³; base↔plate min gap 0.0
  (no interference); flat contact 3 121 mm²; rib→wall ≥0.20; rib→floor 0.20; 4 holes open.
- Print: panel outer face (z=−65) on bed, +z up. Footprint 180 × 184, height 110.
- Open risk: full-width panel may touch cam_mount_bottom if that column extends past the
  arm_base rear edge (not modeled).

---

## 10. 2026-09-28 — Screw check against the OFFICIAL base (SUPERSEDED by §11: user switched to M5)

Source: `Base_SO101_official.stl` = TheRobotStudio SO-ARM100 `STL/SO101/Individual/Base_SO101.stl`
(downloaded 2026-09-28). Same frame/bounds as our reference base, vol 122 690 mm³, and it is
**watertight** — use it for exact 3D booleans (our `Base_SO101_spotface.stl` is not).

**Interlock with the official base: OK.** 3D boolean base∩mount = 0.0 mm³; the base flat bears on
3 120 mm² of the plate top; rib→pocket wall ≥ 0.14 mm (diagonal pixel; 0.2 as designed).

**Base hole:** Ø5.0 through (teardrop toward +z), 45° countersink from r2.5 @ y_b=15.2 to the pad
top @ 17.5. Head column above is clear.

**Screw:** #8-32 × 3/4" button head socket cap, ASME B18.3: head **Ø0.312" (7.92) × 0.087" (2.21)**,
major Ø0.164" (4.17), length 19.05 under the head. The dome is on top; the **underside is a flat
bearing face**, so the dome does not help it sink into the countersink. The Ø7.92 head rests on
the 45° cone at r=3.96 → **y_b=16.68**. (The first pass wrongly used socket-cap head Ø6.86 → 16.14.)
Spotfaced base: flat seat at 15.2 out to r≥4.5 → the Ø7.92 head sits flat at 15.2.

**Stack-up** (plate frame; plate top y=0, bottom −7.2; nut pocket ceiling −3.5, nut 2.24 thick):

| base | head seat | tip (−19.05) | nut −5.74…−3.50 |
|---|---|---|---|
| official (stock) | 14.28 | **−4.77** | **short 0.97**: 1.27/2.24 mm threaded (~1.6 threads) |
| spotfaced | 12.80 | −6.25 | through, +0.51 |

Render: `render_screw_section.png` (built by `render_screw_section.py`). Sections along the front
face (normal z) at both bolt rows (z=−37.3 and 32.475), whole width, plus zooms: base grey,
mount blue, screw red, nut orange; per-bolt seat/tip/engagement labels.

**Unverified:** the nut thickness 2.24 comes from the old notes. A standard #8-32 hex nut is
3.18 (1/8") thick → with that nut the stock-base case is ~1.9 short.

**Fix options offered to the user (none chosen yet):**
1. Print `Base_SO101_spotface.stl` (recommended; 3/4" passes the nut, head seats flat). No mount change.
2. Official base + 7/8" screws: tip −7.95 sticks out 0.75 below the plate bottom. Needs the plate
   thickened locally (~1.5 mm) around the bolts, **not** at the cam-mount tabs (they must stay 7.2).
3. Raising the nut is no longer viable: it would need 0.97 mm, which cuts into the rib pockets.

Change `NUT_CEIL`, `NUT_T`, `SCREW_L`, `T_PAD`, `BUTTON_D`, `CSK_D` in `build_mount_L.py`, then rerun
`python3 build_mount_L.py && python3 render_L.py && python3 render_screw_section.py` and check
the printed stack-up line plus the section render. `render_screw_section.py` measures the seat
itself from the base mesh, so a changed base file shows up there directly.

**Tooling:** python3 with trimesh, manifold3d, shapely, cv2, scipy, PIL (matplotlib broken, P15).
The Onshape document is **not** updated with the L-mount; it exists only as these local meshes.

---

## 11. 2026-09-29 — M5 × 25 screws, thicker bolt pad (current)

**User input:** screws on hand are **M5 × 25 button head** and **M5 × 25 countersunk**. Button heads
go in the **rear row** (z=−37.3, near the back panel), countersunk in the **front row** (z=32.475),
so the low-profile heads sit where the arm swings. The extra length is used for a thicker plate so
the captive nuts sit deep and stable. (The #8-32 flat socket head from Grainger 811YX3 was dropped:
a 3/4" flat head ended 0.9–1.6 short of passing the nut on any base.)

**Base:** use the **official** `Base_SO101` (its 90° countersink, cone y = 10.3 + r in plate frame,
r 2.5…4.8, pad top 15.1). Do **not** use `Base_SO101_spotface` any more: the flat seat would leave
the countersunk heads unsupported. (The build still reads the spotface STL only for the underside
rib heightmap; the underside is identical.)

**Mount changes** (`build_mount_L.py`):
- `T_PAD = 12.0`: pad below arm_base down to y=−12, footprint = arm_base section + notch + bridge,
  limited to |x| ≤ `PAD_X = 46`. The cam-mount side tabs (|x| ≳ 56) stay 7.2 thick.
- `HOLE_D = 5.5` (M5 normal clearance, teardrop +z).
- Nut pockets: M5 ISO 4032 nut 8 AF × 4.7 → `NUT_AF = 8.3`, `NUT_T = 4.7`, `NUT_CEIL = −3.8`
  (web 1.2 to the rib-pocket floor at −2.6; was 0.9). Pocket open from y=−12 up to −3.8.

**Stack-up** (plate frame; measured on the official base by `render_screw_section.py`):

| row | screw | head | seat | tip | nut −8.50…−3.80 | tip to pad bottom −12 |
|---|---|---|---|---|---|---|
| rear | M5×25 button (ISO 7380, Ø9.5 × 2.75, L under head) | flat underside drops 0.05 into the cone | underside 15.07 | −9.93 | full 4.70, +1.43 past | 2.07 inside |
| front | M5×25 countersunk (ISO 10642/DIN 7991, 90°, Ø10, L overall) | cone on cone | top 15.32 (0.22 proud of pad) | −9.68 | full 4.70, +1.18 past | 2.32 inside |

Countersunk head Ø range 9.43–10 → head top 15.0–15.3, tip −10.0…−9.7: all pass the nut, none reach
the pad bottom.

**Verified:** watertight, 1 body, vol 444 929 mm³; 3D boolean vs official base = 0.0 mm³; 4 holes
open; pad bottom −12 at the bolts; tabs at x=±66, 60 still bottom −7.2.
Renders: `render_screw_section.png` (both rows + zooms), `render_L_*.png`, and from
`render_assembly_M5.py`: assembly stills `render_asm_{iso,top,front,right}.png` (grid
`render_asm_views.png`) and 5 s section-sweep videos `slice_front_pair.mp4` / `slice_rear_pair.mp4`
(plane normal z swept ±7 mm through each bolt row, top-view inset shows the cut).
**2026-09-29 fix:** `render.py` projected with `basis` instead of `basis.T`, so all older 3D stills
were viewed from the wrong side (old "bottom" render was really the top). Fixed; `render_L_*` re-rendered.

**Risks / unverified:**
- The official base hole is Ø5.0 as modelled; M5 shank is 4.82–4.98. A printed Ø5.0 hole usually
  comes out undersize → run a 5.2–5.5 mm drill through the base holes if the screws bind.
- Countersunk Ø10 head stands 0.2 above the base pad (base countersink tops out at Ø9.6).
  Deburr/ream the base countersink if it must be flush.
- Nut thickness 4.7 (standard) assumed; a thin nut (ISO 4035, 2.7) still works (more tip past).

## 12. How the design was made (method, for re-doing it)

1. **Reference geometry:** SO-ARM100 `arm_base.stl` (overhead-cam mount, already interlocks with the
   base footprint and bolt pattern) and the official `Base_SO101.stl`, both placed in the old
   Onshape plate frame (x lateral, y up, plate top y=0, `z_plate = z_base − 29.8`).
2. **Interlock:** ray-cast heightmap of the base underside → mask where it drops below its flat
   (ribs, hole bosses) → cv2 contours → shapely polygons, +0.2 clearance → extruded 2.6 deep and
   subtracted from the plate top. The base flat bears on the remaining plate top.
3. **Solid modelling:** manifold3d booleans in a work frame W = (x, −z, y) so every 2D profile
   extrudes along plate y. Union: arm_base + rear-notch fill + bridge + back panel (Onshape Part
   Studio 1 size) + bolt pad; subtract: rib pockets, teardrop holes, hex nut pockets from below.
   `simplify(0.001)` before export removes coplanar slivers.
4. **Screw fit:** section the official base + mount through each bolt row (plane normal z); seat each
   head by ray-casting the base within the head radius (button: highest surface; countersunk:
   lowest head-top where the 90° cone clears the base); tip = seat − length; compare with nut zone
   and pad bottom.
5. **Checks each rebuild:** watertight/1 body, 3D boolean vs official base = 0, holes open, tabs
   unchanged, then look at `render_screw_section.png` yourself.
6. **Print:** back panel outer face (z=−65) on the bed, +z up; hole teardrops and nut hex vertices
   point +z so they bridge without support.
