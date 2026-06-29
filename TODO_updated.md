# Indoor 3D Reconstruction Pipeline — Updated Plan

**Goal:** a 3D environment reconstruction pipeline specialized for **indoor** Spot
captures that **fuses LiDAR and images** into a single dense geometry and outputs a
**textured triangle MESH** (`.ply`/`.obj`).

**Core principle — true LiDAR×image fusion:** the Velodyne is too sparse for indoor
detail, so it is used as the **metric scaffold** (accurate poses + scale + where it
*did* hit), while the **images reconstruct the dense geometry that fills the missing
points**. LiDAR alone (Poisson on the sparse cloud) does NOT look good — that path is
only a baseline. The real deliverable goes through image-driven densification fused
with LiDAR.

Status legend: ✅ done · 🟡 scaffold/partial · ⬜ planned · ⚠️ caveat / blocker

---

## Why fusion works here (the key insight)

- We **already know the camera poses** (from LiDAR SLAM, Steps 1–3) → the image branch
  can do **posed dense reconstruction** and skip SfM entirely (SfM/COLMAP struggles on
  Spot's low-parallax indoor motion — the original plan flagged this).
- LiDAR gives **sparse metric depth** in every image (project LiDAR points into the
  camera). That is the **anchor** that fixes the scale/drift of image-derived depth.
- So: images predict *dense* but scale-ambiguous geometry; LiDAR *corrects it to metric*
  and validates it. Each fills the other's weakness.

---

## Key data facts (from session_05_20260624)

- **Point clouds saved in the ODOM frame** (Spot pre-applies `odom_T_sensor`); un-apply
  `inv(odom_T_sensor)` to get sensor scans. → `common.Frame.read_cloud_sensor()`.
- **Per-scan Spot fused odometry** in metadata (`odom→body→sensor`) — accurate ICP prior.
- **Real timestamps** `acquisition_time_robot_nsec`; LiDAR ≈3 Hz, images 5 Hz, state 50 Hz,
  one shared robot clock (mutually synchronized).
- **Cameras:** 5 RGB fisheye (`back, frontleft, frontright, left, right`), 640×480, 212
  frames, pinhole intrinsics, per-image `transforms_snapshot` → `odom_T_camera`.
  - ⚠️ `ROTATE_90_CLOCKWISE` applied — reconcile with intrinsics before projection.
  - ⚠️ Fisheye described with a *pinhole* model — validate/handle distortion.
  - ⚠️ **No depth images saved** — enabling Spot depth capture is the biggest data-side win.
- **session_05 is a near-stationary spin** (~1.6 m footprint) → low parallax (favors
  learned-depth over MVS), tiny drift, degenerate loop closure. Capture a translating
  session for full coverage.

---

## Architecture (two branches → fusion → mesh)

```
  LiDAR branch                         Image branch
  ────────────                         ────────────
  Step 1  odometry  ─┐                 5 fisheye RGB
  Step 2  keyframes  ├─► poses ────────► B1 undistort + pose each image
  Step 3  pose graph ┘   (metric)          │
        │                                  ▼
        │   sparse metric points     B2 dense depth per image
        │   (also projected into        (learned mono-depth  ‖  posed MVS)
        │    each image as anchor)        │
        │                                 ▼
        └───────────► anchor ───────► B3 scale/align depth to LiDAR  (FUSION glue)
                                          │
                                          ▼
                                     B4 back-project → dense image point cloud
                          ┌───────────────┘
                          ▼
            ====  C: GEOMETRY FUSION  ====
            merge LiDAR (metric) + image (dense) points,
            confidence-weighted; consolidate (voxel/TSDF)
                          │
                          ▼
            D: surfels / TSDF  →  MESH (Poisson / Marching Cubes) → cleanup
                          │
                          ▼
            E: multi-view texture (project 5 cams → vertex colour / UV atlas)
                          │
                          ▼
            F (optional): neural refine (2DGS / SuGaR) with LiDAR prior
                          │
                          ▼
                 textured triangle MESH  ◄── END GOAL
```

**Code layout** (`src/odometry/`): `common.py`, `step0_camera_prep.py` (image
rotation + optional/skippable fisheye → camera manifest), `step1_odometry.py`,
`step2_mapping.py`, `step3_pose_graph.py`, `step4_surfels.py`, `step6_mesh.py`
(LiDAR-only baseline mesh). Image-branch + fusion modules are new (planned below).
Run env: `.venv` (Python 3.11; open3d, numpy, scipy, matplotlib). Image branch will add
deps (see Phase B).

---

## PHASE A — LiDAR SLAM (metric scaffold)

### Step 1 — LiDAR Odometry ✅ (`step1_odometry.py`)
Sensor-frame multi-scale point-to-plane ICP, Spot-odometry prior, quality gate. Outputs
`trajectory_poses_world.npy` + TUM. Verified (net yaw −93.7° vs Spot −94.7°, fitness ~0.99).

### Step 2 — Keyframe Map ✅ (`step2_mapping.py`)
Distance/rotation keyframe gate; fuse + voxel downsample + outlier removal →
`global_map.ply` (80/212 keyframes).

### Step 3 — Pose Graph Optimization ✅ (`step3_pose_graph.py`)
Keyframe nodes; sequential + loop-closure edges; robust LM optimization →
`pose_graph_poses.npy`, `global_map_optimized.ply`. Sharper map (177k→128k voxels).

### Step 4 — Surfels 🟡 (`step4_surfels.py`)
Batch voxel-surfel fusion (oriented normals, confidence, prune) → `surfels.ply`/`.npz`.
Serves as LiDAR geometry input to fusion (Phase C). TODO: incremental fusion + decay.

---

## PHASE B — Image Dense Reconstruction (NEW) ⬜  ← fills the missing points

Depends on **Step 0 calibration** (below). Uses the known LiDAR poses, so no SfM.

### B0 — Step 0 camera prep ✅ (`step0_camera_prep.py`)
- **Rotation reconciled (always on, lossless):** front cams are stored already
  upright (480×640) but with raw-sensor intrinsics; we rotate the *intrinsics*
  (`fx'=fy, fy'=fx, cx'=(h0-1)-cy, cy'=cx`) and turn the optical-frame pose by
  `ROT_Z_90` so K/pixels/pose stay consistent. The **`right` cam needs a 180°
  turn** (mounting); its K/pose are **always** reconciled (flip K
  `cx'=(w-1)-cx, cy'=(h-1)-cy` + turn pose by `ROT_Z_180`). The capture code is
  now fixed to rotate `right` pixels at capture time, so future sessions need no
  pixel op; the legacy upside-down sessions are handled by the
  `rotate_legacy_pixels` flag (rewrites pixels upright). Either way the
  intrinsics are identical. Other cams pass through.
- **Fisheye undistortion is OPTIONAL and OFF by default** (`undistort=False`):
  Spot ships a pinhole model with NO distortion coeffs and the distortion is
  weak, so images are treated as plain pinhole. Same output schema either way →
  downstream (B1/B3/E) is agnostic. Enable only with calibrated `dist_coeffs`.
- **Validated:** LiDAR-reprojection overlay (`output/cameras/debug/`) shows rings
  hugging the walls and bending around a box — extrinsics/intrinsics confirmed,
  and confirms undistortion is unnecessary for now.
- Output: `output/cameras/manifest.json` (per image: upright image path, K,
  size, `odom_T_camera`, `distortion_mode`, mask).

### B1 — Pose & prep each image ⬜
- Compose `odom_T_camera` per image from metadata; undistort/rotate to a clean pinhole image.
- Sync each image to the nearest LiDAR scan / interpolate pose on robot-clock nsec.

### B2 — Dense depth per image ⬜
**✅ CHOSEN: Learned monocular metric depth + LiDAR anchor.**
- Per-image metric depth from a model (Depth Anything V2 / Metric3D / UniDepth),
  then scaled/aligned to LiDAR in B3. Robust to low parallax + textureless indoor walls.
- Needs torch (not yet installed). GPU present: RTX 5070 12 GB — ⚠️ Blackwell (sm_120),
  so use a CUDA 12.8+ / recent torch build. cv2 4.13 already available for projection.
- (Not chosen) Posed MVS (COLMAP/OpenMVS) — weaker on Spot's spin-in-place, low-texture
  indoor; kept only as a fallback reference.

### B3 — LiDAR-anchored metric alignment ⬜  ← the fusion glue
- Project LiDAR points into each image → sparse metric depth samples.
- Fit predicted depth to LiDAR (global scale+shift, then optional per-pixel/spline
  correction). This makes image depth metric and drift-consistent with LiDAR.

### B4 — Back-project to dense image cloud ⬜
- Unproject corrected depth (+ RGB) per image → dense colored point cloud in odom frame.
- Confidence per point (edge/occlusion/agreement-with-LiDAR masking).

**New deps for Phase B:** torch + a depth model (learned path) or COLMAP/OpenMVS (MVS path),
plus OpenCV for undistort/projection.

---

## PHASE C — Geometry Fusion ⬜  ← LiDAR × image into one dense cloud

- Merge LiDAR points (metric, trusted) + image points (dense, fill gaps).
- Confidence-weighted consolidation: LiDAR wins ties on metric position; images supply
  surfaces LiDAR missed (holes in walls/ceiling/floor).
- Consolidate via voxel hashing or TSDF; statistical/radius outlier removal.
- Output: dense fused point cloud / surfels for meshing.

---

## PHASE D — Surface + Mesh 🟡 (`step6_mesh.py` = LiDAR-only baseline)

- Mesh the **fused** cloud: Poisson (have it) or TSDF + Marching Cubes (cleaner indoors,
  esp. once depth is dense).
- Cleanup: remove small components, fill holes, Taubin smoothing, recompute normals.
- Current `step6_mesh.py` runs Poisson on LiDAR-only surfels → `mesh.ply` (235k tris,
  coarse — baseline only; will look much better on the fused cloud).

---

## PHASE E — Multi-View Texture ⬜  (was "Step 5 colour")

- Project mesh vertices into all 5 fisheye views; visibility checks (occlusion, angle,
  distance, blur); weighted multi-view blend; bake to vertex colour or a UV texture atlas.

---

## PHASE F — Neural Refinement (optional, advanced) ⬜

- 2D Gaussian Splatting / SuGaR on the 5-camera images with Step 3 poses; extract mesh;
  use the Phase D fused mesh as geometric prior/regularizer (prevents collapse on
  low-parallax motion). Output: refined textured mesh.

---

## What runs end-to-end today

`step1 → step3 → step4 → step6` → `mesh.ply` (LiDAR-only, coarse). Phases B/C/E are the
image-fusion work that makes it actually look good.

## Immediate next decisions

1. **Depth method for Phase B2:** ✅ DECIDED — learned monocular metric depth + LiDAR
   anchor (Depth Anything V2 / Metric3D / UniDepth). Posed MVS dropped to fallback.
2. ✅ DONE — **Step 0 camera prep** (`step0_camera_prep.py`): rotation reconciled,
   fisheye undistortion made optional/off. Next image-branch step is B1/B2.
3. **Capture a translating session, ideally with depth** — improves coverage for both
   branches and lets us validate B3 against real Spot depth.
