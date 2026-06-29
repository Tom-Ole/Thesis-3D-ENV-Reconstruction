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
(LiDAR-only baseline mesh), `step_b1_image_poses.py` (image branch: sync + refine
camera poses), `step_b2_depth.py` (dense monocular metric depth per image),
`step_b3_align.py` (LiDAR-anchored metric alignment of that depth).
Remaining image-branch + fusion modules are new (planned below).
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
- Output: `output/cameras/manifest.json` (per image: upright `image_path`, `K`,
  size, `odom_T_camera`, `t_nsec`, `distortion_mode`, `mask_path`). **This is the
  single contract for the whole image branch** — downstream steps read the
  manifest, never `captures/images/` directly. `image_path` + `K` are always
  mutually consistent, and `odom_T_camera` shares the odom frame with the LiDAR
  clouds. (Note: paths are absolute — re-run Step 0 if the repo moves.)
- **Read API in `common.py`:** `load_camera_manifest(data_root) -> list[Camera]`
  with arrays parsed. `Camera` exposes `.read_image()`, `.read_mask()`,
  `.cam_T_odom`, `.size`, `.t_sec`, and
  `.project(pts_odom) -> (uv, depth, keep)` (drops behind-camera/out-of-frame;
  `keep` indexes back into the input). `.project()` is the B3 anchor primitive.

### B1 — Pose & prep each image ✅ (`step_b1_image_poses.py`)
- ✅ `odom_T_camera` + clean upright pinhole image already in the B0 manifest.
- ✅ **Temporal sync:** each image bound to its time-nearest LiDAR scan on the
  shared robot clock (median dt 32 ms, max 133 ms) — the B3 anchor scan.
- ✅ **Pose refinement:** raw Spot per-image pose made drift-consistent with the
  LiDAR SLAM trajectory. Per scan `C_i = ref_i @ inv(spot_i)` (ref = Step 1 dense
  trajectory, or Step 3 pose graph via `use_pose_graph`); SE(3)-interpolate `C`
  at the image's `t_nsec` and left-apply: `odom_T_camera_refined = C(t) @ raw`.
  Exact because `body_T_camera` is static and `C` acts on the shared odom frame.
  On session_05 the refinement moves camera centres ~109 mm median (max 215 mm,
  rot ≤3.8°) — non-trivial even on a near-stationary spin.
- **Output:** `output/cameras/poses_b1.json` (per image: `odom_T_camera_refined`,
  `lidar_index`/`lidar_t_nsec`/`dt_sec`, `pose_source`, correction magnitude).
  `common.load_camera_manifest` merges it transparently — `Camera.pose`,
  `.cam_T_odom`, `.project()` use the refined pose automatically once B1 has run,
  and `.anchor_scan(dataset)` returns the B3 scan. Downstream never branches.
- **Validated:** raw-vs-refined LiDAR reprojection overlays (`output/cameras/
  debug/*_b1.jpg`) — rings hug walls/objects equally well in both, confirming the
  refinement preserves alignment while putting all images in one global frame.

### B2 — Dense depth per image ✅ (`step_b2_depth.py`)
- **Model:** Depth Anything V2, **Metric Indoor ViT-L** (`depth-anything/
  Depth-Anything-V2-Metric-Indoor-Large-hf`) via HF transformers. Outputs depth
  directly in METRES (good init for B3) — B3 still re-anchors to LiDAR. Wrapped
  behind a `DepthBackend` so UniDepth v2 / Metric3D can drop in later.
- **Env (resolved):** torch 2.12.1+cu132 already installed; RTX 5070 / sm_120
  verified computing fp16 on GPU. Only added dep was `transformers`. Ran all
  **1060 images in 64 s (60 ms/img)** in fp16 on CUDA.
- **Output:** `output/depth/<stem>.npy` (float16 (H,W), ~650 MB total) +
  `depth_b2.json` (`model_id`, `depth_kind="metric_m"`, per-image paths). Read
  via `Camera.read_depth()` — `common.load_camera_manifest` merges it
  transparently, like B1.
- **Validated:** depth maps crisp on both landscape (back/left/right 640×480) and
  portrait (front 480×640) cams — objects cleanly separated from receding floor /
  far walls. LiDAR cross-check (project anchor scan, raw pose) gives median
  pred-vs-LiDAR r≈0.82 and a ~2× global scale offset (expected; B3 fixes it).
  Previews in `output/depth/debug/<stem>_depth.jpg`.
- (Not chosen) Posed MVS (COLMAP/OpenMVS) — weaker on Spot's spin-in-place, low-texture
  indoor; kept only as a fallback reference.

### B3 — LiDAR-anchored metric alignment ✅ (`step_b3_align.py`)  ← the fusion glue
- Per image, in the **refined odom frame**: project the B1 anchor scan (placed by
  its Step-1 pose) → sparse metric LiDAR samples; sample B2 depth at those `uv`
  (edge anchors dropped). All wired via `Camera.project()` / `.anchor_scan()` /
  `.read_depth()`.
- **Robust global affine in inverse-depth:** `1/z_lidar ≈ a·(1/z_pred)+b`
  (IRLS-trimmed). Disparity space linearises the monocular error far better than
  metres (probe-confirmed), and `a` absorbs B2's ~2× scale.
- **Spline (smoothed residual grid):** coarse robust-binned, hole-filled,
  Gaussian-smoothed disparity-residual correction on top of the affine,
  **gated by held-out error** (kept on 1051/1060; rejected where it didn't help).
- **Quality + confidence:** honest held-out residual + inlier fraction →
  per-image `fit_quality` (median 0.72; 7.7% of frames <0.3, flagged for
  down-weighting — fit quality spans 0.0–0.9, so this matters). Per-pixel
  `Camera.confidence()` = quality × edge-weight × range-validity.
- **Result:** held-out median error **58 mm (affine) → 18 mm (affine+spline)**
  across 1060 images, in 15 s.
- **Output — params-only (no second depth copy):** `output/depth/align_b3.json`
  (a, b, quality, held-out residual, grid ref) + `aligned/<stem>_corr.npy` (tiny
  coarse grids, **538 KB total**). Read via `Camera.read_depth_aligned()` (applies
  affine+grid on the fly) and `.confidence()`; merged transparently by
  `common.load_camera_manifest`. Checks in `output/depth/debug/<stem>_align.jpg`.

### B4 — Back-project to dense image cloud ⬜  ← NEXT
- Inputs wired: `Camera.read_depth_aligned()` (B3 metric depth), `.confidence()`
  (per-pixel weight), `.read_image()` (RGB), `.pose` (B1 refined odom pose).
- Unproject corrected depth (+ RGB) per image → dense colored point cloud in the
  refined odom frame (`X_odom = pose · K⁻¹·[u,v,1]·z`). Same frame as the LiDAR.
- Carry per-point confidence; threshold/voxel-downsample to keep it tractable
  (1060 images × ~300k px is huge — subsample per image and across the 5 Hz
  near-stationary frames).

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
2. ✅ DONE — **Step 0 camera prep**, **B1 image poses**, **B2 dense depth** (Depth
   Anything V2 metric-indoor, 1060 maps), and **B3 LiDAR-anchored alignment**
   (`step_b3_align.py`, held-out err 18 mm). Next is **B4** (back-project the
   aligned depth → dense colored cloud), then **Phase C** fusion.
3. **Capture a translating session, ideally with depth** — improves coverage for both
   branches and lets us validate B3 against real Spot depth.
