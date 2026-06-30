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
`step_b3_align.py` (LiDAR-anchored metric alignment of that depth),
`step_b3b_consistency.py` (multi-view consistency filtering → per-frame weight),
`step_b4_cloud.py` (back-project → dense colored image cloud),
`step_c_fuse.py` (LiDAR × image geometry fusion → `fused_cloud`),
`step_d_mesh.py` (TSDF/Poisson mesh → `mesh_fused.ply`; TSDF default = clean).
Remaining: Phase E textures the mesh (multi-view projection).
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

### B3b — Multi-view consistency filtering ✅ (`step_b3b_consistency.py`)  ← fixes blobby objects
- **The in-environment fix for noisy/blobby objects** (no gsplat / no CUDA build —
  pure torch/numpy, runs on torch 2.12+cu132). Classical MVS geometric
  consistency applied to the B3 aligned depth: back-project each ref pixel,
  reproject into covisible neighbours (selected by optical-axis similarity), and
  check depth agreement (TAU=5%).
- **Key subtlety — `seen` vs `agree`:** only *judge* a pixel when ≥`MIN_SEEN`(3)
  neighbours actually saw it; cull only confirmed-inconsistent pixels (the object
  blobs, seen by many, disagreeing). Pixels too few neighbours could check (the
  grazing floor strips that barely overlap across the spin) are **unverifiable →
  kept**. This was essential: a naïve count-only gate culled the whole floor.
- **Output:** `output/depth/consistency/<stem>.npy` (uint8 weight) +
  `consistency.json`. `Camera.confidence()` multiplies it in, so the existing
  confidence-gated TSDF (Phase D) automatically meshes only consistent geometry.
  Median 67% of valid depth kept; ~27–95 s.
- **Result:** objects (table, box) become distinct structures instead of blobs,
  walls cleaner, **floor preserved** — see Phase D. Still bounded by the monocular
  + low-parallax ceiling (not photoreal); learned-MV (VGGT/DUSt3R, also build-free)
  or better capture remain the further levers.

### B4 — Back-project to dense image cloud ✅ (`step_b4_cloud.py`)
- Unproject each image's B3 aligned depth → refined odom frame
  (`X_odom = pose · z·K⁻¹·[u,v,1]`), same frame as the LiDAR map.
- **Normals from the depth map** (neighbour cross-product, oriented toward the
  camera — correctly oriented for free, good for Phase D meshing).
- **Confidence taper (small B3 addition):** B3 now stores `anchor_zmax` (95th-pct
  anchored depth, median 3.0 m); `Camera.confidence()` decays to 0 between
  `anchor_zmax` and 1.5× it, so extrapolated far-depth never enters the cloud.
  This pulled the cloud's x-extent from −9.4 m back inside the LiDAR's −7.3 m.
- **Streaming confidence-weighted voxel hash** (1.5 cm): per-image reduce, then
  one vectorised group-by → memory bounded by occupied voxels. **18.6 M pts from
  978/1060 images → 3.89 M voxels in ~90 s.** The 82 dropped images are exactly
  the low-quality frames B3 flagged (tapered to zero confidence — working as
  designed).
- **Gap-fill confirmed:** 518k floor points recovered below the LiDAR z-band
  [1.88, 4.78] — geometry the Velodyne never saw. Overlap-surface alignment is
  consistent with B3's 18 mm (the residual NN-to-LiDAR distance is dominated by
  inter-ring gaps + the map's own 5 cm voxels, not misalignment).
- ⚠️ Known noise: radial smearing from grazing-angle / uncertain monocular depth
  — left for Phase C (LiDAR fusion + statistical/radius outlier removal). A
  view-incidence-angle confidence term is a candidate refinement.
- **Output:** `output/fusion/image_cloud.npz` (`points, colors, normals,
  confidence`) + `image_cloud.ply` (view). Read via `common.load_image_cloud`.

**New deps for Phase B:** torch + a depth model (learned path) or COLMAP/OpenMVS (MVS path),
plus OpenCV for undistort/projection.

---

## PHASE C — Geometry Fusion ✅ (`step_c_fuse.py`)  ← LiDAR × image into one dense cloud

- **LiDAR scaffold rebuilt in the B1 frame:** the saved `surfels.npz` is in a
  mixed Step-1/Step-3 frame (~2.6 cm off the image cloud), so Phase C re-places
  the scans with `trajectory_poses_world` (Step-1 dense) and reuses
  `step4_surfels.fuse_surfels` → 71.5k oriented surfels, frame-consistent.
- **Image cloud pre-cleaned** (radius-outlier, dropped 94.5k smear pts).
- **Confidence-weighted voxel hash @ 2 cm, hard per-voxel LiDAR-wins:** a voxel
  with any LiDAR takes its geometry from LiDAR; image-only voxels are
  confidence-weighted image points. Geometry and colour use separate weights so
  colourless LiDAR never darkens image colour (LiDAR-only voxels go grey, pending
  Phase E texture).
- **Cleanup:** drop weak image-only voxels + radius-outlier (LiDAR protected).
- **Globally consistent normals:** oriented toward the trajectory centroid
  (indoor inside-out scan → all normals face the interior) so Poisson behaves.
- **Result:** **1.27 M fused voxels** (71.5k LiDAR-backed + 1.20 M image-only),
  111.8k floor voxels recovered below the LiDAR z-band, bbox hugging the LiDAR
  extent, in ~19 s. Provenance render confirms LiDAR backbone + image gap-fill
  coexisting cleanly.
- ⚠️ Minor residual: grazing-angle LiDAR ring streaks at the edges (LiDAR-backed,
  so protected here) — Phase D density-trim + bbox-crop remove them.
- **Output:** `output/fusion/fused_cloud.npz` (`points, colors, normals,
  confidence, is_lidar`) + `fused_cloud.ply`. Read via `common.load_fused_cloud`;
  drops straight into `step6_mesh.poisson_mesh`.

---

## PHASE D — Surface + Mesh ✅ (`step_d_mesh.py`; `step6_mesh.py` = LiDAR-only baseline)

Two paths in `step_d_mesh.py`; **TSDF is the default** because Poisson-on-points
was too noisy.

- **TSDF + Marching Cubes (DEFAULT, the clean mesh).** Volumetrically integrate
  the B3 aligned depth maps into a truncated SDF (`ScalableTSDFVolume`), now
  **gated by B3b multi-view consistency** via `Camera.confidence()` (consistency
  is the primary gate, so CONF_THR dropped to 0.12 and voxel to **2 cm / trunc
  6 cm** for crisper objects), + per-frame quality gate (skip `fit_quality`
  < 0.35), then Marching Cubes. Cleanup: small-component removal + Taubin (5).
  Result: **~918 k tris, 484 k verts, coloured, in ~25 s**, 943/1060 frames.
  **Objects (table/box) are now distinct, walls clean, floor preserved**
  (z-min −0.63). Big step up from the pre-consistency blobby TSDF.
- ⚠️ Still bounded by the monocular + ≈1.6 m-spin ceiling: residual edge fringe
  + not photoreal. Further (build-free) levers: learned multi-view (VGGT/DUSt3R)
  or a translating/orbit recapture.
- **Poisson (baseline, `method="poisson"`).** Poisson on Phase C's `fused_cloud`
  (reuses `step6`): depth 10 → density trim → bbox crop → component removal →
  Taubin → simplify. Meshes ~1.2 M individually-noisy monocular-depth points, so
  the surface is bumpy — kept only for comparison.
- **Why TSDF wins:** the noise was per-view depth disagreement + grazing smear
  baked into points; TSDF resolves it by weighted averaging in the SDF *before* a
  surface exists, instead of fitting a surface through the noise.
- **Output:** `output/fusion/mesh_fused.ply` + `mesh_fused_preview.png` (shaded
  offscreen render). Read via `common.load_fused_mesh`; Phase E's input.
- 🔭 Further levers if needed: integrate LiDAR depth into the TSDF too (metric
  backbone on bare walls); smaller voxel (2 cm) for more detail; trim open-boundary
  fringe; or per-pixel incidence-angle weighting to cut grazing-floor noise.

### ⚠️ DIAGNOSIS — the mesher is NOT the bottleneck (objects stay noisy/blobby)

Goal is sharp, realistic OBJECTS (table, box), not just flat walls. Tested and
grounded:
- **Single-frame B2 depth is crisp** (box/monitor edges are clean), but
  **monocular depth is not multi-view consistent** — each frame guesses an
  object's 3D shape slightly differently.
- Fusion must then either **average** those disagreements (coarse TSDF 3 cm/12 cm
  → blobby, rounded objects) or **preserve** them (fine TSDF 1.5 cm/4 cm →
  sharper but noisy/fragmented). Verified both — you cannot get sharp **and**
  clean from this data by tuning the mesher.
- **TSDF, Poisson, BPA, Dual Contouring all average the same inconsistent
  inputs** → none fixes it. BPA/Advancing-Front are *interpolating* → reproduce
  noise (worse). Screened Poisson == our `method="poisson"` baseline already.
  Dual Contouring helps sharp corners we don't lack, and amplifies noise on a
  noisy SDF. So the meshing-algorithm menu is a dead end for this symptom.

**Root cause:** multi-view-inconsistent monocular depth + the near-stationary
spin (≈1.6 m footprint → almost no parallax, can't see object sides/backs).

### Real paths to sharp/realistic objects (ranked)
1. **Phase F neural with multi-view photometric consistency** — 2D Gaussian
   Splatting → mesh, or SuGaR. Fits ONE geometry to all RGB views, so edges that
   mono-depth averaging destroys come back. Use the LiDAR mesh/cloud as
   prior+regularizer for the low parallax. **This is the in-data answer.**
   (Raw 3DGS novel-view renders look more photoreal than any extracted mesh — use
   that if the deliverable can be renders rather than a strict mesh.)
2. **Sharper depth model in B2 — DepthPro** (metric, sharp boundaries, drop-in via
   the pluggable `DepthBackend`). Cheap; sharpens per-frame edges, but it's still
   monocular → still multi-view-inconsistent → objects improve but stay soft.
3. **Better capture (the true fix):** orbit/translate around objects + enable Spot
   depth. A spin-in-place fundamentally can't do realistic furniture; no
   algorithm fully overcomes it.

**RESOLVED (in-environment, no gsplat):** multi-view consistency filtering
(`step_b3b_consistency.py`, B3b) + consistency-gated finer TSDF → objects become
distinct, floor preserved. This is the working fix given the fixed torch
2.12/cu132 env. 2DGS remains blocked (below) and is optional upside, not required.

**Status of (c):**
- ✅ **DepthPro done** (B2 now defaults to `apple/DepthPro-hf` via `HFDepthBackend`;
  DA-V2 still selectable). Metric scale much better (x0.78 vs DA-V2 x0.51), B3
  held-out 17 mm, walls a bit cleaner — but **furniture/floor still noisy/blobby**.
  Confirms the ceiling: a sharper *monocular* model doesn't fix multi-view
  inconsistency. (DA-V2 mesh snapshot kept at `mesh_fused_dav2.ply`.)
- ⛔ **2DGS BLOCKED on environment.** gsplat 1.5.3 (latest) won't build its CUDA
  kernels against **torch 2.12.1+cu132** on Windows: (1) it calls torch-internal
  `_jit_compile` with an outdated signature (torch 2.12 added `with_sycl`/
  `extra_sycl_cflags`), (2) it passes `/Wno-attributes` to MSVC `cl.exe`. nvcc
  12.8 *does* target Blackwell sm_120 fine; the problem is torch 2.12 is too new
  for gsplat. Blackwell forced us onto new torch, which gsplat hasn't caught up to.
  **To unblock:** a dedicated venv with torch ~2.7/2.8 **+cu128** (supports sm_120
  AND is gsplat-compatible) + ninja + the installed VS2022 MSVC. Then build the
  actual 2DGS train + mesh-extraction pipeline (substantial). Even then, the
  ≈1.6 m spin is parallax-starved — the LiDAR-mesh prior is essential, and a
  translating/orbit capture remains the bigger lever.

---

## PHASE E — Multi-View Texture ⬜  (was "Step 5 colour")  ← NEXT

- Inputs wired: `common.load_fused_mesh` (Phase D geometry) + `load_camera_manifest`
  (5 fisheye cams with refined `pose`, `K`, `.read_image()`, `.confidence()`,
  `.project()`). Same refined odom frame throughout.
- Project mesh vertices into all 5 fisheye views; visibility checks (occlusion via
  z-buffer / mesh raycast, incidence angle, distance, blur); weighted multi-view
  blend; bake to vertex colour or a UV texture atlas.
- Replaces Phase D's rough Poisson vertex colours (and fills the grey LiDAR-only
  patches) with real image texture → the END-GOAL textured mesh.

---

## PHASE F — Neural Refinement (the real path to sharp objects) ⬜

- 2D Gaussian Splatting / SuGaR on the 5-camera images with the refined poses;
  extract mesh; use the Phase D fused mesh as geometric prior/regularizer
  (prevents collapse on low-parallax motion). Output: refined textured mesh.
- ⛔ **Env blocker (see Phase D diagnosis):** gsplat 1.5.3 ✗ torch 2.12+cu132 on
  Windows. Needs a dedicated **torch 2.7/2.8 +cu128** venv (sm_120 + gsplat-
  compatible) + ninja + VS2022 MSVC (installed). Bounded setup, then a real
  train/extract pipeline.
- Note: raw 3DGS novel-view renders look more photoreal than any extracted mesh —
  use that if the deliverable can be renders rather than a strict mesh.

---

## What runs end-to-end today

**LiDAR branch:** `step1 → step3 → step4 → step6` → `mesh.ply` (LiDAR-only, coarse).
**Image branch:** `step0 → step_b1 → step_b2 (DepthPro) → step_b3 → step_b3b → step_b4`.
**Phase D** (`step_d_mesh`, TSDF gated by B3b consistency) → `mesh_fused.ply`
(~918 k tris; distinct objects, clean walls, floor). The Poisson path + Phase C
`fused_cloud` remain for the point-cloud route. Phase E (texture) is next.

## Immediate next decisions

1. **Depth method for Phase B2:** ✅ DECIDED — learned monocular metric depth + LiDAR
   anchor (Depth Anything V2 / Metric3D / UniDepth). Posed MVS dropped to fallback.
2. ✅ DONE — image branch **B0–B4**, **Phase C fusion**, and **Phase D mesh**
   (`step_d_mesh.py`, **TSDF** → `mesh_fused.ply`, 614 k tris, flat walls + floor;
   Poisson kept as noisy baseline). Next is **Phase E**: project the 5 fisheye
   cameras onto the mesh for real multi-view texture.
3. **Capture a translating session, ideally with depth** — improves coverage for both
   branches and lets us validate B3 against real Spot depth.
