"""
STEP B4 - Back-project to dense image cloud  (image branch, after B3)
=====================================================================

Unprojects every image's LiDAR-anchored metric depth (B3) into the refined odom
frame and consolidates the 5-camera sweep into ONE dense, colored, normal-bearing
point cloud with per-point confidence.  This is the dense image-side geometry
that fills the holes the sparse Velodyne left; Phase C fuses it with the LiDAR.

Per image:
  * Back-project  X_odom = pose . (z . K^-1 . [u,v,1])  on a strided pixel grid,
    in the refined odom frame (B1 pose) - the same frame as the LiDAR map.
  * Normals from the depth map (cross product of neighbouring back-projected
    points), oriented toward the camera -> correctly-oriented normals for free
    (better than estimating orientation from the merged cloud).
  * Colour (BGR->RGB) and per-pixel confidence (B3 .confidence(): quality x edge
    x range-validity x extrapolation taper) carried per point.

Consolidation is a STREAMING, confidence-WEIGHTED voxel hash: each image is first
reduced to its own voxels, then all images are combined in one vectorised
group-by.  Memory is bounded by occupied voxels (room surface), not total pixels,
so all 1060 x ~300k px stay tractable.  LiDAR is NOT merged here - that is Phase C.

Run (from repo root, after step1 -> step_b1 -> step_b2 -> step_b3):
    python src/odometry/step_b4_cloud.py

Output (under <session>/output/fusion/):
    image_cloud.npz   points, colors, normals, confidence (the Phase C product)
    image_cloud.ply   colored + normals, for viewing
"""

import os
import time

import numpy as np
import open3d as o3d

import common


# --- Tunables ---------------------------------------------------------------
VOXEL = 0.015            # consolidation voxel (m)
PIXEL_STRIDE = 4         # subsample pixels per image
CONF_THR = 0.30          # drop pixels below this confidence
MAX_RANGE = 10.0         # hard depth cap (the B3 taper does the real work)
MIN_WEIGHT = 0.30        # drop voxels with less total accumulated confidence

# Voxel-key packing: room indices fit comfortably in +/-2^20 per axis.
_OFF = 1 << 20


def _pack(keys_xyz: np.ndarray) -> np.ndarray:
    """(N,3) int voxel indices -> (N,) unique int64 key."""
    k = keys_xyz.astype(np.int64) + _OFF
    return (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]


# --- Per-image back-projection ----------------------------------------------

def backproject(cam, stride=PIXEL_STRIDE, conf_thr=CONF_THR, max_range=MAX_RANGE):
    """One image -> (points, colors, normals, conf) in refined odom, or None."""
    z = cam.read_depth_aligned()
    if z is None:
        return None
    conf = cam.confidence(z=z)
    img = cam.read_image()                          # BGR
    H, W = z.shape

    # Camera-frame points on the full grid (general K, handles skew).
    Kinv = np.linalg.inv(cam.K)
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    pix = np.stack([uu.ravel(), vv.ravel(), np.ones(H * W)])
    dirs = (Kinv @ pix).T.reshape(H, W, 3)
    Xc = dirs * z[..., None]                         # (H,W,3)

    # Normals from depth-map gradients, oriented toward the camera (origin).
    n = np.cross(np.gradient(Xc, axis=1), np.gradient(Xc, axis=0))
    n /= np.linalg.norm(n, axis=2, keepdims=True) + 1e-9
    flip = np.sum(n * Xc, axis=2) > 0
    n[flip] *= -1

    # To odom (rotation for normals, full transform for points).
    R, t = cam.pose[:3, :3], cam.pose[:3, 3]
    Xo = Xc.reshape(-1, 3) @ R.T + t
    No = n.reshape(-1, 3) @ R.T
    rgb = (img[..., ::-1] / 255.0).reshape(-1, 3)

    # Subsample + validity mask.
    smask = np.zeros((H, W), bool)
    smask[::stride, ::stride] = True
    s = smask.ravel()
    zz = z.ravel()
    cc = conf.ravel()
    good = (s & np.isfinite(zz) & (zz > 0.2) & (zz < max_range)
            & (cc >= conf_thr) & np.isfinite(No).all(1))
    if good.sum() == 0:
        return None
    return Xo[good], rgb[good], No[good], cc[good]


def reduce_image(pts, cols, normals, conf, voxel=VOXEL):
    """Confidence-weighted reduction of one image's points to its voxels."""
    keys = _pack(np.floor(pts / voxel).astype(np.int64))
    uniq, inv = np.unique(keys, return_inverse=True)
    m = len(uniq)
    w = np.zeros(m)
    np.add.at(w, inv, conf)
    wp = np.zeros((m, 3)); np.add.at(wp, inv, pts * conf[:, None])
    wc = np.zeros((m, 3)); np.add.at(wc, inv, cols * conf[:, None])
    wn = np.zeros((m, 3)); np.add.at(wn, inv, normals * conf[:, None])
    return uniq, w, wp, wc, wn


# --- Driver -----------------------------------------------------------------

def run(data_root: str, voxel=VOXEL, stride=PIXEL_STRIDE, conf_thr=CONF_THR,
        min_weight=MIN_WEIGHT, limit: int | None = None, lidar_check=True):
    cams = common.load_camera_manifest(data_root)
    if any(c.align_a is None for c in cams):
        raise RuntimeError("Aligned depth missing - run step_b3_align.py first.")
    if limit:
        cams = cams[:limit]

    out_dir = os.path.join(os.path.abspath(data_root), "output", "fusion")
    os.makedirs(out_dir, exist_ok=True)

    keys_l, w_l, wp_l, wc_l, wn_l = [], [], [], [], []
    n_used, n_pts = 0, 0
    t0 = time.time()
    for i, cam in enumerate(cams):
        bp = backproject(cam, stride, conf_thr)
        if bp is None:
            continue
        n_used += 1
        n_pts += len(bp[0])
        uniq, w, wp, wc, wn = reduce_image(*bp, voxel=voxel)
        keys_l.append(uniq); w_l.append(w)
        wp_l.append(wp); wc_l.append(wc); wn_l.append(wn)
        if i % 100 == 0:
            print(f"[{i + 1:04d}/{len(cams)}] {cam.camera:10s} "
                  f"+{len(bp[0])} pts")

    # One vectorised global group-by over all per-image voxels.
    keys = np.concatenate(keys_l)
    W = np.concatenate(w_l)
    WP = np.concatenate(wp_l); WC = np.concatenate(wc_l); WN = np.concatenate(wn_l)
    uniq, inv = np.unique(keys, return_inverse=True)
    M = len(uniq)
    sw = np.zeros(M); np.add.at(sw, inv, W)
    sp = np.zeros((M, 3)); np.add.at(sp, inv, WP)
    sc = np.zeros((M, 3)); np.add.at(sc, inv, WC)
    sn = np.zeros((M, 3)); np.add.at(sn, inv, WN)

    keep = sw >= min_weight
    points = (sp[keep] / sw[keep, None]).astype(np.float32)
    colors = np.clip(sc[keep] / sw[keep, None], 0, 1).astype(np.float32)
    normals = sn[keep]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9
    normals = normals.astype(np.float32)
    confidence = sw[keep].astype(np.float32)

    npz_path = os.path.join(out_dir, "image_cloud.npz")
    np.savez_compressed(npz_path, points=points, colors=colors,
                        normals=normals, confidence=confidence,
                        voxel=np.float32(voxel))
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.colors = o3d.utility.Vector3dVector(colors)
    pc.normals = o3d.utility.Vector3dVector(normals)
    ply_path = os.path.join(out_dir, "image_cloud.ply")
    o3d.io.write_point_cloud(ply_path, pc)

    dt = time.time() - t0
    print(f"\nB4: {n_pts} back-projected pts from {n_used}/{len(cams)} images "
          f"-> {len(points)} voxels @ {voxel*100:.1f}cm in {dt:.1f}s")
    print(f"    bbox min {np.round(points.min(0), 2)} max {np.round(points.max(0), 2)}")
    print(f"Saved -> {npz_path}\n        {ply_path}")

    if lidar_check:
        _lidar_check(data_root, points)
    return points, colors, normals, confidence


def _lidar_check(data_root, points, sample=20000):
    """Nearest-LiDAR distance distribution (alignment sanity) + combined PLY."""
    lp = os.path.join(os.path.abspath(data_root), "output",
                      "global_map_optimized.ply")
    if not os.path.isfile(lp):
        return
    lidar = o3d.io.read_point_cloud(lp)
    kdt = o3d.geometry.KDTreeFlann(lidar)
    idx = np.random.default_rng(0).choice(
        len(points), min(sample, len(points)), replace=False)
    d = np.array([np.sqrt(kdt.search_knn_vector_3d(points[k], 1)[2][0])
                  for k in idx])
    print(f"    nearest-LiDAR dist: median {np.median(d)*100:.1f}cm  "
          f"p75 {np.percentile(d, 75)*100:.1f}cm  p90 {np.percentile(d, 90)*100:.1f}cm  "
          f"frac<5cm {100*(d < 0.05).mean():.0f}%")


def main():
    data_root = "./captures/session_05_20260624"

    voxel = VOXEL           # consolidation voxel size (m)
    stride = PIXEL_STRIDE   # pixel subsample per image
    conf_thr = CONF_THR     # confidence floor
    limit = None            # int for a quick first-N check

    run(data_root, voxel=voxel, stride=stride, conf_thr=conf_thr, limit=limit)


if __name__ == "__main__":
    main()
