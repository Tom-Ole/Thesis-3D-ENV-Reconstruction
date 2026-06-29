"""
PHASE C - Geometry Fusion  (LiDAR x image -> one dense cloud)
============================================================

Merges the trusted-but-sparse LiDAR scaffold with the dense-but-noisy B4 image
cloud into a single clean, oriented, colored point cloud ready for meshing
(Phase D). The guiding rule from the plan:

    LiDAR wins ties on metric position; images supply the surfaces LiDAR missed
    (floor / ceiling / between-ring wall gaps).

Steps:
  1. LiDAR scaffold IN THE B1 FRAME.  The saved surfels.npz is in a mixed
     Step-1/Step-3 frame (~2.6 cm off the image cloud), so we rebuild it: place
     the scans with `trajectory_poses_world` (Step-1 dense - the exact frame B1
     posed the cameras in) and reuse `step4_surfels.fuse_surfels` for oriented
     normals + per-surfel confidence.
  2. Clean the image cloud (radius-outlier) to cut B4's radial smear.
  3. Confidence-weighted voxel hash with HARD per-voxel LiDAR-wins: a voxel that
     contains any LiDAR takes its geometry from LiDAR (colour still from images
     if available); image-only voxels are confidence-weighted image points.
     Geometry and colour use SEPARATE weights so colourless LiDAR never darkens
     the averaged image colour.
  4. Cleanup: drop weak image-only voxels + radius-outlier on the fused cloud
     (LiDAR voxels are always protected).
  5. Orient all normals toward the trajectory centroid (indoor inside-out scan ->
     normals point to the interior), so Poisson gets globally consistent normals.

Run (from repo root, after the LiDAR branch through step3 and the image branch
through step_b4_cloud):
    python src/odometry/step_c_fuse.py

Output (under <session>/output/fusion/):
    fused_cloud.npz   points, colors, normals, confidence, is_lidar
    fused_cloud.ply   oriented + colored, drops into step6_mesh (Phase D)
"""

import os
import time

import numpy as np
import open3d as o3d

import common
import step2_mapping as step2
import step4_surfels as step4


# --- Tunables ---------------------------------------------------------------
VOX = 0.02               # fusion voxel (m): between image 1.5 cm and surfel 4 cm
IMG_PRECLEAN = True      # radius-outlier the image cloud before fusing
PRECLEAN_NB = 8          # min neighbours within PRECLEAN_RADIUS to survive
PRECLEAN_RADIUS = 0.05
MIN_IMG_WEIGHT = 1.0     # drop image-only voxels below this summed confidence
POST_RADIUS_NB = 6       # fused-cloud radius-outlier (image voxels only)
POST_RADIUS_R = 0.06

_OFF = 1 << 20


def _pack(idx_xyz: np.ndarray) -> np.ndarray:
    """(N,3) int voxel indices -> (N,) unique int64 key."""
    k = idx_xyz.astype(np.int64) + _OFF
    return (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


def _accumulate(keys, weights, *vecs):
    """Per-voxel sums: returns (uniq_keys, summed_weight, [weighted vec sums])."""
    uniq, inv = np.unique(keys, return_inverse=True)
    M = len(uniq)
    W = np.zeros(M)
    np.add.at(W, inv, weights)
    outs = []
    for v in vecs:
        acc = np.zeros((M, v.shape[1]))
        np.add.at(acc, inv, v * weights[:, None])
        outs.append(acc)
    return uniq, W, outs


# --- Inputs -----------------------------------------------------------------

def build_lidar_scaffold(ds):
    """Surfels in the Step-1 dense frame (matches the B1/B4 image cloud)."""
    poses1 = np.load(os.path.join(ds.output_dir, "trajectory_poses_world.npy"))
    kf = np.array(step2.select_keyframes(poses1))
    pts = step4.gather_world_points(ds, poses1, kf)
    surf = step4.fuse_surfels(pts)
    centroid = poses1[:, :3, 3].mean(0)
    return surf, centroid


def load_clean_image(root):
    pts, col, nrm, conf = common.load_image_cloud(root)
    n0 = len(pts)
    if IMG_PRECLEAN:
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts)
        _, ind = pc.remove_radius_outlier(nb_points=PRECLEAN_NB,
                                          radius=PRECLEAN_RADIUS)
        ind = np.asarray(ind)
        pts, col, nrm, conf = pts[ind], col[ind], nrm[ind], conf[ind]
    return pts, col, nrm, conf, n0


# --- Driver -----------------------------------------------------------------

def run(data_root: str, vox=VOX):
    ds = common.SessionDataset(data_root)
    out_dir = os.path.join(os.path.abspath(data_root), "output", "fusion")
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    surf, centroid = build_lidar_scaffold(ds)
    lpos, lnrm, lconf = surf.positions, surf.normals, surf.confidence.astype(float)
    ipos, icol, inrm, iconf, n_img0 = load_clean_image(data_root)
    print(f"LiDAR surfels: {len(lpos)}  |  image pts: {len(ipos)} "
          f"(pre-clean dropped {n_img0 - len(ipos)})")

    # Per-source voxel accumulation.
    ul, lW, (lWP, lWN) = _accumulate(
        _pack(np.floor(lpos / vox).astype(np.int64)), lconf, lpos, lnrm)
    ui, iW, (iWP, iWC, iWN) = _accumulate(
        _pack(np.floor(ipos / vox).astype(np.int64)), iconf, ipos, icol, inrm)

    # Union of occupied voxels, scatter both sources onto it.
    all_keys = np.union1d(ul, ui)
    M = len(all_keys)
    LW = np.zeros(M); LWP = np.zeros((M, 3)); LWN = np.zeros((M, 3))
    has_l = np.zeros(M, bool)
    pl = np.searchsorted(all_keys, ul)
    LW[pl] = lW; LWP[pl] = lWP; LWN[pl] = lWN; has_l[pl] = True
    IW = np.zeros(M); IWP = np.zeros((M, 3)); IWC = np.zeros((M, 3)); IWN = np.zeros((M, 3))
    has_i = np.zeros(M, bool)
    pi = np.searchsorted(all_keys, ui)
    IW[pi] = iW; IWP[pi] = iWP; IWC[pi] = iWC; IWN[pi] = iWN; has_i[pi] = True

    # Hard LiDAR-wins for geometry; colour always from images.
    is_lidar = has_l
    eL = np.maximum(LW, 1e-9)[:, None]
    eI = np.maximum(IW, 1e-9)[:, None]
    points = np.where(is_lidar[:, None], LWP / eL, IWP / eI)
    normals = np.where(is_lidar[:, None], LWN, IWN)
    colors = np.where(has_i[:, None], IWC / eI, 0.5)          # grey if no image
    confidence = np.where(is_lidar, LW, IW)

    # Cleanup: weak image-only voxels, then radius-outlier (protect LiDAR).
    keep = is_lidar | (IW >= MIN_IMG_WEIGHT)
    points, normals, colors = points[keep], normals[keep], colors[keep]
    confidence, is_lidar, has_i = confidence[keep], is_lidar[keep], has_i[keep]

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    _, ind = pc.remove_radius_outlier(nb_points=POST_RADIUS_NB, radius=POST_RADIUS_R)
    keep2 = np.zeros(len(points), bool)
    keep2[np.asarray(ind)] = True
    keep2 |= is_lidar                                         # never drop LiDAR
    points, normals, colors = points[keep2], normals[keep2], colors[keep2]
    confidence, is_lidar = confidence[keep2], is_lidar[keep2]

    # Globally consistent normals: point toward the interior (sensor side).
    normals = _normalize(normals)
    flip = np.sum(normals * (centroid - points), axis=1) < 0
    normals[flip] *= -1

    points = points.astype(np.float32)
    colors = np.clip(colors, 0, 1).astype(np.float32)
    normals = normals.astype(np.float32)
    confidence = confidence.astype(np.float32)

    npz_path = os.path.join(out_dir, "fused_cloud.npz")
    np.savez_compressed(npz_path, points=points, colors=colors, normals=normals,
                        confidence=confidence, is_lidar=is_lidar,
                        voxel=np.float32(vox))
    fused = o3d.geometry.PointCloud()
    fused.points = o3d.utility.Vector3dVector(points)
    fused.colors = o3d.utility.Vector3dVector(colors)
    fused.normals = o3d.utility.Vector3dVector(normals)
    ply_path = os.path.join(out_dir, "fused_cloud.ply")
    o3d.io.write_point_cloud(ply_path, fused)

    n_lidar = int(is_lidar.sum())
    zmin = lpos[:, 2].min()
    floor = int((points[:, 2] < zmin).sum())
    print(f"\nPhase C: fused {len(points)} voxels @ {vox*100:.1f}cm in "
          f"{time.time()-t0:.1f}s")
    print(f"    {n_lidar} LiDAR-backed + {len(points)-n_lidar} image-only voxels")
    print(f"    {floor} floor voxels below LiDAR z<{zmin:.2f} (image gap-fill)")
    print(f"    bbox min {np.round(points.min(0),2)} max {np.round(points.max(0),2)}")
    print(f"Saved -> {npz_path}\n        {ply_path}")
    return points, colors, normals, confidence, is_lidar


def main():
    data_root = "./captures/session_05_20260624"
    run(data_root)


if __name__ == "__main__":
    main()
