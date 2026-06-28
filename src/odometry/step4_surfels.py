"""
STEP 4 - Surfel-Based Surface Reconstruction   [SCAFFOLD]
========================================================

Fuses the per-scan LiDAR points (placed by the Step 1-3 trajectory) into a
surfel map: a set of small oriented disks that denoise and stabilize the raw
point cloud and act as the geometric backbone for Steps 5-6.

A surfel stores:
    position   (3,)   confidence-weighted mean of contributing points
    normal     (3,)   local surface orientation (PCA)
    radius     ()     disk size ~ local point spacing
    confidence ()     how many observations support it (lifecycle weight)
    color      (3,)   from cameras  -> STEP 5 (placeholder for now)

This scaffold implements a BATCH voxel-surfel fusion (one pass over all
keyframes): group world points by voxel, average position, estimate normals,
count observations as confidence, prune unstable surfels. That captures the
TODO's properties (position averaging, confidence, lifecycle pruning) without a
full incremental association loop.

TODO (later passes):
    - incremental per-frame fusion with point<->surfel association + running
      average + temporal decay/removal of unstable surfels (true ElasticFusion
      style), instead of one batch pass
    - color_from_cameras() -> STEP 5 (project the 5 fisheye images per surfel)
    - feed surfels as the geometry prior into STEP 6 (3D Gaussian Splatting)

Run Step 1 (and ideally Step 3) first.
"""

from dataclasses import dataclass

import numpy as np
import open3d as o3d
import matplotlib

import common
import step2_mapping as step2


# --- Tunables ---------------------------------------------------------------
SURFEL_VOXEL = 0.04          # surfel spacing (m)
PER_FRAME_VOXEL = 0.02       # pre-downsample each scan before fusing
MIN_CONFIDENCE = 3           # prune surfels with fewer contributing points
NORMAL_RADIUS = SURFEL_VOXEL * 3


@dataclass
class SurfelMap:
    positions: np.ndarray    # (M,3)
    normals: np.ndarray      # (M,3)
    radii: np.ndarray        # (M,)
    confidence: np.ndarray   # (M,)
    colors: np.ndarray       # (M,3) -- placeholder until Step 5

    def __len__(self):
        return len(self.positions)

    def to_point_cloud(self):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(self.positions)
        pc.normals = o3d.utility.Vector3dVector(self.normals)
        pc.colors = o3d.utility.Vector3dVector(self.colors)
        return pc


def load_world_poses(dataset):
    """
    Per-frame world poses, preferring Step 3's optimized keyframe poses (spread
    back over the frames they represent) and falling back to Step 1.
    """
    poses = step2.load_poses(dataset)            # Step 1, (N,4,4)
    pg = f"{dataset.output_dir}/pose_graph_poses.npy"
    kf = f"{dataset.output_dir}/pose_graph_keyframes.npy"
    import os
    if os.path.isfile(pg) and os.path.isfile(kf):
        opt = np.load(pg)
        keyframes = np.load(kf)
        for k, idx in enumerate(keyframes):
            poses[idx] = opt[k]                   # use optimized at keyframes
        print("Using Step 3 optimized poses at keyframes.")
    else:
        keyframes = np.array(step2.select_keyframes(poses))
        print("Step 3 output not found; using Step 1 poses.")
    return poses, keyframes


def gather_world_points(dataset, poses, keyframes):
    """Stack all keyframe scans (sensor->world), lightly pre-downsampled."""
    chunks = []
    for idx in keyframes:
        c = dataset[int(idx)].read_cloud_sensor().voxel_down_sample(PER_FRAME_VOXEL)
        c.transform(poses[int(idx)].copy())
        chunks.append(np.asarray(c.points))
    return np.concatenate(chunks, axis=0)


def fuse_surfels(points, voxel=SURFEL_VOXEL, min_conf=MIN_CONFIDENCE):
    """Batch voxel-surfel fusion: position averaging + confidence + normals."""
    keys = np.floor(points / voxel).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True,
                               return_counts=True)
    sums = np.zeros((counts.shape[0], 3))
    np.add.at(sums, inv, points)
    centers = sums / counts[:, None]
    confidence = counts.astype(np.float64)

    # Lifecycle prune: drop surfels with too few observations (unstable).
    keep = confidence >= min_conf
    centers, confidence = centers[keep], confidence[keep]

    # Normals via local PCA (Open3D), radius from surfel spacing.
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(centers)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=NORMAL_RADIUS, max_nn=30))
    pc.orient_normals_consistent_tangent_plane(15)
    normals = np.asarray(pc.normals)
    radii = np.full(len(centers), voxel * 0.7)

    colors = _placeholder_colors(confidence)     # STEP 5 replaces this
    return SurfelMap(centers, normals, radii, confidence, colors)


def _placeholder_colors(confidence):
    """Colour by confidence (viridis) until Step 5 supplies camera colour."""
    t = (confidence - confidence.min()) / (np.ptp(confidence) + 1e-9)
    return matplotlib.colormaps["viridis"](t)[:, :3]


def color_from_cameras(surfels, dataset):
    """STEP 5: project the 5 fisheye images per surfel for multi-view colour."""
    raise NotImplementedError("Multi-camera colour fusion is Step 5.")


def save_surfels(dataset, surfels, name="surfels"):
    out = dataset.ensure_output_dir()
    np.savez(f"{out}/{name}.npz",
             positions=surfels.positions, normals=surfels.normals,
             radii=surfels.radii, confidence=surfels.confidence,
             colors=surfels.colors)
    o3d.io.write_point_cloud(f"{out}/{name}.ply", surfels.to_point_cloud())
    print(f"Saved {len(surfels)} surfels -> {out}/{name}.ply (+ .npz)")


def visualize(surfels):
    pc = surfels.to_point_cloud()
    # A few normals drawn as short lines so orientation is visible by default.
    geoms = [pc, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)]
    common.draw(geoms, title="Step 4 - surfels (colour=confidence; normals via 'n')")


def main():
    data_root = "./captures/session_05_20260624"
    dataset = common.SessionDataset(data_root)

    poses, keyframes = load_world_poses(dataset)
    points = gather_world_points(dataset, poses, keyframes)
    print(f"Fusing {len(points)} points from {len(keyframes)} keyframes...")

    surfels = fuse_surfels(points)
    print(f"Surfels: {len(surfels)} "
          f"(confidence mean {surfels.confidence.mean():.1f}, "
          f"max {surfels.confidence.max():.0f})")

    save_surfels(dataset, surfels)
    visualize(surfels)


if __name__ == "__main__":
    main()
