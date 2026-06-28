"""
STEP 1 - LiDAR Odometry
=======================

Frame-to-frame LiDAR odometry: multi-scale (coarse-to-fine) point-to-plane ICP,
seeded with Spot's onboard fused-odometry prior.

Outputs: trajectory estimates T_i  +  aligned point clouds.

Step 0 not done yet: no per-scan deskew and no IMU-interpolated prior. The Spot
fused-odometry prior is the best available substitute; swap in deskew + IMU
interpolation once Step 0 lands.
"""

import copy

import numpy as np
import open3d as o3d

import common


# --- Tunables ---------------------------------------------------------------
VOXEL_SIZES = [0.20, 0.10, 0.05]          # coarse -> fine (metres)
MAX_CORR_DISTANCES = [0.40, 0.20, 0.10]   # ~2x voxel per level
MAX_ITERATIONS = [60, 40, 20]
NORMAL_RADIUS_FACTOR = 2.0

# Quality gate: reject an ICP result and trust the prior instead when the
# registration is weak or it disagrees with Spot's odometry by too much.
MIN_FITNESS = 0.30                        # inlier overlap ratio
MAX_TRANS_DISAGREE = 0.50                 # metres vs prior translation
MAX_ROT_DISAGREE_DEG = 20.0               # degrees vs prior rotation


def _pyramid(pcd):
    """Pre-compute the downsample+normals pyramid once per scan."""
    levels = []
    for voxel in VOXEL_SIZES:
        down = pcd.voxel_down_sample(voxel)
        down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel * NORMAL_RADIUS_FACTOR, max_nn=30))
        levels.append(down)
    return levels


def multiscale_icp(src_pyr, tgt_pyr, init):
    """Coarse-to-fine point-to-plane ICP over pre-built pyramids."""
    transformation = init.copy()
    result = None
    for level, (max_corr, iters) in enumerate(
            zip(MAX_CORR_DISTANCES, MAX_ITERATIONS)):
        result = o3d.pipelines.registration.registration_icp(
            src_pyr[level], tgt_pyr[level], max_corr, transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=iters))
        transformation = result.transformation
    return transformation, result


def _disagreement(a, b):
    """Translation (m) and rotation (deg) difference between two transforms."""
    dt = np.linalg.norm(a[:3, 3] - b[:3, 3])
    dR = a[:3, :3] @ b[:3, :3].T
    cos = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    return dt, np.degrees(np.arccos(cos))


def run_odometry(data_root):
    """
    Estimate the LiDAR trajectory frame-to-frame.

    Returns (dataset, poses_world, spot_poses), all in the odom frame so the
    ICP estimate and Spot's raw odometry are directly comparable.
    """
    dataset = common.SessionDataset(data_root)
    n = len(dataset)
    if n < 2:
        raise RuntimeError(f"Need >=2 frames, found {n}")
    print(f"Loaded {n} frames from {dataset.pointclouds_dir}")

    spot_poses = dataset.spot_trajectory()      # odom_T_sensor per scan

    poses_world = [spot_poses[0].copy()]        # anchor trajectory in odom
    # Clouds are saved in odom; register them in the SENSOR frame.
    tgt_pyr = _pyramid(dataset[0].read_cloud_sensor())

    rejected = 0
    for i in range(1, n):
        src_pyr = _pyramid(dataset[i].read_cloud_sensor())

        # Spot's predicted relative motion (frame i -> frame i-1).
        prior = np.linalg.inv(spot_poses[i - 1]) @ spot_poses[i]

        T_rel, result = multiscale_icp(src_pyr, tgt_pyr, prior)

        dt, dr = _disagreement(T_rel, prior)
        if (result.fitness < MIN_FITNESS or dt > MAX_TRANS_DISAGREE
                or dr > MAX_ROT_DISAGREE_DEG):
            T_rel = prior            # trust the robot over a bad registration
            rejected += 1
            flag = "  REJECTED->prior"
        else:
            flag = ""

        poses_world.append(poses_world[-1] @ T_rel)
        tgt_pyr = src_pyr

        print(f"[{i:04d}/{n-1}] fitness={result.fitness:.3f} "
              f"rmse={result.inlier_rmse:.4f} "
              f"|t|={np.linalg.norm(T_rel[:3, 3]):.3f}m "
              f"(d_prior: {dt:.3f}m/{dr:.1f}deg){flag}")

    print(f"Done. {rejected}/{n-1} pairs fell back to the Spot prior.")
    return dataset, np.stack(poses_world), spot_poses


def save_trajectory(dataset, poses_world, prefix="trajectory"):
    """Save poses as 4x4 matrices (.npy) and in TUM format with real times."""
    from scipy.spatial.transform import Rotation
    out_dir = dataset.ensure_output_dir()
    npy_path = f"{out_dir}/{prefix}_poses_world.npy"
    tum_path = f"{out_dir}/{prefix}_tum.txt"

    np.save(npy_path, poses_world)

    times = dataset.timestamps_sec()
    with open(tum_path, "w") as f:
        for t, T in zip(times, poses_world):
            tx, ty, tz = T[:3, 3]
            qx, qy, qz, qw = Rotation.from_matrix(T[:3, :3]).as_quat()
            f.write(f"{t:.9f} {tx:.6f} {ty:.6f} {tz:.6f} "
                    f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")

    print(f"Saved -> {npy_path}\n        {tum_path}")


def visualize(dataset, poses_world, spot_poses, show_map=True):
    """
    Blue  = Spot's raw fused odometry trajectory.
    Orange = ICP-refined trajectory.
    Plus the fused world-frame cloud (drift shows up as smearing/ghosting).
    """
    geoms = [
        common.trajectory_lineset(spot_poses, color=(0.0, 0.4, 1.0)),
        common.trajectory_lineset(poses_world, color=(1.0, 0.5, 0.0)),
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5),
    ]
    if show_map:
        # every=2 keeps the preview light over 200+ scans.
        geoms.insert(0, common.build_map(dataset, poses_world,
                                         voxel=0.05, every=2))
    common.draw(geoms, title="Step 1 - LiDAR odometry (blue=Spot, orange=ICP)")


def main():
    data_root = "./captures/session_05_20260624"
    show_map = True

    dataset, poses_world, spot_poses = run_odometry(data_root)
    save_trajectory(dataset, poses_world)

    drift = np.linalg.norm(poses_world[-1][:3, 3] - spot_poses[-1][:3, 3])
    print(f"End-pose gap ICP vs Spot odometry: {drift:.3f} m")

    visualize(dataset, poses_world, spot_poses, show_map=show_map)


if __name__ == "__main__":
    main()
