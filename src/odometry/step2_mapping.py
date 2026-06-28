"""
STEP 2 - Keyframe Selection and Global Point Cloud Map

Output: <session>/output/global_map.ply  +  the selected keyframe indices.
"""

import numpy as np
import open3d as o3d

import common


# --- Tunables ---------------------------------------------------------------
KEYFRAME_DIST_M = 0.25            # min translation since last keyframe
KEYFRAME_ROT_DEG = 10.0           # OR min rotation since last keyframe

MAP_VOXEL = 0.03                  # fusion voxel (3 cm indoor)

# Statistical outlier removal: drop points whose mean neighbour distance is
# > std_ratio std-devs above the average.
SOR_NB_NEIGHBORS = 20
SOR_STD_RATIO = 2.0

# Radius outlier removal: drop points with too few neighbours in a ball.
ROR_NB_POINTS = 12
ROR_RADIUS = MAP_VOXEL * 3.0


def load_poses(dataset, prefix="trajectory"):
    path = f"{dataset.output_dir}/{prefix}_poses_world.npy"
    return np.load(path)


def _rel_motion(a, b):
    """Translation (m) and rotation (deg) from pose a to pose b."""
    dt = np.linalg.norm(b[:3, 3] - a[:3, 3])
    dR = a[:3, :3].T @ b[:3, :3]
    dr = np.degrees(np.arccos(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)))
    return dt, dr


def select_keyframes(poses_world,
                     dist_m=KEYFRAME_DIST_M, rot_deg=KEYFRAME_ROT_DEG):
    """Distance/rotation-gated keyframe selection. Returns kept indices."""
    keep = [0]
    last = poses_world[0]
    for i in range(1, len(poses_world)):
        dt, dr = _rel_motion(last, poses_world[i])
        if dt >= dist_m or dr >= rot_deg:
            keep.append(i)
            last = poses_world[i]
    if keep[-1] != len(poses_world) - 1:
        keep.append(len(poses_world) - 1)     # always keep the last scan
    return keep


def build_global_map(dataset, poses_world, keyframes, voxel=MAP_VOXEL):
    """Fuse keyframes into one cloud, then voxel-downsample + denoise."""
    fused = o3d.geometry.PointCloud()
    for i in keyframes:
        cloud = dataset[i].read_cloud_sensor().transform(poses_world[i].copy())
        fused += cloud.voxel_down_sample(voxel)

    n_raw = len(fused.points)
    fused = fused.voxel_down_sample(voxel)
    n_voxel = len(fused.points)

    fused, _ = fused.remove_statistical_outlier(
        nb_neighbors=SOR_NB_NEIGHBORS, std_ratio=SOR_STD_RATIO)
    n_sor = len(fused.points)

    fused, _ = fused.remove_radius_outlier(
        nb_points=ROR_NB_POINTS, radius=ROR_RADIUS)
    n_ror = len(fused.points)

    print(f"Map points: fused={n_raw} -> voxel={n_voxel} "
          f"-> stat-filter={n_sor} -> radius-filter={n_ror}")
    return fused


def save_map(dataset, global_map, name="global_map.ply"):
    out = dataset.ensure_output_dir()
    path = f"{out}/{name}"
    o3d.io.write_point_cloud(path, global_map)
    print(f"Saved map -> {path}")
    return path


def visualize(poses_world, keyframes, global_map):
    geoms = [
        global_map,
        common.trajectory_lineset(poses_world, color=(1.0, 0.5, 0.0)),
    ]
    for T in poses_world[keyframes]:
        s = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        s.paint_uniform_color([0.1, 0.9, 0.1])
        geoms.append(s.translate(T[:3, 3]))
    common.draw(geoms, title="Step 2 - global map (green=keyframes)")


def main():
    data_root = "./captures/session_05_20260624"
    dataset = common.SessionDataset(data_root)
    poses_world = load_poses(dataset)

    keyframes = select_keyframes(poses_world)
    print(f"Keyframes: {len(keyframes)}/{len(poses_world)} "
          f"(gate: {KEYFRAME_DIST_M} m / {KEYFRAME_ROT_DEG} deg)")

    global_map = build_global_map(dataset, poses_world, keyframes)
    save_map(dataset, global_map)
    visualize(poses_world, keyframes, global_map)


if __name__ == "__main__":
    main()
