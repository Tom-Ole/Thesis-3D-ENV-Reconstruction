"""Step 3 -- Multi-Sensor Fusion Mapping (dense fusion, TSDF, confidence).

Turns Step 2's globally-consistent keyframe trajectory into an actual fused
map:

- **3.1 Dense Point Cloud Fusion** -- every dense LiDAR scan transformed into
  one global frame, merged, and cleaned with statistical outlier removal.
- **3.2 TSDF Volume Integration** -- a watertight coarse mesh via real TSDF
  volume integration.
- **3.3 Surface Normals and Confidence Estimation** -- a per-point confidence
  field from viewing angle, sensor distance and multi-view consistency.

**Step 2 only optimizes keyframe poses**, but todo2's Step 3.1 asks to
transform *all* scans -- so the correction each keyframe received from Step 2
is SE(3)-interpolated across the dense (non-keyframe) frames between
bracketing keyframes and applied to each one (see ``propagate_corrections``).

**TSDF from a 360-degree LiDAR scan (research finding):** Open3D's TSDF
integration -- both the legacy ``pipelines.integration.ScalableTSDFVolume``
and the newer tensor ``t.geometry.VoxelBlockGrid`` -- only accepts pinhole
camera depth images (confirmed via direct API inspection: `integrate(image,
intrinsic, extrinsic)`, no raw-point-cloud path in either). There is also no
Windows wheel for VDBFusion (the standard point-cloud-native LiDAR TSDF tool)
on PyPI. Per an explicit decision to implement todo2's Step 3.2 literally
rather than substitute Poisson reconstruction, this synthesizes ``NUM_TILES``
virtual pinhole depth images tiled around each 360-degree scan (rasterized
from the real points with a vectorized z-buffer), and integrates each one
through Open3D's real TSDF pipeline. The virtual cameras' vertical FOV is
derived empirically from the data itself (``estimate_vertical_fov_deg``)
rather than assumed from a specific sensor model -- on this dataset it comes
out to ~±15 degrees, which does match a Velodyne VLP-16, but nothing here
hardcodes that. The tile-rotation and TSDF-extrinsic sign conventions were
validated on a single frame before being trusted across a whole session (see
the project's Step 3 plan): mesh vertices from a one-scan integration landed
a median 1.7 cm from real measured points, well inside the truncation band.

Open3D's `integrate()` has no per-pixel weight argument, so todo2's
"weighted integration based on viewing angle + distance" is approximated the
only way the API actually allows: points at a grazing incidence angle or
beyond a reliable range are dropped from the synthetic depth image before
integration (a hard-gated approximation of continuous weighting, not claimed
to be identical to it).

Produces, under ``captures/<session>/output/step3/``:
- ``fused_cloud.ply`` / ``fused_cloud_provenance.npz`` -- 3.1's clean global cloud
- ``tsdf_mesh.ply`` -- 3.2's watertight coarse mesh
- ``confidence.npz`` -- 3.3's per-point normals + confidence field
- ``debug/*.png`` -- outlier-removal comparison, mesh preview, confidence render

Runnable standalone: edit the variables at the bottom of this file and run
either ``python -m src.reconstruction.fusion`` or
``python src/reconstruction/fusion.py`` directly from the project root.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):
    # Allow `python src/reconstruction/fusion.py` (no -m) by putting the
    # project root on sys.path so the `src.*` absolute imports below resolve.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import open3d as o3d

from src.reconstruction.common import (
    LidarFrame,
    coordinate_frame,
    crop_to_trajectory,
    draw,
    ensure_dir,
    fuse_with_provenance,
    interpolate_pose,
    load_lidar_frames,
    points_to_point_cloud,
)
from src.reconstruction.pose_graph import load_keyframes

# --- 3.1 Dense Point Cloud Fusion -------------------------------------------
FUSION_VOXEL_SIZE = 0.03
OUTLIER_NB_NEIGHBORS = 20
OUTLIER_STD_RATIO = 2.0

# --- 3.2 TSDF Volume Integration ---------------------------------------------
TSDF_VOXEL_SIZE = 0.03
TSDF_TRUNC_VOXELS = 3
NUM_TILES = 8
TILE_FOV_DEG = 360.0 / NUM_TILES
TILE_FOV_MARGIN_DEG = 2.0  # small overlap between tiles, avoids seam gaps
HORIZONTAL_ANGULAR_RES_DEG = 0.5
VERTICAL_ANGULAR_RES_DEG = 0.5
MAX_RANGE_M = 10.0
MAX_INCIDENCE_DEG = 75.0  # grazing-angle gate: the "viewing angle" half of todo2's weighting
NORMAL_RADIUS = 0.15

# --- 3.3 Confidence Estimation ------------------------------------------------
CONFIDENCE_DISTANCE_D0 = 5.0
CONFIDENCE_MULTIVIEW_TARGET = 4


# ---------------------------------------------------------------------------
# Shared foundation: propagate Step 2's keyframe correction onto every dense frame
# ---------------------------------------------------------------------------


def propagate_corrections(
    poses_step1_dense: np.ndarray, keyframe_dense_positions: list[int], poses_step2: np.ndarray
) -> np.ndarray:
    """SE(3)-interpolate Step 2's per-keyframe correction across every dense frame.

    Step 2 only optimizes keyframe poses; this applies the correction each
    keyframe received (``step2_pose @ inv(step1_pose)``) to every frame in
    between by interpolating that correction between bracketing keyframes and
    applying it to the frame's own Step 1 pose -- preserving each frame's
    fine local estimate while pulling it into the globally-consistent frame.
    """
    poses_step3 = np.empty_like(poses_step1_dense)
    corrections = [
        poses_step2[k] @ np.linalg.inv(poses_step1_dense[pos])
        for k, pos in enumerate(keyframe_dense_positions)
    ]

    for k in range(len(keyframe_dense_positions) - 1):
        a, b = keyframe_dense_positions[k], keyframe_dense_positions[k + 1]
        for i in range(a, b + 1):
            fraction = 0.0 if b == a else (i - a) / (b - a)
            correction = interpolate_pose(corrections[k], corrections[k + 1], fraction)
            poses_step3[i] = correction @ poses_step1_dense[i]

    return poses_step3


def load_step3_poses(session_dir: Path) -> tuple[list[LidarFrame], np.ndarray]:
    """Dense Step 1 trajectory pulled into global consistency by Step 2's correction."""
    step1_dir = session_dir / "output" / "step1"
    step2_dir = session_dir / "output" / "step2"
    if not (step1_dir / "odometry_refined_poses.npy").exists():
        raise FileNotFoundError(f"No Step 1 output at {step1_dir}; run Step 1 first.")
    if not (step2_dir / "pose_graph_poses.npy").exists():
        raise FileNotFoundError(f"No Step 2 output at {step2_dir}; run Step 2 first.")

    frames = load_lidar_frames(session_dir)
    poses_step1_dense = np.load(step1_dir / "odometry_refined_poses.npy")
    poses_step2 = np.load(step2_dir / "pose_graph_poses.npy")
    keyframes = load_keyframes(step1_dir)

    index_to_position = {frame.index: position for position, frame in enumerate(frames)}
    keyframe_positions = [index_to_position[kf.scan_index] for kf in keyframes]

    poses_step3 = propagate_corrections(poses_step1_dense, keyframe_positions, poses_step2)
    return frames, poses_step3


# ---------------------------------------------------------------------------
# 3.1 -- Dense Point Cloud Fusion
# ---------------------------------------------------------------------------


def run_dense_fusion(
    frames: list[LidarFrame], poses_step3: np.ndarray, output_dir: Path
) -> tuple[o3d.geometry.PointCloud, np.ndarray, np.ndarray, np.ndarray]:
    print(f"[Step 3.1] Fusing {len(frames)} scans (voxel={FUSION_VOXEL_SIZE} m) ...")
    points, sensor_origins, source_frame_idx = fuse_with_provenance(
        frames, poses_step3, voxel_size=FUSION_VOXEL_SIZE
    )
    print(f"[Step 3.1]   {points.shape[0]} points before outlier removal")

    cloud = points_to_point_cloud(points)
    clean_cloud, keep_idx = cloud.remove_statistical_outlier(OUTLIER_NB_NEIGHBORS, OUTLIER_STD_RATIO)
    keep_mask = np.zeros(points.shape[0], dtype=bool)
    keep_mask[np.asarray(keep_idx)] = True
    outlier_points = points[~keep_mask]
    print(f"[Step 3.1]   removed {outlier_points.shape[0]} outlier points ({100 * outlier_points.shape[0] / points.shape[0]:.1f}%)")

    sensor_origins = sensor_origins[keep_mask]
    source_frame_idx = source_frame_idx[keep_mask]

    o3d.io.write_point_cloud(str(output_dir / "fused_cloud.ply"), clean_cloud)
    np.savez(
        output_dir / "fused_cloud_provenance.npz",
        sensor_origins=sensor_origins,
        source_frame_idx=source_frame_idx,
    )
    return clean_cloud, sensor_origins, source_frame_idx, outlier_points


# ---------------------------------------------------------------------------
# 3.2 -- TSDF Volume Integration via tiled virtual cameras
# ---------------------------------------------------------------------------


def estimate_vertical_fov_deg(frames: list[LidarFrame], sample_size: int = 5) -> tuple[float, float]:
    """Robust (1st-99th percentile) elevation-angle range, derived from real scans.

    Deliberately not hardcoded from a specific Velodyne model's spec sheet --
    no capture metadata confirms the exact sensor model, so this measures the
    actual data instead.
    """
    step = max(1, len(frames) // sample_size)
    elevations = []
    for frame in frames[::step][:sample_size]:
        points = np.asarray(frame.read_cloud_sensor().points)
        ranges = np.linalg.norm(points, axis=1)
        valid = ranges > 1e-3
        elevations.append(np.degrees(np.arcsin(np.clip(points[valid, 2] / ranges[valid], -1.0, 1.0))))
    all_elevations = np.concatenate(elevations)
    return float(np.percentile(all_elevations, 1)), float(np.percentile(all_elevations, 99))


def _tile_rotation_sensor_from_camera(yaw_deg: float) -> np.ndarray:
    """3x3 rotation: camera-local (OpenCV: +X right, +Y down, +Z forward) -> sensor-local.

    Validated empirically (see module docstring): mesh vertices from a
    single-frame TSDF integration using this convention land a median 1.7 cm
    from real measured points.
    """
    yaw = np.radians(yaw_deg)
    forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    world_up = np.array([0.0, 0.0, 1.0])
    x_cam = np.cross(forward, world_up)
    x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(forward, x_cam)
    return np.column_stack([x_cam, y_cam, forward])


def _rasterize_tile_depth(
    points_sensor: np.ndarray,
    rotation_sensor_from_camera: np.ndarray,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    max_range: float,
) -> np.ndarray:
    """Vectorized nearest-wins z-buffer rasterization into one virtual pinhole view."""
    points_camera = points_sensor @ rotation_sensor_from_camera
    z = points_camera[:, 2]
    ranges = np.linalg.norm(points_sensor, axis=1)
    valid = (z > 1e-3) & (ranges < max_range)
    points_camera, z = points_camera[valid], z[valid]

    u = fx * points_camera[:, 0] / z + cx
    v = fy * points_camera[:, 1] / z + cy
    ui, vi = np.round(u).astype(np.int64), np.round(v).astype(np.int64)
    in_bounds = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    ui, vi, z = ui[in_bounds], vi[in_bounds], z[in_bounds]

    depth = np.zeros((height, width), dtype=np.float32)
    order = np.argsort(-z)  # farthest first -> nearest is written last and wins
    depth[vi[order], ui[order]] = z[order]
    return depth


def _incidence_angle_deg(points: np.ndarray, cloud_with_normals: o3d.geometry.PointCloud) -> np.ndarray:
    """Angle between each point's normal and the ray back to the sensor origin."""
    normals = np.asarray(cloud_with_normals.normals)
    ranges = np.linalg.norm(points, axis=1, keepdims=True)
    view_dirs = -points / np.clip(ranges, 1e-6, None)
    cos_angle = np.clip(np.abs(np.sum(normals * view_dirs, axis=1)), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def _gate_by_viewing_angle(points_sensor: np.ndarray) -> np.ndarray:
    """Drop grazing-incidence points -- the "viewing angle" half of todo2's weighting."""
    cloud = points_to_point_cloud(points_sensor)
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS, max_nn=30))
    cloud.orient_normals_towards_camera_location(np.zeros(3))
    incidence = _incidence_angle_deg(points_sensor, cloud)
    return points_sensor[incidence < MAX_INCIDENCE_DEG]


def _keep_largest_component(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    largest_cluster = cluster_n_triangles.argmax()
    mesh.remove_triangles_by_mask(triangle_clusters != largest_cluster)
    mesh.remove_unreferenced_vertices()
    return mesh


def run_tsdf_fusion(frames: list[LidarFrame], poses_step3: np.ndarray, output_dir: Path) -> o3d.geometry.TriangleMesh:
    elev_min, elev_max = estimate_vertical_fov_deg(frames)
    fov_v = (elev_max - elev_min) + 2 * TILE_FOV_MARGIN_DEG
    fov_h = TILE_FOV_DEG + 2 * TILE_FOV_MARGIN_DEG
    width = max(8, int(round(fov_h / HORIZONTAL_ANGULAR_RES_DEG)))
    height = max(8, int(round(fov_v / VERTICAL_ANGULAR_RES_DEG)))
    fx = (width / 2.0) / np.tan(np.radians(fov_h / 2.0))
    fy = (height / 2.0) / np.tan(np.radians(fov_v / 2.0))
    cx, cy = width / 2.0, height / 2.0
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    print(
        f"[Step 3.2] Empirical vertical FOV ~[{elev_min:.1f}, {elev_max:.1f}] deg; "
        f"{NUM_TILES} tiles x {width}x{height} px (fov_h={fov_h:.1f} deg, fov_v={fov_v:.1f} deg)"
    )

    tile_rotations = [_tile_rotation_sensor_from_camera(k * TILE_FOV_DEG) for k in range(NUM_TILES)]
    tile_poses_in_sensor = []
    for rotation in tile_rotations:
        pose = np.eye(4)
        pose[:3, :3] = rotation
        tile_poses_in_sensor.append(pose)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=TSDF_VOXEL_SIZE,
        sdf_trunc=TSDF_TRUNC_VOXELS * TSDF_VOXEL_SIZE,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
    )
    dummy_color = o3d.geometry.Image(np.zeros((height, width, 3), dtype=np.uint8))

    total_pixels, hit_pixels = 0, 0
    for i, frame in enumerate(frames):
        points_sensor = np.asarray(frame.read_cloud_sensor().points)
        if points_sensor.shape[0] == 0:
            continue
        gated_points = _gate_by_viewing_angle(points_sensor)

        for tile_pose, rotation in zip(tile_poses_in_sensor, tile_rotations):
            depth = _rasterize_tile_depth(gated_points, rotation, width, height, fx, fy, cx, cy, MAX_RANGE_M)
            total_pixels += depth.size
            hit_pixels += int(np.count_nonzero(depth))

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                dummy_color,
                o3d.geometry.Image(depth),
                depth_scale=1.0,
                depth_trunc=MAX_RANGE_M,
                convert_rgb_to_intensity=False,
            )
            camera_pose_world = poses_step3[i] @ tile_pose
            extrinsic = np.linalg.inv(camera_pose_world)
            volume.integrate(rgbd, intrinsic, extrinsic)

        if (i + 1) % 50 == 0 or i == len(frames) - 1:
            print(f"[Step 3.2]   integrated {i + 1}/{len(frames)} scans")

    print(f"[Step 3.2] {hit_pixels}/{total_pixels} synthetic depth pixels had a real return ({100 * hit_pixels / total_pixels:.1f}%)")

    print("[Step 3.2] Extracting mesh ...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    mesh = _keep_largest_component(mesh)
    print(f"[Step 3.2] Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

    o3d.io.write_triangle_mesh(str(output_dir / "tsdf_mesh.ply"), mesh)
    return mesh


# ---------------------------------------------------------------------------
# 3.3 -- Surface Normals and Confidence Estimation
# ---------------------------------------------------------------------------


def run_confidence_estimation(
    cloud: o3d.geometry.PointCloud,
    sensor_origins: np.ndarray,
    source_frame_idx: np.ndarray,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    print("[Step 3.3] Estimating normals + confidence ...")
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS, max_nn=30))
    points = np.asarray(cloud.points)
    normals = np.asarray(cloud.normals)

    # Orient each normal using ITS OWN originating sensor position -- unlike a
    # typical SfM cloud, provenance tracking (3.1) means the real originating
    # viewpoint per point is actually known here, not just guessed.
    to_sensor = sensor_origins - points
    flip = np.sum(normals * to_sensor, axis=1) < 0
    normals[flip] *= -1
    cloud.normals = o3d.utility.Vector3dVector(normals)

    ranges = np.linalg.norm(to_sensor, axis=1)
    view_dirs = to_sensor / np.clip(ranges, 1e-6, None)[:, None]
    cos_incidence = np.clip(np.sum(normals * view_dirs, axis=1), 0.0, 1.0)
    angle_confidence = cos_incidence

    distance_confidence = 1.0 / (1.0 + (ranges / CONFIDENCE_DISTANCE_D0) ** 2)

    multiview_confidence = _multiview_confidence(points, source_frame_idx)

    confidence = angle_confidence * distance_confidence * multiview_confidence
    print(
        f"[Step 3.3]   confidence stats: mean={confidence.mean():.3f} "
        f"min={confidence.min():.3f} max={confidence.max():.3f}"
    )

    np.savez(output_dir / "confidence.npz", points=points, normals=normals, confidence=confidence)
    return normals, confidence


def _multiview_confidence(points: np.ndarray, source_frame_idx: np.ndarray) -> np.ndarray:
    """Fraction of a target multi-view count reached, per occupied voxel.

    Voxel-hashes the fused cloud at the TSDF voxel size and counts *distinct
    originating frames* per voxel -- points confirmed by many different scans
    score higher than points seen by only one or two.
    """
    voxel_keys = np.floor(points / TSDF_VOXEL_SIZE).astype(np.int64)
    _, inverse = np.unique(voxel_keys, axis=0, return_inverse=True)

    # Collapsing to unique (voxel, frame) pairs turns "count distinct frames
    # per voxel" into a single bincount -- no per-voxel Python loop needed.
    voxel_frame_pairs = np.unique(np.stack([inverse, source_frame_idx], axis=1), axis=0)
    distinct_counts = np.bincount(voxel_frame_pairs[:, 0], minlength=inverse.max() + 1)

    per_point_count = distinct_counts[inverse]
    return np.clip(per_point_count / CONFIDENCE_MULTIVIEW_TARGET, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------


def plot_outlier_removal(
    outlier_points: np.ndarray,
    clean_cloud: o3d.geometry.PointCloud,
    poses_step3: np.ndarray,
    debug_dir: Path,
    show: bool,
) -> None:
    # A handful of noisy far-range LiDAR returns can sit tens of metres from
    # the real scene (exactly the kind of thing outlier removal is for); left
    # in, they'd dominate the camera auto-framing the same way they did for
    # Step 1/2's map view, so crop to the trajectory's real footprint first.
    outliers = crop_to_trajectory(points_to_point_cloud(outlier_points), poses_step3)
    outliers.paint_uniform_color([0.9, 0.1, 0.1])
    kept = crop_to_trajectory(clean_cloud, poses_step3)
    kept.paint_uniform_color([0.6, 0.6, 0.6])
    draw(
        [kept, outliers, coordinate_frame(size=0.5)],
        title="Step 3.1 - statistical outlier removal (red=removed)",
        screenshot_path=debug_dir / "outlier_removal.png",
        show=show,
        point_size=1.5,
        top_down=True,
    )


def plot_mesh_preview(mesh: o3d.geometry.TriangleMesh, debug_dir: Path, show: bool) -> None:
    mesh.paint_uniform_color([0.75, 0.75, 0.8])
    draw(
        [mesh, coordinate_frame(size=0.5)],
        title="Step 3.2 - TSDF mesh (watertight coarse geometry)",
        screenshot_path=debug_dir / "tsdf_mesh.png",
        show=show,
    )


def plot_confidence(points: np.ndarray, confidence: np.ndarray, poses_step3: np.ndarray, debug_dir: Path, show: bool) -> None:
    """Red (low confidence) -> green (high confidence) heatmap.

    Confidence is a *product* of three [0, 1] factors (todo2's viewing
    angle/distance/multi-view criteria), so even a "decent on every count"
    point (each factor ~0.7) lands around 0.35 -- correct, but it would
    visually flatten this plot toward red. The saved ``confidence.npz``
    values stay the honest product; only the on-screen color uses a sqrt
    (equivalent to a gamma boost) to spread that range back out for legibility.
    """
    cloud = points_to_point_cloud(points)
    cloud.colors = o3d.utility.Vector3dVector(
        np.stack([1.0 - np.sqrt(confidence), np.sqrt(confidence), np.zeros_like(confidence)], axis=1)
    )
    cloud = crop_to_trajectory(cloud, poses_step3)
    draw(
        [cloud, coordinate_frame(size=0.5)],
        title="Step 3.3 - confidence (red=low, green=high)",
        screenshot_path=debug_dir / "confidence.png",
        show=show,
        point_size=1.5,
        top_down=True,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_step3(session_dir: Path, output_dir: Optional[Path] = None, show: bool = True) -> Path:
    session_dir = Path(session_dir)
    output_dir = Path(output_dir) if output_dir else session_dir / "output" / "step3"
    debug_dir = ensure_dir(output_dir / "debug")

    frames, poses_step3 = load_step3_poses(session_dir)
    print(f"[Step 3] Loaded {len(frames)} scans with globally-consistent poses")

    clean_cloud, sensor_origins, source_frame_idx, outlier_points = run_dense_fusion(
        frames, poses_step3, output_dir
    )

    mesh = run_tsdf_fusion(frames, poses_step3, output_dir)

    normals, confidence = run_confidence_estimation(clean_cloud, sensor_origins, source_frame_idx, output_dir)

    print("[Step 3] Rendering visualizations ...")
    plot_outlier_removal(outlier_points, clean_cloud, poses_step3, debug_dir, show)
    plot_mesh_preview(mesh, debug_dir, show)
    plot_confidence(np.asarray(clean_cloud.points), confidence, poses_step3, debug_dir, show)

    print(f"[Step 3] Done. Output in {output_dir}")
    return output_dir


def _default_session_dir() -> Optional[Path]:
    captures_root = Path("captures")
    if not captures_root.exists():
        return None
    sessions = sorted(p for p in captures_root.iterdir() if p.is_dir() and p.name.startswith("session_"))
    return sessions[-1] if sessions else None


if __name__ == "__main__":
    # Edit these to point at a different session / tune Step 3, then re-run this file.
    SESSION_DIR = _default_session_dir()
    OUTPUT_DIR = None  # defaults to captures/<session>/output/step3
    SHOW = True

    if SESSION_DIR is None:
        raise SystemExit("No captures/session_* directory found; set SESSION_DIR explicitly.")

    run_step3(SESSION_DIR, OUTPUT_DIR, show=SHOW)
