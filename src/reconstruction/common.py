"""Shared LiDAR data loading and visualization helpers for the reconstruction pipeline.

Reads the on-disk capture format written by ``src/capture``
(pointclouds/metadata, see docs/DATASET.md) directly -- there is no separate
preprocessing stage. Frame-tree composition reuses
``bosdyn.client.frame_helpers`` on the saved ``transforms_snapshot`` dict (it
is a serialized ``FrameTreeSnapshot`` proto) instead of re-implementing
quaternion composition.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
from bosdyn.api import geometry_pb2
from bosdyn.client.frame_helpers import get_a_tform_b
from google.protobuf import json_format
from scipy.spatial.transform import Rotation, Slerp

from src.data_models import CAPTURE_TYPE_POINTCLOUD

ODOM_FRAME = "odom"
BODY_FRAME = "body"
LIDAR_FRAME = "sensor"


def transform_between(transforms_snapshot: dict, frame_a: str, frame_b: str) -> np.ndarray:
    """Return the 4x4 rigid transform ``a_T_b`` from a saved frame-tree snapshot."""
    snapshot = geometry_pb2.FrameTreeSnapshot()
    json_format.ParseDict(transforms_snapshot, snapshot)
    se3 = get_a_tform_b(snapshot, frame_a, frame_b)
    if se3 is None:
        raise ValueError(f"No transform path from '{frame_a}' to '{frame_b}' in snapshot")
    return se3.to_matrix()


def rotation_angle_deg(delta: np.ndarray) -> float:
    """Angle of the rotation part of a 4x4 transform, in degrees."""
    cos_angle = np.clip((np.trace(delta[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def interpolate_pose(pose_a: np.ndarray, pose_b: np.ndarray, fraction: float) -> np.ndarray:
    """Blend two 4x4 rigid transforms: linear translation + slerp rotation."""
    fraction = float(np.clip(fraction, 0.0, 1.0))
    translation = (1.0 - fraction) * pose_a[:3, 3] + fraction * pose_b[:3, 3]
    rotations = Rotation.from_matrix(np.stack([pose_a[:3, :3], pose_b[:3, :3]]))
    rotation = Slerp([0.0, 1.0], rotations)(fraction).as_matrix()
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


# ---------------------------------------------------------------------------
# Point cloud I/O
# ---------------------------------------------------------------------------


def read_point_cloud(path: Path) -> o3d.geometry.PointCloud:
    return o3d.io.read_point_cloud(str(path))


def points_to_point_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.ascontiguousarray(points, dtype=np.float64))
    return cloud


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid transform to an (N, 3) point array."""
    homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    return (matrix @ homogeneous.T).T[:, :3]


# ---------------------------------------------------------------------------
# LiDAR frames
# ---------------------------------------------------------------------------


@dataclass
class LidarFrame:
    """One recorded LiDAR scan, with pose(s) resolved from its own snapshot."""

    index: int
    path: Path
    t_nsec: int
    num_points: int
    odom_T_body: np.ndarray
    odom_T_sensor: np.ndarray
    metadata: dict = field(repr=False)

    def read_cloud(self) -> o3d.geometry.PointCloud:
        """Cloud as stored on disk: already in the odom frame."""
        return read_point_cloud(self.path)

    def read_cloud_sensor(self) -> o3d.geometry.PointCloud:
        """Cloud back in the LiDAR sensor frame (centred on the sensor).

        The capture pre-applies Spot's ``odom_T_sensor`` before saving, so the
        raw ``.ply`` is in odom. Un-applying it recovers the true sensor scan,
        which is what frame-to-frame odometry must register.
        """
        cloud = self.read_cloud()
        cloud.transform(np.linalg.inv(self.odom_T_sensor))
        return cloud


def load_lidar_frames(session_dir: Path) -> list[LidarFrame]:
    """Parse every LiDAR scan's metadata JSON into a time-sorted frame list."""
    session_dir = Path(session_dir)
    frames: list[LidarFrame] = []

    for meta_path in sorted((session_dir / "metadata").glob("*.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("capture_type") != CAPTURE_TYPE_POINTCLOUD:
            continue
        snapshot = meta["transforms_snapshot"]
        frames.append(
            LidarFrame(
                index=meta["index"],
                path=session_dir / meta["file"],
                t_nsec=meta["acquisition_time_robot_nsec"],
                num_points=meta["num_points"],
                odom_T_body=transform_between(snapshot, ODOM_FRAME, BODY_FRAME),
                odom_T_sensor=transform_between(snapshot, ODOM_FRAME, LIDAR_FRAME),
                metadata=meta,
            )
        )

    frames.sort(key=lambda f: f.t_nsec)
    return frames


# ---------------------------------------------------------------------------
# Multi-scale ICP (shared by Step 1 odometry and Step 2 loop-closure checks)
# ---------------------------------------------------------------------------

VOXEL_SIZES = [0.20, 0.10, 0.05]
MAX_CORR_DISTANCES = [0.40, 0.20, 0.10]
MAX_ITERATIONS = [60, 40, 20]
NORMAL_RADIUS_FACTOR = 2.0


def pyramid(pcd: o3d.geometry.PointCloud) -> list[o3d.geometry.PointCloud]:
    """Pre-compute the downsample+normals pyramid once per scan."""
    levels = []
    for voxel in VOXEL_SIZES:
        down = pcd.voxel_down_sample(voxel)
        down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * NORMAL_RADIUS_FACTOR, max_nn=30)
        )
        levels.append(down)
    return levels


def multiscale_icp(
    source_pyramid: list[o3d.geometry.PointCloud], target_pyramid: list[o3d.geometry.PointCloud], init: np.ndarray
):
    """Coarse-to-fine point-to-plane ICP over pre-built pyramids."""
    transformation = init.copy()
    result = None
    for level, (max_corr, iterations) in enumerate(zip(MAX_CORR_DISTANCES, MAX_ITERATIONS)):
        result = o3d.pipelines.registration.registration_icp(
            source_pyramid[level],
            target_pyramid[level],
            max_corr,
            transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iterations),
        )
        transformation = result.transformation
    return transformation, result


def pose_disagreement(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Translation (m) and rotation (deg) difference between two transforms."""
    translation_m = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
    delta_rotation = a[:3, :3] @ b[:3, :3].T
    return translation_m, rotation_angle_deg(delta_rotation)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def coordinate_frame(size: float = 0.3, pose: Optional[np.ndarray] = None) -> o3d.geometry.TriangleMesh:
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    if pose is not None:
        frame.transform(pose)
    return frame


def trajectory_lineset(poses: np.ndarray, color=(1.0, 0.5, 0.0)) -> o3d.geometry.LineSet:
    """A connected polyline through the translation of every pose."""
    points = poses[:, :3, 3]
    lines = [[i, i + 1] for i in range(len(points) - 1)]
    lineset = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    lineset.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return lineset


def build_map(
    frames: list[LidarFrame], poses: np.ndarray, voxel_size: float = 0.05, every: int = 1
) -> o3d.geometry.PointCloud:
    """Place each scan (sensor frame) at its estimated pose and fuse into one cloud."""
    fused = o3d.geometry.PointCloud()
    for i in range(0, len(frames), every):
        cloud = frames[i].read_cloud_sensor()
        cloud.transform(poses[i])
        fused += cloud.voxel_down_sample(voxel_size)
    return fused.voxel_down_sample(voxel_size)


def fuse_with_provenance(
    frames: list[LidarFrame], poses: np.ndarray, voxel_size: float = 0.05
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform + voxel-downsample each frame independently, then concatenate.

    Unlike ``build_map``, this does not do a second global voxel merge, so
    every output point can still be traced back to the frame (and thus sensor
    position) it came from -- needed for TSDF viewing-angle weighting and
    confidence estimation (Step 3).

    Returns ``(points (N, 3), sensor_origins (N, 3), source_frame_idx (N,))``.
    """
    all_points, all_origins, all_frame_idx = [], [], []
    for i, frame in enumerate(frames):
        cloud = frame.read_cloud_sensor()
        cloud.transform(poses[i])
        points = np.asarray(cloud.voxel_down_sample(voxel_size).points)
        if points.shape[0] == 0:
            continue
        all_points.append(points)
        all_origins.append(np.tile(poses[i][:3, 3], (points.shape[0], 1)))
        all_frame_idx.append(np.full(points.shape[0], i, dtype=np.int64))

    return (
        np.concatenate(all_points, axis=0),
        np.concatenate(all_origins, axis=0),
        np.concatenate(all_frame_idx, axis=0),
    )


def crop_to_trajectory(
    pcd: o3d.geometry.PointCloud, poses: np.ndarray, xy_margin: float = 4.0, z_margin: float = 1.0
) -> o3d.geometry.PointCloud:
    """Crop a fused map to a slab around a trajectory's own real extent.

    A handful of noisy/reflective long-range LiDAR returns (including
    spurious high ones, e.g. through gaps or reflections) can push a map's
    bounding box out to tens of metres in every direction; left in, they
    dominate camera auto-framing and squeeze the actual scene into a corner
    of the image. XY gets a generous margin so walls stay in frame; Z is
    cropped tightly around the trajectory's own height (the robot obviously
    never flew into the ceiling) so a top-down view has a clear, unoccluded
    line of sight to the true, unmodified path -- the ceiling and any stray
    far returns simply aren't in the cropped map to block it.
    """
    centers = poses[:, :3, 3]
    margin = np.array([xy_margin, xy_margin, z_margin])
    box = o3d.geometry.AxisAlignedBoundingBox(centers.min(axis=0) - margin, centers.max(axis=0) + margin)
    return pcd.crop(box)


def _scene_bounds(geometries: list) -> Optional[np.ndarray]:
    min_bound, max_bound = None, None
    for geometry in geometries:
        try:
            box = geometry.get_axis_aligned_bounding_box()
        except Exception:
            continue
        if min_bound is None:
            min_bound, max_bound = np.asarray(box.min_bound), np.asarray(box.max_bound)
        else:
            min_bound = np.minimum(min_bound, box.min_bound)
            max_bound = np.maximum(max_bound, box.max_bound)
    if min_bound is None:
        return None
    return (min_bound + max_bound) / 2.0


def _scene_camera(geometries: list, top_down: bool = False) -> dict:
    """Fixed camera aimed at the scene's bbox centre (dataset-agnostic).

    Two presets:
    - Oblique 3/4 (default): looks down and across the scene, good for
      showing wall height on a room-shaped point cloud.
    - Top-down (``top_down=True``): looks straight down the odom Z axis
      (world-up for Spot). Needed whenever a floor-level trajectory has to
      stay visible against a wall/ceiling map -- from an oblique angle the
      walls of a corridor occlude a path running through its open middle;
      from directly above, the open volume has a clear line of sight down to
      the floor.
    """
    lookat = _scene_bounds(geometries)
    if lookat is None:
        lookat = [0.0, 0.0, 0.0]
    if top_down:
        return {"lookat": lookat, "front": [0.0, 0.0, 1.0], "up": [0.0, 1.0, 0.0], "zoom": 0.6}
    return {"lookat": lookat, "front": [0.5, -0.6, 0.6], "up": [0.0, 0.0, 1.0], "zoom": 0.55}


def draw(
    geometries: list,
    *,
    title: str,
    screenshot_path: Optional[Path] = None,
    show: bool = True,
    point_size: float = 2.0,
    top_down: bool = False,
) -> None:
    """Save an offscreen screenshot and/or pop up an interactive Open3D window.

    Both use the same fixed camera angle (see ``_scene_camera``) so
    screenshots are consistently framed instead of depending on auto-fit.
    """
    camera = _scene_camera(geometries, top_down=top_down)
    if screenshot_path is not None:
        _render_offscreen(geometries, Path(screenshot_path), camera, point_size=point_size)
    if show:
        o3d.visualization.draw_geometries(
            geometries, window_name=title, width=1280, height=800, **camera
        )


def _render_offscreen(geometries: list, path: Path, camera: dict, *, point_size: float = 2.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=1280, height=800)
        for geometry in geometries:
            vis.add_geometry(geometry)
        opt = vis.get_render_option()
        opt.point_size = point_size
        opt.background_color = np.array([1.0, 1.0, 1.0])
        opt.point_color_option = o3d.visualization.PointColorOption.Color
        opt.light_on = False
        opt.line_width = 5.0
        view_control = vis.get_view_control()
        view_control.set_lookat(camera["lookat"])
        view_control.set_front(camera["front"])
        view_control.set_up(camera["up"])
        view_control.set_zoom(camera["zoom"])
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(str(path), do_render=True)
        vis.destroy_window()
    except Exception as exc:  # noqa: BLE001 - screenshots are best-effort
        print(f"WARNING: offscreen render failed for {path.name}: {exc}")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
