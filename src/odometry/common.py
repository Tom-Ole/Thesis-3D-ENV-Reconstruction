"""
Shared dataset + visualization helpers for the SLAM pipeline (Steps 1-3).

The capture format (per session):
    <session>/
        pointclouds/  NNNN_velodyne-point-cloud.ply   (points in the LiDAR
                                                        sensor frame)
        metadata/     NNNN_velodyne-point-cloud.json   (per-scan transforms +
                                                        robot-clock timestamp)
        imu/state_log.jsonl   high-rate robot state (50 Hz) - used in Step 0
        session.json          calibration / rates / time-sync notes
        output/               written by the pipeline steps

Key fact exploited by Step 1: each scan's metadata carries Spot's *fused
odometry* as the transform chain  odom -> body -> sensor.  Composing it gives
`odom_T_sensor` per scan, i.e. a real (drifting but locally excellent) motion
estimate we can seed ICP with instead of guessing.
"""

import os
import re
import json
import glob
from dataclasses import dataclass

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation


# --- Frame transforms -------------------------------------------------------

def _xyzw(rot: dict):
    """Bosdyn rotation dict -> (x, y, z, w), defaulting to identity."""
    return [rot.get("x", 0.0), rot.get("y", 0.0),
            rot.get("z", 0.0), rot.get("w", 1.0)]


def _xyz(pos: dict):
    """Bosdyn position dict -> (x, y, z), defaulting to origin."""
    return [pos.get("x", 0.0), pos.get("y", 0.0), pos.get("z", 0.0)]


def pose_to_matrix(position, quaternion_xyzw) -> np.ndarray:
    """Build a 4x4 homogeneous transform from a position + (x,y,z,w) quat."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(quaternion_xyzw).as_matrix()
    T[:3, 3] = position
    return T


def _edge_matrix(edge: dict) -> np.ndarray:
    """parent_tform_child of one transforms_snapshot edge -> 4x4 matrix."""
    p = edge.get("parent_tform_child", {})
    return pose_to_matrix(_xyz(p.get("position", {})),
                          _xyzw(p.get("rotation", {})))


# --- Dataset ----------------------------------------------------------------

@dataclass
class Frame:
    """One LiDAR scan plus its synchronized pose metadata."""
    index: int
    ply_path: str
    meta_path: str
    t_nsec: int                 # robot-clock acquisition time (nanoseconds)
    odom_T_body: np.ndarray     # Spot fused-odometry body pose (4x4)
    body_T_sensor: np.ndarray   # static LiDAR extrinsic (4x4)

    @property
    def odom_T_sensor(self) -> np.ndarray:
        """LiDAR pose in the odom frame (Spot's prior for this scan)."""
        return self.odom_T_body @ self.body_T_sensor

    @property
    def t_sec(self) -> float:
        return self.t_nsec * 1e-9

    def read_cloud(self) -> o3d.geometry.PointCloud:
        """Cloud as stored on disk: ALREADY in the odom frame."""
        return o3d.io.read_point_cloud(self.ply_path)

    def read_cloud_sensor(self) -> o3d.geometry.PointCloud:
        """
        Cloud back in the LiDAR sensor frame (centred on the sensor).

        The capture pre-applies Spot's odom_T_sensor before saving, so the raw
        .ply is in odom. Un-applying it recovers the true sensor scan, which is
        what frame-to-frame odometry must register.
        """
        c = o3d.io.read_point_cloud(self.ply_path)
        c.transform(np.linalg.inv(self.odom_T_sensor))
        return c


def _natural_key(path: str):
    name = os.path.basename(path)
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", name)]


class SessionDataset:
    """
    Loads LiDAR frames + their pose metadata for one capture session.
    Indexable / iterable: dataset[i] -> Frame, len(dataset) -> frame count.
    """

    def __init__(self, data_root: str):
        self.root = os.path.abspath(data_root)
        self.pointclouds_dir = os.path.join(self.root, "pointclouds")
        self.metadata_dir = os.path.join(self.root, "metadata")
        self.output_dir = os.path.join(self.root, "output")
        if not os.path.isdir(self.pointclouds_dir):
            raise FileNotFoundError(
                f"No 'pointclouds' folder at {self.pointclouds_dir}")

        self.frames = self._load_frames()
        if not self.frames:
            raise RuntimeError(f"No LiDAR frames found under {self.root}")

    def _load_frames(self):
        frames = []
        for ply in sorted(glob.glob(
                os.path.join(self.pointclouds_dir, "*.ply")), key=_natural_key):
            stem = os.path.splitext(os.path.basename(ply))[0]
            meta_path = os.path.join(self.metadata_dir, stem + ".json")
            if not os.path.isfile(meta_path):
                # No pose metadata -> skip; Step 1 needs the prior.
                continue
            with open(meta_path) as f:
                meta = json.load(f)

            edges = (meta.get("transforms_snapshot", {})
                         .get("child_to_parent_edge_map", {}))
            odom_T_body = _edge_matrix(edges["body"]) if "body" in edges \
                else np.eye(4)
            body_T_sensor = _edge_matrix(edges["sensor"]) if "sensor" in edges \
                else np.eye(4)

            frames.append(Frame(
                index=meta.get("index", len(frames) + 1),
                ply_path=ply,
                meta_path=meta_path,
                t_nsec=int(meta.get("acquisition_time_robot_nsec", 0)),
                odom_T_body=odom_T_body,
                body_T_sensor=body_T_sensor,
            ))
        return frames

    def spot_trajectory(self) -> np.ndarray:
        """(N, 4, 4) Spot odometry sensor poses in the odom frame."""
        return np.stack([f.odom_T_sensor for f in self.frames])

    def timestamps_sec(self) -> np.ndarray:
        return np.array([f.t_sec for f in self.frames])

    def ensure_output_dir(self) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        return self.output_dir

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        return self.frames[i]


# --- Visualization helpers --------------------------------------------------

def trajectory_lineset(poses_world: np.ndarray, color=(1.0, 0.5, 0.0)):
    """A polyline through the translation of every pose (drawable)."""
    pts = poses_world[:, :3, 3]
    lines = [[i, i + 1] for i in range(len(pts) - 1)]
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines))
    ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return ls


def pose_axes(poses_world: np.ndarray, size=0.3, every=1):
    """Coordinate frames placed at selected poses (orientation check)."""
    out = []
    for T in poses_world[::every]:
        out.append(o3d.geometry.TriangleMesh
                   .create_coordinate_frame(size=size).transform(T))
    return out


def build_map(dataset: "SessionDataset", poses_world: np.ndarray,
              voxel=0.05, every=1):
    """
    Place each scan with its estimated pose and stack into one cloud.

    Clouds are taken in the SENSOR frame and transformed by poses_world, so the
    map follows the refined trajectory. (Reading the on-disk odom clouds and
    transforming again would double-apply Spot's pose.) Used by Step 1 as a
    drift check; Step 2 replaces it with a keyframed, outlier-filtered map.
    """
    fused = o3d.geometry.PointCloud()
    for i in range(0, len(poses_world), every):
        cloud = dataset[i].read_cloud_sensor().transform(poses_world[i].copy())
        fused += cloud.voxel_down_sample(voxel)
    return fused.voxel_down_sample(voxel)


def draw(geometries, title="view"):
    """Thin wrapper so every step opens a window the same way."""
    o3d.visualization.draw_geometries(
        geometries,
        window_name=title,
        zoom=0.4459,
        front=[0.9288, -0.2951, -0.2242],
        lookat=[1.6784, 2.0612, 1.4451],
        up=[-0.3402, -0.9189, -0.1996])
