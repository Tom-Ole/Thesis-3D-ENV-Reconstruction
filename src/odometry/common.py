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


def body_T_frame(edges: dict, frame: str) -> np.ndarray:
    """
    Compose body_T_frame by walking a transforms_snapshot up to 'body'.

    Each edge stores parent_tform_child + the parent's name, so we accumulate
    body_T_child = body_T_parent @ parent_tform_child along the chain. Used to
    place any sensor frame (e.g. 'frontleft_fisheye') in the common body frame.
    """
    T = np.eye(4)
    cur = frame
    seen = set()
    while cur in edges and cur != "body" and cur not in seen:
        seen.add(cur)
        T = _edge_matrix(edges[cur]) @ T
        cur = edges[cur].get("parent_frame_name", "body")
    return T


def odom_T_frame(edges: dict, frame: str) -> np.ndarray:
    """frame's pose in the (fixed) odom frame, composed from one snapshot."""
    body_T_odom = _edge_matrix(edges["odom"])      # odom's parent is body
    return np.linalg.inv(body_T_odom) @ body_T_frame(edges, frame)


# --- Camera intrinsics ------------------------------------------------------

def intrinsics_matrix(camera_model: dict) -> np.ndarray:
    """Bosdyn pinhole camera_model -> 3x3 intrinsics K (raw sensor frame)."""
    intr = camera_model["intrinsics"]
    fx = intr["focal_length"]["x"]
    fy = intr["focal_length"]["y"]
    cx = intr["principal_point"]["x"]
    cy = intr["principal_point"]["y"]
    skew = intr.get("skew", {}).get("x", 0.0)
    return np.array([[fx, skew, cx],
                     [0.0, fy,  cy],
                     [0.0, 0.0, 1.0]])


# A 90deg-clockwise IMAGE rotation corresponds to a +90deg rotation of the
# camera optical frame about its own optical (z) axis. Keeping image *and*
# intrinsics *and* extrinsics consistent means applying this same rotation to
# the optical-frame pose (see rotate_intrinsics_90cw / Step 0).
ROT_Z_90 = np.array([[0.0, -1.0, 0.0, 0.0],
                     [1.0,  0.0, 0.0, 0.0],
                     [0.0,  0.0, 1.0, 0.0],
                     [0.0,  0.0, 0.0, 1.0]])

# 180deg image rotation == 180deg about the optical (z) axis (self-inverse).
ROT_Z_180 = np.array([[-1.0,  0.0, 0.0, 0.0],
                      [0.0,  -1.0, 0.0, 0.0],
                      [0.0,   0.0, 1.0, 0.0],
                      [0.0,   0.0, 0.0, 1.0]])


def rotate_intrinsics_90cw(K: np.ndarray, w0: int, h0: int):
    """
    Map raw intrinsics onto an image that was rotated 90deg CLOCKWISE.

    Spot stores frontleft/frontright already rotated upright (w0xh0 sensor ->
    h0xw0 image) but leaves the intrinsics describing the raw sensor. For a CW
    rotation a raw pixel (u,v) lands at (u',v') = ((h0-1)-v, u); solving for the
    intrinsics that reproduce that (with the optical frame turned by ROT_Z_90)
    gives a clean upper-triangular K':

        fx' = fy,  fy' = fx,  cx' = (h0-1) - cy,  cy' = cx

    Returns (K_rot, (w_rot, h_rot)) with the rotated image size.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    K_rot = np.array([[fy,  0.0, (h0 - 1) - cy],
                      [0.0, fx,  cx],
                      [0.0, 0.0, 1.0]])
    return K_rot, (h0, w0)


def rotate_intrinsics_180(K: np.ndarray, w: int, h: int):
    """Map intrinsics onto a 180deg-rotated image (size unchanged).

    A 180deg rotation sends pixel (u,v) -> (w-1-u, h-1-v); the focal lengths are
    unchanged and the principal point mirrors through the centre (paired with a
    ROT_Z_180 turn of the optical frame). Returns (K_rot, (w, h)).
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    K_rot = np.array([[fx,  0.0, (w - 1) - cx],
                      [0.0, fy,  (h - 1) - cy],
                      [0.0, 0.0, 1.0]])
    return K_rot, (w, h)


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


# --- Camera images ----------------------------------------------------------

# The five RGB fisheye sources, by the suffix used in their filenames.
CAMERA_SOURCES = ("back", "frontleft", "frontright", "left", "right")


@dataclass
class ImageFrame:
    """One RGB image plus its intrinsics, pose and rotation bookkeeping.

    Reads as stored on disk: front cameras are already rotated upright, the rest
    are landscape. ``rotation_applied`` records what the capture did so Step 0
    can reconcile the intrinsics (see common.rotate_intrinsics_90cw).
    """
    index: int
    image_path: str
    meta_path: str
    t_nsec: int
    source: str                  # e.g. "frontleft_fisheye_image"
    sensor_frame: str            # e.g. "frontleft_fisheye"
    rotation_applied: str | None  # e.g. "ROTATE_90_CLOCKWISE" or None
    K_raw: np.ndarray            # 3x3, describes the RAW (unrotated) sensor
    raw_size: tuple              # (w0, h0) of the raw sensor (cols, rows)
    odom_T_sensor: np.ndarray    # raw optical-frame pose in odom (4x4)

    @property
    def t_sec(self) -> float:
        return self.t_nsec * 1e-9

    @property
    def camera_name(self) -> str:
        """Short name ('frontleft', 'back', ...) from the source string."""
        return self.source.replace("_fisheye_image", "")

    def read_image(self) -> np.ndarray:
        """BGR image as stored on disk (already upright for front cameras)."""
        import cv2
        return cv2.imread(self.image_path, cv2.IMREAD_COLOR)


def load_image_frames(data_root: str):
    """All RGB image frames for a session, sorted by filename, with poses."""
    root = os.path.abspath(data_root)
    images_dir = os.path.join(root, "images")
    metadata_dir = os.path.join(root, "metadata")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"No 'images' folder at {images_dir}")

    frames = []
    for img in sorted(glob.glob(os.path.join(images_dir, "*.jpg")),
                      key=_natural_key):
        stem = os.path.splitext(os.path.basename(img))[0]
        meta_path = os.path.join(metadata_dir, stem + ".json")
        if not os.path.isfile(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("capture_type") != "image":
            continue

        edges = (meta.get("transforms_snapshot", {})
                     .get("child_to_parent_edge_map", {}))
        sensor_frame = meta.get("frame_name_image_sensor", "")
        odom_T_sensor = (odom_T_frame(edges, sensor_frame)
                         if sensor_frame in edges else np.eye(4))

        frames.append(ImageFrame(
            index=meta.get("index", len(frames) + 1),
            image_path=img,
            meta_path=meta_path,
            t_nsec=int(meta.get("acquisition_time_robot_nsec", 0)),
            source=meta.get("source", stem),
            sensor_frame=sensor_frame,
            rotation_applied=meta.get("rotation_applied"),
            K_raw=intrinsics_matrix(meta["camera_model"]),
            raw_size=(int(meta.get("cols", 0)), int(meta.get("rows", 0))),
            odom_T_sensor=odom_T_sensor,
        ))
    return frames


# --- Prepared cameras (Step 0 manifest, read side) --------------------------

@dataclass
class Camera:
    """A prepared camera from Step 0's manifest, with arrays already parsed.

    The contract guaranteed by Step 0: ``image_path`` and ``K`` always match
    (rotation/undistortion already reconciled), and ``odom_T_camera`` is in the
    same odom frame as the LiDAR clouds (Frame.read_cloud()). So projecting a
    LiDAR point is just ``K @ (cam_T_odom @ X_odom)``.

    Step B1 (step_b1_image_poses.py) optionally fills the ``*_refined`` /
    ``lidar_*`` fields: a pose made drift-consistent with the LiDAR SLAM
    trajectory and the time-nearest LiDAR scan to use as the B3 metric anchor.
    When present, ``.pose`` (and therefore ``.cam_T_odom`` / ``.project``)
    transparently uses the refined pose; otherwise it falls back to Step 0's raw
    Spot-odometry pose. Downstream code never has to branch on whether B1 ran.
    """
    source: str
    camera: str
    t_nsec: int
    image_path: str
    width: int
    height: int
    K: np.ndarray               # 3x3
    odom_T_camera: np.ndarray   # 4x4 (raw Spot per-image odometry, from Step 0)
    distortion_mode: str
    mask_path: str | None
    source_image: str
    # --- Step B1 enrichment (None until step_b1_image_poses.py has run) ------
    odom_T_camera_refined: np.ndarray | None = None  # 4x4, LiDAR-SLAM consistent
    lidar_index: int | None = None      # nearest LiDAR scan (SessionDataset idx)
    lidar_t_nsec: int | None = None     # that scan's timestamp
    dt_sec: float | None = None         # |image - scan| time gap (seconds)
    pose_source: str | None = None      # "interpolated" | "clamped"
    # --- Step B2 enrichment (None until step_b2_depth.py has run) ------------
    depth_path: str | None = None       # per-pixel depth map (.npy) on disk
    depth_kind: str | None = None       # "metric_m" | "inverse_relative"
    # --- Step B3 enrichment (None until step_b3_align.py has run) ------------
    align_a: float | None = None        # disparity affine: 1/z_lidar ~ a/z + b
    align_b: float | None = None
    align_grid_path: str | None = None  # coarse disparity-residual grid (.npy)
    align_quality: float | None = None  # per-image fit_quality in [0, 1]
    align_anchor_zmax: float | None = None  # range LiDAR validated (m); taper beyond
    depth_aligned_kind: str | None = None  # "metric_aligned_m"

    @property
    def t_sec(self) -> float:
        return self.t_nsec * 1e-9

    @property
    def size(self) -> tuple:
        return (self.width, self.height)

    @property
    def pose(self) -> np.ndarray:
        """Best available odom->camera pose: B1-refined if present, else raw."""
        return (self.odom_T_camera_refined if self.odom_T_camera_refined
                is not None else self.odom_T_camera)

    @property
    def cam_T_odom(self) -> np.ndarray:
        """World(odom)->camera transform, ready for projection."""
        return np.linalg.inv(self.pose)

    def anchor_scan(self, dataset: "SessionDataset"):
        """The time-nearest LiDAR Frame chosen by B1 (the B3 anchor scan).

        Returns None until step_b1_image_poses.py has populated ``lidar_index``.
        """
        if self.lidar_index is None:
            return None
        return dataset.frames[self.lidar_index]

    def read_image(self) -> np.ndarray:
        """The prepared BGR image (upright, matching self.K)."""
        import cv2
        return cv2.imread(self.image_path, cv2.IMREAD_COLOR)

    def read_mask(self):
        """Valid-pixel mask if undistortion produced one, else None."""
        if not self.mask_path:
            return None
        import cv2
        return cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)

    def read_depth(self):
        """B2's dense depth map as float32 (H,W), or None until B2 has run.

        Units follow ``depth_kind``: "metric_m" is metres (pre-B3, so scale is
        only approximate); "inverse_relative" is disparity-like (larger=closer).
        """
        if not self.depth_path:
            return None
        return np.load(self.depth_path).astype(np.float32)

    def read_depth_aligned(self):
        """B3 LiDAR-anchored metric depth (H,W) m, or None until B3 has run.

        Applies the disparity affine (and smoothed residual grid, if B3 kept
        one) to ``read_depth()`` on the fly: ``z = 1/(a/z_pred + b + grid)``.
        Pixels where the corrected disparity is non-positive are NaN.
        """
        if self.align_a is None:
            return None
        z = self.read_depth()
        if z is None:
            return None
        D = self.align_a / np.clip(z, 1e-6, None) + self.align_b
        if self.align_grid_path:
            import cv2
            g = np.load(self.align_grid_path).astype(np.float32)
            D = D + cv2.resize(g, (z.shape[1], z.shape[0]),
                               interpolation=cv2.INTER_LINEAR)
        return np.where(D > 1e-6, 1.0 / D, np.nan).astype(np.float32)

    def confidence(self, max_range: float = 12.0, edge_k: float = 0.5,
                   z: np.ndarray | None = None):
        """Per-pixel confidence in [0,1] for the aligned depth (B4 weighting).

        ``fit_quality`` (per-image, from B3) x an edge weight (low at depth
        discontinuities, where monocular depth is least reliable) x range
        validity x an extrapolation taper that decays to 0 between
        ``align_anchor_zmax`` and 1.5x it (depth past where LiDAR could check
        it is untrustworthy). Pass ``z`` to reuse an already-loaded aligned
        depth. None until B3 has run.
        """
        if z is None:
            z = self.read_depth_aligned()
        if z is None:
            return None
        zc = np.nan_to_num(z, nan=max_range)
        valid = np.isfinite(z) & (z > 0.2) & (z < max_range)
        logz = np.log(np.clip(zc, 0.2, max_range))
        gy, gx = np.gradient(logz)
        edge = np.exp(-np.hypot(gx, gy) / edge_k)
        q = self.align_quality if self.align_quality is not None else 1.0
        conf = valid * edge * q
        if self.align_anchor_zmax:
            zt = self.align_anchor_zmax
            over = np.clip((zc - zt) / (0.5 * zt), 0.0, 1.0)  # 1@zt -> 0@1.5zt
            conf = conf * (1.0 - over)
        return conf.astype(np.float32)

    def project(self, pts_odom: np.ndarray):
        """Project Nx3 odom-frame points -> (uv Mx2, depth M, keep-mask N).

        Keeps only points in front of the camera and inside the image; ``keep``
        indexes back into the input rows so callers can carry colour/labels.
        """
        pts = np.asarray(pts_odom, dtype=float).reshape(-1, 3)
        cam = (self.cam_T_odom @ np.hstack(
            [pts, np.ones((len(pts), 1))]).T).T[:, :3]
        z = cam[:, 2]
        uv = (self.K @ cam.T).T
        uv = uv[:, :2] / uv[:, 2:3]
        keep = (z > 1e-6) & (uv[:, 0] >= 0) & (uv[:, 0] < self.width) \
            & (uv[:, 1] >= 0) & (uv[:, 1] < self.height)
        return uv[keep], z[keep], keep


def load_camera_manifest(data_root: str):
    """Load Step 0's output/cameras/manifest.json as a list of Camera records.

    If Step B1 has written output/cameras/poses_b1.json, each camera is enriched
    in place with the refined pose + nearest-LiDAR-scan fields (matched by
    ``source_image``), so ``.pose`` / ``.project()`` use the drift-consistent
    pose automatically.
    """
    root = os.path.abspath(data_root)
    path = os.path.join(root, "output", "cameras", "manifest.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No camera manifest at {path}. Run step0_camera_prep.py first.")
    with open(path) as f:
        manifest = json.load(f)

    cams = []
    for c in manifest.get("cameras", []):
        cams.append(Camera(
            source=c["source"],
            camera=c["camera"],
            t_nsec=int(c["t_nsec"]),
            image_path=c["image_path"],
            width=int(c["width"]),
            height=int(c["height"]),
            K=np.array(c["K"], dtype=float),
            odom_T_camera=np.array(c["odom_T_camera"], dtype=float),
            distortion_mode=c.get("distortion_mode", "none"),
            mask_path=c.get("mask_path"),
            source_image=c.get("source_image", c["image_path"]),
        ))

    _merge_image_poses(root, cams)
    _merge_depth(root, cams)
    _merge_align(root, cams)
    return cams


def _merge_image_poses(root: str, cams: list):
    """Fold Step B1's poses_b1.json (if present) into the Camera records."""
    path = os.path.join(root, "output", "cameras", "poses_b1.json")
    if not os.path.isfile(path):
        return
    with open(path) as f:
        poses = json.load(f)
    by_src = {p["source_image"]: p for p in poses.get("cameras", [])}
    for cam in cams:
        p = by_src.get(cam.source_image)
        if p is None:
            continue
        cam.odom_T_camera_refined = np.array(
            p["odom_T_camera_refined"], dtype=float)
        cam.lidar_index = int(p["lidar_index"])
        cam.lidar_t_nsec = int(p["lidar_t_nsec"])
        cam.dt_sec = float(p["dt_sec"])
        cam.pose_source = p.get("pose_source")


def _merge_depth(root: str, cams: list):
    """Fold Step B2's depth_b2.json (if present) into the Camera records."""
    path = os.path.join(root, "output", "depth", "depth_b2.json")
    if not os.path.isfile(path):
        return
    with open(path) as f:
        depth = json.load(f)
    kind = depth.get("depth_kind")
    by_src = {d["source_image"]: d for d in depth.get("images", [])}
    for cam in cams:
        d = by_src.get(cam.source_image)
        if d is None:
            continue
        cam.depth_path = d["depth_path"]
        cam.depth_kind = kind


def _merge_align(root: str, cams: list):
    """Fold Step B3's align_b3.json (if present) into the Camera records."""
    path = os.path.join(root, "output", "depth", "align_b3.json")
    if not os.path.isfile(path):
        return
    with open(path) as f:
        align = json.load(f)
    kind = align.get("depth_aligned_kind")
    by_src = {a["source_image"]: a for a in align.get("images", [])}
    for cam in cams:
        a = by_src.get(cam.source_image)
        if a is None:
            continue
        cam.align_a = float(a["a"])
        cam.align_b = float(a["b"])
        cam.align_grid_path = a.get("align_grid_path")
        cam.align_quality = float(a["fit_quality"])
        cam.align_anchor_zmax = (float(a["anchor_zmax"])
                                 if a.get("anchor_zmax") is not None else None)
        cam.depth_aligned_kind = kind

def load_image_cloud(data_root: str):
    """Load Step B4's dense image cloud -> (points, colors, normals, confidence).

    Arrays are Nx3 / Nx3 / Nx3 / N, in the refined odom frame (same frame as the
    LiDAR map). Raises if B4 has not been run.
    """
    root = os.path.abspath(data_root)
    path = os.path.join(root, "output", "fusion", "image_cloud.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No image cloud at {path}. Run step_b4_cloud.py first.")
    d = np.load(path)
    return d["points"], d["colors"], d["normals"], d["confidence"]


def load_fused_cloud(data_root: str):
    """Load Phase C's fused LiDAR+image cloud as an Open3D oriented point cloud.

    Returns (pcd, attrs) where pcd has points/colors/normals and attrs carries
    the per-point ``confidence`` and ``is_lidar`` arrays. In the refined odom
    frame; ready for Phase D meshing.
    """
    root = os.path.abspath(data_root)
    path = os.path.join(root, "output", "fusion", "fused_cloud.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No fused cloud at {path}. Run step_c_fuse.py first.")
    d = np.load(path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(d["points"])
    pcd.colors = o3d.utility.Vector3dVector(d["colors"])
    pcd.normals = o3d.utility.Vector3dVector(d["normals"])
    return pcd, {"confidence": d["confidence"], "is_lidar": d["is_lidar"]}


def load_fused_mesh(data_root: str):
    """Load Phase D's fused triangle mesh (geometry for Phase E texturing)."""
    root = os.path.abspath(data_root)
    path = os.path.join(root, "output", "fusion", "mesh_fused.ply")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No fused mesh at {path}. Run step_d_mesh.py first.")
    return o3d.io.read_triangle_mesh(path)


# Example:
# import common

# cams = common.load_camera_manifest("./captures/session_05_20260624")
# lidar = common.SessionDataset("./captures/session_05_20260624")

# for cam in cams:
#     img = cam.read_image()                         # upright BGR, matches cam.K
#     lf  = min(lidar.frames, key=lambda f: abs(f.t_nsec - cam.t_nsec))  # B3 sync
#     uv, depth, keep = cam.project(lf.read_cloud().points)  # sparse metric anchor
#     # B2: run depth model on img; B3: fit predicted depth to `depth` at `uv`


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
