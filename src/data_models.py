"""Plain data models shared across the capture layer and the GUI.

These dataclasses are intentionally dependency-free (only stdlib + typing) so
they can be imported and unit-tested without the Spot SDK or PySide6 present.
``numpy`` arrays are referenced via ``Any`` to keep this module import-light.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# Capture type discriminators used in filenames / metadata.
CAPTURE_TYPE_IMAGE = "image"
CAPTURE_TYPE_POINTCLOUD = "pointcloud"


@dataclass
class ImageSourceInfo:
    """Lightweight description of a Spot image source (camera)."""

    name: str
    rows: int = 0
    cols: int = 0
    image_type: str = "IMAGE_TYPE_UNKNOWN"
    pixel_formats: list[str] = field(default_factory=list)


@dataclass
class PointCloudSourceInfo:
    """Lightweight description of a Spot point-cloud source (e.g. EAP LiDAR)."""

    name: str
    frame_name_sensor: str = ""


@dataclass
class CapturedImage:
    """Decoded, in-memory result of a single image capture, ready to persist.

    Exactly one of ``jpeg_bytes`` (raw JPEG passthrough) or ``array`` (decoded
    pixels written via OpenCV) is set, depending on the source format and
    whether camera rotation was applied.
    """

    source: str
    image_format: str
    pixel_format: str
    rows: int
    cols: int
    extension: str = "jpg"
    jpeg_bytes: Optional[bytes] = None
    array: Optional[Any] = None  # numpy.ndarray when set
    acquisition_time: Optional[str] = None  # ISO 8601, robot clock
    acquisition_time_robot_nsec: Optional[int] = None  # robot clock, nanoseconds
    frame_name_image_sensor: str = ""
    camera_model: dict = field(default_factory=dict)  # intrinsics + distortion
    transforms_snapshot: dict = field(default_factory=dict)
    rotation_applied: Optional[str] = None


@dataclass
class CapturedPointCloud:
    """Decoded, in-memory result of a single point-cloud capture."""

    source: str
    num_points: int
    points: Any  # numpy.ndarray of shape (N, 3), float32
    encoding: str = ""
    acquisition_time: Optional[str] = None  # ISO 8601, robot clock
    acquisition_time_robot_nsec: Optional[int] = None  # robot clock, nanoseconds
    frame_name_sensor: str = ""
    transforms_snapshot: dict = field(default_factory=dict)


@dataclass
class CaptureResult:
    """Metadata describing a single persisted capture asset on disk."""

    index: int
    capture_type: str
    source: str
    file_path: str
    metadata_path: str
    timestamp: str
    group_id: str
    rows: Optional[int] = None
    cols: Optional[int] = None
    num_points: Optional[int] = None
    preview_path: Optional[str] = None


@dataclass
class CaptureBatchResult:
    """Outcome of a single capture action (which may produce several assets)."""

    event_id: str
    session_name: str
    results: list[CaptureResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class ConnectionStatus:
    """Snapshot of the current Spot connection state, safe for the GUI."""

    connected: bool = False
    hostname: str = ""
    robot_name: str = ""
    message: str = ""
    has_point_cloud: bool = False
    battery_percent: Optional[float] = None


@dataclass
class ConnectResult:
    """Combined result of a connect-and-discover action."""

    status: ConnectionStatus
    image_sources: list[ImageSourceInfo] = field(default_factory=list)
    point_cloud_sources: list[PointCloudSourceInfo] = field(default_factory=list)
