"""GUI-facing orchestration for the capture workflow.

Ties together :class:`SpotService` (robot I/O), :class:`SessionManager`
(folders/numbering) and :class:`DiskWriter` (persistence). It is Qt-free so it
can be exercised from background worker threads or tests; the GUI only needs to
call :meth:`connect_and_discover`, :meth:`start_new_session` and
:meth:`capture`.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable, List, Optional

from src.capture.capture_session import (
    CaptureSession,
    SessionManager,
    sanitize_name,
)
from src.capture.spot_service import SpotService
from src.capture.writter import DiskWriter
from src.config import Config
from src.data_models import (
    CAPTURE_TYPE_IMAGE,
    CAPTURE_TYPE_POINTCLOUD,
    CaptureBatchResult,
    CapturedImage,
    CapturedPointCloud,
    CaptureResult,
    ConnectResult,
    ConnectionStatus,
)

logger = logging.getLogger(__name__)

LogFn = Callable[[str], None]


def _noop(_: str) -> None:
    pass


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat()


class CaptureController:
    """Orchestrates connect, session and capture-and-persist operations."""

    def __init__(self, config: Config, spot: SpotService, sessions: SessionManager):
        self.config = config
        self.spot = spot
        self.sessions = sessions
        self.writer = DiskWriter()

    # -- connection --------------------------------------------------------

    def connect_and_discover(
        self,
        hostname: str,
        username: str,
        password: str,
        log: LogFn = _noop,
    ) -> ConnectResult:
        """Connect to the robot and enumerate available capture sources."""
        status = self.spot.connect(hostname, username, password, log=log)
        image_sources = self.spot.list_image_sources()
        pc_sources = self.spot.list_point_cloud_sources()
        log(
            f"Found {len(image_sources)} image source(s) and "
            f"{len(pc_sources)} point-cloud source(s)."
        )
        return ConnectResult(
            status=status,
            image_sources=image_sources,
            point_cloud_sources=pc_sources,
        )

    def disconnect(self, log: LogFn = _noop) -> ConnectionStatus:
        self.spot.disconnect(log=log)
        return ConnectionStatus(connected=False, message="Disconnected")

    # -- sessions ----------------------------------------------------------

    @property
    def active_session(self) -> Optional[CaptureSession]:
        return self.sessions.active_session

    def start_new_session(self) -> CaptureSession:
        session = self.sessions.start_new_session()
        logger.info("Started capture session %s at %s", session.name, session.path)
        return session

    def ensure_session(self) -> CaptureSession:
        """Return the active session, creating one automatically if needed."""
        if self.sessions.active_session is None:
            return self.start_new_session()
        return self.sessions.active_session

    def prepare_recording(self, log: LogFn = _noop) -> CaptureSession:
        """Ensure a session exists and write its calibration manifest.

        Called once when a recording starts. The manifest consolidates the
        static, per-session information needed for Step-0 data preparation:
        camera intrinsics + distortion, the body<->sensor frame tree (nominal
        extrinsics), the robot/client clock skew, and the configured rates.
        Manifest failures are logged but never block the recording.
        """
        session = self.ensure_session()
        try:
            self._write_session_manifest(session, log)
        except Exception as exc:  # noqa: BLE001 - manifest is best-effort
            logger.exception("Failed to write session manifest")
            log(f"WARNING: could not write calibration manifest: {exc}")
        return session

    def state_log_path(self, session: CaptureSession):
        """Path of the IMU / odometry JSONL log for a session."""
        return session.state_log_path

    def _write_session_manifest(self, session: CaptureSession, log: LogFn) -> None:
        manifest = {
            "session": session.name,
            "created_at": _now_iso(),
            "robot_hostname": self.spot.hostname,
            "time_sync": {
                "clock_skew_sec": self.spot.clock_skew_sec(),
                "note": (
                    "All sensor timestamps are in the robot clock. Subtract "
                    "clock_skew_sec to map them to the local/client clock. "
                    "Because LiDAR, cameras and IMU share the robot clock they "
                    "are mutually synchronised; interpolate on robot-clock "
                    "nanoseconds (acquisition_time_robot_nsec)."
                ),
            },
            "capture_rates_hz": {
                "image": self.config.image_sample_rate,
                "lidar": self.config.lidar_sample_rate,
                "state_poll": self.config.state_sample_rate,
            },
            "imu_source": (
                "robot-state-streaming (raw IMU, high-rate)"
                if self.spot.has_state_streaming
                else "robot-state polling (fused odometry pose)"
            ),
            "camera_rotation_applied": self.config.apply_camera_rotation,
            "cameras": self.spot.get_camera_models(),
            "point_cloud_sources": [
                {"name": s.name, "frame_name_sensor": s.frame_name_sensor}
                for s in self.spot.list_point_cloud_sources()
            ],
            "frame_tree_snapshot": self.spot.get_frame_tree_snapshot(),
            "extrinsics_note": (
                "Per-asset metadata carries each sensor's transforms_snapshot "
                "and sensor frame name. Compose through the common 'body'/'odom' "
                "frame to obtain LiDAR<->camera and LiDAR<->body(IMU) transforms; "
                "these are the nominal/factory extrinsics to validate or refine."
            ),
            "deskew_note": (
                "Spot's PointCloud API exposes one acquisition timestamp per "
                "scan, not per-point timestamps. Deskew by interpolating the "
                "high-rate pose/IMU log across the scan period (assume uniform "
                "angular sweep)."
            ),
        }
        self.writer.write_json(session.manifest_path, manifest)
        log(f"Wrote calibration manifest {session.manifest_path.name}")

    # -- capture -----------------------------------------------------------

    def capture(
        self,
        image_sources: List[str],
        point_cloud_sources: List[str],
        log: LogFn = _noop,
    ) -> CaptureBatchResult:
        """Capture the requested sources and persist them under one event id.

        All assets produced by a single call share one ``event_id`` so that
        related images and point clouds can be matched later (e.g. for
        "Capture Both"). Image and point-cloud failures are isolated so one does
        not prevent the other from being saved.
        """
        session = self.ensure_session()
        event_id = session.next_event_id()
        batch = CaptureBatchResult(event_id=event_id, session_name=session.name)
        log(f"Capture event {event_id} in {session.name}")

        if image_sources:
            try:
                for image in self.spot.capture_images(image_sources):
                    batch.results.append(
                        self._save_image(session, image, event_id, log)
                    )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Image capture failed")
                batch.errors.append(f"Image capture failed: {exc}")

        if point_cloud_sources:
            try:
                for cloud in self.spot.capture_point_clouds(point_cloud_sources):
                    batch.results.append(
                        self._save_point_cloud(session, cloud, event_id, log)
                    )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Point-cloud capture failed")
                batch.errors.append(f"Point-cloud capture failed: {exc}")

        return batch

    # -- persistence -------------------------------------------------------

    def _save_image(
        self,
        session: CaptureSession,
        image: CapturedImage,
        event_id: str,
        log: LogFn,
    ) -> CaptureResult:
        index = session.next_image_index()
        base = f"{index:04d}_{sanitize_name(image.source)}"
        file_path = session.images_dir / f"{base}.{image.extension}"

        if image.jpeg_bytes is not None:
            self.writer.write_image_bytes(file_path, image.jpeg_bytes)
        elif image.array is not None:
            self.writer.write_image_array(file_path, image.array)
        else:
            raise ValueError(f"No image data decoded for source {image.source}")

        timestamp = _now_iso()
        metadata = {
            "index": index,
            "capture_type": CAPTURE_TYPE_IMAGE,
            "group_id": event_id,
            "session": session.name,
            "source": image.source,
            "saved_at": timestamp,
            "acquisition_time": image.acquisition_time,
            "acquisition_time_robot_nsec": image.acquisition_time_robot_nsec,
            "clock_skew_sec": self.spot.clock_skew_sec(),
            "image_format": image.image_format,
            "pixel_format": image.pixel_format,
            "rows": image.rows,
            "cols": image.cols,
            "rotation_applied": image.rotation_applied,
            "frame_name_image_sensor": image.frame_name_image_sensor,
            "camera_model": image.camera_model,
            "transforms_snapshot": image.transforms_snapshot,
            "file": str(file_path.relative_to(session.path)),
        }
        metadata_path = session.metadata_dir / f"{base}.json"
        self.writer.write_json(metadata_path, metadata)
        log(f"Saved image {file_path.name}")

        return CaptureResult(
            index=index,
            capture_type=CAPTURE_TYPE_IMAGE,
            source=image.source,
            file_path=str(file_path),
            metadata_path=str(metadata_path),
            timestamp=timestamp,
            group_id=event_id,
            rows=image.rows,
            cols=image.cols,
            preview_path=str(file_path),
        )

    def _save_point_cloud(
        self,
        session: CaptureSession,
        cloud: CapturedPointCloud,
        event_id: str,
        log: LogFn,
    ) -> CaptureResult:
        index = session.next_pointcloud_index()
        base = f"{index:04d}_{sanitize_name(cloud.source)}"
        file_path = session.pointclouds_dir / f"{base}.ply"
        self.writer.write_point_cloud_ply(file_path, cloud.points)

        timestamp = _now_iso()
        metadata = {
            "index": index,
            "capture_type": CAPTURE_TYPE_POINTCLOUD,
            "group_id": event_id,
            "session": session.name,
            "source": cloud.source,
            "saved_at": timestamp,
            "acquisition_time": cloud.acquisition_time,
            "acquisition_time_robot_nsec": cloud.acquisition_time_robot_nsec,
            "clock_skew_sec": self.spot.clock_skew_sec(),
            "num_points": cloud.num_points,
            "encoding": cloud.encoding,
            "frame_name_sensor": cloud.frame_name_sensor,
            "transforms_snapshot": cloud.transforms_snapshot,
            "file": str(file_path.relative_to(session.path)),
        }
        metadata_path = session.metadata_dir / f"{base}.json"
        self.writer.write_json(metadata_path, metadata)
        log(f"Saved point cloud {file_path.name} ({cloud.num_points} points)")

        return CaptureResult(
            index=index,
            capture_type=CAPTURE_TYPE_POINTCLOUD,
            source=cloud.source,
            file_path=str(file_path),
            metadata_path=str(metadata_path),
            timestamp=timestamp,
            group_id=event_id,
            num_points=cloud.num_points,
        )
