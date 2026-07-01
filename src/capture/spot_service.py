"""Central Spot connection / service layer.

Owns the lifecycle of the Spot SDK objects and the per-capability client
wrappers, and exposes a small, GUI-friendly facade for discovery and capture.

Design notes
------------
* Pure data capture (images, point clouds, robot state) requires authentication
  and time-sync only. It does **not** require a lease or E-Stop -- those are for
  commanding motion -- so this service deliberately does not acquire them.
* Clients are created once at connect time and reused for every capture.
* The EAP point-cloud (LiDAR) service is optional; if it is not registered the
  service connects without it and image capture still works.
* This object is Qt-free and is intended to be driven from background worker
  threads (see :mod:`src.gui.workers`).
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

import bosdyn.client
from bosdyn.client.auth import InvalidLoginError
from bosdyn.client.exceptions import RpcError
from bosdyn.client.image import ImageClient
from bosdyn.client.point_cloud import PointCloudClient
from bosdyn.client.robot_state import RobotStateClient
from google.protobuf import json_format

try:  # high-rate IMU service; not present in every SDK build / on every robot
    from bosdyn.client.robot_state import RobotStateStreamingClient
except Exception:  # pragma: no cover - optional capability
    RobotStateStreamingClient = None

from src.capture.image_client import ImageClientWrapper
from src.capture.lidar_client import PointCloudClientWrapper
from src.capture.state_client import StateClientWrapper
from src.config import Config
from src.data_models import (
    CapturedImage,
    CapturedPointCloud,
    ConnectionStatus,
    ImageSourceInfo,
    PointCloudSourceInfo,
)

logger = logging.getLogger(__name__)

LogFn = Callable[[str], None]


def _noop(_: str) -> None:
    pass


class SpotConnectionError(RuntimeError):
    """Raised for user-facing connection/authentication failures."""


class SpotService:
    """Owns the Spot connection and capture clients for the whole app."""

    SDK_NAME = "SpotCaptureApp"

    def __init__(self, config: Config):
        self.config = config
        self._sdk = None
        self._robot = None
        self.hostname = ""
        self.connected = False

        self.image_wrapper: Optional[ImageClientWrapper] = None
        self.pc_wrapper: Optional[PointCloudClientWrapper] = None
        self.state_wrapper: Optional[StateClientWrapper] = None
        self._state_stream_client = None  # RobotStateStreamingClient when available

    # -- lifecycle ---------------------------------------------------------

    def connect(
        self,
        hostname: str,
        username: str,
        password: str,
        log: LogFn = _noop,
    ) -> ConnectionStatus:
        """Authenticate, sync time and create the capture clients.

        Raises :class:`SpotConnectionError` with a user-friendly message on
        failure. The service is left disconnected on any error.
        """
        self.disconnect()
        try:
            log(f"Connecting to {hostname} ...")
            sdk = bosdyn.client.create_standard_sdk(self.SDK_NAME)
            robot = sdk.create_robot(hostname)
            robot.authenticate(username, password)
            log("Authenticated. Synchronizing time ...")
            robot.time_sync.wait_for_sync()

            image_client = robot.ensure_client(ImageClient.default_service_name)
            state_client = robot.ensure_client(RobotStateClient.default_service_name)
            self.image_wrapper = ImageClientWrapper(
                image_client, rotate=self.config.apply_camera_rotation
            )
            self.state_wrapper = StateClientWrapper(state_client)
            self._state_stream_client = self._try_state_stream_client(robot, log)

            self.pc_wrapper = self._try_point_cloud_client(robot, log)

            self._sdk = sdk
            self._robot = robot
            self.hostname = hostname
            self.connected = True

            status = ConnectionStatus(
                connected=True,
                hostname=hostname,
                robot_name=self._robot_name(robot),
                has_point_cloud=self.pc_wrapper is not None,
                battery_percent=self.get_battery_percent(),
                message="Connected",
            )
            log(f"Connected to {status.robot_name or hostname}.")
            return status

        except InvalidLoginError as exc:
            self.disconnect()
            raise SpotConnectionError(
                "Authentication failed - check username and password."
            ) from exc
        except RpcError as exc:
            self.disconnect()
            raise SpotConnectionError(
                f"Could not reach the robot at {hostname}: {exc}"
            ) from exc
        except Exception as exc:  # noqa: BLE001 - surface anything else cleanly
            self.disconnect()
            raise SpotConnectionError(f"Connection failed: {exc}") from exc

    def disconnect(self, log: LogFn = _noop) -> None:
        """Tear down clients and the time-sync thread (safe to call anytime)."""
        if self._robot is not None:
            try:
                self._robot.time_sync.stop()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass
            log("Disconnected.")
        self._sdk = None
        self._robot = None
        self.image_wrapper = None
        self.pc_wrapper = None
        self.state_wrapper = None
        self._state_stream_client = None
        self.connected = False

    # The EAP Velodyne payload registers this exact name in Spot's service
    # directory. PointCloudClient.default_service_name is 'point-cloud' (the
    # generic SDK name) which does not appear in the directory on an EAP robot,
    # causing ensure_client to raise UnregisteredServiceNameError.
    _VELODYNE_SERVICE_NAME = "velodyne-point-cloud"

    def _try_point_cloud_client(
        self, robot, log: LogFn
    ) -> Optional[PointCloudClientWrapper]:
        """Create the point-cloud client if the EAP service is available."""
        try:
            pc_client = robot.ensure_client(self._VELODYNE_SERVICE_NAME)
            pc_client.list_point_cloud_sources()  # verify it actually responds
            log("Point-cloud / LiDAR service detected (EAP payload).")
            return PointCloudClientWrapper(pc_client)
        except Exception as exc:  # noqa: BLE001 - optional payload, degrade gracefully
            logger.warning(
                "No point-cloud / LiDAR service found (%s: %s); image capture only.",
                type(exc).__name__,
                exc,
            )
            log("No point-cloud / LiDAR service found; image capture only.")
            return None

    def _try_state_stream_client(self, robot, log: LogFn):
        """Create the high-rate IMU streaming client if the service exists."""
        if RobotStateStreamingClient is None:
            return None
        try:
            client = robot.ensure_client(
                RobotStateStreamingClient.default_service_name
            )
            log("High-rate IMU (robot-state-streaming) available.")
            return client
        except Exception:  # noqa: BLE001 - optional, fall back to polling
            log("robot-state-streaming not available; IMU will be polled.")
            return None

    # -- discovery & capture ----------------------------------------------

    @property
    def has_point_cloud(self) -> bool:
        return self.pc_wrapper is not None

    def list_image_sources(self) -> List[ImageSourceInfo]:
        self._require_connected()
        return self.image_wrapper.list_sources()

    def list_point_cloud_sources(self) -> List[PointCloudSourceInfo]:
        if self.pc_wrapper is None:
            return []
        return self.pc_wrapper.list_sources()

    def capture_images(self, source_names: List[str]) -> List[CapturedImage]:
        self._require_connected()
        if not source_names:
            return []
        return self.image_wrapper.capture(source_names)

    def capture_point_clouds(
        self, source_names: List[str]
    ) -> List[CapturedPointCloud]:
        if not source_names:
            return []
        if self.pc_wrapper is None:
            raise SpotConnectionError(
                "No point-cloud / LiDAR service is available on this robot."
            )
        return self.pc_wrapper.capture(source_names)

    def get_battery_percent(self) -> Optional[float]:
        if self.state_wrapper is None:
            return None
        return self.state_wrapper.get_battery_percent()

    # -- calibration / synchronisation metadata ---------------------------

    @property
    def has_state_streaming(self) -> bool:
        return self._state_stream_client is not None

    def clock_skew_sec(self) -> Optional[float]:
        """Robot clock minus client clock, in seconds.

        Spot timestamps every sensor reading in the *robot* clock; subtracting
        this skew maps those timestamps onto the local/client clock. Because all
        sensors share the robot clock, they are already mutually synchronised --
        downstream code only needs to interpolate on that common timeline.
        """
        try:
            skew = self._robot.time_sync.endpoint.clock_skew
            return skew.seconds + skew.nanos * 1e-9
        except Exception:  # noqa: BLE001 - best-effort
            return None

    def get_camera_models(self) -> dict:
        """Per-camera intrinsics + distortion for the calibration manifest."""
        self._require_connected()
        return self.image_wrapper.get_camera_models()

    def get_frame_tree_snapshot(self) -> dict:
        """Body <-> sensor frame tree (nominal extrinsics) at this instant."""
        self._require_connected()
        try:
            return self.state_wrapper.get_transforms_snapshot()
        except Exception:  # noqa: BLE001 - best-effort
            return {}

    def record_state(
        self,
        jsonl_path,
        stop_event,
        target_hz: float,
        log: LogFn = _noop,
    ) -> int:
        """Continuously record IMU / odometry to a JSONL file until stopped.

        One JSON object per line. Prefers the high-rate ``robot-state-streaming``
        service (raw IMU + joint/kinematic state, ~hundreds of Hz) and falls back
        to polling ``get_robot_state`` (fused odometry pose) at ``target_hz``.

        Returns the number of samples written.
        """
        self._require_connected()
        path = Path(jsonl_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(path, "w", encoding="utf-8") as handle:
            use_stream = (
                self.config.prefer_state_streaming
                and self._state_stream_client is not None
            )
            if use_stream:
                try:
                    log("Recording high-rate IMU via robot-state-streaming ...")
                    count = self._record_state_stream(handle, stop_event)
                    return count
                except Exception as exc:  # noqa: BLE001 - fall back to polling
                    logger.warning("State streaming failed: %s", exc)
                    log(f"IMU streaming failed ({exc}); polling instead.")
            log(f"Recording IMU/odometry by polling at ~{target_hz:.0f} Hz ...")
            count = self._record_state_poll(handle, stop_event, target_hz)
        return count

    def _record_state_stream(self, handle, stop_event) -> int:
        count = 0
        for response in self._state_stream_client.get_robot_state_stream():
            if stop_event.is_set():
                break
            record = {
                "local_receive_time": datetime.now().astimezone().isoformat(),
                "source": "robot-state-streaming",
                "data": json_format.MessageToDict(
                    response, preserving_proto_field_name=True
                ),
            }
            handle.write(json.dumps(record, default=str) + "\n")
            count += 1
        return count

    def _record_state_poll(self, handle, stop_event, target_hz: float) -> int:
        interval = 1.0 / target_hz if target_hz > 0 else 0.02
        count = 0
        next_t = time.monotonic()
        while not stop_event.is_set():
            now = time.monotonic()
            if now < next_t:
                stop_event.wait(min(next_t - now, 0.02))
                continue
            try:
                sample = self.state_wrapper.sample_state()
                handle.write(json.dumps(sample, default=str) + "\n")
                count += 1
            except Exception as exc:  # noqa: BLE001 - keep recording through blips
                logger.debug("State poll sample failed: %s", exc)
            next_t = time.monotonic() + interval
        return count

    # -- helpers -----------------------------------------------------------

    def _require_connected(self) -> None:
        if not self.connected or self.image_wrapper is None:
            raise SpotConnectionError("Not connected to a robot.")

    @staticmethod
    def _robot_name(robot) -> str:
        try:
            return robot.get_cached_robot_id().nickname or ""
        except Exception:  # pragma: no cover - id is best-effort
            return ""
