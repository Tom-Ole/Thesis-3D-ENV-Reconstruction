"""Wrapper around the Spot PointCloudClient.

On Spot the point-cloud / "LiDAR" data is produced by the EAP payload's
Velodyne sensor and exposed through ``PointCloudClient`` (service name
``velodyne-point-cloud``). This is a real LiDAR-derived point cloud, but the
payload is optional, so :class:`~src.capture.spot_service.SpotService` only
constructs this wrapper when the service is actually present.

The wrapper is named for what it returns (point clouds) and can later be
extended to other point-cloud sources/payloads without touching callers.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np
from bosdyn.api import point_cloud_pb2
from bosdyn.client.point_cloud import PointCloudClient
from google.protobuf import json_format

from src.data_models import CapturedPointCloud, PointCloudSourceInfo

logger = logging.getLogger(__name__)


class PointCloudClientWrapper:
    """Reads and decodes point clouds from Spot point-cloud sources."""

    def __init__(self, client: PointCloudClient):
        self.client = client

    def list_sources(self) -> List[PointCloudSourceInfo]:
        """Return the point-cloud sources currently advertised by the robot."""
        return [
            PointCloudSourceInfo(
                name=src.name,
                frame_name_sensor=getattr(src, "frame_name_sensor", ""),
            )
            for src in self.client.list_point_cloud_sources()
        ]

    def capture(self, source_names: List[str]) -> List[CapturedPointCloud]:
        """Capture and decode point clouds from the given sources."""
        responses = self.client.get_point_cloud_from_sources(list(source_names))
        return [self._decode(resp) for resp in responses]

    def _decode(self, response) -> CapturedPointCloud:
        pc = response.point_cloud
        source = pc.source
        encoding = point_cloud_pb2.PointCloud.Encoding.Name(pc.encoding)

        if pc.encoding == point_cloud_pb2.PointCloud.ENCODING_XYZ_32F:
            points = np.frombuffer(pc.data, dtype=np.float32).reshape(-1, 3)
        else:
            logger.warning(
                "Unsupported point-cloud encoding %s for %s; saving empty cloud",
                encoding,
                source.name,
            )
            points = np.empty((0, 3), dtype=np.float32)

        return CapturedPointCloud(
            source=source.name,
            num_points=pc.num_points,
            points=points,
            encoding=encoding,
            acquisition_time=_timestamp_to_iso(source.acquisition_time),
            acquisition_time_robot_nsec=_timestamp_to_nsec(source.acquisition_time),
            frame_name_sensor=source.frame_name_sensor,
            transforms_snapshot=json_format.MessageToDict(
                source.transforms_snapshot, preserving_proto_field_name=True
            ),
        )


def _timestamp_to_iso(timestamp):
    try:
        return timestamp.ToDatetime().isoformat()
    except Exception:  # pragma: no cover - defensive
        return None


def _timestamp_to_nsec(timestamp):
    """Robot-clock nanoseconds, the common timeline for cross-sensor sync."""
    try:
        return int(timestamp.ToNanoseconds())
    except Exception:  # pragma: no cover - defensive
        return None
