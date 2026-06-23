"""Wrapper around bosdyn RobotStateClient.

Surfaces lightweight status (battery) for the GUI and produces serialisable
robot-state samples for the IMU / odometry recorder. Robot state does not
require a lease, so this is safe to call during pure data capture.

Note on IMU: the standard ``get_robot_state`` RPC does not expose raw IMU
packets -- it provides the fused ``kinematic_state`` (odometry pose in the
``odom``/``vision`` frames plus body velocities), which is the practical signal
for pose interpolation and LiDAR deskewing. True high-rate raw IMU comes from
the ``robot-state-streaming`` service (see :class:`SpotService.record_state`).
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from bosdyn.client.robot_state import RobotStateClient
from google.protobuf import json_format

logger = logging.getLogger(__name__)


class StateClientWrapper:
    def __init__(self, client: RobotStateClient):
        self.client = client

    def get_battery_percent(self) -> Optional[float]:
        """Return the first reported battery charge percentage, if available."""
        try:
            state = self.client.get_robot_state()
            for battery in state.battery_states:
                if battery.HasField("charge_percentage"):
                    return battery.charge_percentage.value
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not read robot state: %s", exc)
        return None

    def sample_state(self) -> dict:
        """Return one JSON-serialisable kinematic-state sample.

        Includes the robot-clock acquisition timestamp (the common timeline for
        cross-sensor synchronisation) and the local wall-clock receive time.
        """
        state = self.client.get_robot_state()
        kinematic = state.kinematic_state
        return {
            "local_receive_time": datetime.now().astimezone().isoformat(),
            "source": "robot-state-poll",
            "acquisition_time_robot_nsec": _ts_nsec(kinematic.acquisition_timestamp),
            "kinematic_state": json_format.MessageToDict(
                kinematic, preserving_proto_field_name=True
            ),
        }

    def get_transforms_snapshot(self) -> dict:
        """Return the current frame-tree snapshot (body <-> sensor extrinsics)."""
        state = self.client.get_robot_state()
        return json_format.MessageToDict(
            state.kinematic_state.transforms_snapshot,
            preserving_proto_field_name=True,
        )


def _ts_nsec(timestamp) -> Optional[int]:
    try:
        return int(timestamp.ToNanoseconds())
    except Exception:  # pragma: no cover - defensive
        return None
