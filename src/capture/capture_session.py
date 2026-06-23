"""Capture session management: deterministic numbering, folder layout, naming.

A *session* is one folder under the capture root that groups everything saved
between pressing "Start new session" and starting the next one:

    captures/
      session_01_YYYYMMDD/
        images/
        pointclouds/
        metadata/

This module is dependency-free (stdlib only) so it can be tested without the
Spot SDK or a GUI present.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

# Matches "session_01_20260623" and captures the numeric index.
_SESSION_RE = re.compile(r"^session_(\d+)_\d{8}$")
# Leading zero-padded index of an asset filename, e.g. "0007_frontleft.jpg".
_ASSET_INDEX_RE = re.compile(r"^(\d+)_")


def sanitize_name(name: str) -> str:
    """Make a source name safe to embed in a filename."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return cleaned.strip("._") or "source"


class CaptureSession:
    """A single capture session and the file/index bookkeeping inside it."""

    def __init__(self, root: Path, index: int, date_str: str):
        self.index = index
        self.date_str = date_str
        self.name = f"session_{index:02d}_{date_str}"
        self.path = Path(root) / self.name
        self.images_dir = self.path / "images"
        self.pointclouds_dir = self.path / "pointclouds"
        self.metadata_dir = self.path / "metadata"
        self.imu_dir = self.path / "imu"
        self.manifest_path = self.path / "session.json"
        self.state_log_path = self.imu_dir / "state_log.jsonl"

        # Resume counters from whatever already exists on disk so re-opening an
        # existing session folder never overwrites prior captures.
        self._image_index = _max_asset_index(self.images_dir)
        self._pointcloud_index = _max_asset_index(self.pointclouds_dir)
        self._event_index = max(self._image_index, self._pointcloud_index)

    def create_dirs(self) -> None:
        """Create the session folder and its subfolders (idempotent)."""
        for d in (
            self.images_dir,
            self.pointclouds_dir,
            self.metadata_dir,
            self.imu_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def next_image_index(self) -> int:
        self._image_index += 1
        return self._image_index

    def next_pointcloud_index(self) -> int:
        self._pointcloud_index += 1
        return self._pointcloud_index

    def next_event_id(self) -> str:
        """Return a fresh capture-event id shared by all assets of one action.

        Used to link images and point clouds captured together (e.g. via
        "Capture Both") so they can later be matched during reconstruction.
        """
        self._event_index += 1
        return f"event_{self._event_index:04d}"


class SessionManager:
    """Owns the capture root and the currently active session."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.active_session: Optional[CaptureSession] = None

    def start_new_session(self, now: Optional[datetime] = None) -> CaptureSession:
        """Create the next session folder and make it active."""
        now = now or datetime.now()
        index = self._next_session_index()
        session = CaptureSession(self.root, index, now.strftime("%Y%m%d"))
        session.create_dirs()
        self.active_session = session
        return session

    def _next_session_index(self) -> int:
        """Return max existing session index across the whole root, plus one."""
        highest = 0
        if self.root.exists():
            for entry in self.root.iterdir():
                if not entry.is_dir():
                    continue
                match = _SESSION_RE.match(entry.name)
                if match:
                    highest = max(highest, int(match.group(1)))
        return highest + 1


def _max_asset_index(directory: Path) -> int:
    """Return the largest leading numeric index among files in a directory."""
    highest = 0
    if directory.exists():
        for entry in directory.iterdir():
            match = _ASSET_INDEX_RE.match(entry.name)
            if match:
                highest = max(highest, int(match.group(1)))
    return highest
