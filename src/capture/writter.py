"""Handles writing capture assets and metadata to disk.

Kept deliberately small and free of Spot-SDK types: the client wrappers decode
protobufs into plain bytes / numpy arrays / dicts, and this writer only knows
how to persist those. OpenCV is imported lazily so that point-cloud and JSON
writing work even if OpenCV is unavailable.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class DiskWriter:
    """Persists images, point clouds and JSON metadata to disk."""

    @staticmethod
    def write_image_bytes(path: Path, data: bytes) -> None:
        """Write already-encoded image bytes (e.g. JPEG passthrough)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)

    @staticmethod
    def write_image_array(path: Path, array: Any) -> None:
        """Encode and write a decoded image array using OpenCV.

        Supports 8-bit colour/greyscale and 16-bit depth (saved as PNG).
        """
        import cv2  # local import: only needed for raw/rotated images

        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), array):
            raise IOError(f"OpenCV failed to write image to {path}")

    @staticmethod
    def write_point_cloud_ply(path: Path, points: Any) -> None:
        """Write an (N, 3) float32 array as a binary little-endian PLY file.

        PLY is a widely supported point-cloud format (Open3D, MeshLab,
        CloudCompare, COLMAP), which keeps captures usable by future 3D tooling
        without adding a heavyweight dependency here.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        pts = np.ascontiguousarray(np.asarray(points, dtype=np.float32).reshape(-1, 3))
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {pts.shape[0]}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "end_header\n"
        )
        with open(path, "wb") as f:
            f.write(header.encode("ascii"))
            f.write(pts.tobytes())

    @staticmethod
    def write_json(path: Path, payload: dict) -> None:
        """Write a metadata dictionary as pretty-printed JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
