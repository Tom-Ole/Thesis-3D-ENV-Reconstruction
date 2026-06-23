"""Wrapper around the Spot ImageClient for reading body cameras.

Handles source discovery, image requests, decoding (JPEG passthrough vs. raw /
depth), optional rotation of the body fisheye cameras (they are mounted at
angles), and extraction of intrinsics + frame transforms for later 3D work.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import cv2
import numpy as np
from bosdyn.api import image_pb2
from bosdyn.client.image import ImageClient, build_image_request
from google.protobuf import json_format

from src.models import CapturedImage, ImageSourceInfo

logger = logging.getLogger(__name__)

# Body fisheye cameras are physically mounted rotated; rotate captures upright.
# Sources not listed here (depth, hand, etc.) are saved as-is.
CAMERA_ROTATION = {
    "back_fisheye_image": None,
    "left_fisheye_image": None,
    "right_fisheye_image": cv2.ROTATE_180,
    "frontleft_fisheye_image": cv2.ROTATE_90_CLOCKWISE,
    "frontright_fisheye_image": cv2.ROTATE_90_CLOCKWISE,
}

_ROTATION_NAMES = {
    cv2.ROTATE_90_CLOCKWISE: "ROTATE_90_CLOCKWISE",
    cv2.ROTATE_180: "ROTATE_180",
    cv2.ROTATE_90_COUNTERCLOCKWISE: "ROTATE_90_COUNTERCLOCKWISE",
}


class ImageClientWrapper:
    """Reads and decodes images from Spot body cameras."""

    def __init__(self, client: ImageClient, rotate: bool = True):
        self.client = client
        self.rotate = rotate

    def list_sources(self) -> List[ImageSourceInfo]:
        """Return the image sources currently advertised by the robot."""
        sources = []
        for src in self.client.list_image_sources():
            sources.append(
                ImageSourceInfo(
                    name=src.name,
                    rows=src.rows,
                    cols=src.cols,
                    image_type=image_pb2.ImageSource.ImageType.Name(src.image_type),
                    pixel_formats=[
                        image_pb2.Image.PixelFormat.Name(p)
                        for p in getattr(src, "pixel_formats", [])
                    ],
                )
            )
        return sources

    def get_camera_models(self) -> dict:
        """Return per-source camera models (intrinsics + distortion).

        Used to build the session calibration manifest. The model is read
        straight from the robot's advertised :class:`ImageSource`, giving the
        factory/nominal calibration that the user's own OpenCV calibration can
        validate or refine.
        """
        models = {}
        for src in self.client.list_image_sources():
            models[src.name] = {
                "rows": src.rows,
                "cols": src.cols,
                "image_type": image_pb2.ImageSource.ImageType.Name(src.image_type),
                "model": _extract_camera_model(src),
                "raw": json_format.MessageToDict(
                    src, preserving_proto_field_name=True
                ),
            }
        return models

    def capture(
        self,
        source_names: List[str],
        quality_percent: float = 75.0,
        pixel_format: Optional[int] = None,
    ) -> List[CapturedImage]:
        """Capture and decode images from the given sources in one request."""
        requests = [
            build_image_request(
                name, quality_percent=quality_percent, pixel_format=pixel_format
            )
            for name in source_names
        ]
        responses = self.client.get_image(requests)
        return [self._decode(resp) for resp in responses]

    def _decode(self, response) -> CapturedImage:
        shot = response.shot
        image = shot.image
        source = response.source.name

        captured = CapturedImage(
            source=source,
            image_format=image_pb2.Image.Format.Name(image.format),
            pixel_format=image_pb2.Image.PixelFormat.Name(image.pixel_format),
            rows=image.rows,
            cols=image.cols,
            acquisition_time=_timestamp_to_iso(shot.acquisition_time),
            acquisition_time_robot_nsec=_timestamp_to_nsec(shot.acquisition_time),
            frame_name_image_sensor=shot.frame_name_image_sensor,
            camera_model=_extract_camera_model(response.source),
            transforms_snapshot=json_format.MessageToDict(
                shot.transforms_snapshot, preserving_proto_field_name=True
            ),
        )

        rotation = CAMERA_ROTATION.get(source) if self.rotate else None

        if image.format == image_pb2.Image.FORMAT_JPEG:
            if rotation is not None:
                arr = cv2.imdecode(
                    np.frombuffer(image.data, dtype=np.uint8), cv2.IMREAD_UNCHANGED
                )
                captured.array = cv2.rotate(arr, rotation)
                captured.rotation_applied = _ROTATION_NAMES.get(rotation)
            else:
                captured.jpeg_bytes = image.data
            captured.extension = "jpg"
        elif image.format == image_pb2.Image.FORMAT_RAW:
            arr = _decode_raw(image)
            if rotation is not None and arr is not None:
                arr = cv2.rotate(arr, rotation)
                captured.rotation_applied = _ROTATION_NAMES.get(rotation)
            captured.array = arr
            captured.extension = "png"
        else:
            # Unknown/RLE encodings: keep the raw bytes so nothing is lost.
            logger.warning("Unhandled image format for %s; storing raw bytes", source)
            captured.jpeg_bytes = image.data
            captured.extension = "bin"

        return captured


def _decode_raw(image) -> Optional[np.ndarray]:
    """Decode a FORMAT_RAW image into a numpy array based on its pixel format."""
    pf = image_pb2.Image
    layouts = {
        pf.PIXEL_FORMAT_DEPTH_U16: (np.uint16, 1),
        pf.PIXEL_FORMAT_GREYSCALE_U8: (np.uint8, 1),
        pf.PIXEL_FORMAT_GREYSCALE_U16: (np.uint16, 1),
        pf.PIXEL_FORMAT_RGB_U8: (np.uint8, 3),
        pf.PIXEL_FORMAT_RGBA_U8: (np.uint8, 4),
    }
    dtype, channels = layouts.get(image.pixel_format, (np.uint8, 1))
    arr = np.frombuffer(image.data, dtype=dtype)
    if channels == 1:
        arr = arr.reshape(image.rows, image.cols)
    else:
        arr = arr.reshape(image.rows, image.cols, channels)
        # OpenCV writes BGR(A); Spot delivers RGB(A) -> convert before saving.
        if channels == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif channels == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
    return arr


def _intrinsics_dict(intrinsics) -> dict:
    return {
        "focal_length": {"x": intrinsics.focal_length.x, "y": intrinsics.focal_length.y},
        "principal_point": {
            "x": intrinsics.principal_point.x,
            "y": intrinsics.principal_point.y,
        },
        "skew": {"x": intrinsics.skew.x, "y": intrinsics.skew.y},
    }


def _extract_camera_model(source) -> dict:
    """Pull the camera model (intrinsics + distortion) from an ImageSource.

    Handles both the plain ``pinhole`` model and ``pinhole_brown_conrady``
    (which adds Brown-Conrady distortion coefficients k1,k2,k3,p1,p2 needed for
    undistortion). Field access is defensive so this degrades gracefully across
    SDK versions; unknown models fall back to ``type: unknown``.
    """
    try:
        if source.HasField("pinhole_brown_conrady"):
            pbc = source.pinhole_brown_conrady
            dist = pbc.distortion
            return {
                "type": "pinhole_brown_conrady",
                "intrinsics": _intrinsics_dict(pbc.intrinsics),
                "distortion": {
                    k: getattr(dist, k, None) for k in ("k1", "k2", "k3", "p1", "p2")
                },
            }
        if source.HasField("pinhole"):
            return {
                "type": "pinhole",
                "intrinsics": _intrinsics_dict(source.pinhole.intrinsics),
            }
    except Exception:  # noqa: BLE001 - never let metadata extraction crash a capture
        logger.debug("Could not extract camera model for %s", source.name)
    return {"type": "unknown"}


def _timestamp_to_iso(timestamp) -> Optional[str]:
    try:
        return timestamp.ToDatetime().isoformat()
    except Exception:  # pragma: no cover - defensive
        return None


def _timestamp_to_nsec(timestamp) -> Optional[int]:
    """Robot-clock nanoseconds, the common timeline for cross-sensor sync."""
    try:
        return int(timestamp.ToNanoseconds())
    except Exception:  # pragma: no cover - defensive
        return None
